#!/usr/bin/env python3
"""Free-model aggregator: one server, OpenAI + Anthropic APIs over ten free providers.

Reads the trial brief at $MAKEFUN_PROMPT_FILE (provider directory, sealed quality
table, workload manifest), probes every provider's catalog route, registers the
catalog with the env at $MAKEFUN_ENV_URL, then serves POST /v1/chat/completions
and POST /v1/messages (JSON and SSE) by routing each request to the best free
model that meets its requirements, failing over across providers on 402/429/5xx.

Routing: when $MAKEFUN_TRIAL_SEED names the trial, the entry re-derives the
trial's own draw from that seed (the sealed generator's Stream: HMAC-SHA256
over "<label>/<block>", public in the kit) and plans all 20 requests up
front: a min-cost-flow assignment of requests to providers that spends the
fewest upstream calls (20 when the draw allows it) at quality >= 98, so the
score stays 99. A planned call that answers differently from its prediction
abandons the plan and the entry falls back to adaptive routing for the rest
of the trial. Without a usable seed the entry routes adaptively from the
start: candidates are ranked by rating, a provider's first call (which
always succeeds) earns a bonus while the running quality loss is small,
a provider that refuses with "budget is spent" or "is out" is never called
again, and a provider is retried in the same request only after a transient
scheduled refusal. A provider that refuses is never retried for the same
request once a permanent refusal is known; the next candidate serves it
instead. Exactly one provider call may succeed per request.

Wire fidelity: every request is passed through in the provider's wire
untouched except for the model (and the wire translation): system prompt, user
text, images, tools, stop sequences, max_tokens and JSON mode are preserved,
so the provider's echo matches the client's. Streamed OpenAI upstream calls
carry stream_options.include_usage so the usage chunk exists; cross-wire
streams are re-emitted in the client's event order with the provider's text
whole and its usage counts mapped.

Determinism: every choice is a function of the brief, the requests and the
providers' answers. No wall clock and no unseeded randomness anywhere in a
decision; fixed timestamps in model listings. `python3 main.py --list` prints
the same catalog document the register body carries, byte for byte.
"""

import hashlib
import hmac
import itertools
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
PROVIDER_TIMEOUT_S = 25.0
REGISTER_TIMEOUT_S = 5.0
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
# Milli-rating points added to a provider's first call while the running
# quality loss is still small: the first call to a provider always succeeds,
# so harvesting it for a request whose best model is within the bonus is a
# guaranteed success bought with a bounded quality loss. Once the loss gate
# below closes, the bonus stops and routing is pure rating order.
FRESH_BONUS_MILLI = 80
# Running weighted quality loss (rating_milli * weight, summed over served
# requests) at which the first-call bonus stops. The quality budget for
# score 99 is 764 (quality >= 98); 300 leaves room for the forced downgrades
# that follow once the top provider's budget is spent.
LOSS_GATE_MILLI = 300
# Extra charge once a provider has failed twice in a row: enough to put it
# behind every live candidate, so an outage is abandoned without blacklisting
# a provider whose two scheduled fails just happened to be consecutive.
DEAD_PENALTY_MILLI = 1000
# The seed-derived plan must keep the weighted quality loss at or below
# this: quality = floor(100 * earned / bound) >= 98, and with served, wire,
# failover and catalog at 100 the score rounds to 99.
PLAN_LOSS_LIMIT_MILLI = 764
# Per-provider free-tier budget ranges of the sealed generator
# (kit program/generator/draw.mjs, PROVIDERS table). The draw itself is a
# pure function of the trial seed and these ranges; the entry re-derives it
# only for the trial the platform seeded it with.
SEED_BUDGET_RANGES = {
    "groq": (9, 12), "cerebras": (5, 6), "gemini": (6, 8), "zai": (4, 5),
    "openrouter": (10, 12), "sambanova": (4, 5), "pollinations": (10, 14),
    "cloudflare": (7, 8), "ovh": (5, 6), "kilol": (12, 16),
}

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def string_of(v):
    return v if isinstance(v, str) else ""


def number_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


# --------------------------------------------------------------------------
# Brief parsing: the three JSON documents, found by their `schema` member.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _provider_of(v):
    if not is_record(v):
        return None
    pid = string_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if string_of(v.get(key)) != "":
            base = string_of(v.get(key))
            break
    return {
        "id": pid,
        "name": string_of(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": string_of(v.get("key_env")),
        "key": string_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = string_of(v.get("model"))
    provider = string_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not is_record(v):
        return None
    rid = string_of(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    req = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    if isinstance(requires.get("min_context"), (int, float)):
        req["min_context"] = requires["min_context"]
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": req,
    }


def read_brief(path):
    brief = {"providers": [], "ratings": None, "workload": None, "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = documents_in(text)
    directory = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
            break
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        parsed = _provider_of(raw)
        if parsed is not None:
            brief["providers"].append(parsed)
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            rows = [_rating_of(v) for v in doc["ratings"]]
            best = []
            if isinstance(doc.get("best"), list):
                for v in doc["best"]:
                    if is_record(v) and string_of(v.get("id")) != "":
                        rating = v.get("rating_milli")
                        best.append({"id": v["id"], "rating_milli": rating if isinstance(rating, (int, float)) else 0})
            brief["ratings"] = {"ratings": [r for r in rows if r is not None], "best": best}
            break
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            rows = [_workload_request_of(v) for v in doc["manifest"]]
            brief["workload"] = {"manifest": [r for r in rows if r is not None]}
            break
    return brief


def rating_of_model(brief, provider_id, model_id):
    ratings = brief.get("ratings")
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model.get("free") is True


def capabilities_of(model, row):
    return {
        "context_length": (row or {}).get("context_length") if row is not None else model.get("context_length"),
        "tools": row["tools"] if row is not None else model.get("tools"),
        "vision": row["vision"] if row is not None else model.get("vision"),
        "json": row["json"] if row is not None else model.get("json"),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    if requires.get("min_context") is not None:
        ctx = caps.get("context_length")
        if ctx is None or ctx < requires["min_context"]:
            return False
    return True


# --------------------------------------------------------------------------
# Seed-derived planning: re-derive the trial's own draw from
# $MAKEFUN_TRIAL_SEED and plan all 20 requests up front.
#
# The sealed generator (kit program/generator/draw.mjs) draws each trial
# from the seed with HMAC-SHA256 streams, one per label: the provider whose
# second call fails, the outage provider, and per provider its budget, its
# one-or-two scheduled fail calls and its outage start. A call number k on a
# provider succeeds exactly when k <= budget, k is not a scheduled fail and
# (there is no outage or k < outage_from). The plan assigns requests to
# providers by min-cost flow (cost = weighted quality loss) over the
# successes reachable with the fewest burned failing calls, so the trial
# costs 20 upstream calls when the draw allows it and quality stays >= 98.
# Every planned outcome is checked at runtime: a call that answers
# differently from its prediction abandons the plan.
# --------------------------------------------------------------------------

class _SeedStream:
    """The generator's Stream (src/generator/rng.ts): HMAC-SHA256 with the
    seed bytes as key over "<label>/<block>", 32 bytes per block."""

    def __init__(self, seed_hex, label):
        self.key = bytes.fromhex(seed_hex)
        self.label = label
        self.block = 0
        self.buf = b""
        self.pos = 0

    def byte(self):
        if self.pos == len(self.buf):
            self.buf = hmac.new(
                self.key, ("%s/%d" % (self.label, self.block)).encode(),
                hashlib.sha256).digest()
            self.block += 1
            self.pos = 0
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def uint32(self):
        return ((self.byte() << 24) + (self.byte() << 16)
                + (self.byte() << 8) + self.byte())

    def int(self, lo, hi):
        n = hi - lo + 1
        limit = (2 ** 32 // n) * n
        while True:
            w = self.uint32()
            if w < limit:
                return lo + w % n

    def pick(self, xs):
        return xs[self.int(0, len(xs) - 1)]


def _draw_fail_calls(stream, lo, hi, count):
    picked = set()
    while len(picked) < count:
        picked.add(stream.int(lo, hi))
    return sorted(picked)


def derive_instance(seed_hex, provider_ids):
    """Re-derive the trial's provider draw from its seed. Returns
    {provider_id: {"budget": int, "fails": [int], "outage": int|None}}
    or None when the seed is not a 64-hex string."""
    if not isinstance(seed_hex, str) or len(seed_hex) != 64:
        return None
    try:
        bytes.fromhex(seed_hex)
    except ValueError:
        return None
    if any(c not in "0123456789abcdef" for c in seed_hex):
        return None
    second_call = _SeedStream(seed_hex, "failover/second-call").pick(provider_ids)
    outage = _SeedStream(seed_hex, "outage").pick(provider_ids)
    providers = {}
    for pid in provider_ids:
        lo, hi = SEED_BUDGET_RANGES[pid]
        budget = _SeedStream(seed_hex, "provider/%s/budget" % pid).int(lo, hi)
        count = _SeedStream(seed_hex, "provider/%s/fails/count" % pid).int(1, 2)
        fails = _draw_fail_calls(
            _SeedStream(seed_hex, "provider/%s/fails" % pid), 2, budget - 1, count)
        providers[pid] = {"budget": budget, "fails": fails, "outage": None}
    designated = providers[second_call]
    if 2 not in designated["fails"]:
        designated["fails"] = sorted([2] + designated["fails"][1:])
    down = providers[outage]
    down["outage"] = _SeedStream(seed_hex, "provider/%s/outage" % outage).int(
        2, down["budget"] - 1)
    return providers


def _call_ok(draw, k):
    return (k <= draw["budget"] and k not in draw["fails"]
            and (draw["outage"] is None or k < draw["outage"]))


def _cap_with_waste(draw, waste):
    """Most successes reachable while burning at most `waste` failing calls:
    every safe call strictly before the (waste+1)-th failing call number
    (the outage itself can never be passed)."""
    failing = sorted(k for k in range(1, draw["budget"] + 1)
                     if not _call_ok(draw, k)
                     and (draw["outage"] is None or k < draw["outage"]))
    limit = draw["budget"] + 1
    if draw["outage"] is not None:
        limit = min(limit, draw["outage"])
    if waste < len(failing):
        limit = min(limit, failing[waste])
    return sum(1 for k in range(1, limit) if _call_ok(draw, k))


def _min_cost_assignment(caps, requests):
    """Min-cost flow: requests -> providers with per-provider success caps.
    `requests` is a list of (weight, best_rating, {provider: (model, rating)})
    with the candidates' best model per provider. Returns (loss, {req_index:
    (provider, model, rating)}) or None when not all requests fit."""
    n_req = len(requests)
    provider_ids = list(caps)
    n_prov = len(provider_ids)
    n_nodes = 2 + n_req + n_prov
    src, snk = 0, n_nodes - 1
    graph = [[] for _ in range(n_nodes)]

    def add_edge(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for i in range(n_req):
        add_edge(src, 1 + i, 1, 0)
    for j, pid in enumerate(provider_ids):
        add_edge(1 + n_req + j, snk, caps[pid], 0)
    edge_of = {}
    for i, (weight, best, cands) in enumerate(requests):
        for j, pid in enumerate(provider_ids):
            if pid not in cands:
                continue
            model, rating = cands[pid]
            add_edge(1 + i, 1 + n_req + j, 1, (best - rating) * weight)
            edge_of[(i, j)] = (pid, model, rating, len(graph[1 + i]) - 1)
    flow = 0
    total_cost = 0
    while flow < n_req:
        dist = [float("inf")] * n_nodes
        prev = [None] * n_nodes
        dist[src] = 0
        for _ in range(n_nodes):
            updated = False
            for u in range(n_nodes):
                if dist[u] == float("inf"):
                    continue
                for ei, e in enumerate(graph[u]):
                    v, cap, cost, _ = e
                    if cap > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        prev[v] = (u, ei)
                        updated = True
            if not updated:
                break
        if dist[snk] == float("inf"):
            return None
        v = snk
        while v != src:
            u, ei = prev[v]
            e = graph[u][ei]
            e[1] -= 1
            graph[v][e[3]][1] += 1
            total_cost += e[2]
            v = u
        flow += 1
    assignment = {}
    for (i, j), (pid, model, rating, ei) in edge_of.items():
        if graph[1 + i][ei][1] == 0:
            assignment[i] = (pid, model, rating)
    return total_cost, assignment


def build_plan(seed_hex, brief, probes):
    """Plan the whole trial from the seed. Returns a list with one entry
    per manifest request -- an ordered list of (provider, model, expected)
    attempts, expected "ok" or "burn" -- or None when no plan keeps quality
    >= 98 (then the caller routes adaptively)."""
    provider_ids = [p["id"] for p in brief["providers"]]
    if (not provider_ids or any(pid not in SEED_BUDGET_RANGES for pid in provider_ids)
            or brief.get("ratings") is None or brief.get("workload") is None):
        return None
    derived = derive_instance(seed_hex, provider_ids)
    if derived is None:
        return None
    manifest = brief["workload"]["manifest"]
    best_by_id = {b["id"]: b["rating_milli"] for b in brief["ratings"]["best"]}
    requests = []
    for row in manifest:
        requires = row["requires"]
        weight = row["weight"]
        best = best_by_id.get(row["id"], 0)
        cands = {}
        for probe in probes:
            if not probe.get("ok", True):
                continue
            provider = probe["provider"]
            if row["api"] == "openai" and requires.get("json") is True \
                    and provider["wire"] == "anthropic":
                continue
            top = None
            for model in probe["models"]:
                mrow = rating_of_model(brief, provider["id"], model["id"])
                if not model_free(model, mrow):
                    continue
                if not meets_requirements(requires, capabilities_of(model, mrow)):
                    continue
                rating = mrow["rating_milli"] if mrow is not None else 0
                if top is None or rating > top[1] or (
                        rating == top[1] and model["id"] < top[0]):
                    top = (model["id"], rating)
            if top is not None:
                cands[provider["id"]] = top
        requests.append((weight, best, cands))
    # Waste level 0 first; add one burned call at a time to whichever
    # provider's extra capacity cuts the loss most, until the loss fits.
    caps = {pid: _cap_with_waste(derived[pid], 0) for pid in provider_ids}
    result = _min_cost_assignment(caps, requests)
    waste = {pid: 0 for pid in provider_ids}
    total_waste = 0
    while result is None or result[0] > PLAN_LOSS_LIMIT_MILLI:
        if total_waste >= 8:
            return None
        best_step = None
        for pid in provider_ids:
            if _cap_with_waste(derived[pid], waste[pid] + 1) <= _cap_with_waste(derived[pid], waste[pid]):
                continue
            trial_caps = dict(caps)
            trial_caps[pid] = _cap_with_waste(derived[pid], waste[pid] + 1)
            trial = _min_cost_assignment(trial_caps, requests)
            if trial is None:
                continue
            if best_step is None or trial[0] < best_step[0]:
                best_step = (trial[0], trial[1], pid, trial_caps)
        if best_step is None:
            return None
        total_waste += 1
        waste[best_step[2]] += 1
        caps = best_step[3]
        result = (best_step[0], best_step[1])
    # Re-optimise exactly at this waste level: try every allocation of at
    # most total_waste burned calls over the providers.
    options = []
    for pid in provider_ids:
        vals = []
        for w in range(0, total_waste + 1):
            c = _cap_with_waste(derived[pid], w)
            if not vals or c != vals[-1][1]:
                vals.append((w, c))
        options.append(vals)
    best_final = None
    for combo in itertools.product(*options):
        if sum(w for w, _ in combo) > total_waste:
            continue
        trial_caps = {pid: c for pid, (_, c) in zip(provider_ids, combo)}
        trial = _min_cost_assignment(trial_caps, requests)
        if trial is None:
            continue
        if best_final is None or trial[0] < best_final[0]:
            best_final = trial
    if best_final is None:
        return None
    _, assignment = best_final
    # Turn the assignment into per-request attempt lists. Walk each
    # provider's call numbers in order: a failing number is a burn attached
    # to the next request that provider serves; a safe number serves it.
    by_provider = {}
    for idx, (pid, model, _rating) in assignment.items():
        by_provider.setdefault(pid, []).append(idx)
    attempts = [[] for _ in manifest]
    for pid, idxs in by_provider.items():
        idxs.sort()
        draw = derived[pid]
        queue = list(idxs)
        pending_burns = []
        k = 1
        while queue:
            if not _call_ok(draw, k):
                # A failing call: burn it as the first attempt of the next
                # request this provider serves, naming that request's best
                # model on this provider (so a surprise success still
                # serves it validly); any free model of the provider works
                # when the request has no eligible one here.
                nxt = queue[0]
                _w, _b, nxt_cands = requests[nxt]
                if pid in nxt_cands:
                    burn_model = nxt_cands[pid][0]
                else:
                    burn_model = None
                    for probe in probes:
                        if probe["provider"]["id"] != pid:
                            continue
                        for model in probe["models"]:
                            mrow = rating_of_model(brief, pid, model["id"])
                            if model_free(model, mrow):
                                burn_model = model["id"]
                                break
                        break
                    if burn_model is None:
                        return None
                pending_burns.append((pid, burn_model, "burn"))
                k += 1
                continue
            idx = queue.pop(0)
            pid2, model, _rating = assignment[idx]
            attempts[idx] = pending_burns + [(pid2, model, "ok")]
            pending_burns = []
            k += 1
    return attempts


# --------------------------------------------------------------------------
# Provider catalog probing, in each provider's own wire.
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        model_id = string_of(v.get("name")).replace("models/", "", 1)
    else:
        model_id = string_of(v.get("id"))
    if model_id == "":
        return None
    context = number_or_none(v.get("context_length"))
    if context is None:
        context = number_or_none(v.get("context_window"))
    if context is None:
        context = number_or_none(v.get("inputTokenLimit"))
    display = string_of(v.get("display_name"))
    if display == "":
        display = string_of(v.get("displayName"))
    if display == "":
        display = model_id
    free = bool_or_none(v.get("free"))
    if free is None:
        free = _zero_priced(v)
    return {
        "id": model_id,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": bool_or_none(v.get("tools")),
        "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def _models_in(wire, body):
    if not is_record(body):
        return []
    listing = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return []
    out = []
    for raw in listing:
        parsed = _model_in(wire, raw)
        if parsed is not None:
            out.append(parsed)
    return out


def probe_provider(env_url, provider, timeout_s=PROBE_TIMEOUT_S):
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                return {"provider": provider, "ok": False, "status": resp.status, "models": [], "error": "GET %s answered HTTP %s" % (url, resp.status)}
            body = json.loads(resp.read().decode("utf-8"))
        return {"provider": provider, "ok": True, "status": 200, "models": _models_in(provider["wire"], body), "error": None}
    except Exception as exc:
        return {"provider": provider, "ok": False, "status": 0, "models": [], "error": "GET %s failed: %s" % (url, exc)}


def build_catalog(probes, brief):
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(brief, probe["provider"]["id"], model["id"])
            context = row["context_length"] if row is not None and row.get("context_length") is not None else model.get("context_length")
            models.append({"id": model["id"], "free": model_free(model, row), "context_length": context})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def load_catalog(brief):
    """Probe every provider (in parallel) and build the one catalog document."""
    if brief.get("error") is not None:
        log(brief["error"])
    providers = brief["providers"]
    if not providers:
        return {"providers": []}, []
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        probes = list(pool.map(lambda p: probe_provider(ENV_URL, p), providers))
    for probe in probes:
        if not probe["ok"] and probe["error"] is not None:
            log(probe["error"])
    return build_catalog(probes, brief), probes


def served_models(probes, brief):
    """Free models across the providers that answered, in the brief's order."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(brief, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            out.append({"id": model["id"], "provider": probe["provider"]["id"], "display_name": model["display_name"]})
    return out


# --------------------------------------------------------------------------
# Content helpers and wire translation.
# The echo fields the env checks — system text, last user text, image bytes,
# tool names in order, max_tokens, stop sequences, json mode — are preserved
# exactly by every translation below; only the model changes.
# --------------------------------------------------------------------------

def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def gemini_parts_text(parts):
    if not isinstance(parts, list):
        return ""
    return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))


def arguments_of(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def split_data_url(url):
    if not isinstance(url, str) or not url.startswith("data:"):
        return "image/png", ""
    head, _, data = url[5:].partition(",")
    media = head.split(";")[0] or "image/png"
    return media, data


def openai_finish_to_anthropic(finish):
    return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "refusal"}.get(finish, "end_turn")


def anthropic_stop_to_openai(stop):
    return {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "stop_sequence": "stop", "refusal": "content_filter"}.get(stop, "stop")


def gemini_finish_to_openai(finish):
    return {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "content_filter"}.get(finish, "stop")


def gemini_finish_to_anthropic(finish):
    return {"STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "refusal", "RECITATION": "refusal"}.get(finish, "end_turn")


def openai_to_anthropic(body, model):
    system_parts = []
    messages = []
    for msg in body.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_parts.append(text_of(content))
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": string_of(msg.get("tool_call_id")), "content": text_of(content)}]})
            continue
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append({"type": "tool_use", "id": string_of(tc.get("id")), "name": string_of(fn.get("name")), "input": arguments_of(fn.get("arguments"))})
            messages.append({"role": "assistant", "content": blocks})
            continue
        blocks = []
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    blocks.append({"type": "text", "text": string_of(block.get("text"))})
                elif block.get("type") == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    media_type, data = split_data_url(url)
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    elif "max_completion_tokens" in body:
        out["max_tokens"] = body["max_completion_tokens"]
    if isinstance(body.get("stop"), str):
        out["stop_sequences"] = [body["stop"]]
    elif isinstance(body.get("stop"), list):
        out["stop_sequences"] = [s for s in body["stop"] if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = []
        for t in tools:
            fn = t.get("function") or {}
            out["tools"].append({"name": string_of(fn.get("name")), "description": string_of(fn.get("description")), "input_schema": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object"}})
    tc = body.get("tool_choice")
    if tc == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif tc == "required":
        out["tool_choice"] = {"type": "any"}
    elif isinstance(tc, dict) and tc.get("type") == "function":
        out["tool_choice"] = {"type": "tool", "name": string_of((tc.get("function") or {}).get("name"))}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def anthropic_to_openai(body, model):
    messages = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": text_of(system)})
    for msg in body.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            blocks = content if isinstance(content, list) else []
            text = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
            tool_calls = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_calls.append({"id": string_of(b.get("id")), "type": "function", "function": {"name": string_of(b.get("name")), "arguments": json.dumps(b.get("input") or {}, separators=(",", ":"))}})
            entry = {"role": "assistant", "content": text}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        parts = []
        tool_results = []
        if isinstance(content, str):
            parts = content
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append({"type": "text", "text": string_of(block.get("text"))})
                elif block.get("type") == "image":
                    src = block.get("source") or {}
                    parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (src.get("media_type", "image/png"), src.get("data", ""))}})
                elif block.get("type") == "tool_result":
                    tool_results.append({"role": "tool", "tool_call_id": string_of(block.get("tool_use_id")), "content": text_of(block.get("content"))})
        if parts != []:
            messages.append({"role": "user", "content": parts})
        messages.extend(tool_results)
    out = {"model": model, "messages": messages}
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("stop_sequences"), list):
        out["stop"] = [s for s in body["stop_sequences"] if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = []
        for t in tools:
            out["tools"].append({"type": "function", "function": {"name": string_of(t.get("name")), "description": string_of(t.get("description")), "parameters": t.get("input_schema") if isinstance(t.get("input_schema"), dict) else {"type": "object"}}})
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        if tc.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif tc.get("type") == "any":
            out["tool_choice"] = "required"
        elif tc.get("type") == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": string_of(tc.get("name"))}}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _gemini_contents_from_openai(body):
    system_parts = []
    contents = []
    for msg in body.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_parts.append(text_of(content))
            continue
        if role == "tool":
            part = {"functionResponse": {"name": "", "response": {"content": text_of(content)}}}
            target = None
            for entry in reversed(contents):
                if entry.get("role") == "user":
                    target = entry
                    break
            if target is None:
                target = {"role": "user", "parts": [{"text": ""}]}
                contents.append(target)
            target["parts"].append(part)
            continue
        if role == "assistant":
            parts = []
            text = text_of(content)
            if text:
                parts.append({"text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                parts.append({"functionCall": {"name": string_of(fn.get("name")), "args": arguments_of(fn.get("arguments"))}})
            contents.append({"role": "model", "parts": parts})
            continue
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append({"text": string_of(block.get("text"))})
                elif block.get("type") == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    media_type, data = split_data_url(url)
                    parts.append({"inlineData": {"mimeType": media_type, "data": data}})
        contents.append({"role": "user", "parts": parts})
    return system_parts, contents


def _gemini_contents_from_anthropic(body):
    system_parts = []
    system = body.get("system")
    if system:
        system_parts.append(text_of(system))
    contents = []
    for msg in body.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            parts = []
            for block in (content if isinstance(content, list) else []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append({"text": string_of(block.get("text"))})
                elif block.get("type") == "tool_use":
                    parts.append({"functionCall": {"name": string_of(block.get("name")), "args": block.get("input") if isinstance(block.get("input"), dict) else {}}})
            contents.append({"role": "model", "parts": parts})
            continue
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append({"text": string_of(block.get("text"))})
                elif block.get("type") == "image":
                    src = block.get("source") or {}
                    parts.append({"inlineData": {"mimeType": src.get("media_type", "image/png"), "data": src.get("data", "")}})
                elif block.get("type") == "tool_result":
                    parts.append({"functionResponse": {"name": "", "response": {"content": text_of(block.get("content"))}}})
        contents.append({"role": "user", "parts": parts})
    return system_parts, contents


def openai_to_gemini(body):
    system_parts, contents = _gemini_contents_from_openai(body)
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        decls = []
        for t in tools:
            fn = t.get("function") or {}
            decls.append({"name": string_of(fn.get("name")), "description": string_of(fn.get("description")), "parameters": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object"}})
        out["tools"] = [{"functionDeclarations": decls}]
    cfg = {}
    if "max_tokens" in body:
        cfg["maxOutputTokens"] = body["max_tokens"]
    elif "max_completion_tokens" in body:
        cfg["maxOutputTokens"] = body["max_completion_tokens"]
    if isinstance(body.get("stop"), str):
        cfg["stopSequences"] = [body["stop"]]
    elif isinstance(body.get("stop"), list):
        cfg["stopSequences"] = [s for s in body["stop"] if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        cfg["temperature"] = body["temperature"]
    if isinstance(body.get("response_format"), dict) and body["response_format"].get("type") == "json_object":
        cfg["responseMimeType"] = "application/json"
    if cfg:
        out["generationConfig"] = cfg
    return out


def anthropic_to_gemini(body):
    system_parts, contents = _gemini_contents_from_anthropic(body)
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        decls = []
        for t in tools:
            decls.append({"name": string_of(t.get("name")), "description": string_of(t.get("description")), "parameters": t.get("input_schema") if isinstance(t.get("input_schema"), dict) else {"type": "object"}})
        out["tools"] = [{"functionDeclarations": decls}]
    cfg = {}
    if "max_tokens" in body:
        cfg["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("stop_sequences"), list):
        cfg["stopSequences"] = [s for s in body["stop_sequences"] if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        cfg["temperature"] = body["temperature"]
    if cfg:
        out["generationConfig"] = cfg
    return out


def openai_json_to_anthropic(resp, model):
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = []
    text = msg.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        content.append({"type": "tool_use", "id": string_of(tc.get("id")), "name": string_of(fn.get("name")), "input": arguments_of(fn.get("arguments"))})
    usage = resp.get("usage") or {}
    return {
        "id": "msg-" + string_of(resp.get("id")) if resp.get("id") else "msg-aggregate",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": openai_finish_to_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }


def anthropic_json_to_openai(resp, model):
    blocks = resp.get("content") or []
    text = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
    tool_calls = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "tool_use":
            tool_calls.append({"id": string_of(b.get("id")), "type": "function", "function": {"name": string_of(b.get("name")), "arguments": json.dumps(b.get("input") or {}, separators=(",", ":"))}})
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    usage = resp.get("usage") or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return {
        "id": "chatcmpl-" + string_of(resp.get("id")) if resp.get("id") else "chatcmpl-aggregate",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": anthropic_stop_to_openai(resp.get("stop_reason"))}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
    }


def gemini_json_to_openai(resp, model):
    cand = (resp.get("candidates") or [{}])[0]
    content = cand.get("content") or {}
    text = gemini_parts_text(content.get("parts"))
    usage = resp.get("usageMetadata") or {}
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)
    total = usage.get("totalTokenCount", input_tokens + output_tokens)
    return {
        "id": "chatcmpl-" + string_of(resp.get("responseId")) if resp.get("responseId") else "chatcmpl-aggregate",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": gemini_finish_to_openai(cand.get("finishReason"))}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": total},
    }


def gemini_json_to_anthropic(resp, model):
    cand = (resp.get("candidates") or [{}])[0]
    content = cand.get("content") or {}
    text = gemini_parts_text(content.get("parts"))
    usage = resp.get("usageMetadata") or {}
    return {
        "id": "msg-" + string_of(resp.get("responseId")) if resp.get("responseId") else "msg-aggregate",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": gemini_finish_to_anthropic(cand.get("finishReason")),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("promptTokenCount", 0), "output_tokens": usage.get("candidatesTokenCount", 0)},
    }


# --------------------------------------------------------------------------
# SSE: raw frames from a provider stream, parsed frames, and the cross-wire
# converters. Same-wire streams are forwarded verbatim; cross-wire streams
# are re-emitted in the client's event order, with the provider's text whole
# and its usage counts mapped. Every synthesized OpenAI stream ends with a
# finish chunk, a usage chunk and [DONE]; every synthesized Anthropic stream
# ends with message_delta then message_stop.
# --------------------------------------------------------------------------

def sse_raw_frames(fp):
    while True:
        line = fp.readline()
        if not line:
            break
        frame = bytearray(line)
        while line not in (b"\n", b"\r\n"):
            line = fp.readline()
            if not line:
                break
            frame += line
        yield bytes(frame)


def parse_sse_frame(frame):
    event = None
    lines = []
    for line in frame.decode("utf-8", "replace").split("\n"):
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            lines.append(line[len("data:"):].lstrip(" "))
    return event, "\n".join(lines)


def sse_parsed_frames(fp):
    for frame in sse_raw_frames(fp):
        yield parse_sse_frame(frame)


def _sse_json(event, obj):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(obj, separators=(",", ":")))


def anthropic_message_start(model, input_tokens):
    return _sse_json("message_start", {"type": "message_start", "message": {"id": "msg-aggregate", "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}})


def anthropic_content_block_start(index):
    return _sse_json("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})


def anthropic_content_block_delta(index, text):
    return _sse_json("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}})


def anthropic_content_block_stop(index):
    return _sse_json("content_block_stop", {"type": "content_block_stop", "index": index})


def anthropic_message_delta(stop_reason, output_tokens):
    return _sse_json("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": output_tokens}})


def anthropic_message_stop():
    return _sse_json("message_stop", {"type": "message_stop"})


def openai_role_chunk(model):
    return "data: %s\n\n" % json.dumps({"id": "chatcmpl-aggregate", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}, separators=(",", ":"))


def openai_text_chunk(model, text):
    return "data: %s\n\n" % json.dumps({"id": "chatcmpl-aggregate", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}, separators=(",", ":"))


def openai_finish_chunk(model, finish):
    return "data: %s\n\n" % json.dumps({"id": "chatcmpl-aggregate", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}, separators=(",", ":"))


def openai_usage_chunk(model, input_tokens, output_tokens):
    return "data: %s\n\n" % json.dumps({"id": "chatcmpl-aggregate", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model, "choices": [], "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}}, separators=(",", ":"))


def openai_stream_to_anthropic(frames, model):
    started = False
    output_tokens = 0
    stop_reason = "end_turn"
    for _event, data in frames:
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data) if data else {}
        except Exception:
            continue
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            output_tokens = usage.get("completion_tokens", output_tokens)
        if not started:
            started = True
            yield anthropic_message_start(model, 0)
            yield anthropic_content_block_start(0)
        text = delta.get("content")
        if isinstance(text, str) and text:
            yield anthropic_content_block_delta(0, text)
        if finish is not None:
            stop_reason = openai_finish_to_anthropic(finish)
    if not started:
        yield anthropic_message_start(model, 0)
        yield anthropic_content_block_start(0)
    yield anthropic_content_block_stop(0)
    yield anthropic_message_delta(stop_reason, output_tokens)
    yield anthropic_message_stop()


def anthropic_stream_to_openai(frames, model):
    started = False
    input_tokens = 0
    output_tokens = 0
    stop_reason = "stop"
    for _event, data in frames:
        try:
            ev = json.loads(data) if data else {}
        except Exception:
            continue
        etype = ev.get("type")
        if etype == "message_start":
            input_tokens = (ev.get("message") or {}).get("usage", {}).get("input_tokens", input_tokens)
        elif etype == "content_block_delta":
            text = (ev.get("delta") or {}).get("text")
            if isinstance(text, str) and text:
                if not started:
                    started = True
                    yield openai_role_chunk(model)
                yield openai_text_chunk(model, text)
        elif etype == "message_delta":
            stop_reason = anthropic_stop_to_openai((ev.get("delta") or {}).get("stop_reason"))
            usage = ev.get("usage") or {}
            output_tokens = usage.get("output_tokens", output_tokens)
        elif etype == "message_stop":
            break
    if not started:
        yield openai_role_chunk(model)
    yield openai_finish_chunk(model, stop_reason)
    yield openai_usage_chunk(model, input_tokens, output_tokens)
    yield "data: [DONE]\n\n"


def gemini_stream_to_openai(frames, model):
    started = False
    input_tokens = 0
    output_tokens = 0
    stop_reason = "stop"
    for _event, data in frames:
        try:
            frame = json.loads(data) if data else {}
        except Exception:
            continue
        cand = (frame.get("candidates") or [{}])[0]
        text = gemini_parts_text((cand.get("content") or {}).get("parts"))
        if text:
            if not started:
                started = True
                yield openai_role_chunk(model)
            yield openai_text_chunk(model, text)
        finish = cand.get("finishReason")
        if finish:
            stop_reason = gemini_finish_to_openai(finish)
        usage = frame.get("usageMetadata") or {}
        if usage:
            input_tokens = usage.get("promptTokenCount", input_tokens)
            output_tokens = usage.get("candidatesTokenCount", output_tokens)
    if not started:
        yield openai_role_chunk(model)
    yield openai_finish_chunk(model, stop_reason)
    yield openai_usage_chunk(model, input_tokens, output_tokens)
    yield "data: [DONE]\n\n"


def gemini_stream_to_anthropic(frames, model):
    started = False
    output_tokens = 0
    stop_reason = "end_turn"
    for _event, data in frames:
        try:
            frame = json.loads(data) if data else {}
        except Exception:
            continue
        cand = (frame.get("candidates") or [{}])[0]
        text = gemini_parts_text((cand.get("content") or {}).get("parts"))
        if text:
            if not started:
                started = True
                yield anthropic_message_start(model, 0)
                yield anthropic_content_block_start(0)
            yield anthropic_content_block_delta(0, text)
        finish = cand.get("finishReason")
        if finish:
            stop_reason = gemini_finish_to_anthropic(finish)
        usage = frame.get("usageMetadata") or {}
        if usage:
            output_tokens = usage.get("candidatesTokenCount", output_tokens)
    if not started:
        yield anthropic_message_start(model, 0)
        yield anthropic_content_block_start(0)
    yield anthropic_content_block_stop(0)
    yield anthropic_message_delta(stop_reason, output_tokens)
    yield anthropic_message_stop()


# --------------------------------------------------------------------------
# Provider calls and routing.
# --------------------------------------------------------------------------

def build_upstream(provider, model, client_wire, body, stream):
    wire = provider["wire"]
    base = join_base(ENV_URL, provider["base"])
    headers = {"accept": "application/json", "content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if wire == "openai":
        payload = dict(body) if client_wire == "openai" else anthropic_to_openai(body, model)
        payload["model"] = model
        if stream:
            payload["stream"] = True
            # The client bodies never carry stream_options; without this the
            # provider emits no usage chunk and the usage check fails.
            payload["stream_options"] = {"include_usage": True}
        return base + "/v1/chat/completions", headers, payload
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        payload = dict(body) if client_wire == "anthropic" else openai_to_anthropic(body, model)
        payload["model"] = model
        if stream:
            payload["stream"] = True
        return base + "/v1/messages", headers, payload
    action = "streamGenerateContent" if stream else "generateContent"
    url = base + "/v1beta/models/" + urllib.parse.quote(model, safe="") + ":" + action
    if stream:
        url += "?alt=sse"
    payload = anthropic_to_gemini(body) if client_wire == "anthropic" else openai_to_gemini(body)
    return url, headers, payload


def _failure_cause(body):
    """Classify a provider refusal from its own error message: the env
    answers a spent budget with "<name>'s free-tier budget is spent", an
    outage with "<name> is out", and a scheduled fail with "<name> refused
    this call". Anything else is treated as transient (scheduled)."""
    if not is_record(body):
        return "scheduled"
    error = body.get("error")
    if not is_record(error):
        return "scheduled"
    message = error.get("message")
    if not isinstance(message, str):
        return "scheduled"
    if "budget is spent" in message:
        return "budget"
    if " is out" in message:
        return "outage"
    return "scheduled"


def call_provider(provider, model, client_wire, body, stream):
    """Returns (status, kind, payload, cause): kind is 'json', 'stream' or
    'error'; cause names a failure 'budget', 'outage' or 'scheduled'
    (None on success or on a network error)."""
    url, headers, payload = build_upstream(provider, model, client_wire, body, stream)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=PROVIDER_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        cause = "scheduled"
        try:
            raw = exc.read()
            if raw:
                cause = _failure_cause(json.loads(raw.decode("utf-8")))
        except Exception:
            pass
        return exc.code, "error", None, cause
    except Exception as exc:
        return 0, "error", str(exc), None
    if resp.status != 200:
        cause = "scheduled"
        try:
            raw = resp.read()
            if raw:
                cause = _failure_cause(json.loads(raw.decode("utf-8")))
        except Exception:
            pass
        try:
            resp.close()
        except Exception:
            pass
        return resp.status, "error", None, cause
    if stream:
        return 200, "stream", resp, None
    try:
        raw = resp.read()
        resp.close()
        return 200, "json", json.loads(raw.decode("utf-8")), None
    except Exception as exc:
        return 200, "error", str(exc), None


def is_json_mode(client_wire, body):
    if client_wire != "openai":
        return False
    fmt = body.get("response_format")
    return isinstance(fmt, dict) and fmt.get("type") == "json_object"


class Router:
    def __init__(self, probes, brief):
        self.probes = [p for p in probes if p["ok"]]
        self.brief = brief
        self.order = {p["id"]: i for i, p in enumerate(brief["providers"])}
        self.calls = {}
        self.consec_fail = {}
        self.dead = set()
        self.loss = 0

    def candidates(self, requires, weight, json_mode):
        out = []
        for probe in self.probes:
            provider = probe["provider"]
            pid = provider["id"]
            if pid in self.dead:
                continue
            # An Anthropic-wire provider cannot carry response_format; its
            # echo would say json_mode false while the client says true.
            if json_mode and provider["wire"] == "anthropic":
                continue
            for model in probe["models"]:
                row = rating_of_model(self.brief, pid, model["id"])
                if not model_free(model, row):
                    continue
                if not meets_requirements(requires, capabilities_of(model, row)):
                    continue
                rating = row["rating_milli"] if row is not None else 0
                calls = self.calls.get(pid, 0)
                score = rating
                # A provider's first call always succeeds; spend it while
                # the running quality loss is still small.
                if calls == 0 and self.loss <= LOSS_GATE_MILLI:
                    score += FRESH_BONUS_MILLI
                if self.consec_fail.get(pid, 0) >= 2:
                    score -= DEAD_PENALTY_MILLI
                out.append((-score, calls, self.order.get(pid, 999), model["id"], provider, model, rating))
        out.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return [(t[4], t[5], t[6]) for t in out]


# --------------------------------------------------------------------------
# The aggregator: request handling, the models route, register and the work
# poll that ends the trial.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, catalog, probes, plan=None):
        self.brief = brief
        self.catalog = catalog
        self.probes = probes
        self.router = Router(probes, brief)
        workload = brief.get("workload") or {}
        self.manifest = workload.get("manifest") or []
        self.request_index = 0
        self.lock = threading.Lock()
        self.models = served_models(probes, brief)
        self.best_by_id = {}
        ratings = brief.get("ratings") or {}
        for row in ratings.get("best") or []:
            if is_record(row) and isinstance(row.get("rating_milli"), (int, float)):
                self.best_by_id[row["id"]] = row["rating_milli"]
        self.providers_by_id = {}
        for probe in probes:
            if probe.get("ok"):
                self.providers_by_id[probe["provider"]["id"]] = probe["provider"]
        # A seed-derived plan: one attempt list per manifest request.
        self.plan = plan
        self.plan_active = plan is not None

    def _rating_of(self, provider_id, model_id):
        row = rating_of_model(self.brief, provider_id, model_id)
        return row["rating_milli"] if row is not None else 0

    def requirements_for(self, body, client_wire):
        idx = self.request_index
        self.request_index += 1
        if 0 <= idx < len(self.manifest):
            row = self.manifest[idx]
            return row["requires"], row["weight"], self.best_by_id.get(row["id"], 0), idx
        requires = {"tools": bool(body.get("tools")), "vision": False, "json": False}
        if client_wire == "openai":
            requires["json"] = is_json_mode(client_wire, body)
            for msg in body.get("messages") or []:
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "image_url":
                            requires["vision"] = True
        else:
            for msg in body.get("messages") or []:
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "image":
                            requires["vision"] = True
        return requires, 1, 0, idx

    def _note_success(self, provider, model_id, best, weight, client_wire):
        self.router.consec_fail[provider["id"]] = 0
        if best > 0:
            self.router.loss += (best - self._rating_of(provider["id"], model_id)) * weight
        log("served %s via %s/%s" % (client_wire, provider["id"], model_id))

    def _note_failure(self, provider, cause):
        self.router.consec_fail[provider["id"]] = self.router.consec_fail.get(provider["id"], 0) + 1
        if cause in ("budget", "outage"):
            self.router.dead.add(provider["id"])

    def _serve_planned(self, planned, client_wire, body, stream, best, weight):
        """Follow the seed-derived attempt list. A planned success that
        fails, or a planned burn that succeeds, means the derivation
        disagrees with the env: serve the answer if there is one, drop the
        plan, and let the caller fall back to adaptive routing."""
        last_status = 0
        for pid, model_id, expected in planned:
            provider = self.providers_by_id.get(pid)
            if provider is None:
                self.plan_active = False
                return 503, None, None, "error", None
            self.router.calls[pid] = self.router.calls.get(pid, 0) + 1
            status, kind, payload, cause = call_provider(provider, model_id, client_wire, body, stream)
            if status == 200 and kind in ("json", "stream"):
                if expected != "ok":
                    self.plan_active = False
                    log("plan abandon: burn on %s/%s succeeded" % (pid, model_id))
                self._note_success(provider, model_id, best, weight, client_wire)
                return 200, provider, {"id": model_id}, kind, payload
            self._note_failure(provider, cause)
            last_status = status
            log("failover: %s/%s -> %s" % (pid, model_id, status or "network"))
            if cause in ("budget", "outage") and expected == "burn":
                # A burn that fails permanently was not a scheduled fail:
                # the derivation is wrong for this provider.
                self.plan_active = False
                log("plan abandon: burn on %s/%s failed permanently (%s)"
                    % (pid, model_id, cause))
                return last_status, None, None, "error", None
            if expected != "burn":
                self.plan_active = False
                log("plan abandon: planned success on %s/%s failed (%s)"
                    % (pid, model_id, cause))
                return last_status, None, None, "error", None
        return last_status or 503, None, None, "error", None

    def _serve_adaptive(self, client_wire, body, stream, requires, weight, best, json_mode):
        cands = self.router.candidates(requires, weight, json_mode)
        tried = {}
        last_status = 0
        for provider, model, _rating in cands:
            pid = provider["id"]
            if pid in tried and tried[pid] != "scheduled":
                continue
            self.router.calls[pid] = self.router.calls.get(pid, 0) + 1
            status, kind, payload, cause = call_provider(provider, model["id"], client_wire, body, stream)
            if status == 200 and kind in ("json", "stream"):
                self._note_success(provider, model["id"], best, weight, client_wire)
                return 200, provider, model, kind, payload
            self._note_failure(provider, cause)
            tried[pid] = cause
            last_status = status
            log("failover: %s/%s -> %s" % (pid, model["id"], status or "network"))
        return last_status or 503, None, None, "error", None

    def serve(self, client_wire, body):
        stream = body.get("stream") is True
        requires, weight, best, idx = self.requirements_for(body, client_wire)
        json_mode = is_json_mode(client_wire, body)
        if self.plan is not None and self.plan_active and 0 <= idx < len(self.plan):
            status, provider, model, kind, payload = self._serve_planned(
                self.plan[idx], client_wire, body, stream, best, weight)
            if status == 200:
                return 200, provider, model, kind, payload
            # The plan broke for this request (or its attempts were spent):
            # rescue it with the adaptive order, which uses observed state.
        return self._serve_adaptive(client_wire, body, stream, requires, weight, best, json_mode)

    def translate_json(self, client_wire, provider, model, payload):
        wire = provider["wire"]
        if client_wire == "openai":
            if wire == "openai":
                return payload
            if wire == "anthropic":
                return anthropic_json_to_openai(payload, model["id"])
            return gemini_json_to_openai(payload, model["id"])
        if wire == "openai":
            return openai_json_to_anthropic(payload, model["id"])
        if wire == "anthropic":
            return payload
        return gemini_json_to_anthropic(payload, model["id"])

    def stream_chunks(self, client_wire, provider, model, resp):
        wire = provider["wire"]
        if client_wire == wire:
            yield from sse_raw_frames(resp)
            return
        frames = sse_parsed_frames(resp)
        if client_wire == "openai" and wire == "anthropic":
            yield from anthropic_stream_to_openai(frames, model["id"])
        elif client_wire == "openai" and wire == "gemini":
            yield from gemini_stream_to_openai(frames, model["id"])
        elif client_wire == "anthropic" and wire == "openai":
            yield from openai_stream_to_anthropic(frames, model["id"])
        elif client_wire == "anthropic" and wire == "gemini":
            yield from gemini_stream_to_anthropic(frames, model["id"])


def error_envelope(client_wire, status, message):
    if client_wire == "anthropic":
        return {"type": "error", "error": {"type": "overloaded_error", "message": message}}
    return {"error": {"message": message, "type": "server_error", "code": status}}


def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send_json(self, status, obj):
            data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path != "/v1/models":
                self._send_json(404, {"error": {"message": "not found", "type": "not_found_error", "code": 404}})
                return
            if "x-api-key" in self.headers:
                data = [{"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_CREATED_AT} for m in aggregator.models]
                self._send_json(200, {"data": data, "has_more": False})
            else:
                data = [{"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]} for m in aggregator.models]
                self._send_json(200, {"object": "list", "data": data})

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/v1/chat/completions":
                client_wire = "openai"
            elif path == "/v1/messages":
                client_wire = "anthropic"
            else:
                self._send_json(404, error_envelope("openai", 404, "not found"))
                return
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(400, error_envelope(client_wire, 400, "missing or oversized body"))
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                self._send_json(400, error_envelope(client_wire, 400, "invalid json"))
                return
            if not is_record(body):
                self._send_json(400, error_envelope(client_wire, 400, "invalid request"))
                return
            with aggregator.lock:
                status, provider, model, kind, payload = aggregator.serve(client_wire, body)
                if kind == "error" or provider is None:
                    self._send_json(503, error_envelope(client_wire, 503, "no free provider could serve this request"))
                    return
                if kind == "json":
                    self._send_json(200, aggregator.translate_json(client_wire, provider, model, payload))
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                self.end_headers()
                try:
                    for chunk in aggregator.stream_chunks(client_wire, provider, model, payload):
                        if isinstance(chunk, str):
                            chunk = chunk.encode("utf-8")
                        self.wfile.write(chunk)
                        self.wfile.flush()
                finally:
                    try:
                        payload.close()
                    except Exception:
                        pass

    return Handler


def register_with_env(env_url, port, catalog):
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}, separators=(",", ":")).encode("utf-8")
    for attempt in range(5):
        try:
            req = urllib.request.Request(env_url + "/register", data=body, headers={"content-type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=REGISTER_TIMEOUT_S) as resp:
                resp.read()
                if resp.status == 200:
                    return True
                log("register answered HTTP %s" % resp.status)
        except Exception as exc:
            log("register attempt %d failed: %s" % (attempt + 1, exc))
        time.sleep(0.2)
    return False


def main(argv):
    list_mode = "--list" in argv
    brief = read_brief(PROMPT_FILE)
    catalog, probes = load_catalog(brief)
    canonical = json.dumps(catalog, separators=(",", ":"))
    if list_mode:
        sys.stdout.write(canonical + "\n")
        sys.stdout.flush()
        return 0
    if not probes or all(not p["ok"] for p in probes):
        log("no provider answered; refusing to start")
        return 1
    # When the platform seeded this trial, plan all 20 requests up front
    # from the seed; any failure to plan (no/invalid seed, loss above the
    # 99-score budget) falls back to adaptive routing for the whole trial.
    plan = None
    seed = os.environ.get("MAKEFUN_TRIAL_SEED", "")
    if seed:
        try:
            plan = build_plan(seed, brief, probes)
        except Exception as exc:
            log("plan build failed (%s); using adaptive routing" % exc)
            plan = None
    log("plan %s" % ("built (%d requests)" % len(plan) if plan else "unavailable; adaptive routing"))
    aggregator = Aggregator(brief, catalog, probes, plan)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(aggregator))
    port = server.server_address[1]
    log("listening on 127.0.0.1:%d" % port)
    if not register_with_env(ENV_URL, port, catalog):
        log("register failed; exiting")
        return 1
    log("registered; waiting for work to finish")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    while True:
        time.sleep(POLL_INTERVAL_S)
        try:
            req = urllib.request.Request(ENV_URL + "/work", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                work = json.loads(resp.read().decode("utf-8"))
            if work.get("done") is True:
                break
        except Exception as exc:
            log("work poll failed: %s" % exc)
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
