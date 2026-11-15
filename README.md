# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/rabakta/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/rabakta/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, Python 3.13 standard library only, serving an OpenAI-compatible
(`POST /v1/chat/completions`) and an Anthropic-compatible (`POST /v1/messages`)
API over ten simulated free-tier providers. Every request is routed to the
best free model that meets its requirements — tools, vision, JSON mode, long
context — with failover across providers on 402/429/5xx, and the catalog it
read is reported back byte for byte.

Practice trials against the kit's own env + scorer: **99/100 on all three**
(quality 99, served 100, wire 100, failover 100, catalog 100; 20/20 valid).
99 is the ceiling: the single 1475 model has ~10 successful calls for 17
requests that want it, so at least 7 requests must drop one tier to 1470.

## Run it

```sh
MAKEFUN_PROMPT_FILE=prompt.txt MAKEFUN_ENV_URL=http://127.0.0.1:8080 python3 main.py
python3 main.py --list
```

`main.py` reads the trial brief at `$MAKEFUN_PROMPT_FILE` (provider directory,
sealed quality table, workload manifest), probes every provider's catalog
route, registers its catalog with the env at `$MAKEFUN_ENV_URL`, serves the
20 requests, polls `$MAKEFUN_ENV_URL/work` every 100 ms and exits 0 when the
work is done. `python3 main.py --list` prints the catalog document — the same
bytes the register body carries. No wall clock, no unseeded randomness, no
dependencies: only the standard library.

Routing, in one line: candidates are ranked by `rating_milli` minus a load
penalty of 5 milli per call already made to their provider (divided by the
request's weight, so weight-2 requests spend top-provider budget first), with
a heavy penalty for a provider that just failed twice in a row; a provider
that refuses is never retried for the same request. Wire fidelity is exact:
system text, last user text, image bytes, tool names in order, max_tokens,
stop sequences and JSON mode are preserved through every translation, and
streamed OpenAI upstream calls carry `stream_options.include_usage` so the
usage chunk exists.

---

<sub>Entered on [make.fun](https://make.fun/?ref=wm6bb8x7&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
