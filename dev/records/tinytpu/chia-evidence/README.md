# CHIA evidence (trimmed)

What backs the claims in `../README.md` and `docs/source/extensions/chia.rst`.
Only verdicts, diffs, costs and summaries are kept here (~200 KB). The raw
runs -- worker logs, cosim/stress logs, csynth XMLs, generated spec copies --
and the full `chia-isa` branch history are on the tag
**`chia-isa-run1-evidence`**, at `chia_runs/<same directory name>/`:

    git show chia-isa-run1-evidence:chia_runs/isa-run1-20260919/front-end/worker.log

| directory | what it is |
| --- | --- |
| `isa-run1-20260919/` | the first paid run on CHIA2026 ($28.54): its README, `run.json` (incl. the pre-flight record), `summary.json`, `opencode_sessions.json` (per-session cost from opencode's DB), both workers' `variants.jsonl` (every candidate's diff and verdict), `swarm.log`, and the independent acceptance of the one claimed win (`accept-front-end-iter1/`) and of its parametric re-expression (`param_burst.diff`, `accept-param-burst/`). `param_burst.diff` is evidence, not a landed design change. |
| `accept-control-476a70d8/` | no-diff acceptance control of the shipped design: 172 / 262 / 418 / 484 / 686, the baseline `accept.py` records |
| `timeline-476a70d8-16x16x16/` | per-process cosim timeline of the shipped design at 16x16x16; the source of run 1's seed facts, on the agent's reading list |
| `harness-test-*/` | LLM-free harness runs (`results.json`: every case, expected vs. got) and their `accept.py` on case b |
| `isa-smoke-20260919-035443/` | the capped smoke run on the old design and old project (test-adrs): no candidate completed |

Paths inside the JSON (e.g. `/home/sk3463/allo-chia-isa/chia_runs/...`) are
where the files were written at the time; the same relative paths exist on
the tag.
