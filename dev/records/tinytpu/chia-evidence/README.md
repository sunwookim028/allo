# CHIA evidence: every run, what it asked, and what it cost

This is the index of the series. One row per agent run, in order, whether or
not it produced a result and whether or not its evidence survived -- a null
result is only interpretable next to the question it was asked, so the
question is recorded here even when the answer was "nothing".

**Adding a run.** Write the pre-registration first, as
`prereg-run<N>-<YYYYMMDD>.md`, and **push it to `main` before the first paid
call** (`examples/tinytpu/chia_agent/README.md`, "Before you spend"). Add the
row below at the same time with the result column empty, and fill it in when
the run ends. `python3 examples/tinytpu/chia_agent/spend.py run <RUN_DIR>`
gives the cost; `swarm.py --status <RUN_DIR>` gives the rest.

## The runs

| # | run | date | question pre-registered | result | deviation | cost | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -- | capped smoke | 2026-09-19 | **none**; two angles only, in `run.json` (is the `vru` tier paying for itself; is the per-`mm` weight prologue worth double-buffering). Old design, old project | **Null, and not about the design.** 0 candidates completed, neither worker finished iteration 1, no cycle number produced. Exposed five harness defects, all fixed in `97308da0` | n/a | **$15.15**, hard cap fired. Billed `test-adrs`, **outside** the CHIA2026 cap | `isa-smoke-20260919-035443/` |
| 1 | `isa-run1-20260919` | 2026-09-19 | **none**; the task template plus two angles in `run.json` (what in the prologue is data dependence and what is ordering; where the tail's time goes) | **One win of four.** `dma_ld` operand bursts widened to 4 rows: 172 / 262 / 376 / 425 / 627, -160 cycles, bit-exact, at 2.3x the block RAM. **Not landable as written** -- hard-coded `T = 4`, deleted the 260-line docstring; both holes became guards. Also caught a false win claim from a debug session | Both workers stopped on the soft cap before iteration 3. One later correction, in the run's own README: "main supports only T=4" withdrawn -- it is the harness that needs `MAXDIM/T >= 3` | **$28.54** of $30, 100 min. Three of five sessions timed out and report $0.00 on `usage` while the DB charges **$22.17** of it | `isa-run1-20260919/`; raw logs on tag `chia-isa-run1-evidence` |
| 2 | `codesign-run2-20260922-143234` | 2026-09-22 | **yes**, in `dev/records/codesign_mapspace.rst`: can an agent find a two-part change where neither part shows anything alone -- a wider AGU term budget *and* a mechanism giving `acc` its step, both in one iteration? Predicted a mechanism reaching only Kt=2 with cycles flat, or a candidate that widens the AGU and stops | **Null; the predicted second branch, in both arms.** Encodable nests rose (3->8, 3->7), **cycles unchanged at 169 / 686**, chosen nest unchanged, area up: `regression`. The less-directed arm made 17 nests encodable of which **10 computed the wrong answer**, the mapper's pick among them; the frozen reference sweep caught all ten | **Four.** Stopped at iteration 2 of 5 by a **cap defect, not its budget** (it summed the whole account); all six calls hit the 2400 s timeout; the "unguided" label is **withdrawn** (a shared system message reached both arms, so there is no unguided condition); "16 wrong nests" corrected to 10 of 17 | **$47.65** of $68 (the broken cap read $52.60) | **none here.** `git show 2fa1b6b9:docs/source/records/codesign_mapspace.rst` -- not an ancestor of `main`; also on tag `archive/codesign-record`. Its commit message carries corrections found nowhere else |
| 3 | `isa-run3-20260924` | 2026-09-24 | **yes**, `prereg-run3-20260924.md`, committed as `b18b8e19` before the run: on the design as it ships (175 / 265 / 421 / 482 / 674), does a two-worker three-iteration search over the **decomposed** design reach a **graded verdict**, and beat the control the same run measures? Predicted yes, "0 or 1 wins", and named in advance that a win would be a rediscovery of `TPU_DMA_WIDEN` | **Primary met, one win, rediscovery as named.** Six evaluations, **all six graded** -- none died at setup, policy, import, invariant or tamper. `tail` iter 3: **175 / 265 / 386 / 435 / 627** against a control of 175 / 265 / 421 / 482 / 674, -129 cycles, every gate clean, 2.431 ns. A 22-line diff; a $0 follow-up shows the burst widening is worth the **entire** -129 and the channel depth **zero** | **Three, in the results rather than the plan.** The prediction held for a softer reason (headroom not spent); the objective is now known **partial** (the widening is worth zero at the scored config but -287 / -583 at MAXDIM=64), with the pre-registration left as written; and the billing instrument failed live | **$20.72** of $60, 167 min. All four large calls ran the full 2400 s timeout and returned no `usage`, so on that field the whole run reads **$0.00** | `isa-run3-20260924/`, incl. `spend.json` |

Cumulative on CHIA2026 after run 3: **$124.26** of the **$500**
`CHIA_TOTAL_CAP_USD` ceiling (`isa-run3-20260924/spend.json`; recomputed by
`spend.py report`). The smoke run's $15.15 is outside it, on `test-adrs`.
About $75 of the cumulative is attributed to no run in `spend.py report`:
sessions from `smoke.py`, from the abstraction track and from manual use, all
of which the cutover rule counts against CHIA2026 because opencode stores no
GCP project with a session.

## Everything else in this directory

None of these is an agent run and none cost money.

| directory | what it is |
| --- | --- |
| `prereg-run3-20260924.md` | run 3's pre-registration, and the template for the next one |
| `accept-control-476a70d8/` | no-diff acceptance control of the design at `476a70d8`: 172 / 262 / 418 / 484 / 686, the baseline `accept.py` records |
| `timeline-476a70d8-16x16x16/` | per-process cosim timeline of that design at 16x16x16; the source of run 1's and run 3's seed facts, on the agent's reading list |
| latest $0 verification | **90/90 cases, 39.4 min** on `--phases s,control,e,c,g,loop` (2026-09-24) -- the first pass of the loop phase since the objective merge, which had left the frozen set not closed under the evaluation's imports, so every candidate died at stage `model` |
| `harness-test-*/` | LLM-free `test_harness.py` runs (`results.json`: every case, expected against measured) and their `accept.py` on case b |
| `codesign-suite-20260922.json` | the co-design harness's own suite, 11/11 cases, $0 |

## What is kept, and what is on the tag

Only verdicts, diffs, costs and summaries are kept here (~200 KB). The raw
runs -- worker logs, cosim and stress logs, csynth XMLs, generated spec copies
-- and the full `chia-isa` branch history are on the tag
**`chia-isa-run1-evidence`**, at `chia_runs/<same directory name>/`:

    git show chia-isa-run1-evidence:chia_runs/isa-run1-20260919/front-end/worker.log

Paths inside the JSON (e.g. `/home/sk3463/allo-chia-isa/chia_runs/...`) are
where the files were written at the time; the same relative paths exist on the
tag.

A committed run directory is enough to reconstruct the run with no memory of
it: `swarm.py --status <dir>` reads it back, and `spend.py run <dir>` prices
it. Both work on every directory above that has a `run.json`.
