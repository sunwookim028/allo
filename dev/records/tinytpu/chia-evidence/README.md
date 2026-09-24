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
| -- | capped smoke | 2026-09-19 | **None.** Two angles only, in `run.json`: is the `vru` tier paying for itself, and is the per-`mm` serial weight prologue worth double-buffering. Old design, old project | **Null, and not a result about the design.** 0 candidates completed; neither worker finished iteration 1; no cycle number was produced for any proposal. It exposed five harness defects (60 s MCP timeout against a 2-4 min cosim, a hung evaluator blocking the tool server, a 900 s deadlocked candidate, a silent retry of a 40-minute timed-out prompt, unified diffs as the dominant failure mode), all fixed in `97308da0` | n/a | **$15.15**, hard cap fired. Billed **`test-adrs`, not CHIA2026**, so outside the $500 cap | `isa-smoke-20260919-035443/` |
| 1 | `isa-run1-20260919` | 2026-09-19 | **None.** The question is the task template plus two seeded angles in `run.json`: lower cosim cycles; `front-end` = what in the prologue is data dependence and what is ordering; `tail` = where the tail's time goes | **One win of four candidates.** `front-end` iter 1 widened `dma_ld`'s operand bursts to 4 rows: re-accepted on a clean checkout at 172 / 262 / 376 / 425 / 627 (-160 cycles), bit-exact, stress 492/492, at 2.3x the block RAM. **Not landable as written** -- the diff hard-coded `T = 4` and deleted the 260-line design docstring; re-expressed parametrically at $0 it gives the same numbers. Both holes became the guards in the README's item 3d. Also caught a false win claim: `tail`'s debug session reported "172 (down from 680)" for a diff that was one `pass` and scored exactly baseline | n/a. Both workers stopped on the soft cap before iteration 3. One later correction to the write-up, inside the run's own README: the claim that main's design supports only `T=4` is withdrawn -- it is the harness that needs `MAXDIM/T >= 3` | **$28.54** of a $30 cap, 100 min. Three of five sessions hit the 40-minute timeout and report $0.00 on `usage` while the database charges **$22.17** of that total | `isa-run1-20260919/` (raw logs on tag `chia-isa-run1-evidence`) |
| 2 | `codesign-run2-20260922-143234` | 2026-09-22 | **Yes**, in `dev/records/codesign_mapspace.rst` ("Pre-registration: what the search is being asked, and what I expect"), written before the run: can an agent find a two-part change where neither part shows anything on its own -- a wider AGU address-term budget *and* a mechanism giving `acc` its step, both right in one iteration. Predicted most likely a mechanism reaching only Kt=2 with cycles flat; second most likely a candidate that widens the AGU and stops | **Null, and the predicted second branch held in both arms.** Both arms passed every gate at iteration 2 and raised encodable nests (3->8 directed, 3->7 less directed), with **cycles unchanged at 169 / 686**, the chosen nest unchanged, and area up -- classified `regression`. The less-directed arm's iteration 1 made 17 nests encodable of which **10 computed the wrong answer**, the mapper's pick among them; the frozen reference-model sweep caught all ten | **Four, all recorded.** (1) Stopped after iteration 2 of 5 by a **harness defect, not its budget**: the per-run cap summed the whole account ($52.60) instead of the run's own sessions ($47.65), and iteration 3 would have started. Fixed -- every cap now keys on the run's title tag. (2) All six model calls hit the 2400 s timeout, so none is a finished agent answer. (3) The "unguided" label on the second arm is **withdrawn**: a shared system message handed both arms the corrected diagnosis, so the run contains no unguided condition; the arms are "directed" and "less directed". (4) "16 wrong nests" corrected to 10 of 17 | **$47.65** run-own of a $68 cap (the broken cap read $52.60) | **No directory here.** The write-up is on commit `2fa1b6b9`, which is not an ancestor of `main`: `git show 2fa1b6b9:docs/source/records/codesign_mapspace.rst`, also on tag `archive/codesign-record`. Its commit message carries five corrections found nowhere else |
| 3 | `isa-run3-20260924` | 2026-09-24 | **Yes**, `prereg-run3-20260924.md`, committed as `b18b8e19` before the run: on the design as it ships (175 / 265 / 421 / 482 / 674), does a two-worker three-iteration search over the **decomposed** design produce candidates that reach a **graded verdict**, and does any beat the control the same run measures? Predicted yes, "0 or 1 accepted wins", and named in advance that any win would be a rediscovery of `TPU_DMA_WIDEN`, already in the tree and off by default | **Primary met, one win, and the named rediscovery held.** Six evaluations, **all six graded** -- none died at setup, policy, import, invariant or tamper. `tail` iter 3 accepted: **175 / 265 / 386 / 435 / 627** against a control of 175 / 265 / 421 / 482 / 674, i.e. -129 cycles (-7.0% at 16x16x16, -9.8% at 16x16x8), every gate clean, 2.431 ns. The 22-line diff is `_WIDEN = TpuParams.widest_burst(T, MAXDIM)` plus an `ac2sp` channel deepened `QD`->32; a $0 follow-up (`accept-burst-only/`) shows the burst widening is worth the **entire** -129 and the channel depth exactly **zero**. Workers edited `ip/units/`; 4 of 6 candidates were inexpressible under the old editable set | **Three, all in the results section rather than the plan.** (1) The prediction held for a softer reason than expected -- the headroom was not spent. (2) A **late qualification**: the objective (summed cosim cycles at 4x4x4 + 16x16x16) is now known to be partial -- the burst widening is worth zero cycles on the model suite at the scored config but -287 / -583 at MAXDIM=64. The pre-registration is left as written. (3) The **billing instrument failed live across the whole run** | **$20.72** of a $60 cap, 167 min. All four large iteration calls ran the full 2400 s timeout and returned no `usage`, so on the `usage` field the entire run reads **$0.00**. `spend.json` is the database figure | `isa-run3-20260924/` |

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
