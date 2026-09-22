# isa-run1-20260919: the first paid CHIA run on CHIA2026

2 workers (`front-end`, `tail`), at most 3 iterations each, per-run cap $30,
model `google-vertex/gemini-3.1-pro-preview` @ global, project
`chia2026-tinytpu` (billing account CHIA2026, `01BF39-94AA3F-36BACB`), harness
@ `c93f0153` (main @ `476a70d8`; baseline 172 / 262 / 418 / 484 / 686).
Tool servers on loopback (127.0.0.1:8000 and :8001). Pre-flight passed before
any worker started (`swarm.log`).

On main this directory is trimmed (verdicts, diffs, costs); `worker.log`,
the cosim/stress logs and `csynth.xml` of both acceptances, and the workers'
spec copies are on the tag `chia-isa-run1-evidence` under
`chia_runs/isa-run1-20260919/`.

## Cost and time (opencode's DB, `opencode_sessions.json`)

| worker | sessions | USD |
| --- | --- | --- |
| front-end | 2 | 5.17 + 8.78 = **13.95** |
| tail | 3 | 4.36 + 1.20 + 9.03 = **14.59** |
| total | 5 | **$28.54** (371 model messages) |

Wall time 99.9 min (17:16 - 18:56 UTC). All five sessions are after the
cutover and google-vertex, so all $28.54 counts against the CHIA2026 cap
(cumulative $28.54 of $100 after this run, including the $0.0013 check call).
Three of the five sessions hit opencode's 40-minute timeout; the loop's
per-call figure records those as $0.00 (`llm spend (this worker)` in
`worker.log` says $5.17 and $1.20), while the DB charges them in full. The
DB figure is the true one, and it is what the run cap and the cumulative cap
use. Both workers stopped on the soft cap before iteration 3.

## Per worker (verified from `variants.jsonl` and the evaluator's verdicts)

**front-end** (angle: 211 cycles of `dma_ld` before the first MAC)

- iter 1 -- widened `dma_ld`'s operand burst loops to 4 DRAM rows (16 packed
  words, 64 bytes) per iteration, with the trip count rounded up. In-loop:
  cosim 172 / 627, bit-exact, stress_isa 492/492, clock 2.431 ns. **Accepted
  by the loop (-59).** The diff ALSO deleted the 260-line module docstring,
  replaced `T = int(os.environ.get("TPU_T", 4))` with `T = 4`, and added a
  dead `MAXDIM = 16` (overridden a dozen lines later by the original env
  read), plus junk comments (`# empty`, `# (truncated microarch_isa.py for
  space)`).
- iter 2 -- rewrote `gemm_program` to delegate to the hand-written program and
  restructured the program layout (M padding, new loop nest). The session hit
  the 40-min timeout; the spec it left deadlocks: `gate:bench_isa` TIMEOUT at
  240 s. Rejected; the debug session was not started (soft cap).

**tail** (angle: accu/dma_st drain after the last PE)

- iter 1 -- the session tried a contiguous `dma_st` write-back through a local
  block buffer, timed out, and left `1/0` lines in the spec; the gate failed.
  The debug session removed those lines "and the dead code", and reported
  "172 (down from 680)" and "686 (down from 1457)". **That report is wrong**:
  the final diff is one added `pass`, i.e. the unmodified design, and it scored
  exactly the baseline 172 / 686. Rejected (+0).
- iter 2 -- added relay kernels between `accu` and `dma_st` (and replaced the
  module docstring with `"""Docstring removed."""`). Timed out; the spec it
  left passed the gate (492/492) and scored 176 / 686: worse. Rejected (+4).

## Independent verification of the one claimed improvement

`accept.py` on front-end iter 1's diff, clean checkout
(`accept-front-end-iter1/`): **172 / 262 / 376 / 425 / 627**, all five TBs
`mismatches = 0`, stress_isa 492/492, `TPU_TB=stress` RTL testbench 0
mismatches over 6 calls at every shape, est. clock 2.431 ns. `claim: win`,
-160 over the five shapes (0 / 0 / -42 / -59 / -59). Area (csynth
estimate, recorded not scored): BRAM18K 42 -> **98** (+133%), LUT 26583 ->
31929 (+20%), FF 17481 -> 24465 (+40%), DSP 14 unchanged -- the price of
partitioning `rbA`/`rbB` for 16 writes an iteration. Worth weighing before
landing: -59 cycles at 16x16x16 (8.6%) for 2.3x the block RAM.

Read by hand: the functional change is only `dma_ld`'s two burst loops. Rows
beyond a program's span are read (the rounded-up group), but only up to a
multiple of 4 <= 16, so every read is inside the MAXDIM x MAXDIM operand and
the extra rows land in `rbA`/`rbB` words no instruction of that program
names. Nothing touches `accu`, `ar`, the AR_RAW_DIST contract, or any
initialisation. At MAXDIM=8 and 12 the candidate still builds and is exact
(`param_check.py` at those configs, 69/69 and 186/186), because the literal
`MAXDIM = 16` is dead. The literal `T = 4` is live, and hard-coding it is why
this candidate is not landable.

> **Correction, 2026-09-22.** This paragraph originally went on to say that
> "T=4 is also the only T main's design supports (it fails `check_program` at
> `TPU_T=8`)". **That is false**, and was the same false claim carried by
> `docs/source/extensions/chia.rst` and `evaluate.py`. Measured on main:
> `TPU_T=8 TPU_MAXDIM=32 param_check.py` prints `PARAM OK: 408/408 runs
> exact`, and `TPU_T=8 TPU_MAXDIM=32 bench_isa.py 32 32 32` prints
> `ALL EXACT`. What fails at T=8 is the HARNESS, and only at MAXDIM=16: three
> test-program generators address column block 2, which exists only when
> MAXDIM/T >= 3, so at T=8/MAXDIM=16 nine of 24 random seeds cannot be
> generated and `param_check` refuses for want of programs (62/63 runs exact,
> no wrong answer). The assumption is MAXDIM/T >= 3, not T == 4.

**Not landable as written**: it deletes the design docstring and hard-codes T.

**The idea survives a parametric re-expression** (`param_burst.diff`, no
model call): `DMA_ROWS` = the largest divisor of MAXDIM whose rows fit one
64-byte beat, docstring intact, the old "bursts are hidden" comment corrected
against the measured timeline. `accept.py` (`accept-param-burst/`): identical
cycles 172 / 262 / 376 / 425 / 627, all bit-exact, stress_isa 492/492, RTL
stress 0 mismatches at all five shapes, 2.431 ns, the same area as the
candidate; exact at MAXDIM=8 and 12. Nothing is landed on main.
