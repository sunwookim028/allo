# ═══════════════════════════════════════════════════════════════════════
# HANDOFF — START HERE (simulator shell) · updated 2026-07-27
# ═══════════════════════════════════════════════════════════════════════

## 2026-07-27: Task 1 DONE. Task 2 in progress (plumbing committed, detection next).

**Resume point: write `_emit_deadlock_guard` in `simulator.py`, then wire it into the
3 blocking spin sites (lines ~1365 cross-call, ~1666/~1776 local).** Design below.

Task 1 (cost model) is complete — commits `ebd572f` (cycle read-out), `676f059`
(charge before lowering), `a48eacb` (real per-op latencies + II).
- **`mod.get_cycles()` now exists** → `SimCycles{per_pe, makespan}`. There was no way
  to read a cycle count at all before; DSE can now score a design.
- The clock is **no longer "all-ones" and no longer purely logical** (the old text
  lower in this file is stale on both counts). Typed latencies: fp32 add 4 / mul 3 /
  div 14, int mul 1, int div ≈ bitwidth, BRAM load 2 / store 1, register-resident
  scalars 0, structural 0, and `(n-1)*II + body` for pipelined loops.
- **Calibrated against Vitis csynth: 7 modelled vs 6 reported** on a blocking
  producer/consumer. Caveat — that is ONE point, two trivial integer PEs; float and
  BRAM latencies remain uncalibrated. Say "cycle-approximate, 1.17× at one point",
  not "cycle-approximate" flat.
- Biggest finding: **NB designs csynth to `undef` latency** — there is no static
  schedule to ingest for exactly the designs this work targets, which kills
  OmniSim-style schedule ingestion as an accuracy route. Blocking designs report
  fine. Details + the csynth recipe: memory [[csynth-cost-model-ground-truth]].
- Known gaps, deliberately NOT chased (they shift every DSE score by a constant, so
  they don't change rankings): `load_buf`/`store_res` wrappers aren't clocked at all
  (Vitis charges `store_res0` 11–12 cycles); per-op microbenchmark calibration.

## 2026-07-28: Task 2 DONE — all 3 deadlock classes detected, no more silent hangs.

`ed99cc2` plumbing → `3831021` peer-done proof → `62d5e4b` circular watchdog.
Was: 3 of 4 cases hang forever with no output. Now all 4 resolve in 1–3 s.

| case | before | after |
|---|---|---|
| `control` | COMPLETED | COMPLETED |
| `mismatch_get` | hangs | DEADLOCK (peer-done, names stream+side) |
| `overfill_put` | hangs | DEADLOCK (peer-done, names stream+side) |
| `circular` | hangs | DEADLOCK (circular wait) |

**Two detectors, different confidence — keep them distinguished.**
- **peer-done = a proof.** A PE blocked on a stream whose peer already returned can
  never unblock. No threshold, no false positives. Keys off `_CLOCK_DONE_BIT`.
- **circular = a heuristic.** Counts consecutive polls where every PE is
  blocked-or-finished (`ALLO_SIM_DEADLOCK_POLLS`, default 16384 ≈ 1 s). An
  *instantaneous* all-blocked reading is NOT sufficient (a PE can be counted blocked
  with its condition already satisfied) — the consecutive-poll run is what makes it
  robust. Its message says the reporting PE is one participant, not the root cause.

**Independent of the cost model** — verified by running the whole matrix under
`ALLO_SIM_CLOCK=cycle`, which bypasses the per-op latency table entirely. Detection
keys off *completion*, not simulated time. But it DOES need the clock plumbing:
no `sim.clock` PEs → `n_pes == 0` → guard is a no-op and the design hangs as before.

**Trap that cost real time — `MemRefDCE.cpp:28` erases ANY op with results and no
uses, with no side-effect check.** All 10 `memref.atomic_rmw` ops were emitted and
then silently deleted before LLVM, so the watchdog was dead code and `circular` still
hung with zero diagnostics. Same trap as [[allo-try-put-unused-ok-dce]]. Fix: sink the
atomic's old value into a live global slot. **Always verify emitted-vs-survived at the
IR level** (`str(module).count(...)` before the pipeline vs after) — the helper was
provably called 10 times and still produced nothing.

Still uncovered: the `try_put`/`try_get` **time barriers** (`_emit_read_barrier` /
`_emit_write_barrier`) are not instrumented, so a PE parked there is not counted
blocked — undercounts, costing detection rather than causing false positives.
**Livelock** in NB retry loops is missed entirely (they keep ticking and re-entering,
so they never look blocked) — likely the bigger gap for the agents' generator.

---

Task 2 history — `ed99cc2` had phase 0 + phase 1 plumbing, no behaviour
change yet. Baseline: `control` COMPLETED, the other 3 cases hang and are killed at 25 s.
- `DeadlockError` exists; `deadlock_worker.py` prints COMPLETED / DEADLOCK / nothing.
- Read-out global widened N → N+3, extra slots `[flag, stream_id, role]`; declared by
  `_declare_sim_global` BEFORE stream lowering (spin loops emit `get_global` against it).
- `_SIM_CTX`/`_stream_id` carry `n_pes` + per-stream ids into the deep lowering chain.
- **The design idea:** `_CLOCK_DONE_BIT` (added for the read-out) makes detection
  *exact*. A PE blocked on a stream whose peer has already finished is a proof, not a
  heuristic — no threshold, no false positives — and covers starvation +
  back-pressure (2 of 3 classes). Plan: in the spin `after` block, if the peer clock
  has DONE set, write `(stream_id, role)` + flag into the global; add `flag == 0` to
  the `before` condition so every blocked PE unwinds; Python raises after the run.
  Note the `after` block is already terminated when the guard is added — insert with
  `InsertionPoint(beforeOperation=<after_block terminator>)`.
- Still open: **circular wait** needs a no-progress watchdog (heuristic, phase 2), and
  **livelock** — NB retry loops call `_tick_clock`, so their clocks keep advancing and
  the frozen-clock signature misses them entirely. May matter more than circular wait
  since the agents' generator emits NB wirings.

Env note: HOME quota hit 100% mid-session and blocked ALL writes (`EDQUOT`). Freed by
`conda clean --all` (miniconda3/pkgs was 3.8G). Watch it — Vitis projects are ~20 MB each.

# ═══════════════════════════════════════════════════════════════════════

This shell works on **Allo's dataflow JIT simulator** (`df.build(target="simulator")`,
file `allo/backend/simulator.py`, branch `wire`). A **separate "agents" shell** works
on interface/IP generation (`agents/` folder). **The two tracks join at DSE**: the
agents shell *generates* interconnect designs; this shell provides the *evaluate* half
(cost model + deadlock detection) that scores them. So the top-priority items below are
exactly what the agents' DSE loop will need.

## Current state (committed, branch `wire`)
The **DAM-lite timing layer is done and non-blocking is deterministic**:
- `ca437b0` spin-advance (failed try_get ticks the clock)
- `340363e` D2+D3+occupancy — write-side barrier; **`nb_nondeterminism.py` 23 distinct → 1**
- `198b415` clock-threading crash fix (non-stream / helper-calling PEs)  ← HEAD

Validated: `nb_nondeterminism` single outcome; `read_barrier_test.py` `[8]`; `test_df_unit`
(all 4), `test_stream_ops_sim`, `test_region_stateful`, nb_simple/nb_scalar pass;
systolic 2×2 err ~0. The distributed timing layer works; details in memory
`simulator-timing-layer-wip` and the design sections lower in this file.

## Environment / how to validate
```bash
source /home/zsm9/miniconda3/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build-rhel8   # build-rhel8 ONLY
export PYTHONPATH=/home/zsm9/allo_sup && export OMP_NUM_THREADS=8
python tests/dataflow/test_df_unit.py            # golden
python tests/dataflow/test_region_stateful.py
python simulator_profiling/nb_nondeterminism.py  # determinism metric (expect single outcome)
python simulator_profiling/read_barrier_test.py  # expect [8]
```
Gotchas (memory): interactive `conda activate` needed (plain `conda run` → "Unknown function
top"; overriding LLVM_BUILD_DIR to `build/` → GLIBC abort). Force `PYTHONPATH=/home/zsm9/allo_sup`
(import otherwise grabs installed `/home/zsm9/allo`). See [[simulator-llvm-build-dir-run]],
[[allo-two-checkouts-trap]].

## TASKS (priority order)

1. **Cost model for DSE (the #1 join-point item).** The clock is all-ones today =
   a **logical** clock (deterministic ordering only, NOT real cycles). For DSE
   *performance numbers*, refine `_op_latency` in `simulator.py` toward real per-op /
   II latencies. Supervisor stance is explicit: **correct, not cycle-accurate** — "try
   many things." This is what lets a generated design be *scored*, so do it first.

2. **Deadlock detection.** The sim currently **hangs forever** on deadlock with no
   diagnostic (profiling finding #5). Add detection (watchdog on no-progress, or the
   converge-fails-with-error of task 4). The agents' generator will produce broken
   wirings — it needs "deadlock: here" instead of an infinite hang.

3. **Wire / valid-only / valid-ready IN THE SIMULATOR.** These exist **only for the
   SystemC backend today** (`allo/ir/types.py` `Wire`/`Channel`; guarded in
   `allo/backend/hls.py`). The simulator has only `Stream` (blocking + the new
   deterministic non-blocking). Bring them to the sim so it can model + DSE the
   protocols the agent picks. Design = the unified `max(producer_time, consumer_time)`
   handshake-over-timestamps rule (see "Wire / ValidOnly / ValidReady as one handshake"
   section below). Rules: **pure-wire-connected kernels → fuse into one (latency 0)**;
   a bare wire doesn't self-synchronize; every feedback cycle needs a registered
   element (Stream) or it's a combinational loop. Op×primitive: `try_*` works on
   Stream/Channel, NOT Wire; only Stream has empty/full. Ref: `agents/INTERCONNECT.md`,
   memory [[wire-channel-dataflow-types]].

4. **(Bigger, optional) Three-phase DES rewrite — the ARC2HS model.** A colleague's
   event-driven simulator (setup → block(converge) → commit) gives determinism,
   combinational loops, AND deadlock detection **structurally**: the block phase
   iterates same-time to a fixed point with **monotone pressure** (only add
   backpressure during block, resolve releases in commit → provably terminates; a real
   combinational loop hits the iteration cap → *errors* instead of hanging) + **lazy
   cancellation**. Recommended shape = **hybrid**: keep JIT'd PE bodies as threads,
   adopt the three-phase converge only for the **channel/interconnect layer**. De-risk
   first: how a PE gets "stepped" (thread-per-PE + per-cycle barrier vs coroutine).
   This would subsume tasks 2 and much of 3. ARC2HS spec + NoC/Local model code are in
   the 2026-07-26 conversation; consider saving them under `simulator_papers/` or a note.

## Key decisions / facts (don't relearn)
- All-ones clock = **logical** (determinism); real cycles need an HLS II/latency schedule
  or DAM-style annotations — refine via `_op_latency`. Cycle-*approximate* is the ceiling.
- **Wire/Channel = SystemC-only; Stream = universal.** Non-blocking (`try_*`) = Stream &
  Channel; **never Wire** (a wire has no "empty" — always its current value).
- Spin-wait `scf.while` loops are excluded from clock increments (else non-deterministic);
  failed NB polls tick the clock so spin-waits still advance sim time.
- The agents shell's blocks (routers/PE) run on the **current** sim (Stream) — no new sim
  feature is needed to *run* them; new features (1–4) are for *scoring* them.

## Pointers
- Memory: [[simulator-timing-layer-wip]] (resume doc), [[simulator-profiling-harness]],
  [[wire-channel-dataflow-types]], [[sim-direction-wires-dse-agents]] (supervisor
  direction), [[simulator-combinational-wire-design]].
- Deep reference: the rest of THIS file (DAM-lite design; current-sim analysis; paper
  analysis incl. OmniSim/DAM). Profiling harness + findings: `simulator_profiling/`.
- Notes: `notes/PITFALLS_DATAFLOW_REGION.md`, `STATE.md`, `BRANCHES.md`.

# ═══════════════════════════════════════════════════════════════════════

# claude_simulator — Session Goals

This file tracks the goals and working context for the **simulator** shell:
extending Allo's dataflow simulator.

## Goals

1. **Non-blocking reads/writes**
   Extend Allo's simulator to support non-blocking `get`/`put` (stream
   read/write) semantics — a read/write that does not stall the caller when the
   channel is empty/full, but instead reports success/failure.

2. **Wires and handshake protocols in a dataflow region**
   Extend Allo's simulator to support:
   - **Wires** — raw combinational connections (unbuffered).
   - **ValidOnly** protocol — data qualified by a `valid` signal only.
   - **ValidReady** protocol — full back-pressured handshake (`valid` + `ready`).

3. **Read related papers**
   The user will add papers that may be relevant. Read them to gather ideas for
   the simulator design (protocols, scheduling, combinational cycles, etc.).

4. **Analyze Allo's current simulator in detail**
   Produce a detailed analysis of how Allo's dataflow simulator currently works,
   as the foundation for the extensions above.

## Working context

- Branch: `wire`
- Related notes/memory:
  - `notes/PITFALLS_DATAFLOW_REGION.md`
  - combinational-wire design + wire/channel dataflow types (auto-memory)
  - nb-streams file map (auto-memory)

## Status / progress log

- 2026-07-23: **Profiling round 1 — directly measured where sim time goes**
  (turning the `mininpu research team.docx` inferred claim ">95% OS scheduling"
  into a measured curve). Harness: `scratchpad/profile_worker.py` +
  `profile_driver.py` (one subprocess per config, `OMP_NUM_THREADS=PE`,
  `getrusage` deltas around a warm N=30 loop, build isolated). Host = **64 cores**
  (doc's host = 16), so we can see the transition. Parametric systolic GEMM.

  | config | PE | PE/core | build s | sim ms ±sd | busy cores | sys_frac | ivcsw/sim | RSS MB |
  |---|--:|--:|--:|--:|--:|--:|--:|--:|
  | 2×2×2 | 12 | 0.25 | 0.84 | 0.60 ±0.21 | 14.8 | 0.005 | 0.1 | 269 |
  | 4×4×4 | 32 | 0.56 | 2.16 | 0.87 ±0.45 | 30.4 | 0.062 | 45 | 274 |
  | 6×6×6 | 60 | 1.0 | 4.18 | 1.55 ±1.02 | 46.9 | 0.046 | 627 | 280 |
  | 8×8×8 | 96 | 1.56 | 7.59 | 2.63 ±2.54 | 51.3 | 0.777 | 29,161 | 287 |
  | 10×10×8 | 140 | 2.25 | 11.9 | 23.0 ±16.4 | 58.0 | 0.794 | 718,672 | 296 |
  | 12×12×8 | 192 | 3.06 | 17.7 | 61.3 ±41.3 | 57.3 | 0.795 | 2,140,751 | 311 |
  | 14×14×8 | 252 | 4.0 | 25.4 | 99.3 ±44.3 | 58.7 | 0.790 | 4,376,961 | 328 |
  | 16×16×8 | 320 | 5.06 | 35.4 | 179.8 ±62.9 | 58.8 | 0.794 | 7,790,120 | 347 |

  **Findings:** (1) Two regimes with the knee **exactly at core count (64)**:
  below 1 PE/core the sim is **user-mode spin-bound** (`sys_frac`≈0.005–0.06,
  `ivcsw`≈0), NOT scheduling-bound; above it, `sys_frac` plateaus at **~0.79**
  (kernel/futex/nanosleep-bound) — the doc's regime, measured as ~80% not >95%.
  (2) **Involuntary context switches** are the fingerprint and explode
  super-linearly to **7.8M per single sim** at 324 PE (~43M/s). (3) Parallelism
  **saturates at ~59 cores**; beyond 64 PE it's pure context-switch overhead
  (sim 2.6→180 ms for the same work-shape). (4) Variance grows to `sd/mean`≈71%
  (scales the proven non-determinism). (5) NOT bottlenecks: memory (RSS 269→347 MB,
  stacks page lazily) and compute (never appears). (6) Build dominates single-shot
  (0.84→35.4 s, ~200× a warm sim at 324 PE), ∝ PE count. Raw:
  `simulator_profiling/profile_results.json`. Baseline for the wire/timing-layer
  work: the metric to beat is ivcsw/sim + sys_frac + variance.

- 2026-07-23: **Profiling round 2 — syscall breakdown of the kernel time**
  (`strace -f -c`, per-sim = counts(NRUNS=1)−counts(NRUNS=0); harness
  `simulator_profiling/profile_strace.py`, log `strace_output.txt`). **Robust
  result: `futex` is 81–95% of syscall TIME at every config** → the ~0.79 sys_frac
  is dominated by threads parking/waking on **OpenMP futexes** (the
  `omp.critical`/atomic on head/tail, the region-join barrier, libomp thread-pool
  coordination) — NOT `usleep`. The spin-loop syscalls `sched_yield` (from
  `omp.taskyield`) and `nanosleep` (from `usleep(1)`) are high-COUNT, low-TIME:
  counts climb with PE (yield 0→8.9k, nsleep 16→5.5k per sim) but each is cheap.
  **Caveat:** strace serializes execution so it under-reproduces the oversubscribed
  futex storm — the subtracted futex/sim *counts* are noisy/near-zero (build's own
  futexes cancel out); treat counts as composition, magnitude = the clean unstraced
  ivcsw/sim. **Implication:** the fix is fewer/coarser synchronizations (timestamp
  handoff instead of per-element critical sections), not `usleep` tuning (doc's <1%).

- 2026-07-23: **Profiling round 3 — libomp knob sweep** (harness
  `simulator_profiling/profile_knobs.py`, log `knobs_output.txt`; 6 knobs × 2
  oversubscribed configs, N=20). **HEADLINE: `OMP_WAIT_POLICY=passive` = free
  4–6.5× speedup, no code change.** 8×8×8: 11.76→1.81 ms (6.5×); 12×12×8:
  9.83→2.50 ms (3.9×). **Involuntary ctx switches collapse ~1000–4000×** (185,934→30;
  259,950→190), busy_cores 56→5 (exposes that the design is compute-light — baseline
  was ~all wasted spinning), and **variance nearly vanishes** (sd 16.6→0.20). Cause:
  LLVM libomp default (`KMP_BLOCKTIME≈200ms`) spins-before-sleeping at OMP sync
  points; passive makes threads sleep immediately so the OS stops preempting
  hundreds of spinners. So most of the doc's ">95% OS scheduling" is **recoverable
  busy-wait, not irreducible oversubscription.** Other knobs: `wait=active` helps at
  low oversub (5.7× at 1.56×) but is catastrophic at high (0.11× / 9× slower at
  3.06× — all threads spin); **pinning (`OMP_PROC_BIND`) harmful when oversubscribed**
  (0.11–0.40×). All knobs PASS correctness. **Caveats:** (1) perf win, NOT a fix for
  NB *semantic* non-determinism (still needs the timing layer; passive only cuts
  timing jitter); (2) speedup is large because this design is compute-light — confirm
  on a compute-heavy PE. **Action:** set `OMP_WAIT_POLICY=passive` by default next to
  the `OMP_MAX_ACTIVE_LEVELS` set in `build_dataflow_simulator` (`simulator.py:1432`),
  before the ExecutionEngine loads libomp.

- 2026-07-23: **IMPLEMENTED `OMP_WAIT_POLICY=passive` default** in
  `build_dataflow_simulator` (`simulator.py`, next to `OMP_MAX_ACTIVE_LEVELS`;
  only set if unset → respects user override). **Re-verified with the full 8-config
  sweep** (`simulator_profiling/sweep_output_passive.txt`,
  `profile_results_passive.json`; shell `OMP_WAIT_POLICY` unset so it tests the CODE
  default — every config records `omp_wait_policy=passive`). Speedup vs the baseline
  sweep grows with size: **0.82× (2×2×2, undersubscribed — tiny sub-ms cost) →
  2.9× (6×6×6) → 12.1× (8×8×8) → 28.1× (10×10×8) → 20.5× (16×16×8)**. ivcsw/sim
  collapses ~7000× at 324 PE (3.1M→456); busy cores 55–59→4–12; variance ±sd
  25–86 ms→0.15–3.5 ms. **All 8 PASS** (worst err 4.8e-7). Only cost: ~0.8× when
  PE<cores (spin wakes faster than futex sleep when cores are free) — vanishes once
  PE≥cores, the regime the sim is actually used in. Baseline table preserved in
  `sweep_output.txt`.

- 2026-07-23: **Profiling round 4 — compute-heavy test** (does "compute negligible"
  hold, and does passive still win when PEs actually work?). Harness
  `simulator_profiling/profile_worker_heavy.py` + `profile_heavy_driver.py`: same
  systolic topology, but each interior PE does FLOP loop-carried madds/step; swept
  FLOP 0→1,048,576 at 10×10×8 (144 PE), active vs passive, N=15. **Findings:**
  (1) **"Compute negligible" is an artifact of a compute-light benchmark** — under
  passive, sys_frac collapses 0.35→0.026, busy_cores climb 6.7→23.5, sim_ms grows
  2→100 ms tracking FLOPs. Holds only near FLOP=0. (2) **Passive wins at EVERY
  intensity and never hurts**: passive/active speedup 32×(FLOP=0) → 3.3×(FLOP=1M),
  shrinking but always ≥3.3× even fully compute-bound — resolves the earlier caveat.
  Cause: under active, ~51 cores stay busy but only ~23-worth is real compute; the
  rest is busy-wait spinners **stealing cores from compute threads** under
  oversubscription. (3) **Active masks compute, passive reveals it** — under active
  sys_frac stays ~0.7–0.8 and sim_ms barely moves over a 128,000× FLOP increase
  (66→328 ms); passive makes sim_ms ∝ work → a more honest perf model. Passive
  variance far tighter (±0.2–9 ms vs ±27–61 ms). (4) **Cross-policy correctness
  confirmed** (seeded inputs): active & passive give identical checksum,
  max_dev=0 — passive changes nothing but speed. (Initial run showed spurious
  checksum DIFF from unseeded per-subprocess RNG; fixed with `np.random.seed(0)`.)
  Raw: `profile_heavy_results.json`, log `heavy_output.txt`.

- 2026-07-23: **Profiling round 5 — deadlock detection** (does the sim actually
  catch deadlock, its stated purpose per the doc?). Harness
  `simulator_profiling/deadlock_worker.py` + `deadlock_driver.py`: a correct control
  + 3 deliberately-deadlocking designs, each under `timeout -s KILL 20`. **Result:**
  control COMPLETES (1.6 s); **all 3 deadlock classes HANG forever** — starvation
  (get with no put), back-pressure (put into never-drained full FIFO), circular wait
  (mutual get-before-put). **The sim does NOT detect deadlock — it exposes it only as
  an unbounded hang**: no error, no diagnostic, no localization (which FIFO/PE),
  can't distinguish deadlock from slow/running; needs an external timeout. Qualifies
  the doc's "checks for deadlock" claim (exposes ≠ detects). **Note:** with the new
  passive default, a deadlocked sim now SLEEPS (futex-park) instead of busy-spinning →
  a quiet idle hang with no CPU spike, *harder* to notice. Real detection is a
  capability the DAM/OmniSim timing layer would unlock (blocked-no-progress /
  task-tracker-zero). Log `deadlock_output.txt`.
  **Allo gotcha found:** `S.put(i)` with a raw `range` loop index fails to build
  (`memref.store` index-vs-i32 mismatch — the induction var is `index`-typed); assign
  to a typed local first (`x: int32 = i; S.put(x)`). Bare-index put ≠ typed put.

- 2026-07-23: **Profiling round 6 — feedback (cyclic dataflow)** (all prior tests
  were feed-forward; does the run-to-completion OMP + blocking-FIFO model handle a
  graph cycle?). Harness `simulator_profiling/feedback_worker.py` +
  `feedback_driver.py`; log `feedback_output.txt`. **Results:** a properly-**primed**
  feedback loop works — ring2 (2-kernel cycle) → got=10=expected, ring3 (3-kernel)
  → got=20=expected, both **deterministic** across 3 runs. The **same topology
  unprimed** (consume-before-produce) **deadlocks** (killed at 15 s). **Findings:**
  (1) buffered feedback (FIFO cycles) already works correctly & deterministically —
  NOT a gap; (2) feedback correctness is the *design's* responsibility (priming +
  FIFO depth), which the sim faithfully reflects (works when correct, hangs when not);
  (3) buffered feedback is deterministic (blocking = timing-immune even with cycles),
  unlike the NB case. **Implication:** narrows the extension's hard case — the
  timing-layer/fixed-point scheduler is needed only for **combinational** feedback
  (zero-latency wire loops that can't be primed), NOT for ordinary FIFO feedback.

---

# Design: correct non-blocking via a default cost model (DAM-lite)

Goal: make `try_get`/`try_put`/`empty`/`full` **deterministic and order-faithful**
instead of scheduler-decided (round 0 proved the current ops are not). Idea from
**DAM (ISCA'24)** — per-PE local simulated time + timestamped tokens; NB = a
timestamp comparison. The *timestamps* come from a **default cost model** (no HLS
schedule needed); accuracy can be refined later toward Allo's II/latency schedule
(OmniSim's approach).

## The clock: a default cost model
Each PE (OMP thread) gets a monotonic integer **simulated-time counter `T^S`**
(init 0), advanced by a static **latency table**:
| op | default latency (ticks) |
|---|---|
| arith / logic / compare | 1 |
| memref load / store | 1 |
| index cast / address calc | 0 |
| stream put / get (the transfer) | 1 |
| control-flow branch | 0 |
| loop (body `B`, trip `n`) | `n · lat(B)` sequential; `(n-1)·II + lat(B)` if pipelined |

Cheapest usable version: **every op = 1 tick**, so `T^S` = count of ops that PE
has executed. Deterministic and monotonic — enough to fix NB non-determinism;
the table just makes it less coarse. Implementation: at lowering, precompute a
**static per-basic-block latency** and insert one `T^S += <block_latency>` per block.

## Timestamped FIFOs + NB semantics
- Ring-buffer slot becomes `(value, t_produced)`. `put(v)`: store
  `(v, T^S_producer)`, then `T^S_producer += 1`.
- Publish each PE's `T^S` in shared memory (atomic) so peers can read it.
- **Blocking `get`**: dequeue head, `T^S = max(T^S, head.t_produced) + 1`.
- **`try_get()` / `empty()` at query time `T = T^S_consumer`** — two steps:
  1. **Time barrier (the crux):** wait until the producer's published clock
     `T^S_producer ≥ T` (or the producer has terminated). Only then is the answer
     *determined* — otherwise an earlier-timestamped element could still arrive.
     This wait is on *simulated-time progress*, which is deterministic, so the
     **answer is deterministic even though the thread still physically blocks.**
  2. **Answer:** success iff `head.t_produced ≤ T`. On success dequeue and advance;
     else return `(dummy, false)`. `empty()` = same predicate, no dequeue.
- **`try_put()` / `full()`**: symmetric, using a reverse **read-cursor timestamp**
  (consumer publishes the sim-time at which it freed each slot) so "full at time T"
  is well-defined (DAM's response-channel back-pressure).
- **Termination:** a finished PE publishes `T^S = +∞` so waiters don't hang forever.

## Why this is correct
The NB result is a pure function of the two PEs' *deterministic* clocks and the
timestamped tokens — not of OS scheduling. Same program + input → same answer every
run (fixing round 0). The threads still run concurrently; they synchronize only by
reading each other's monotonic `T^S` (DAM's SVA/SVP: atomic peek, park only if
ahead). This is **conservative (Chandy–Misra–Bryant) parallel simulation**.

## Implementation progress
- **Phase 0 (DONE, uncommitted) — refactor.** Extracted the four NB/status op
  lowerings (`empty`/`full`/`try_put`/`try_get`), previously duplicated in the
  cross-call and local paths of `_process_function_streams`, into one helper
  **`_lower_nb_stream_op(...)`** (`allo/backend/simulator.py`). Both paths now call it;
  net −267 lines. Unified on the cleaner local version (drops a redundant head-load
  the cross-call path had). Zero behavior change — validated: `test_stream_nb_simple`,
  `test_stream_nb_scalar`, `test_stream_ops_sim` all PASS; feedback/blocking paths
  unchanged. This is the single site Phase 3 will edit to add the timestamp/time-barrier
  NB semantics. Scoped to non-blocking (wire deferred). *Not committed — working diff.*
- **Phase 1 (DONE, uncommitted) — per-PE simulated-time clock.** Added to
  `allo/backend/simulator.py`: `_op_latency`/`_block_latency` (all-ones cost model),
  `_collect_blocks` (recursive, **skips `scf.while` spin loops** so waiting doesn't
  advance the clock non-deterministically), `_insert_clock_increments` (emits
  `clock += static_block_latency` at each block start — loop bodies charged
  per-iteration for free), and `_add_pe_clocks` (adds a trailing `memref<i64>` clock
  arg to each PE, alloc+init 0 at the **caller entry** so it dominates the calls even
  after omp injection, recreates each `func.call` with the clock operand, inserts the
  increments). Wired into `build_dataflow_simulator` after PE discovery, before omp
  injection. **Clock is inert (written, never read) → results unchanged** — validated:
  nb_simple/nb_scalar/stream_ops_sim PASS, feedback ring2/ring3 correct, systolic
  2×2×2 err 0.0 / 6×6×6 err 1.2e-7. Key fix found: per-call clock allocs caused a
  dominance violation once calls move into `omp.section`; solved by allocating at the
  caller entry block (`InsertionPoint.at_block_begin`).
- **Phase 2 IN PROGRESS (uncommitted).** Chunk 1: FIFO struct extended `{ring,head,
  tail}`→`{ring,head,tail,ts_ring}` (`ts_ring: memref<(depth+1)×i64>`, inert). Chunk 2
  (reorder): split Phase-1 `_add_pe_clocks` into `_add_pe_clock_args` (runs BEFORE
  `_process_function_streams`, tags PEs `sim.clock`) + `_insert_pe_clock_increments`
  (runs after) — so put/get lowering can reference the PE clock. Realization: put-stamp
  and get-max each use the PE's OWN clock (ts_ring carries the producer's timestamp
  across), so NO struct clock-pointers needed for Phase 2 (those are Phase 3's
  time-barrier). Both chunks validated behavior-preserving (nb/stream_ops/systolic/
  feedback all pass, clocks still inert).
- **Phase 2 chunk 3 (NB path) DONE (uncommitted).** Added `_clock_of` (PE's clock =
  last arg if `sim.clock`-tagged, else None), `_stamp_put_ts` (`ts_ring[slot]=clock`),
  `_advance_get_ts` (`clock=max(clock, ts_ring[slot])` via `arith.maxsi`). Threaded
  `ts_ptr` (struct_get idx 3) + `clock_arg` (`_clock_of(func)`) into `_lower_nb_stream_op`
  from both preambles; `try_put` stamps, `try_get` max-updates. Validated: nb_simple/
  nb_scalar/stream_ops_sim/systolic all pass (results unchanged — clock read/written but
  nothing branches on it yet). IR check on a try_put/try_get region: 2 `sim.clock` funcs,
  1 `arith.maxsi` (consumer) — ops really emitted, not skipped.
- **Phase 2 chunk 3b DONE (uncommitted) — Phase 2 COMPLETE.** Added the same
  `_stamp_put_ts`/`_advance_get_ts` calls at the 4 blocking put/get sites
  (cross-call + local, before each atomic index-update critical). Now EVERY put
  (blocking or NB) stamps `ts_ring[slot]=clock` and EVERY get does
  `clock=max(clock, ts_ring[slot])`. Validated: nb/stream_ops/systolic(err 0.0)/
  feedback all pass; blocking design IR-verified to emit `arith.maxsi`. Results
  still unchanged (clock written, not branched on).
- **Next — Phase 3 (the payoff): the time-barrier.** `try_get`/`empty` at consumer
  time T: (a) wait until the PRODUCER's live clock ≥ T, (b) answer `head.ts ≤ T`.
  Needs the producer's live clock reachable by the consumer → FIFO struct gains
  `prod_clock`/`cons_clock` pointers, which requires per-stream producer/consumer
  identification. Success metric: `nb_nondeterminism.py` → ONE answer / 30 runs.

## Scope / caveats
- **Blocking ops are unchanged in behavior** (already timing-immune); this only adds
  a clock so NB has meaning. Buffered feedback (round 6) still works.
- Accuracy is coarse under the default table (all-ones); ordering/determinism is
  exact. Refine latencies toward the HLS schedule for cycle-accuracy later.
- **Combinational wires (zero-latency)** are still the hard case — a 0-latency link
  makes the time barrier trivial only if acyclic; a combinational *loop* needs the
  fixed-point step (separate from this NB work).
- Overhead: one extra atomic clock per PE + one timestamp per FIFO slot + the
  time-barrier wait (replaces the current single-shot `scf.if`).

## Wire / ValidOnly / ValidReady as one handshake-over-timestamps rule
Once each PE has a simulated-time clock `T^S`, **"same cycle" = "same timestamp"**,
so the three link protocols are the SAME timestamp machinery with different
"who waits?" rules — no separate mechanism per protocol:

- **valid at cycle `T`** = producer has a token whose timestamp ≤ `T` (asserts valid).
- **ready at cycle `T`** = consumer can accept at its cycle `T`.
- A **transfer** resolves to one number: `T_transfer = max(T_producer_has_data,
  T_consumer_ready)`. Both advance past it; whichever side became ready *earlier*
  **stalls** (waits for the peer's monotonic `T^S` to reach `T_transfer`). That
  wait IS the back-pressure — deterministic, since it's a wait on the peer's clock,
  not on OS scheduling.

| link | rule | who stalls |
|---|---|---|
| **ValidReady** | transfer at `max(valid_time, ready_time)` | both — full rendezvous |
| **ValidOnly** | consumer reads whatever token is valid (ts ≤ its `T`); producer never waits | consumer only (may miss / overwrite) |
| **Wire** | read the value valid at cycle `T`; no valid, no ready | neither (correct only if externally synced) |

Key consequence: **ValidReady is just a timestamped depth-0/1 rendezvous — the same
`max(producer_time, consumer_time)` rule as the blocking `get` handoff**, plus an
explicit ready signal. So it needs no new machinery: it's the blocking-FIFO timing
rule at capacity 0. **ValidOnly** relaxes it (drop the ready-wait; producer fires
and the consumer samples whatever is valid at its cycle). **Wire** relaxes it further
(drop valid too; consumer just reads the last value at its cycle). This unifies the
non-blocking work (§ above) and all three link types under one clock + one
`max`-of-timestamps rule; the only genuinely separate case remains a **combinational
loop** (zero-latency cycle), which still needs the fixed-point step.


- 2026-07-22: Completed detailed analysis of the current simulator (below).
- 2026-07-22: **Empirically proved NB non-determinism.** A Type-C test (producer:
  200 `try_put`, consumer: 200 `try_get`, both fixed-count/no-spin, depth-8 stream)
  run 30× on the same compiled sim with identical input produced **23 distinct
  `(put_ok, got)` outcomes**, `put_ok`/`got` ranging the full **32..200**. Same
  program, same input → 23 answers, because cross-PE ordering is set by the OS
  scheduler (no clock). On hardware this is a single fixed number. Also exposes a
  correctness artifact: many runs show `got < put_ok` (items left stranded in the
  FIFO). Script: `scratchpad/nb_nondeterminism.py`. Confirms: NB ops are *lowered*
  but not *semantically faithful* — the same gap that blocks Wire/valid_ready.

---

# Analysis: Allo's Current Dataflow Simulator

_All code references are to `allo/backend/simulator.py` (1587 lines) unless noted._

## 1. What the simulator *is*

The "simulator" is **not** a discrete-event or cycle-accurate RTL simulator. It is a
**source-to-source MLIR pass + a JIT execution harness** that turns an Allo dataflow
region into a **multithreaded shared-memory C-like program** and runs it on the CPU via
LLVM's `ExecutionEngine`. Concurrency between PEs (processing elements / `@df.kernel`s)
is modeled with **OpenMP** (`omp.parallel` > `omp.sections` > `omp.section`, one section
per PE call). Streams become **shared ring-buffer FIFOs in memory**; blocking is modeled
by **busy-wait spin loops** (`scf.while` + `omp.flush` + `omp.taskyield` + `usleep(1)`).

There is **no notion of a clock**. Progress is driven purely by the OS scheduler running
the OMP threads. This is the single most important fact for the extension goals: Wires
and handshake protocols are inherently *cycle/timing* concepts, and the current model
deliberately lacks a clock (see §7).

## 2. Top-level entry & pipeline

- `dataflow.py:730` — `df.build(target="simulator")` → `LLVMOMPModule(s.module, s.top_func_name)`.
- `LLVMOMPModule.__init__` (`:1520`) is the whole pipeline:
  1. Parse module into a fresh `Context` (registers the `allo` dialect).
  2. `get_func_inputs_outputs(func)` — record top-level I/O types for the JIT calling convention.
  3. `decompose_library_function(self.module)` — expand library ops.
  4. **`build_dataflow_simulator(module, top_func_name)`** — the core transform (§3–§5).
  5. Tag top func with `llvm.emit_c_interface` + `top`.
  6. Two `PassManager` lowering pipelines (bufferize → linalg-to-affine; then
     lower-affine → scf-to-cf → memref/func/index/arith/cf/**openmp**-to-llvm → canonicalize).
  7. `allo_d.lower_composite_type` / `lower_bit_ops` — lower the `StructType` FIFO handle
     and any bit ops.
  8. **`convert_critical_write_to_atomic_write`** (`:1490`) — peephole: an `omp.critical`
     wrapping a single `llvm.store` + terminator → `omp.atomic.write` (cheaper).
  9. Build `ExecutionEngine` with `libmlir_runner_utils`, `libmlir_c_runner_utils`,
     **`libomp`** shared libs. (Requires correct `LLVM_BUILD_DIR` — see PITFALLS.)

## 3. FIFO data structure (how a Stream is represented)

For each `allo.stream_construct` (`_process_function_streams`, `:165`–`:221`), the pass
materializes a **ring buffer + head/tail**, packaged in a `StructType`:

- `fifo`  : `memref<(depth+1) x elem>` — ring buffer, **depth+1** slots (one slot is
  sacrificed to disambiguate full vs. empty, the classic ring-buffer trick).
- `head`  : `memref<i32>` (scalar) — read index, init 0.
- `tail`  : `memref<i32>` (scalar) — write index, init 0.
- The three are bundled by `allo.struct_construct` into a `StructType`, stored into a
  `memref<struct>` tagged with the stream's `name` attribute.

Element type handling: scalar (`i*`/`f*`) → `memref<depth+1 x elem>`; array element
(`memref<...>`) → `memref<(depth+1) x shape...>` (extra leading ring dimension).

**Empty ⇔ `head == tail`. Full ⇔ `(tail+1) % (depth+1) == head`.**

## 4. Stream op lowering (the heart of the simulator)

Two nearly-identical code paths handle stream ops (this duplication is a maintenance
smell and a prime refactor target before extending):

- **Cross-call path** (`:224`–`:834`): streams passed as `func.call` arguments. First it
  rewrites the callee signature + call operands so the FIFO `memref<struct>` is threaded
  through as an argument (`arg_stream_table`), then rewrites each stream op inside the callee.
- **Local path** (`:838`–`:1383`): streams constructed and used within the *same* function.
  Same lowering logic, re-implemented.

Op-by-op semantics produced:

| Allo op            | Blocking? | Lowered to |
|--------------------|-----------|------------|
| `stream_put`       | **yes**   | spin `while (head == (tail+1)%N)` { flush; taskyield; usleep(1) }, then store elem(s) at `tail`, `omp.critical`/atomic update `tail`, flush. |
| `stream_get`       | **yes**   | spin `while (head == tail)` {…}, then load elem(s) at `head`, critical update `head`. |
| `stream_try_put`   | **no**    | `scf.if (not full)` → store + update tail, yield `true`; else yield `false` (`i1`). |
| `stream_try_get`   | **no**    | `scf.if (not empty)` → load + update head, yield `(data, true)`; else yield `(dummy, false)`. |
| `stream_empty`     | probe     | flush; `head == tail` → `i1`. |
| `stream_full`      | probe     | flush; `(tail+1)%N == head` → `i1`. |

**Non-blocking support (Goal 1) already exists for Streams** — `StreamTryGetOp`,
`StreamTryPutOp`, `StreamEmptyOp`, `StreamFullOp` are all lowered here (35 stream-op
references in the file). The `try_*` ops return an `i1` success flag alongside the data
via `scf.if` results; `empty`/`full` are pure probes. So Goal 1's *simulator* piece is
largely **done**; remaining Goal-1 work (if any) is likely at the frontend/other backends
or hardening/coverage, not core sim semantics. **Verify against tests**
`test_stream_nb_simple.py`, `test_stream_nb_scalar.py`, `test_stream_ops_sim.py`.

### Memory-consistency machinery
Because PEs are concurrent OMP threads sharing the FIFO memory, correctness relies on:
- `omp.flush` before every head/tail read and after every update (visibility).
- `omp.critical` (later peepholed to `omp.atomic.write`) around the single index update.
- `usleep(1)` inside spin loops to avoid CPU starvation / livelock under oversubscription.
This is a **software shared-memory** concurrency model, not hardware handshake semantics.

## 5. PE discovery & concurrency injection

- `_process_function_streams` (`:78`) recursively walks from the top function, following
  `func.call`s (both top-level and **nested** inside `affine.for`/`scf.if`, `:136`–`:150`)
  to make sure every callee with streams gets lowered (otherwise stray `stream_*` ops
  survive to LLVM lowering and crash `convert-func-to-llvm`).
- PE calls are accumulated per-function in `all_pe_calls_by_func`.
- `_inject_omp_parallel_sections` (`:1401`) wraps each function's set of PE `func.call`s in
  `omp.parallel { omp.sections { omp.section { call } … } }` — i.e. **every PE runs as an
  independent, unordered concurrent section**. Nested parallelism is enabled via
  `OMP_MAX_ACTIVE_LEVELS=4` (`:1432`) so a PE that itself contains a sub-region doesn't deadlock.

**This is exactly the scheduler that must change for Wires** (see §7): independent
`omp.section`s have *no ordering guarantee*, which is fine for back-pressured FIFOs but
wrong for zero-latency combinational wires.

## 6. Gap analysis vs. the session goals

| Goal | Current state |
|------|---------------|
| **Non-blocking read/write** | ✅ Streams: `try_get/try_put/empty/full` fully lowered (§4). Possibly needs test coverage / other-backend parity, but sim core exists. |
| **Wire** in a region | ❌ Simulator has **0** references to `WireConstruct/Get/PutOp`. Frontend + dialect + SystemC backend support Wire (`ir/types.py:361`, `AlloOps.td:1137`), but the **simulator does not lower it** — a `@df.region` using `Wire[T]` would leave `allo.wire_*` ops that crash LLVM lowering. |
| **ValidOnly / ValidReady Channel** in a region | ❌ Same: **0** references to `ChannelConstruct/Get/Put/TryGet/TryPutOp` in the simulator. Frontend/dialect define `Channel[T, valid_only|valid_ready]` (`ir/types.py:392`, `AlloAttrs.td:84`), SystemC backend maps them (Connections::Combinational), but the **simulator does not**. |

So the concrete simulator work is: **(a)** add lowering branches for `wire_*` and
`channel_*` ops, and **(b)** — the hard part — give them *correct scheduling semantics*.

## 7. The core architectural tension (Wire/Channel scheduling)

From the existing design notes ([[simulator-combinational-wire-design]]):

- **Wire = single-slot cell, not a ring buffer.** Construct → `memref<elem>` (scalar cell,
  no head/tail/struct). `put` → plain `store` (no full-check, no critical). `get` → plain
  `load` (no empty-check). Lift the cell to a shared `memref` arg like the FIFO struct.
- **Feed-forward (acyclic) case is easy:** if producer runs fully before consumer, a wire
  is essentially just a shared `memref` — but then it must **not** be an independent
  `omp.section`; PEs must run in **topological (producer-before-consumer) order** in
  `_inject_omp_parallel_sections`, and combinational **cycles must be rejected** via
  topo-sort. In this case `omp.flush` can even be dropped (pure store→load).
- **General case (fine-grained coupling / feedback) is a big change:** a real wire needs a
  notion of a **clock cycle** that the async run-to-completion OMP model deliberately lacks.
  Correct general behavior ⇒ abandon run-to-completion and adopt a **cycle-stepped,
  settle-to-fixed-point** scheduler (RTL-sim style: evaluate the combinational cone to a
  fixed point each cycle, then latch registers). This is the central design decision the
  papers should inform.
- **Cosim reality check** (from [[wire-channel-dataflow-types]]): a bare Wire is only
  correct when the consumer's read is synchronized to the producer's write (companion
  handshake or lock-step). `ValidReady` = full handshake (safe at any timing);
  `ValidOnly` = data + valid, lock-step, no back-pressure; `Wire` = raw, externally
  synchronized. These are **three distinct hardware timing patterns** the simulator must
  eventually distinguish, not cosmetic variants.

## 8. Design questions the extension must answer

1. **Scheduling model:** keep async-OMP + topological ordering (only supports feed-forward,
   no feedback), or move to a cycle-stepped fixed-point scheduler (supports everything, but
   a rewrite of §5)? The papers (`DAM_ISCA24`, `omnisim`, `M100`, AMD-NPU) are the input here.
2. **ValidReady vs ValidOnly semantics** in a *software* sim with no clock — how to model
   back-pressure (ready) vs. drop/overwrite (valid-only) without a cycle notion.
3. **Cycle detection** for combinational loops (reject vs. support via fixed-point).
4. **Refactor first?** The two duplicated stream-lowering paths (§4) should probably be
   unified into a table-driven `_lower_link_op(kind, protocol, ...)` before adding
   wire/channel branches, to avoid a third and fourth copy.

## 9. Key file/line index (for the implementation phase)

- `allo/backend/simulator.py`
  - `_process_function_streams` `:78` — FIFO materialization + op lowering (add wire/channel here).
  - FIFO struct build `:165`; cross-call op lowering `:224`; local op lowering `:838`.
  - `_inject_omp_parallel_sections` `:1401` — PE concurrency (add topological ordering here).
  - `build_dataflow_simulator` `:1427`; `LLVMOMPModule` `:1520`.
- `allo/dataflow.py:730` — simulator target dispatch.
- `allo/ir/types.py:361` `Wire`, `:392` `Channel`, `:350` `ChannelProtocol`.
- Dialect ops: `mlir/include/allo/Dialect/AlloOps.td:1137` (Wire), Channel ops nearby;
  protocol enum `AlloAttrs.td:84`.
- Tests to keep green: `tests/dataflow/test_stream_ops_sim.py`,
  `test_stream_nb_simple.py`, `test_stream_nb_scalar.py`, `test_nested_subregion_streams.py`.

## 10. Papers to read (in `simulator_papers/`)

- `DAM_ISCA24_dataflow_abstract.pdf` — Dataflow Abstract Machine (ISCA'24); likely the
  closest match for a principled dataflow simulation/scheduling model.
- `2508.19299v1_omnisim.pdf` — OmniSim.
- `2604.17862v1_m100.pdf` — M100.
- `2504.03083v1_unlockin_the_amd_npu.pdf` — AMD NPU.

(Next step per session goals: read these for scheduling/protocol ideas, then design the
Wire/Channel simulator extension around §7–§8.)

---

# Paper Analysis — Ideas for the Extension (read 2026-07-22)

The four papers split cleanly into **two that propose a simulator architecture**
(DAM, OmniSim — the important ones) and **two that give handshake/link semantics**
(M100, AMD-NPU). Below: each paper's transferable idea, then a synthesis (§P5).

## P1. OmniSim (HLS sim; *most directly relevant — it names Allo*)
"Simulating Hardware with C Speed and RTL Accuracy for HLS Designs." Successor to
LightningSim. **It explicitly classifies Allo as supporting "Type A" only** and its
taxonomy is literally a roadmap for our exact task.

- **Technique:** don't step a clock. (a) Instrument LLVM IR to emit a **trace of
  executed basic blocks** (native-speed functional sim); (b) combine trace with the
  **static C-synthesis schedule** (loop II, latencies, pipeline stages) to build a
  **simulation graph** of *events* (FIFO accesses, calls) with RAW/WAR dependency
  edges; **total cycles = longest path**. **0.09 % mean cycle error** vs RTL co-sim,
  **up to 35.9× (geomean 30.7×)** faster than co-sim.
- **The Type A/B/C taxonomy = our roadmap:**
  - **Type A** (Allo today): blocking + acyclic → infinite-depth assumption, timing
    can't change functionality → simulate single-threaded, compute cycles after.
  - **Type B:** non-blocking / cyclic / infinite loops, but NB outcome doesn't change
    control flow → functional sim needs threads but is cycle-independent.
  - **Type C:** NB outcome **changes later behavior** (dropped elements, `if(write_nb)`)
    → functional sim itself becomes cycle-dependent, must couple to perf sim.
  Adding non-blocking + valid_only/valid_ready + wires moves Allo into **B and C.**
- **Link mapping (directly usable):** back-pressure = a dependency edge resolved by FIFO
  depth; **Wire = depth-0 / zero-latency edge (same-cycle RAW ordering)**;
  **valid_only = no back-pressure edge** (producer never stalls); **valid_ready =
  back-pressure edge resolved by depth.**
- **Runtime shape (close to Allo's):** one OS **thread per dataflow module** (≈ Allo's
  OMP sections) for functional sim; **one dedicated Perf-Sim thread** owns a **Query
  Pool** + partial sim graph + a **cycle-stamped FIFO read/write table**. A func thread
  runs natively until it hits an NB access / `empty()`/`full()` / blocking-empty, then
  **pauses and emits a query**; the perf thread resolves queries against exact HW cycles
  (Table 2: NB write with FIFO size S succeeds if w≤S else compare source cycle vs
  (w−S)-th read's cycle). **Deadlock = Task Tracker active-thread count hits 0** with
  queries pending (co-sim just hangs; OmniSim reports it).
- **Most portable single structure:** the **cycle-stamped FIFO R/W table** — replaces
  Allo's occupancy-counter ring buffer and is what makes NB `empty()/full()` correct
  despite OS-scheduling skew.
- **Cost:** requires ingesting Allo's **static II/latency schedule** to stamp cycles —
  Allo doesn't surface that to the simulator today.

## P2. DAM — Dataflow Abstract Machine (ISCA'24)
A **thread/context + local-time** model; **conservative** (CMB-style), not cycle-stepped,
not event-driven. Timing-*approximate* but functionally exact/deterministic. Up to **4
orders of magnitude** faster than cycle-by-cycle sims; calibrated to **~0.3 % (0.8 cycle)**
of RTL.

- **Core idea:** every PE ("context") owns a **monotonic local simulated time `T^S`**;
  it may jump forward freely, never backward. **No global clock, no barrier** →
  "asynchronous distributed time," contexts run thousands of cycles apart. Latency is
  injected by the model author with `time.incr(...)` calls.
- **Channels are "time-bridging":** backed by a **data channel of timestamped values**
  (sender→receiver) **+ a response channel of dequeue-timestamps** (receiver→sender);
  the reverse timestamps are how **back-pressure propagates without a clock**.
- **Non-blocking = a timestamp peek** (SVA, acquire/release atomics — zero extra x86
  instructions): is a value stamped ≤ my `T^S` present? Only **futex-park** (SVP) when
  the reader is genuinely ahead of the writer. This is far cleaner than Allo's spin loops.
- **Link mapping:** **valid_ready** = bounded channel with the reverse response-timestamp
  back-pressure; **valid_only** = unbounded channel (paper notes unbounded "cannot
  simulate backpressure" — exactly valid_only); **Wire** = capacity-0, `send-recv-delta=0`
  channel (sender/receiver share `T^S` at transfer) → needs intra-timestamp ordering.
- **Speed knobs:** blocking dequeue **jumps local time** to the front element's timestamp
  (skips idle spans, no stepping); prefer **non-boosting `SCHED_FIFO`** over CFS (**2.3×**).
- **Cost:** requires **augmenting the OMP run-to-completion model** with per-PE `T^S`,
  timestamped elements, and peer-timestamp waits. No fixed-point iteration (single-pass).

## P3. M100 — Orchestrated Dataflow Architecture (handshake semantics)
Coarse-grained, **memory-mediated** (all links are SRAM buffers + HW counters; *no*
combinational wire — don't model Wire from this paper). Gives a clean **credit/valid-ready**
model:
- **Dual Synchronization Counters** (Fig. 3): producer bumps a **Producer-Updated SC**
  = `valid`; consumer bumps a **Consumer-Updated SC** = `ready`/credit return. →
  **valid_only = producer SC only; valid_ready = both counters.**
- **Blocking read = "wait until counter ≥ expected value"** (SU monitor with a threshold);
  **non-blocking read = same counter test without parking.** Sync granularity is
  software-controlled (every element or every N elements).
- Scheduler invariant worth copying: **order preserved only within a functional unit;
  global execution out-of-order, data-driven.** Double-buffering is first-class for
  breaking loop-carried deps.

## P4. AMD NPU / AIE (IRON) — applied, thin on our topic
Real-hardware GPT-2 offload paper; **no simulator, no valid/ready, no wires** discussed.
Transferable bits only: **(a)** DMA as an **independent data mover decoupled from PE
compute** (channel ≠ PE); **(b)** lock-guarded **acquire → compute → release** over a
bounded **double buffer** = the Channel back-pressure pattern (the semaphore lock *is* the
credit); **(c)** **fixed-latency static scheduling** as the timing model (no stalls; compiler
avoids hazards). ObjectFifo is named but not specified.

## P5. Synthesis → design directions

**Two viable simulator architectures, a clear tradeoff:**

| | **OmniSim-style** (trace + sim-graph, longest-path) | **DAM-style** (per-PE local time, timestamped channels) |
|---|---|---|
| Accuracy | Cycle-accurate (0.09 %) | Cycle-approx (~0.3 %, author-tuned) |
| Needs | Allo's static II/latency schedule; a Perf-Sim orchestrator thread + query pool | Per-PE `T^S`; timestamped elements; SVA/SVP sync |
| Fits Allo's OMP model? | Yes — thread-per-PE already matches; add 1 perf thread | Partial — must add local time to each OMP section |
| Handles Type C (NB changes behavior)? | Yes (its whole point) | Yes (data-driven) |
| Effort | Higher (schedule ingestion + graph engine) | Medium (augment threads + channels) |

**Where all papers agree (build these regardless of architecture):**
1. **Handshake = counters/credits, not spin.** valid_only = one forward signal (producer
   SC / valid); valid_ready = add a reverse credit (consumer SC / response-timestamp /
   FIFO-depth back-pressure edge). Matches M100 **and** DAM **and** OmniSim.
2. **Non-blocking = one test, no park** — a timestamp/counter/table peek returning
   success. Replaces Allo's current `scf.if`-with-no-spin (which is already correct
   *functionally* but has no timing meaning). This is the bridge from our existing NB
   Stream lowering to a *timed* one.
3. **Wire is the odd one out** — only OmniSim and DAM give it a principled model
   (depth-0 zero-latency edge / capacity-0 zero-delta channel); M100 & AMD-NPU can't
   express it (everything buffered). Confirms our own note ([[simulator-combinational-wire-design]]):
   a wire needs **intra-cycle ordering** — either topological scheduling (feed-forward
   only) or a settle/fixed-point step (feedback). Neither DAM nor OmniSim needs a
   fixed-point *because both carry explicit cycle/time stamps* that order the same-cycle
   read after the write — an alternative to fixed-point iteration.

**Recommendation (to discuss):** the pragmatic path is a **DAM-lite** layer — give each
OMP PE a monotonic `T^S`, replace ring buffers with **timestamped single/multi-slot
cells**, and implement the three links as (Wire = 0-delta capacity-0 cell, valid_only =
unbounded/never-block, valid_ready = bounded + reverse credit). It reuses Allo's existing
thread-per-PE OMP structure and existing NB `scf.if` lowering, avoids needing Allo's
static schedule, and gives *timing-approximate* wire/handshake behavior. If true
cycle-accuracy is later required, adopt OmniSim's sim-graph/longest-path engine (needs the
static schedule). Open question for §8: do we need feedback/combinational-loop support on
day one? If not, topological ordering (our existing note) + timestamps covers feed-forward
wires without a fixed-point solver.
