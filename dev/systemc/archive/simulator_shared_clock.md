# Shared Simulation Clock — Design

*Supersedes the per-PE local-time engine for synchronous designs. Companion to
`simulator_concept.md` (which chose DAM-lite per-PE time) — this document records why that
choice breaks, and what replaces it.*

---

## 1. The failure it fixes

The DAM-lite engine gives every PE its own monotonic clock `T^S`, charged **per scalar op**
(`_op_latency` / `_block_latency`, applied at the start of *every* block by
`_insert_clock_increments`, `simulator.py:280`). An element is stamped with the producer's
clock; `try_get` only sees it once the consumer's clock reaches that stamp
(`_emit_read_barrier`, `:561`, condition `ts[head] <= T`).

**Consequence: a compute-heavy PE's output is permanently invisible to lighter peers.**
Measured on a 1x1 NoC — identical structure, only the router body differs:

| router body | per-PE clocks | delivered |
|---|---|---|
| bare forward (1 `try_get` + 1 `try_put`) | router 49, drv 39, col_k 50 | **yes** |
| full (decode + 5x5 arbitration + crossbar) | router **6442-6730**, drv 39, col_k 50 | **no** |

A router pass runs ~50+ loop iterations of scalar work — a 5-entry clear, a 5-entry decode, a
5-entry route, a **5x5 arbitration double loop**, a 5-entry crossbar — ≈270 charged cycles.
A collector pass charges ~2. The collector would need ~3000 iterations to reach the router's
stamps, so the packet sits in the FIFO, correctly written, forever unread. No deadlock, no
error, zero delivered.

Ruled out as causes (each tested): failed non-blocking polls (removing 4 dead perimeter polls
moved the clock only 6730 → 6442); mapped kernels / indexed stream arrays (a mapped forwarder
passes); the separate `try_put`-unused-`ok` DCE bug (real, independent, fixed by consuming the
flag).

## 2. Why per-op charging is the wrong model here

Two independent reasons:

1. **It contradicts the hardware.** A router's decode + arbitration + crossbar is **one
   pipelined cycle** (II=1), not 270. Allo schedules that loop body at II=1; the simulator
   charges it per scalar op. The error is not a constant factor — it scales with how much
   combinational logic a PE contains, so it grows precisely for the PEs we care about.
2. **A mesh NoC is synchronous.** Every router ticks the same edge. Per-PE *local* time models
   asynchronous dataflow with decoupled rates (DAM's domain); imposing it on a single clock
   domain is a category error. The divergence above is that mismatch surfacing.

`simulator_concept.md` §7 decision 3 explicitly deferred deriving latencies from Allo's
schedule, on the grounds that determinism only needs latencies to be a *deterministic function
of the program*, not an accurate one. That reasoning is sound for determinism and wrong for
**visibility**: the read barrier compares clocks *across* PEs, so inaccuracy that is unequal
between PEs breaks data transfer outright.

## 3. The model: one loop iteration = one cycle, globally barriered

**Invariant.** All PEs share one simulated cycle counter. A PE performs the work of cycle *k*,
then waits at a barrier until every live PE has finished cycle *k*; then all advance to *k+1*.

This matches the always-fire dataflow style these designs are written in — the outer
`for t in range(NUM_IT)` loop *is* the cycle loop — and makes timestamps trivially comparable,
because there is only one clock.

Two changes to `simulator.py`:

- **(A) Cost model → per-iteration.** Charge **+1 at the PE's top-level loop body**, instead of
  `_block_latency` at every block. A PE's clock then counts *cycles*, not ops, and every PE
  advances at the same rate. This alone should restore visibility (§5, Stage 1).
- **(B) Global cycle barrier.** At the end of each top-level iteration, every PE waits for all
  others. This is what makes it a true cycle-step rather than merely comparable rates.

### Why a plain `omp.barrier` will not do

PEs have **different trip counts** (a router runs `NUM_IT`; a one-shot loader runs once), and
an OMP barrier requires every thread in the team to arrive. Mismatched arrivals are UB.
The barrier must therefore be a **software barrier with a dynamic participant count**:

- a global `live_count`, decremented when a PE retires (hooks onto the existing
  `_insert_clock_termination`, `:378`, which already marks a PE done);
- a global `arrived` counter and a sense flag; the last arriver of cycle *k* flips the sense
  and resets `arrived`;
- a retiring PE must decrement `live_count` **and** flip the sense if it was the last one
  outstanding, or the remaining PEs hang on a participant that will never arrive.

That retirement race is the main correctness risk in this design and is where prototype effort
should concentrate.

## 4. What this simplifies

With one clock, most of the timestamp machinery becomes redundant:

- `_emit_read_barrier` / `_emit_write_barrier` (`:561`, `:595`) — spin loops comparing local `T`
  against a peer's clock cell. Under a global barrier, everything written in cycle *k* is
  visible in *k+1* by construction. These can degrade to a plain occupancy check.
- `ts_ring` / `free_ts_ring` / `_advance_get_ts` / `_mark_occupied` — per-element timestamps
  exist to order events across incomparable clocks. One clock makes them unnecessary.
- `_tick_clock` on failed polls (`:492`) — the hack that stopped spin-waits freezing a clock.
  A cycle-step advances time regardless of poll outcome.
- `_CLOCK_DONE_BIT` (`:370`) stays: retirement still has to release waiters.

Deleting these is *not* part of the prototype — they are inert once the barrier is in, and
removing them is a separate cleanup once the model is validated.

## 5. Staged plan

- **Stage 1 — cost model only** (`ALLO_SIM_CLOCK=cycle`). Charge +1 per top-level iteration;
  leave barriers and timestamps untouched. Predicts: router/drv/collector clocks converge to
  ~NUM_IT, and the 1x1 and 2x2 NoC tests deliver. Cheap, reversible, and isolates whether
  *comparable rates* alone are sufficient.
- **Stage 2 — global cycle barrier.** Add the dynamic-count software barrier. Gives true
  same-cycle semantics and deterministic ordering independent of PE weight.
- **Stage 3 — validate.** Golden `test_df_unit.py` / `test_region_stateful.py`, the stream
  tests, `nb_nondeterminism.py` (must stay at 1 outcome), then the 4x4 NoC.
- **Stage 4 — cleanup.** Retire the now-inert timestamp rings and spin barriers.

**Gate between stages:** Stage 1 must make the 1x1 and 2x2 tests deliver. If it does not,
comparable clock rates are not sufficient and the diagnosis in §1 is incomplete — stop and
re-measure rather than proceeding to the barrier.

## 6. Known costs and open questions

- **Barrier cost.** A global sync per simulated cycle across N PEs (80+ for a 4x4 mesh with
  edges). This is the cost `simulator_concept.md` §4 avoided. Mitigating consideration: the
  current spin-and-flush barriers already cost ~280 cycles/pass and made the 4x4 time out at
  every budget, so a clean cycle-step may well be *cheaper* than what it replaces. To be
  measured, not assumed.
- **What is "one cycle" for a non-loop PE?** A one-shot loader with a `meta_for` burst has no
  cycle loop. Options: treat the whole body as one cycle, or leave such PEs unbarriered.
  Undecided — the prototype should not need it, but the 4x4 loader will.
- **Nested loops.** Charging only the top-level loop makes an inner `for` free. Correct for a
  pipelined II=1 body; wrong for a genuinely sequential inner loop. A later refinement is to
  use Allo's II metadata (the Stage-4 bridge `simulator_concept.md` deferred).
- **Does this regress async designs?** Producer/consumer pairs with genuinely decoupled rates
  are modelled *less* accurately by a shared clock. If that matters, the model should be
  selectable per region rather than global.
