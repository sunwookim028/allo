# Simulator Extension — High-Level Design Concept

*Companion to `claude_simulator.md` (current-simulator analysis + paper synthesis).
This doc is the **high-level concept + staged roadmap**, not implementation detail.*

Scope: extend Allo's **dataflow simulator** (`allo/backend/simulator.py`, the LLVM/OMP
CPU sim — NOT the SystemC/Catapult translation backend) to support **non-blocking
read/write** and the three link protocols **Wire / ValidOnly / ValidReady**.

---

## 1. The core obstacle (one sentence)

The simulator models PEs as unordered OpenMP threads over shared-memory ring-buffer FIFOs
with **no notion of a clock** — but Wire, ValidOnly, ValidReady, and *faithful* non-blocking
are all **timing/ordering concepts**. So the real task is not "add three op lowerings"; it is
**"introduce just enough ordering to make link timing meaningful, without paying for a full
cycle-accurate RTL sim."**

Evidence this matters: the existing NB ops (`try_get/try_put`) are *lowered* but not
*faithful* — same program + same input gave **23 distinct outcomes in 30 runs** because the
OS scheduler, not the design, decides ordering (`examples/nb_nondeterminism.py`).

---

## 2. Unifying idea: one *link* abstraction, not four op families

Today there is one link type (Stream = blocking bounded FIFO). Instead of bolting on separate
`wire_*` and `channel_*` families, treat **every PE-to-PE connection as one "link"** described
by three orthogonal properties, plus a per-operation access mode:

| Property | Wire | ValidOnly | ValidReady | Stream (today) |
|---|---|---|---|---|
| **Depth** (HW storage) | **0** (combinational) | **0** (no buffer) | **0** (no buffer) | **N** (bounded FIFO) |
| **`valid` wire** | – | ✓ | ✓ | (implicit) |
| **`ready` wire / backpressure** | – | **No** (lossy) | **Yes** (stall until ready, lossless) | Yes (full/empty) |
| **Latency / ordering** | same-cycle (combinational) | same-cycle | same-cycle | buffered (≥ next) |
| **Access mode** (per op, orthogonal) | blocking / **non-blocking** — a *test-and-proceed* flag on get/put, independent of link type |

**Key correction (`Channel` has NO depth param; docstring: "no buffering"):** Wire / ValidOnly /
ValidReady are **one combinational depth-0 family**, differing only by which handshake wires
exist (raw / +valid / +valid+ready). **`Stream` is the odd one out — the only buffered (FIFO
depth-N) link.** Matches the SystemC backend (both channels → `Connections` "combinational, no
buffer"; wire → raw `sc_signal`). ValidReady's `ready` = backpressure, but the link is still
zero-buffer — a *rendezvous/synchronous handshake*, not a FIFO.

Two payoffs:
- **Conceptual:** Wire / ValidOnly / ValidReady are three combinational points (which handshake
  wires); Stream is the buffered point. Blocking vs NB is a separate axis on the *operation*.
- **Implementation:** collapses today's **two duplicated stream-lowering paths** into one
  table-driven `_lower_link_op(kind, protocol, mode, …)` — do this refactor *first* so we
  don't spawn copy #3 and #4 (`claude_simulator.md` §8 Q4).

---

## 3. The engine decision, reframed as a spectrum (not either/or)

"How do links get correct timing?" has three known answers, increasing in cost and fidelity.
The key move: **don't pick one — stage them**, because the abstraction in §2 lets timing be
added underneath a stable op interface.

| Engine | Ordering authority | Gets us | Cost | Feedback? |
|---|---|---|---|---|
| **A. Topological** (feed-forward) | producer-before-consumer PE order | functional wire/channel, no crashes | low | no |
| **B. DAM-lite** (per-PE local time) | each PE's monotonic `T^S` + timestamped links | **deterministic** NB, approx-timed valid_ready, backpressure without a clock, feedback | medium | yes |
| **C. OmniSim** (trace + static schedule) | ingest Allo II/latency schedule, longest-path | ~cycle-accurate (0.09% err) | high (needs schedule surfaced to sim) | yes |

DAM (B) is the center of gravity: it maps directly onto Allo's *existing thread-per-PE OMP
structure* (add a local-time counter per section), gives determinism + backpressure **without
a global clock or fixed-point iteration**, and its timestamped-channel model expresses all
three protocols cleanly (ValidReady = bounded + reverse dequeue-timestamps; ValidOnly =
unbounded/never-block; Wire = capacity-0, zero-delta). OmniSim (C) is the upgrade path *iff*
we later need RTL-matching cycle counts and are willing to surface Allo's static schedule.

---

## 4. Recommended concept: **"timed links," built in stages**

A single mental model to build toward: **each PE owns a monotonic simulated time `T^S`; each
link is a typed channel carrying `(value, timestamp)`; the link's protocol decides
buffering + backpressure; a get returns the value visible at the reader's `T^S`; NB = a peek
that returns a success flag instead of advancing/parking.** Blocking = advance `T^S` to the
element's timestamp (skip idle spans — DAM's speed trick). This is Engine B, and Engines A
and C are the "before" and "after" of the same interface.

Why this and not a cycle-stepped fixed-point RTL sim: fixed-point iteration is the classic way
to settle combinational wires, but **both DAM and OmniSim avoid it** — explicit timestamps
already order a same-cycle read after its write, which is cheaper and fits our async model.

---

## 5. Staged roadmap

- **Stage 0 — Refactor (prereq).** Unify the two stream-lowering paths into one table-driven
  `_lower_link_op(kind, protocol, mode)`. No behavior change; keeps existing tests green. This
  is the seam every later stage plugs into.

- **Stage 1 — Functional links, feed-forward.** Add lowering so a region using Wire/Channel
  *runs* (no LLVM crash), untimed:
  - **Wire** → single-slot `memref` cell, plain store/load; **topological PE ordering** in
    `_inject_omp_parallel_sections`; **reject combinational cycles** with a clear error.
  - **ValidOnly** → never-blocking cell (producer overwrites, consumer reads latest + a valid bit).
  - **ValidReady** → reuse the existing bounded ring-buffer FIFO (backpressure = blocking).
  - Deliverable: designs using these types simulate functionally. Unblocks EVA-style *acyclic*
    work today. (Determinism/timing still not faithful — that's Stage 2.)

- **Stage 2 — DAM-lite timing.** Give each OMP PE a monotonic `T^S`; make links carry
  timestamps; NB becomes a deterministic timestamp peek; ValidReady backpressure via reverse
  dequeue-timestamps. **Turns `nb_nondeterminism.py` from 23 outcomes → 1.** This is the
  "faithful" milestone for Goal 1 + ValidReady.

- **Stage 3 — Feedback / combinational loops.** Allow cyclic links (the EVA bidirectional
  mesh) via timestamp-ordered same-cycle transfer (no global fixed-point). Wire in a loop
  needs intra-timestamp ordering rules — design here draws on DAM §V and OmniSim's depth-0 edge.

- **Stage 4 (optional) — Cycle-accuracy.** If RTL-matching numbers are required, add
  OmniSim-style longest-path over a sim-graph built from Allo's static II/latency schedule.
  Bigger lift; only if Stage-2 approximate timing proves insufficient.

Value lands early (Stage 1 unblocks the types), and the hard architectural change (local time)
is isolated to Stage 2 behind the Stage-0 seam — so we are not forced to commit to the whole
timed engine before seeing the functional version work.

---

## 6. Cross-cutting: protocol → simulator mapping (target semantics)

| Construct | HW depth | Sim device (Stage 1) | Producer when downstream not ready | Consumer when no data | NB variant |
|---|---|---|---|---|---|
| **Wire** | 0 | single-slot cell | — (drives value) | reads current driven value | NB read = "is a value driven this Tˢ?" |
| **ValidOnly** | 0 | 1-slot cell + valid | overwrite (never stall, lossy) | reads latest; valid=0 if none | NB read returns valid flag |
| **ValidReady** | 0 (rendezvous) | **depth-1 blocking** (functional stand-in) | stall until ready (blocking) / fail (NB) | stall / fail (NB) | test-and-proceed + success flag |
| **Stream** (exists) | N | ring buffer (exists) | stall on full | stall on empty | `try_*` already lowered |

*ValidReady is zero-buffer in HW (a rendezvous); the depth-1 blocking device is only a Stage-1
functional stand-in that yields identical data — true same-cycle timing is Stage 2.*

---

## 7. Decisions (resolved 2026-07-22)

1. **Accuracy bar → DAM-lite *approximate* is enough for now.** OmniSim cycle-accuracy
   (Stage 4) deferred until/unless approximate timing proves insufficient.
2. **Feedback → feed-forward first is enough.** Stage 1/2 acyclic only; combinational cycles
   rejected with a clear error. Feedback (Stage 3) is a later increment.
3. **`T^S` advance source → lightweight op/iteration-count latency model inside the lowering**
   (NOT Allo's static schedule). Rationale: the Stage-2 **determinism** win depends only on
   latencies being a *deterministic function of the program*, not on their *accuracy* — any
   deterministic per-PE increment collapses the 23-outcome nondeterminism to 1. Deriving real
   latencies from Allo's II/latency metadata is the accuracy upgrade (bridge to Stage 4),
   deferred so Stage 2 stays self-contained (no schedule surfacing needed).
4. **ValidOnly → latest-wins single-slot cell + valid bit, producer never stalls** (NOT a
   lossy FIFO). Grounded in the SystemC backend: Channel maps to `Connections::In/Out<T>`
   *"combinational, no buffer"* — a FIFO would contradict RTL. **Corollary (locked):** the
   simulator's Channel NB ops are **`try_get`/`try_put` only, never `empty()`/`full()`** —
   the backend errors channel `empty/full` as having "no synthesizable Connections
   equivalent," so the sim must not express what RTL rejects. (Open detail for Stage-1 impl:
   does a read *clear* valid (one-shot delivery) or leave it *held* — pin against RTL then.)

---

## 8. Validation strategy

- **Ground truth = the SystemC/Catapult RTL cosim path** (already emits wire/channel). Each
  sim stage validated against RTL cosim numbers on the same design.
- **Determinism regression:** `examples/nb_nondeterminism.py` must collapse 23→1 outcome at
  Stage 2 — promote it to a CI test.
- **Keep green:** `tests/dataflow/test_stream_ops_sim.py`, `test_stream_nb_simple.py`,
  `test_stream_nb_scalar.py`, `test_nested_subregion_streams.py`, plus the golden
  `test_df_unit.py` / `test_region_stateful.py`.
- **New per-protocol unit tests:** minimal 2-PE producer/consumer for Wire, ValidOnly,
  ValidReady (feed-forward at Stage 1; add a feedback case at Stage 3).

---

## 9. What each source paper contributes (pointer, detail in `claude_simulator.md`)

- **DAM (ISCA'24)** — the engine model for Stage 2/3: per-PE local time, timestamped
  channels, backpressure via reverse timestamps, NB = peek, no fixed-point.
- **OmniSim** — the Type-A/B/C taxonomy (our roadmap: this work moves Allo A→B→C) and the
  Stage-4 cycle-accurate longest-path engine; also the cleanest link→edge mapping
  (Wire = depth-0 edge, valid_only = no-backpressure edge, valid_ready = depth-bounded edge).
- **M100** — handshake as counters/credits (ValidOnly = producer SC; ValidReady = + consumer SC).
- **AMD NPU/IRON** — applied confirmation only (decoupled data-mover = channel; lock/credit
  double-buffer = backpressure); no wire/valid-ready model to borrow.
