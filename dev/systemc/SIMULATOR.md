# Allo dataflow simulator — architecture, timing model, and cycle accuracy

Merged 2026-08-12 from five files: `claude_simulator.md`, `simulator_concept.md`,
`simulator_shared_clock.md`, `simulator_cycle_model.md`, `simulator_lightningsim_plan.md`.
All originals are in `archive/` — `claude_simulator.md` in particular keeps the dated
session log and the full paper analysis, which are not reproduced here.

Scope: the **JIT dataflow simulator** (`df.build(target="simulator")`,
`allo/backend/simulator.py`) — *not* the SystemC/Catapult translation backend, which is
`BACKEND.md`.

---

## 0. Status

**Done.** The DAM-lite timing layer (per-PE local clocks, timestamped FIFOs) and
deterministic non-blocking: `nb_nondeterminism.py` went from **23 distinct outcomes in 30
runs to 1**. Commits `ca437b0` (spin-advance), `340363e` (write-side barrier + occupancy),
`198b415` (clock-threading crash fix). Deadlock detection also landed — peer-done proof
plus a circular-wait watchdog, replacing what used to be a silent infinite hang.

**The blocking problem.** The per-PE clock model **cannot run a mesh**. See §3.2 — a
compute-heavy PE's output becomes permanently invisible to lighter peers. The shared-clock
design in §3.3 is the proposed replacement and is **not implemented**. In the meantime,
mesh work uses the `simulator_nb_nondet.py` fallback.

**Open.** Livelock is still undetected. Cycle accuracy is uncalibrated (§4), though
ranking already correlates with Vitis at τ = +1.00, which may make calibration low
priority — see §4.1.

### Validate

```bash
source /home/zsm9/miniconda3/etc/profile.d/conda.sh && conda activate allo   # interactive!
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build-rhel8      # build-rhel8 ONLY
export PYTHONPATH=/home/zsm9/allo_sup && export OMP_NUM_THREADS=8
python tests/dataflow/test_df_unit.py            # golden
python tests/dataflow/test_region_stateful.py
# the profiling harness now lives OUTSIDE the repo, at /home/zsm9/simulator_profiling
python /home/zsm9/simulator_profiling/nb_nondeterminism.py  # expect a SINGLE outcome
python /home/zsm9/simulator_profiling/read_barrier_test.py  # expect [8]
```

Plain `conda run` fails with "Unknown function top"; overriding `LLVM_BUILD_DIR` to
`build/` aborts on GLIBC. `PYTHONPATH` must be forced or the import silently grabs the
installed `/home/zsm9/allo` instead of this checkout.

---

## 1. What the simulator is

Not a discrete-event or cycle-accurate RTL simulator. It is a **source-to-source MLIR pass
plus a JIT harness** that turns a dataflow region into a multithreaded shared-memory
program and runs it through LLVM's `ExecutionEngine`.

- **PEs** → OpenMP `omp.parallel` > `omp.sections` > `omp.section`, one section per PE
  call (`_inject_omp_parallel_sections`, `:1401`). Sections are **unordered**.
- **Streams** → shared ring-buffer FIFOs: `fifo: memref<(depth+1) x elem>` plus scalar
  `head`/`tail`, bundled into a `StructType`. The extra slot disambiguates full from
  empty. Empty ⇔ `head == tail`; full ⇔ `(tail+1) % (depth+1) == head`.
- **Blocking** → busy-wait spin loops (`scf.while` + `omp.flush` + `omp.taskyield` +
  `usleep(1)`).
- **Consistency** → `omp.flush` around every head/tail access, `omp.critical` on the index
  update (peepholed to `omp.atomic.write`).

**There is no clock.** Progress is driven by the OS scheduler. This is the single most
important fact about the design, because Wire and the handshake protocols are inherently
timing concepts.

Op lowering exists in **two nearly identical paths** — cross-call (`:224`–`:834`, streams
passed as `func.call` arguments, threaded through via `arg_stream_table`) and local
(`:838`–`:1383`, constructed and used in one function). Unifying them into one
table-driven `_lower_link_op(kind, protocol, mode)` is the prerequisite refactor for
anything below; otherwise every new link type spawns copies #3 and #4.

| Allo op | Blocking | Lowered to |
|---|---|---|
| `stream_put` | yes | spin while full, store at `tail`, atomic update, flush |
| `stream_get` | yes | spin while empty, load at `head`, atomic update |
| `stream_try_put` | no | `scf.if (not full)` → store + yield `true`, else `false` |
| `stream_try_get` | no | `scf.if (not empty)` → load + yield `(data, true)`, else `(dummy, false)` |
| `stream_empty` / `stream_full` | probe | flush, compare indices |

Key file/line index: `_process_function_streams` `:78` (FIFO materialization; where wire /
channel lowering would go), FIFO struct build `:165`, `build_dataflow_simulator` `:1427`,
`LLVMOMPModule` `:1520`. Frontend types: `allo/ir/types.py:361` `Wire`, `:392` `Channel`,
`:350` `ChannelProtocol`. Dialect: `AlloOps.td:1137`, protocol enum `AlloAttrs.td:84`.

---

## 2. The link abstraction

Rather than bolting on separate `wire_*` and `channel_*` op families, treat every PE-to-PE
connection as **one link** described by orthogonal properties, with blocking-vs-nonblocking
as a per-operation flag rather than a link property.

| | Wire | ValidOnly | ValidReady | Stream |
|---|---|---|---|---|
| **Depth** | 0 | 0 | 0 | **N** |
| **`valid`** | – | ✓ | ✓ | implicit |
| **`ready` / backpressure** | – | no (lossy) | yes (lossless) | yes |
| **Latency** | same-cycle | same-cycle | same-cycle | buffered |

The correction that makes this clean: **`Channel` has no depth parameter** — its docstring
says "no buffering". So Wire / ValidOnly / ValidReady are **one combinational depth-0
family** differing only in which handshake wires exist, and **`Stream` is the odd one
out**, the only buffered link. This matches the SystemC backend, where both channel
protocols map to Connections ("combinational, no buffer") and Wire maps to a raw
`sc_signal`. ValidReady's `ready` is backpressure, but the link is still zero-buffer — a
rendezvous, not a FIFO.

**Simulator status: neither is lowered.** There are **zero** references to
`WireConstruct/Get/Put` or `ChannelConstruct/Get/Put/TryGet/TryPut` in `simulator.py`. A
region using either leaves `allo.wire_*` ops that crash LLVM lowering. Frontend, dialect
and SystemC backend all support them; only the simulator does not.

### Target semantics

| Construct | Sim device | Producer, downstream not ready | Consumer, no data | NB variant |
|---|---|---|---|---|
| **Wire** | single-slot cell | drives value | reads current value | "is a value driven this cycle?" |
| **ValidOnly** | 1-slot cell + valid bit | overwrite, never stall (lossy) | reads latest, valid=0 if none | returns valid flag |
| **ValidReady** | depth-1 blocking (stand-in) | stall / fail | stall / fail | test-and-proceed + flag |
| **Stream** | ring buffer (exists) | stall on full | stall on empty | already lowered |

Locked decisions: **ValidOnly is a latest-wins single-slot cell with a valid bit, never a
lossy FIFO** — a FIFO would contradict the RTL. And the simulator's Channel NB ops are
**`try_get`/`try_put` only, never `empty()`/`full()`**, because the backend rejects channel
empty/full as having no synthesizable Connections equivalent; the sim must not express what
RTL refuses. `try_*` works on Stream and Channel but **never on Wire** — a wire has no
"empty", only its current value.

**Hardware caveat, confirmed in cosim:** a bare Wire gives zero storage *and* zero
alignment, so it is only correct when the consumer's read is synchronised to the producer's
write. Pure-wire-connected kernels should fuse into one (latency 0), and every feedback
cycle needs a registered element or it is a combinational loop.

---

## 3. Timing

### 3.1 Why ordering is the real task

The existing NB ops are *lowered* but not *faithful*: unordered OMP sections mean the OS
scheduler, not the design, decides who wins. Hence 23 outcomes from one program. The task
was never "add three op lowerings" — it is **introduce just enough ordering to make link
timing meaningful without paying for a full cycle-accurate RTL sim.**

Three engines, increasing cost: **topological** (producer-before-consumer order; functional
but untimed, no feedback), **DAM-lite** (per-PE monotonic time + timestamped links;
deterministic NB and backpressure with no global clock), **OmniSim** (trace + static
schedule, longest path; ~cycle-accurate). Notably both DAM and OmniSim **avoid
fixed-point iteration** — explicit timestamps already order a same-cycle read after its
write.

### 3.2 DAM-lite, and why it breaks on a mesh

Implemented model: every PE owns a monotonic clock `T^S`, charged **per scalar op**
(`_op_latency`/`_block_latency`, applied at every block by `_insert_clock_increments`,
`:280`). Elements carry the producer's stamp; `try_get` only sees one once the consumer's
clock reaches it (`_emit_read_barrier`, `:561`, `ts[head] <= T`).

**Consequence: a compute-heavy PE's output is permanently invisible to lighter peers.**
Measured on a 1×1 NoC where only the router body differs:

| router body | per-PE clocks | delivered |
|---|---|---|
| bare forward (1 `try_get` + 1 `try_put`) | router 49, drv 39, col 50 | **yes** |
| full (decode + 5×5 arbitration + crossbar) | router **6442–6730**, drv 39, col 50 | **no** |

A router pass charges ≈270 cycles of scalar work; a collector pass charges ≈2. The
collector would need ~3000 iterations to reach the router's stamps, so the packet sits in
the FIFO — correctly written, forever unread. No deadlock, no error, zero delivered.

Ruled out by test: failed NB polls (removing 4 dead perimeter polls moved the clock only
6730 → 6442), mapped kernels, indexed stream arrays, and the separate `try_put`-unused-`ok`
DCE bug (real, independent, fixed by consuming the flag).

Two independent reasons per-op charging is wrong here:

1. **It contradicts the hardware.** A router's decode + arbitration + crossbar is **one
   pipelined cycle at II=1**, not 270. The error is not a constant factor — it scales with
   how much combinational logic a PE contains, i.e. it grows precisely for the PEs we care
   about.
2. **A mesh is synchronous.** Per-PE *local* time models asynchronous dataflow with
   decoupled rates. Imposing it on a single clock domain is a category error.

The earlier decision to defer real latencies — on the grounds that determinism needs
latencies to be a *deterministic* function of the program, not an accurate one — is sound
for **determinism** and wrong for **visibility**: the read barrier compares clocks *across*
PEs, so inaccuracy that is unequal between PEs breaks data transfer outright.

### 3.3 The replacement: one shared clock (designed, NOT implemented)

**Invariant.** All PEs share one simulated cycle counter. A PE performs the work of cycle
*k*, waits at a barrier until every live PE has finished cycle *k*, then all advance. This
matches the always-fire style these designs are written in — the outer `for t in
range(NUM_IT)` **is** the cycle loop — and makes timestamps trivially comparable.

Two changes: **(A)** charge **+1 at the PE's top-level loop body** instead of
`_block_latency` at every block, so a clock counts cycles rather than ops; **(B)** a global
cycle barrier at the end of each top-level iteration.

**A plain `omp.barrier` will not work** — PEs have different trip counts (a router runs
`NUM_IT`, a one-shot loader runs once) and an OMP barrier requires every thread to arrive.
It must be a **software barrier with a dynamic participant count**: a global `live_count`
decremented on retirement (hooking `_insert_clock_termination`, `:378`), an `arrived`
counter and a sense flag, with the last arriver flipping the sense. A retiring PE must
decrement `live_count` **and** flip the sense if it was the last one outstanding, or the
others hang forever on a participant that will never arrive. **That retirement race is the
main correctness risk** and is where prototype effort should go.

Once the barrier is in, most timestamp machinery becomes inert — `_emit_read_barrier` /
`_emit_write_barrier` (`:561`, `:595`) degrade to occupancy checks, the `ts_ring` family
becomes unnecessary, and `_tick_clock` on failed polls (`:492`) is redundant since a
cycle-step advances regardless. `_CLOCK_DONE_BIT` (`:370`) stays. Deleting them is cleanup
*after* validation, not part of the prototype.

**Staging, with a hard gate.** Stage 1 = cost model only (`ALLO_SIM_CLOCK=cycle`), no
barrier — cheap and reversible, and isolates whether comparable *rates* alone suffice.
**Stage 1 must make the 1×1 and 2×2 NoC tests deliver; if it does not, the §3.2 diagnosis
is incomplete — stop and re-measure rather than building the barrier.** Then Stage 2
barrier, Stage 3 validation (golden tests, stream tests, `nb_nondeterminism` must stay at
1 outcome, then 4×4), Stage 4 cleanup.

Open questions: barrier cost at 80+ PEs (though the current spin-and-flush barriers already
cost ~280 cycles/pass and made 4×4 time out at every budget, so a clean cycle-step may be
*cheaper* — measure, don't assume); what "one cycle" means for a non-loop PE such as a
one-shot loader; nested loops (charging only the top level makes an inner `for` free —
correct for a pipelined II=1 body, wrong for a genuinely sequential one); and whether this
regresses genuinely asynchronous producer/consumer pairs, which a shared clock models
*less* accurately. If so the model should be selectable per region.

---

## 4. Cycle accuracy

### 4.1 What "accurate" has to mean

| consumer | needs | a constant offset is |
|---|---|---|
| **DSE ranking** | correct *ordering* | harmless |
| **Reporting** | correct *magnitude* | fatal |

Today's model is calibrated at **one point** (7 modelled vs 6 reported) on two trivial
integer PEs. Float, BRAM and pipelined-loop latencies are wholly uncalibrated, and makespan
is 0.4× because the `load_buf`/`store_res` wrappers are not clocked at all. One data point
cannot distinguish a good model from a lucky one.

But note: **ranking already matches Vitis at τ = +1.00**, so if DSE only needs ordering,
much of the calibration work may not be worth doing. Resolve which bar applies before
investing.

### 4.2 The correction that reopened schedule ingestion

An earlier finding — recorded in several places — said non-blocking designs give **no**
static schedule, every loop reporting `lat=- II=- depth=-`. **That was too strong.** What
is `undef` is the **aggregate** latency and trip counts, necessarily, because retry loops
are unbounded. **Per-iteration latency and II are reported**, including for the NB retry
loops themselves.

At operation granularity it is even clearer: `…/solution1/.autopilot/db/*.verbose.sched.rpt`
schedules **every operation**, names the hardware core, and gives latency and II — and
these reports exist and are complete for the NB project. The NB retry loop is a single
state, II=1, so **a failed `try_put` attempt costs exactly 1 cycle**.

This is exactly the split trace-based simulation is built on: **the static schedule supplies
per-iteration timing; the dynamic trace supplies the trip counts the tool cannot know.**

Measured cores (`xcu280`, 3.33 ns) against our table: `RAM` latency **1** vs our
`ARRAY_LOAD_LATENCY = 2` (**too high**); register scalars and `Adder`/`Cmp` free
(**confirms** `_is_register_memref` and `_FREE_OPS`); NB retry 1 cycle (**correct**).

> **Chaining caveat.** `FIFO_SRL` reports `Latency = 0`, yet the write still occupies its
> own state because its 1.21 ns delay will not chain under a 3.33 ns clock. Ingest the
> **state distance (`ST_n`)**, not the core's structural latency — charging streams zero
> would be wrong.

### 4.3 The plan

**Delete the hand-written per-op latency table and charge the PE clock with per-loop timing
ingested from csynth instead.** The architecture is not the problem; the guessed numbers
are.

Our simulator **already is the trace** — that is the leverage. LightningSim and OmniSim
need two passes (instrumented run → trace, then join to schedule and take a longest path).
We execute the real program per PE with a local clock already, so measured latencies inject
directly into the running simulation: OmniSim's *accuracy source* on DAM's *runtime*.

| supplies | provided by |
|---|---|
| compute latency (how long a loop body takes) | csynth static schedule |
| trip counts (how often it runs) | our execution, free |
| stall time (full/empty FIFOs, contention) | our simulation, dynamically |

Mechanism: csynth each `@df.kernel` body **once**; join report → MLIR by loop name (already
true by construction, `EmitVivadoHLS.cpp:972`, with a source-line fallback for NB retry
loops that come back as `VITIS_LOOP_24_1`); charge the ingested per-iteration latency in
`_insert_clock_increments` (the `(n−1)·II + body` formula already exists — only the numbers
change); **cache per kernel body**, since DSE varies mapping/tiling/FIFO depth rather than
the PE body, so one ~35 s csynth amortises across a sweep; ingest the wrappers too
(`store_res0_1` reports latency 12 / II 4 and is currently clocked at **zero** — that is
the 0.4× makespan gap); keep today's table as a fallback so the simulator runs without
Vitis.

**Measure before ingesting.** With one calibration point you cannot tell whether ingestion
helped. Run a ground-truth sweep first — 10–20 small designs spanning int/float,
blocking/NB, varying FIFO depth and 2–8 PEs — then report the **error distribution before
and after**, not a single ratio.

**The load-bearing assumption: that a kernel's schedule is stable across the configurations
DSE varies.** If Vitis re-schedules per mapping, per-kernel caching breaks and synthesis
cost lands back in the inner loop. Untested; the sweep should answer it first.

**Unresolved risk:** we emit SystemC → Catapult → Xcelium, and *that* path is proven
bit-exact. Vitis is a different scheduler with different cores and clock assumptions, so
calibrating to Vitis while shipping through Catapult targets the wrong tool. **Resolve
before building the ingestion.**

Cheaper ideas worth doing first: **sensitivity analysis** (results are deterministic, so
perturb one latency constant at a time — probably 3 of ~30 matter); **validate rankings,
not magnitudes** (Kendall-τ over design pairs, far cheaper); **emit an interval, not a
point** (an interval that brackets the truth beats a point confidently 20% off); and
**critical-path attribution** (we already have per-PE clocks, so we can say which PE or
stream dominates — more actionable for a generator than a scalar, and needs no accuracy
improvement at all).

---

## 5. LightningSim

Installed as conda env `lightningsim`, package `lightningsim 0.2.6`. Scratch area
`/home/zsm9/lightningsim/`. It is a **whole-tool integration, not a library call**:
`lightningsim [--cli|--gui] <vitis_hls_solution_dir>` consumes a Vitis solution, links an
instrumented build of the csim testbench, runs it to produce a trace, joins the trace to
the static schedule, and reports cycles. So the pipeline is Allo → HLS C++ → `vitis_hls
csynth` → solution dir → LightningSim, sitting *beside* our simulator. **Vitis-only** — the
same wrong-scheduler risk as §4.3.

**Stage 0 — PASS.** LightningSim `top` = 21 cycles = csynth `top` = 21, exact. Details in
`/home/zsm9/simulator_profiling/lightningsim_stage0/`. It also **measured** the 0.4× makespan gap:
ours says 7, both oracles say 21, and `store_res0.1` spans cycles 6–20 — the unclocked
wrappers, exactly as hypothesised.

Three Allo-side blockers, all fixable in the emitter: `host.cpp` is an **OpenCL** host, not
a csim testbench (needs a native `tb.cpp` calling `top()` directly); generated `kernel.h`
uses `int32_t` **without including `<cstdint>`** and only survives because other headers
precede it; and `#pragma HLS pipeline II=1 rewind` link-errors on `_ssdm_op_Return`
(dropping `rewind` fixes it but perturbs the design, 19–20 → 21 — a proper fix stubs the
intrinsic).

**Stage 1 — PASS.** `node_0_0` = 1014 cycles = csynth 1014, exact, on a PE body cut out of
the **cyclic** EVA mesh. The whole mesh segfaults; a PE extracted from it traces exactly —
so the **per-kernel oracle idea is validated**, and it produces precisely the per-kernel
numbers §4.3 wants to ingest.

> **The finding worth the whole exercise:** that PE's main loop runs at **II ≈ 3** (1012
> cycles / 336 iterations) despite `#pragma HLS pipeline II=1`. Our cost model assumes
> `DEFAULT_II = 1`, measured on a trivial loop. A 3× miss on the dominant loop of a real PE.

Harness requirements: no `hls::stream` in the top signature (top-level stream ports make
Vitis emit `streamcpy_hls` glue LightningSim does not provide — wrap with `pe_feed`/
`pe_drain`); drain counts must match production **exactly** or `builder.finish()` raises
`incomplete edges remain`; and 0.2.6 needs a one-line patch at `trace_file.py:~418`.

**Path C — WORKS, and it handles cycles.** Import `lightningsim._core` (the compiled Rust
engine) and build the simulation graph ourselves — no Vitis bitcode, no instrumentation, no
testbench. The solver accepts hand-built graphs, models backpressure (depth 1 → 37 cycles,
depth 8 → 25), does native FIFO-depth DSE returning latency *and* BRAM, and **resolves a
primed cycle while correctly flagging an unprimed one as deadlock.**

**That changes the picture:** LightningSim cannot *run* a cyclic design, but the **engine**
has no such limit — the Type A restriction lives in the front end, not the solver. Path C
reaches designs that invoking the tool never can. Caveats: `_core.pyi` is a private,
unstable API on a compiled binary, and these were synthetic graphs of a few nodes, so
nothing yet proves it scales to a real mesh.

**Why the mesh segfaults** (worth not re-diagnosing): the error `unknown trace entry type
'fifo_wr'` is misleading — the trace is 4866 good lines ending in a truncated entry, and
the testbench returncode is `-11`, i.e. SIGSEGV. The cause is **not** non-blocking; that
design has **zero** `read_nb`/`write_nb` and 200 blocking accesses. It is a 2×2 **mesh**,
i.e. cyclic, and LightningSim's Type A is blocking **and acyclic**. This is precisely why
OmniSim was written.

---

## 6. What the papers contribute

- **DAM (ISCA'24)** — the per-PE local-time engine: timestamped channels, backpressure via
  reverse dequeue timestamps, NB as a peek, no fixed-point. Reports ~0.3% (0.8 cycle) vs
  RTL, with *author-tuned* latencies — the ceiling is the table, not the architecture.
- **OmniSim** — the Type A/B/C taxonomy (this work moves Allo A→B→C), the cleanest
  link→edge mapping (Wire = depth-0 edge, valid_only = no-backpressure, valid_ready =
  depth-bounded), and the cycle-accurate longest-path engine. 0.09% mean error.
- **LightningSim (FPGA'23)** — OmniSim's predecessor and the cleaner statement of the
  method. 99.9% accurate, up to 95× faster than RTL cosim.
- **M100** — handshake as counters/credits (ValidOnly = producer credit; ValidReady = plus
  consumer credit).
- **AMD NPU / IRON** — applied confirmation only; no wire/valid-ready model to borrow.

Worth reading next, by what they would contribute: **latency-insensitivity testing for
dataflow HLS designs** (FPGA'25, Edinburgh) — OmniSim's Type A/B/C question posed as a
testing problem, the most relevant single paper for the non-blocking/livelock gap;
**Aladdin** (ISCA'14) for why the dependence graph is built the way it is; **FIFOAdvisor**
and the INRIA max-plus FIFO-sizing work as a possible cheap Tier 0 for DSE pruning;
**Carloni on backpressure** for making valid_only/valid_ready precise rather than ad hoc.
Learned surrogates (IronMan, hierarchical-GNN QoR) need a large labelled corpus — they
*raise* the ground-truth bar rather than removing it. Do not start there.

PDFs are in `papers/`. Full per-paper analysis is in `archive/claude_simulator.md` §P1–P5.

---

## 7. Validation strategy

- **Ground truth = the SystemC/Catapult RTL cosim path**, which is proven bit-exact and,
  crucially, yields **exact cycles for NB designs** where csynth reports `undef`. Two
  complementary oracles: csynth (static, cheap, partial for NB) and RTL cosim (dynamic,
  exact, slow, works for everything). LightningSim is a third (trace-based, exact for
  Type A, ~95× faster than cosim).
- **Determinism regression:** `nb_nondeterminism.py` must stay at 1 outcome — promote to CI.
- **Keep green:** `test_stream_ops_sim.py`, `test_stream_nb_simple.py`,
  `test_stream_nb_scalar.py`, `test_nested_subregion_streams.py`, plus golden
  `test_df_unit.py` / `test_region_stateful.py`.
- **Per-protocol unit tests:** minimal 2-PE producer/consumer for Wire, ValidOnly,
  ValidReady.
