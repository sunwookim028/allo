# Routes to an accurate cycle model

Written 2026-07-28. Companion to `simulator_concept.md` (architecture) and the paper
analysis in `claude_simulator.md` §P1–P5. Decision note, not an implementation plan.

---

## 0. Correction to a recorded finding — the highest-accuracy route is NOT closed

`claude_simulator.md` and the `csynth-cost-model-ground-truth` note both record:

> Non-blocking designs give NO static schedule. Every loop reports `lat=- II=- depth=-`
> and overall latency `undef` … **schedule ingestion cannot be an accuracy route for
> NB designs.**

**That is too strong.** Re-reading `nb_stream.prj/out.prj/solution1/syn/report/csynth.rpt`:

```
|                  Modules & Loops         | … | Latency | Iteration | Interval | Trip | Pipelined |
| + top_nb*                                | … |    -    |     -     |    -     |  -   | dataflow  |
|  o l_S_i_0_i                             | … |    -    |     -     |    -     |  -   |    no     |
|   o VITIS_LOOP_24_1                      | … |    -    |     1     |    1     |  -   |   yes     |   <--
|   o VITIS_LOOP_44_1                      | … |    -    |     1     |    1     |  -   |   yes     |   <--
|  o l_S_store_res0_store_res0_l_0         | … |   11    |     9     |    1     |  4   |   yes     |   <--
```

What is `undef` is **aggregate latency and trip counts** — necessarily so, because the
retry loops are unbounded and the tool cannot know how many attempts happen.
**Per-iteration latency and II are reported**, including for the NB retry loops
themselves (1 cycle per attempt).

This is exactly the split trace-based simulation is built on: **the static schedule
supplies per-iteration timing; the dynamic trace supplies the trip counts the tool
cannot know.** The thing we concluded was missing is the thing the trace provides.

Caveat: the *outer* loops (`l_S_i_0_i`) still report `-`, so the schedule is partial —
usable for the pipelined inner regions, not a complete static timing of the design.

### 0a. Confirmed at operation granularity — `.verbose.sched.rpt`

`csynth.rpt` is the wrong file to read anyway. `…/solution1/.autopilot/db/*.verbose.sched.rpt`
schedules **every operation**, names the hardware core, and gives its latency and II.
**These reports exist and are complete for the NB project**, despite its `undef` aggregate.

The NB retry loop is a single state:

```
State 1 <SV = 0> <Delay = 1.21>
  ST_1 : 'nbwrite'  @_ssdm_op_NbWrite.ap_fifo  [kernel.cpp:27]   <Delay = 1.21>
  ST_1 : 'br' %v4 → cleanup.cont | for.inc.exitStub               (the retry branch)
```

**One state, II=1 → a failed `try_put` attempt costs exactly 1 cycle**, matching the
`csynth.rpt` row (`VITIS_LOOP_24_1: Iteration Latency = 1`). Vitis supplies
cost-per-attempt; our simulator supplies the attempt count; neither alone yields the
total. That is §3.2 demonstrated on the hardest case.

Measured cores, both projects (`xcu280`, 3.33 ns clock):

| core | latency | our table says |
|---|---|---|
| `RAM` | **1** | `ARRAY_LOAD_LATENCY = 2` — **too high** |
| `FIFO_SRL` | 0 | `stream_put/get = 1` — see the chaining caveat |
| register scalar (no core) | 0 | 0 — **confirms `_is_register_memref`** |
| `Adder`, `Cmp` | 0 | free — **confirms `_FREE_OPS`** |
| NB retry iteration | 1 cycle | `stream_try_put = 1` — **correct** |

**Chaining caveat — core latency ≠ scheduled cost.** `FIFO_SRL` reports `Latency = 0`,
yet in `producer_0` the write still occupies its own state because its 1.21 ns delay
will not chain under a 3.33 ns clock. So the quantity to ingest is the **state distance
(`ST_n`)**, not the core's structural latency. Reading `Latency = 0` and charging streams
zero would be wrong.

**Join is easier than assumed:** Vitis extracts each pipelined loop into its own function
(`producer_0_Pipeline_VITIS_LOOP_24_1`), so the extracted function name carries the loop
identity — join on that rather than needing a per-source-line fallback.

Cheap first win that needs no join at all: harvest `(opcode, core, latency, delay)`
across a sweep and fit the table empirically.

---

## 1. What "accurate" has to mean here

Two different consumers, two different bars:

| consumer | needs | a constant offset is… |
|---|---|---|
| **DSE ranking** (the agents' generator picking a design) | correct *ordering* of candidates | **harmless** |
| **Reporting** ("this design takes N cycles") | correct *magnitude* | **fatal** |

Today's model is calibrated at exactly **one point** (7 modelled vs 6 reported, 1.2×) on
two trivial integer PEs. Float, BRAM and pipelined-loop latencies are wholly
uncalibrated, and the makespan is 0.4× because `load_buf`/`store_res` are not clocked at
all. So we can currently defend neither bar — not because the model is bad, but because
**one data point cannot distinguish a good model from a lucky one.**

---

## 2. The three routes

### Route A — trace + static schedule (LightningSim / OmniSim)

Instrument LLVM IR to record executed basic blocks, join that trace to the csynth
schedule, build a graph of FIFO-access events with dependency edges, take the longest
path. **LightningSim: 99.9 % accurate, up to 95× faster than RTL co-sim. OmniSim:
0.09 % mean error, up to 35.9× over co-sim and 6.61× over LightningSim.**

- **Fit:** strong. Allo already lowers MLIR → LLVM, and loop names already join MLIR to
  the report by construction (`EmitVivadoHLS.cpp:972`).
- **Now viable for NB designs** — see §0. OmniSim's whole contribution is precisely the
  dataflow/NB designs commercial tools decline to simulate.
- **Real cost, and it is not the one we thought:** it needs a **csynth run per design**
  (~35 s). That is fatal *inside* a DSE inner loop, but it is not needed there — what
  varies in DSE is usually mapping/tiling/FIFO depth, not the PE body. **Schedule per
  kernel body can be extracted once and cached across configurations.**
- Retry-loop names come back as `VITIS_LOOP_24_1` (source-line) rather than the joined
  `l_…` form, so the MLIR↔report join needs a fallback for exactly the NB loops.

### Route B — per-PE local time (DAM-style; what we have)

Each PE owns a monotonic local time, advanced by a hand-authored latency table;
channels carry timestamps. DAM reports **~0.3 % (0.8 cycle)** vs RTL, up to 4 orders of
magnitude faster than cycle-stepped simulation.

- **Fit:** already built and working; no tool dependency; fast enough for a DSE inner loop.
- **Ceiling is the table, not the architecture.** DAM's 0.3 % is with *author-tuned*
  latencies. Ours are educated guesses validated once.
- This is the right **Tier 1** engine. It does not need replacing — it needs calibrating.

### Route C — calibrate against ground truth at scale

Not an alternative engine; the **measurement infrastructure both A and B need.**

- We already have a **proven** oracle: SystemC → Catapult → **Xcelium RTL co-sim**,
  demonstrated bit-exact on an allo-emitted design (2026-07-26). Crucially this yields
  **exact cycles for NB designs**, where csynth reports `undef`.
- So there are two complementary oracles: **csynth** = static, cheap, partial for NB;
  **RTL co-sim** = dynamic, exact, slow, works for everything.

---

## 3. The plan — keep our runtime, replace guessed latencies with measured ones

**One sentence: delete the hand-written per-op latency table and charge the PE clock
with per-loop timing ingested from csynth instead.**

Today `_op_latency` guesses (`fp add = 4`, `mul = 3`, …). Those guesses are the dominant
error source — the architecture is not the problem, the numbers are.

### 3.1 Why this fits us and not the papers' shape

**Our simulator already *is* the trace.** That is the whole leverage point.

LightningSim and OmniSim need two passes: an instrumented functional run producing a
trace, then a post-pass joining trace to schedule and taking a longest path. We need
neither — we already execute the real program, per PE, with a local clock. So the
measured latencies can be injected **directly into the running simulation**.

Result: OmniSim's *accuracy source* (real HLS schedule numbers) on DAM's *runtime*
(no trace pass, no event graph, non-blocking handled natively).

### 3.2 The division of labour — the actual point

| supplies | provided by |
|---|---|
| **compute latency** — how long a loop body takes | csynth static schedule (iteration latency, II) |
| **trip counts** — how many times it runs | our execution, free |
| **stall time** — waiting on full/empty FIFOs, contention | our simulation, dynamically |

This is why §0 matters. csynth *cannot* give trip counts for NB designs (`undef`) — and
it does not need to, because that is the half we already have. **Each side supplies
exactly what the other cannot.** A static-schedule-only route dies on NB; a
guessed-latency-only route (today) dies on accuracy; together they cover each other.

### 3.3 Mechanism

1. csynth each `@df.kernel` body **once**; read per-loop `{iteration latency, II,
   pipelined}`.
2. Join report → MLIR by loop name. The naming already does this by construction
   (`l_<op>_<loop>`, `EmitVivadoHLS.cpp:972`), with a source-line fallback for the NB
   retry loops that come back as `VITIS_LOOP_24_1`.
3. In `_insert_clock_increments`, charge the **ingested** per-iteration latency in place
   of summed guesses. The `(n−1)·II + body` formula already exists — only the numbers
   change.
4. **Cache per kernel body.** DSE varies mapping/tiling/FIFO depth, not the PE body, so
   one csynth run (~35 s) amortises across a whole sweep and stays out of the inner loop.
5. Ingest the wrappers too: `store_res0_1` reports latency 12 / II 4 and is currently
   clocked at **zero** — this is what makes makespan 0.4×.
6. Keep today's table as the fallback when no report exists, so the simulator still runs
   without Vitis.

### 3.4 Why not the alternatives

- **Full trace+graph (Route A):** buys little over the above while costing a trace pass
  and an event-graph builder, because it reconstructs dynamic information we already have.
- **Learned surrogate:** needs a large labelled corpus — i.e. the oracle below. It
  *raises* the measurement bar rather than removing it. Not a shortcut.
- **Analytical / max-plus:** closed-form and fast, but assumes static rates; NB and
  data-dependent control are precisely where it breaks. Candidate Tier 0 for coarse
  pruning, not the main model.

### 3.5 First step is the oracle, not the ingestion

With one calibration point we could not tell whether ingestion helped. So:

1. **Ground-truth sweep first** — 10–20 small designs spanning int/float, blocking/NB,
   varying FIFO depth and 2–8 PEs, pushed through both oracles (csynth; Catapult/Xcelium
   RTL co-sim, which is exact and works for NB). Emit *design → predicted → csynth → RTL*.
2. **Then ingest**, and report the **error distribution before and after** — not a single
   ratio.

**The assumption the whole plan rests on:** that a kernel's schedule is stable across the
configurations DSE varies. If Vitis re-schedules per mapping, per-kernel caching breaks
and the synthesis cost lands back in the inner loop. **This is untested and should be the
first thing the sweep answers.**

---

## 4. Open questions

- **Does Vitis re-schedule per mapping?** (§3.5 — the load-bearing assumption.)
- Do rankings actually need absolute accuracy, or only ordering? Decides how much of the
  wrapper/offset work is worth doing.
- Can Catapult/Xcelium co-sim be scripted headlessly for a sweep, or is it interactive?
- The `load_buf`/`store_res` gap: constant offset (harmless for ranking) or
  design-dependent (fatal for both)? Measurable with the §3.5 sweep.
- Outer loops still report `-`, so ingestion is partial. How much of a typical PE's time
  sits in regions the schedule does not cover?

## 5. Further literature (beyond the four already read)

Grouped by what each would actually contribute. **Bold = read these first.**

### 5a. Trace-based / pre-RTL simulation (same family as OmniSim)

- **LightningSim** (FPGA'23) — OmniSim's predecessor and the cleaner statement of the
  method: trace LLVM IR, map static HLS scheduling onto the trace, compute stalls and
  deadlocks from inter-function interaction. **99.9 % accurate, up to 95× faster than
  RTL co-sim.** Closest thing to a reference implementation for Route A, and it operates
  at the level Allo already lowers to.
- **Aladdin** (ISCA'14) — pre-RTL power/performance from a **dynamic data dependence
  graph**, no RTL generated; within 10 % of RTL flows. Older and coarser than
  LightningSim, but it is the origin of the "constrain an unconstrained DDDG" framing
  and is worth reading for *why* the graph is built the way it is. `gem5-Aladdin`
  extends it to accelerator+memory-system co-simulation.

### 5b. Analytical throughput models (no trace, no tool run — cheapest tier)

Potentially a **Tier 0** below our current engine: closed-form throughput for a dataflow
graph, fast enough to evaluate thousands of candidates.

- **Throughput and FIFO sizing for latency-insensitive designs** (INRIA) — max-plus /
  marked-graph analysis giving optimal FIFO sizes at maximum achievable throughput.
  Directly relevant: FIFO depth is a DSE knob we currently model only dynamically.
- **FIFOAdvisor** (arXiv 2510.20981, 2025) — a DSE framework for automated FIFO sizing
  of HLS designs; SDF buffer sizing via static analysis plus an SDC optimisation model.
  The most current statement of this line and closest to our DSE use case.
- **The role of back-pressure in latency-insensitive systems** (Carloni, Columbia) — the
  foundational semantics for what back-pressure *is*. Useful for making
  `valid_only` / `valid_ready` / wire precise rather than ad hoc.

### 5c. Learned surrogates (predict cycles instead of simulating them)

Relevant only if DSE needs to score far more candidates than either tier can simulate.

- **IronMan** (GLSVLSI'21) — GNN performance predictor + RL DSE; reduces HLS tool
  prediction error by 5.7× in timing, 10.9× in resources.
- **Hierarchical GNN source-to-post-route QoR** (DATE'24) — predicts *post-route* QoR
  from C source ([code](https://github.com/sjtu-zhao-lab/hierarchical-gnn-for-hls)).
- Caveat worth stating plainly: a surrogate needs a **large labelled corpus**, which is
  the §3.1 harness again. Learned models do not remove the ground-truth requirement —
  they raise it. Do not start here.

### 5d. Directly adjacent to our open problems

- **Latency-insensitivity testing for dataflow HLS designs** (FPGA'25, Edinburgh) —
  *does a design's result depend on timing?* This is OmniSim's Type A/B/C question posed
  as a testing problem, and it is the same property our determinism work asserts.
  **Most relevant single new paper for the non-blocking/livelock gap.**
- **StreamTensor** (arXiv 2509.13694, 2025) — streaming dataflow accelerators for LLM
  inference; useful as a workload/topology source once the model needs realistic designs
  rather than 2-PE toys.

## Sources

- [LightningSim (arXiv 2304.11219)](https://arxiv.org/pdf/2304.11219)
- [OmniSim, MICRO'58 (ACM DL)](https://dl.acm.org/doi/full/10.1145/3725843.3756033) ·
  [arXiv 2508.19299](https://arxiv.org/html/2508.19299v1)
- DAM, ISCA'24 — `simulator_papers/DAM_ISCA24_dataflow_abstract.pdf`
- [Aladdin, ISCA'14](https://people.eecs.berkeley.edu/~ysshao/assets/papers/shao2014-isca.pdf) ·
  [code](https://github.com/harvard-acc/ALADDIN)
- [Throughput and FIFO sizing for latency-insensitive designs (INRIA)](https://inria.hal.science/inria-00381644v1/document)
- [FIFOAdvisor (arXiv 2510.20981)](https://arxiv.org/pdf/2510.20981)
- [Back-pressure in latency-insensitive systems (Carloni)](https://www.cs.columbia.edu/~luca/research/rbilsENTCS06.pdf)
- [Latency-insensitivity testing for dataflow HLS designs, FPGA'25](https://www.pure.ed.ac.uk/ws/portalfiles/portal/486717247/ChengEtalFPGA2025LatencyInsensitivityTesting.pdf)
- [IronMan (ACM DL)](https://dl.acm.org/doi/abs/10.1145/3453688.3461495)
- [Hierarchical GNN QoR, DATE'24 (arXiv 2401.08696)](https://arxiv.org/pdf/2401.08696)
- [StreamTensor (arXiv 2509.13694)](https://arxiv.org/pdf/2509.13694)
- Local reports: `nb_stream.prj/…/csynth.rpt`, `blocking_stream_csynth.prj/…`

---

## 6. Other ideas (not yet in the plan)

Roughly in order of value.

1. **Calibrate against Catapult, not Vitis.** We emit SystemC → Catapult → Xcelium, and
   that path is *proven bit-exact*. Vitis is a different scheduler with different cores
   and clock assumptions, so calibrating to Vitis while shipping through Catapult targets
   the wrong tool. Catapult emits its own scheduling reports. **This could invalidate the
   §3 ingestion target and should be resolved before building it** — check which backend
   the generated designs actually go through.
2. **Sensitivity analysis before any calibration.** Results are deterministic, so perturb
   one latency constant at a time and see which move the makespan. Most probably do not —
   the critical path dominates. Turns "calibrate ~30 constants" into "calibrate the 3 that
   matter." No tool runs needed.
3. **Validate rankings, not magnitudes.** If DSE only needs ordering, measure Kendall-τ /
   Spearman over design pairs instead of per-design absolute error. Far cheaper (a modest
   RTL-cosim set suffices) and measures the property we actually care about.
4. **Emit an interval, not a point.** Optimistic/pessimistic bounds rather than one
   number. DSE can prune on bounds, and it is honest about a model calibrated at one
   point — an interval that brackets the truth beats a point confidently 20 % off.
5. **Report critical-path attribution.** We already have per-PE clocks, so we can say
   *which* PE or stream dominates the makespan. For the agents' generator that is more
   actionable than a scalar, and it needs no accuracy improvement at all.
