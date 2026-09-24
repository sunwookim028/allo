# claude_evaluation.md — evaluation shell charter

**Created 2026-08-02. Branch `wire`.** Companion to `claude_simulator.md` (simulator shell).

This shell has **one job: turn the work of the last months into defensible, publishable
evaluation**. Not new features. Where a feature is genuinely missing for a measurement we
want, that is a scoped sub-task justified by the measurement — never the other way round.

Three things to evaluate:

| Track | Subject | Baseline(s) |
|---|---|---|
| **A** | the new **SystemC/Catapult backend** for Allo | the existing Allo backends (Vitis HLS `vhls`, JIT simulator); hand-written SystemC/MatchLib |
| **B** | **EVA in Allo** — old Allo backend (Vitis) **and** new SystemC backend | the **original EVA SystemVerilog RTL** |
| **C** | **NoCs / routers**: our Allo rebuilds using the new link concepts (non-blocking `try_*`, `Channel[valid_ready\|valid_only]`, `Wire`) | **RaveNoC** RTL + at least one more published NoC benchmark |

### Decisions taken (2026-08-02, user)

1. **"Old Allo" = the Vitis HLS (`vhls`) backend.** Track B's middle column is EVA through
   the existing Vitis path on this fork. The JIT simulator is a *functional reference only*,
   not a compared flow (it is impractically slow on EVA and currently regressed).
2. **The EVA baseline is the original SystemVerilog**, `/home/zsm9/allo/EVA/EVA_untouched/EVA/`
   — the real chip RTL with DesignWare FP16 IP and the VCS testbenches in `rsim/`+`tb/`.
   Vitis-generated RTL is *not* the golden; it is one of the things being evaluated.
3. **NoC baselines: all four**, but in two tiers (see §4.C5) — RaveNoC and MatchLib
   `WHVCRouter` are **run** baselines (we execute them here, same simulator, matched config);
   CONNECT and OpenSMART are **literature anchors** (we cite their reported values and match
   their metric format, we do not re-run their flows).
4. **Output = a thesis chapter.** Consequences: full methodology defence in-line, and
   **negative results and the design-space narrative are in scope** — the Wire-has-no-
   synchronisation result, the fusion 3.9× cost, the `-IO_MODE super` saga, the
   compile-time-constant-index expressiveness cost. These are chapter material, not
   embarrassments to hide. Can be condensed into a paper later; the reverse is expensive.

---

## 0. Ground rules (read before running anything)

**Two-checkouts trap.** Always `PYTHONPATH=/home/zsm9/allo_sup`. A bare import grabs the
installed `/home/zsm9/allo` and silently runs *stale* code.
(`[[allo-two-checkouts-trap]]`)

**Env blocks** — these are load-bearing and each cost hours to rediscover:

```bash
# --- SystemC csim / csynth / cosim (Catapult 2024.2 + Xcelium) ---
export MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home
export SYSTEMC_HOME=$MGC_HOME/shared
export MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu
export CDS_LIC_FILE=5280@en-license-05.coecis.cornell.edu
export NC_ROOT=/opt/cadence/XCELIUM2403          # also NCSim_NC_ROOT
export ALLO_CXX_EXTRA="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"   # GLIBCXX_3.4.26
export PATH=/home/zsm9/miniconda3/envs/allo/bin:$MGC_HOME/bin:$PATH        # conda python3 FIRST
unset LD_PRELOAD                                  # libtinfo/ncurses wrong-ELF noise

# --- JIT dataflow simulator (reference) ---
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build-rhel8
export OMP_NUM_THREADS=8

# --- RaveNoC RTL benches ---
conda activate ravenoc     # cocotb 1.9.2, NOT 2.x (2.x removed cocotb.fork)
```

**Disk.** `/tmp` is a shared 15 G mount and *has* filled from Catapult projects. Do all
sweep work in `/scratch` (400 G local) or `/work/shared/users/zsm9`. Home quota is 20 G
soft and has blocked every `df.build` before.

**Process isolation.** Dataflow designs SEGFAULT at exit (OMP/GC race). Any sweep must run
one test *per subprocess*, or one crash kills the whole run.

**Catapult build-subdir quirk.** Synthesize in a *build subdir*, never the source dir, or
`Connections` In/Out ports silently degrade to raw `sc_signal` → SCHD-30 and RTL with **no
backpressure logic at all**. Any artifact archived from a flat build must be regenerated.
(`[[catapult-connections-cwd-quirk]]`)

**Rebuild after touching the emitter:**
`cd mlir/build && conda run -n allo ninja tools/allo/_mlir/libAlloMLIRAggregateCAPI.so`

---

## 1. Evidence inventory — what we already have

This section is the honest starting balance. Everything below is *measured*, with the
caveats attached. Do not re-derive; do re-verify anything marked ⚠.

### Track A — SystemC backend

| Level | Status | Source |
|---|---|---|
| emit | never crashes; every dataflow design emits | broad sweep, `[[systemc-backend-test-status]]` |
| csim functional | `test_systemc_backend.py` **28/28**; dataflow cosim-vs-JIT **bit-exact** on every checkable design except 2 known-semantic + stateful gap | ibid. |
| csynth (Catapult) | **22/32** dataflow test files, **1** residual scheduling failure in the whole set (was 2/32 before the `-IO_MODE super` fix) | `/scratch/cosim_work/csynth_results.txt` |
| RTL cosim (Xcelium) | **45 unique designs RTL == C bit-exact, ZERO mismatches** | `[[systemc-csynth-status]]`, cosim sweep 2026-07-30 |
| `mode="cosim"` | first-class (`df.build(..., mode="cosim")`), committed | `[[cosim-first-class-mode]]` |
| csim cycles == RTL cycles | **5/5 exact** (hello_channel 11/11, pe_channel 41/41, pe_stream 43/43, fifo_stream, vc_buffer_channel) | `[[csim-cycles-equal-rtl-cycles]]` |

⚠ The cycle-equivalence claim is **not yet validated for `try_*` designs**, which is exactly
where MatchLib's timing model could diverge. This is a *gating* experiment for Track C.

Known open gaps (each must be either fixed or explicitly declared out of scope in the
write-up, not quietly omitted):
- stateful-region `__stateful_*` undeclared under synth (7 designs)
- `Stream.empty()/full()` on a **cross-kernel** stream is non-functional in SystemC
  (`Connections::In::Empty()` doesn't reflect channel state) — fix designed, 6 edit sites,
  not implemented (`[[channel-peek-canget-canput-deferred]]`)
- `Channel[valid_only]` lowers identically to `valid_ready` — the lighter
  `sc_signal`+valid form is validated by hand but not emitted
- `large_scale_gemm`: depth-1024 `Stream` → register-file FIFO needs a 1025-way mux; needs
  a RAM-based FIFO
- 2D-systolic-in-subregion stream-direction inversion (direction-inference vs DCE ordering)

### Track B — EVA

- `examples/systemc/eva_example/` — reproducible rtprime example, in-repo.
- 1×1 **functional cosim PASS bit-exact** (systolic passthrough, numpy golden).
- **8×8 emits** (139,981-line `kernel.cpp`, 1152 AlloFifo, 192 kernel instances) and **g++
  csim compiles clean**. 8×8 Catapult csynth was launched, outcome unknown — check
  `/scratch/cosim_work/eva8_csyn.prj/eva8_csyn.log`.
- **EVA now synthesizes through Catapult**: fp16 bitcast (CIN-71, `802c9eb`), loop-pragma
  placement (CIN-319, `cd9f1ad`), scheduled top (`9546e4f`) → node loop pipelined at **II=1**.
- **Catapult vs Vitis, same scheduled EVA 1×1**: both II=1. Catapult = ASIC nangate-45nm,
  148 k µm², 487 modules, ~500 MHz (slack −0.08). Vitis = FPGA xczu7ev, 6 DSP, ~8584 FF /
  18469 LUT, Fmax 143.5 MHz. **Not directly comparable as stated** — this needs an
  apples-to-apples framing (see §4.B).
- **Cosim vectors already exist — do NOT regenerate.**
  `Vitis_HLS/bubble_model/final_runs/cosim_8x8/` (`vec_{mmm,fft,mmm2,mmm4,mmm_x2}_eva_prime.npz`
  + `vectors_*.h` + `tb_replay.cpp`), and `final_final/prime/vectors_prime_{1x1,4x4}.h`.
  Wrong turn to avoid: `Allo/EVA/archive/tests/cosim/vec_*.npz` are for the *compile-time-prime*
  variant and give all-zeros on rtprime.
- Original RTL: `/home/zsm9/allo/EVA/EVA_untouched/EVA/`, summarized in
  `pe_core_implementation/EVA_reference.md`. SystemVerilog, DesignWare `DW_fp_*` FP16,
  8×8 cores per `pe_array`, VCS testbenches in `rsim/` + `tb/`.
- `Vitis_HLS/EVA_HLS_vs_RTL_Comparison.docx` exists — **read it first**, it may already
  contain half of Track B.

### Track C — NoC

**RaveNoC reference (ours — RaveNoC publishes nothing).** `agents/noc/ravenoc_bench/`,
committed `e0c8564`. Config: 32-bit flits, depth 2, 2 VCs, 2.0 ns, Xcelium + cocotb 1.9.2.

| design | mode | cycles | throughput | zero-load latency |
|---|---|---|---|---|
| `fifo` | max throughput, 64 flits | 65 | 1.016 cyc/flit | 1 |
| `vc_buffer` | greedy | 65 | 1.016 cyc/flit | 1 |
| `vc_buffer` | backpressure 1-in-3 | 193 | 3.016 cyc/flit | 3 first, mean 4.97 |
| `rr_arbiter` | saturated, 4 req | 65 | 0.985 grants/cyc, `[16,16,16,16]` | steady wait 4.00 |

**Allo counterparts** (`/home/zsm9/final_noc/results/`, both cosim bit-exact):

| | slope cyc/flit | fixed overhead | RTL modules |
|---|---|---|---|
| RaveNoC `fifo.sv` | 1.016 (incl. 1 startup bubble) | +1 | 1 |
| Allo `Stream[int32,2]` | **1.000** | +5 | 21 |
| Allo `Channel[valid_ready]` | **1.000** | +3 | 18 |

→ **Allo's generated links are throughput-equivalent to hand-written RTL**; Channel saves 2
cycles of fixed overhead and 3 RTL modules over Stream. Fits are exact (residual 0.00 at
N=8..128).

**Routers built** (`agents/noc/`): `_ports` (per-port split), `_fused`, `_chan` (Channel),
`_vc` (virtual channels), `_mesh` (2×2, 4×4), `_worm` (wormhole), `_wire`, `_adapt`.
Test IDs T1 ejection / T2 XY / T3 lossless / T4 fairness / T5 two VCs / T6 head-of-line /
M1-M2 mesh / P1-P3 wormhole. The Channel router **csynths clean, csims T1-T4, and cosims
bit-exact** since the `-IO_MODE super` fix.

**Measured design facts worth reporting as results in their own right:**
- Fused vs split: fusion beat the split on fidelity and cost (35→10 streams, 12→3 PEs) but
  serialises 5 ports → ~3.9× more cycles on the critical path (10745 vs 2778).
- VCs demonstrably work — T6 with the NVC=2-on-vc0 **control** proves the win is the virtual
  channel, not the extra buffer.
- Wormhole costs ~2× cycles (21253 vs 10745) and *creates* the head-of-line blocking VCs fix.
- **A `Wire` gives zero storage AND zero alignment** — controlled experiment `pe_split.py`
  (mono PASS / stream PASS / wire FAIL-all-zeros). This is a real negative result.
- Non-blocking ≠ non-consuming: `try_get` consumes, so poll-then-arbitrate destroys losers.
  `Stream` has `empty()`/`full()`; `Channel` and `Wire` have no probe.
- The Catapult `-IO_MODE super` saga: plain Connections ceiling was measured at **exactly 1
  conditional handshake per loop body** under the default `-IO_MODE fixed`; the fix is 2
  lines in `allo/backend/catapult.py`, and it is *behaviour-preserving* (45/45 cosim).

⚠ **PARKED / UNUSABLE:** `final_noc/bench/tb_allo_link.py` reports 2–3 cyc/flit where the
design provably does 1.0. Do not use its numbers. Acceptance test before reuse: greedy on
`hello_channel` must give ~1.0 cyc/flit.
⚠ `final_noc/allo/router_faithful.py` builds+runs but delivers nothing; one fix applied
(blocking `get`→`try_get`) and **not yet re-run**.
⚠ `final_noc/` is **not under git**.

---

## 2. What "properly evaluate" means here

Each track must produce, for every claim:

1. **A claim stated so it can be false.** "The SystemC backend is good" is not a claim.
   "For 45 dataflow designs the Catapult RTL is bit-exact with the C model, and cycle counts
   from csim match RTL cycle counts exactly on 5/5 measured designs" is.
2. **A baseline that is a fair fight.** Same simulator on both sides where possible (we
   already use Xcelium for both RaveNoC and our RTL — that is what makes the NoC comparison
   mean anything). State the configuration match explicitly (flit width, depth, VCs, clock).
3. **A measurement window that is defensible.** Ours is `[first accepted transfer .. last
   accepted transfer]`, excluding reset, because the 17-cycle Catapult reset is a flow
   artifact. Say so in the paper; don't hide it.
4. **A control.** The VC result has one (NVC=2-on-vc0). The fairness result has one (a
   priority encoder hits the same throughput with a `[64,0,0,0]` distribution). Every
   headline claim should get one.
5. **A stated negative / limitation.** Wire has no synchronisation. `empty()/full()` is
   broken cross-kernel. Fusion costs 3.9×. These make the evaluation credible, not weaker.

**Metrics we will report** (superset; per-track selection in §4):

- *Functional*: bit-exactness vs a golden (numpy, JIT sim, or captured RTL vectors); pass
  rate over a design suite.
- *Performance*: cycles; throughput (cyc/flit or flits/cyc); **zero-load latency**;
  **saturation throughput** from a latency-vs-offered-load curve; fairness distribution;
  buffer occupancy.
- *Cost*: Catapult area (nangate-45nm µm², gate count, register bits, module count) and
  clock/slack; Vitis LUT/FF/DSP/BRAM and Fmax where the FPGA path applies.
- *Productivity*: lines of source, number of design variants reachable per unit effort,
  parameter sweeps that are one-line in Allo and a rewrite in RTL. (This is MatchLib's own
  headline metric and is legitimate — but it must be measured, not asserted.)
- *Tool-flow*: emit/csim/csynth/cosim pass rate; wall-clock per stage.

---

## 3. Paper / prior-art survey (do this early — it sets the metric list)

**Purpose:** we must report *the values the field reports*, in *their* format, or a reviewer
cannot place us. Right now we report cyc/flit slope + fixed overhead; the NoC field reports
latency-vs-injection-rate curves under named synthetic traffic. That gap is the single
biggest methodology risk in Track C.

For each paper below, extract into `evaluation/papers/<key>.md`: **what they measure, under
what traffic/workload, on what technology, and the actual numbers**. Do not paraphrase from
memory — open the PDF.

**NoC (Track C):**
- **CONNECT** (Papamichael & Hoe, FPGA'12) — FPGA-focused NoC generator. Already flagged as
  an inspiration repo (`pe_core_implementation/inspo_router_noc/`). Its "one invocation ==
  one clock cycle" discipline is exactly the Wire-alignment argument we made.
- **OpenSMART** (Kwon & Krishna, ISPASS'17) — single-cycle NoC generator, HLS-ish flow.
- **MatchLib / NVIDIA modular VLSI flow** (Khailany et al., DAC'18) + the MatchLib
  Connections papers — *the* direct methodology baseline, since our backend emits exactly
  this. They report area/perf vs hand RTL and a productivity multiplier. Note: MatchLib's
  own `WHVCRouter` ships inside our Catapult install at
  `$MGC_HOME/shared/pkgs/matchlib/cmod/include/WHVCRouter.h` **with unit tests** — a
  synthesizable, same-toolchain, same-library NoC baseline sitting on this machine.
  **This is probably our best second NoC benchmark.**
- **BookSim2 / Garnet (gem5)** — the standard cycle-level NoC simulators; they define the
  canonical load-latency methodology and traffic patterns (uniform random, transpose,
  bit-complement, hotspot, tornado). Even if we don't run them, we should adopt their
  x-axis.
- **Dally & Towles**, *Principles and Practices of Interconnection Networks* — for the
  definitions of zero-load latency and saturation throughput we will cite.
- **RaveNoC** — confirmed to publish **nothing**. Our numbers are the first; say so.

**HLS / dataflow (Tracks A & B):**
- **Allo** (PLDI'24) — the parent system; match its evaluation format for the backend claim.
- **DAM** (ISCA'24, PDF already in `simulator_papers/`) — dataflow-abstract-machine
  simulation; relevant to the "csim cycles == RTL cycles" claim and to the simulator shell.
- `simulator_papers/2508.19299v1_omnisim.pdf`, `2504.03083v1_unlockin_the_amd_npu.pdf`,
  `2604.17862v1_m100.pdf` — already downloaded, mine for metric conventions.
- Spatial / Aetherling / Dahlia / HeteroCL — for what "productivity + quality-of-results"
  claims look like in this community.

**EVA (Track B):** check whether EVA has a publication or internal report with reported
numbers (`EVA_reference.md` cites none; `Vitis_HLS/EVA_HLS_vs_RTL_Comparison.docx` and
`Updates.pptx` are the likely internal sources). If there is no published EVA number, then —
as with RaveNoC — every reference figure is ours to produce, and the comparison must be
**re-measured on both sides under one flow**, not lifted from a slide.

---

## 4. Per-track plan

### Track A — the SystemC backend

**Claim shape:** *Allo's SystemC/Catapult backend takes an unmodified Allo dataflow program
to synthesizable, RTL-verified hardware, at parity with the existing Vitis path on
functionality and at ASIC-flow quality the Vitis path cannot reach — with N designs proven
bit-exact against RTL.*

**A1. Refresh the pass-rate matrix on the current HEAD.** Every count we have is from a
different day and a different emitter state. Produce **one** table, one run, one commit
hash, covering all `tests/dataflow/test_*.py`: `EMIT / CSIM / CSYNTH / COSIM` ×
{OK, FAIL(reason)}. Per-file subprocess, `/scratch`, results to a CSV. Harness exists:
`scratchpad/scpatch_sweep.py` + `run_sweep.sh`; the `mode="cosim"` backend path now does
most of what those plugins hand-rolled — prefer the committed path.
*Also check `/scratch/cosim_work/sweep_results.txt` for the sweep that was in flight.*

**A2. Backend-vs-backend parity table.** Same designs, three columns: JIT simulator (golden),
Vitis `vhls` csim/csynth, SystemC csim/csynth/cosim. This is the "old Allo vs new backend"
axis. Where Vitis can't (ASIC, real interconnect) and where SystemC can't (the stateful gap)
must both appear.

**A3. Quality-of-results.** For a chosen subset (~6 designs spanning stream / mem-port /
systolic / non-blocking), report Catapult area+timing and Vitis LUT/FF/DSP+Fmax side by
side, with the framing fixed (§4.B applies here too). Include the **cost of the link
abstraction**: Stream = AlloFifo + 5 RTL modules + 137 register bits; Channel = 3 wires, 0
modules. That is a clean, quantified backend result.

**A4. The cycle-model claim — DEMOTED (user decision, 2026-08-02).**
**Every reported performance number goes through csynth + RTL cosim. csim alone is not
trusted as a measurement path, regardless of how well it agrees.** The 5/5 csim==RTL
agreement stays in the write-up as an *observation* about the flow's fidelity — never as a
substitute for the RTL number. So A4 is no longer a gate on anything; run it only if we want
that observation extended to the `try_*` family for completeness.

**The cost model this implies (and it is friendlier than "hours per datapoint"):**
- **csynth is per DESIGN VARIANT** — hours, sometimes many (EVA 8×8 ran ~6 h and wasn't done).
- **cosim is per DATAPOINT** — minutes, reusing the already-synthesized RTL. `final_eva_
  performance` already exploits exactly this: one 8×8 csynth at
  `/work/shared/users/zsm9/eva_ts_8x8_rtl/wd`, then ~30 min of cosim per workload.

→ **Sweeping traffic/load/injection-rate is CHEAP. Sweeping architecture is EXPENSIVE.**
The plan must therefore: (a) fix a *small, deliberately chosen* set of design variants, each
paid for once with a csynth; (b) drive as many traffic points as we like through cosim on
each. Any variant sweep (§4.C4: fused-vs-split, NVC ∈ {1,2,4}, wormhole on/off, depth, mesh
size) must be **costed in csynth-hours before it is launched**, not discovered mid-run.

**A5. Decide the fate of each open gap** (list in §1). For each: fix / scope out / document
as limitation. The `Stream.empty()/full()` cross-kernel bug is the one most likely to be
*needed* by Track C (request/grant arbitration depends on a non-consuming probe), so it is
the first candidate for "fix, because a measurement needs it".

### Track B — EVA

**Claim shape:** *A real, published-class accelerator (EVA: FP16 programmable spatial array,
8×8 PEs, dual interconnect) is expressible in Allo and lowers through both backends to
verified hardware; the SystemC path reaches an ASIC flow and matches the original RTL's
function, at a stated cost in area/performance and a stated saving in source and design
effort.*

**B0. Read `Vitis_HLS/EVA_HLS_vs_RTL_Comparison.docx` first**, plus `Updates.pptx`. Do not
re-measure what is already measured; do re-verify anything that pre-dates the current
emitter.

**B1. Fix the comparison framing.** The current Catapult-ASIC-vs-Vitis-FPGA table is not an
apples-to-apples comparison (gates vs LUTs). With the SV `EVA_untouched` as the golden, the
honest version becomes feasible, because **the baseline can be run through both flows**:

```
                         ASIC (nangate-45nm)          FPGA (xczu7ev)
  SV EVA_untouched   →   Design Compiler          →   Vivado synth
  Allo → SystemC     →   Catapult                 →   (n/a)
  Allo → Vitis       →   (n/a)                    →   Vitis HLS + Vivado
```

Compare **within each column** (Allo-SystemC vs SV on ASIC; Allo-Vitis vs SV on FPGA), then
report the two **ratios** side by side. Never compare a µm² to a LUT count. We already have
the ASIC-flow recipe (`[[asic-flow-from-catapult-rtl]]`): use `concat_rtl.v` **not** `rtl.v`
(the latter omits the `ccs_*_wait_v1` handshake cells and only elaborates for degraded
flat-cwd builds), `rtl.v.dc.sdc` for constraints — and note the SDC names `DFF_X1` from
NangateOpenCellLibrary explicitly, and the clock period is **baked into the HLS schedule**,
so targeting a faster clock means re-running csynth, not just relaxing the SDC.
Caveat to state: no memory macros anywhere — even `AlloMem` synthesizes to flops, so any
array-heavy design is gate-inflated on our side. Point Catapult at a memory library *before*
csynth if this distorts EVA's numbers.

**B2. Functional equivalence at scale, using the vectors that already exist.** Build a
vec-driven systemc runner: load `final_runs/cosim_8x8/vec_*.npz`, emit rtprime at that
config, write `prime_cfg=6` + `in/iv/rin` as `input*.data`, csim, diff `out/rout`. Ladder:
**1×1 (mmm) → 4×4 → 8×8 (fft)**. Then the same under `mode="cosim"` for RTL.
Gotchas already paid for: `prime_cfg` is the **first** positional arg (rtprime's shipped
`run_eva` is stale and omits it); NSTEP needs ≥160 margin or everything drains to zero.

**B3. Three-way table.** For each workload (mmm, fft, …) × each size (1×1, 4×4, 8×8):
functional PASS/FAIL, cycles, and cost, for {original SV RTL, Allo→Vitis, Allo→SystemC}.
The original RTL side runs its own VCS testbenches in `rsim/`+`tb/` — check whether VCS is
available on this box or whether they must be ported to Xcelium (RaveNoC needed one
`output logic` patch for exactly this reason).

**B4. Productivity / expressiveness.** Line counts, parameter-sweep cost (EVA at M=N=1 vs
8×8 is a parameter in Allo; in the SV it is a hierarchy), and the list of things Allo made
easy vs the things it forced (e.g. compile-time-constant stream indices → `meta_for` + a
runtime guard per site — a real expressiveness cost that should be reported, not hidden).

**B5. Close out the 8×8 csynth** — check `eva8_csyn.log`; 192 kernel instances may simply
not synthesize in reasonable time. If it doesn't, that is a *finding* (scaling limit of the
flow), and the 4×4 becomes the largest synthesized point.

### Track C — NoCs and routers

**Claim shape:** *Allo's new link abstractions (non-blocking `try_*`, `Channel`, `Wire`)
express a real NoC router family — including virtual channels and wormhole — and generate
RTL that is throughput-equivalent to hand-written SystemVerilog, at a stated cost, while
making architectural variants cheap enough to sweep.*

**C1. Adopt the field's methodology.** Build a **latency-vs-offered-load harness** producing
the canonical curve under **uniform-random, transpose, bit-complement, hotspot** traffic,
reporting **zero-load latency** and **saturation throughput**. Today we have slope +
intercept for a single link; that is a *microbenchmark*, correct but not what the field
compares. Both sides (RaveNoC RTL and Allo RTL) must be driven by the *same* harness under
the *same* simulator (Xcelium), as we already do.
 - Precondition: fix or replace `bench/tb_allo_link.py`. Acceptance test:
   greedy on `hello_channel` must read ~1.0 cyc/flit.
 - Precondition: A4 (does csim cycle-count hold for `try_*`?). If yes, most of the sweep can
   run in csim at seconds/point instead of hours/point.

**C2. Ladder of matched comparisons.** Each rung is RaveNoC-module vs Allo-equivalent, same
config (32-bit flit, depth 2, 2 VC, 2.0 ns), both cosim-verified:
 1. `fifo` ↔ `Stream[int32,2]` ✅ done
 2. `vc_buffer` ↔ `Channel[valid_ready]` ✅ done
 3. `rr_arbiter` ↔ Allo arbiter — **must reproduce the `[16,16,16,16]` distribution**, not
   just the rate
 4. `output_module` (arbiter + VC buffers)
 5. **full `router_ravenoc`** ↔ `router_rvn_chan` / `router_rvn_vc`
 6. mesh (2×2, 4×4)

**C3. The link-primitive study — DONE 2026-08-03, and it is the distinctive result.**
One router, four link types, everything else identical. Full write-up:
`evaluation/trackC_noc/VARIANTS.md`. Outcome:

| link | csim in a router | area (Genus 45nm) |
|---|---|---|
| `Stream[T,2]` | ✅ T1–T4 | 30,602 µm² |
| `Channel[valid_ready]` | ✅ T1–T4 | **11,991 µm² (2.55× smaller)** |
| `Channel[valid_only]` | ❌ **0/12 delivered** | invalid (broken design) |
| `Wire[T]` | ❌ not expressible (`try_*` rejected) | — |

**Only two of the four work.** The rule: *a link must supply backpressure whenever the
consumer time-multiplexes its ports.* `Channel[valid_ready]` is the sweet spot — it removes
all the storage a `Stream` pays for while keeping the one property the architecture needs.

`valid_only` was implemented for this (emitter, `EmitSystemC.cpp`) against the Connections
reference — see [[connections-protocol-reference]]. It is bit-exact in lock-step
(`hello_vonly`) and useless in a router, which is the finding, not a bug.

**Two microbenchmark claims did NOT survive at router scale** — both retracted in
`VARIANTS.md`: the "Channel has a 2.9× shorter critical path" figure (router-scale: within
1.3 %), and the `hello_vonly`-vs-`hello_channel` area comparison (link is a rounding error
against kernel overhead). **Lesson worth stating in the chapter: link-only microbenchmarks do
not predict router-scale behaviour.** Two independent cases now say so.

**C4. Architecture-variant sweep** — the "cheap variants" productivity claim, quantified:
fused vs split (3.9× cycles, 35→10 streams), NVC ∈ {1,2,4}, wormhole on/off (2× cycles,
creates the HoL blocking VCs fix), depth sweep, mesh size. Each point = a parameter change
in Allo; state what the equivalent would cost in the SV.

**C5. Baselines, in two tiers.**

*Tier 1 — RUN here, matched config, same simulator (Xcelium):*
- **RaveNoC** — done for `fifo`/`vc_buffer`/`rr_arbiter`; ladder continues to
  `output_module` and the full `router_ravenoc`.
- **MatchLib `WHVCRouter`** — `$MGC_HOME/shared/pkgs/matchlib/cmod/include/WHVCRouter.h`,
  with `unittests/WHVCRouterTop`, synthesizable, Connections-based. This is *hand-written
  SystemC/Connections* against our *generated* SystemC/Connections: same tool, same library,
  same simulator, so it **isolates the compiler from the library** — the cleanest single
  comparison available to us. Dependency note: MatchLib needs the nvhls stack + boost;
  header-only boost is already installed in the `allo` conda env
  (`conda remove -n allo libboost-headers` to undo), and `/tmp/matchlib_toolkit` was cloned
  (re-clone if `/tmp` was wiped). Include order matters:
  `connections/connections.h` **before** `nvhls_connections_buffered_ports.h`.

*Tier 2 — LITERATURE anchors, cited not re-run:*
- **CONNECT** (FPGA'12) and **OpenSMART** (ISPASS'17). We adopt their metric format
  (latency-vs-injection-rate, zero-load latency, saturation throughput, area/Fmax) and place
  our numbers on the same axes, stating clearly that the technology and flow differ so the
  comparison is *positional, not head-to-head*. This is the standard and honest way to use
  them; re-running their flows is out of scope.

Note on the answer given: "RaveNoC only" was selected alongside the others. Read as *keep
RaveNoC as the primary ladder* — which the tiering above does. If the intent was actually to
cut scope to RaveNoC alone, drop Tier 1's MatchLib rung and demote it to Tier 2; say so and
the plan shrinks by roughly one phase.

**C6. Close the open router work**: re-run `router_faithful.py` after the `try_get` fix;
confirm the `router_rvn_chan` csynth that was still running at pause; resolve the CIN-124 ×4
that still fires in build-subdir runs.

---

## 5. Deliverables

**Created 2026-08-02 at `/home/zsm9/evaluation/`** (same level as `allo_sup`, matching
`final_noc/`, `final_eva_performance/`). See its `README.md`.

```
/home/zsm9/evaluation/
  README.md            # disk rule, layout, rules, what to reconcile with
  papers/<key>.md      # per-paper: what they measure + their actual numbers
  trackA_backend/      # matrices, QoR tables
  trackB_eva/          # three-way tables, vector-cosim results
  trackC_noc/          # load-latency curves, ladder results, variant sweep
  harness/             # measurement scripts
  results/results.csv  # one row per (design, flow, metric, value, commit, date)
```

**⚠ DISK: home has only ~2.2 GB headroom (18,254 M / 20,480 M soft).** This tree holds docs,
CSVs and small logs ONLY. Catapult projects, netlists, `schedule.gnt` (182 MB each), VCDs and
sweep scratch go to `/scratch` (279 G free) or `/work/shared/users/zsh9` and are referenced by
path. A full home quota makes every `df.build` fail with `Disk quota exceeded`.

Rules: every number carries **commit hash + date + config**. Every table has a companion raw
log archived. `final_noc/` gets either `git init` or is mirrored into `allo_sup/agents/noc/`
— an unversioned results tree is not a deliverable.

---

## 5b. PHASE 0 FINDINGS (2026-08-02) — the plan needs revising, much of Track B already exists

**Tooling is better than assumed. Both open decisions 2 and 4 resolve favourably:**
- **VCS is installed** (`/opt/synopsys/vcs/W-2024.09/bin/vcs`) and `EVA_untouched/` already
  contains **built `simv` binaries** (`simv_cc`, `simv_grp`, `simv_indep`, `simv_pipe`) and
  `.vcd` traces. The SV testbenches run natively — no Xcelium port needed.
- ~~Design Compiler is installed, so the ASIC comparison is executable.~~ **WRONG — corrected
  below.** `dc_shell` exists but is **NOT licensed** (`Fatal: Design Compiler is not enabled
  (DCSH-1)`). The ASIC flow goes through **Cadence Genus** instead
  (`/opt/cadence/GENUS201/bin/genus`), which works with
  `LD_LIBRARY_PATH=/opt/cadence/GENUS201/tools.lnx86/lib/64bit` (the `SuSE/SLES11` subdir does
  NOT work) and `unset LD_PRELOAD`. Genus reads Catapult's shipped Liberty
  (`$MGC_HOME/pkgs/siflibs/nangate/NangateOpenCellLibrary_typical_ccs.lib`) directly, so no
  `.db` compile — which is the step that would have needed DC. Flow: `final_noc/synth/genus.tcl`
  + `run_genus.sh`. Use **`concat_rtl.v`**, not `rtl.v`. (`[[genus-gate-level-results]]`)

**GATE-LEVEL NUMBERS ALREADY EXIST — and they overturn a Track C conclusion.** Nangate 45nm,
2.0 ns target:

| design | critical path | Fmax | area µm² | cells |
|---|---|---|---|---|
| **Allo `Channel`** (`vc_buffer_channel`) | 470 ps | **2.13 GHz** | 711.8 | 133 |
| **Allo `Stream`** (`fifo_stream`) | 1364 ps | **0.73 GHz** | 1910.7 | 583 |
| RaveNoC `vc_buffer` | 454 ps | 2.20 GHz | 585.7 | 184 |
| RaveNoC `fifo` | 485 ps | 2.06 GHz | 164.9 | 60 |
| RaveNoC `rr_arbiter` | 154 ps | 6.49 GHz | 33.3 | 15 |
| RaveNoC `router_ravenoc` | Genus SEGFAULT | — | — | — |

**Channel beats Stream 2.9× on critical path and 2.7× on area — while the cycle counts said
they were identical** (both slope 1.000 cyc/flit). Throughput measurement alone completely hid
this. This is the strongest evidence yet for the combinational channel and it retroactively
justifies §2's insistence on pairing cycles with cost. **Allo `Channel` 2.13 GHz vs RaveNoC
`vc_buffer` 2.20 GHz — within 3 % of hand-written RTL.**
Caveat that must travel with the *area* column: the Allo designs include producer and consumer
kernels, the RaveNoC modules are the buffer alone — 712 vs 586 µm² is **not** like-for-like.
The Fmax comparison is sound (the path runs through the link logic either way).
Bonus: Catapult's own period sweep put the path at 0.4395 ns, Genus measured 0.470 ns —
**within 7 %**, so `period_sweep.sh` is a usable no-license proxy.
Open: Genus segfaults on `router_ravenoc`, almost certainly the SV `router_if` interfaces —
try the flattened shim.

**The in-flight csynth+cosim sweep FINISHED** (`/scratch/cosim_work/sweep_results.txt`,
`# DONE Sat 1 Aug 14:22`). 65 result rows, on the committed `mode="cosim"` backend:

| CSYNTH | COSIM | count | what it is |
|---|---|---|---|
| PASS | MATCH | **53** | RTL bit-exact vs the C golden (48 originally + 5 recovered, below) |
| PASS | GOLDEN_FAIL | 1 | `test_mlp` — golden *simulation* failed |
| FAIL | GOLDEN_FAIL | 8 | golden **g++ compile** failed — all stateful/mesh (`region_stateful` ×3, `decoupled_message_passing`, `decoupled_2x1_mesh`, `test_2x1`, `test_2x2`, `multiple_blocks`) = the known `__stateful_*` gap |
| FAIL | – | 3 | `test_systolic` (1D), `test_hierachical_function`, `test_convolution` |

→ **54/65 csynth, 53/65 cosim-bit-exact.** Better than the 22/32 figure in §1, which was the
older per-file count.

**The 5 `ERR` rows were a HARNESS bug and are now FIXED + re-verified (2026-08-03).**
They reported `'numpy.int32' object does not support item assignment` — the read-back sites
in `allo/backend/hls.py` did `out_arg[:] = value`, which is right for an ndarray but throws on
a **scalar** output, since numpy scalars are immutable. (The sweep harness builds scalar args
as `np.int32(1)`, so every scalar-output design hit it.) The designs synthesized and ran
correctly the whole time; only the comparison code was wrong.
Fix: new `store_output()` helper (`hls.py` ~L179) applied at the three unguarded sites
(~L1032, L1048, L1243). It distinguishes three cases — immutable scalar (skip, return False),
0-d array (`arr[...]`, which *is* mutable, unlike `arr[:]`), and normal ndarray (`arr[:]`).
**Re-ran the sweep on the affected files: all 5 flipped `ERR` → `MATCH`**
(`test_scalar_empty_full_sim`, `test_scalar_try_put_try_get_sim`, `test_try_put_try_get_sim`,
`test_empty_full_sim`, `test_large_scale_gemm`). Log: `/scratch/cosim_work/sweep_refix.txt`.
Note this also means the non-blocking `try_*` / `empty`/`full` family is now **RTL-verified
bit-exact**, which matters for Track C — those are the ops every router is built on.

**Track B is much further along than this charter assumed — a parallel line of work,
`/home/zsm9/final_eva_performance/`, has already produced most of it.** `MASTER_RESULTS.md`
there reports, for `eva_sb_syscredit_rtprime` (our Allo EVA) at II=2, 8×8:
- RTL cosim correctness: mmm / fft / cordic×4 all **8/8 rows bit-exact** (streamed, B=300).
- Throughput: mmm 0.167 out/cyc, fft 0.045, cordic ~0.036; first-out latency 537 / ~404–457 / ~386 cyc.
- FPGA QoR (Vivado P&R, xczu7ev): full tile 2.962 ns ≈ **338 MHz**, 4,627 LUT; 8×8 top ≈ 233 MHz,
  with a *stated argument* for why tile-level is definitive (critical path is intra-node, links
  are registered → Fmax scale-invariant).

And for the **original SV EVA under VCS with real DesignWare fp16**:
- per-PE: config load 7 cyc, MOV 5 cyc, MAC 7 cyc, **steady-state 0.302 out/cyc** (CPI 3.31),
  90.6 % pipeline utilisation, back-pressure degrade 1.28×.
- 8×8 mesh mmm: program load 50 cyc, per-column first output 42–77 cyc (~6 cyc/col).
- streaming per-lane: mmm **0.250**, fft **0.154** out/cyc.
- FPGA QoR **N/A by construction** — DesignWare cells are not Vivado-synthesizable.

**Consequence for Track B:** the Allo→Vitis column and the SV-RTL column largely exist. The
**gap is the Allo→SystemC/Catapult column**, and specifically the ASIC flow — which is also
the one place the original EVA's QoR can be compared at all (it is ASIC-native). That makes
§4.B1's ASIC comparison not a nice-to-have but *the* missing piece of Track B.
**Action: reconcile with that work before measuring anything** — do not duplicate it.

**The LOC/productivity analysis also already exists**, in
`Vitis_HLS/EVA_HLS_vs_RTL_Comparison.docx` (2026-06-23) — but note it compares a
**hand-written Vitis C++** rebuild, not Allo:
- Tier 1 (PE core + router, tightest apples-to-apples): **598 vs 2,178 lines = 3.6×**
- Tier 2 (+ drivers/collectors + 8×8 array): **1,605 vs 4,822 = 3.0×**
- Explicitly excludes ~8,070 RTL lines with no rebuilt counterpart (extended ISA, config-reg
  plane, memory subsystem, SoC hierarchy) — a methodology we should copy verbatim.
- Datatype-swap footprint: ~13 edits same-width / ~16–18 different-width **before** a
  one-knob policy header, 0 after; vs. RTL where the FPU is hard-bound to DesignWare fp16 and
  a type change means regenerating the arithmetic IP. Flagged there as "the single strongest
  HLS-advantage demo" — and it applies to Allo at least as well.
- Its §8 ("what is usually compared to show HLS advantage over RTL") independently matches
  the metric list in §2 above, which is a good sign for both.
→ **Reuse the RTL line counts and the exclusion methodology; re-measure the Allo side.**

**EVA vector-driven csim already ran** (`/work/shared/users/zsm9/eva_8x8_run/vec_csim/SUMMARY.txt`,
2026-08-02) against the `final_runs/cosim_8x8` vectors — i.e. §4.B2 is partly done:
- mmm (1×2) **PASS** bit-exact, mmm2 (2×2) **PASS**, mmm4 (4×4) **PASS** (margin 1000)
- fft (8×8) **FAIL** — `out_e` only, other 7 arrays match
- mmm_x2 (8×8, ×2 reps) **FAIL** — `out_s` only, **token duplication (32 vs 24), not a margin
  issue**; multi-rep collector at 8×8. Open.
→ So the SystemC path is bit-exact up to 4×4 and has **two specific, localized 8×8 failures**.
That is a far better position than "unverified", and both are concrete debugging targets.

**EVA 8×8 Catapult csynth: RESOLVED — it was OUR OWN TIMEOUT, not a tool or scaling failure.**
Started 09:49:41, log last written 15:37:14 = **20,880 s**, against
`ALLO_COSIM_SYNTH_TIMEOUT=21600` (6 h) — 720 s short, consistent with the last buffered log
flush before an external kill. Evidence it was *not* a failure:
- **0 errors** in the whole 26 MB log; no abort/terminate/signal message anywhere.
- It got **well past scheduling**: `schedule.gnt` (182 MB) written 12:32, `cycle.rpt` 13:38,
  `reg_sharing.tcl` 13:53, and it was running `dpfsm` datapath-FSM transformation at the end.
- Not OOM: peak 9.3 GB on a 376 GB box (320 GB free now), nothing in `dmesg`.
→ **8×8 EVA synthesis is time-bound, not blocked.** Re-run with a much larger
`ALLO_COSIM_SYNTH_TIMEOUT` (≥ 12 h) to get the top-level netlist. It also incidentally proves
the CIN-71 fp16 fix holds at 8×8 scale.

**`/home/zsm9/final_eva_systemc/` — the SystemC EVA track — has a STALE blocker.** Its README
says csynth is "❌ blocked — CIN-71 (fp16 bitcast, `memcpy` on `half*`)" with a proposed fix of
"use `set_data` / the IEEE bit API instead of memcpy". **That fix is already committed on
`wire` as `802c9eb`**, and the 8×8 run above proves it works at scale. So that track's stated
blocker no longer exists — it needs re-running against the current emitter, not fixing.
Also note its scheduling caveat, which matters for any Allo-vs-Allo comparison: the Vitis
"II=2" came from `inject_pragmas_ii2.py`, a **VHLS-text hack** that injects `#pragma HLS
pipeline` because the real Allo schedule was intractable to *emit* at 8×8. That hack does not
apply to SystemC, where **Catapult is the scheduler** and II is a `run.tcl` directive. So the
two flows are not scheduled the same way, and any II/throughput comparison must say so.

**Other jobs are running on this box** (a Catapult sweep at
`/scratch/cosim_work/sweep/sw_q_5e1cli`, ~26 h; a VCS `+REPS=512` cordic run under
`final_eva_performance/logs/streaming/`). Check before launching anything heavy — CPU
contention has corrupted a sweep here before.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Numbers in memory come from different emitter states | A1 re-runs everything on one HEAD before anything is written up |
| `csim cycles == RTL cycles` fails for `try_*` → the whole cheap-measurement strategy collapses | A4 tests it **first**; fallback is csynth-per-point (hours) and a reduced sweep |
| The backpressure bench is broken and its numbers are unusable | Acceptance test on `hello_channel` before any reuse |
| 8×8 EVA csynth may never finish | Report 4×4 as the largest synthesized point; the scaling limit is itself a finding |
| RaveNoC full-router tb is slow (25 min and still going at a 1500 s timeout) | Shrink the config, or drive submodules directly (author-sanctioned) |
| Apples-to-oranges area comparison (gates vs LUTs) | B1 decides the framing **before** measuring |
| Flat-cwd Catapult artifacts silently have no backpressure logic | Regenerate anything archived from a flat build; check for `ccs_in_wait_v1` in the netlist |

---

## 7. Sequencing

**Phase 0 (now):** paper survey (§3) + `EVA_HLS_vs_RTL_Comparison.docx` + A4 (the `try_*`
cycle-model gate) + the `tb_allo_link` acceptance test. These three decide the shape of
everything after.

**Phase 1:** A1 + A2 — one clean, current, complete backend matrix. This is the spine; both
other tracks cite it.

**Phase 2 (parallel):** B2/B3 (EVA vector-driven ladder) and C1/C2 (load-latency harness +
comparison ladder). Independent, both long-running.

**Phase 3:** C3/C4 (link-primitive study + variant sweep) and A3 (QoR) — the distinctive
results, once the infrastructure from Phase 2 exists.

**Phase 4:** C5 (MatchLib WHVCRouter baseline), B4 (productivity), write-up.

---

## 8. Decisions

**Resolved 2026-08-02** (see the block at the top of this file): old Allo = `vhls`;
EVA baseline = SV `EVA_untouched`; NoC baselines tiered (RaveNoC + MatchLib run, CONNECT +
OpenSMART cited); output = thesis chapter; area framing = within-flow comparison plus
ratio-of-ratios (§4.B1).

**Still open:**

1. **Is there a published EVA number** we must match, or are all reference figures ours to
   make (as with RaveNoC)? Phase 0 answers this by reading
   `Vitis_HLS/EVA_HLS_vs_RTL_Comparison.docx` + `Updates.pptx`.
2. **Is VCS available on this box** for the SV EVA testbenches in `rsim/`+`tb/`, or must they
   be ported to Xcelium? (RaveNoC needed one `output logic` patch for exactly this reason —
   Verilator accepts a net assigned in `always_comb`, Xcelium does not.) Affects B3's cost
   substantially.
3. **Does the `-C5` MatchLib rung stay in Tier 1** — see the note at the end of §4.C5.
4. **Memory macros for the ASIC flow** (§4.B1): accept flop-based arrays and state the
   distortion, or point Catapult at a memory library before csynth. Only matters if EVA's
   register files dominate the gate count.
