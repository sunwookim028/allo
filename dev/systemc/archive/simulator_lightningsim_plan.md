# Integrating LightningSim with Allo — status and plan

Written 2026-07-30. Companion to `simulator_cycle_model.md` (why we want this).

---

## 0. It is already installed, and someone already tried

- **Tool:** conda env `lightningsim`, package `lightningsim 0.2.6` from
  `sharc-lab.github.io/LightningSim/repo`. CLI works.
- **Working dir:** `/home/zsm9/lightningsim/` — *not* a source clone, a scratch area with
  two experiments from 2026-07-22:
  - `branch_test/` — hand-written HLS kernel using `read_nb` in a DATAFLOW region
    (a Type C probe). csynth succeeded; `.autopilot/db` was not preserved.
  - `mmmr_test/` — an **Allo-EVA generated 2×2 array** (`allo_prj_2x2_L336_I8_D1`) plus
    two hand-written native testbenches (`tb_replay.cpp`, `tb_synth.cpp`) and a full
    559-file solution db.
- No LightningSim result artefacts exist, i.e. **no run ever completed.**

## 1. Interface — this is a whole-tool integration, not a library call

```
lightningsim [--cli|--gui] <vitis_hls_solution_dir>
```

It consumes a **Vitis HLS solution directory**, links an instrumented build of the
project's **csim testbench**, runs it to produce a trace over a pipe
(`HLSLITESIM_TRACE_FD`), joins the trace to the static schedule, and reports cycles.

Consequences:
- The pipeline is **Allo → HLS C++ → `vitis_hls csynth` → solution dir → LightningSim**.
  It sits *beside* our simulator, not inside it.
- **Vitis-only.** See the Catapult-vs-Vitis risk in `simulator_cycle_model.md` §6.1 — if
  designs ship through Catapult, this measures a different scheduler.

## 2. What actually breaks today (both reproduced 2026-07-30)

### 2a. Allo-EVA 2×2 mesh → testbench SEGFAULTS

```
ValueError: unknown trace entry type 'fifo_wr'
```

Misleading error. The raw trace (captured by wrapping `read_trace`) is **4866 good
lines** ending in a truncated entry with no newline:

```
fifo_write\t0x7fff298c534c
fifo_write\t0x7fff298c532c
fifo_wr        <-- cut off mid-write
```

Testbench `returncode = -11` = **SIGSEGV**. The truncated line is a symptom, not the bug.

**Cause — NOT the non-blocking limitation.** That design has **0** `read_nb`/`write_nb`
and 200 blocking `.read()`/`.write()`. It is a 2×2 **mesh**, i.e. *cyclic*. LightningSim's
Type A is blocking **and acyclic**: a sequential csim can satisfy every read only if the
dataflow is feed-forward. A cyclic design reads an empty stream, which is why the
hand-written testbench needed `-DALLOW_EMPTY_HLS_STREAM_READS` — and under the
instrumented stream implementation that read segfaults.

This is precisely why OmniSim was written: it classifies Allo as Type A only.

### 2b. Blocking producer/consumer → wrong testbench

```
fatal error: CL/cl2.hpp: No such file or directory   (compiling host.cpp)
```

Allo emits `host.cpp`, an **OpenCL host program**, and LightningSim picks it up as the
csim testbench. It needs a *native* testbench that calls the kernel directly — which is
exactly why `tb_synth.cpp` was hand-written in `mmmr_test`.

## 3. The plan

### Stage 0 — DONE 2026-07-30. PASS.

**LightningSim `top` = 21 cycles = csynth `top` = 21, exact.** Sources, reproduction
steps and full notes: `simulator_profiling/lightningsim_stage0/`.

```
[ 0-20] top                     <-- 21 cycles inclusive
        [ 0- 0] entry_proc
        [ 0- 4] producer_0
        [ 1- 5] consumer_0
        [ 6-20] store_res0.1    <-- 15 cycles our simulator charges ZERO
```

Three Allo-side blockers had to be cleared, all fixable in the emitter (Stage 2):

1. `host.cpp` is an OpenCL host → wrote a native `tb.cpp` calling `top()` directly.
2. Generated `kernel.h` uses `int32_t` **without including `<cstdint>`** — `kernel.cpp`
   only survives because other headers precede it. Any native testbench including
   `kernel.h` first fails to compile.
3. `#pragma HLS pipeline II=1 rewind` → link error `undefined reference to
   _ssdm_op_Return`; LightningSim's runtime lacks that intrinsic. Dropping `rewind` fixes
   it, **but perturbs the design** (csynth `top` was 19–20 with it, 21 without). The
   comparison above is self-consistent; a proper fix stubs the intrinsic instead.

**Bonus result — the 0.4× makespan gap is now measured, not guessed.** Our simulator says
makespan 7; both oracles say 21. The breakdown localises all of it: `store_res0.1` spans
cycles 6–20 and our cost model charges the `load_buf`/`store_res` wrappers **zero**
(7 + ~14 ≈ 21). That confirms the §1 hypothesis in `simulator_cycle_model.md`.

### Stage 0 (original spec) — prove the path end-to-end on a Type A design

Nothing below is worth doing until one Allo design produces a cycle count.

- Take the existing **blocking producer/consumer** (`blocking_stream_csynth.prj`) —
  acyclic, blocking, already synthesises, and already has Vitis ground truth
  (`producer_0=6, consumer_0=6, top=19..20`) to check against.
- Write a **native csim testbench** (~30 lines, model on `mmmr_test/tb_synth.cpp`) that
  calls `top()` directly with no OpenCL.
- Re-synthesise with that testbench as `add_files -tb`, then run `lightningsim --cli`.
- **Exit criterion:** a cycle number that matches the csynth report. If this fails, stop
  and reassess — everything downstream assumes it works.

### Stage 1 — DONE 2026-07-30. PASS.

**LightningSim `node_0_0` = 1014 cycles = csynth `node_0_0` = 1014, exact**, on a PE body
extracted from the cyclic EVA mesh. Details: `simulator_profiling/lightningsim_stage1/`.

The per-kernel oracle idea is **validated**: the whole mesh segfaults (cyclic), but a PE
cut out of it traces exactly. Three harness requirements and one upstream patch:

- **No `hls::stream` in the top signature** — top-level stream ports make Vitis emit
  `streamcpy_hls` glue calling `fpga_fifo_*_4`, which LightningSim does not provide.
  Wrap with `pe_feed`/`pe_drain` in a dataflow `top`, streams internal.
- **Drain counts must match production exactly** or `builder.finish()` raises
  `incomplete edges remain`. This PE writes each output once *before* the loop and once
  per iteration → 337, not 336.
- **LightningSim 0.2.6 needs a one-line patch**: `trace_file.py:~418` asserts a FIFO
  write's payload has an instruction source; for 5 FIFOs here it is `None`. Defaulting
  the width lets it complete, but the default is a guess on `ap_uint<26>/<17>` streams —
  the proper fix reads the declared width, which means building from source.

**Finding worth the whole exercise:** the PE's main loop runs at **II ≈ 3** (1012 cycles
/ 336 iterations) despite `#pragma HLS pipeline II=1`. Our cost model assumes
`DEFAULT_II = 1`, measured on a trivial loop. A 3× miss on the dominant loop of a real
PE — exactly what `simulator_cycle_model.md` proposes to fix by ingestion.

### Stage 1 (original spec) — decide the scope honestly

LightningSim is **Type A only**. Our target designs (meshes, non-blocking, wires) are
Type B/C. So it can never be the general answer. Two viable roles:

- **(a) Per-kernel oracle — recommended.** Run LightningSim on *individual PE kernel
  bodies*, which are almost always acyclic and blocking even when the full design is a
  cyclic mesh. This sidesteps the Type A limit entirely **and produces exactly the
  per-kernel numbers `simulator_cycle_model.md` §3 wants to ingest.** The two threads
  converge here.
- **(b) Whole-design oracle — limited.** Only for acyclic blocking designs. Useful as a
  high-confidence calibration point but covers few of our targets.

Do **not** plan on (b) for meshes; that is what the segfault is telling us.

### Path C — DONE 2026-07-30. WORKS, and it handles cycles.

Import `lightningsim._core` (the compiled Rust engine) and build the simulation graph
ourselves — no Vitis bitcode, no instrumentation, no testbench. Details and runnable
probes: `simulator_profiling/lightningsim_pathc/`.

Proven: the solver accepts hand-built graphs; it models **back-pressure** (depth 1 → 37
cycles, depth 8 → 25, converging to unbounded); its **native FIFO-depth DSE** returns
latency *and* BRAM per point; and — decisively — **it resolves a primed cycle and
correctly flags an unprimed one as deadlock.**

**That last result changes the picture.** LightningSim cannot *run* a cyclic design (its
functional sim needs a completed sequential pass; Stage 1 = segfault), but the **engine**
has no such limit. The Type A restriction lives in the front end, not the solver. So Path
C reaches designs that invoking the tool never can.

Caveats: event order must be causally consistent; a malformed graph fails at `finish()`
with `incomplete edges remain`; `stage` numbers still come from the schedule (Path B's
work regardless); `_core.pyi` is a private, unstable API on a compiled binary; and these
were synthetic graphs of a few nodes, so nothing here proves it scales to a real mesh.

### Stage 2 — automate testbench emission

The blocker in 2b is that Allo emits an OpenCL host, not a csim testbench.

- Add a native-testbench emitter (a `tb.cpp` alongside `kernel.cpp`) that drives the top
  function with deterministic inputs and checksums outputs. `mmmr_test/tb_synth.cpp` is a
  working template.
- Emit it from the same place that emits `host.cpp`, gated by a flag so nothing existing
  changes.

### Stage 3 — wire it as a measurement path

- A script `simulator_profiling/lightningsim_oracle.py`: Allo design → emit HLS C++ + tb
  → `vitis_hls csynth` → `lightningsim --cli` → parse cycles → row in the ground-truth
  table from `simulator_cycle_model.md` §3.5.
- This is a **third oracle** next to csynth (static, partial) and Catapult/Xcelium RTL
  co-sim (exact, slow): trace-based, exact for Type A, ~95× faster than co-sim.

## 4. Risks, in order

1. **Vitis-only** — if we ship through Catapult, this calibrates the wrong scheduler.
   Resolve before Stage 2. (`simulator_cycle_model.md` §6.1.)
2. **Type A ceiling is structural**, not a bug to fix. Per-kernel use (Stage 1a) is the
   way around it; whole-mesh use is not.
3. **0.2.6 is a packaged binary**, not a source checkout — patching the trace parser or
   the stream model means building from source. Avoid needing to.
4. Cyclic *per-kernel* bodies (a PE with a feedback loop) would hit the same wall even in
   Stage 1a. Unknown how common; measurable once Stage 0 works.

## 5. Immediate next step

Stage 0 only: hand-write one native testbench for the producer/consumer design and get a
single number out. Everything else is contingent on that.
