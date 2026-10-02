# U1 `vpu_bf16_add` through SystemC -> Catapult, against MiniTPU's RTL (Catapult track)

Host zhang-21 (RHEL 8.10, 64 cores, shared: load ~150 during the DC runs).
Run 2026-10-02 on branch `u1-catapult` (worktree from `origin/u1-pilot` at
`4da88a79`; `mlir/build` symlinked to the main checkout's build). MiniTPU at
`b3ba0a4d`, `src/core/vpu/vpu_bf16_add.sv` (sha256 `c35440ad...`, read-only).
README D-7 / D-9, Catapult-track row.

## Summary

- **Synthesis succeeds with one hand-patch.** The harness's unit as written
  (`native`: one `df.kernel` looping `c[i] = a[i] + b[i]` over `bfloat16[n]`)
  stops in `go analyze` with `CRD-135` (Connections has no `Wrapped<ac::bfloat16>`
  because the emitted `kernel.cpp` includes `ac_std_float.h` *after*
  `mc_connections.h`). Moving that one `#include` up (P1) gets every variant
  through `go extract` in 40-47 s. The known `sc_trace` bug (1) did **not** fire
  in Catapult's front end (it breaks g++ csim only; F1), and the tb bugs (2), (3)
  are csim-only and not on this path.
- **Numerics: bit-exact except two named cases, on every variant.** Over the
  default stimulus (251,936 vectors), Catapult's RTL equals MiniTPU's RTL on
  249,907 and equals IEEE RNE (`ref.ieee_bf16_add`) on 249,908. The 2,029
  differences are all explained: 2,028 NaN results (Catapult emits all-ones
  payload `0x7fff`/`0xffff` keeping a sign; MiniTPU always `+0x7fc0`), and
  1 `(+0)+(-0)` (Catapult `+0`, IEEE; MiniTPU `-0`). Rounding (ties included)
  and subnormals match exactly: the emitter's `AC_RND_CONV` override is right.
- **Cycles: the unit as written is not MiniTPU's shape, and was not pipelined.**
  `native`'s loop lands in the SC_THREAD's *reset action*: 3 cycles/vector,
  measured, while `cycle.rpt` reports Latency `-1`, Throughput `1`. With
  `s.pipeline("add_0:i")` it runs at **1.000 cycle/vector, latency 2** (2.0 ns).
  A closer-shaped expression (free-running `add` kernel between `Stream`s,
  `synth_top=add_0`) needs a Catapult directive Allo cannot express
  (`PIPELINE_STALL_MODE flush`), or the last output never leaves the pipeline.
  A `Wire`-port unit needs a second hand-patch (P3, `CIN-123`) and then gives the
  closest shape: at 3.33 ns, inputs -> adder -> one output register.
- **Same-flow PPA (DC W-2024.09, FreePDK45 `stdcells.db` md5 `f5560259...`).**
  MiniTPU's combinational adder needs **2.38 ns** (misses 2.0 ns by 0.38 ns,
  900 um2); Catapult pipelines its own into 2 stages and closes 2.0 ns.
  At 3.33 ns, the like-for-like pair (MiniTPU adder + output register vs
  Catapult's Wire-port RTL, same ports, same register count) is **883.9 vs
  813.4 um2**, both meeting timing.

## Environment

```bash
cd /work/shared/users/phd/sk3463/scratch/wt-u1-cat
source examples/minitpu/harness/env-zhang21.sh     # allo env, catapult-2024, Xcelium, Verilator; LLVM_BUILD_DIR from the env
export MINITPU_HARNESS_CACHE=/work/shared/users/phd/sk3463/scratch/u1_cat/hcache   # keep Verilator builds out of the MiniTPU clone
```

| tool | version |
| --- | --- |
| Catapult | Catapult Ultra Synthesis 2024.2/1130128 (`MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home`), library `nangate-45nm_beh` |
| Verilator | 5.052 2026-09-05 rev conda-forge build (C++ via the allo env's x86_64-conda-linux-gnu-c++) |
| Design Compiler | W-2024.09 for linux64 - Aug 27, 2024 (`module load synopsys-dc-W-2024.09`, `SNPSLMD_LICENSE_FILE=27020@en-license-05.coecis.cornell.edu`) |
| Python | `$ALLO_PYTHON` (allo conda env) |

## What was built

Scripts are in `scripts/` (run from the worktree root; scratch at
`/work/shared/users/phd/sk3463/scratch/u1_cat/`). Every Catapult run used the
emitted `run.tcl` unchanged except where a hand-patch says so (`catapult/<v>/run.tcl`
is the tcl as run; `catapult/<v>/hand_patches.diff` is emitted -> as-run `kernel.cpp`).

| variant | Allo source | how emitted |
| --- | --- | --- |
| `native_n4`, `native_n251936` | `units/bf16_add.native(n)`, unchanged | `scripts/emit_csyn.py <n> <prj>` |
| `native_ii1_n16`, `native_ii1_n251936` | same + `s.pipeline("add_0:i")` | `scripts/emit_csyn.py <n> <prj> --pipeline` |
| `stream_n16` | `scripts/variants.py:stream`: `src` -> `Stream[bfloat16,2]` x2 -> free-running `add` (`for _ in range(n): sc.put(sa.get() + sb.get())`) -> `Stream` -> `sink`; `synth_top=add_0` | `scripts/emit_var.py stream 16 <prj>` |
| `stream_ii1_n16` | same + `s.pipeline("add_0:_")` | `... --pipeline` |
| `stream_ii1_flush_n16` (`_3p33`) | same + P4 (flush) | `... --pipeline [--clock=3.33]`, then P4 in `run.tcl` |
| `wire_n16`, `wire_ii1_n16` (`_3p33`) | `variants.py:wire`: as `stream` with `Wire[bfloat16]` links, so `add_0` has plain `sc_in`/`sc_out` ports | `scripts/emit_var.py wire 16 <prj> [--pipeline] [--clock=3.33]`, then P3 |

`variants.py:channel` (`Channel[bfloat16, valid_ready]`) emits an `add_0` identical
to `stream`'s (the link type only changes the region top), so it was not synthesized
separately. Catapult was run as `scripts/run_csyn.sh <prj>` (applies P1 [+P3 with
`PATCH_ARGS=--wire-wait`], then `catapult -shell -f ../run.tcl` in `<prj>/build`),
the same command Allo's `mod()` runs.

### Hand-patches (workarounds in generated files, not fixes)

- **P1** (all variants): `#include <ac_std_float.h>` moved above `#include <mc_connections.h>`.
- **P3** (Wire variants): `wait();` in the steady-state `while (1)` also under `__SYNTHESIS__`.
- **P4** (`*_flush*`): `directive set -PIPELINE_STALL_MODE flush` in `run.tcl`.

## Catapult results (clock from `run.tcl`; reports in `catapult/<v>/`)

| variant | clock | cycle.rpt latency / throughput | loop | area (TOTAL AREA after assignment) | slack (ns) | wall |
| --- | --- | --- | --- | --- | --- | --- |
| `native_n4` | 2.0 | -1 / 1 (reset 13) | `l_S_i_0_i` 3 c-steps, **reset action**, not pipelined | 1528.5 | 0.0175 | 47 s |
| `native_n251936` | 2.0 | -1 / 1 (reset 755,809) | same | 1674.4 | 0.0175 | 40 s |
| `native_ii1_n16` | 2.0 | -1 / 1 (reset 19) | II=1, "no flushing", reset action | 1542.3 | 0.0103 | 41 s |
| `native_ii1_n251936` | 2.0 | -1 / 1 (reset 251,939) | same | 1690.5 | 0.0103 | 41 s |
| `stream_n16` | 2.0 | 2 / 3 | `while` 3 c-steps | 1489.7 | 0.0175 | 45 s |
| `stream_ii1_n16` | 2.0 | 2 / 1 | II=1, **no flushing** | 1439.9 | 0.0045 | 44 s |
| `stream_ii1_flush_n16` | 2.0 | 2 / 1 | II=1, flushing (P4) | 1539.2 | 0.0057 | 42 s |
| `wire_n16` | 2.0 | 3 / 3 | `while` 3 c-steps (P3) | 1099.5 | 0.0011 | 44 s |
| `wire_ii1_n16` | 2.0 | 2 / 1 | II=1, no flushing (P3) | 1494.0 | **-0.0142** | 42 s |
| `stream_ii1_flush_n16_3p33` | 3.33 | 1 / 1 | II=1, flushing | 1625.8 | 0.0052 | 42 s |
| `wire_ii1_n16_3p33` | 3.33 | 1 / 1 | II=1, no flushing | 1863.1 | 0.0033 | 41 s |

Catapult area is in its score units. It does **not** rank these designs the way
DC does (`wire_ii1_n16_3p33` is Catapult's largest and DC's smallest, below).

## Catapult RTL vs MiniTPU RTL in Verilator

`harness/rtl.py` gained a `stream` shape (independent valid/ready per input
port, output ready held high or dropped one cycle in `out_ready_period`, accept
and emit cycles stamped) and `reset`/`warmup` on `bare`;
`units/bf16_add.catapult_rtl(v1_dir, ...)` is the `RtlUnit` for Catapult's
`add_0`/`top` (`concat_sim_rtl.v`, ports `v6`, `v7` -> `v8`, `rst` active low).
Driver: `scripts/cmp_rtl.py <v1_dir> [--top top] [--shape bare --latency L --warmup W] [--ready-period P]`.
Logs in `verilator/` (each ~4-12 s for 251,936 vectors, Verilator build included).

| RTL | shape | measured | vs MiniTPU RTL | vs IEEE RNE |
| --- | --- | --- | --- | --- |
| `native_n251936` | stream | 755,809 cycles = **3.000 cyc/vector**, latency 2 | 249,907 / 251,936 | 249,908 |
| `native_ii1_n251936` | stream | 251,939 cycles = **1.000**, latency 2, first output cycle 3 | 249,907 | 249,908 |
| `stream_n16` | stream | 3.000, latency 2 | 249,907 | 249,908 |
| `stream_ii1_n16` (no flush) | stream | **stalled**: 251,935 of 251,936 outputs (below) | -- | -- |
| `stream_ii1_flush_n16` | stream | 1.000, latency 2 | 249,907 | 249,908 |
| same, output ready low 1 cycle in 3 | stream | 1.500 (= the sink's rate), latency 2-4, nothing lost | 249,907 | 249,908 |
| `wire_ii1_n16` | bare, reset, warmup 3 | result 2 edges after the input is sampled (`latency=1` in `bare` terms) | 249,907 | 249,908 |
| `stream_ii1_flush_n16_3p33` | stream | 1.000, latency 1 | 249,907 | 249,908 |
| `wire_ii1_n16_3p33` | bare, reset, warmup 1 | one output register (`latency=0` in `bare` terms) | 249,907 | 249,908 |

The 2,029 differences against MiniTPU, identical on every row:

```
   2028  NaN payload: ac_std_float all-ones 0x7fff/0xffff, rtl +0x7fc0  e.g. 0000+7fc0: catapult 7fff minitpu 7fc0, 0000+7f81: catapult 7fff minitpu 7fc0, 0000+7fff: catapult 7fff minitpu 7fc0
      1  (+0)+(-0): allo +0 (IEEE), rtl -0  e.g. 0000+8000: catapult 0000 minitpu 8000
```

Against the Allo simulator (`verilator/cmp_sim_vs_catapult.txt`): 249,908 equal;
all 2,028 differences are NaN-vs-NaN. The simulator gives `7fc0`/`ffc0`
(1,048 / 980), Catapult `7fff`/`ffff` (1,052 / 976), and the sign differs on 12
invalid operations, e.g. `7f80+ff80` (inf + -inf): Catapult `7fff`, simulator `ffc0`.

The `bare` latency/warmup of the Wire RTLs were found by search on 3,000 non-special
vectors (the only combination that matched) -- a measurement of a port shape with no
handshake, not a declared contract.

## Same-flow PPA: Design Compiler

One script for every RTL (`scripts/dc/dc_u1.tcl`, driven by `scripts/dc/run_dc.sh <name> <top> <clk|none> <period> <src>...`),
modelled on mflowgen's `synopsys-dc-synthesis` node as used for TinyTPU
(`examples/tinytpu/asic_synthesis/`), run directly rather than through mflowgen:

| setting | value |
| --- | --- |
| target library | `stdcells.db`, FreePDK45 `view-standard`, md5 **`f5560259ca91a4b67336b715b729f94d`** (the md5 every committed TinyTPU run records), from `/scratch/users/sk3463/build_T4_MAXDIM16_shipped_baseline/1-freepdk-45nm/view-standard/` (`adk.tcl` md5 `5e884494...`, `adk-base.tcl` md5 `67261d8b...`) |
| synthetic library | `dw_foundation.sldb` (md5 `d955c61b84f986958ce94ccd738f45e7`) |
| compile | `ungroup -start_level 2 -all -flatten`; `compile_ultra -gate_clock`; `set_dont_use */SDFF*` (ADK overlay) |
| mode | **non-topographical** (differs from the TinyTPU runs, which were topographical) |
| constraints | `create_clock -period P` on `clk` (virtual clock for the combinational unit); input and output delay **0** (TinyTPU's: 0.5P in, 0 out); driving cell `INV_X2`; `ADK_TYPICAL_ON_CHIP_LOAD` on outputs; max fanout 20; max transition 0.25P |
| cores | asked for 8; DC used 1 (`UIO-231`, host load 150) |
| sources | MiniTPU: `vpu_bf16_add.sv` (+ `scripts/dc/mtpu_add_reg.sv` / `mtpu_add_oreg.sv` register wrappers); Catapult: `concat_rtl.v` (sha256 in `SHA256SUMS.txt`) |

Reports in `dc/<run>/` (area, qor, timing, clock gating, reference; `wall.txt`).
Areas are um2 of FreePDK45 cells, pre-layout, no wires.

| run | clock (ns) | design | total area | comb. | non-comb. | seq. cells | crit. path (ns) | slack (ns) | DC wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `minitpu_comb_2p0` | 2.0 (virtual) | MiniTPU adder alone | 899.9 | 899.9 | 0 | 0 | 2.38 | **-0.38** | 111 s |
| `minitpu_oreg_2p0` | 2.0 | MiniTPU adder + output reg | 980.7 | 876.7 | 104.0 | 16 | 2.39 | **-0.42** | 79 s |
| `minitpu_reg_2p0` | 2.0 | MiniTPU adder, in + out regs | 1079.4 | 857.1 | 222.4 | 48 | 2.18 | **-0.22** | 95 s |
| `cat_stream_ii1_flush_2p0` | 2.0 | Catapult, Stream ports, II=1, 2 stages | 1201.8 | 765.0 | 436.8 | 83 | 1.96 | 0.00 | 73 s |
| `cat_stream_ii3_2p0` | 2.0 | Catapult, Stream ports, II=3 | 1289.3 | 853.1 | 436.2 | 83 | 1.96 | 0.00 | 68 s |
| `cat_wire_ii1_2p0` | 2.0 | Catapult, Wire ports, II=1, 2 stages | 1050.2 | 772.2 | 278.0 | 53 | 1.96 | 0.00 | 69 s |
| `minitpu_comb_3p33` | 3.33 (virtual) | MiniTPU adder alone | 811.6 | 811.6 | 0 | 0 | 3.32 | 0.01 | 72 s |
| `minitpu_oreg_3p33` | 3.33 | **MiniTPU adder + output reg** | **883.9** | 787.4 | 96.6 | 16 | 3.20 | 0.09 | 72 s |
| `minitpu_reg_3p33` | 3.33 | MiniTPU adder, in + out regs | 1038.7 | 793.7 | 245.0 | 48 | 3.26 | 0.03 | 71 s |
| `cat_wire_ii1_3p33` | 3.33 | **Catapult, Wire ports, 1 stage** | **813.4** | 717.7 | 95.8 | 18 | 3.28 | 0.01 | 69 s |
| `cat_stream_ii1_flush_3p33` | 3.33 | Catapult, Stream ports, II=1, 1 stage | 1092.5 | 793.2 | 299.3 | 57 | 3.29 | 0.00 | 72 s |

The apples-to-apples pair is the bold one: same ports (`a`, `b` in, 16-bit
registered result out), same clock, same flow. Catapult's has 2 more flops
(reset/`done`). Its datapath is **9% smaller** (717.7 vs 787.4 um2 comb.); the
two adders differ in algorithm (`ac_std_float::add_generic` vs MiniTPU's
17-bit align/normalize), and not in what they compute except NaN and `+0 + -0`.
At 2.0 ns MiniTPU's adder does not fit in one cycle (2.38 ns); Catapult met
2.0 ns by inserting a second stage, which a combinational MiniTPU unit cannot
do. The handshake costs ~280 um2 (Stream vs Wire at 3.33 ns).

## Errors, verbatim

**E1. `native`, unpatched, `go analyze`** (`catapult/native_n4/csyn_unpatched.log.gz`):

```
# Error: $MGC_HOME/shared/include/connections/marshaller.h(208): class "ac::bfloat16" has no member "Marshall" (CRD-135)
# Error: $MGC_HOME/shared/include/connections/marshaller.h(208):           detected during: (CRD-135)
# Error: $MGC_HOME/shared/include/connections/marshaller.h(208):             instantiation of "void Wrapped<T>::Marshall(Marshaller<Size> &) [with T=ac::bfloat16, Size=16U]" at line 2506 of "/opt/siemens/catapult/2024.2/Mgc_home/shared/include/connections/connections.h" (CRD-135)
# Error: $MGC_HOME/shared/include/connections/marshaller.h(208):             instantiation of "void Connections::InBlocking<Message, Connections::SYN_PORT>::read_msg(Message &) [with Message=ac::bfloat16]" at line 2349 of "/opt/siemens/catapult/2024.2/Mgc_home/shared/include/connections/connections.h" (CRD-135)
# Error: $MGC_HOME/shared/include/connections/marshaller.h(208):             instantiation of "Connections::InBlocking<Message, Connections::SYN_PORT>::InBlocking(const char *) [with Message=ac::bfloat16]" at line 2946 of "/opt/siemens/catapult/2024.2/Mgc_home/shared/include/connections/connections.h" (CRD-135)
# Error: $MGC_HOME/shared/include/connections/marshaller.h(208):             instantiation of "Connections::In<Message, Connections::SYN_PORT>::In(const char *) [with Message=ac::bfloat16]" at line 382 of "/work/shared/users/phd/sk3463/scratch/u1_cat/native_n4.prj/kernel.cpp" (CRD-135)
# Error: Compilation aborted (CIN-5)
# Error: go analyze: Failed analyze
CSYN_RAISED RuntimeError: Failed to synthesize the design with Catapult HLS in csyn mode after 11.8s
```

Cause: `connections/marshaller.h` defines `Wrapped<ac::bfloat16>` only inside
`#if defined(__AC_STD_FLOAT_H) && !defined(__MARSHALLER_AC_STD_FLOAT_H)` (line
444), and the emitted file includes `ac_std_float.h` after `mc_connections.h`.
The emitted comment "bf16 needs NOTHING here: ac::bfloat16 already has one" is
true only for the other include order. Allo's exception names none of this; the
errors are only in `<prj>/build/catapult.log`.

**E2. Wire variant, P1 only, `go compile`** (from the run before P3):

```
# Error: $PROJECT_HOME/../kernel.cpp(428): Loop 'while' in thread 'run' must have a wait; (CIN-123)
# Warning: $PROJECT_HOME/../kernel.cpp(422): Path with no waits detected in process '/add_0/run'.  This can cause SystemC simulation to hang. (CIN-165)
# Warning: $PROJECT_HOME/../kernel.cpp(422): SystemC thread 'run' should be scheduled with 'iomode=fixed' since it writes to sc_signal 'v8'. (CIN-124)
# Error: Compilation aborted (CIN-5)
# Error: go compile: Failed compile
```

**E3. Pipelined Stream kernel without flush, in Verilator** (`verilator/cmp_stream_ii1_noflush.txt`):

```
RuntimeError: add_0 driver exited 4: stalled at cycle 261938: 251935 of 251936 outputs
```

Catapult: `Loop '/add_0/run/while' is pipelined with initiation interval 1 and no flushing (SCHD-43)`.
The last datum sits in the pipeline until another input arrives, which never does.

**E4. Asking Allo for the flush** (`s.pipeline("add_0:_", style="flp")`, `target="systemc"`):

```
RuntimeError pipeline: the systemc emitter does not write `style=`, so the pipeline control style on _ (style=flp) would be dropped and the RTL would use that tool's default. This is refused rather than dropped because a style is chosen to stop an RTL deadlock that no simulation shows. Build for vitis_hls/vivado_hls/pynq, or drop `style=` if the style is not needed on systemc.
```

(Correct behaviour per D-1, "refuse, not drop"; it leaves no way to ask.)

Warnings on every run: `CIN-124 ... should be scheduled with 'iomode=fixed' since it
writes to sc_signal 'done'`; `LIB-83 Component library 'nangate-45nm_beh' created
with a newer version of Catapult Library Builder, 2025.1/1129627 > 2024.2/1130128`;
`LIB-142 Extrapolation detected`; `CRD-111 statement is unreachable` (the post-loop
`done.write(true)` of a steady-state kernel).

## Findings (D-9 classes)

| # | class | finding | proposal (not applied; no core patches here) |
| --- | --- | --- | --- |
| F1 | **bug** (emitter) | bf16 over any Connections port fails `go analyze` (E1): `ac_std_float.h` is included after `mc_connections.h`. csim cannot see it: checked with Catapult's g++ 10.3 `-fsyntax-only` on the emitted `native_n4` file with only the `sc_trace` patch applied, it compiles without `__SYNTHESIS__` and fails with `marshaller.h:208:18: error: 'class ac::bfloat16' has no member named 'Marshall'` with it. Conversely the `sc_trace` bug (1) breaks that g++ compile (`no matching function for call to 'sc_trace(sc_core::sc_trace_file*&, const ac::bfloat16&, std::string&)'`) but not Catapult's front end. So csim and csyn each hit a different one of the two bf16 bugs. | `EmitSystemC.cpp`: emit `#include <ac_std_float.h>` (after the `AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE` define) before `mc_connections.h`; fix the "needs NOTHING" comment. A `csyn` smoke test with a bf16 port would have caught it. |
| F2 | **bug** (emitter) | A steady-state kernel whose body has no `Pop`/`Push` (Wire-only) has no `wait()` under `__SYNTHESIS__` -> `CIN-123` (E2). | Emit `wait()` in the synthesis branch when the body contains no blocking Connections op (or always for `Wire` ports). |
| F3 | **missing abstraction** | Allo has no way to say "a combinational unit with ports a, b, result". The harness's `native` is an array-mapping kernel; Catapult makes it a one-shot loop in the reset action, re-armed only by reset. The closest expressible shape is a free-running kernel with `Wire` ports (`synth_top=add_0`), which Catapult turns into inputs -> adder -> output register, still clocked, with reset, `done`, and a warm-up (3 cycles at 2.0 ns, 1 at 3.33 ns) that nothing in the interface states. `@df.unit` ports are `Stream`s only (`stream_ports.rst`), so the unit cannot be written as a unit either. | A combinational-function unit form (ports in, ports out, no thread), emitted as `SC_METHOD` / a Catapult `ccore`; `Wire` combinational mode (`c7402f9f`, the track's dependency) is the nearest existing piece. |
| F4 | **missing abstraction** | Pipeline stall mode. Catapult's default for a pipelined free-running loop is "no flushing", which strands the tail of a finite stream (E3); `style=` exists only for Vitis and is refused for SystemC (E4). The Allo simulator delivers every element, so the two disagree on what the program does. | Map `style="flp"` to Catapult `-PIPELINE_STALL_MODE flush` on that loop (and `"stp"` to `stall`) in the SystemC/Catapult emitters, instead of refusing. |
| F5 | **workaround** | II=1 needs an explicit `s.pipeline`; without it every variant is 3 cycles/vector at 2.0 ns. Matching "a vector every cycle" is a schedule decision the unit's text does not carry. | Record; per-unit schedule in the unit file. |
| F6 | **semantic mismatch** | NaN encoding: Catapult `ac_std_float` all-ones payload with a sign; Allo simulator (numpy/LLVM) `0x7fc0`/`0xffc0`, with a different sign rule on 12 invalid ops; MiniTPU canonical `+0x7fc0`. Three tools, three answers, for 2,028 / 251,936 vectors. | Triage: whether Allo's `bfloat16` should pin a NaN rule (the type does not say). |
| F7 | **semantic mismatch** | `(+0)+(-0)`: Catapult and Allo give IEEE `+0`, MiniTPU `-0` (already in `ref.py`). | Record (MiniTPU's deviation). |
| F8 | **finding** (reports) | `cycle.rpt` for `native` says Latency `-1`, Throughput `1`; the measured rate is 3 cycles/vector (1 with pipeline). A reset-action loop is invisible to Catapult's process-level summary; read the Loops table. Catapult's area score also ranks designs differently from DC (`wire_ii1_n16_3p33`: largest score, smallest cells). | Read throughput from RTL simulation, area from DC; do not quote `cycle.rpt` process rows for reset-action kernels. |
| F9 | **finding** (harness) | `rtl.py`'s existing drivers write `dut.p = (decltype(dut.p))in[k]`, which casts a `uint64_t` lvalue to `SData&` (a reinterpreting reference cast); it reads the low 16 bits only because the host is little-endian. The new `stream` driver assigns by value. | Replace the cast with a value conversion in `valid`/`bare`/`comb`. |

Not run: `cosim` and `mode="ppa"` (SCVerify needs a C++ testbench; the SystemC
`sc_main` testbench does not count, `catapult.rst` "The testbench"); the emitted
testbench's csim (bugs 2, 3 belong to `systemc-u1-fixes`).

## Reproduce

```bash
cd /work/shared/users/phd/sk3463/scratch/wt-u1-cat
source examples/minitpu/harness/env-zhang21.sh
R=dev/records/minitpu/u1_bf16_add_catapult_2026-10-02/scripts
S=/work/shared/users/phd/sk3463/scratch/u1_cat      # scripts assume $S holds variants.py/patch_kernel.py/run_csyn.sh
$ALLO_PYTHON $R/emit_csyn.py 251936 $S/native_ii1_n251936.prj --pipeline   # fails at csyn (E1); then:
$S/run_csyn.sh $S/native_ii1_n251936.prj                                  # P1 + Catapult, ~41 s
$ALLO_PYTHON $R/emit_var.py wire 16 $S/wire_ii1_n16_3p33.prj --pipeline --clock=3.33
PATCH_ARGS=--wire-wait $S/run_csyn.sh $S/wire_ii1_n16_3p33.prj
export MINITPU_HARNESS_CACHE=$S/hcache
$ALLO_PYTHON $R/cmp_rtl.py $S/native_ii1_n251936.prj/build/Catapult/top.v1 --top top
$ALLO_PYTHON $R/cmp_rtl.py $S/wire_ii1_n16_3p33.prj/build/Catapult/add_0.v1 --shape bare --latency 0 --warmup 1
$R/dc/run_dc.sh cat_wire_ii1_3p33 add_0 clk 3.33 $S/wire_ii1_n16_3p33.prj/build/Catapult/add_0.v1/concat_rtl.v
$R/dc/run_dc.sh minitpu_oreg_3p33 mtpu_add_oreg clk 3.33 \
    /work/shared/users/phd/sk3463/minitpu/src/core/vpu/vpu_bf16_add.sv $R/dc/mtpu_add_oreg.sv
```

`run_dc.sh` writes to `$S/dc/`. Generated RTL is not kept here; `SHA256SUMS.txt`
holds the sha256 of every `concat_rtl.v` (identical to its `concat_sim_rtl.v`),
`rtl.v`, emitted and as-run `kernel.cpp`, and DC netlist.
