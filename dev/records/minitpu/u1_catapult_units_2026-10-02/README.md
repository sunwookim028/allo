# U1 remaining units through SystemC -> Catapult, against MiniTPU's RTL (Catapult track)

Host zhang-21 (RHEL 8.10, 64 cores, load ~8). Run 2026-10-02 on branch
`u1-catapult2` (worktree `scratch/wt-u1-cat2` from `origin/u1-pilot` at
`7da58494`). MiniTPU at `b3ba0a4d`, read-only. README D-1 / D-7 / D-9,
Catapult-track row. Follows the pilot (`../u1_bf16_add_catapult_2026-10-02/`,
findings F1-F9) and the pipe record (`../u1_pipe_2026-10-02.rst`, L1-L5).

Units: `vpu_bf16_mul` (comb), `vpu_bf16_mul_pipe` (latency 2),
`mxu_bf16_mul_acc24` (comb), `mxu_acc24_add_pipe` (latency 3), `vpu_alu`
(latency 3).

## Summary

- **No hand-patch to `kernel.cpp`.** All 45 Catapult runs below synthesize the
  SystemC as Allo emitted it. The pilot's P1 (`ac_std_float.h` include order,
  CRD-135) and P3 (Wire-only kernel lacks `wait()`, CIN-123) are fixed on
  `u1-pilot` (`c87462f7`, `bf6e4608`): the bf16-port kernels (`bf16_mul
  native`, `alu native`) pass `go analyze`, and every Wire kernel compiles.
  S4 (bf16 -> f32 widening) is fixed as well (`a80981a7`): `mul_acc24 native`
  now synthesizes as written, with no `native_bitext` workaround. The only
  hand edits are to `run.tcl`, to pin latency (below).
- **Every `bits`-family variant is bit-exact on the full stimulus**, in Verilator
  against MiniTPU's RTL, at both clocks and in both port styles: `bf16_mul bits`
  and `bf16_mul_pipe bits_pipe` (6,019,104), `mul_acc24 bits` (6,019,104),
  `acc24_add_pipe bits` (352,116), `alu bits` (5,079,552). Each `native` RTL
  differs from MiniTPU only on named rules, and **on exactly the vectors
  SystemC csim does** (mul 5,011,890; mul_acc24 4,941,087; acc24 349,935;
  alu 5,064,720). For these units csim predicts Catapult's RTL values exactly.
  The Allo simulator does not.
- **Declared latency through Connections ports: honoured on all five units.**
  With the hand-added `cycle set {<out>.Push()} -from {<in>.Pop()} -equal L` on
  every in/out pair, the RTL runs at II=1 with measured latency equal to the
  declared one (mul_pipe 2, acc24 3, alu 3 at 2.0 and 3.33 ns). Under
  backpressure nothing is lost. Unpinned, the latency is the clock's (L2
  again): mul_pipe 3 at 2.0 ns and 1 at 3.33 ns; alu 3 and 2. A declared 0 for
  the comb units is **refused** (SCHD-30), and 1 is honoured.
- **New: latency on Wire ports cannot be pinned (N2).** A `sc_in` read is not
  an op Catapult schedules. The I/O constraint has no anchor. A `cycle set` on
  the write op is accepted and has no effect, and the loop c-step constraint
  sets latency only by coincidence: acc24 3 -> 3, mul_pipe 2 -> 2 but 3, 4 -> 2,
  alu 3/4/5 -> 2 at 3.33 ns. It can shorten (mul 2 -> 1) but not lengthen.
- **Same-flow PPA (DC W-2024.09, FreePDK45 `stdcells.db` md5 `f5560259...`).**
  Like for like means the same port shape and clock: a MiniTPU unit plus an
  output register, or MiniTPU's own pipeline, against Catapult's Wire-port RTL.
  Catapult's RTL is **larger on 4 of 5 units at 3.33 ns**: mul +55 %,
  mul_acc24 +9 %, mul_pipe -4 %, acc24 +72 %, alu +18 % (at latency 2 against
  3). Every design closes both clocks in DC, including the ones Catapult's own
  `rtl.rpt` reports at -0.3 to -1.8 ns of slack (N4).

## Environment

```bash
cd /work/shared/users/phd/sk3463/scratch/wt-u1-cat2
source examples/minitpu/harness/env-zhang21.sh        # allo env, catapult-2024, Verilator 5.052
export MINITPU_HARNESS_CACHE=/work/shared/users/phd/sk3463/scratch/u1_cat2/hcache
```

| tool | version |
| --- | --- |
| Catapult | Catapult Ultra Synthesis 2024.2/1130128, library `nangate-45nm_beh` |
| Verilator | 5.052 2026-09-05 rev conda-forge build |
| Design Compiler | W-2024.09 for linux64 - Aug 27, 2024 (`module load synopsys-dc-W-2024.09`) |
| Allo bindings | a private copy of `scratch/wt-u1`'s snapshot of the `wt-sc-fix` build at `e238e410` (`mlir/build/SNAPSHOT`) |

**Bindings pitfall (environment).** Copying `mlir/build/tools/allo/_mlir` is not
enough to make the bindings private. `_mlir_libs/_allo*.so` has `RUNPATH
[$ORIGIN:/work/.../wt-sc-fix/mlir/build/tools/allo/_mlir]`, and
`libAlloMLIRAggregateCAPI.so` sits in `_mlir/`, not in `$ORIGIN` (`_mlir_libs/`).
So `ldd` resolved the CAPI library to **`wt-sc-fix`'s build**, the one the
integration run saw rebuilt mid-run. Fixed for this copy with a symlink
`_mlir_libs/libAlloMLIRAggregateCAPI.so.22.0git -> ../libAlloMLIRAggregateCAPI.so.22.0git`.
After that, `ldd` resolves every library inside the worktree. The
`u1_integration` snapshot has the same RUNPATH.

## What was built

Scripts in `scripts/` (run from the worktree root; projects in
`/work/shared/users/phd/sk3463/scratch/u1_cat2/<name>.prj`, not kept):

| script | does |
| --- | --- |
| `emit_csyn.py <unit> <variant> <prj> [...]` | builds the unit's variant for `target="systemc", mode="csyn"` and runs Catapult (`catapult -shell -f ../run.tcl` from `<prj>/build`, as `mod()` does). `--io-all L` adds the I/O cycle constraint for every (Push, Pop) pair of the kernel; `--tcl` adds raw tcl after `go architect`. `run.tcl.emitted` is Allo's tcl, `run.tcl` the one that ran. |
| `gen_wire.py <unit> <variant>` | writes `wire/wire_<unit>_<variant>.py`: the variant's kernel body **copied by text**, with only the ports changed (`<p>[i]` -> one `Wire.get()` per input, `<out>[i] = e` -> `wc.put(e)`, `for _`), between `src`/`sink` kernels, as the pilot's `variants.py:wire`. Generated files are committed. |
| `run_batch.sh <list> [P]` | one `emit_csyn.py` per line, P in parallel (`batch_stream.txt`, `batch_wire.txt`, `batch_wire_probes.txt`) |
| `cmp_rtl.py <unit> <prj> [--top K --shape bare] [--ready-period P]` | Catapult RTL (`concat_sim_rtl.v`) and MiniTPU's unit in Verilator via `harness/rtl.py`, full stimulus. `stream` shape: per-vector latency and cycles per vector. `bare` (Wire): one build, every edge's output recorded, latency **measured** as the output offset that matches on the first 3,000 vectors, then the whole stimulus is compared at that offset. Differences are classified by the unit's `EXPLAIN`. |
| `run_cmp.sh <out>` | `cmp_rtl.py` per line of stdin (`cmp_stream.txt`, `cmp_wire.txt`) |
| `summarize_csyn.py`, `dc/summarize_dc.py` | the tables below |
| `dc/run_dc.sh`, `dc/dc_u1.tcl` | the pilot's DC flow, unchanged (`dc_u1.tcl` byte-identical), output to `scratch/u1_cat2/dc`; `dc/dc_runs.txt` is the run list (`$M` MiniTPU `src/core`, `$S` scratch, `$R` scripts) |
| `dc/mtpu_mul_oreg.sv`, `dc/mtpu_macc_oreg.sv` | MiniTPU comb unit + output register (the pilot's `mtpu_add_oreg.sv` shape) |

Two port styles per unit:

- **Stream** (region `top`, `synth_top` default): the unit as the harness writes
  it, its arrays become Connections ports; `s.pipeline("<k>_0:i")`, the full
  stimulus length as trip count (pipe record H1: a non-flushing pipeline holds
  its tail otherwise). Pipelined units pinned with `--io-all L`.
- **Wire** (`gen_wire.py`, `synth_top=<k>_0`, `s.pipeline("<k>_0:_")`): plain
  `sc_in`/`sc_out` ports, the port shape of a MiniTPU unit; used for the
  like-for-like DC rows.

ALU and acc24 `bits` need `s.unroll("leading_zeros17:offset")` /
`("leading_zeros19:offset")` for II=1 (pipe L5, ALU C10), as before.

## Results per unit

Latency: rising edges from input capture to visible result (`harness/rtl.py`).
"stream" = Connections ports, region `top`; "wire" = Wire ports. Values are
full-stimulus counts against MiniTPU's RTL. Catapult area is its score
(`rtl.rpt` TOTAL AREA after assignment). Slack is `rtl.rpt`'s worst. All raw
lines are in `verilator/rtl_cmp_stream.txt`, `verilator/rtl_cmp_wire.txt` and
`catapult/csyn_summary.txt`.

### `vpu_bf16_mul` (comb, declared 0), 6,019,104 vectors

| build | clock | values | measured latency / rate | Catapult area | slack |
| --- | --- | --- | --- | --- | --- |
| stream `bits` | 2.0 | **6,019,104** | 2 / 1.000 | 1686.4 | 0.336 |
| stream `bits` | 3.33 | **6,019,104** | 2 / 1.000 | 1663.1 | 0.499 |
| stream `bits`, I/O `-equal 1` | 3.33 | **6,019,104** | **1** / 1.000 | 1588.5 | 0.221 |
| stream `bits`, I/O `-equal 0` | 3.33 | **refused**, SCHD-30 (E1) | -- | -- | -- |
| stream `native` | 2.0 | 5,011,890 | 2 / 1.000 | 1819.5 | 0.006 |
| stream `native` | 3.33 | 5,011,890 | 1 / 1.000 | 1561.7 | 0.518 |
| wire `bits` | 2.0 / 3.33 | **6,019,104** | 2 / 2 | 1061.8 / 1075.9 | 0.234 / 1.199 |
| wire `bits`, loop `-equal 1` | 2.0 / 3.33 | **6,019,104** | **1** / **1** | 1531.0 / 1203.5 | 0.126 / 0.386 |
| wire `native` | 3.33 | 5,011,890 | 1 | 949.6 | 0.838 |

`native` (Catapult `ac::bfloat16` `*`) equals `ref.ieee_bf16_mul` on all
6,019,104 vectors. The 1,007,214 differences from MiniTPU are its flush rules
and NaN (classified: 514,904 + 36,401 subnormal operand flushed, 51,958 + 53
tiny product flushed, 403,886 NaN operand, 12 `Inf x 0`). Catapult's NaN for
`*` is `0x7fc0` with sign `a ^ b` (`0000 x ffc0 -> ffc0`). For `+` (pilot) it
is the all-ones `0x7fff` (N5).

### `vpu_bf16_mul_pipe` (declared 2), 6,019,104 vectors

| build | clock | values | measured latency / rate | Catapult area | slack |
| --- | --- | --- | --- | --- | --- |
| stream `bits_pipe`, unpinned | 2.0 / 3.33 | **6,019,104** | **3** / **1**, 1.000 | 1576.9 / 1321.9 | 0.186 / 0.480 |
| stream `bits_pipe`, I/O `-equal 2` | 2.0 / 3.33 | **6,019,104** | **2** / **2**, 1.000 | 1453.8 / 1441.0 | 0.186 / 0.579 |
| same, output ready low 1 cycle in 3 | 3.33 | **6,019,104** | 2-4, 1.500 (sink's rate), none lost | | |
| stream `native`, I/O `-equal 2` | 2.0 / 3.33 | 5,011,890 | 2 / 2 | 2096.9 / 1655.4 | 0.038 / 0.515 |
| wire `bits_pipe`, unpinned | 2.0 / 3.33 | **6,019,104** | 3 / 1 | 1037.7 / 849.0 | 0.389 / 0.443 |
| wire `bits_pipe`, loop `-equal 2` | 2.0 / 3.33 | **6,019,104** | **2** / **2** | 1084.6 / 989.2 | 0.120 / 0.260 |
| wire, loop `-equal 3` / `4` (+ write op `-equal 2`/`3`) | 3.33 | 6,019,104 | **2** (not 3) | 996.9-1002.2 | 0.260 |

### `mxu_bf16_mul_acc24` (comb, declared 0), 6,019,104 vectors

| build | clock | values | measured latency / rate | Catapult area | slack |
| --- | --- | --- | --- | --- | --- |
| stream `bits` | 2.0 / 3.33 | **6,019,104** | 1 / 1, 1.000 | 1454.5 / 1415.4 | 0.095 / 0.635 |
| stream `native` (as written: S4 fixed) | 2.0 / 3.33 | 4,941,087 | 2 / 1 | 3171.9 / 2409.4 | -0.023 / 0.199 |
| wire `bits` | 2.0 / 3.33 | **6,019,104** | 1 / 1 | 873.2 / 868.9 | 0.344 / 1.014 |
| wire `native` | 3.33 | 4,941,087 | 1 | 1787.3 | 0.257 |

`native`: 403,886 NaN operand (Catapult `ffc000`, sign kept), 12 `Inf x 0`,
514,840 + 70,862 subnormal operand flushed, 88,417 tiny product flushed. Against
`ref.ieee_bf16_mul_acc24`: 6,019,073. The 31 others are W1's double rounding
of float32 subnormal products, now in RTL. MiniTPU's flush hides them.

### `mxu_acc24_add_pipe` (declared 3), 352,116 vectors

| build | clock | values | measured latency / rate | Catapult area | slack |
| --- | --- | --- | --- | --- | --- |
| stream `bits` + unroll LZC, I/O `-equal 3` | 2.0 / 3.33 | **352,116** | **3** / **3**, 1.000 | 4070.3 / 3924.0 | **-1.626 / -0.334** |
| same, output ready low 1 in 3 | 3.33 | **352,116** | 3-5, 1.500, none lost | | |
| stream `native`, I/O `-equal 3` | 2.0 / 3.33 | 349,935 | 3 / 3 | 3088.0 / 2893.6 | 0.008 / 0.004 |
| wire `bits` + unroll, unpinned | 3.33 | **352,116** | 2 | 5251.5 | -1.964 |
| wire `bits` + unroll, loop `-equal 3` | 2.0 / 3.33 | **352,116** | **3** / **3** | 4718.5 / 5487.2 | -1.785 / -0.583 |
| wire `native`, loop `-equal 3` | 3.33 | 349,935 | 3 | 2697.4 | 0.028 |

`native`'s 2,181: **2,067 NaN results become zeros** (pipe V1, now in Catapult
RTL: `0000+7fc000 -> 800000`), 113 double roundings, 1 `(+0)+(-0)`. Unpinned
acc24 at 5.0/2.0 ns is in the pipe record (2/3).

### `vpu_alu` (declared 3), 5,079,552 vectors (16 op codes)

| build | clock | values | measured latency / rate | Catapult area | slack |
| --- | --- | --- | --- | --- | --- |
| stream `bits` + unroll, unpinned | 2.0 / 3.33 | **5,079,552** | **3** / **2**, 1.000 | 5209.7 / 4645.4 | -1.338 / -1.403 |
| stream `bits` + unroll, I/O `-equal 3` | 2.0 / 3.33 | **5,079,552** | **3** / **3**, 1.000 | 5209.7 / 4725.4 | -1.338 / -1.403 |
| same, output ready low 1 in 3 | 3.33 | **5,079,552** | 3-5, 1.500, none lost | | |
| stream `native`, I/O `-equal 3` | 2.0 / 3.33 | 5,064,720 | 3 / 3 | 4091.1 / 4327.6 | 0.004 / -0.039 |
| wire `bits` + unroll, loop `-equal 3` | 2.0 | **5,079,552** | **3** | 5995.0 | -1.537 |
| wire `bits` + unroll, unpinned / loop `-equal 3`, `4`, `5` | 3.33 | **5,079,552** | **2** on all four | 5685.6-5708.3 | -1.15 |
| wire `native`, loop `-equal 3` | 3.33 | 5,064,720 | 3 | 4326.2 | 0.000 |

`native`'s 14,832 match csim's classes exactly: NaN `0x7fff` 6,249, MUL flush
1,320 + 4,745, zero signs 2, and **C8 in RTL**: 2,514 + 2 MAX/MIN vectors where
`std::max`/`std::min` order NaN and `+-0` as the RTL does not.

## Catapult RTL vs MiniTPU RTL: verdicts

| unit | bit-exact variant(s) | latency = declared? | throughput |
| --- | --- | --- | --- |
| `vpu_bf16_mul` | `bits` (stream, wire) | declared 0: **not reachable** (refused, E1); 1 is the least, pinned | II=1 |
| `vpu_bf16_mul_pipe` | `bits_pipe` (stream, wire) | **yes, 2**: stream pinned at both clocks; wire loop-pinned at both | II=1 |
| `mxu_bf16_mul_acc24` | `bits` (stream, wire) | declared 0: as `vpu_bf16_mul`; Catapult gives 1 unpinned | II=1 |
| `mxu_acc24_add_pipe` | `bits` (stream, wire) | **yes, 3**: stream pinned; wire loop-pinned (by coincidence, N2) | II=1 |
| `vpu_alu` | `bits` (stream, wire) | **yes, 3** on stream (pinned) and on wire at 2.0 ns; **no on wire at 3.33 ns** (2, not raisable, N2) | II=1 |

## Same-flow PPA: Design Compiler

The flow is the pilot's, unchanged: `scripts/dc/dc_u1.tcl` (byte-identical),
driven by `scripts/dc/run_dc.sh <name> <top> <clk|none> <period> <src>...`, with
the runs listed in `scripts/dc/dc_runs.txt`.

| setting | value |
| --- | --- |
| target library | `stdcells.db`, FreePDK45 `view-standard`, md5 **`f5560259ca91a4b67336b715b729f94d`** (= pilot), from `/scratch/users/sk3463/build_T4_MAXDIM16_shipped_baseline/1-freepdk-45nm/view-standard/` (`adk.tcl` md5 `5e884494824b75be2350b2e43ea9608a`, `adk-base.tcl` md5 `67261d8b88f45c69bea1853713e9e58a`) |
| synthetic library | `dw_foundation.sldb`, `/opt/synopsys/syn/W-2024.09/libraries/syn/`, md5 `d955c61b84f986958ce94ccd738f45e7` (= pilot) |
| compile | `ungroup -start_level 2 -all -flatten`; `compile_ultra -gate_clock`; ADK dont-use list; non-topographical |
| constraints | `create_clock -period P` on the clock port (`clk`; MiniTPU `clk_i`; virtual for comb); input/output delay 0; driving cell `INV_X2`; `ADK_TYPICAL_ON_CHIP_LOAD`; max fanout 20; max transition 0.25P |
| cores | `-max_cores 8`; DC: "Running optimization using a maximum of 8 cores" (OPT-1500); 8 runs in parallel, host load ~8 |
| sources | MiniTPU `src/core` at `b3ba0a4d` (sha256 in `SHA256SUMS.txt`), `MINITPU_MXU_ACC_USE_DSP` 0 (fabric); Catapult `concat_rtl.v` per run (sha256 in `SHA256SUMS.txt`) |
| wall | 62-75 s per run (`dc/<run>/wall.txt`) |

Reports: `dc/<run>/` (area, hierarchical area, timing, qor, reference, power,
clock gating); `dc/dc_summary.txt`. Areas are um2, pre-layout.

**Like for like** (same port shape, same clock): MiniTPU's comb unit plus an
output register, or its own pipeline, against Catapult's Wire-port RTL at the
measured latency shown.

| unit | clock | MiniTPU design | um2 | flops | slack | Catapult Wire RTL | lat. | um2 | flops | slack | Cat / MiniTPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16_mul | 3.33 | `vpu_bf16_mul` + oreg | 508.9 | 16 | 0.67 | `bits`, loop 1 | 1 | 790.8 | 46 | 0.58 | **1.55** |
| bf16_mul | 2.0 | same | 572.2 | 16 | 0.00 | same | 1 | 834.7 | 43 | 0.00 | 1.46 |
| bf16_mul_acc24 | 3.33 | `mxu_bf16_mul_acc24` + oreg | 517.1 | 24 | 1.49 | `bits` | 1 | 562.6 | 26 | 1.18 | **1.09** |
| bf16_mul_acc24 | 2.0 | same | 517.1 | 24 | 0.16 | same | 1 | 559.1 | 26 | 0.13 | 1.08 |
| bf16_mul_pipe | 3.33 | `vpu_bf16_mul_pipe` | 711.8 | 63 | 1.65 | `bits_pipe`, loop 2 | 2 | 681.0 | 44 | 0.53 | **0.96** |
| bf16_mul_pipe | 2.0 | same | 711.8 | 63 | 0.32 | same | 2 | 779.9 | 53 | 0.00 | 1.10 |
| acc24_add_pipe | 3.33 | `mxu_acc24_add_pipe` (+valid) | 1424.4 | 90 | 0.03 | `bits`, loop 3 | 3 | 2448.0 | 101 | 0.07 | **1.72** |
| acc24_add_pipe | 2.0 | same | 1556.6 | 90 | 0.01 | same | 3 | 2364.7 | 134 | 0.01 | 1.52 |
| alu | 3.33 | `vpu_alu` (+valid) | 2394.0 | 224 | 0.24 | `bits`, loop 3 | **2** | 2826.3 | 186 | 0.00 | **1.18** |
| alu | 2.0 | same | 2612.9 | 224 | 0.00 | same | 3 | 3947.2 | 298 | 0.00 | 1.51 |

Other DC rows (`dc/dc_summary.txt`):

| run | clock | um2 | flops | slack |
| --- | --- | --- | --- | --- |
| MiniTPU `vpu_bf16_mul` alone (virtual clock) | 2.0 / 3.33 | 489.4 / 438.4 | 0 | 0.01 / 0.63 (crit 1.99 / 2.70 ns) |
| MiniTPU `mxu_bf16_mul_acc24` alone | 2.0 / 3.33 | 408.6 / 408.6 | 0 | 0.15 / 1.48 (crit 1.85 ns) |
| Catapult stream `bf16_mul bits` (lat 2) | 2.0 / 3.33 | 1285.6 / 1296.2 | 116 / 121 | 0.00 / 0.10 |
| Catapult stream `bf16_mul_pipe bits_pipe`, I/O 2 | 2.0 / 3.33 | 1213.0 / 1167.5 | 116 / 115 | 0.00 / 0.45 |
| Catapult stream `mul_acc24 bits` (lat 1) | 2.0 / 3.33 | 1098.6 / 1079.2 | 97 | 0.00 / 0.88 |
| Catapult stream `acc24 bits`, I/O 3 | 2.0 / 3.33 | 2533.4 / 2563.7 | 219 / 193 | 0.01 / 0.14 |
| Catapult stream `alu bits`, I/O 3 | 2.0 / 3.33 | 3860.5 / 3121.8 | 350 / 253 | 0.00 / 0.01 |
| Catapult wire `native`: mul / mul_acc24 / acc24 (lat 3) / alu (lat 3) | 3.33 | 726.2 / 931.3 / 2126.7 / 2115.2 | 18 / 26 / 110 / 55 | 0.86 / 0.02 / 0.01 / 0.00 |

Reading the table:

- Both MiniTPU comb units fit one 2.0 ns cycle: 1.99 and 1.85 ns critical
  paths. The pilot's adder did not (2.38 ns). MiniTPU's acc24 and ALU pipelines
  also close 2.0 ns.
- Catapult's Wire RTL is smaller only for `bf16_mul_pipe` at 3.33 ns, by 4 %.
  The largest gap is `acc24_add_pipe`: 1.72x, and the comb area alone is 1.9x
  (1916 vs 1017 um2). Hypothesis, not tested: the transcription's
  workarounds cost logic that MiniTPU's text does not have. Those are the
  19-arm align `case` written as a 40-bit shift plus a jam mask (M5), the B1
  spare bits, and an unrolled 19-step priority loop for the LZC. `bf16_mul`
  (+55 %): with the loop pinned to 1, Catapult keeps its FSM state
  register and registers mid-path values (`reg_v8_ftd*`, `while_v98_qr_*`; 46
  flops against MiniTPU's 16) rather than one output register.
- The `native` Wire RTLs are not the same function (IEEE subnormals,
  float32 double rounding), so they are not like-for-like. The ALU's is
  smaller than MiniTPU's (2115 vs 2394 um2) and differs on 14,832 vectors.
- Stream (handshake) against Wire: +430 to +540 um2 on the three
  multipliers at either clock. On acc24 it is +116 (3.33 ns) and +169 (2.0 ns).
  On the ALU it is +296 at 3.33 ns, where the Wire RTL has latency 2, and -87
  at 2.0 ns. The two port styles get different schedules, so the difference is
  not only the handshake.

## Errors and messages, verbatim

**E1. Declared latency 0 through Connections ports** (`catapult/mul_bits_3p33_io0/csyn.log.gz`):

```
# User constraint applied between 'v1.Pop()' and 'v2.Push()',  min = 0  max = 0 (CNS-4)
# Error: Schedule failed, please check the following constraints. (SCHD-30)
# $PROJECT_HOME/../kernel.cpp(489):  kernel.cpp(489,20,8): chained data dependency at time 120cy+2.664 (SCHD-6)
# $PROJECT_HOME/../kernel.cpp(489):    from operation MODULAR_IO "v0.Pop()" DELAY 0 ON "ccs_connections.ccs_conn_in_wait(3,16,0,0,0,0)" (SCHD-6)
# $PROJECT_HOME/../kernel.cpp(831):  kernel.cpp(831,6,13): user constraint at time 120cy+2.664 (SCHD-6)
# $PROJECT_HOME/../kernel.cpp(831):    from operation MODULAR_IO "v2.Push()" DELAY 0 ON "ccs_connections.ccs_conn_out_wait(5,16,0,0,0,0)" (SCHD-6)
```

The 2.66 ns multiplier fits a 3.33 ns cycle, but a `Push` cannot chain after
a `Pop` in the same c-step: zero latency is infeasible on Catapult's
latency-insensitive ports at any clock.

**E2. A cycle constraint on a Wire write, accepted and ignored** (`catapult/wprobe_mulpipe_3p33_w3/csyn.log.gz`):

```
# > cycle set {/mul_0/run/v8.write:asn(v8)} -equal 3  ;# [hand-patch: declared latency]
# /mul_0/run/v8.write:asn(v8)/CSTEPS_FROM {{.. == 3}}
# $PROJECT_HOME/../kernel.cpp(523): Prescheduled LOOP '/mul_0/run/while' (1 c-steps) (SCHD-7)
```

No warning; the RTL is identical to the unconstrained one (score 849.0,
measured latency 1). `cycle find_op` finds no read op for an `sc_in`. `*v6*`
matches only `while:else:v63:asn(...)`, and `*read*` gives `No operation found
matching '*read*'`. The input has nothing for `-from` to name.

**Messages on every run** (counts over the 45 kept runs): `CIN-124` (thread
should be `iomode=fixed`, writes `done`) 128x, `LIB-83` (library built with a
newer Catapult Library Builder) 45x, `LIB-142` (extrapolation) 45x,
`CRD-111` (statement unreachable: the post-loop `done.write(true)`) 19x;
nothing else. No `CIN-123`, no `CRD-135`.

## Findings (D-9 classes)

The pilot's F1-F9 and the pipe record's L1-L5 hold where they apply. New or
changed:

| # | class | finding | proposal (not applied) |
| --- | --- | --- | --- |
| N1 | **match** (bug fixed) | F1/P1, F2/P3 and S4 are fixed on `u1-pilot`, and that holds in Catapult on 45 runs: bf16 Connections ports, Wire-only kernels and bf16->f32 widening synthesize as emitted. | Close the pilot's P1/P3 as fixed. |
| N2 | **missing abstraction** (backend) | **A Wire-port unit's latency cannot be pinned.** An `sc_in` read is not a schedulable op, so there is no `-from` anchor. A write-op constraint is accepted and silently ignored (E2). The loop c-step constraint is not latency: it lengthens acc24 2 -> 3, but mul_pipe stays 2 at loop 3/4 and the ALU stays 2 at loop 3/4/5 (3.33 ns), while it shortens bf16_mul 2 -> 1. Catapult honours a latency only on handshake ports. | The latency proposal (pipe record) should say: Catapult honours `latency=` through the I/O constraint on Connections ports; on `Wire` ports it **refuses**, or emits the loop form and treats it as unchecked until the harness measures it. |
| N3 | **semantic mismatch** (refines triage 4) | Latency 0 is refused through Connections ports (E1); 1 is the least Catapult can give a comb unit, and unpinned it gives 1 or 2 by clock (mul 2 at both clocks, mul_acc24 1). | Keep triage item 4's "recorded deviation"; the comb leaves' Catapult latency is 1 when pinned. |
| N4 | **finding** (reports) | `rtl.rpt` slack is negative for every `bits` acc24/ALU build (-0.33 to -1.96 ns), with no error and the latency honoured, yet DC closes the same RTL at the same clock (slack 0.00-0.14 ns). Catapult's scheduler and its post-assignment timing disagree, and both differ from DC. | Do not use `rtl.rpt` slack as a timing verdict (as F8 for `cycle.rpt`). For Wire kernels, `cycle.rpt` latency matched the measured latency on all 19 builds. For region-top Stream kernels it says `-1` (F8). |
| N5 | **semantic mismatch** | The NaN that Catapult's `ac_std_float` returns depends on the op. For `+` (pilot) it is all-ones `0x7fff`/`0xffff`. For `*` it is `0x7fc0` with sign `a ^ b`, equal to `ref.ieee_*`. binary32 `*` keeps the operand sign (`ffc000`). For binary32 `+`, the acc24 round turns NaN into `+-0` (V1, now in RTL). | Triage item 3 (NaN policy) covers it; add the per-op rule to the record of what each backend's `bfloat16` means. |
| N6 | **match** (csim = RTL) | For every `native` unit Catapult's RTL differs from MiniTPU on exactly csim's vectors, including C8 (`std::max` NaN/zero order, 2,516 MAX/MIN) and V1 (NaN -> 0, 2,067). | csim is a faithful value oracle for Catapult; the simulator is not (N1 of `u1_mul`, C7). |
| N7 | **finding** (PPA) | Same flow and ports: Catapult's RTL from the bit-exact Allo text is 0.96-1.72x MiniTPU's area at 3.33 ns, worst on acc24 (1.72x) and bf16_mul (1.55x); for the pilot's adder it was 0.92x. | Before U3: synthesize acc24 `bits` with the M5 shift written as a `case` (if the notation allows) to test the hypothesis that the workarounds cost the area. |
| N8 | **environment** | A copied `_mlir` still loads `wt-sc-fix`'s `libAlloMLIRAggregateCAPI.so` through its absolute RUNPATH (above). | Document in `dev/toolchains.rst`: a private bindings copy needs the CAPI library next to `_mlir_libs/` (symlink) or `LD_LIBRARY_PATH`. |

Not run: cosim / `mode="ppa"` (SCVerify needs a C++ testbench, as the pilot);
RTLGen and AMC on these units; `stages`/`staged`/`netlist` variants through
Catapult. The pipe record has acc24 `staged` at latency 5. The ALU's `netlist`
and `bits_dispatch` were not built here. The pilot's `vpu_bf16_add(_pipe)`
was not rerun.

## Reproduce

```bash
cd /work/shared/users/phd/sk3463/scratch/wt-u1-cat2
source examples/minitpu/harness/env-zhang21.sh
export MINITPU_HARNESS_CACHE=/work/shared/users/phd/sk3463/scratch/u1_cat2/hcache
R=dev/records/minitpu/u1_catapult_units_2026-10-02/scripts
S=/work/shared/users/phd/sk3463/scratch/u1_cat2
$ALLO_PYTHON $R/gen_wire.py alu bits                       # regenerate a Wire form (committed in $R/wire/)
$R/run_batch.sh $R/batch_stream.txt 6                      # 26 Catapult runs, ~45 s each -> $S/batch_stream.out
$R/run_batch.sh $R/batch_wire.txt 6                        # 19 Wire runs
$R/run_cmp.sh $S/cmp_stream.out < $R/cmp_stream.txt        # Verilator, full stimulus, ~10 s each
$R/run_cmp.sh $S/cmp_wire.out   < $R/cmp_wire.txt
sed "s#\$S#$S#g; s#\$M#/work/shared/users/phd/sk3463/minitpu/src/core#g; s#\$R#$PWD/$R#g" $R/dc/dc_runs.txt \
  | xargs -P 8 -L 1 $R/dc/run_dc.sh                        # 38 DC runs, ~65 s each
python3 $R/dc/summarize_dc.py $S/dc; python3 $R/summarize_csyn.py $S/*.prj
```

`run_batch.sh` and `run_cmp.sh` hold this worktree's and scratch paths. The
generated RTL is not kept. `SHA256SUMS.txt` has the sha256 of every emitted
`kernel.cpp`, `concat_rtl.v`/`concat_sim_rtl.v` and DC netlist, and of the
MiniTPU sources.
