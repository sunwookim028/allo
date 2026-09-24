# MiniTPU model -- first gate, 2026-09-24

`examples/minitpu/run.py` on zhang-21, worktree `/home/sk3463/allo-minitpu`,
branch `minitpu-model`, `OMP_NUM_THREADS=8`,
`LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build`, bindings built in the
worktree.

```
  16x16x16    28 cmds  build 81.5s  run 11.2s  EXACT  rel err 0.162%  (bound 0.391%)
  16x64x16   100 cmds  build 87.8s  run 11.5s  EXACT  rel err 0.360%  (bound 0.781%)
  16x32x32   104 cmds  build 83.0s  run 11.2s  EXACT  rel err 0.246%  (bound 0.552%)
  32x64x16   168 cmds  build 82.0s  run 11.4s  EXACT  rel err 0.306%  (bound 0.781%)
  PASS
```

* **EXACT** = bit-identical to `examples/minitpu/reference.py`. That is a
  check on the *model*, not on the reading: `reference.py` is our own
  reimplementation of MiniTPU's rounding rules. No golden vector set exists
  yet; MiniTPU's owner is building one.
* **rel err** is the relative Frobenius-norm error against float64, the
  measure MiniTPU uses (`board_package/gpt2_kernels.py`,
  `_GEMM_ERROR_LIMIT = 0.01`). All four are well inside the 1 % bar.
* **bound** is `sqrt(K/16) * 2^-8`, the derived bound for summing `K/16`
  BF16-rounded partials.
* `16x16x16` is the array's own ISA-level tile, which MiniTPU's compiler would
  never emit. `32x64x16` is `[32,64]@[64,16]`, the smallest product
  `minitpu-cc` will build.
* ~280 concurrent kernels (256 PEs + 16 edges + 8 control), one OS thread each
  (`allo/backend/simulator.py:1388`).

**These are model figures and say nothing about MiniTPU's cycles.** Nothing in
the model counts cycles; the 168-cycle 16x16x16, the 52-cycle matrix step and
the 85-cycle result latency are not reproduced and were never targeted.

## Double rounding, measured -- a bit-exactness result, not an accuracy one

`acc24` is emulated as float32 rounded to 15 fraction bits. acc24 has
float32's 8-bit exponent field and bias, so the **normal** range matches
(smallest normal `2^-126`); the edges differ, since the significand is
narrower -- largest finite `(2 - 2^-15)·2^127` against `(2 - 2^-23)·2^127`, and
the smallest non-zero differs by `2^8`. Both edges are outside the operating
range for BF16 inputs.

`mxu_acc24_add_pipe` rounds the **exact** sum to acc24 once per add, so
round-once is the silicon behaviour and the float32 path is the divergent one.
Over 24 million random acc24 pairs the two disagree on **6,933 (0.03 %)**.
MiniTPU's owner reproduced it over 4,000,000 pairs across 40 binades:
**0.0636 %, exactly one acc24 ulp every time**. Their
`tools/accum_precision.py` takes the same float32 path.

**Not an accuracy figure.** One acc24 ulp is `2^-15`; the BF16 output is
rounded at `2^-8`, 128x coarser, so the difference reaches the result about one
time in 128 -- roughly `5e-6` per add, far below the BF16 floor every tolerance
in either project derives from. It matters for **bit-exactness**: a model that
reaches acc24 through float32 cannot be held bit-exact to silicon, and the
failure looks like a handful of one-ulp elements with no pattern.

## Fork gates re-run on this branch, 2026-09-24

| gate | result |
| --- | --- |
| `examples/tinytpu/reproduce.sh` | `REPRODUCED` 175 / 265 / 421 / 482 / 674, 0 mismatches |
| `gen_isa.py --check` | `ISA OK` |
| `lift_units.py --check` | `UNITS OK` |
| `stress_isa.py` (TPU_MAXDIM=16) | `STRESS OK: 492/492` |
| `act_compile.py --gate` | `ACT GATE OK: 12/12` |
| `check_numbers.py --reports examples/tinytpu/asic_synthesis/reports` | `AREA NUMBERS OK` |
| `pytest tests/dataflow tests/ip` | 137 passed, 18 skipped, 2 xfailed, 1 failed |
| `pytest tests/dataflow/test_bf16_dataflow.py` | 5 passed |
| docs `-W --keep-going`, fresh build dir | `build succeeded` |

The one `pytest` failure is `tests/dataflow/test_hierachical_mesh.py::test_2x2`
-- `AssertionError: vitis_hls is not available` (`allo/backend/hls.py:836`), an
absent tool rather than a regression; the test does not guard on availability.
`tests/dataflow/aie/` is excluded: `ModuleNotFoundError: No module named 'aie'`,
also an absent toolchain.
