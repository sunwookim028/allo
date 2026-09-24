# after_unsigned_bit_slice — EVA NSTEP=215, bit slices declared UNSIGNED

This is the current toolchain, unmodified.

| | |
|---|---|
| Allo commit | the commit that adds this directory; parent `dbbc54ae` |
| Patch on top | none |
| Emitter | `mlir/lib/Translation/EmitSystemC.cpp` blob `d562f22e` — identical to the `before/` side |
| `allo/ir/builder.py` | blob `e7a813e3`, carrying `3de74846` and `094ab413` (#612) |
| Bindings | rebuilt from that tree; `LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build` |
| Invocation | `cd examples/eva && python cosim_eva_systemc.py` (with the corrected argument order that this commit lands) |
| NSTEP = LANELEN | **215** |
| M, N, K | 1, 1, 6 — 1x1 passthrough, ramp `1..6` into WEST at cycle 15 |
| `PRIME_TOKENS` | 6 |
| Host | `ace-01`, no Catapult, no Xcelium — **emission only, the csim was not run** |
| `kernel.cpp` sha256 | `f246c0cf7d7da763c4d29fafe99cb8a2880e7bc3ad6c0616d9c4f4e14ac9e8ae` |

Run: `export MGC_HOME=...; ./csim.sh` (~15 s). `output1.data` is `out_e`; its
first six values must be `1 2 3 4 5 6`.
