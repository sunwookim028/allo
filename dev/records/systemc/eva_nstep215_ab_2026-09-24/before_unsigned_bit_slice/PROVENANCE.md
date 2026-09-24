# before_unsigned_bit_slice — EVA NSTEP=215, bit slices declared SIGNED

This is the side that reproduces the **committed reference's** types
(`examples/eva/generated/kernel.cpp`, emitted at `72c70dcb`, 2026-08-20).

| | |
|---|---|
| Allo commit | the commit that adds this directory; parent `dbbc54ae` |
| Patch on top | `../unsigned_slice_revert.patch`, applied to `allo/ir/builder.py` (blob `e7a813e3`), and nothing else |
| Emitter | `mlir/lib/Translation/EmitSystemC.cpp` blob `d562f22e` — unmodified, identical to the `after/` side |
| Bindings | rebuilt from that tree; `LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build` |
| Invocation | `cd examples/eva && python cosim_eva_systemc.py` (with the corrected argument order that this commit lands) |
| NSTEP = LANELEN | **215** |
| M, N, K | 1, 1, 6 — 1x1 passthrough, ramp `1..6` into WEST at cycle 15 |
| `PRIME_TOKENS` | 6 |
| Host | `ace-01`, no Catapult, no Xcelium — **emission only, the csim was not run** |
| `kernel.cpp` sha256 | `2ed2e0a89ba2e5958cd8d5fc094c87bb92fec328b848a3ecc24b0a27e334755b` |

Why this is a reconstruction rather than a checkout of `72c70dcb`: the C++
emitter at `72c70dcb` was measured to produce byte-identical output to the
current one on this design (see `../README.md`), so the only thing that has to
move to recover the reference's types is the one `allo/ir/builder.py`
attribute — and holding everything else at the current commit is what keeps the
pair a one-variable comparison.

Verified: at NSTEP=55 this build reproduces the committed reference exactly
except for the 29 `input<k>.data` index statements, which come from the
independent region-argument-order change described in `../README.md`.

Run: `export MGC_HOME=...; ./csim.sh` (~15 s). `output1.data` is `out_e`; its
first six values must be `1 2 3 4 5 6`.
