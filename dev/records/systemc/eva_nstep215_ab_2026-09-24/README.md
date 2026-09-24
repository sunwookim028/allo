# EVA at NSTEP=215: with and without the unsigned-bit-slice rule — 2026-09-24

Emitted on `zhang-21`'s counterpart (`ace-01`, no Catapult, no Xcelium) so the
licensed host can compile and run both sides and diff them against the ramp
golden. **Neither side has been run here**: `cosim_eva_systemc.py` emits, writes
`input*.data`, and then dies in `_find_catapult_binary()`. There are therefore no
`output*.data` in either directory — they are what the csim produces.

```bash
cd <either directory>
export MGC_HOME=...                     # Catapult Mgc_home
./csim.sh                               # ~15 s; writes output*.data
```

`out_e` is `output1.data` (the region's second `out` argument, order below).
The workload is the 1x1 passthrough: a 6-long ramp `1..6` enters WEST at cycle
15 and must leave EAST unchanged, so `output1.data[0:6] == 1 2 3 4 5 6` after the
collector compacts. Anything else — in particular all zeros — is a fail.

## What the two directories are

| Directory | Bit slices |
|---|---|
| `before_unsigned_bit_slice/` | declared **signed** (`ac_int<N,true>`, `int8_t`) |
| `after_unsigned_bit_slice/` | declared **unsigned** (`ac_int<N,false>`, `uint8_t`) |

`kernel_cpp.diff` is the whole difference: **286 lines, 143 declarations, every
one of them signed → unsigned.** Nothing else differs — same SSA names, same
module structure, same `input*.data` (byte-identical in both directories), same
NSTEP. That is the point of the pair: whatever the csim does differently is
caused by those 143 declarations and by nothing else.

## Which change this actually is — the emitter is not the axis

The handoff asked for "the old emitter", meaning a commit of
`mlir/lib/Translation/EmitSystemC.cpp`. **Measured, that is the wrong axis.**

The committed reference `examples/eva/generated/kernel.cpp` was last regenerated
at `72c70dcb` (2026-08-20). Reverting `mlir/lib/Translation/` **and**
`mlir/include/allo/Translation/` to `72c70dcb` — all five files that changed
since, 546 lines, including all of the IP-instantiation work — and rebuilding the
bindings changes the EVA emission in **zero** lines after SSA-name normalisation.
The C++ emitter is not what moved.

What moved is `allo/ir/builder.py`, in two commits that arrived within three
hours of each other on 2026-09-18:

- `3de74846` "hls: emit bit slices as unsigned" (this fork)
- `094ab413` "[Bugfix][IR][HLS] Preserve unsigned bit-slice types during HLS codegen (#612)" (upstream)

Both attach the `unsigned` `UnitAttr` to `GetIntSliceOp`, which is how signedness
reaches the emitters. `3de74846` also reorders `fixUnsignedType`/`emitValue` in
`VhlsModuleEmitter::emitGetSlice`; that half is inert on the SystemC path, whose
`emitGetSlice` already ordered them correctly. So on this design the whole
observed change is one Python attribute.

`unsigned_slice_revert.patch` is the exact, minimal patch that produced the
`before/` side: it removes those attachments from `allo/ir/builder.py` and
nothing else.

## Provenance

| | |
|---|---|
| Allo commit | this commit; parent `dbbc54ae` |
| `mlir/lib/Translation/EmitSystemC.cpp` | blob `d562f22e` (identical on both sides) |
| `allo/ir/builder.py` | blob `e7a813e3`; `before/` = that blob with `unsigned_slice_revert.patch` applied |
| bindings | built in a worktree of this commit against `LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build` |
| invocation | `cd examples/eva && python cosim_eva_systemc.py` |
| NSTEP / LANELEN | 215 (`pc=15` + 200) |
| M, N, K | 1, 1, 6 |
| `PRIME_TOKENS` | 6 |
| `kernel.cpp` sha256 (before) | `2ed2e0a89ba2e5958cd8d5fc094c87bb92fec328b848a3ecc24b0a27e334755b` |
| `kernel.cpp` sha256 (after) | `f246c0cf7d7da763c4d29fafe99cb8a2880e7bc3ad6c0616d9c4f4e14ac9e8ae` |

## The other thing that changed, and why it is not in the pair

The committed NSTEP=55 reference differs from today's emission in **two**
independent ways, not one. After SSA normalisation, 344 lines:

- **286 lines** — the signed → unsigned rule above.
- **58 lines (29 statements)** — the testbench's `input<k>.data` indices. The
  region's MLIR argument order used to be a *discovery* order, args pulled
  forward in the order the kernels declaring them were visited, which put
  `prime_cfg` first. It is now the **declared** order, and `prime_cfg` is last.

The second one is not visible in `kernel_cpp.diff` because both sides here were
emitted from the same (current) frontend. It matters anyway, because
`cosim_eva_systemc.py` still passed the discovery order as of `dbbc54ae`, and
that does not raise: the arity still matches, so `write_tensor_to_file` pairs
each array with the wrong slot and the csim reads a 1-value `prime_cfg` file
where the testbench wants 215 halfs. Both runs here use the corrected order,
which this commit also lands in the script together with an assertion that
compares each argument's dtype and shape against the module's signature.

Declared order, which is what the `input<k>.data` indices now follow:

```
in_w in_e in_n in_s | out_w out_e out_n out_s | rin_w rin_e rin_n rin_s
                    | rout_w rout_e rout_n rout_s | iv_w iv_e iv_n iv_s | prime_cfg
```

`input0` = `in_w` (the ramp at index 15..20), `input7` = `rin_s` (the program
packets), `input8` = `iv_w` (the valid mask, 1 at 15..20), `input12` =
`prime_cfg` = 6. Verified on both sides before they were copied here.

## What this pair can and cannot settle

It can settle whether the unsigned declarations break the EVA passthrough: run
both, diff `output1.data` against the ramp.

It cannot settle whether *signed* was ever right. The reference's own csim runs
with no inputs and outputs all zeros (see `../reports/zhang21_2026-09-24/`), and
`before/` is a reconstruction of the reference's types, not the reference's
build. If `after/` passes, the rule is a correction and the committed reference
should be regenerated. If `before/` passes and `after/` does not, the rule breaks
something on this design and that is the finding worth having.
