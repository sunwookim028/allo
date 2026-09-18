# Pitfalls: @df.region() with Multi-Kernel Arg Mapping

## Bug: top-func arg order reordered when scalar n appears before array in kernel args

**Root cause:** `_build_top` in `allo/dataflow.py` populated `input_types` in
kernel-appearance order (first kernel's args first). If a scalar `int32[1]` arg
appeared in an earlier kernel's `args=[arr_in, n]` list than a large array that
only appeared in a later kernel (`args=[arr_out]`), the top-level MLIR function
ended up with `(arr_in, n, arr_out)` instead of the declared `(arr_in, arr_out, n)`.
Python callers invoking `mod(inp, out, n)` would silently corrupt args.

**Fix:** `_build_top` now pre-seeds `used_args` and `input_types` from
`s.func_args[s.top_func_name]` (the region's canonical declaration order) before
iterating kernels. Kernel args resolve to canonical positions via `dtensor.top_name`
(set by `allo/ir/infer.py` from `args=[...]` annotation).

**File changed:** `allo/dataflow.py` — `_build_top()` function, argument mapping
section (~line 481).

## Scalar region args: bare `int32` is rejected; use `int32[1]` → `m_axi`

**Corrected 2026-09-17.** An earlier version of this section claimed that a bare
`int32` (no brackets) `@df.region()` argument yields a true AXI-Lite scalar port
(`s_axilite`). That held only on the short-lived
`feature/region-bare-scalar-axilite` branch, which was reverted (`a7ae144f`,
reverting `dbc60b43`) after upstream PR #577 settled on rejecting scalars in
`args=[...]`.

On `main` today:

- `args=[...]` takes array types only. `int32[N]` for any N (including N=1)
  maps to an AXI-MM pointer (`m_axi`); there is no scalar port path.
- `s_axilite` appears nowhere in `allo/backend/vitis.py` (only in
  `allo/backend/pynq.py`), and
  `tests/dataflow/test_df_unit.py::test_region_bare_scalar_arg` no longer
  exists.
- The auto-capture → `s_axilite` redesign is still open. It is item 4 of the
  hierarchical-region design record (`notes/archive/HIERARCHY_DESIGN.md`) and
  is tracked in fork issue #7.

## Observation: OMP segfault at Python GC exit

When running simulator with OMP threads and the module is not freed before
interpreter exit, Python GC (dict_traverse / func_traverse) can segfault due to
OMP thread-local storage teardown racing with the GC. Workaround: set
`OMP_NUM_THREADS=N` explicitly and run each region in its own process. This
is a Python/OpenMP interaction issue, not an Allo bug.

## Observation: LLVM_BUILD_DIR must be set BY YOU -- corrected 2026-09-18

This section used to say the conda `allo` env "already sets `LLVM_BUILD_DIR`" to
`/work/shared/common/llvm-project-main/build-rhel8` and must not be overridden.
**That is false on this host and contradicted `CLAUDE.md`, which is right.**
Checked directly:

```
$ conda activate allo && echo "${LLVM_BUILD_DIR:-<unset>}"
<unset>
```

Neither `conda activate allo` nor `conda run` sets it, and the simulator asserts
`LLVM_BUILD_DIR is not set` without it. Export it yourself:

```bash
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build
```

What survives of the original warning is the reason it was written: **which**
build you point at matters. A build made against a newer glibc than the host's
fails at simulator init with a GLIBC_2.33 error, which is why an RHEL8-compatible
build was named here in the first place. The path above is the build every
result in this repo was produced with.

## Inspecting MLIR: one process per dump

(Extracted from the retired `ALLO_LESSONS.md` on 2026-09-17; still true. That
file was deleted 2026-09-18 once this was the only part of it left standing --
`git show f1c3aad1^:notes/ALLO_LESSONS.md`.)

The MLIR context cannot be re-initialized in the same Python process — a second
`customize()` in one interpreter aborts with
`LLVM ERROR: Option 'fast' already exists!`. Use one process per module dump:

    conda run -n allo python -c "...customize and print str(s.module)..."

and pipe through `head` / `sed -n` for line ranges; generated MLIR carries no
source attribution, so match by function name (`func.func @decoder_0`), never by
line number.

These errors are library bugs, not user-code bugs — the right move is a minimal
reproducer and a compiler fix, not a rewrite of the kernel to dodge them:

- `cannot be converted to LLVM IR: missing LLVMTranslationDialectInterface`
- `Assertion 'value' failed`
- `Failure while creating the ExecutionEngine`
- `AttributeError: 'ASTContext' object has no attribute 'global_op_cache'`
- `LLVM ERROR: Option 'fast' already exists!`
