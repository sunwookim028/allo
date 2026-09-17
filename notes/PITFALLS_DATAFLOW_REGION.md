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

## Observation: LLVM_BUILD_DIR must NOT be overridden

The conda `allo` env already sets `LLVM_BUILD_DIR` to the RHEL8-compatible build
(`/work/shared/common/llvm-project-main/build-rhel8`). Overriding it with the
non-RHEL8 build (`build/`) causes a GLIBC_2.33 crash at simulator init.

## Inspecting MLIR: one process per dump

(Extracted from the retired `ALLO_LESSONS.md` on 2026-09-17; still true.)

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
