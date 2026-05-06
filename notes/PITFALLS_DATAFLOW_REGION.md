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

## Scalar region args: bare `int32` → `s_axilite`, `int32[N]` → `m_axi`

Use bare `int32` (no brackets) as a `@df.region()` argument to obtain a true
AXI-Lite scalar port (`s_axilite`) in the generated HLS and a pass-by-value
parameter in the simulator. `int32[N]` for any N (including N=1) always maps
to an AXI-MM pointer (`m_axi`).

Implementation coverage (all three layers updated):
- **`allo/dataflow.py : _build_top()`** — emits bare MLIR `i32` for shape-`()` args
  instead of the earlier `memref<i32>` (0-D memref).
- **`allo/backend/vitis.py : postprocess_hls_code()`** — collects scalar args and
  emits `#pragma HLS interface s_axilite port=<arg> bundle=control` after the `m_axi` block.
- **`allo/backend/llvm.py : LLVMModule.__call__()`** — accepts `np.integer`,
  `np.floating`, and 0-d `np.ndarray` in the scalar slot (converts via `.item()`).

Test: `tests/dataflow/test_df_unit.py::test_region_bare_scalar_arg`.

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
