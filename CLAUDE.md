# allo — Coding Agent Notes

## Quick pitfalls

- **Region arg-order reordering** when scalar in early kernel `args=[...]` → fix in `_build_top`; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **OMP segfault at exit** when GC races with OMP threads — set `OMP_NUM_THREADS=N`, run regions in separate processes; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **Wrong `LLVM_BUILD_DIR`** — conda `allo` env sets it already (`build-rhel8`); overriding with `build/` → GLIBC_2.33 crash; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **Scalar `@df.region()` args** — use bare `int32` (no brackets) to get `s_axilite`; `int32[N]` (any N, including 1) always maps to `m_axi`; details: `notes/PITFALLS_DATAFLOW_REGION.md`

## Environment

```bash
conda activate allo
# LLVM_BUILD_DIR already set in conda env (don't override)
export OMP_NUM_THREADS=8   # required for multi-kernel simulator
```

## Golden test for dataflow simulator

```bash
OMP_NUM_THREADS=8 conda run -n allo python tests/dataflow/test_df_unit.py
OMP_NUM_THREADS=8 conda run -n allo python tests/dataflow/test_region_stateful.py
```

## Notes from AGENTS.md

See `AGENTS.md` for build instructions, testing, and code style guidelines.
