# allo — Coding Agent Notes

## Quick pitfalls

- **Region arg-order reordering** when scalar in early kernel `args=[...]` → fix in `_build_top`; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **OMP segfault at exit** when GC races with OMP threads — set `OMP_NUM_THREADS=N`, run regions in separate processes; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **Wrong `LLVM_BUILD_DIR`** — conda `allo` env sets it already (`build-rhel8`); overriding with `build/` → GLIBC_2.33 crash; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **Scalar `@df.region()` args** — bare `int32` in `args=[...]` is **rejected** (PR #577); use `int32[1]` → `m_axi`. Auto-capture → `s_axilite` redesign pending upstream; details: `notes/PITFALLS_DATAFLOW_REGION.md`

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

## Project state

Live state is judged from git/GitHub/notes, not a checked-in status file:
`git branch -vv`, `gh pr list -R cornell-zhang/allo`,
`gh issue list -R sunwookim028/allo`, and `notes/`.

- Living feature map (fork vs upstream): the pinned fork issue
  https://github.com/sunwookim028/allo/issues/13. It is the single living
  picture of what the fork carries vs upstream; project state is otherwise
  judged from git/GitHub, not checked-in `.md` snapshots.
- Fork-local file inventory: fork issue #5
  (https://github.com/sunwookim028/allo/issues/5#issuecomment-4977128476).
- `notes/MAINTENANCE_CHECKLIST.md`: the upstream-merge procedure.

## Notes from AGENTS.md

See `AGENTS.md` for build instructions, testing, and code style guidelines.
