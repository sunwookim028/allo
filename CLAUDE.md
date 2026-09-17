# allo — Coding Agent Notes

## Quick pitfalls

- **Region arg-order reordering** when scalar in early kernel `args=[...]` → fix in `_build_top`; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **OMP segfault at exit** when GC races with OMP threads — set `OMP_NUM_THREADS=N`, run regions in separate processes; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **`LLVM_BUILD_DIR` is NOT set by the conda env** — neither `conda activate allo` nor `conda run` sets it (verified 2026-09-17), and the simulator asserts `LLVM_BUILD_DIR is not set` without it. Export it explicitly; details: `notes/PITFALLS_DATAFLOW_REGION.md`
- **Scalar `@df.region()` args** — bare `int32` in `args=[...]` is **rejected** (PR #577); use `int32[1]` → `m_axi`. Auto-capture → `s_axilite` redesign pending upstream; details: `notes/PITFALLS_DATAFLOW_REGION.md`

## Environment

```bash
conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build   # the env does NOT set this

# OMP_NUM_THREADS no longer has to exceed the kernel-instance count: the
# simulator now sets the OpenMP team size to the section count itself.
# Before that fix a PE blocked on a stream spun in its section, so a team
# smaller than the section count never started the sections that would
# unblock it and the region hung SILENTLY. See notes/ALLO_SHORTCOMINGS.md #11.
export OMP_NUM_THREADS=8
```

## Golden test for dataflow simulator

```bash
# `conda run` does not source the activate scripts, so export the env first.
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
python tests/dataflow/test_df_unit.py
python tests/dataflow/test_region_stateful.py
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
