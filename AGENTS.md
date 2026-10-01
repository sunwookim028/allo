# Building
- Always run `conda activate allo` before building or running tests
- Run `pip install -v -e .` to build the full project (includes MLIR/C++ backend)
- Read `docs/source/dive/frontend_syntax.rst` for comprehensive Allo frontend syntax reference
- Read `docs/source/dive/dataflow.rst` for the dataflow programming model (regions, kernels, streams)

# Testing
- Run `bash scripts/lint/task_lint.sh` for formatting checks
- Run `python3 -m pytest --ignore=tests/dataflow tests -v` for tests
  - Prefer running a single test file instead of the full suite (full suite is slow)
  - Use only software simulators (`target="llvm"` or `target="simulator"`)
  - If Vitis HLS tests are needed, ask the user to run them manually

# Code style
- Make small, targeted diffs rather than large refactors, and always be concise
- Prefer general solutions instead of one-off `if/else` patches
- Place Python frontend code in `allo/`
- Place MLIR dialects and passes code in `mlir/`
- Add tests and documentation for new features in `tests/` and `docs/`

# Branching policy
- `main` is the fork's integration HEAD, its default branch, and the home for all fork-local docs. This is the working branch. It is NOT a mirror of upstream.
- Upstream (`cornell-zhang/allo`) is tracked via the `upstream` remote; compare against `upstream/main`. There is no local mirror branch and no `next` branch.
- Remote convention: `origin` = `sunwookim028/allo` (the fork), `upstream` = `cornell-zhang/allo`.
- Two kinds of branch:
  - **Upstream PR branches** (`feature/*`, `fix/*`) are based on `upstream/main`, one branch per upstream PR, and carry no fork-local files.
  - **Fork work branches** are based on `main` and merge back into `main` once their result is solid.
- Fork-local files (fork docs pages, CLAUDE.md) live on `main` only - never on upstream PR branches.
- Direction, decisions and milestones are in `README.md`; defects are fork GitHub issues; the fork-vs-upstream feature map is the pinned fork issue https://github.com/sunwookim028/allo/issues/13.
- Commit messages follow upstream style: `[Tag][Tag] Imperative summary`.

# Fork exceptions
- Vitis HLS runs (csynth, cosim) are part of the fork's measurement flow (README, D-1); agents may run them, but tell the user first when a run takes longer than a few minutes.

# Don'ts
- Do not modify repository structure without approval
- Do not install system packages without explicit user confirmation
