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
- New features/fixes go on `feature/*` or `fix/*` branches based on `upstream/main`, one branch per upstream PR.
- Fork-local files (docs, notes, CLAUDE.md, STATE.md, BRANCHES.md) live on `main` only — never on feature branches destined for upstream.
- See `BRANCHES.md` for current open PRs and branch dependencies.

# Don'ts
- Do not modify repository structure without approval
- Do not install system packages without explicit user confirmation
