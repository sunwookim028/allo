# Maintenance checklist

Durable procedure for reconciling `main` with `upstream/main`. Project state
(open PRs, branch dependencies) is judged from git/GitHub, not checked-in `.md`
snapshots; the living fork-vs-upstream feature map is the pinned fork issue
https://github.com/sunwookim028/allo/issues/13.

## When a PR merges upstream

1. `git fetch upstream` to refresh `upstream/main`.
2. Delete the merged branch locally and on `origin` (the fork).
3. Reconcile `main` with the merged commit. `main` is not a fast-forward of
   `upstream/main`, so this goes through the main<->upstream reconciliation
   (fork issue #5) and MUST preserve the fork-local features inventoried in
   fork issue #5 (nb-stream primitives, Catapult nb-stream support):
   https://github.com/sunwookim028/allo/issues/5#issuecomment-4977128476.
4. Update the affected fork issues (`gh issue list -R sunwookim028/allo`) and
   any relevant `notes/`.
5. Rebase any live feature/fix/wip branches that carried the merged branch as
   a dependency onto the refreshed `upstream/main` to drop the merged-in
   commits; check `git branch -a` for the current set (branches come and go,
   so do not hardcode names here).

## Branch layout (as of 2026-09-17)

| Branch | Lineage | Role |
| --- | --- | --- |
| `main` | cornell-zhang | Fork integration branch: upstream plus fork-local features. **Not** a mirror of upstream — 81 ahead, 0 behind `upstream/main` (`8bafb0dc`) as of this writing. |
| `upstream` | cornell-zhang | Mirror of `upstream/main`, tracking the `upstream` remote. Refresh it to see what has landed; diff `main` against it to see what the fork carries. **Rebasing `main` onto it is never automatic — it is an explicit call.** |
| `chia-codesign` | `kkkaishao/allo` (ACT) | The CHIA / TinyTPU co-design artifact. A separate codebase, not a feature branch — see below. |

`main` and `chia-codesign` are the only working branches; `upstream` is just
the mirror.
`fix/vhls-mlir-percent-alloc-csim` is gone: upstream PR #554 merged and is now
the tip of `upstream/main`.

Read-only lineages on the `kai` remote, for reference rather than merging:
`kai/main` (has `dataflow.py`, plus `frontend/ harness/ primitives/`),
`kai/allov2` (`compiler/ lang/ operators/ schedule`, no `dataflow.py`, no ACT),
and `kai/act` (the allov2 lineage plus `exp/dsa` — the ACT compiler flow that
`chia-codesign` descends from).

Tags, in place of branches that were retired because their history is reachable
elsewhere: `tinytpu-rtlgen-base` (`882f7dd6`, the retired branch tip, now an
ancestor of `chia-codesign`; the actual fork point between `main` and
`chia-codesign` is `76130c63`) and `wip-u280-rescue` (`1bd3c5a6`, an unreviewed
u280 / nb-stream / fp16 rescue point from 2026-07-06).

### `chia-codesign` is a different codebase, not a feature branch

It shares only a March 2026 ancestor (`76130c63`, upstream #555) with `main`.
Kai's ACT fork re-architected the package:

| | `main` | `chia-codesign` |
| --- | --- | --- |
| `allo/` | 98 files: `dataflow.py`, `ir/`, `passes.py`, `customize.py`, `_mlir/`, `autoscheduler/` | 117 files: `compiler/`, `lang/`, `operators/`, `schedule/`, `exp/` |

None of the first column's files exist in the second. A trial merge
(`git merge-tree main chia-codesign`) reports 26 conflicts, 24 of them "core
file deleted in chia-codesign and modified in main". **The two lineages are
maintained separately and are not expected to converge**; the upstream-merge
procedure above applies to `main` only. `chia-codesign` carries its own
frontmatter (`CODESIGN.md`) and checkpoint (`notes/CHIA_CHECKPOINT.md`).
**Neither file exists on `main`** -- do not go looking for them in this tree.
They live on the `chia-codesign` branch; on this host that is the worktree at
`/home/sk3463/allo-chia-wt`, so the checkpoint reads as
`/home/sk3463/allo-chia-wt/notes/CHIA_CHECKPOINT.md` (or
`git show chia-codesign:notes/CHIA_CHECKPOINT.md`).

## Two LLVM builds, one submodule: check the version before trusting a build

There are **two** LLVM/MLIR builds on this host and they are different LLVM
versions. Verified 2026-09-18:

| Build | Size | `llvm-config --version` | `VCSRevision.h` `LLVM_REVISION` | Configured `LLVM_SOURCE_DIR` |
| --- | --- | --- | --- | --- |
| `/home/sk3463/llvm-allo-6b09f739/build` (external, what `LLVM_BUILD_DIR` points at) | 11 GB | `22.0.0git` | `6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb` | its own out-of-tree checkout |
| `externals/llvm-project/build` (in-tree) | 3.9 GB | `23.0.0git` | `040a641988f6ed6f4fab250706ca2b620c1de2d8` | `/home/sk3463/allo/externals/llvm-project/llvm` |

**`git status` showing `M externals/llvm-project` is the correct state, not
dirt. Do not "fix" it.** `main` records the pin `6b09f739` (LLVM 22), but the
working checkout is deliberately at `040a6419` (LLVM 23), because the in-tree
build above was configured against those sources and `chia-codesign` links
against that build.

This was gotten wrong once, on 2026-09-18: the drift was read as accidental and
the submodule was checked back out to the pin. Nothing on `main` noticed --
`main` does not use this submodule at all, its `allo/_mlir` points at the
external `llvm-allo-6b09f739` tree -- but it silently put LLVM 22 sources under
`chia-codesign`'s LLVM 23 binaries (see the worktree section below). It was
restored the same day.

So: the checked-out revision serves `chia-codesign`, the recorded pin serves
`main`, and they are not the same revision. That is a structural consequence of
two worktrees sharing one submodule, and the real fix is rule 2 below, not a
`git submodule update`.

Practical rule: **`externals/llvm-project/build` is not the project's build.**
`LLVM_BUILD_DIR` and `mlir/build` both point at the external 6b09f739 tree
(`mlir/build/CMakeCache.txt` records
`LLVM_DIR=/home/sk3463/llvm-allo-6b09f739/build/lib/cmake/llvm`). Before using
any LLVM build here, run `<build>/bin/llvm-config --version` and compare
`VCSRevision.h` against `git submodule status`; a 3.9 GB directory in the right
place is not evidence.

*Not verified:* the reason given for the LLVM 23 build was that CIRCT requires
LLVM 23, and that CIRCT was removed from `externals/` on 2026-09-18. What is
checkable today: `externals/` on `main` holds only `llvm-project` and
`past-python-bindings`, and **`main` has never tracked `externals/circt`** --
its `.gitmodules` has no such entry and no commit on `main` touches one. CIRCT
is a `chia-codesign` submodule (`git ls-tree chia-codesign externals/` lists it,
and `/home/sk3463/allo-chia-wt/externals/circt` is populated). So there was
nothing tracked on `main` to delete; if a CIRCT tree was removed it was
untracked, and that cannot be confirmed from git. The 2026-09-18 submodule
checkout is likewise inferred from the mtime of `externals/llvm-project/.git`,
not from a reflog.

## Never point one worktree's build at another worktree's tree

cmake preserves mtimes, so a cross-worktree build dependency leaves **nothing
looking stale**. No rebuild is triggered, no warning is printed, and the symptom
surfaces much later as an unexplained ABI or dialect mismatch.

The current arrangement, verified 2026-09-18 -- the second worktree already
does this:

| | `/home/sk3463/allo` (`main`) | `/home/sk3463/allo-chia-wt` (`chia-codesign`) |
| --- | --- | --- |
| `allo/_mlir` | symlink `-> ../mlir/build/tools/allo/_mlir`, **relative, stays inside the worktree** | a **real directory** of installed output, not a symlink |
| extension ABI tag | `_allo.cpython-312-*.so` | `_allo.cpython-314-*.so` |
| runtime soname | `libAlloDataflowRuntime.so.22.0git` | `libAlloDataflowRuntime.so.23.0git` |
| build's `LLVM_DIR` | `/home/sk3463/llvm-allo-6b09f739/build/lib/cmake/llvm` | `/home/sk3463/allo/externals/llvm-project/build/lib/cmake/llvm` -- **the other worktree** |

Worse, individual files inside `/home/sk3463/allo-chia-wt/allo/_mlir` are
absolute symlinks out of the worktree: `ir.py`, `passmanager.py`, `rewrite.py`
and `execution_engine.py` all point into
`/home/sk3463/allo/externals/llvm-project/mlir/python/mlir/`, i.e. into `main`'s
submodule checkout. Only `schedule.py` stays local
(`-> /home/sk3463/allo-chia-wt/mlir/python/allo/schedule.py`).

**This is the mechanism by which the 2026-09-18 mistake above did its damage.**
Checking out a different revision of `main`'s submodule swapped four of
`chia-codesign`'s Python binding files to a different LLVM version, with no
build step, no warning, and nothing in either worktree's `git status` pointing
at it. The failure this produces is an ABI or dialect error that appears to come
from the *other* branch's code.

Rules:

1. A worktree's `allo/_mlir` symlink target must be **relative** and must stay
   inside that worktree. `main`'s is correct; copy that shape.
2. Each worktree gets its own LLVM/MLIR build directory, or they share a
   read-only external one (like `llvm-allo-6b09f739`) that **neither** worktree's
   `externals/` can be checked out from under.
3. Never run a build in worktree A that names a path under worktree B in
   `LLVM_DIR`/`MLIR_DIR`. Checking out a branch in B then silently changes A's
   sources.
4. On any "impossible" ABI or dialect error, first `readlink -f allo/_mlir`,
   then `grep -E '^(LLVM|MLIR)_DIR' <build>/CMakeCache.txt`, and check the
   soname version suffix in `allo/_mlir/_mlir_libs/`. Those three answer it
   faster than any rebuild.
