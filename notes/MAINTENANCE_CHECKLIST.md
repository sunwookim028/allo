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

## Branch layout (as of 2026-09-07)

| Branch | Lineage | Role |
| --- | --- | --- |
| `main` | cornell-zhang | Fork integration branch: upstream plus fork-local features. **Not** a mirror of upstream — 68 ahead, 8 behind as of this writing. |
| `upstream` | cornell-zhang | Mirror of `upstream/main`, tracking the `upstream` remote. Refresh it to see what has landed; diff `main` against it to see what the fork carries. **Rebasing `main` onto it is never automatic — it is an explicit call.** |
| `chia-codesign` | `kkkaishao/allo` (ACT) | The CHIA / TinyTPU co-design artifact. A separate codebase, not a feature branch — see below. |
| `fix/vhls-mlir-percent-alloc-csim` | cornell-zhang | Live: upstream PR #554, open since Feb 2026. |

Tags, in place of branches that were retired because their history is reachable
elsewhere: `tinytpu-rtlgen-base` (`882f7dd6`, the commit `chia-codesign` forks
from) and `wip-u280-rescue` (`1bd3c5a6`, an unreviewed u280 / nb-stream / fp16
rescue point from 2026-07-06).

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
