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
   `notes/FORK_LOCAL_FEATURES.md` (nb-stream primitives, Catapult nb-stream
   support).
4. Update the affected fork issues (`gh issue list -R sunwookim028/allo`) and
   any relevant `notes/`.
5. If `feature/mesh-accelerator` or `wip/simulator-deep-scan` carried that
   branch as a merge dep, rebase them onto the refreshed `upstream/main` to drop
   the merged-in commits. (Current local branches: `git branch` ->
   `main`, `feature/mesh-accelerator`, `wip/simulator-deep-scan`.)
