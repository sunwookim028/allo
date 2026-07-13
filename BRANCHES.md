# Branch bookkeeping (pointer)

This file is temporary scratch. Judge live branch / PR state from:

- Local branches and their upstreams: `git branch -vv`, `git worktree list`.
- Open upstream PRs: `gh pr list -R cornell-zhang/allo` (currently #554, #593).
- Open fork work: `gh issue list -R sunwookim028/allo`
  (open: #4, #5, #7-#12; #6 closed).
- Upstream baseline: `git log upstream/main`.
- Merge procedure (when a PR merges upstream):
  `notes/MAINTENANCE_CHECKLIST.md`.
- Fork-local features to preserve during reconciliation:
  `notes/FORK_LOCAL_FEATURES.md` (fork issue #5).
