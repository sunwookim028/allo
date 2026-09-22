# Uncommitted work, snapshotted 2026-09-22 (second pass)

Insurance only, taken before an expected session interruption. **Committed work
is on `origin`** — every branch was verified pushed, and every per-worktree ref
is contained in a named branch that is on the remote. Look there first.

This holds only what was *not* committed at the moment of the snapshot: the
tracked-file diff from each live worktree, and a list of its untracked files.
A resuming agent's own worktree is authoritative; this is stale the moment it
moves. Delete the branch once those worktrees have landed.

| worktree | branch | mid-flight on |
| --- | --- | --- |
| `allo-codesign` | `codesign-loop` | the two-worker design-point search |
| `allo-spec` | `isa-spec` | the two-ceiling reconciliation |
| `wt-big-shapes` | `big-shapes` | pricing the encoding margins |
| `agent-a8259683…` | `perf-deficit` | attributing the residual deficit |
