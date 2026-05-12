# Branch bookkeeping — sunwookim028/allo

Single source of truth for open PRs, in-flight feature branches, and
their dependency relationships. Update whenever a branch is created,
a PR is opened/merged, or a dependency changes.

Companion: `STATE.md` — what is implemented and what is planned.

---

## Branch roles

| Branch | Role |
|---|---|
| `main` | Mirror of `cornell-zhang/allo:main`. Never commit here. |
| `next` | Integration HEAD; fork default; editable install for allo-npu. Re-merge feature branches as they advance; no rebase/force-push. |
| `feature/*`, `fix/*` | One branch ↔ one upstream PR, each based on `origin/main`. |
| `wip/*` | Known-broken or incomplete work parked for later. |

---

## Open upstream PRs

| PR | Branch | Status | Blocking notes |
|---|---|---|---|
| [#554](https://github.com/cornell-zhang/allo/pull/554) | `fix/vhls-mlir-percent-alloc-csim` | OPEN | Re-pinged @chhzh123 2026-05-12; CI green |
| [#577](https://github.com/cornell-zhang/allo/pull/577) | `fix/hierarchical-dataflow-codegen` | OPEN | `EmitVivadoHLS.cpp` restored to upstream (b5ee250); pylint fix (eab77a8); CI green; @chhzh123 pre-approved; awaiting @Fangtangtang final pass |

**Recently merged:** [#579](https://github.com/cornell-zhang/allo/pull/579) (`fix/fp16-hls-half-type`) merged 2026-05-11 (commit `ad8da09`). Branch deleted from local + fork.

---

## Feature/fix branches not yet PR'd

**Items 5–8 of the upstream queue are deferred** to a fresh session per
the consolidation verdict in `/work/shared/users/phd/sk3463/projects/ALLO_HIERARCHY_DESIGN.md`.
That file frames region-scope `@Stateful`, vhls file-scope statefuls, nested
sub-region streams, and auto-capture `s_axilite` as one structural gap. No
individual upstream issues/PRs should be filed until the fresh session decides
the consolidation form.

| Branch | Description | PR status | Notes |
|---|---|---|---|
| `feature/region-bare-scalar-axilite` | Bare `int32` in `@df.region()` args → `s_axilite` (args=[] path) | Not opened — reference only | Reverted from `next` 2026-05-10; conflicts with #577 scalar rejection. Redesign targets auto-capture → s_axilite. **Deferred to fresh-session consolidation.** |
| `feature/region-scope-stateful` | Region-scope `@Stateful` shared across inner kernels | Not opened | Pushed to fork/. **Deferred to fresh-session consolidation** (paired with vhls file-scope @Stateful — same root). |
| `feature/nb-streams` | Non-blocking stream primitives (`try_put`/`try_get`/`empty`/`full`) | Not opened | Local-only; vhls scope agreed for upstream, Tapa/Catapult fork-local. **Postponed** until hierarchical items land. |
| `feature/mesh-accelerator-v2` | Tile-based hierarchical dataflow regression tests | Not opened | Local-only; stacked on `feature/nb-streams` — rebase after nb-streams is clean. |
| `feature/catapult-review` | Catapult HLS NB stream support + quickstart docs | Not opened — fork-local indefinitely | Pushed to fork/. Not in upstream queue. |
| `fix/simulator-nested-call-streams` | Conservative deep-scan for sub-region calls nested in control flow | Not opened | Pushed to fork/; merged into `next`. **Deferred to fresh-session consolidation** (one of the four hierarchical-dataflow items). |

---

## Dependency graph

```
origin/main  (#579 merged 2026-05-11)
  ├── fix/vhls-mlir-percent-alloc-csim       (PR #554)  ─┐
  ├── fix/hierarchical-dataflow-codegen      (PR #577)  ─┤─→ feature/nb-streams
  │                                                       │
  │                                                       └─→ feature/mesh-accelerator-v2
  ├── feature/region-bare-scalar-axilite     (reference — deferred)
  ├── feature/region-scope-stateful          (deferred to consolidation)
  ├── feature/catapult-review                (fork-local indefinitely)
  └── fix/simulator-nested-call-streams      (deferred to consolidation)
```

`feature/nb-streams` and `feature/mesh-accelerator-v2` still carry merge commits
from PR #577 and the (now-merged) #579. When #554 / #577 land, rebase them onto
the updated `origin/main` to drop the merged-in commits.

---

## Parked / WIP

| Branch | Reason parked |
|---|---|
| `wip/simulator-deep-scan` | Aggressive deep-scan caused NaN in allo-tpu L2. Conservative fix landed as `fix/simulator-nested-call-streams`. Kept for reference. |

---

## Housekeeping: when a PR merges upstream

1. `git fetch origin && git checkout main && git merge --ff-only origin/main`
2. Delete the merged branch locally and on `fork/`.
3. Rebuild `next`: checkout `next`, drop the stale merge commit for that branch, re-merge the remaining branches.
4. Update this file and `STATE.md`.
5. If `feature/nb-streams` or `feature/mesh-accelerator-v2` carried that branch as a merge dep, rebase them onto the new `origin/main`.
