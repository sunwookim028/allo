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
| `next` | Integration HEAD; fork default; editable install for allo-tpu. Re-merge feature branches as they advance; no rebase/force-push. |
| `feature/*`, `fix/*` | One branch ↔ one upstream PR, each based on `origin/main`. |
| `wip/*` | Known-broken or incomplete work parked for later. |

---

## Open upstream PRs

| PR | Branch | Status | Blocking notes |
|---|---|---|---|
| [#554](https://github.com/cornell-zhang/allo/pull/554) | `fix/vhls-mlir-percent-alloc-csim` | OPEN | Rebased + tests added; awaiting @chhzh123 re-review |
| [#577](https://github.com/cornell-zhang/allo/pull/577) | `fix/hierarchical-dataflow-codegen` | OPEN | Fixes pushed 2026-05-06 (42df618): `move_before` removed; scalar-in-args error in infer.py; OMP regression test added — re-request review |
| [#579](https://github.com/cornell-zhang/allo/pull/579) | `fix/fp16-hls-half-type` | OPEN | Switched to `hls::` for all FP math + fp16 test; awaiting @Fangtangtang ack |

---

## Feature/fix branches not yet PR'd

| Branch | Description | PR status | Notes |
|---|---|---|---|
| `feature/region-bare-scalar-axilite` | Bare `int32` in `@df.region()` args → `s_axilite` | Not opened | Clean; local-only (not pushed to fork/) |
| `feature/region-scope-stateful` | Region-scope `@Stateful` shared across inner kernels | Not opened | Pushed to fork/; verify no rebase needed before PR |
| `feature/nb-streams` | Non-blocking stream primitives (`try_put`/`try_get`/`empty`/`full`) | Not opened | Local-only; merges in #554, #577, #579 — must rebase after those land |
| `feature/mesh-accelerator-v2` | Tile-based hierarchical dataflow regression tests | Not opened | Local-only; stacked on `feature/nb-streams` — rebase after nb-streams is clean |
| `feature/catapult-review` | Catapult HLS NB stream support + quickstart docs | Not opened | Pushed to fork/; rebase check needed before PR |
| `fix/simulator-nested-call-streams` | Conservative deep-scan for sub-region calls nested in control flow | Not opened | Pushed to fork/; merged into `next` |

---

## Dependency graph

```
origin/main
  ├── fix/vhls-mlir-percent-alloc-csim       (PR #554)  ─┐
  ├── fix/hierarchical-dataflow-codegen      (PR #577)  ─┤─→ feature/nb-streams
  ├── fix/fp16-hls-half-type                 (PR #579)  ─┘        │
  │                                                                └─→ feature/mesh-accelerator-v2
  ├── feature/region-bare-scalar-axilite     (standalone)
  ├── feature/region-scope-stateful          (standalone)
  ├── feature/catapult-review                (standalone)
  └── fix/simulator-nested-call-streams      (standalone)
```

`feature/nb-streams` and `feature/mesh-accelerator-v2` carry merge commits from
the three open PRs. When #554, #577, #579 land upstream, these two branches must
be rebased onto the updated `origin/main` (drop the merged-in commits).

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
