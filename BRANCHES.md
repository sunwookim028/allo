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

Posture: fork-first. All work lands on `fork/next`; only generic and clean
changes are PR'd to `cornell-zhang/allo` upstream; mininpu-specific work stays
on the fork.

## Open upstream PRs

| PR | Branch | Status | Blocking notes |
|---|---|---|---|
| [#554](https://github.com/cornell-zhang/allo/pull/554) | `fix/vhls-mlir-percent-alloc-csim` | OPEN | CI green; rebase-clean; re-ping @chhzh123 pending |

**Recently merged:**
- [#577](https://github.com/cornell-zhang/allo/pull/577) (`fix/hierarchical-dataflow-codegen`) merged 2026-05-13 (commit `2211c69`). Local branch deleted; fork copy retained. Post-merge housekeeping done.
- [#579](https://github.com/cornell-zhang/allo/pull/579) (`fix/fp16-hls-half-type`) merged 2026-05-11 (commit `ad8da09`). Branch deleted from local + fork.

**Closed / superseded:** [#578](https://github.com/cornell-zhang/allo/pull/578), [#563](https://github.com/cornell-zhang/allo/pull/563), [#562](https://github.com/cornell-zhang/allo/pull/562) closed, superseded by #577 / #579 (kept here for history).

**Upstream baseline:** `origin/main` advanced to `36bc03e` (#577 `2211c69` + #586 AIE XRT/driver compat). Local `main` fast-forwarded `ad8da09` -> `36bc03e`. Merging `origin/main` into `next` conflicts in 6 files (`allo/backend/simulator.py`, `allo/backend/vitis.py`, `allo/ir/builder.py`, `allo/ir/visitor.py`, `mlir/lib/Translation/EmitVivadoHLS.cpp`, `tests/test_vhls.py`); the merge was aborted and deferred to a dedicated reconciliation session. `next` remains on `3458db1`.

---

## Feature/fix branches not yet PR'd

**Items 5–8 of the upstream queue are deferred** to a fresh session per
the consolidation verdict in `notes/HIERARCHY_DESIGN.md`.
That file frames region-scope `@Stateful`, vhls file-scope statefuls, nested
sub-region streams, and auto-capture `s_axilite` as one structural gap. No
individual upstream issues/PRs should be filed until the fresh session decides
the consolidation form.

| Branch | Description | PR status | Notes |
|---|---|---|---|
| `feature/region-bare-scalar-axilite` | Bare `int32` in `@df.region()` args → `s_axilite` (args=[] path) | Not opened — reference only | Reverted from `next` 2026-05-10 (revert `a7ae144`); conflicts with #577 scalar rejection. Kept as reference only. Redesign targets auto-capture → s_axilite. **Deferred to fresh-session consolidation.** |
| `feature/region-scope-stateful` | Region-scope `@Stateful` shared across inner kernels | Not opened | Pushed to fork/. **Deferred to fresh-session consolidation** (paired with vhls file-scope @Stateful — same root). |
| `feature/nb-streams` | Non-blocking stream primitives (`try_put`/`try_get`/`empty`/`full`) | Not opened | Local-only; vhls scope agreed for upstream, Tapa/Catapult fork-local. **Postponed** until hierarchical items land. |
| `feature/mesh-accelerator-v2` | Tile-based hierarchical dataflow regression tests (`b73b555`); maintained successor, already merged into `next` | Not opened | Local-only; stacked on `feature/nb-streams` — rebase after nb-streams is clean. |
| `feature/mesh-accelerator` | Predecessor line (`06ce561`): 22 unique exploratory commits not in `-v2` (Catapult-v1 end-to-end, mesh-v1, decoupled-mesh perf-eval framework) | Not opened — archival | Intentionally PARKED divergent line, kept for archival, not integrated. `-v2` is the maintained successor. |
| `feature/catapult-review` | Catapult HLS NB stream support + quickstart docs | Not opened — fork-local indefinitely | Pushed to fork/. Not in upstream queue. |
| `fix/simulator-nested-call-streams` | Conservative deep-scan for sub-region calls nested in control flow | Not opened | Pushed to fork/; merged into `next`. **Deferred to fresh-session consolidation** (one of the four hierarchical-dataflow items). |

---

## Dependency graph

```
origin/main  (36bc03e: #577 merged 2026-05-13, #579 2026-05-11, #586 AIE compat)
  ├── fix/vhls-mlir-percent-alloc-csim       (PR #554 - OPEN)          ─┐
  ├── feature/nb-streams                     (fork-local; deferred)     │
  ├── feature/mesh-accelerator-v2            (b73b555; in next)  ────────┘
  ├── feature/mesh-accelerator               (06ce561; parked, archival)
  ├── feature/region-bare-scalar-axilite     (reference - deferred)
  ├── feature/region-scope-stateful          (deferred to consolidation)
  ├── feature/catapult-review                (fork-local indefinitely)
  └── fix/simulator-nested-call-streams      (deferred to consolidation)
```

`feature/nb-streams` and `feature/mesh-accelerator-v2` still carry merge commits
from the now-merged #577 and #579. Rebase them onto the updated `origin/main`
(`36bc03e`) to drop the merged-in commits when the `next` reconciliation happens.

---

## Parked / WIP

| Branch | Reason parked |
|---|---|
| `wip/simulator-deep-scan` | Aggressive deep-scan caused NaN in allo-tpu L2. Conservative fix landed as `fix/simulator-nested-call-streams`. Kept for reference. |

### Stash triage (2026-07-06)

The 3 previously-undocumented stashes were reviewed. Two redundant ones were
dropped (bare-scalar WIP already on `feature/region-bare-scalar-axilite`; a stale
`.claude/scheduled_tasks.lock`). One is retained (`stash@{0}`,
"region-scope-stateful cross-branch edits"): its untracked payload is unlanded
innovation absent from `next` (729 lines: `tests/dataflow/hls_synth_fp16.py`,
`tests/dataflow/test_stream_nb_patterns.py`, `tests/u280_hw_deploy.py`). Land or
discard deliberately later. No more hidden loose ends.

---

## Housekeeping: when a PR merges upstream

1. `git fetch origin && git checkout main && git merge --ff-only origin/main`
2. Delete the merged branch locally and on `fork/`.
3. Rebuild `next`: checkout `next`, drop the stale merge commit for that branch, re-merge the remaining branches.
4. Update this file and `STATE.md`.
5. If `feature/nb-streams` or `feature/mesh-accelerator-v2` carried that branch as a merge dep, rebase them onto the new `origin/main`.
