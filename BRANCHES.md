# Branch bookkeeping — sunwookim028/allo

Single source of truth for open PRs, in-flight feature branches, and
their dependency relationships. Update whenever a branch is created,
a PR is opened/merged, or a dependency changes.

Companion: `STATE.md` — what is implemented and what is planned.

---

## Remotes

| Remote | Points at | Role |
|---|---|---|
| `origin` | `sunwookim028/allo` | The fork. Default push/pull. Holds `main` (default) and all `feature/*`/`fix/*`/`wip/*` branches. |
| `upstream` | `cornell-zhang/allo` | Upstream. Read-only baseline; compare against `upstream/main`. |

## Branch roles

| Branch | Role |
|---|---|
| `main` | Integration HEAD; fork default; home for all fork-local docs; editable install for allo-npu. This IS the working branch. NOT a mirror of upstream. |
| `feature/*`, `fix/*` | One branch ↔ one upstream PR, each based on `upstream/main`. |
| `wip/*` | Known-broken or incomplete work parked for later. |

There is no `next` branch and no local upstream-mirror branch. The former
`next` integration line was renamed to `main` and the old upstream-mirror
`main` was retired (upstream is reached via the `upstream` remote).

---

Posture: fork-first. All work lands on `origin/main`; only generic and clean
changes are PR'd to `cornell-zhang/allo` upstream; mininpu-specific work stays
on the fork.

## Open upstream PRs

| PR | Branch | Status | Blocking notes |
|---|---|---|---|
| [#554](https://github.com/cornell-zhang/allo/pull/554) | `fix/vhls-mlir-percent-alloc-csim` | OPEN | CI green; rebase-clean; re-ping @chhzh123 pending |
| [#593](https://github.com/cornell-zhang/allo/pull/593) | `fix/region-toparg-aliasing` | OPEN | Fixes #592 (region top-arg dedup keyed by bound name). Polished (imports + comment) at `3fd034d`. **Already cherry-picked onto `main` as `effd2bb`.** |

**Recently merged:**
- [#577](https://github.com/cornell-zhang/allo/pull/577) (`fix/hierarchical-dataflow-codegen`) merged 2026-05-13 (commit `2211c69`). Local branch deleted; fork copy retained. Post-merge housekeeping done.
- [#579](https://github.com/cornell-zhang/allo/pull/579) (`fix/fp16-hls-half-type`) merged 2026-05-11 (commit `ad8da09`). Branch deleted from local + fork.

**Closed / superseded:** [#578](https://github.com/cornell-zhang/allo/pull/578), [#563](https://github.com/cornell-zhang/allo/pull/563), [#562](https://github.com/cornell-zhang/allo/pull/562) closed, superseded by #577 / #579 (kept here for history).

**Upstream baseline:** `upstream/main` is at `36bc03e` (#577 `2211c69` + #586 AIE XRT/driver compat). Reconciling `main` with `upstream/main` conflicts in 6 files (`allo/backend/simulator.py`, `allo/backend/vitis.py`, `allo/ir/builder.py`, `allo/ir/visitor.py`, `mlir/lib/Translation/EmitVivadoHLS.cpp`, `tests/test_vhls.py`); that merge is deferred to the main<->upstream reconciliation (fork issue #5). `main` is at `effd2bb` (former `next` tip `d02719e` + the #592 polish cherry-pick).

> **FORK-LOCAL, NOT UPSTREAM — preserve during main<->upstream reconciliation (fork issue #5).**
> Two feature sets live only on `main` and have no upstream equivalent; the
> reconciliation must not drop them:
> 1. **Non-blocking stream primitives** - `try_put`/`try_get`/`empty`/`full`
>    (`allo/ir/types.py`; MLIR `StreamTryGet`/`Put`/`Empty`/`Full` ops;
>    `EmitVivadoHLS.cpp` `emitStreamTry*`).
> 2. **Full Catapult HLS backend** - `allo/backend/catapult.py`,
>    `mlir/lib/Translation/EmitCatapultHLS.cpp` (+ CAPI + bindings),
>    `harness/catapult`, docs, `tests/test_catapult_hls.py`.

**#592 fix state on `main`:** cherry-picked (commit `effd2bb`); the polished
upstream PR is #593 (`fix/region-toparg-aliasing`, `3fd034d`).

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
| `feature/region-scope-stateful` | Region-scope `@Stateful` shared across inner kernels | Not opened | Pushed to origin. **Deferred to fresh-session consolidation** (paired with vhls file-scope @Stateful — same root). |
| `feature/nb-streams` | Non-blocking stream primitives (`try_put`/`try_get`/`empty`/`full`) | Not opened | Local-only; vhls scope agreed for upstream, Tapa/Catapult fork-local. **Postponed** until hierarchical items land. |
| `feature/mesh-accelerator-v2` | Tile-based hierarchical dataflow regression tests (`b73b555`); maintained successor, already merged into `main` | Not opened | Local-only; stacked on `feature/nb-streams` — rebase after nb-streams is clean. |
| `feature/mesh-accelerator` | Predecessor line (`06ce561`): 22 unique exploratory commits not in `-v2` (Catapult-v1 end-to-end, mesh-v1, decoupled-mesh perf-eval framework) | Not opened — archival | Intentionally PARKED divergent line, kept for archival, not integrated. `-v2` is the maintained successor. |
| `feature/catapult-review` | Catapult HLS NB stream support + quickstart docs | Not opened — fork-local indefinitely | Pushed to origin. Not in upstream queue. |
| `fix/simulator-nested-call-streams` | Conservative deep-scan for sub-region calls nested in control flow | Not opened | Pushed to origin; merged into `main`. **Deferred to fresh-session consolidation** (one of the four hierarchical-dataflow items). |

---

## Dependency graph

```
upstream/main  (36bc03e: #577 merged 2026-05-13, #579 2026-05-11, #586 AIE compat)
  ├── fix/vhls-mlir-percent-alloc-csim       (PR #554 - OPEN)          ─┐
  ├── fix/region-toparg-aliasing             (PR #593 - OPEN; on main) │
  ├── feature/nb-streams                     (fork-local; deferred)     │
  ├── feature/mesh-accelerator-v2            (b73b555; in main)  ───────┘
  ├── feature/mesh-accelerator               (06ce561; parked, archival)
  ├── feature/region-bare-scalar-axilite     (reference - deferred)
  ├── feature/region-scope-stateful          (deferred to consolidation)
  ├── feature/catapult-review                (fork-local indefinitely)
  └── fix/simulator-nested-call-streams      (deferred to consolidation)
```

`feature/nb-streams` and `feature/mesh-accelerator-v2` still carry merge commits
from the now-merged #577 and #579. Rebase them onto `upstream/main`
(`36bc03e`) to drop the merged-in commits when the main<->upstream
reconciliation happens.

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
innovation absent from `main` (729 lines: `tests/dataflow/hls_synth_fp16.py`,
`tests/dataflow/test_stream_nb_patterns.py`, `tests/u280_hw_deploy.py`). Land or
discard deliberately later. No more hidden loose ends.

---

## Housekeeping: when a PR merges upstream

1. `git fetch upstream` to refresh `upstream/main`.
2. Delete the merged branch locally and on `origin` (the fork).
3. Reconcile `main` with the merged commit. `main` is not a fast-forward of `upstream/main`, so this goes through the main<->upstream reconciliation (fork issue #5) and MUST preserve the fork-local features listed above (nb-stream primitives, Catapult backend).
4. Update this file and `STATE.md`.
5. If `feature/nb-streams` or `feature/mesh-accelerator-v2` carried that branch as a merge dep, rebase them onto the refreshed `upstream/main`.
