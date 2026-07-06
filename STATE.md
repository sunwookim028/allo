# Fork state — `sunwookim028/allo`

This file lives on the `main` branch (fork default) and tracks what
is implemented and what is planned. For open PRs, branch dependencies,
and housekeeping steps see `BRANCHES.md`. Update this file with any
merge into `main` or any change to planned work. Consumer repo:
`sunwookim028/allo-npu` consumes this fork as its editable Allo install.

## Remotes and branches

Remotes: `origin` = `sunwookim028/allo` (the fork), `upstream` =
`cornell-zhang/allo`. Upstream is a read-only baseline reached via
`upstream/main`; there is no local mirror branch and no `next` branch.

| Branch | Role | Tracks |
|---|---|---|
| `main` | integration HEAD and fork default. The working branch; pulled by allo-npu. NOT a mirror of upstream. | `origin/main` |
| `feature/*`, `fix/*` | one branch ↔ one upstream PR, each based on `upstream/main` | `origin/<same>` |
| `wip/*` | known-broken or incomplete work parked for later | `origin/<same>` |

## Implemented in `main` (delta vs `upstream/main`)

- **Bare-scalar `@df.region()` args → `s_axilite`** — *(reverted from `next`
  2026-05-10 via revert `a7ae144`; branch `feature/region-bare-scalar-axilite`
  kept as reference only)* — conflicted with PR #577's scalar
  rejection (`42df618`). Redesign planned on auto-capture path: auto-captured
  scalars in a `@df.region` should emit `s_axilite` in vitis_hls instead of
  m_axi. Upstream issue TBD. Use `int32[1]` (m_axi) in the meantime.
- **Region-scope `@ Stateful`** — propagates `stateful_var_map` /
  `stateful_counter` through `ASTContext.copy()`, anchors
  `memref.get_global` at each function's entry block. Lets a
  buffer declared at `@df.region` body scope be shared across
  every `@df.kernel` in the region (Gemmini-style decoder + driver
  split). Branch: `feature/region-scope-stateful`.
- **Hierarchical dataflow simulator deadlock fix** — recursive
  OMP parallel-section injection per function, not just the top.
  *Merged upstream as PR #577 (commit `2211c69`, 2026-05-13).*
  Branch `fix/hierarchical-dataflow-codegen` deleted from local
  (fork copy retained). Superseded the earlier closed attempts
  PR #578 / #563 / #562.
- **fp16 (half) HLS support** — `f16 → half` ctype mapping, fp16
  scalar math dispatch, `hls::xxx` qualified math ops uniformly.
  *Merged upstream as PR #579 (commit `ad8da09`, 2026-05-11).*
  Branch `fix/fp16-hls-half-type` deleted from local + fork.
- **VHLS csim emitter cleanup** — `%alloc` MLIR-name stripping in
  postprocess; `ap_int<N>`/`ap_uint<N>` resolution in nanobind
  wrapper. Branch: `fix/vhls-mlir-percent-alloc-csim` (PR #554).
- **Non-blocking stream primitives** — `try_put`/`try_get`/
  `empty`/`full` for VHLS and Tapa. Branch: `feature/nb-streams`.
- **Nested-subregion stream lowering** — conservative deep-scan in
  `_process_function_streams` so callees nested inside `affine.for` /
  `scf.if` have their streams lowered before LLVM conversion. Does NOT
  add nested calls to `pe_call_define_ops` (avoids OMP over-wrap).
  Branch: `fix/simulator-nested-call-streams` (merged into `main`).
- **Mesh-accelerator tile tests** — tile-based hierarchical
  dataflow regression set. Branch: `feature/mesh-accelerator-v2`
  (`b73b555`), the maintained successor, already merged into `main`.
  Predecessor `feature/mesh-accelerator` (`06ce561`) is an
  intentionally-parked divergent line (22 unique exploratory commits
  not in v2: Catapult-v1 end-to-end, mesh-v1, decoupled-mesh
  perf-eval framework), kept for archival, not integrated.
- **Catapult HLS NB stream support** — Catapult backend additions
  + quickstart docs. Branch: `feature/catapult-review`.

## Live upstream PRs (cornell-zhang/allo)

Posture: fork-first. All work lands on `origin/main`; only generic and
clean changes are PR'd to `cornell-zhang/allo` upstream; mininpu-specific
work stays on the fork.

| PR | Branch | State | Notes |
|---|---|---|---|
| #554 | `fix/vhls-mlir-percent-alloc-csim` | OPEN | Rebased + tests added; CI green; rebase-clean; re-ping @chhzh123 pending |
| #593 | `fix/region-toparg-aliasing` | OPEN | Fixes #592 (region top-arg dedup by bound name). Polished at `3fd034d`; **cherry-picked onto `main` as `effd2bb`**. |

Merged/closed upstream:

- **#577** (`fix/hierarchical-dataflow-codegen`) merged 2026-05-13 (commit `2211c69`).
  Local branch deleted; fork copy retained.
- **#579** (`fix/fp16-hls-half-type`) merged 2026-05-11 (commit `ad8da09`).
- **#578 / #563 / #562** closed, superseded by #577 / #579 (kept here for history).

Upstream baseline: `upstream/main` is at `36bc03e` (adds #577 `2211c69`
plus #586 AIE-backend XRT/driver compat). Reconciling `main` with
`upstream/main` conflicts in 6 files (`allo/backend/simulator.py`,
`allo/backend/vitis.py`, `allo/ir/builder.py`, `allo/ir/visitor.py`,
`mlir/lib/Translation/EmitVivadoHLS.cpp`, `tests/test_vhls.py`) - the fork's
reverted/redesigned lines diverge from the merged-upstream #577 form - so the
reconciliation is deferred to fork issue #5. `main` is at `effd2bb` (former
`next` tip `d02719e` plus the #592 polish cherry-pick).

### FORK-LOCAL features (NOT upstream) - preserve during reconciliation (fork issue #5)

These live only on `main` and have no upstream equivalent. The
main<->upstream reconciliation must not drop them:

1. **Non-blocking stream primitives** - `try_put`/`try_get`/`empty`/`full`
   (`allo/ir/types.py`; MLIR `StreamTryGet`/`Put`/`Empty`/`Full` ops;
   `EmitVivadoHLS.cpp` `emitStreamTry*`).
2. **Full Catapult HLS backend** - `allo/backend/catapult.py`,
   `mlir/lib/Translation/EmitCatapultHLS.cpp` (+ CAPI + bindings),
   `harness/catapult`, docs, `tests/test_catapult_hls.py`.

**#592 fix state on `main`:** cherry-picked (commit `effd2bb`); upstream PR
#593 (`fix/region-toparg-aliasing`, `3fd034d`).

## Known broken / parked

_(none currently — `wip/simulator-deep-scan` was fixed and landed as
`fix/simulator-nested-call-streams`, merged into `main` on 2026-05-01.)_

### Pending investigation

- **L2 NaN** — allo-tpu L2 systolic case may still produce NaN/1e35
  magnitude outputs. The aggressive deep-scan from `feature/nb-streams`
  (commit `8f407bf`) was the suspected cause; replaced by the conservative
  fix. Re-validate allo-tpu L2 baseline after pulling updated `main`.

## Planned (not started)

Sourced from `notes/ALLO_SHORTCOMINGS.md`.
Items #1, #3 are partially handled above.

| # | Item | Priority |
|---|---|---|
| 2 | `@ Stateful` inside `@df.kernel` bodies | High — currently forces 8 single-element region-scope arrays in L2 decoder |
| 4 | `math.exp` / `math.log` recognized in `@df.kernel` (currently `KeyError`) | Low — `allo.exp` works |
| 5 | Region-param vs kernel-local name shadowing diagnostic | Medium — silent error |
| 6 | Local `int32` decls in `elif` branches don't dominate uses | Medium — bloats decoder |
| 7 | Bitwise `&`/`\|` operators in expression DSL | Low — arithmetic emulation works |
| 8 | Multiple sub-region call sites under mutually-exclusive branches | Medium — forces unnatural code structure |
| 9 | Sim-cache hashing covers imported helpers | High (allo-tpu side) — repeated stale-cache "validated" |
| 10 | Source-position attribution from MLIR errors back to user `tpu.py` | Low |

## Conventions for agents working here

- See `BRANCHES.md` for open PR status, not-yet-PR branches, and dependency graph.
- `notes/HIERARCHY_DESIGN.md` (moved into the repo from the former
  `ALLO_HIERARCHY_DESIGN.md`) is the pending consolidation lens for items 5–8
  of the upstream plan (region-scope `@Stateful`, vhls file-scope statefuls,
  nested-call streams, auto-capture s_axilite). Read it before pitching
  maintainers; items are deferred to a fresh session.
- Stash triage (2026-07-06): the 3 previously-undocumented stashes were reviewed.
  Two redundant ones were dropped (bare-scalar WIP already on
  `feature/region-bare-scalar-axilite`; a stale `.claude/scheduled_tasks.lock`).
  One is retained (`stash@{0}`, "region-scope-stateful cross-branch edits") because
  its untracked payload holds unlanded innovation absent from `main`: 729 lines
  across `tests/dataflow/hls_synth_fp16.py`, `tests/dataflow/test_stream_nb_patterns.py`,
  and `tests/u280_hw_deploy.py`. Land or discard deliberately in a later session.
- Consumer-side handoff doc is at `/work/shared/users/phd/sk3463/projects/allo-npu/handoff.md`
  (moved out of this repo 2026-05-12).
- Allo and allo-npu are separate sessions. Never edit the other
  repo. See `notes/ALLO_LESSONS.md`.
- Before claiming a fix is validated: `rm -rf .cache/llvm_sim/`
  in allo-tpu and rebuild. Cache is keyed only on level `tpu.py`
  + top-level `tpu_config.py`.
- One MLIR Context per Python process.
  `LLVM ERROR: Option 'fast' already exists!` means a second
  `customize()` was attempted in the same process.
- `main` is the working/integration branch: commit fork-local work here.
  Do not force-push `main`; integrate feature branches by merge. Never
  commit fork-local docs onto `feature/*`/`fix/*` branches bound for upstream.
