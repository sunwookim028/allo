# Fork state — `sunwookim028/allo`

This file lives on the `next` branch (fork default) and tracks what
is implemented and what is planned. For open PRs, branch dependencies,
and housekeeping steps see `BRANCHES.md`. Update this file with any
merge into `next` or any change to planned work. Consumer repo:
`sunwookim028/allo-npu` consumes this fork as its editable Allo install.

## Branches

| Branch | Role | Tracks |
|---|---|---|
| `main` | mirror of upstream `cornell-zhang/allo:main`. Never commit. | `origin/main` |
| `next` | integration HEAD. Pulled by allo-npu. Re-merge as features advance. | `fork/next` |
| `feature/*`, `fix/*` | one branch ↔ one upstream PR, each based on `origin/main` | `fork/<same>` |
| `wip/*` | known-broken or incomplete work parked for later | `fork/<same>` |

## Implemented in `next` (delta vs `origin/main`)

- **Bare-scalar `@df.region()` args → `s_axilite`** — *(reverted from `next`
  2026-05-10; branch kept as reference)* — conflicted with PR #577's scalar
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
  Branch: `fix/hierarchical-dataflow-codegen` (PR #577).
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
  Branch: `fix/simulator-nested-call-streams` (merged into `next`).
- **Mesh-accelerator tile tests** — tile-based hierarchical
  dataflow regression set. Branch: `feature/mesh-accelerator-v2`.
- **Catapult HLS NB stream support** — Catapult backend additions
  + quickstart docs. Branch: `feature/catapult-review`.

## Live upstream PRs (cornell-zhang/allo)

| PR | Branch | State | Notes |
|---|---|---|---|
| #554 | `fix/vhls-mlir-percent-alloc-csim` | OPEN | Rebased + tests added; re-pinged @chhzh123 2026-05-12 |
| #577 | `fix/hierarchical-dataflow-codegen` | OPEN | EmitVivadoHLS.cpp restored to main (b5ee250); pylint fix (eab77a8); CI green; @chhzh123 pre-approved merge; awaiting @Fangtangtang final pass |

#579 merged 2026-05-11 (commit `ad8da09`).

## Known broken / parked

_(none currently — `wip/simulator-deep-scan` was fixed and landed as
`fix/simulator-nested-call-streams`, merged into `next` on 2026-05-01.)_

### Pending investigation

- **L2 NaN** — allo-tpu L2 systolic case may still produce NaN/1e35
  magnitude outputs. The aggressive deep-scan from `feature/nb-streams`
  (commit `8f407bf`) was the suspected cause; replaced by the conservative
  fix. Re-validate allo-tpu L2 baseline after pulling updated `next`.

## Planned (not started)

Sourced from `/work/shared/users/phd/sk3463/projects/ALLO_SHORTCOMINGS.md`.
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
- `/work/shared/users/phd/sk3463/projects/ALLO_HIERARCHY_DESIGN.md` is the pending
  consolidation lens for items 5–8 of the upstream plan (region-scope `@Stateful`,
  vhls file-scope statefuls, nested-call streams, auto-capture s_axilite). Read it
  before pitching maintainers; items are deferred to a fresh session.
- Consumer-side handoff doc is at `/work/shared/users/phd/sk3463/projects/allo-npu/handoff.md`
  (moved out of this repo 2026-05-12).
- Allo and allo-npu are separate sessions. Never edit the other
  repo. See `/work/shared/users/phd/sk3463/projects/ALLO_LESSONS.md`.
- Before claiming a fix is validated: `rm -rf .cache/llvm_sim/`
  in allo-tpu and rebuild. Cache is keyed only on level `tpu.py`
  + top-level `tpu_config.py`.
- One MLIR Context per Python process.
  `LLVM ERROR: Option 'fast' already exists!` means a second
  `customize()` was attempted in the same process.
- Don't commit to `main`. Don't force-push `next` (only re-merge).
