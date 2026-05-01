# Fork state — `sunwookim028/allo`

This file lives on the `next` branch (fork default) and is the
single source of truth for what is implemented, what is planned,
and which branches and PRs are live. Update it with any merge into
`next` or any PR open/close. Companion repo: `sunwookim028/allo-tpu`
consumes this fork as its editable Allo install.

## Branches

| Branch | Role | Tracks |
|---|---|---|
| `main` | mirror of upstream `cornell-zhang/allo:main`. Never commit. | `origin/main` |
| `next` | integration HEAD. Pulled by allo-tpu. Re-merge as features advance. | `fork/next` |
| `feature/*`, `fix/*` | one branch ↔ one upstream PR, each based on `origin/main` | `fork/<same>` |
| `fork-mgmt` | fork-only files (HANDOFF, gitignore, this STATE.md). Rebase onto `origin/main`; never upstream. | `fork/fork-mgmt` |
| `wip/*` | known-broken or incomplete work parked for later | `fork/<same>` |

## Implemented in `next` (delta vs `origin/main`)

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
  Branch: `fix/fp16-hls-half-type` (PR #579).
- **VHLS csim emitter cleanup** — `%alloc` MLIR-name stripping in
  postprocess; `ap_int<N>`/`ap_uint<N>` resolution in nanobind
  wrapper. Branch: `fix/vhls-mlir-percent-alloc-csim` (PR #554).
- **Non-blocking stream primitives** — `try_put`/`try_get`/
  `empty`/`full` for VHLS and Tapa. Branch: `feature/nb-streams`.
  **⚠ MLIR `.so` not yet rebuilt with new ops; `StreamTryGetOp` etc.
  absent from compiled extension → `dataflow.py` refs crash at import.**
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
| #554 | `fix/vhls-mlir-percent-alloc-csim` | OPEN | Rebased + tests added; awaiting @chhzh123 re-review |
| #577 | `fix/hierarchical-dataflow-codegen` | OPEN | Review fixes pushed (7a3212c); 2 questions out to @Fangtangtang (`move_before` vs #557; scalar-in-args type-infer) |
| #579 | `fix/fp16-hls-half-type` | OPEN | Switched to `hls::` for all FP math + fp16 test added; awaiting @Fangtangtang ack |

A scheduled remote agent (`trig_01GAZWpbV4Qq8BREhRWF62L2`) checks
PR state on 2026-05-08 09:00 ET.

## Known broken / parked

- **`wip/simulator-deep-scan`** — extends
  `_process_function_streams` with a recursive `func_d.CallOp`
  scan (so PE callees nested inside `affine.for`/`affine.if` get
  visited). Builds successfully against allo-tpu L2 but produces
  NaN / 1e35-magnitude outputs — likely double-processing of stream
  args. **Blocks allo-tpu L2 baseline.** See `HANDOFF.md` on
  `fork-mgmt` for diagnosis notes. **Owner: next Allo session.**

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

- Read `fork-mgmt:HANDOFF.md` before resuming WIP simulator work.
- Allo and allo-tpu are separate sessions. Never edit the other
  repo. See `/work/shared/users/phd/sk3463/projects/ALLO_LESSONS.md`.
- Before claiming a fix is validated: `rm -rf .cache/llvm_sim/`
  in allo-tpu and rebuild. Cache is keyed only on level `tpu.py`
  + top-level `tpu_config.py`.
- One MLIR Context per Python process.
  `LLVM ERROR: Option 'fast' already exists!` means a second
  `customize()` was attempted in the same process.
- Don't commit to `main`. Don't force-push `next` (only re-merge).
