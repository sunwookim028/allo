# Fork-local features to preserve (fork issue #5)

Concrete file inventory for the feature sets that live only on
`sunwookim028/allo` `main` and have no upstream equivalent. The
main<->upstream reconciliation (fork issue #5) must not drop them.

Derived from `git diff upstream/main..main` against upstream baseline
`upstream/main` `437bf53` (current `upstream/main` tip, now an ancestor of
`main`). Regenerate with:

```
git diff --stat upstream/main..main -- '*.py' '*.cpp' '*.h' '*.td'
```

## Reconciliation status

As of merge `ffa2d0c` (`chore(main): reconcile with upstream/main`), upstream
PRs #594, #586, #577, and #579 are absorbed and `main` is **0 behind**
`upstream/main` (`git rev-list --left-right --count main...upstream/main` -> `61	0`).
The merge kept the fork-local `allo/ir/visitor.py` stateful-propagation block
byte-identical to its pre-merge state. The files that used to conflict on a
straight `main` <-> `upstream/main` merge no longer conflict; the residual
`upstream/main..main` diff below is the genuine fork-local delta, not
unmerged-upstream drift.

## 1. Non-blocking stream primitives

DSL methods `try_put` / `try_get` / `empty` / `full` on streams, the MLIR
ops that back them, and their VHLS / Catapult / simulator lowering. This
whole set is fork-local.

MLIR ops: `stream_try_get`, `stream_try_put`, `stream_empty`, `stream_full`
(`StreamTryGetOp` / `StreamTryPutOp` / `StreamEmptyOp` / `StreamFullOp`).

- `allo/ir/types.py` - `put` / `get` / `try_put` / `try_get` / `empty` / `full`
  DSL methods on stream types.
- `allo/ir/builder.py` - build the `StreamTry*` ops from the AST.
- `allo/dataflow.py` - dataflow wiring for the primitives.
- `allo/backend/simulator.py` - simulator support for the ops.
- `mlir/include/allo/Dialect/AlloOps.td` - `StreamTryGetOp`, `StreamTryPutOp`,
  `StreamEmptyOp`, `StreamFullOp` op definitions.
- `mlir/include/allo/Dialect/Visitor.h` - visitor dispatch for the new ops.
- `mlir/include/allo/Translation/EmitBaseHLS.h`,
  `mlir/include/allo/Translation/EmitVivadoHLS.h` - `emitStreamTry*` declarations.
- `mlir/lib/Translation/EmitVivadoHLS.cpp` - `emitStreamTryGet` /
  `emitStreamTryPut` / `emitStreamEmpty` / `emitStreamFull` for VHLS.
- `docs/source/backends/nonblocking_streams.rst` - user documentation.

Tests (all fork-local, added files): `tests/dataflow/test_stream_nb_simple.py`,
`tests/dataflow/test_stream_ops_ir.py`, `tests/dataflow/test_stream_ops_sim.py`,
`tests/dataflow/test_stream_ops_hls.py`, `tests/dataflow/hls_synth_streams.py`,
`tests/test_backend_utils.py`.

## 2. Catapult HLS non-blocking stream support

NOTE: the Catapult backend itself already EXISTS upstream
(`allo/backend/catapult.py`, `mlir/lib/Translation/EmitCatapultHLS.cpp` and
`mlir/include/allo/Translation/EmitCatapultHLS.h`). The fork-local delta is the
non-blocking-stream emitters added to it, not a whole new backend.

- `allo/backend/catapult.py` - modified (nb-stream path).
- `mlir/lib/Translation/EmitCatapultHLS.cpp` - modified: `emitStreamTryGet` /
  `emitStreamTryPut` / `emitStreamEmpty` / `emitStreamFull`. Uses
  `ac_channel::available()` (the synthesizable subset lacks `.empty()`,
  EDG CIN-59).
- `notes/CATAPULT_QUICKSTART.md` - quickstart doc.

Byte-identical to upstream (no fork change): `EmitCatapultHLS.h`,
`tests/test_catapult_hls.py`.

## 3. Region-scoped `@Stateful`, hierarchical dataflow, mesh accelerator

Region-scoped stateful propagation, hierarchical / nested-subregion dataflow,
and the mesh-accelerator experiments. Source support is shared with the deltas
listed in section 4; the tests below are fork-local added files.

Tests (all fork-local, added files):
`tests/dataflow/test_region_stateful.py`,
`tests/dataflow/test_region_toparg_aliasing.py`,
`tests/dataflow/test_nested_subregion_streams.py`,
`tests/dataflow/test_hierachical_mesh.py`,
`tests/dataflow/test_decoupled_mesh.py`,
`tests/dataflow/hls_synth_decoupled.py`,
`tests/dataflow/mesh_perf.py`.

## 4. Other fork-local source deltas

The reconciliation also touches these, which carry a mix of the above plus the
region-scope `@Stateful` and hierarchical-dataflow work; check each diff before
dropping any lines:

- `allo/ir/builder.py`, `allo/ir/visitor.py`, `allo/ir/infer.py`,
  `allo/backend/simulator.py`, `allo/backend/vitis.py`, `allo/backend/hls.py`,
  `allo/backend/ip.py`, `allo/passes.py`, `mlir/lib/Translation/Utils.cpp`.

Notable single-purpose deltas within the above:

- `allo/backend/vitis.py` - `postprocess_hls_code` strips the MLIR SSA-name
  `%` sigil (`re.sub(r"%(\w)", r"\1", ...)`) from emitted VHLS.
- `allo/ir/visitor.py` - region-scoped stateful propagation (kept byte-identical
  across the `ffa2d0c` reconciliation).

All 18 modified source files (`.py`/`.cpp`/`.h`/`.td`) and 13 added test files
above appear in the current `git diff --stat upstream/main..main`; none of them
have been byte-aligned back to upstream by the #577/#579 squash-merge. The only
file that WAS previously a fork-local diff and is now byte-identical to upstream
is `tests/test_vhls.py` (no longer in the diff).
