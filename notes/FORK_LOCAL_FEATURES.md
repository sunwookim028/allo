# Fork-local features to preserve (fork issue #5)

Concrete file inventory for the feature sets that live only on
`sunwookim028/allo` `main` and have no upstream equivalent. The
main<->upstream reconciliation (fork issue #5) must not drop them.

Derived from `git diff upstream/main..main` against upstream baseline
`upstream/main` `437bf53`. Regenerate with:

```
git diff --stat upstream/main..main -- '*.py' '*.cpp' '*.h' '*.td'
```

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

## Other fork-local source deltas

The reconciliation also touches these, which carry a mix of the above plus the
region-scope `@Stateful` and hierarchical-dataflow work; check each diff before
dropping merged-upstream lines:

- `allo/ir/builder.py`, `allo/ir/visitor.py`, `allo/ir/infer.py`,
  `allo/backend/simulator.py`, `allo/backend/vitis.py`, `allo/backend/hls.py`,
  `allo/backend/ip.py`, `allo/passes.py`, `mlir/lib/Translation/Utils.cpp`.

The six files that conflict on a straight `main` <-> `upstream/main` merge are
tracked in fork issue #5.
