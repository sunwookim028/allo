/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_REGIONGRAPH_H
#define ALLO_SCHEDULING_REGIONGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::allo {

enum class RegionKind { Loop, StraightLine };

/// A scheduling region: a single region-bearing op, or a maximal run of other
/// ops.
struct SchedRegion {
  RegionKind kind;
  /// Top-level ops of the region; a Loop region holds exactly its loop op.
  SmallVector<Operation *> ops;

  Operation *anchor() const { return ops.front(); }
};

/// Partition a block into scheduling regions (loops + maximal straight-line
/// runs). The scheduler recurses this into imperfect-nest bodies.
///
/// In a nested block (loop/while body, `if` branch), every synchronous
/// sub-kernel call is isolated into its own region: the enclosing container
/// re-runs the block per iteration, gated on the child's real `done`.
///
/// In the function's entry block, only an indeterminate call
/// (`isIndeterminateCall`) is isolated: its completion cycle is data-dependent,
/// so sharing a span would schedule siblings against a meaningless start time.
/// This applies only where the call is a leaf CallUnit; a
/// `composesOnStructuralTop` function wires every call as a concurrent process.
SmallVector<SchedRegion> enumerateRegions(Block &block);

/// Partition `func`'s entry block into scheduling regions (loops + maximal
/// straight-line runs).
SmallVector<SchedRegion> enumerateRegions(func::FuncOp func);

/// What a region's own boundary expression evaluates to: a counted loop's
/// runtime bounds, a guard's predicate. An entry the anchor does not need (a
/// constant bound, an `scf` region carrying its bound as an operand already)
/// stays null.
struct EntryCone {
  Value lower, upper;
  Value predicate;
};

/// \p anchor's boundary values, read off the `allo.volatile` marker that
/// `expand-region-bounds` placed immediately before it. Which slots the marker
/// carries follows from the anchor's own maps, so this takes the values in the
/// order the pass wrote them.
EntryCone entryConeOf(Operation *anchor);

/// A region's controller shape. The reifier reads it to charge a region's
/// boundary cost, the emitter reads it to pick a controller family.
enum class RegionShape {
  /// Runs a schedule itself: an II-paced pipeline or a straight-line
  /// sequential. A `dcp.instance` inside one is a fixed-latency datapath node
  /// (a `CallUnit`), not a child to sequence.
  Leaf,
  /// Drives child regions in its body (a loop wrapping an inner loop, or a
  /// sequential wrapper), one hierarchical pass per outer iteration.
  Container,
  /// Predicates its children: a `dcp.select`, run-once under its `condition`.
  Guard,
  /// Hands off to an instantiated module: a counted loop whose entire body is
  /// one `dcp.instance`, advanced by the child's real `done` rather than a
  /// pipeline cadence. On the instance substrate, so not a `Container`: it has
  /// no child regions.
  CallNode,
};

/// The shape of a reified region op (`dcp.pipeline` / `dcp.sequential` /
/// `dcp.select`), read off its body. Order matters: a select is a guard
/// whichever arms it has, a region holding child regions sequences them, and
/// only then does a lone-instance counted loop become the instance hand-off.
RegionShape dcpRegionShape(Operation *regionOp);

/// The same shape, asked of the source counted loop before its body is
/// materialized. The scheduler, its composer and the reifier read this rather
/// than `dcpRegionShape`, so they agree on which loops sequence children.
///
/// Asked of every loop in a nest, including outer levels of a perfect band:
/// each level above the innermost drives its child as a container.
RegionShape countedLoopShape(LoopLikeOpInterface loop);

/// Whether a straight-line region carries a datapath, i.e. materializes into a
/// `dcp.sequential` at all. A span of nothing but declarations is left in place
/// (`isDeclarationOp`), so it forms no region and occupies no cycle.
bool spanFormsRegion(ArrayRef<Operation *> ops);

/// A declaration: an op that names storage or a literal and binds no hardware.
/// A region of nothing but these carries no datapath, so the reifier leaves it
/// in place, and a level of only these plus child loops has no work to
/// schedule.
bool isDeclarationOp(Operation *op);

/// A synchronous sub-kernel call: a plain (non-async) `func.call`, scheduled as
/// an opaque fixed-latency node. An async call composes structurally as
/// dataflow, ordered by its streams rather than by the schedule.
bool isSyncSubKernelCall(Operation *op);

/// The kernel a `func.call` names, whichever container the phase has: a
/// `func.func` while scheduling, a `dcp.module` once reified. Not filtered by
/// op type, since null reads as an indeterminate callee rather than an error.
Operation *calleeOf(Operation *call);

/// A callee's whole-kernel static latency, from whichever carrier the current
/// phase has: `allo.sched.latency` on a `func.func` while scheduling, the
/// `dcp.module`'s own `latency` once reified. Empty when the callee's length is
/// data-dependent. Reification is post-order over the call graph, so a caller
/// always sees its callee's final reified latency.
std::optional<int64_t> calleeStaticLatency(Operation *callee);

/// A sync call whose callee carries no static latency: its results and its
/// writes land on the child's `done`, at a cycle no static schedule can name.
/// The region partitioner isolates such a call; see `enumerateRegions`.
bool isIndeterminateCall(Operation *op);

/// Whether \p block holds a synchronous sub-kernel call anywhere under it. A
/// `while` body must decompose whenever it does: the flushing-pipeline schedule
/// issues an iteration per cycle, which no re-fired child instance can follow.
bool blockHasSyncCall(Block &block);

/// Whether \p op hands off through a stream anywhere under it, so the emitter
/// wraps it in a stall shell (`HWEmitter::emitRegion`) and back-pressure, not
/// its schedule, decides when it finishes.
///
/// Such a region carries no static span (`composeSpan`): back-pressure can
/// stretch it by an amount no static analysis names, so its whole kernel is
/// indeterminate and callers gate on its real `done`.
///
/// Structural only, not a proof a channel never stalls; keys on the stream ops,
/// which reification keeps verbatim.
bool isElastic(Operation *op);

/// Whether a sync call can be modelled as a leaf CallUnit: every operand is a
/// memref or scalar and every result a scalar. It excludes a stream operand,
/// which is a latency-insensitive hand-off the leaf datapath cannot time.
bool callLowerable(func::CallOp call);

/// Whether \p func composes its children on the structural top rather than the
/// leaf: it has an `await` spawn (async), or wires children through a stream (a
/// plain KPN-style call whose operand is a `Stream`, concurrent even without
/// `await`). Read before reification; `spawnsConcurrently` is the same question
/// asked of one reified child.
bool composesOnStructuralTop(func::FuncOp func);

/// Whether \p invoke is a concurrent child: an `await` spawn, or a call wired
/// to a sibling through a `Stream`. Either way completion is ordered by
/// back-pressure rather than a schedule, making its container a process
/// network. The reified counterpart of `composesOnStructuralTop`.
bool spawnsConcurrently(Operation *invoke);

/// Whether \p op is part of a concurrent container's own structure: the calls
/// it composes, the channels / buffers / constant tables it declares, and the
/// constants feeding them. Everything else in such a container is loose
/// datapath, which `outline-loose-processes` lifts into its own process.
bool isContainerStructure(Operation &op);

/// Whether a counted loop's body decomposes into sub-regions, becoming a
/// sequential wrapper that runs children in program order rather than one flat
/// modulo problem. True when the body nests a loop, or holds a sub-kernel call
/// alongside anything else: a flat modulo schedule has one issue cadence, which
/// a per-iteration child re-fire advancing on that child's `done` cannot share.
bool loopBodyDecomposes(LoopLikeOpInterface loop);

/// Topologically sort the synchronous call graph, as callsites. Fails on a
/// cycle, diagnosed on the callsites that form it. For a consumer that binds
/// per callsite: two calls to one kernel may pass different arrays.
llvm::FailureOr<SmallVector<Operation *>>
buildAndSortCallsiteGraph(func::FuncOp root);

/// The kernels reachable from \p root, callees before callers, with \p root
/// last. One entry per function; external callees are dropped, having no body
/// to work on. This order lets a caller read a fact its callee already
/// published, e.g. the callee latency `isIndeterminateCall` depends on.
llvm::FailureOr<SmallVector<func::FuncOp>>
callGraphPostOrder(func::FuncOp root);
} // namespace mlir::allo

#endif // ALLO_SCHEDULING_REGIONGRAPH_H
