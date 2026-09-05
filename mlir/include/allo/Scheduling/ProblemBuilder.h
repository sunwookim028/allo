/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_PROBLEMBUILDER_H
#define ALLO_SCHEDULING_PROBLEMBUILDER_H

#include "allo/Scheduling/DependenceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::allo {

class OperatorLibrary;
struct DeviceModel;

/// Build a cyclic scheduling problem for one counted loop (`affine.for` or
/// `scf.for`): its body ops, their dependences with inter-iteration distances,
/// and its iter_arg recurrences. \p ProblemT is a `CyclicProblem` subclass.
template <class ProblemT>
ProblemT buildCyclicProblem(LoopLikeOpInterface loop, DependenceAnalysis &deps);

/// The operation defining the value carried into iter_arg \p iterArg of the
/// counted loop with body \p body and terminator \p yield, and how many
/// iterations back it sits: 1 for a direct recurrence, P for a P-slot rotated
/// accumulator, following any chain of iter_arg-to-iter_arg shifts.
/// `{nullptr, 0}` where there is no such definer.
std::pair<Operation *, unsigned> iterArgSource(Block *body, Operation *yield,
                                               unsigned iterArg);

/// Whether an `scf.while` forwards all before-args to the after region 1:1
/// (identity forwarding, equal arity), the shape `buildWhileProblem` schedules,
/// which aligns inits/before-args/after-args/yield/results by one slot index.
bool whileHasIdentityForwarding(scf::WhileOp w);

/// Whether an `scf.while`'s continue-condition settles the cycle the loop
/// issues, so the while can flushing-pipeline. False when the before region
/// holds a multi-cycle op per \p lib (a memory read, a latency IP) or a
/// sub-kernel call, whose length is its callee's schedule and no row \p lib can
/// answer for. Either routes the while to the sequential check/run controller.
bool conditionIsCombinational(scf::WhileOp w, const DeviceModel &dev);

/// Whether \p w takes the flushing-pipeline schedule rather than decomposing
/// into sub-regions run in program order: it nests no loop (whose per-iteration
/// length is data-dependent), its condition is combinational, and its body
/// holds no sub-kernel call. Only a while on this path must forward its
/// loop-carried values 1:1.
bool whileFlushingPipelines(scf::WhileOp w, const DeviceModel &dev);

/// Build a cyclic scheduling problem for an uncounted `scf.while`, its before
/// and after regions scheduled as one iteration: both regions' ops and deps,
/// the non-speculative condition gate, the state recurrence at distance 1, and
/// a side-effect anchor before `scf.yield`. Requires
/// `whileHasIdentityForwarding(w)`.
template <class ProblemT>
ProblemT buildWhileProblem(scf::WhileOp w, DependenceAnalysis &deps);

/// Build an acyclic scheduling problem for a straight-line region (the
/// top-level \p ops of a maximal non-loop run): the ops and their intra-span
/// dependences, with the last program-order op as the unique sink, so that
/// minimizing it schedules the span ASAP.
template <class ProblemT>
ProblemT buildAcyclicProblem(ArrayRef<Operation *> ops,
                             DependenceAnalysis &deps);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_PROBLEMBUILDER_H
