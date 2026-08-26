/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_DEPENDENCEANALYSIS_H
#define ALLO_SCHEDULING_DEPENDENCEANALYSIS_H

#include "circt/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>

namespace mlir::allo {

/// A constant range `[lb, ub]` (inclusive) on an SSA value, either endpoint
/// open when unknown. Distilled from the `allo.assume.ssa` value facts.
struct AssumedRange {
  std::optional<int64_t> lb;
  std::optional<int64_t> ub;
};

/// What one counted loop runs: an iteration count plus whether that count is
/// only an upper bound.
struct LoopTrip {
  /// Iterations, or empty when nothing bounds them.
  std::optional<int64_t> count;
  /// `count` is a worst case from an `allo.assume.ssa` range, not a
  /// compile-time constant, so every span composed from it is a bound, not an
  /// exact cycle count.
  bool bounded = false;
};

/// The dependence distance carried by the counted loop at 1-based nesting depth
/// \p level among a dependence's shared enclosing loops, projected from its
/// components (outermost -> innermost). Sets \p drop when an outer loop already
/// satisfies the dependence by its sequential execution. Sets \p valid = false
/// when \p level is deeper than the shared loop nest. A loop-independent
/// dependence has no components and maps to distance 0.
int64_t
carriedDistanceAtLevel(llvm::ArrayRef<affine::DependenceComponent> comps,
                       unsigned level, bool &drop, bool &valid);

/// Memory + stream dependence analysis over a `func.func`. Affine memref
/// accesses and Allo stream get/put ops are recorded into one
/// MemoryDependenceResult that scheduling problem construction consumes.
class DependenceAnalysis {
public:
  explicit DependenceAnalysis(func::FuncOp funcOp);

  /// Dependences whose destination is \p op (may be empty).
  llvm::ArrayRef<circt::analysis::MemoryDependence>
  getDependences(Operation *op) {
    return results[op];
  }

  /// The constant range a value is known to lie in, distilled from the
  /// `allo.assume.ssa` facts, or nullopt when no such fact constrains it.
  std::optional<AssumedRange> getAssumedRange(Value v) const {
    auto it = assumedRanges.find(v);
    return it == assumedRanges.end() ? std::nullopt : std::optional(it->second);
  }

  /// All distilled value ranges, keyed by SSA value.
  const llvm::DenseMap<Value, AssumedRange> &getAssumedRanges() const {
    return assumedRanges;
  }

  /// Whether the polyhedral test cannot model \p op's access.
  bool isNonPolyhedral(Operation *op) const {
    return nonPolyhedral.contains(op);
  }

  /// Whether the dependences between \p a and \p b come from the polyhedral
  /// test alone: both accesses are in its reach and it decided the pair. A
  /// pair on the conservative path carries blanket distances no consumer may
  /// treat as exact.
  bool isExactPair(Operation *a, Operation *b) const {
    return !nonPolyhedral.contains(a) && !nonPolyhedral.contains(b) &&
           !undecided.contains(a < b ? std::make_pair(a, b)
                                     : std::make_pair(b, a));
  }

  /// What \p loop (an `affine.for` or `scf.for`) runs: its exact count where
  /// that is compile-time, else the worst case its symbolic bounds admit under
  /// the `allo.assume.ssa` ranges, else empty. Memoized.
  LoopTrip tripOf(Operation *loop) const;

  /// How many accesses sit outside the polyhedral test's reach and how many
  /// pairs it accepted but could not decide: the population the conservative
  /// path owns, reported so it is watched rather than rediscovered.
  unsigned conservativeAccesses() const { return nonPolyhedral.size(); }
  unsigned undecidedPairs() const { return undecided.size(); }

private:
  func::FuncOp func;
  circt::analysis::MemoryDependenceResult results;
  llvm::DenseMap<Value, AssumedRange> assumedRanges;
  llvm::SmallDenseSet<Operation *> nonPolyhedral;
  /// Pairs the polyhedral test accepted but could not decide (see isExactPair).
  llvm::DenseSet<std::pair<Operation *, Operation *>> undecided;
  mutable llvm::DenseMap<Operation *, LoopTrip> trips;
};

/// Whether \p op carries a memory effect this analysis does not model
/// (`memref.copy`, `atomic_rmw`, `dma_*`). Such an op joins no access list, so
/// its dependences would be dropped and anything scheduled around it may race;
/// `verify-rtl-legality` rejects one before scheduling. The complement of the
/// access kinds the constructor's walk collects, so edit the two together.
bool isUnmodeledMemoryAccess(Operation *op);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_DEPENDENCEANALYSIS_H
