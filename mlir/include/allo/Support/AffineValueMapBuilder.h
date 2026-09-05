/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SUPPORT_AFFINE_VALUE_MAP_BUILDER_H
#define ALLO_SUPPORT_AFFINE_VALUE_MAP_BUILDER_H

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::allo {
// Strip away index casts, extension/truncation ops, which do not affect the
// value as an affine expression.
Value stripCast(Value value);

// Incrementally raises raw index-typed SSA values (constants, affine
// dims/symbols, affine.apply, and arith add/sub/mul/div/rem chains) into a
// canonical `affine::AffineValueMap`, so ops outside affine.load/store form can
// be analyzed with affine machinery.
struct AffineValueMapBuilder {
  MLIRContext *ctx;
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> syms;
  SmallVector<Value, 4> results;
  llvm::SmallDenseSet<Value, 4> exprFailureCache;
  SmallVector<AffineExpr, 4> exprs;

  explicit AffineValueMapBuilder(MLIRContext *ctx) : ctx(ctx) {}

  // Import a single value as an affine expression.
  LogicalResult importValue(Value v) {
    auto result = importValueInternal(v);
    if (failed(result))
      return failure();
    exprs.push_back(*result);
    return success();
  }
  // Import an affine map and its operands. If allowMultiResults is false, the
  // map must have exactly one result.
  LogicalResult importMapAndOperands(AffineMap map, ValueRange dims,
                                     ValueRange syms,
                                     bool allowMultiResults = false);
  // Compose the imported expressions and simplify the resulting map.
  affine::AffineValueMap compose() const;
  // Reset internal state to reuse the builder for another map. The failure
  // cache is kept, so a builder must not outlive a rewrite that could make a
  // cached failure importable.
  void reset();
  // Add results to the final value map. Optional when only the composition of
  // the results matters.
  void addResults(ArrayRef<Value> results) {
    llvm::append_range(this->results, results);
  }

private:
  LogicalResult cacheFailure(Value v) {
    exprFailureCache.insert(v);
    return failure();
  }
  FailureOr<AffineExpr> importValueInternal(Value v);
  AffineExpr addDim(Value v);
  AffineExpr addSym(Value v);
};
} // namespace mlir::allo

#endif // ALLO_SUPPORT_AFFINE_VALUE_MAP_BUILDER_H
