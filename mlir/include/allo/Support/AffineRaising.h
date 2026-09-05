/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SUPPORT_AFFINE_RAISING_H
#define ALLO_SUPPORT_AFFINE_RAISING_H

#include "allo/Support/AffineValueMapBuilder.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include <string>

namespace mlir::allo {

// Raising ops into affine form, which is what puts them inside the polyhedral
// dependence test. An op is raised only when the affine form computes what the
// original computed.
//
// Each entry point builds its own `AffineValueMapBuilder`: the builder keeps an
// import FAILURE across `reset`, while raising a loop is exactly what turns a
// value that could not be imported into one that can.

/// Raise a `memref.load` / `memref.store` whose subscripts are affine functions
/// of the enclosing induction variables and loop-invariant values. Fails,
/// changing nothing, when a subscript is not such a function.
LogicalResult raiseAffineAccess(RewriterBase &rewriter, Operation *op);

/// Raise every access under \p root that `raiseAffineAccess` accepts, returning
/// how many.
unsigned raiseAffineAccesses(RewriterBase &rewriter, Operation *root);

/// The affine bound \p root states, expanding the max (for a lower bound) or
/// min (for an upper bound) it may be built from, in `arith`, `affine` or
/// select-of-compare form.
FailureOr<affine::AffineValueMap> matchAffineBound(Value root,
                                                   bool isLowerBound);

/// Raise \p forOp to an `affine.for` with the same body, bounds, step and
/// iter_args, and raise the accesses inside it. \p reason names what stopped it
/// on failure, in which case nothing is changed.
FailureOr<affine::AffineForOp>
raiseToAffineFor(RewriterBase &rewriter, scf::ForOp forOp, std::string &reason);

/// Raise \p parOp to an `affine.parallel` with the same body, per-dimension
/// bounds and steps, and raise the accesses inside it. A reduction is not
/// modeled. \p reason names what stopped it on failure.
FailureOr<affine::AffineParallelOp>
raiseToAffineParallel(RewriterBase &rewriter, scf::ParallelOp parOp,
                      std::string &reason);

} // namespace mlir::allo

#endif // ALLO_SUPPORT_AFFINE_RAISING_H
