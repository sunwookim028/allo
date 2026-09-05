/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/AffineRaising.h"
#include "allo/TransformOps/AlloTransformOps.h"
#include "allo/TransformOps/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::allo;

DiagnosedSilenceableFailure transform::RaiseToAffineOp::applyToOne(
    TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  std::string reason;

  if (auto forOp = dyn_cast<scf::ForOp>(target)) {
    auto raised = raiseToAffineFor(rewriter, forOp, reason);
    if (failed(raised))
      return emitSilenceableFailure(forOp)
             << "cannot raise to affine.for: " << reason;
    results.push_back(*raised);
    return DiagnosedSilenceableFailure::success();
  }
  if (auto parOp = dyn_cast<scf::ParallelOp>(target)) {
    auto raised = raiseToAffineParallel(rewriter, parOp, reason);
    if (failed(raised))
      return emitSilenceableFailure(parOp)
             << "cannot raise to affine.parallel: " << reason;
    results.push_back(*raised);
    return DiagnosedSilenceableFailure::success();
  }
  if (isa<affine::AffineForOp, affine::AffineParallelOp>(target)) {
    // Already an affine loop, so only the accesses inside it are left to raise.
    raiseAffineAccesses(rewriter, target);
    results.push_back(target);
    return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableFailure(target)
         << "expected scf.for or scf.parallel, but got " << target->getName();
}
