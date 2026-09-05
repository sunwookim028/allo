/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/AffineRaising.h"
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::allo {
#define GEN_PASS_DEF_RAISETOAFFINEPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

struct RaiseToAffinePass
    : public allo::impl::RaiseToAffinePassBase<RaiseToAffinePass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func.getContext());

    // Outermost first, because raising a loop is what makes its induction
    // variable a valid affine dim, and that is a precondition for raising
    // anything nested under it: a triangular nest (`for k in range(i, j)`)
    // takes the enclosing loop's variable as its own bound. `walk` defaults to
    // POST-order, which is exactly the wrong direction. The ops nested in a
    // raised loop are MOVED rather than cloned, so their addresses survive.
    SmallVector<scf::ForOp> loops;
    func.walk<WalkOrder::PreOrder>(
        [&](scf::ForOp loop) { loops.push_back(loop); });

    unsigned raisedLoops = 0;
    std::string reason;
    for (scf::ForOp loop : loops) {
      if (succeeded(raiseToAffineFor(rewriter, loop, reason)))
        ++raisedLoops;
      else
        debug(Stage::Prep, loop) << "Left as scf.for: " << reason;
    }

    // Whatever is left sits outside any loop this pass could raise.
    unsigned raisedAccesses = raiseAffineAccesses(rewriter, func);
    if (!raisedLoops && !raisedAccesses)
      return;
    info(Stage::Prep, func)
        << "Raised " << raisedLoops << " loop(s) and " << raisedAccesses
        << " further memref access(es) to affine form, so the polyhedral "
           "dependence test decides them rather than the conservative fallback";
  }
};

} // namespace
