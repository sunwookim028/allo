/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"
#include "allo/Transforms/ReductionUtils.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_REASSOCIATEREDUCTIONSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// A loop-carried iter_arg (not the induction variable) of an enclosing
// affine.for.
bool isLoopCarried(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg)
    return false;
  auto forOp = dyn_cast<affine::AffineForOp>(arg.getOwner()->getParentOp());
  return forOp && llvm::is_contained(forOp.getRegionIterArgs(), v);
}

struct ReductionChain {
  SmallVector<ReductionStep> steps; // chain steps, tail first
  SmallVector<Value> leaves;        // the operands the chain folds together
};

// The affine.store the chain result reaches, following single-use forwarding
// ops (a reciprocal multiply, a cast). Null if the result fans out or is not
// stored.
affine::AffineStoreOp closingStore(Value chainResult) {
  Value v = chainResult;
  while (v.hasOneUse()) {
    Operation *u = *v.user_begin();
    if (auto st = dyn_cast<affine::AffineStoreOp>(u))
      return st.getValueToStore() == v ? st : affine::AffineStoreOp();
    if (u->getNumResults() != 1)
      return {};
    v = u->getResult(0);
  }
  return {};
}

// A leaf load that reads, from an earlier iteration, what `store` writes: the
// reduction's memory-carried recurrence tap (a stencil's `A[i, j-1]`), folded
// in at the tree root to keep that recurrence one operator deep.
bool isCarriedTap(Value leaf, affine::AffineStoreOp store) {
  auto load = leaf.getDefiningOp<affine::AffineLoadOp>();
  if (!load || load.getMemRef() != store.getMemRef())
    return false;
  unsigned depth = affine::getInnermostCommonLoopDepth(
      {load.getOperation(), store.getOperation()});
  affine::MemRefAccess src(store), dst(load);
  SmallVector<affine::DependenceComponent, 2> comps;
  return depth &&
         affine::checkMemrefAccessDependence(src, dst, depth,
                                             /*dependenceConstraints=*/nullptr,
                                             &comps)
                 .value == affine::DependenceResult::HasDependence &&
         !comps.empty() && comps.back().lb.value_or(0) > 0;
}

// Flatten the maximal chain of `proto`'s operator: absorb any single-use step
// of the same operator/idiom, collecting every non-chain operand (peeled
// through the idiom's extends) as a leaf. Absorbed steps are recorded so their
// ops can be erased once the chain is rebalanced.
void flatten(Value v, const ReductionStep &proto, ReductionChain &chain) {
  ReductionStep s = matchReductionStep(v);
  if (s && sameReduction(s, proto) && v.hasOneUse()) {
    chain.steps.push_back(s);
    auto [a, b] = reductionOperands(s);
    flatten(a, proto, chain);
    flatten(b, proto, chain);
    return;
  }
  chain.leaves.push_back(v);
}

// Erase a rewritten step's ops (idiom: trunc, core, both extends), once their
// results are dead. Steps are erased tail-first, so each op is use-empty by the
// time it is reached.
void eraseStep(RewriterBase &b, const ReductionStep &s) {
  Operation *e0 = s.widened() ? s.core->getOperand(0).getDefiningOp() : nullptr;
  Operation *e1 = s.widened() ? s.core->getOperand(1).getDefiningOp() : nullptr;
  for (Operation *op : {s.trunc, s.core, e0, e1})
    if (op && op->use_empty())
      b.eraseOp(op);
}

// A float division by a finite non-zero constant becomes a multiply by its
// reciprocal, trading a divider IP for a multiply. Inexact, so it rides the
// same `float-reassoc` fast-math gate as the reassociation.
void reciprocalizeConstDivs(func::FuncOp fn) {
  SmallVector<arith::DivFOp> divs;
  fn.walk([&](arith::DivFOp op) {
    APFloat c(0.0);
    if (matchPattern(op.getRhs(), m_ConstantFloat(&c)) && c.isFiniteNonZero())
      divs.push_back(op);
  });
  OpBuilder b(fn.getContext());
  for (arith::DivFOp op : divs) {
    APFloat c(0.0);
    matchPattern(op.getRhs(), m_ConstantFloat(&c));
    APFloat recip(c.getSemantics(), 1);
    recip.divide(c, APFloat::rmNearestTiesToEven);
    b.setInsertionPoint(op);
    Value k = arith::ConstantOp::create(b, op.getLoc(),
                                        b.getFloatAttr(op.getType(), recip));
    auto mul = arith::MulFOp::create(b, op.getLoc(), op.getLhs(), k);
    mul->setAttrs(op->getAttrs());
    op.getResult().replaceAllUsesWith(mul.getResult());
    op.erase();
  }
}

struct ReassociateReductionsPass
    : public allo::impl::ReassociateReductionsPassBase<
          ReassociateReductionsPass> {
  using ReassociateReductionsPassBase::ReassociateReductionsPassBase;

  void runOnOperation() override {
    if (floatReassoc)
      reciprocalizeConstDivs(getOperation());

    // Process tails first (reverse program order) so each chain is rebalanced
    // from its outermost step inward and its absorbed links are skipped. Only
    // the integer widening idiom is exactly associative; float needs opt-in.
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (op->getNumResults() == 1 && matchReductionStep(op->getResult(0)))
        candidates.push_back(op);
    });

    DenseSet<Operation *> consumed;
    IRRewriter b(&getContext());
    for (Operation *op : llvm::reverse(candidates)) {
      if (consumed.contains(op))
        continue;
      ReductionStep tail = matchReductionStep(op->getResult(0));
      if (tail.isFloat() && !floatReassoc)
        continue;

      ReductionChain chain;
      chain.steps.push_back(tail);
      auto [lhs, rhs] = reductionOperands(tail);
      flatten(lhs, tail, chain);
      flatten(rhs, tail, chain);
      if (chain.steps.size() < 2) // nothing absorbed: a lone step, no chain
        continue;

      // A loop-carried accumulator is folded in last so its recurrence spans
      // one operator; the remaining leaves form a balanced tree. A memory-
      // carried stencil tap (a load reading an earlier iteration's store) plays
      // the same role, so it is folded in last too.
      affine::AffineStoreOp store = closingStore(op->getResult(0));
      SmallVector<Value> carried, rest;
      for (Value leaf : chain.leaves)
        (isLoopCarried(leaf) || (store && isCarriedTap(leaf, store)) ? carried
                                                                     : rest)
            .push_back(leaf);

      // A bare integer chain carries no cast marking it as a reduction, so a
      // loop-carried accumulator is its key. Without one it is ordinary
      // integer arithmetic, such as the address expressions later passes read.
      if (!tail.isFloat() && !tail.widened() && carried.empty())
        continue;

      // Rewrite only when the depth strictly improves: a carried chain drops
      // its recurrence from N operators to 1; a straight-line chain drops its
      // depth from N to ceil(log2(N)).
      unsigned n = chain.leaves.size();
      bool improves = carried.empty() ? llvm::Log2_32_Ceil(n) < n - 1 : n >= 3;
      if (!improves)
        continue;

      for (const ReductionStep &s : chain.steps) {
        consumed.insert(s.core);
        if (s.trunc)
          consumed.insert(s.trunc);
      }

      b.setInsertionPoint(op);
      Value acc = rest.empty() ? buildBalancedTree(b, tail, carried)
                               : buildBalancedTree(b, tail, rest);
      if (!rest.empty())
        for (Value c : carried)
          acc = buildReductionStep(b, tail, acc, c);

      op->getResult(0).replaceAllUsesWith(acc);
      for (const ReductionStep &s : chain.steps)
        eraseStep(b, s);
    }
  }
};

} // namespace
