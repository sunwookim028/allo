/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORMS_REDUCTIONUTILS_H
#define ALLO_TRANSFORMS_REDUCTIONUTILS_H

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

// Helpers shared by the reduction-restructuring passes
// (raise-memory-reductions, rotate-reductions, tree-height-reduction), which
// recognize associative reductions and rebuild them as balanced trees / rotated
// accumulators.
//
// Three operator shapes are recognized:
//   * float: a bare `arith.addf` / `arith.mulf`;
//   * integer, widened: the frontend's width-extension idiom
//       trunc_w( core( ext(x), ext(y) ) )
//     (`core` = arith.addi/muli, `ext` = extsi/extui). Since
//       trunc(ext a `core` ext b) == (a `core` b) mod 2^w,
//     the reduction is exactly associative, and rebuilding the same idiom
//     leaves operand widths unchanged.
//   * integer, bare: a same-width `arith.addi` / `arith.muli`, what the idiom
//     collapses to once operand widths are narrowed to what the consumer reads.
//
// A bare integer core carries nothing marking it as a reduction, so shape alone
// is not the key: a caller walking arbitrary ops must anchor it on a
// loop-carried accumulator among the chain's leaves, which leaves index and
// address arithmetic untouched. A caller that matched the loop shape first, as
// rotate-reductions does, already holds that anchor.
namespace mlir::allo {

inline bool isFloatReductionOp(Operation *op) {
  return isa<arith::AddFOp, arith::MulFOp>(op);
}

inline bool isIntReductionOp(Operation *op) {
  return isa<arith::AddIOp, arith::MulIOp>(op);
}

inline bool isIntExtendOp(Operation *op) {
  return isa<arith::ExtSIOp, arith::ExtUIOp>(op);
}

// One associative reduction step produced by a value: the combine operator,
// plus the extend/truncate wrappers when it is the integer idiom.
struct ReductionStep {
  Operation *core = nullptr; // arith.add{f,i} / mul{f,i}
  Operation *extProto =
      nullptr;                // an ext feeding `core` (idiom only), else null
  Operation *trunc = nullptr; // the narrowing trunci (idiom only), else null

  explicit operator bool() const { return core != nullptr; }
  bool widened() const { return trunc != nullptr; }
  bool isFloat() const { return isFloatReductionOp(core); }
  bool isMul() const { return isa<arith::MulFOp, arith::MulIOp>(core); }
  // The narrow (logical) result value / type this step produces.
  Value result() const {
    return widened() ? trunc->getResult(0) : core->getResult(0);
  }
  Type type() const { return result().getType(); }
};

// Classify `v` as the result of a reduction step, or an invalid step if `v` is
// not produced by one. A bare integer core matches on shape alone; only a
// caller holding the structural anchor may act on it.
inline ReductionStep matchReductionStep(Value v) {
  ReductionStep step;
  Operation *d = v.getDefiningOp();
  if (!d)
    return step;
  if (isFloatReductionOp(d)) {
    step.core = d;
    return step;
  }
  // `index` stays out: it is what address arithmetic is expressed in.
  if (isIntReductionOp(d) && isa<IntegerType>(d->getResult(0).getType())) {
    step.core = d;
    return step;
  }
  if (isa<arith::TruncIOp>(d)) {
    Operation *core = d->getOperand(0).getDefiningOp();
    if (core && isIntReductionOp(core)) {
      Operation *e0 = core->getOperand(0).getDefiningOp();
      Operation *e1 = core->getOperand(1).getDefiningOp();
      Type narrow = v.getType();
      if (e0 && e1 && isIntExtendOp(e0) && e0->getName() == e1->getName() &&
          e0->getOperand(0).getType() == narrow &&
          e1->getOperand(0).getType() == narrow) {
        step.core = core;
        step.extProto = e0;
        step.trunc = d;
      }
    }
  }
  return step;
}

// The two logical operands `step` combines (peeling the extends of the idiom).
inline std::pair<Value, Value> reductionOperands(const ReductionStep &step) {
  if (step.widened())
    return {step.core->getOperand(0).getDefiningOp()->getOperand(0),
            step.core->getOperand(1).getDefiningOp()->getOperand(0)};
  return {step.core->getOperand(0), step.core->getOperand(1)};
}

// Same operator/idiom (so two steps belong to one reduction chain)?
inline bool sameReduction(const ReductionStep &a, const ReductionStep &b) {
  if (a.core->getName() != b.core->getName() || a.widened() != b.widened())
    return false;
  if (!a.widened())
    return a.type() == b.type();
  return a.extProto->getName() == b.extProto->getName() &&
         a.type() == b.type() &&
         a.core->getResult(0).getType() == b.core->getResult(0).getType();
}

// A fresh one-operand op (ext / trunc) matching `proto`, producing type `ty`.
inline Value cloneCastOp(OpBuilder &b, Operation *proto, Value x, Type ty) {
  OperationState state(proto->getLoc(), proto->getName());
  state.addOperands(x);
  state.addTypes(ty);
  return b.create(state)->getResult(0);
}

// A fresh instance of `proto`'s binary operator over (x, y), inserted at `b`.
inline Value cloneBinaryOp(OpBuilder &b, Operation *proto, Value x, Value y) {
  OperationState state(proto->getLoc(), proto->getName());
  state.addOperands({x, y});
  state.addTypes({x.getType()});
  return b.create(state)->getResult(0);
}

// Rebuild one reduction step combining narrow values (x, y) with `proto`'s
// operator/idiom. For the idiom, re-extend, combine wide, then re-truncate.
inline Value buildReductionStep(OpBuilder &b, const ReductionStep &proto,
                                Value x, Value y) {
  if (!proto.widened())
    return cloneBinaryOp(b, proto.core, x, y);
  Type wide = proto.core->getResult(0).getType();
  Value ex = cloneCastOp(b, proto.extProto, x, wide);
  Value ey = cloneCastOp(b, proto.extProto, y, wide);
  Value c = cloneBinaryOp(b, proto.core, ex, ey);
  return cloneCastOp(b, proto.trunc, c, proto.type());
}

// Combine `values` with `proto`'s operator/idiom as a balanced binary tree.
inline Value buildBalancedTree(OpBuilder &b, const ReductionStep &proto,
                               ValueRange values) {
  SmallVector<Value> level(values.begin(), values.end());
  while (level.size() > 1) {
    SmallVector<Value> next;
    for (unsigned i = 0, e = level.size(); i + 1 < e; i += 2)
      next.push_back(buildReductionStep(b, proto, level[i], level[i + 1]));
    if (level.size() & 1)
      next.push_back(level.back());
    level = std::move(next);
  }
  return level.front();
}

// A leaf/subtree paired with its arrival weight for the weighted tree builder.
using WeightedValue = std::pair<Value, double>;

// Build a minimum-weighted-height tree over `nodes`, merging the two lightest
// each round; a merged node's weight is `max(child weights) + opWeight`.
// `combine(a, b)` emits the operator. Equal or zero weights give a plain
// balanced tree. `nodes` must be non-empty and is consumed.
inline Value
buildWeightedTree(SmallVectorImpl<WeightedValue> &nodes, double opWeight,
                  llvm::function_ref<Value(Value, Value)> combine) {
  while (nodes.size() > 1) {
    llvm::stable_sort(nodes,
                      [](const WeightedValue &a, const WeightedValue &b) {
                        return a.second < b.second;
                      });
    Value a = nodes[0].first, b = nodes[1].first;
    double w = std::max(nodes[0].second, nodes[1].second) + opWeight;
    nodes.erase(nodes.begin(), nodes.begin() + 2);
    nodes.push_back({combine(a, b), w});
  }
  return nodes.front().first;
}

// A maximal chain of one reduction operator/idiom: the steps absorbed (tail
// first) and the leaves they fold together.
struct ReductionChain {
  SmallVector<ReductionStep> steps;
  SmallVector<Value> leaves;
};

// Flatten the maximal chain of `proto`'s operator into `chain`: absorb any
// single-use step of the same operator/idiom, collecting every non-chain
// operand (peeled through the idiom's extends) as a leaf.
inline void flattenChain(Value v, const ReductionStep &proto,
                         ReductionChain &chain) {
  ReductionStep s = matchReductionStep(v);
  if (s && sameReduction(s, proto) && v.hasOneUse()) {
    chain.steps.push_back(s);
    auto [a, b] = reductionOperands(s);
    flattenChain(a, proto, chain);
    flattenChain(b, proto, chain);
    return;
  }
  chain.leaves.push_back(v);
}

// Erase a rewritten step's ops (idiom: trunc, core, both extends) once dead.
// Steps are erased tail first, so each op is use-empty when reached.
inline void eraseStep(const ReductionStep &s) {
  Operation *e0 = s.widened() ? s.core->getOperand(0).getDefiningOp() : nullptr;
  Operation *e1 = s.widened() ? s.core->getOperand(1).getDefiningOp() : nullptr;
  for (Operation *op : {s.trunc, s.core, e0, e1})
    if (op && op->use_empty())
      op->erase();
}

// A loop-carried iter_arg (not the induction variable) of an enclosing
// affine.for.
inline bool isLoopCarried(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg)
    return false;
  auto forOp = dyn_cast<affine::AffineForOp>(arg.getOwner()->getParentOp());
  return forOp && llvm::is_contained(forOp.getRegionIterArgs(), v);
}

// The affine.store the value reaches, following single-use forwarding ops (a
// reciprocal multiply, a cast). Null if the value fans out or is not stored.
inline affine::AffineStoreOp closingStore(Value chainResult) {
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

// A leaf load that reads, from an earlier iteration, what `store` writes: a
// reduction's memory-carried recurrence tap (an accumulator, a stencil's
// `A[i, j-1]`), which belongs at the tree root to keep that recurrence one
// operator deep.
inline bool isCarriedTap(Value leaf, affine::AffineStoreOp store) {
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

} // namespace mlir::allo

#endif // ALLO_TRANSFORMS_REDUCTIONUTILS_H
