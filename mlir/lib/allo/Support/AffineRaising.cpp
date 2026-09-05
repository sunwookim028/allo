/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/AffineRaising.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::allo;

//===----------------------------------------------------------------------===//
// Accesses
//===----------------------------------------------------------------------===//

// Whether the SSA tree computing \p index contains arithmetic whose affine
// counterpart is a DIFFERENT function: `arith.divsi` / `arith.remsi` truncate
// toward zero where an AffineExpr floor-divides, and a truncation wraps where
// an affine expression does not.
static bool hasNonAffineSemantics(Value index) {
  SmallVector<Value> worklist{index};
  llvm::SmallDenseSet<Value> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !seen.insert(v).second)
      continue;
    if (isa<arith::DivSIOp, arith::RemSIOp, arith::TruncIOp>(def))
      return true;
    llvm::append_range(worklist, def->getOperands());
  }
  return false;
}

// The access map of \p indices, or failure when a subscript is not an affine
// function of the enclosing induction variables and loop-invariant values.
static FailureOr<affine::AffineValueMap>
accessMap(AffineValueMapBuilder &builder, OperandRange indices) {
  builder.reset();
  for (Value idx : indices) {
    if (hasNonAffineSemantics(idx) || failed(builder.importValue(idx)))
      return failure();
  }
  affine::AffineValueMap map = builder.compose();
  // The builder admits operands an affine op would reject (an Allo worker-id
  // query is loop-invariant but not a valid affine symbol), so hold the
  // composed map to the dialect's own rule.
  for (unsigned i = 0, e = map.getNumOperands(); i < e; ++i) {
    Value operand = map.getOperand(i);
    bool ok = i < map.getNumDims() ? affine::isValidDim(operand)
                                   : affine::isValidSymbol(operand);
    if (!ok)
      return failure();
  }
  return map;
}

LogicalResult mlir::allo::raiseAffineAccess(RewriterBase &rewriter,
                                            Operation *op) {
  auto load = dyn_cast<memref::LoadOp>(op);
  auto store = dyn_cast<memref::StoreOp>(op);
  if (!load && !store)
    return failure();

  AffineValueMapBuilder builder(op->getContext());
  auto map = accessMap(builder, load ? load.getIndices() : store.getIndices());
  if (failed(map))
    return failure();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  Operation *raised;
  if (load)
    raised =
        affine::AffineLoadOp::create(rewriter, op->getLoc(), load.getMemRef(),
                                     map->getAffineMap(), map->getOperands());
  else
    raised = affine::AffineStoreOp::create(
        rewriter, op->getLoc(), store.getValueToStore(), store.getMemRef(),
        map->getAffineMap(), map->getOperands());
  // The schedule id and key ride on the access and name it to the user, so they
  // must survive the change of form.
  raised->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  rewriter.replaceOp(op, raised->getResults());
  return success();
}

unsigned mlir::allo::raiseAffineAccesses(RewriterBase &rewriter,
                                         Operation *root) {
  // Collect first: an access is replaced, so rewriting inside the walk would
  // erase the node the traversal is standing on.
  SmallVector<Operation *> accesses;
  root->walk([&](Operation *op) {
    if (isa<memref::LoadOp, memref::StoreOp>(op))
      accesses.push_back(op);
  });
  unsigned raised = 0;
  for (Operation *op : accesses)
    raised += succeeded(raiseAffineAccess(rewriter, op));
  return raised;
}

//===----------------------------------------------------------------------===//
// Loop bounds
//===----------------------------------------------------------------------===//

// Read `select(cmp, a, b)` as min/max, returning {isMax, x, y} such that the
// result is `isMax ? max(x, y) : min(x, y)`.
static FailureOr<std::tuple<bool, Value, Value>>
matchSelectAsMinMax(arith::SelectOp sel) {
  auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return failure();

  using P = arith::CmpIPredicate;
  P pred = cmp.getPredicate();
  bool isGE =
      pred == P::sge || pred == P::sgt || pred == P::uge || pred == P::ugt;
  bool isLE =
      pred == P::sle || pred == P::slt || pred == P::ule || pred == P::ult;
  if (!isGE && !isLE)
    return failure(); // eq / ne order nothing

  Value lhs = cmp.getLhs(), rhs = cmp.getRhs();
  Value t = sel.getTrueValue(), f = sel.getFalseValue();
  bool swapped;
  if (t == lhs && f == rhs)
    swapped = false;
  else if (t == rhs && f == lhs)
    swapped = true;
  else
    return failure();

  return std::make_tuple(isGE != swapped, lhs, rhs);
}

static LogicalResult collectBoundExprs(AffineValueMapBuilder &builder,
                                       Value value, bool isLowerBound);

static LogicalResult collectBinaryBoundExprs(AffineValueMapBuilder &builder,
                                             Value lhs, Value rhs,
                                             bool isLowerBound) {
  if (failed(collectBoundExprs(builder, lhs, isLowerBound)))
    return failure();
  return collectBoundExprs(builder, rhs, isLowerBound);
}

// Collect \p value into the builder as bound expressions. An affine bound is a
// MAP whose results are maxed (lower) or minned (upper), so a max/min in the
// bound's own arithmetic flattens into extra results.
static LogicalResult collectBoundExprs(AffineValueMapBuilder &builder,
                                       Value value, bool isLowerBound) {
  Operation *defOp = stripCast(value).getDefiningOp();

  if (isLowerBound) {
    if (auto max = dyn_cast_or_null<arith::MaxSIOp>(defOp))
      return collectBinaryBoundExprs(builder, max.getLhs(), max.getRhs(), true);
    if (auto max = dyn_cast_or_null<arith::MaxUIOp>(defOp))
      return collectBinaryBoundExprs(builder, max.getLhs(), max.getRhs(), true);
  } else {
    if (auto min = dyn_cast_or_null<arith::MinSIOp>(defOp))
      return collectBinaryBoundExprs(builder, min.getLhs(), min.getRhs(),
                                     false);
    if (auto min = dyn_cast_or_null<arith::MinUIOp>(defOp))
      return collectBinaryBoundExprs(builder, min.getLhs(), min.getRhs(),
                                     false);
  }

  if (auto sel = dyn_cast_or_null<arith::SelectOp>(defOp)) {
    if (auto match = matchSelectAsMinMax(sel); succeeded(match)) {
      auto [isMax, x, y] = *match;
      if (isMax == isLowerBound)
        return collectBinaryBoundExprs(builder, x, y, isLowerBound);
    }
  }

  if (auto max = dyn_cast_or_null<affine::AffineMaxOp>(defOp)) {
    if (!isLowerBound)
      return failure();
    return builder.importMapAndOperands(
        max.getAffineMap(), max.getDimOperands(), max.getSymbolOperands(),
        /*allowMultiResults=*/true);
  }
  if (auto min = dyn_cast_or_null<affine::AffineMinOp>(defOp)) {
    if (isLowerBound)
      return failure();
    return builder.importMapAndOperands(
        min.getAffineMap(), min.getDimOperands(), min.getSymbolOperands(),
        /*allowMultiResults=*/true);
  }

  // Leaf: a dim, a symbol, a constant or an affine.apply chain.
  return builder.importValue(value);
}

FailureOr<affine::AffineValueMap>
mlir::allo::matchAffineBound(Value root, bool isLowerBound) {
  AffineValueMapBuilder builder(root.getContext());
  if (failed(collectBoundExprs(builder, root, isLowerBound)))
    return failure();
  return builder.compose();
}

static std::optional<int64_t> getConstPositiveStep(Value step) {
  IntegerAttr::ValueType cst;
  if (!matchPattern(stripCast(step), m_ConstantInt(&cst)))
    return std::nullopt;
  int64_t value = cst.getSExtValue();
  return value > 0 ? std::optional(value) : std::nullopt;
}

//===----------------------------------------------------------------------===//
// Loops
//===----------------------------------------------------------------------===//

FailureOr<affine::AffineForOp>
mlir::allo::raiseToAffineFor(RewriterBase &rewriter, scf::ForOp forOp,
                             std::string &reason) {
  auto lb = matchAffineBound(forOp.getLowerBound(), true);
  if (failed(lb)) {
    reason = "lower bound is not an affine expression of the enclosing loops";
    return failure();
  }
  auto ub = matchAffineBound(forOp.getUpperBound(), false);
  if (failed(ub)) {
    reason = "upper bound is not an affine expression of the enclosing loops";
    return failure();
  }
  auto step = getConstPositiveStep(forOp.getStep());
  if (!step) {
    reason = "step is not a positive constant";
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);
  auto affineLoop = affine::AffineForOp::create(
      rewriter, forOp.getLoc(), lb->getOperands(), lb->getAffineMap(),
      ub->getOperands(), ub->getAffineMap(), *step, forOp.getInitArgs());
  affineLoop->setDiscardableAttrs(forOp->getDiscardableAttrDictionary());

  // Swap the terminators, then move the body across. A loop built WITH
  // iter_args gets no implicit terminator at all, one built without gets an
  // empty `affine.yield`, so the block is emptied rather than the terminator
  // erased.
  Block *body = affineLoop.getBody();
  if (!body->empty())
    rewriter.eraseOp(&body->back());
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  rewriter.setInsertionPoint(yield);
  affine::AffineYieldOp::create(rewriter, yield.getLoc(), yield->getOperands());
  rewriter.eraseOp(yield);

  // The source block's arguments are the induction variable followed by the
  // loop-carried ones, in that order.
  SmallVector<Value> argRepls{affineLoop.getInductionVar()};
  llvm::append_range(argRepls, affineLoop.getRegionIterArgs());
  rewriter.mergeBlocks(forOp.getBody(), body, argRepls);

  // Only now is the induction variable a valid affine dim, so the accesses
  // under it become raisable at exactly this point.
  raiseAffineAccesses(rewriter, affineLoop);
  rewriter.replaceOp(forOp, affineLoop);
  return affineLoop;
}

// Rebuild every bound map over ONE shared operand list, which is what
// `affine.parallel` takes: its per-dimension maps carry no operands of their
// own.
static FailureOr<std::pair<SmallVector<AffineMap>, SmallVector<Value>>>
normalizeParallelBounds(ArrayRef<affine::AffineValueMap> bounds) {
  if (bounds.empty())
    return failure();
  MLIRContext *ctx = bounds.front().getAffineMap().getContext();

  llvm::SmallMapVector<Value, unsigned, 8> dims, syms;
  for (const affine::AffineValueMap &bound : bounds)
    for (Value operand : bound.getOperands()) {
      Value v = stripCast(operand);
      if (affine::isValidDim(v)) // prefer a dim where both are legal
        dims.insert({v, dims.size()});
      else if (affine::isValidSymbol(v))
        syms.insert({v, syms.size()});
      else
        return failure();
    }

  auto globalExpr = [&](Value v) {
    v = stripCast(v);
    auto it = dims.find(v);
    return it != dims.end() ? getAffineDimExpr(it->second, ctx)
                            : getAffineSymbolExpr(syms.find(v)->second, ctx);
  };

  SmallVector<AffineMap> maps;
  for (const affine::AffineValueMap &bound : bounds) {
    auto operands = bound.getOperands();
    unsigned nDims = bound.getNumDims();
    SmallVector<AffineExpr> dimExprs, symExprs;
    for (Value v : operands.take_front(nDims))
      dimExprs.push_back(globalExpr(v));
    for (Value v : operands.drop_front(nDims))
      symExprs.push_back(globalExpr(v));

    SmallVector<AffineExpr> results;
    for (AffineExpr expr : bound.getAffineMap().getResults())
      results.push_back(expr.replaceDimsAndSymbols(dimExprs, symExprs));
    maps.push_back(AffineMap::get(dims.size(), syms.size(), results, ctx));
  }

  SmallVector<Value> operands(dims.keys());
  llvm::append_range(operands, syms.keys());
  return std::make_pair(maps, operands);
}

FailureOr<affine::AffineParallelOp>
mlir::allo::raiseToAffineParallel(RewriterBase &rewriter, scf::ParallelOp parOp,
                                  std::string &reason) {
  if (parOp.getNumReductions() != 0) {
    reason = "a parallel reduction is not modeled";
    return failure();
  }

  SmallVector<affine::AffineValueMap> lbs, ubs;
  SmallVector<int64_t> steps;
  for (unsigned i = 0, e = parOp.getNumLoops(); i < e; ++i) {
    auto lb = matchAffineBound(parOp.getLowerBound()[i], true);
    auto ub = matchAffineBound(parOp.getUpperBound()[i], false);
    auto step = getConstPositiveStep(parOp.getStep()[i]);
    if (failed(lb) || failed(ub) || !step) {
      reason = "dimension " + std::to_string(i) +
               " has a bound or step that is not affine";
      return failure();
    }
    lbs.push_back(*lb);
    ubs.push_back(*ub);
    steps.push_back(*step);
  }

  auto normLBs = normalizeParallelBounds(lbs);
  auto normUBs = normalizeParallelBounds(ubs);
  if (failed(normLBs) || failed(normUBs)) {
    reason = "bounds do not share one affine input space";
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(parOp);
  SmallVector<arith::AtomicRMWKind> reductions;
  auto affineParallel = affine::AffineParallelOp::create(
      rewriter, parOp.getLoc(), TypeRange{}, reductions, normLBs->first,
      normLBs->second, normUBs->first, normUBs->second, steps);
  affineParallel->setDiscardableAttrs(parOp->getDiscardableAttrDictionary());

  Block *body = affineParallel.getBody();
  if (!body->empty())
    rewriter.eraseOp(&body->back());
  auto reduce = cast<scf::ReduceOp>(parOp.getBody()->getTerminator());
  rewriter.setInsertionPoint(reduce);
  affine::AffineYieldOp::create(rewriter, reduce.getLoc(),
                                reduce->getOperands());
  rewriter.eraseOp(reduce);

  rewriter.mergeBlocks(parOp.getBody(), body,
                       ValueRange(affineParallel.getIVs()));
  raiseAffineAccesses(rewriter, affineParallel);
  rewriter.replaceOp(parOp, affineParallel->getResults());
  return affineParallel;
}
