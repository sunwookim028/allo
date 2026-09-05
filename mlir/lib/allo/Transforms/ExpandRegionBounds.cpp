/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/MemoryModel.h" // kIndexWidth
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h" // getConstantTripCount
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h" // expandAffineExpr
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir::allo {
#define GEN_PASS_DEF_EXPANDREGIONBOUNDSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;

namespace {

// The value an `affine.for` bound evaluates to at \p b's insertion point: the
// max of the lower-bound map's results, the min of the upper-bound map's. A
// trivial map hands an operand straight back and builds nothing.
Value expandLoopBound(OpBuilder &b, AffineForOp loop, bool isLower) {
  AffineMap map = isLower ? loop.getLowerBoundMap() : loop.getUpperBoundMap();
  ValueRange operands =
      isLower ? loop.getLowerBoundOperands() : loop.getUpperBoundOperands();
  Location loc = loop.getLoc();
  SmallVector<Value> parts;
  for (AffineExpr e : map.getResults())
    parts.push_back(expandAffineExpr(b, loc, e,
                                     operands.take_front(map.getNumDims()),
                                     operands.drop_front(map.getNumDims())));
  Value bound = parts.front();
  for (Value v : llvm::drop_begin(parts))
    bound = isLower ? arith::MaxSIOp::create(b, loc, bound, v).getResult()
                    : arith::MinSIOp::create(b, loc, bound, v).getResult();
  return bound;
}

// The `i1` an `affine.if`'s integer set evaluates to at \p b's insertion point:
// the conjunction of its constraints, each `expr >= 0`, or `== 0` for an
// equality.
Value expandGuardPredicate(OpBuilder &b, AffineIfOp guard) {
  Location loc = guard.getLoc();
  IntegerSet set = guard.getIntegerSet();
  SmallVector<Value> operands(guard.getOperands());
  ArrayRef<Value> args(operands);
  unsigned numDims = set.getNumDims();
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value cond;
  for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
    Value aff =
        expandAffineExpr(b, loc, set.getConstraint(i), args.take_front(numDims),
                         args.drop_front(numDims));
    auto pred =
        set.isEq(i) ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge;
    Value cmp = arith::CmpIOp::create(b, loc, pred, aff, zero);
    cond = cond ? arith::AndIOp::create(b, loc, cond, cmp).getResult() : cmp;
  }
  if (!cond) // an empty set is always true
    cond = arith::ConstantIntOp::create(b, loc, /*value=*/1, /*width=*/1);
  return cond;
}

// Rebuild \p built at `datapathWidth`, rewriting \p cone to the new values and
// erasing the originals. `expandAffineExpr` works in `index`, which carries no
// width for an operator row to be priced at or for an IP signature to match.
// That width is what the emitted bound always had, so no value changes.
void retypeToIndexWidth(OpBuilder &b, IntegerType iw,
                        ArrayRef<Operation *> built,
                        SmallVectorImpl<Value> &cone) {
  IRMapping map;
  // A value from outside the cone is index-typed wherever an affine operand is,
  // so it casts in. The frontend's own `iN` to `index` cast usually meets this
  // one and the pair folds away.
  auto lower = [&](Value v) -> Value {
    if (Value known = map.lookupOrNull(v))
      return known;
    if (!isa<IndexType>(v.getType()))
      return v;
    Value cast = arith::IndexCastOp::create(b, v.getLoc(), iw, v);
    map.map(v, cast);
    return cast;
  };

  for (Operation *op : built) {
    Operation *retyped;
    // A constant's value attribute is typed, so it is rebuilt rather than
    // cloned. Only an index one moves; the guard predicate's `i1` stays.
    if (auto c = dyn_cast<arith::ConstantOp>(op);
        c && isa<IndexType>(c.getType())) {
      retyped = arith::ConstantOp::create(
          b, c.getLoc(),
          b.getIntegerAttr(iw, cast<IntegerAttr>(c.getValue()).getInt()));
    } else {
      SmallVector<Value> operands;
      for (Value v : op->getOperands())
        operands.push_back(lower(v));
      SmallVector<Type> results;
      for (Type t : op->getResultTypes())
        results.push_back(isa<IndexType>(t) ? iw : t);
      OperationState state(op->getLoc(), op->getName(), operands, results,
                           op->getAttrs());
      retyped = b.create(state);
    }
    for (auto [from, to] : llvm::zip(op->getResults(), retyped->getResults()))
      map.map(from, to);
  }
  for (Value &v : cone)
    v = lower(v);
  for (Operation *op : llvm::reverse(built))
    op->erase();
}

struct ExpandRegionBoundsPass
    : public allo::impl::ExpandRegionBoundsPassBase<ExpandRegionBoundsPass> {
  using ExpandRegionBoundsPassBase::ExpandRegionBoundsPassBase;

  void runOnOperation() override {
    SmallVector<Operation *> anchors;
    getOperation().walk([&](Operation *op) {
      if (isa<AffineForOp, AffineIfOp>(op))
        anchors.push_back(op);
    });

    auto iw = IntegerType::get(&getContext(), kIndexWidth);
    for (Operation *anchor : anchors) {
      OpBuilder b(anchor);
      Operation *before = anchor->getPrevNode();
      // Slot order is the marker's contract: `entryConeOf` reads it back by
      // asking these same two questions of the same loop.
      SmallVector<Value> cone;
      if (auto loop = dyn_cast<AffineForOp>(anchor)) {
        if (!loop.hasConstantLowerBound())
          cone.push_back(expandLoopBound(b, loop, /*isLower=*/true));
        if (!getConstantTripCount(loop))
          cone.push_back(expandLoopBound(b, loop, /*isLower=*/false));
      } else {
        cone.push_back(expandGuardPredicate(b, cast<AffineIfOp>(anchor)));
      }
      if (cone.empty())
        continue;

      SmallVector<Operation *> built;
      for (Operation *op = before ? before->getNextNode()
                                  : &anchor->getBlock()->front();
           op != anchor; op = op->getNextNode())
        built.push_back(op);
      retypeToIndexWidth(b, iw, built, cone);
      VolatileOp::create(b, anchor->getLoc(), cone);
    }
  }
};

} // namespace
