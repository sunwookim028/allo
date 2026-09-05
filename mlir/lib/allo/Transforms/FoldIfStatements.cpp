/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h" // StreamGetOp / StreamPutOp
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_FOLDIFSTATEMENTSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

//===----------------------------------------------------------------------===//
// Hyperblock predication: speculate both branches into a select/masked-store
// datapath so the scheduler sees no control flow.
//===----------------------------------------------------------------------===//

// A stream get/put fires only where its i1 predicate holds, so masking it means
// ANDing the branch condition into that predicate instead of speculating the
// side effect. This keeps a conditional stream access inside the single
// pipelined region rather than serializing it as a guard region.
bool isMaskableStream(Operation *op) {
  return isa<StreamGetOp, StreamPutOp>(op);
}

// An op inside an if body may be speculated (hoisted unconditionally) iff it is
// a load/store (loads are safe on FPGA, stores are predicated), a stream
// get/put (masked by its predicate), or a region-free pure op. Anything else
// cannot be masked, so the enclosing if is left alone.
bool speculatable(Operation *op) {
  if (isa<affine::AffineLoadOp, affine::AffineStoreOp, memref::LoadOp,
          memref::StoreOp>(op))
    return true;
  if (isMaskableStream(op))
    return true;
  if (op->getNumRegions() != 0)
    return false;
  return isMemoryEffectFree(op);
}

// The first branch op that cannot be speculated (why `ifOp` is un-convertible),
// or null if both branches are fully speculatable.
Operation *firstBlocker(Operation *ifOp) {
  for (Region &region : ifOp->getRegions()) {
    if (region.empty()) // an absent else region
      continue;
    for (Operation &op : region.front().without_terminator())
      if (!speculatable(&op))
        return &op;
  }
  return nullptr;
}

// Both branch bodies of `ifOp` (affine.if or scf.if) are fully speculatable.
bool canConvert(Operation *ifOp) { return !firstBlocker(ifOp); }

// `op` is nested (however deep) inside a loop, so leaving it un-converted
// blocks that loop from pipelining across it.
bool insideLoop(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(p))
      return true;
  return false;
}

// A branch op that consumes the predicate when speculated: a store (predicated
// read-modify-write) or a maskable stream op (predicate operand). A branch with
// none needs no condition materialized.
bool consumesPredicate(Operation *op) {
  return isa<affine::AffineStoreOp, memref::StoreOp>(op) ||
         isMaskableStream(op);
}

bool needsPredicate(Block *block) {
  return llvm::any_of(*block,
                      [](Operation &op) { return consumesPredicate(&op); });
}

// Hoist a stream get/put out of the branch and gate it on `pred` (ANDed with
// any predicate it already carries), so it fires only where the branch is
// taken. A get's result is forwarded to its uses and is garbage when it does
// not fire, which is safe because every such use is itself predicated.
void maskStreamOp(Operation *op, Operation *ifOp, Value pred, RewriterBase &b) {
  b.setInsertionPoint(ifOp);
  Value existing = isa<StreamGetOp>(op) ? cast<StreamGetOp>(op).getPred()
                                        : cast<StreamPutOp>(op).getPred();
  Value gated =
      existing
          ? arith::AndIOp::create(b, op->getLoc(), existing, pred).getResult()
          : pred;
  if (auto get = dyn_cast<StreamGetOp>(op)) {
    auto nw = StreamGetOp::create(b, get.getLoc(), get.getValue().getType(),
                                  get.getStream(), get.getIndices(), gated);
    b.replaceOp(op, nw.getResult());
  } else {
    auto put = cast<StreamPutOp>(op);
    StreamPutOp::create(b, put.getLoc(), put.getStream(), put.getIndices(),
                        put.getValue(), gated);
    b.eraseOp(op);
  }
}

// Hoist a branch body before `ifOp`, predicating each store under `pred`
// (`store select(pred, storedValue, load)`) and gating each stream get/put on
// `pred`; every other op is speculated.
void predicateBranch(Block *body, Operation *ifOp, Value pred,
                     RewriterBase &b) {
  for (Operation &op : llvm::make_early_inc_range(body->without_terminator())) {
    if (isMaskableStream(&op)) {
      maskStreamOp(&op, ifOp, pred, b);
    } else if (auto store = dyn_cast<affine::AffineStoreOp>(&op)) {
      b.setInsertionPoint(ifOp);
      Value old = affine::AffineLoadOp::create(
          b, store.getLoc(), store.getMemRef(), store.getAffineMap(),
          store.getMapOperands());
      Value sel = arith::SelectOp::create(b, store.getLoc(), pred,
                                          store.getValue(), old);
      affine::AffineStoreOp::create(b, store.getLoc(), sel, store.getMemRef(),
                                    store.getAffineMap(),
                                    store.getMapOperands());
      b.eraseOp(&op);
    } else if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      b.setInsertionPoint(ifOp);
      Value old = memref::LoadOp::create(b, store.getLoc(), store.getMemRef(),
                                         store.getIndices());
      Value sel = arith::SelectOp::create(b, store.getLoc(), pred,
                                          store.getValueToStore(), old);
      memref::StoreOp::create(b, store.getLoc(), sel, store.getMemRef(),
                              store.getIndices());
      b.eraseOp(&op);
    } else {
      b.moveOpBefore(&op, ifOp);
    }
  }
}

// Predicate both branches under `cond` and replace the if's results with
// selects. `cond` is consumed only by a predicated store, a masked stream op or
// a result select, so a guard with none of those may pass a null `cond`.
void convertCore(Operation *ifOp, Block *thenBlock, Block *elseBlock,
                 Value cond, RewriterBase &b) {
  Location loc = ifOp->getLoc();
  auto yieldOperands = [](Block *block) {
    return SmallVector<Value>(block->getTerminator()->getOperands());
  };

  // Capture each branch's yields *after* predicating it: masking a stream get
  // replaces the op and RAUWs the yield operand, so a copy taken beforehand
  // would dangle.
  predicateBranch(thenBlock, ifOp, cond, b);
  SmallVector<Value> thenYield = yieldOperands(thenBlock);
  SmallVector<Value> elseYield;
  if (elseBlock) {
    Value notCond; // only the else stores / stream ops need it
    if (needsPredicate(elseBlock)) {
      b.setInsertionPoint(ifOp);
      Value one = arith::ConstantIntOp::create(b, loc, 1, /*width=*/1);
      notCond = arith::XOrIOp::create(b, loc, cond, one);
    }
    predicateBranch(elseBlock, ifOp, notCond, b);
    elseYield = yieldOperands(elseBlock);
  }

  // Value results: select between the two speculated branches.
  b.setInsertionPoint(ifOp);
  for (auto [result, thenVal, elseVal] :
       llvm::zip(ifOp->getResults(), thenYield, elseYield))
    result.replaceAllUsesWith(
        arith::SelectOp::create(b, loc, cond, thenVal, elseVal));
  b.eraseOp(ifOp);
}

// The affine.if condition (IntegerSet + operands) as an i1: AND of one cmpi per
// constraint (`expr >= 0`, or `expr == 0` for an equality). An empty set is
// vacuously true.
Value materializeCondition(RewriterBase &b, Location loc,
                           affine::AffineIfOp ifOp) {
  IntegerSet set = ifOp.getIntegerSet();
  SmallVector<Value> operands(ifOp.getOperands());
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value cond;
  for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
    AffineMap map = AffineMap::get(set.getNumDims(), set.getNumSymbols(),
                                   set.getConstraint(i), b.getContext());
    Value v = affine::AffineApplyOp::create(b, loc, map, operands);
    Value c = arith::CmpIOp::create(b, loc,
                                    set.isEq(i) ? arith::CmpIPredicate::eq
                                                : arith::CmpIPredicate::sge,
                                    v, zero);
    cond = cond ? arith::AndIOp::create(b, loc, cond, c).getResult() : c;
  }
  if (!cond)
    cond = arith::ConstantIntOp::create(b, loc, 1, /*width=*/1);
  return cond;
}

// affine.if: the condition is an integer set, so materialize it, but only when
// a value result or a predicated op consumes it, leaving no unused condition
// ops behind.
void convert(RewriterBase &b, affine::AffineIfOp ifOp) {
  OpBuilder::InsertionGuard g(b);
  Block *thenBlock = ifOp.getThenBlock();
  Block *elseBlock = ifOp.hasElse() ? ifOp.getElseBlock() : nullptr;
  bool needCond = ifOp.getNumResults() > 0 || needsPredicate(thenBlock) ||
                  (elseBlock && needsPredicate(elseBlock));
  b.setInsertionPoint(ifOp);
  Value cond =
      needCond ? materializeCondition(b, ifOp.getLoc(), ifOp) : Value();
  convertCore(ifOp, thenBlock, elseBlock, cond, b);
}

// scf.if: the condition is already an i1 value, used directly
void convert(RewriterBase &b, scf::IfOp ifOp) {
  OpBuilder::InsertionGuard g(b);
  convertCore(ifOp, ifOp.thenBlock(), ifOp.elseBlock(), ifOp.getCondition(), b);
}

template <typename IfOpTy>
struct HyperblockPredication : OpRewritePattern<IfOpTy> {
  using OpRewritePattern<IfOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOpTy ifOp,
                                PatternRewriter &rewriter) const override {
    if (!canConvert(ifOp))
      return failure();
    Location loc = ifOp.getLoc();
    convert(rewriter, ifOp);
    log(Level::Info, Stage::Prep, loc)
        << "Performing if-conversion on hyperblock";
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Guard-to-bound: fold an affine.if that guards a whole loop body into the
// loop's bounds (index-set splitting), eliminating the conditional.
//===----------------------------------------------------------------------===//

// Concatenate `b`'s single result onto `a` over a merged operand list. A
// multi-result affine.for lower bound is a `max`, an upper bound a `min`, so
// appending a result tightens the bound. Operand order is (a dims, b dims, a
// symbols, b symbols) to match AffineMap's dim-then-symbol layout.
void combineBounds(AffineMap a, ValueRange aOps, AffineMap b, ValueRange bOps,
                   AffineMap &outMap, SmallVector<Value> &outOps) {
  unsigned nda = a.getNumDims(), nsa = a.getNumSymbols();
  unsigned ndb = b.getNumDims(), nsb = b.getNumSymbols();
  SmallVector<AffineExpr> results(a.getResults().begin(), a.getResults().end());
  for (AffineExpr e : b.getResults())
    results.push_back(e.shiftDims(ndb, nda).shiftSymbols(nsb, nsa));
  outMap = AffineMap::get(nda + ndb, nsa + nsb, results, a.getContext());
  outOps.assign(aOps.begin(), aOps.begin() + nda); // a dims
  outOps.append(bOps.begin(), bOps.begin() + ndb); // b dims
  outOps.append(aOps.begin() + nda, aOps.end());   // a symbols
  outOps.append(bOps.begin() + ndb, bOps.end());   // b symbols
}

// Whether `e` is affine in dim `pos`: it decomposes exactly as
// `coeff * d_pos + residual` with a constant `coeff` and `residual` independent
// of the dim. Feeding the dim to `mod`/`floordiv`/`ceildiv` makes `e`
// quasi-affine in it, where no such decomposition exists.
static bool isAffineInDim(AffineExpr e, unsigned pos) {
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return true; // a leaf: this dim, another dim/symbol, or a constant
  switch (e.getKind()) {
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
    return isAffineInDim(bin.getLHS(), pos) && isAffineInDim(bin.getRHS(), pos);
  default: // Mod / FloorDiv / CeilDiv
    return !e.isFunctionOfDim(pos);
  }
}

struct FoldGuardIntoLoopBound : OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineIfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Fold only a pure guard (no results, no else) that wraps the *entire* body
    // of its enclosing affine.for.
    if (ifOp.hasElse() || ifOp.getNumResults() != 0)
      return failure();
    auto forOp = dyn_cast<affine::AffineForOp>(ifOp->getParentOp());
    if (!forOp)
      return failure();
    Block *body = forOp.getBody();
    if (&body->front() != ifOp.getOperation() ||
        ifOp->getNextNode() != body->getTerminator())
      return failure();

    // A single inequality `constraint >= 0` that is affine in the loop IV.
    IntegerSet set = ifOp.getIntegerSet();
    if (set.getNumConstraints() != 1 || set.isEq(0))
      return failure();
    unsigned numDims = set.getNumDims(), numSyms = set.getNumSymbols();
    ValueRange operands = ifOp.getOperands();
    Value iv = forOp.getInductionVar();
    std::optional<unsigned> ivPos;
    for (unsigned k = 0; k < numDims; ++k)
      if (operands[k] == iv) {
        ivPos = k;
        break;
      }
    if (!ivPos)
      return failure();

    // constraint = coeff*iv + residual; only a unit coefficient maps cleanly to
    // a single bound (`iv >= -residual` if +1, `iv <= residual` if -1).
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr c = set.getConstraint(0);
    AffineExpr ivDim = getAffineDimExpr(*ivPos, ctx);
    // The two-point probe below is only valid when the constraint is affine in
    // the IV: on a quasi-affine one (a floordiv/mod of a coalesced nest's IV)
    // it reads a finite difference as the coefficient and drops a real guard.
    if (!isAffineInDim(c, *ivPos))
      return failure();
    AffineExpr residual = simplifyAffineExpr(
        c.replace(ivDim, getAffineConstantExpr(0, ctx)), numDims, numSyms);
    AffineExpr coeffExpr = simplifyAffineExpr(
        c.replace(ivDim, getAffineConstantExpr(1, ctx)) - residual, numDims,
        numSyms);
    auto coeff = dyn_cast<AffineConstantExpr>(coeffExpr);
    if (!coeff || (coeff.getValue() != 1 && coeff.getValue() != -1))
      return failure();
    bool lower = coeff.getValue() == 1;

    // Re-express the bound over the `if`'s operands minus the IV dim.
    SmallVector<AffineExpr> dimRepl;
    SmallVector<Value> boundOperands;
    unsigned newDim = 0;
    for (unsigned k = 0; k < numDims; ++k) {
      if (k == *ivPos) {
        dimRepl.push_back(getAffineConstantExpr(0, ctx)); // dead, unreferenced
        continue;
      }
      dimRepl.push_back(getAffineDimExpr(newDim++, ctx));
      boundOperands.push_back(operands[k]);
    }
    SmallVector<AffineExpr> symRepl;
    for (unsigned k = 0; k < numSyms; ++k)
      symRepl.push_back(getAffineSymbolExpr(k, ctx));
    for (unsigned k = 0; k < numSyms; ++k)
      boundOperands.push_back(operands[numDims + k]);
    // Lower: `iv >= -residual`. Upper is exclusive: `iv <= residual` == `iv <
    // residual + 1`.
    AffineExpr boundExpr =
        lower ? -residual : residual + getAffineConstantExpr(1, ctx);
    boundExpr = simplifyAffineExpr(
        boundExpr.replaceDimsAndSymbols(dimRepl, symRepl), newDim, numSyms);
    AffineMap extraMap = AffineMap::get(newDim, numSyms, boundExpr);

    // Tighten the existing bound (multi-result max/min) and canonicalize.
    AffineMap curMap =
        lower ? forOp.getLowerBoundMap() : forOp.getUpperBoundMap();
    SmallVector<Value> curOperands(lower ? forOp.getLowerBoundOperands()
                                         : forOp.getUpperBoundOperands());
    AffineMap combinedMap;
    SmallVector<Value> combinedOperands;
    combineBounds(curMap, curOperands, extraMap, boundOperands, combinedMap,
                  combinedOperands);
    affine::canonicalizeMapAndOperands(&combinedMap, &combinedOperands);

    rewriter.modifyOpInPlace(forOp, [&]() {
      if (lower)
        forOp.setLowerBound(combinedOperands, combinedMap);
      else
        forOp.setUpperBound(combinedOperands, combinedMap);
    });

    // Inline the guarded body and drop the `if`.
    for (Operation &op :
         llvm::make_early_inc_range(ifOp.getThenBlock()->without_terminator()))
      rewriter.moveOpBefore(&op, ifOp);
    rewriter.eraseOp(ifOp);
    info(Stage::Prep, forOp)
        << "Folded an affine.if guard into the enclosing loop's "
        << (lower ? "lower" : "upper") << " bound";
    return success();
  }
};

struct FoldIfStatementsPass
    : public allo::impl::FoldIfStatementsPassBase<FoldIfStatementsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Guard-to-bound is preferred (it deletes dead iterations rather than
    // masking them), so give it the higher benefit.
    patterns.add<FoldGuardIntoLoopBound>(&getContext(), /*benefit=*/2);
    patterns.add<HyperblockPredication<affine::AffineIfOp>,
                 HyperblockPredication<scf::IfOp>>(&getContext(),
                                                   /*benefit=*/1);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();

    // Any conditional still in a loop body could not be predicated/folded, so
    // it schedules as one opaque unit the loop cannot pipeline across.
    getOperation().walk([&](Operation *op) {
      if (!isa<affine::AffineIfOp, scf::IfOp>(op) || !insideLoop(op))
        return;
      Operation *blocker = firstBlocker(op);
      assert(blocker && "a convertible if should have been converted");
      warn(Stage::Prep, op)
          << "Conditional left as an opaque scheduling unit because '"
          << blocker->getName().getStringRef()
          << "' cannot be predicated; the enclosing loop cannot pipeline "
             "across it";
    });
  }
};

} // namespace
