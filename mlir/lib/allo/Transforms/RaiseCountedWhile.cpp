/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::allo {
#define GEN_PASS_DEF_RAISECOUNTEDWHILEPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// The direction the exit test expects the induction variable to move.
enum class Dir { Increasing, Decreasing };

// A matched affine while: the IV iter-arg, its init and tested bound, the
// signed constant step, and whether the ordered test was inclusive. The IV
// evolves as `init + k*delta` and the loop exits when the test against `bound`
// first fails.
struct AffineWhile {
  unsigned ivIndex;
  Value init;
  Value bound;
  int64_t delta;
  bool inclusive;
  Type cmpTy;    // the width the exit test compares at, at least the IV width
  bool ivSigned; // the IV was sign extended to cmpTy rather than zero extended
};

// Whether `v` is loop-invariant w.r.t. `w`: not defined inside either region.
static bool isInvariant(Value v, scf::WhileOp w) {
  Operation *def = v.getDefiningOp();
  return !def || !w->isProperAncestor(def);
}

// Look through the width-adjusting integer casts the frontend wraps integer
// arithmetic in for overflow semantics: trunci(addi(extsi(iv), c)) is iv + c in
// the IV's own width, so the recurrence underneath is what counts.
static Value peelCast(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (!isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(def))
      break;
    v = def->getOperand(0);
  }
  return v;
}

// The predicate for the same comparison with its operands swapped, used to
// normalize a `bound <pred> iv` test to `iv <pred> bound`. eq/ne are symmetric.
static arith::CmpIPredicate swapOperands(arith::CmpIPredicate p) {
  using P = arith::CmpIPredicate;
  switch (p) {
  case P::slt:
    return P::sgt;
  case P::sle:
    return P::sge;
  case P::sgt:
    return P::slt;
  case P::sge:
    return P::sle;
  case P::ult:
    return P::ugt;
  case P::ule:
    return P::uge;
  case P::ugt:
    return P::ult;
  case P::uge:
    return P::ule;
  case P::eq:
  case P::ne:
    return p;
  }
  llvm_unreachable("unhandled CmpIPredicate");
}

// Read an IV-on-lhs ordered predicate as (direction, inclusive). eq/ne are not
// a monotone counted exit, so they yield nullopt.
static std::optional<std::pair<Dir, bool>>
classifyPredicate(arith::CmpIPredicate p) {
  using P = arith::CmpIPredicate;
  switch (p) {
  case P::ult:
  case P::slt:
    return std::make_pair(Dir::Increasing, false);
  case P::ule:
  case P::sle:
    return std::make_pair(Dir::Increasing, true);
  case P::ugt:
  case P::sgt:
    return std::make_pair(Dir::Decreasing, false);
  case P::uge:
  case P::sge:
    return std::make_pair(Dir::Decreasing, true);
  default:
    return std::nullopt;
  }
}

// Match a counted while as an affine-IV model: a pure ordered test of one IV
// against a loop-invariant bound, with a constant-step self-update whose sign
// agrees with the test. Returns nullopt on any deviation.
static std::optional<AffineWhile> matchCountedWhile(scf::WhileOp w) {
  if (!w.getBefore().hasOneBlock() || !w.getAfter().hasOneBlock())
    return std::nullopt;
  Block &before = w.getBefore().front();
  scf::ConditionOp cond = w.getConditionOp();
  scf::YieldOp yield = w.getYieldOp();

  // Identity forwarding: the condition passes the before args through 1:1, so
  // before / after / inits / results share one index space.
  if (cond.getArgs().size() != before.getNumArguments())
    return std::nullopt;
  for (auto [i, arg] : llvm::enumerate(cond.getArgs()))
    if (arg != before.getArgument(i))
      return std::nullopt;

  // `before` is pure and holds only the comparison plus the width casts a
  // cross-width compare wraps its operands in. This is the fence that keeps a
  // data-dependent exit (a load or a multi-cycle IP op in `before`) on the
  // flushing path.
  auto cmp = cond.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp->getBlock() != &before)
    return std::nullopt;
  for (Operation &op : before.without_terminator())
    if (&op != cmp.getOperation() &&
        !isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(&op))
      return std::nullopt;

  // One cmp operand is a before block-arg (the IV), possibly widened by a
  // promotion cast; the other is a loop-invariant bound at the compare width.
  Value ivSide = cmp.getLhs(), bound = cmp.getRhs();
  auto ivArg = dyn_cast<BlockArgument>(peelCast(ivSide));
  bool ivOnLhs = true;
  if (!ivArg || ivArg.getOwner() != &before) {
    ivSide = cmp.getRhs();
    bound = cmp.getLhs();
    ivArg = dyn_cast<BlockArgument>(peelCast(ivSide));
    ivOnLhs = false;
  }
  if (!ivArg || ivArg.getOwner() != &before)
    return std::nullopt;

  // The IV side is raw or a single extension to the compare width; a truncation
  // is not monotone and an index cast is opaque to peelCast, so both are left
  // on the flushing path.
  Type cmpTy = cmp.getLhs().getType();
  bool ivSigned = true;
  if (ivSide != Value(ivArg)) {
    Operation *c = ivSide.getDefiningOp();
    if (!isa<arith::ExtSIOp, arith::ExtUIOp>(c) || c->getOperand(0) != ivArg)
      return std::nullopt;
    ivSigned = isa<arith::ExtSIOp>(c);
  }

  // The bound is invariant, possibly behind the promotion cast; that cast
  // reconstructs before the loop from its invariant operand.
  Value boundRoot = bound;
  if (Operation *bc = bound.getDefiningOp(); bc && w->isProperAncestor(bc)) {
    if (!isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(bc))
      return std::nullopt;
    boundRoot = bc->getOperand(0);
  }
  if (!isInvariant(boundRoot, w))
    return std::nullopt;

  // IV self-update: yield[k] is `addi(iv, c)` or `subi(iv, c)` with a constant
  // c, giving the signed step, read through the overflow casts. `subi(c, iv)`
  // reflects the IV rather than stepping it, so it is not a counted recurrence.
  unsigned k = ivArg.getArgNumber();
  Value ivAfter = w.getAfterArguments()[k];
  Operation *upd = peelCast(yield.getOperand(k)).getDefiningOp();
  int64_t delta;
  if (auto add = dyn_cast_or_null<arith::AddIOp>(upd)) {
    Value l = peelCast(add.getLhs()), r = peelCast(add.getRhs());
    Value stepV = l == ivAfter ? r : r == ivAfter ? l : Value();
    std::optional<int64_t> c =
        stepV ? getConstantIntValue(stepV) : std::nullopt;
    if (!c)
      return std::nullopt;
    delta = *c;
  } else if (auto sub = dyn_cast_or_null<arith::SubIOp>(upd)) {
    if (peelCast(sub.getLhs()) != ivAfter)
      return std::nullopt;
    std::optional<int64_t> c = getConstantIntValue(peelCast(sub.getRhs()));
    if (!c)
      return std::nullopt;
    delta = -*c;
  } else {
    return std::nullopt;
  }
  if (delta == 0)
    return std::nullopt;

  // Normalize the test to IV-on-lhs and read its direction and inclusivity. A
  // `ne` exit is counted only when a constant step reaches a constant bound
  // exactly; a dynamic bound or a step that overshoots stays uncounted.
  arith::CmpIPredicate pred =
      ivOnLhs ? cmp.getPredicate() : swapOperands(cmp.getPredicate());
  Dir dir;
  bool inclusive;
  if (pred == arith::CmpIPredicate::ne) {
    std::optional<int64_t> ci = getConstantIntValue(w.getInits()[k]);
    std::optional<int64_t> cb = getConstantIntValue(boundRoot);
    if (!ci || !cb)
      return std::nullopt;
    int64_t span = *cb - *ci;
    if (span == 0 || (span > 0) != (delta > 0) || span % delta != 0)
      return std::nullopt;
    dir = delta > 0 ? Dir::Increasing : Dir::Decreasing;
    inclusive = false;
  } else {
    std::optional<std::pair<Dir, bool>> cls = classifyPredicate(pred);
    if (!cls)
      return std::nullopt;
    std::tie(dir, inclusive) = *cls;
  }

  // Direction agreement: the step's sign must match the exit test, else the
  // loop is non-terminating rather than counted.
  if ((dir == Dir::Increasing) != (delta > 0))
    return std::nullopt;

  // Raise any integer or index IV: the rewrite reconstructs the IV from an
  // index counter and casts through the IV's type.
  if (!ivArg.getType().isIntOrIndex())
    return std::nullopt;

  return AffineWhile{k,         w.getInits()[k], bound,   delta,
                     inclusive, cmpTy,           ivSigned};
}

struct RaiseCountedWhile : OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp w,
                                PatternRewriter &rewriter) const override {
    std::optional<AffineWhile> m = matchCountedWhile(w);
    if (!m)
      return failure();

    Location loc = w.getLoc();
    rewriter.setInsertionPoint(w);

    Value init = m->init;
    Type ivType = init.getType();
    Type cmpTy = m->cmpTy;
    bool increasing = m->delta > 0;
    int64_t absDelta = increasing ? m->delta : -m->delta;

    // A constant in a given integer or index type.
    auto constOf = [](OpBuilder &b, Location l, Type ty, int64_t v) -> Value {
      if (ty.isIndex())
        return arith::ConstantIndexOp::create(b, l, v);
      return arith::ConstantIntOp::create(b, l, ty, v);
    };

    // The IV value at counter position `count` (an index): `init + count*delta`
    // in the IV type, which folds back to `count` for a unit-step zero-based
    // increasing loop.
    auto ivAt = [&](OpBuilder &b, Location l, Value count) -> Value {
      Value c = ivType.isIndex()
                    ? count
                    : b.createOrFold<arith::IndexCastOp>(l, ivType, count);
      Value scaled =
          b.createOrFold<arith::MulIOp>(l, c, constOf(b, l, ivType, m->delta));
      return b.createOrFold<arith::AddIOp>(l, scaled, init);
    };

    // The exit compares at `cmpTy`: reconstruct the bound there (its promotion
    // cast, if any, moves out of the loop) and lift the init to it, so the trip
    // is the count of `init + k*delta` values that pass the test.
    Value boundW = m->bound;
    if (Operation *bc = m->bound.getDefiningOp(); bc && w->isProperAncestor(bc))
      boundW = rewriter.clone(*bc)->getResult(0);
    Value initW = init;
    if (cmpTy != ivType)
      initW =
          m->ivSigned
              ? arith::ExtSIOp::create(rewriter, loc, cmpTy, init).getResult()
              : arith::ExtUIOp::create(rewriter, loc, cmpTy, init).getResult();

    // Trip count of init + k*delta: fold the inclusive test into an excluded
    // boundary so all four predicates share one span, then ceil-divide by
    // |delta|. createOrFold keeps a constant trip constant for the scheduler.
    Value boundExcl = boundW;
    if (m->inclusive)
      boundExcl = increasing
                      ? rewriter.createOrFold<arith::AddIOp>(
                            loc, boundW, constOf(rewriter, loc, cmpTy, 1))
                      : rewriter.createOrFold<arith::SubIOp>(
                            loc, boundW, constOf(rewriter, loc, cmpTy, 1));
    Value span =
        increasing
            ? rewriter.createOrFold<arith::SubIOp>(loc, boundExcl, initW)
            : rewriter.createOrFold<arith::SubIOp>(loc, initW, boundExcl);
    Value numer = rewriter.createOrFold<arith::AddIOp>(
        loc, span, constOf(rewriter, loc, cmpTy, absDelta - 1));
    Value trip = rewriter.createOrFold<arith::DivSIOp>(
        loc, numer, constOf(rewriter, loc, cmpTy, absDelta));
    // Clamp a zero-trip loop (init already past the bound) to 0, so the bound
    // is empty and the exit IV below is the untouched init.
    trip = rewriter.createOrFold<arith::MaxSIOp>(
        loc, trip, constOf(rewriter, loc, cmpTy, 0));

    Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value ub = cmpTy.isIndex() ? trip
                               : rewriter.createOrFold<arith::IndexCastOp>(
                                     loc, rewriter.getIndexType(), trip);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Carried (non-IV) iter-args, in order.
    SmallVector<Value> carriedInits;
    SmallVector<unsigned> carriedIdx;
    for (auto [i, in] : llvm::enumerate(w.getInits()))
      if (i != m->ivIndex) {
        carriedInits.push_back(in);
        carriedIdx.push_back(i);
      }

    scf::YieldOp whileYield = w.getYieldOp();
    Block &after = w.getAfter().front();

    // Move the after-region body into the for body: the IV maps to its rebuilt
    // value and the carried args to the iter-args. The IV self-update clones as
    // dead code and canonicalize removes it.
    auto build = [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
      IRMapping map;
      map.map(after.getArgument(m->ivIndex), ivAt(b, l, iv));
      for (auto [r, idx] : llvm::enumerate(carriedIdx))
        map.map(after.getArgument(idx), iterArgs[r]);
      for (Operation &op : after.without_terminator())
        b.clone(op, map);
      SmallVector<Value> yields;
      for (unsigned idx : carriedIdx)
        yields.push_back(map.lookupOrDefault(whileYield.getOperand(idx)));
      scf::YieldOp::create(b, l, yields);
    };

    auto forOp =
        scf::ForOp::create(rewriter, loc, lb, ub, step, carriedInits, build);

    info(Stage::Prep, forOp)
        << "Raising counted while loop into a counted for loop";

    for (auto [r, idx] : llvm::enumerate(carriedIdx))
      rewriter.replaceAllUsesWith(w.getResult(idx), forOp.getResult(r));

    // The IV result is the first value failing the test: init + trip*delta.
    if (!w.getResult(m->ivIndex).use_empty()) {
      rewriter.setInsertionPointAfter(forOp);
      rewriter.replaceAllUsesWith(w.getResult(m->ivIndex),
                                  ivAt(rewriter, loc, ub));
    }
    rewriter.eraseOp(w);
    return success();
  }
};

struct RaiseCountedWhilePass
    : public allo::impl::RaiseCountedWhilePassBase<RaiseCountedWhilePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseCountedWhile>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
