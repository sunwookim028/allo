/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryModel.h" // kIndexWidth
#include "allo/Support/BitAnalysis.h"    // knownBits
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::allo {
#define GEN_PASS_DEF_NARROWDEMANDEDBITSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {

// The two's-complement ring operators, whose low `w` result bits are a function
// of the low `w` bits of the operands alone, which is what makes sinking a
// truncation through them exact. Division, remainder and compare read the high
// bits, so the demand stops at those.
bool isRingOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op);
}

// Bitwise operators are per-bit, so a truncation sinks through them too.
bool isBitwiseOp(Operation *op) {
  return isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
}

//===----------------------------------------------------------------------===//
// Forward value hulls
//===----------------------------------------------------------------------===//

/// A signed hull [lo, hi] of the values an SSA value can carry.
using Hull = std::pair<int64_t, int64_t>;

/// The hull, when it fits int64; unknown on overflow.
std::optional<Hull> mkHull(__int128 lo, __int128 hi) {
  assert(lo <= hi && "a hull is ordered");
  if (lo < std::numeric_limits<int64_t>::min() ||
      hi > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return Hull{(int64_t)lo, (int64_t)hi};
}

/// Significant bits of a hull, the signed convention the datapath sizes by.
unsigned bitsOfHull(Hull h) {
  auto bits = [](int64_t v) {
    return (unsigned)APInt(64, (uint64_t)v, /*isSigned=*/true)
        .getSignificantBits();
  };
  return std::max(bits(h.first), bits(h.second));
}

/// Depth cap on the recursive walk over the value DAG.
constexpr unsigned kHullDepth = 8;

std::optional<Hull> hullOf(Value v, unsigned depth);

/// Interval-evaluate an affine expr; dims and symbols read \p operands.
std::optional<Hull> hullOfExpr(AffineExpr e, unsigned numDims,
                               ValueRange operands, unsigned depth) {
  auto operand = [&](unsigned pos) -> std::optional<Hull> {
    assert(pos < operands.size() &&
           "an affine map's operands cover its dims and symbols");
    return hullOf(operands[pos], depth);
  };
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return Hull{c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return operand(d.getPosition());
  if (auto s = dyn_cast<AffineSymbolExpr>(e))
    return operand(numDims + s.getPosition());
  auto bin = cast<AffineBinaryOpExpr>(e);
  auto lhs = hullOfExpr(bin.getLHS(), numDims, operands, depth);
  auto rhs = hullOfExpr(bin.getRHS(), numDims, operands, depth);
  if (!lhs || !rhs)
    return std::nullopt;
  auto [a, b] = *lhs;
  auto [c, d] = *rhs;
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return mkHull((__int128)a + c, (__int128)b + d);
  case AffineExprKind::Mul: {
    __int128 p[] = {(__int128)a * c, (__int128)a * d, (__int128)b * c,
                    (__int128)b * d};
    return mkHull(*std::min_element(p, p + 4), *std::max_element(p, p + 4));
  }
  case AffineExprKind::FloorDiv:
    if (c != d || c <= 0)
      return std::nullopt;
    return Hull{llvm::divideFloorSigned(a, c), llvm::divideFloorSigned(b, c)};
  case AffineExprKind::CeilDiv:
    if (c != d || c <= 0)
      return std::nullopt;
    return Hull{llvm::divideCeilSigned(a, c), llvm::divideCeilSigned(b, c)};
  case AffineExprKind::Mod:
    if (c != d || c <= 0)
      return std::nullopt;
    return a >= 0 && b < c ? lhs : std::optional<Hull>(Hull{0, c - 1});
  default:
    return std::nullopt;
  }
}

//===----------------------------------------------------------------------===//
// Loop-carried hulls
//===----------------------------------------------------------------------===//

Hull hullUnion(Hull a, Hull b) {
  return {std::min(a.first, b.first), std::max(a.second, b.second)};
}

/// One iteration of a carried value's transfer, decomposed against its own
/// iter-arg: the value either translates by `delta` or resets into `join`.
/// Select, min and max return one of their operands, which makes the
/// alternation exact. `trunc` is the narrowest truncation the walk looked
/// through; the envelope must fit it or the transfer wraps mid-cone.
struct Step {
  Hull delta{0, 0};
  std::optional<Hull> join;
  unsigned trunc = std::numeric_limits<unsigned>::max();
};

/// Recurrences whose hull is in flight, keyed (loop, result index). A transfer
/// side that reads its own or a coupled iter-arg re-enters here and sees
/// unknown rather than recursing forever.
thread_local llvm::DenseSet<std::pair<Operation *, unsigned>> inFlightHulls;

/// Decompose \p v as a transfer of iter-arg \p arg. The seam casts left behind
/// by the narrowing rewrites are value-preserving here: extension exactly,
/// truncation once `recurrenceHull` has held the envelope against the narrowest
/// width the walk passed through.
std::optional<Step> stepOf(Value v, Value arg, unsigned depth) {
  if (v == arg)
    return Step{};
  if (!depth--)
    return std::nullopt;
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (isa<arith::ExtSIOp>(op))
    return stepOf(op->getOperand(0), arg, depth);
  if (auto tr = dyn_cast<arith::TruncIOp>(op)) {
    std::optional<Step> s = stepOf(tr.getIn(), arg, depth);
    if (s)
      s->trunc = std::min(s->trunc, cast<IntegerType>(tr.getType()).getWidth());
    return s;
  }
  // A translated step shifts the reset hulls too: a reset deeper in the cone
  // still rides every operator above it.
  auto translate = [&](Value stepped, Value other,
                       bool negate) -> std::optional<Step> {
    std::optional<Step> s = stepOf(stepped, arg, depth);
    if (!s)
      return std::nullopt;
    std::optional<Hull> e = hullOf(other, depth);
    if (!e)
      return std::nullopt;
    auto shift = [&](Hull h) {
      return negate ? mkHull((__int128)h.first - e->second,
                             (__int128)h.second - e->first)
                    : mkHull((__int128)h.first + e->first,
                             (__int128)h.second + e->second);
    };
    std::optional<Hull> d = shift(s->delta);
    if (!d)
      return std::nullopt;
    s->delta = *d;
    if (s->join) {
      std::optional<Hull> j = shift(*s->join);
      if (!j)
        return std::nullopt;
      s->join = *j;
    }
    return s;
  };
  // Each arm is either a further transfer or an independent reset hull.
  auto alternate = [&](Value a, Value b) -> std::optional<Step> {
    Step out;
    bool stepped = false;
    for (Value arm : {a, b}) {
      if (std::optional<Step> s = stepOf(arm, arg, depth)) {
        out.delta = stepped ? hullUnion(out.delta, s->delta) : s->delta;
        stepped = true;
        out.trunc = std::min(out.trunc, s->trunc);
        if (s->join)
          out.join = out.join ? hullUnion(*out.join, *s->join) : *s->join;
        continue;
      }
      std::optional<Hull> h = hullOf(arm, depth);
      if (!h)
        return std::nullopt;
      out.join = out.join ? hullUnion(*out.join, *h) : *h;
    }
    return out;
  };
  return llvm::TypeSwitch<Operation *, std::optional<Step>>(op)
      .Case<arith::AddIOp>([&](auto) -> std::optional<Step> {
        if (auto s = translate(op->getOperand(0), op->getOperand(1), false))
          return s;
        return translate(op->getOperand(1), op->getOperand(0), false);
      })
      .Case<arith::SubIOp>([&](auto) {
        return translate(op->getOperand(0), op->getOperand(1), true);
      })
      .Case<arith::SelectOp>([&](arith::SelectOp sel) {
        return alternate(sel.getTrueValue(), sel.getFalseValue());
      })
      .Case<arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp, arith::MaxUIOp>(
          [&](auto) { return alternate(op->getOperand(0), op->getOperand(1)); })
      .Default([](auto) { return std::nullopt; });
}

/// The hull of \p fo's \p idx-th carried value: what the body reads, or with
/// \p forResult the loop's final result. Over a constant trip the value stays
/// inside (init u join) + steps * [min(delta, 0), max(delta, 0)]. The refusal
/// always tests the full-trip envelope against the carrier, so no reachable
/// value, intermediate or final, can wrap.
std::optional<Hull> recurrenceHull(affine::AffineForOp fo, unsigned idx,
                                   bool forResult, unsigned depth) {
  auto ity = dyn_cast<IntegerType>(fo.getResult(idx).getType());
  if (!ity)
    return std::nullopt;
  auto key = std::make_pair(fo.getOperation(), idx);
  if (!inFlightHulls.insert(key).second)
    return std::nullopt;
  llvm::scope_exit guard([&] { inFlightHulls.erase(key); });
  std::optional<uint64_t> trip = affine::getConstantTripCount(fo);
  // The envelope multiplies the trip, so bound it to keep the products exact.
  if (!trip || *trip == 0 || *trip > (uint64_t(1) << 32))
    return std::nullopt;
  std::optional<Hull> init = hullOf(fo.getInits()[idx], depth);
  if (!init)
    return std::nullopt;
  Value yielded = cast<affine::AffineYieldOp>(fo.getBody()->getTerminator())
                      .getOperand(idx);
  std::optional<Step> st = stepOf(yielded, fo.getRegionIterArgs()[idx], depth);
  if (!st) {
    // A transfer that never reads its own iter-arg is a plain reset.
    std::optional<Hull> h = hullOf(yielded, depth);
    if (!h)
      return std::nullopt;
    st = Step{{0, 0}, *h};
  }
  Hull base = st->join ? hullUnion(*init, *st->join) : *init;
  auto env = [&](__int128 steps) {
    return mkHull(base.first + steps * std::min<__int128>(st->delta.first, 0),
                  base.second +
                      steps * std::max<__int128>(st->delta.second, 0));
  };
  std::optional<Hull> full = env(*trip);
  unsigned cap = std::min(ity.getWidth(), st->trunc);
  if (!full || bitsOfHull(*full) > cap)
    return std::nullopt;
  return forResult ? full : env(*trip - 1);
}

/// The hull of the value \p v carries: a forward interval walk over constants,
/// constant loop bounds and the monotone arith transfers. Unknown is always
/// sound. A hull the value's own carrier could wrap is refused, so a returned
/// hull is the value's range, never its residue mod 2^width.
std::optional<Hull> hullOf(Value v, unsigned depth) {
  if (!depth--)
    return std::nullopt;
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return cst.getSignificantBits() <= 64
               ? std::optional<Hull>(
                     Hull{cst.getSExtValue(), cst.getSExtValue()})
               : std::nullopt;
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    if (affine::AffineForOp fo = affine::getForInductionVarOwner(v)) {
      if (!fo.hasConstantLowerBound() || !fo.hasConstantUpperBound() ||
          fo.getConstantLowerBound() >= fo.getConstantUpperBound())
        return std::nullopt;
      return Hull{fo.getConstantLowerBound(), fo.getConstantUpperBound() - 1};
    }
    auto fo = dyn_cast<affine::AffineForOp>(ba.getOwner()->getParentOp());
    if (fo && ba.getOwner() == fo.getBody() && ba.getArgNumber() > 0)
      return recurrenceHull(fo, ba.getArgNumber() - 1, /*forResult=*/false,
                            depth);
    return std::nullopt;
  }
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  auto in = [&](unsigned k) { return hullOf(op->getOperand(k), depth); };
  auto rhsConst = [&]() -> std::optional<int64_t> {
    APInt c;
    if (matchPattern(op->getOperand(1), m_ConstantInt(&c)) &&
        c.getSignificantBits() <= 64)
      return c.getSExtValue();
    return std::nullopt;
  };
  auto binary = [&](auto f) -> std::optional<Hull> {
    auto x = in(0), y = in(1);
    if (!x || !y)
      return std::nullopt;
    return f(x->first, x->second, y->first, y->second);
  };
  std::optional<Hull> h =
      llvm::TypeSwitch<Operation *, std::optional<Hull>>(op)
          .Case<affine::AffineApplyOp>([&](affine::AffineApplyOp ap) {
            AffineMap m = ap.getAffineMap();
            return hullOfExpr(m.getResult(0), m.getNumDims(), ap.getOperands(),
                              depth);
          })
          .Case<affine::AffineForOp>([&](affine::AffineForOp fo) {
            return recurrenceHull(fo, cast<OpResult>(v).getResultNumber(),
                                  /*forResult=*/true, depth);
          })
          .Case<arith::AddIOp>([&](auto) {
            return binary([](int64_t a, int64_t b, int64_t c, int64_t d) {
              return mkHull((__int128)a + c, (__int128)b + d);
            });
          })
          .Case<arith::SubIOp>([&](auto) {
            return binary([](int64_t a, int64_t b, int64_t c, int64_t d) {
              return mkHull((__int128)a - d, (__int128)b - c);
            });
          })
          .Case<arith::MulIOp>([&](auto) {
            return binary([](int64_t a, int64_t b, int64_t c, int64_t d) {
              __int128 p[] = {(__int128)a * c, (__int128)a * d, (__int128)b * c,
                              (__int128)b * d};
              return mkHull(*std::min_element(p, p + 4),
                            *std::max_element(p, p + 4));
            });
          })
          .Case<arith::AndIOp>([&](auto) -> std::optional<Hull> {
            // AND with a non-negative mask lands in [0, mask] whatever the
            // other side holds.
            auto c = rhsConst();
            if (!c || *c < 0)
              return std::nullopt;
            return Hull{0, *c};
          })
          .Case<arith::OrIOp, arith::XOrIOp>([&](auto) -> std::optional<Hull> {
            auto x = in(0), y = in(1);
            if (!x || !y || x->first < 0 || y->first < 0)
              return std::nullopt;
            unsigned k = std::max(APInt(64, x->second).getActiveBits(),
                                  APInt(64, y->second).getActiveBits());
            return k > 62 ? std::nullopt
                          : std::optional<Hull>(Hull{0, (int64_t(1) << k) - 1});
          })
          .Case<arith::RemUIOp>([&](auto) -> std::optional<Hull> {
            auto c = rhsConst();
            if (!c || *c <= 0)
              return std::nullopt;
            return Hull{0, *c - 1};
          })
          .Case<arith::RemSIOp>([&](auto) -> std::optional<Hull> {
            auto c = rhsConst();
            if (!c || *c <= 0)
              return std::nullopt;
            auto x = in(0);
            if (x && x->first >= 0)
              return Hull{0, std::min(x->second, *c - 1)};
            return Hull{-(*c - 1), *c - 1};
          })
          .Case<arith::DivSIOp, arith::DivUIOp>(
              [&](auto) -> std::optional<Hull> {
                auto c = rhsConst();
                auto x = in(0);
                if (!c || *c <= 0 || !x)
                  return std::nullopt;
                if (isa<arith::DivUIOp>(op) && x->first < 0)
                  return std::nullopt;
                return Hull{x->first / *c, x->second / *c};
              })
          .Case<arith::ShLIOp>([&](auto) -> std::optional<Hull> {
            auto c = rhsConst();
            auto x = in(0);
            if (!c || *c < 0 || *c > 62 || !x)
              return std::nullopt;
            __int128 p = __int128(1) << *c;
            return mkHull(x->first * p, x->second * p);
          })
          .Case<arith::ShRUIOp, arith::ShRSIOp>(
              [&](auto) -> std::optional<Hull> {
                auto c = rhsConst();
                auto x = in(0);
                if (!c || *c < 0 || *c > 62 || !x)
                  return std::nullopt;
                if (isa<arith::ShRUIOp>(op) && x->first < 0)
                  return std::nullopt;
                int64_t p = int64_t(1) << *c;
                return Hull{llvm::divideFloorSigned(x->first, p),
                            llvm::divideFloorSigned(x->second, p)};
              })
          .Case<arith::SelectOp>([&](auto) -> std::optional<Hull> {
            auto x = in(1), y = in(2);
            if (!x || !y)
              return std::nullopt;
            return Hull{std::min(x->first, y->first),
                        std::max(x->second, y->second)};
          })
          .Case<arith::MinSIOp, arith::MaxSIOp>(
              [&](auto) -> std::optional<Hull> {
                return binary([&](int64_t a, int64_t b, int64_t c, int64_t d) {
                  return isa<arith::MinSIOp>(op)
                             ? std::optional<Hull>(
                                   Hull{std::min(a, c), std::min(b, d)})
                             : std::optional<Hull>(
                                   Hull{std::max(a, c), std::max(b, d)});
                });
              })
          .Case<arith::MinUIOp, arith::MaxUIOp>(
              [&](auto) -> std::optional<Hull> {
                return binary([&](int64_t a, int64_t b, int64_t c,
                                  int64_t d) -> std::optional<Hull> {
                  if (a < 0 || c < 0)
                    return std::nullopt;
                  return isa<arith::MinUIOp>(op)
                             ? Hull{std::min(a, c), std::min(b, d)}
                             : Hull{std::max(a, c), std::max(b, d)};
                });
              })
          .Case<arith::ExtSIOp, arith::IndexCastOp, arith::TruncIOp>(
              [&](auto) { return in(0); })
          .Case<arith::ExtUIOp, arith::IndexCastUIOp>(
              [&](auto) -> std::optional<Hull> {
                // Reinterprets the bits unsigned: exact only where the input
                // is proven non-negative.
                auto x = in(0);
                if (!x || x->first < 0)
                  return std::nullopt;
                return x;
              })
          .Default([](auto) { return std::nullopt; });
  // The wrap refusal: a truncating cast or a ring op computing mod 2^width
  // holds the transfer's hull only when that hull fits the carrier.
  if (h)
    if (auto ity = dyn_cast<IntegerType>(v.getType()))
      if (bitsOfHull(*h) > ity.getWidth())
        return std::nullopt;
  return h;
}

/// The width the datapath builds this carrier at; the width narrowing must
/// beat to be a narrowing at all.
unsigned carrierWidth(Type t) {
  return isa<IndexType>(t) ? kIndexWidth : cast<IntegerType>(t).getWidth();
}

//===----------------------------------------------------------------------===//
// Rewrites
//===----------------------------------------------------------------------===//

// A constant-amount shift sinks through a truncation to width `w`. A left shift
// by `c < w` always does. A right shift by `c < w` does when the bits the
// narrow shift cannot see are known: zero above `w` for a logical shift, or
// replicated sign for an arithmetic one.
bool shiftSinks(Operation *op, IntegerType narrow) {
  if (!isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>(op))
    return false;
  APInt amt;
  if (!matchPattern(op->getOperand(1), m_ConstantInt(&amt)))
    return false;
  unsigned w = narrow.getWidth();
  if (amt.uge(w))
    return false;
  if (isa<arith::ShLIOp>(op))
    return true;
  unsigned orig = cast<IntegerType>(op->getResult(0).getType()).getWidth();
  llvm::KnownBits kb = knownBits(op->getOperand(0));
  if (isa<arith::ShRUIOp>(op))
    return kb.countMinLeadingZeros() >= orig - w;
  return orig - w <
         std::max(kb.countMinLeadingZeros(), kb.countMinLeadingOnes());
}

// trunc_w(op(a, b)) -> op(trunc_w(a), trunc_w(b)), moving the truncation toward
// the leaves so the operator is built at the width its consumer reads; the
// leftover truncations fold into the extends up the chain. Ring and bitwise
// operators commute with truncation, a constant-amount shift does under
// `shiftSinks`, and a select is bit-wise in its arms with its condition passing
// through. Division, remainder, compare, min and max read the high bits and
// stop the sink.
struct SinkTruncThroughOp : OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncIOp trunc,
                                PatternRewriter &rewriter) const override {
    Operation *op = trunc.getIn().getDefiningOp();
    // Without a single use the wide result stays live, so the wide operator
    // survives and this only adds truncations.
    if (!op || !op->hasOneUse())
      return failure();
    auto narrow = cast<IntegerType>(trunc.getType());
    Location loc = op->getLoc();
    if (auto sel = dyn_cast<arith::SelectOp>(op)) {
      Value t =
          arith::TruncIOp::create(rewriter, loc, narrow, sel.getTrueValue());
      Value f =
          arith::TruncIOp::create(rewriter, loc, narrow, sel.getFalseValue());
      rewriter.replaceOpWithNewOp<arith::SelectOp>(trunc, sel.getCondition(), t,
                                                   f);
      return success();
    }
    if (!isRingOp(op) && !isBitwiseOp(op) && !shiftSinks(op, narrow))
      return failure();
    OperationState state(loc, op->getName());
    for (Value operand : op->getOperands())
      state.addOperands(
          arith::TruncIOp::create(rewriter, loc, narrow, operand).getResult());
    state.addTypes(narrow);
    rewriter.replaceOp(trunc, rewriter.create(state)->getResult(0));
    return success();
  }
};

// `x & y` -> `x` when every bit `y` would clear is already zero in `x`: a mask
// over a field the value cannot hold. Writing a bit field splices with such a
// mask on every field after the first, and the splices chain, so each mask
// removed takes a whole AND off the critical path.
struct DropRedundantMask : OpRewritePattern<arith::AndIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AndIOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<IntegerType>(op.getType()))
      return failure(); // an index mask has no width to reason in
    llvm::KnownBits lhs = knownBits(op.getLhs());
    llvm::KnownBits rhs = knownBits(op.getRhs());
    // Bit by bit: the mask keeps this one, or the value never sets it.
    if ((rhs.One | lhs.Zero).isAllOnes()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if ((lhs.One | rhs.Zero).isAllOnes()) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

// `x | y` -> `x` when every bit `y` would set is already one in `x`, or `y`
// never sets it: an OR that contributes no new bits. The dual of the redundant
// AND mask, One and Zero swapped.
struct DropRedundantOr : OpRewritePattern<arith::OrIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::OrIOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<IntegerType>(op.getType()))
      return failure(); // an index or has no width to reason in
    llvm::KnownBits lhs = knownBits(op.getLhs());
    llvm::KnownBits rhs = knownBits(op.getRhs());
    // Bit by bit: the other side already sets this one, or this side never
    // does.
    if ((rhs.Zero | lhs.One).isAllOnes()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if ((lhs.Zero | rhs.One).isAllOnes()) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

// Rebuilds a ring or bitwise op whose value hull needs fewer bits than its
// carrier at the hull's width, with resize casts at the seams. The casts are
// wiring; the operator is built and priced at the width the value spans. An
// `index` operand enters the integer domain here, which is what lets a
// truncation reach it.
struct NarrowFromHull : RewritePattern {
  NarrowFromHull(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isRingOp(op) && !isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op))
      return failure();
    Type ty = op->getResult(0).getType();
    unsigned width = carrierWidth(ty);
    std::optional<Hull> h = hullOf(op->getResult(0), kHullDepth);
    if (!h)
      return failure();
    unsigned w = bitsOfHull(*h);
    if (w >= width)
      return failure();
    bool index = isa<IndexType>(ty);
    Type nty = rewriter.getIntegerType(w);
    Location loc = op->getLoc();
    auto shrink = [&](Value x) -> Value {
      if (index)
        return arith::IndexCastOp::create(rewriter, loc, nty, x);
      return arith::TruncIOp::create(rewriter, loc, nty, x);
    };
    // Rebuilt without the original's attributes, like the sink above: keeping
    // per-site ids apart would stop CSE from merging equal rebuilt cones.
    OperationState state(loc, op->getName());
    state.addOperands({shrink(op->getOperand(0)), shrink(op->getOperand(1))});
    state.addTypes(nty);
    Value narrow = rewriter.create(state)->getResult(0);
    rewriter.replaceOp(
        op,
        index
            ? arith::IndexCastOp::create(rewriter, loc, ty, narrow).getResult()
            : arith::ExtSIOp::create(rewriter, loc, ty, narrow).getResult());
    return success();
  }
};

// Narrow an affine.for's integer iter-args to their recurrence hulls. The
// loop's signature demands the carrier width: once the carried value crosses
// the boundary narrow, its survivor register shrinks with it and the body cone
// follows through the seam casts. The result hull is the full-trip envelope, a
// superset of every value the body reads, so it is the register's width.
struct NarrowIterArgs : OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp fo,
                                PatternRewriter &rewriter) const override {
    if (fo.getInits().empty())
      return failure();
    SmallVector<Type> ntys;
    bool any = false;
    for (OpResult r : fo.getResults()) {
      ntys.push_back(r.getType());
      auto ity = dyn_cast<IntegerType>(r.getType());
      if (!ity || ity.getWidth() <= 1)
        continue;
      // A returned or re-yielded result keeps the carrier: its widening cast
      // would be stranded in a span of its own, spending a region boundary on
      // pure wiring.
      if (llvm::any_of(r.getUsers(), [](Operation *u) {
            return u->hasTrait<OpTrait::IsTerminator>();
          }))
        continue;
      std::optional<Hull> h = hullOf(r, kHullDepth);
      if (!h)
        continue;
      unsigned w = bitsOfHull(*h);
      if (w >= ity.getWidth())
        continue;
      ntys.back() = rewriter.getIntegerType(w);
      any = true;
    }
    if (!any)
      return failure();
    Location loc = fo.getLoc();
    // Both types are integer wherever they differ; the width says which cast.
    auto resize = [&](Value v, Type ty) -> Value {
      if (v.getType() == ty)
        return v;
      if (cast<IntegerType>(ty).getWidth() <
          cast<IntegerType>(v.getType()).getWidth())
        return arith::TruncIOp::create(rewriter, loc, ty, v).getResult();
      return arith::ExtSIOp::create(rewriter, loc, ty, v).getResult();
    };
    SmallVector<Value> inits;
    for (auto [init, nty] : llvm::zip(fo.getInits(), ntys))
      inits.push_back(resize(init, nty));
    auto nw = affine::AffineForOp::create(
        rewriter, loc, fo.getLowerBoundOperands(), fo.getLowerBoundMap(),
        fo.getUpperBoundOperands(), fo.getUpperBoundMap(), fo.getStepAsInt(),
        inits);
    // The directives ride the loop op (pipeline, unroll), so they move.
    nw->setDiscardableAttrs(fo->getDiscardableAttrDictionary());
    Block *body = nw.getBody();
    rewriter.setInsertionPointToStart(body);
    SmallVector<Value> repl{body->getArgument(0)};
    for (auto [k, arg] : llvm::enumerate(fo.getRegionIterArgs()))
      repl.push_back(resize(body->getArgument(k + 1), arg.getType()));
    rewriter.mergeBlocks(fo.getBody(), body, repl);
    auto yield = cast<affine::AffineYieldOp>(body->getTerminator());
    SmallVector<Value> yops;
    for (auto [v, nty] : llvm::zip(yield.getOperands(), ntys)) {
      if (v.getType() == nty) {
        yops.push_back(v);
        continue;
      }
      // The cast goes beside its producer, not at the yield: stranded after
      // the child loops it would reify as a span of its own and spend a region
      // boundary on pure wiring.
      if (Operation *def = v.getDefiningOp(); def && def->getBlock() == body)
        rewriter.setInsertionPointAfter(def);
      else
        rewriter.setInsertionPointToStart(body);
      yops.push_back(resize(v, nty));
    }
    rewriter.modifyOpInPlace(yield, [&] { yield->setOperands(yops); });
    rewriter.setInsertionPointAfter(nw);
    SmallVector<Value> results;
    for (auto [k, r] : llvm::enumerate(nw.getResults()))
      results.push_back(resize(r, fo.getResult(k).getType()));
    rewriter.replaceOp(fo, results);
    return success();
  }
};

// `x & (2^k - 1)` is a zero-extended truncation spelled as a mask. The cast
// form makes the low-bit demand explicit so the truncation can sink into the
// producer; the casts themselves are wiring.
struct MaskToTrunc : OpRewritePattern<arith::AndIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AndIOp op,
                                PatternRewriter &rewriter) const override {
    APInt mask;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&mask)) || !mask.isMask())
      return failure();
    unsigned k = mask.popcount();
    if (k == 0 || k >= carrierWidth(op.getType()))
      return failure();
    Type nty = rewriter.getIntegerType(k);
    Location loc = op.getLoc();
    if (isa<IndexType>(op.getType())) {
      Value t = arith::IndexCastOp::create(rewriter, loc, nty, op.getLhs());
      rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(op, op.getType(), t);
    } else {
      Value t = arith::TruncIOp::create(rewriter, loc, nty, op.getLhs());
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, op.getType(), t);
    }
    return success();
  }
};

// trunci(cast(x: index -> iA) -> iB)  =>  cast(x -> iB)
struct FoldTruncOfIndexCast : OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncIOp op,
                                PatternRewriter &rewriter) const override {
    Operation *inner = op.getIn().getDefiningOp();
    if (!inner || !isa<arith::IndexCastOp, arith::IndexCastUIOp>(inner) ||
        !isa<IndexType>(inner->getOperand(0).getType()))
      return failure();
    // Both steps keep the low bits, so one truncating cast does.
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, op.getType(),
                                                    inner->getOperand(0));
    return success();
  }
};

// A resize that hops through `index` is a resize: fold the pair to one direct
// cast so a truncation keeps moving toward its producer.
//   cast(cast(x: iA -> index) -> iB)  =>  trunci/ext(x -> iB)
struct FoldCastThroughIndex : RewritePattern {
  FoldCastThroughIndex(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<arith::IndexCastOp, arith::IndexCastUIOp>(op) ||
        !isa<IntegerType>(op->getResult(0).getType()))
      return failure();
    Operation *inner = op->getOperand(0).getDefiningOp();
    if (!inner || !isa<arith::IndexCastOp, arith::IndexCastUIOp>(inner))
      return failure();
    Value x = inner->getOperand(0);
    auto ity = dyn_cast<IntegerType>(x.getType());
    if (!ity)
      return failure();
    unsigned a = ity.getWidth();
    Type bty = op->getResult(0).getType();
    unsigned b = cast<IntegerType>(bty).getWidth();
    if (b == a) {
      rewriter.replaceOp(op, x);
    } else if (b < a) {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, bty, x);
    } else if (isa<arith::IndexCastOp>(inner)) {
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, bty, x);
    } else {
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, bty, x);
    }
    return success();
  }
};

struct NarrowDemandedBitsPass
    : public allo::impl::NarrowDemandedBitsPassBase<NarrowDemandedBitsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<SinkTruncThroughOp, DropRedundantMask, DropRedundantOr,
                 NarrowFromHull, NarrowIterArgs, MaskToTrunc,
                 FoldCastThroughIndex, FoldTruncOfIndexCast>(ctx);
    // The cast folds are what make the rewrite chain: without them a sunk
    // truncation stops on top of an extend instead of collapsing into it.
    arith::TruncIOp::getCanonicalizationPatterns(patterns, ctx);
    arith::ExtSIOp::getCanonicalizationPatterns(patterns, ctx);
    arith::ExtUIOp::getCanonicalizationPatterns(patterns, ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(patterns, ctx);
    arith::IndexCastUIOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
