/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/IR/ConstantRange.h"

namespace mlir::allo {
#define GEN_PASS_DEF_FLOATTOINTPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using llvm::ConstantRange;

namespace {

// The analysis works one bit wider than any integer we will emit, so a full
// 64-bit source range never wraps while its endpoints are widened and combined.
constexpr unsigned kMaxIntBW = 64;
constexpr unsigned kW = kMaxIntBW + 1;

// A demotable float cone runs from integer-to-float leaves, through the
// truncation-free float arithmetic, to a float-to-int or float-compare root.
// Each value carries the integer range it can take; a cone is demoted only when
// that range fits the float type's exactly representable integer band, which is
// the proof that no operation in it ever rounded.
struct FloatToIntPass : allo::impl::FloatToIntPassBase<FloatToIntPass> {
  using Base::Base;

  llvm::SmallPtrSet<Operation *, 16> roots;
  llvm::EquivalenceClasses<Operation *> ecs;
  llvm::MapVector<Operation *, ConstantRange> seen;
  llvm::MapVector<Operation *, Value> converted;

  static ConstantRange badRange() { return ConstantRange::getFull(kW); }
  static ConstantRange unknownRange() { return ConstantRange::getEmpty(kW); }

  void mark(Operation *op, ConstantRange r) {
    auto it = seen.find(op);
    if (it == seen.end())
      seen.insert({op, r});
    else
      it->second = r;
  }

  // The ordered and unordered forms of a predicate map to the same signed
  // integer predicate: a value out of an integer-to-float cast is never a NaN,
  // so the two agree. Predicates that only distinguish NaN have no image.
  static std::optional<arith::CmpIPredicate> mapPred(arith::CmpFPredicate p) {
    using F = arith::CmpFPredicate;
    using I = arith::CmpIPredicate;
    switch (p) {
    case F::OEQ:
    case F::UEQ:
      return I::eq;
    case F::ONE:
    case F::UNE:
      return I::ne;
    case F::OGT:
    case F::UGT:
      return I::sgt;
    case F::OGE:
    case F::UGE:
      return I::sge;
    case F::OLT:
    case F::ULT:
      return I::slt;
    case F::OLE:
    case F::ULE:
      return I::sle;
    default:
      return std::nullopt;
    }
  }

  // Float min/max of integer-valued (never-NaN) operands is a plain integer
  // min/max, so all four NaN variants collapse to one signed op.
  static bool isMinMax(Operation *op) {
    return isa<arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
               arith::MinNumFOp>(op);
  }
  static bool isInterior(Operation *op) {
    return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::NegFOp,
               arith::ExtFOp, arith::TruncFOp>(op) ||
           isMinMax(op);
  }
  static bool isRoot(Operation *op) {
    return isa<arith::FPToSIOp, arith::FPToUIOp, arith::CmpFOp>(op);
  }
  // A scalar float or integer value can carry a range; a vector cannot, and its
  // op poisons the cone.
  static bool isScalar(Type t) { return isa<FloatType, IntegerType>(t); }

  void findRoots(func::FuncOp fn) {
    fn.walk([&](Operation *op) {
      if (isa<arith::FPToSIOp, arith::FPToUIOp>(op)) {
        if (isScalar(op->getOperand(0).getType()))
          roots.insert(op);
      } else if (auto c = dyn_cast<arith::CmpFOp>(op)) {
        if (isScalar(c.getLhs().getType()) && mapPred(c.getPredicate()))
          roots.insert(op);
      }
    });
  }

  void walkBackwards() {
    SmallVector<Operation *> work(roots.begin(), roots.end());
    while (!work.empty()) {
      Operation *op = work.pop_back_val();
      if (seen.count(op))
        continue;

      // A cast into float seeds the analysis with its input integer width and
      // stops the walk; the integer operand is left untouched.
      if (auto s = dyn_cast<arith::SIToFPOp>(op)) {
        unsigned bw = s.getIn().getType().getIntOrFloatBitWidth();
        mark(op,
             bw < kW ? ConstantRange::getFull(bw).signExtend(kW) : badRange());
        continue;
      }
      if (auto u = dyn_cast<arith::UIToFPOp>(op)) {
        unsigned bw = u.getIn().getType().getIntOrFloatBitWidth();
        mark(op,
             bw < kW ? ConstantRange::getFull(bw).zeroExtend(kW) : badRange());
        continue;
      }
      if ((isInterior(op) || isRoot(op)) &&
          isScalar(op->getResult(0).getType()))
        mark(op, unknownRange());
      else {
        mark(op, badRange());
        continue;
      }

      for (Value o : op->getOperands()) {
        APFloat apf(0.0);
        if (matchPattern(o, m_ConstantFloat(&apf)))
          continue;
        if (Operation *def = o.getDefiningOp()) {
          ecs.unionSets(op, def);
          if (seen.find(op)->second != badRange())
            work.push_back(def);
        } else {
          // A block argument (loop induction var or carried value) leaves the
          // cone open; the recurrence is not demoted.
          mark(op, badRange());
        }
      }
    }
  }

  // The range of \p op from its operands, or nullopt while an operand is not
  // yet resolved. A poisoned or non-integral constant operand returns badRange.
  std::optional<ConstantRange> calcRange(Operation *op) {
    SmallVector<ConstantRange, 3> in;
    for (Value o : op->getOperands()) {
      if (Operation *def = o.getDefiningOp(); def && seen.count(def)) {
        ConstantRange r = seen.find(def)->second;
        if (r == unknownRange())
          return std::nullopt;
        in.push_back(r);
        continue;
      }
      APFloat apf(0.0);
      if (!matchPattern(o, m_ConstantFloat(&apf)))
        return badRange();
      if (!apf.isFinite() || (apf.isZero() && apf.isNegative()))
        return badRange();
      APFloat rounded = apf;
      if (rounded.roundToIntegral(APFloat::rmNearestTiesToEven) !=
              APFloat::opOK ||
          rounded != apf)
        return badRange();
      APSInt v(kW, /*isUnsigned=*/false);
      bool exact;
      auto st = apf.convertToInteger(v, APFloat::rmNearestTiesToEven, &exact);
      if (st != APFloat::opOK && st != APFloat::opInexact)
        return badRange();
      in.push_back(ConstantRange(APInt(v)));
    }
    if (isa<arith::NegFOp>(op))
      return ConstantRange(APInt::getZero(kW)).sub(in[0]);
    if (isa<arith::AddFOp>(op))
      return in[0].add(in[1]);
    if (isa<arith::SubFOp>(op))
      return in[0].sub(in[1]);
    if (isa<arith::MulFOp>(op))
      return in[0].multiply(in[1]);
    if (isa<arith::ExtFOp, arith::TruncFOp>(op))
      return in[0]; // same integer value; only the float format changes
    if (isa<arith::MaximumFOp, arith::MaxNumFOp>(op))
      return in[0].smax(in[1]);
    if (isa<arith::MinimumFOp, arith::MinNumFOp>(op))
      return in[0].smin(in[1]);
    if (isa<arith::FPToSIOp, arith::FPToUIOp>(op))
      return in[0];
    return in[0].unionWith(in[1]); // cmpf: both operands share one range
  }

  void walkForwards() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &kv : seen) {
        if (kv.second != unknownRange())
          continue;
        if (auto r = calcRange(kv.first)) {
          kv.second = *r;
          changed = true;
        }
      }
    }
  }

  static Value resize(OpBuilder &b, Location l, Value v, Type ty, bool sign) {
    unsigned s = v.getType().getIntOrFloatBitWidth();
    unsigned d = ty.getIntOrFloatBitWidth();
    if (s == d)
      return v;
    if (s < d)
      return sign ? arith::ExtSIOp::create(b, l, ty, v).getResult()
                  : arith::ExtUIOp::create(b, l, ty, v).getResult();
    return arith::TruncIOp::create(b, l, ty, v).getResult();
  }

  Value convert(Operation *op, Type toTy) {
    if (auto it = converted.find(op); it != converted.end())
      return it->second;
    OpBuilder b(op);
    Location l = op->getLoc();

    SmallVector<Value, 3> in;
    if (!isa<arith::SIToFPOp, arith::UIToFPOp>(op))
      for (Value o : op->getOperands()) {
        APFloat apf(0.0);
        if (Operation *def = o.getDefiningOp(); def && seen.count(def))
          in.push_back(convert(def, toTy));
        else if (matchPattern(o, m_ConstantFloat(&apf))) {
          APSInt v(toTy.getIntOrFloatBitWidth(), /*isUnsigned=*/false);
          bool exact;
          apf.convertToInteger(v, APFloat::rmNearestTiesToEven, &exact);
          in.push_back(
              arith::ConstantOp::create(b, l, IntegerAttr::get(toTy, v))
                  .getResult());
        }
      }

    Value nv;
    if (auto s = dyn_cast<arith::SIToFPOp>(op))
      nv = resize(b, l, s.getIn(), toTy, /*sign=*/true);
    else if (auto u = dyn_cast<arith::UIToFPOp>(op))
      nv = resize(b, l, u.getIn(), toTy, /*sign=*/false);
    else if (auto f = dyn_cast<arith::FPToSIOp>(op))
      nv = resize(b, l, in[0], f.getType(), /*sign=*/true);
    else if (auto f = dyn_cast<arith::FPToUIOp>(op))
      nv = resize(b, l, in[0], f.getType(), /*sign=*/false);
    else if (auto c = dyn_cast<arith::CmpFOp>(op))
      nv = arith::CmpIOp::create(b, l, *mapPred(c.getPredicate()), in[0], in[1])
               .getResult();
    else if (isa<arith::NegFOp>(op)) {
      Value zero = arith::ConstantOp::create(b, l, IntegerAttr::get(toTy, 0))
                       .getResult();
      nv = arith::SubIOp::create(b, l, zero, in[0]).getResult();
    } else if (isa<arith::ExtFOp, arith::TruncFOp>(op))
      nv = in[0]; // a float resize is the identity on the integer value
    else if (isa<arith::MaximumFOp, arith::MaxNumFOp>(op))
      nv = arith::MaxSIOp::create(b, l, in[0], in[1]).getResult();
    else if (isa<arith::MinimumFOp, arith::MinNumFOp>(op))
      nv = arith::MinSIOp::create(b, l, in[0], in[1]).getResult();
    else if (isa<arith::AddFOp>(op))
      nv = arith::AddIOp::create(b, l, in[0], in[1]).getResult();
    else if (isa<arith::SubFOp>(op))
      nv = arith::SubIOp::create(b, l, in[0], in[1]).getResult();
    else
      nv = arith::MulIOp::create(b, l, in[0], in[1]).getResult();

    if (roots.count(op))
      op->getResult(0).replaceAllUsesWith(nv);
    converted.insert({op, nv});
    return nv;
  }

  void validateAndTransform() {
    for (const auto *lead : ecs) {
      if (!lead->isLeader())
        continue;
      ConstantRange r = unknownRange();
      bool fail = false;
      bool convertible = false;
      for (Operation *op : ecs.members(*lead)) {
        auto it = seen.find(op);
        if (it == seen.end())
          continue;
        if (it->second.isEmptySet()) {
          fail = true; // an operand never resolved; leave the cone alone
          break;
        }
        r = r.unionWith(it->second);
        if (isRoot(op))
          continue;
        convertible = true;
        // Each op is exact only while its value fits its own float type's
        // representable integer band. Per-op is what makes mixed precision
        // sound: an f64 subexpression may exceed f32's band unless truncated to
        // f32 there.
        unsigned bits = it->second.getMinSignedBits() + 1;
        unsigned band =
            cast<FloatType>(op->getResult(0).getType()).getFPMantissaWidth() -
            1;
        if (bits > band) {
          fail = true; // the float result may have rounded; leave it float
          break;
        }
        // A non-root value must be consumed only inside the cone, or it still
        // leaves as a float and the rewrite would drop that use.
        for (Operation *user : op->getResult(0).getUsers())
          if (!seen.count(user)) {
            fail = true;
            break;
          }
        if (fail)
          break;
      }
      if (fail || !convertible || r.isFullSet() || r.isEmptySet() ||
          r.isSignWrappedSet())
        continue;

      Type toTy = IntegerType::get(&getContext(), r.getMinSignedBits() + 1);
      for (Operation *op : ecs.members(*lead))
        convert(op, toTy);
    }
  }

  void runOnOperation() override {
    roots.clear();
    ecs = {};
    seen.clear();
    converted.clear();

    findRoots(getOperation());
    walkBackwards();
    walkForwards();
    validateAndTransform();

    // The demoted float ops are dead; erase uses before defs. Their float
    // constant operands are left for the downstream canonicalize/cse.
    for (auto &kv : llvm::reverse(converted))
      kv.first->erase();
  }
};

} // namespace
