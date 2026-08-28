/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
               arith::ExtFOp, arith::TruncFOp, arith::SelectOp>(op) ||
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
        // A NaN test on integer valued operands folds to a constant, so uno and
        // ord are roots too even though they have no signed integer predicate.
        using F = arith::CmpFPredicate;
        bool nanTest = c.getPredicate() == F::UNO || c.getPredicate() == F::ORD;
        if (isScalar(c.getLhs().getType()) &&
            (mapPred(c.getPredicate()) || nanTest))
          roots.insert(op);
      }
    });
  }

  void walkBackwards(SmallVector<Operation *> work) {
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

      // A select carries the cone through its two arms; its condition is an i1
      // left untouched, so the scan skips operand 0.
      unsigned first = isa<arith::SelectOp>(op) ? 1 : 0;
      for (unsigned i = first, e = op->getNumOperands(); i < e; ++i) {
        Value o = op->getOperand(i);
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
    unsigned first = isa<arith::SelectOp>(op) ? 1 : 0;
    for (unsigned i = first, e = op->getNumOperands(); i < e; ++i) {
      Value o = op->getOperand(i);
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
    return in[0].unionWith(in[1]); // cmpf and select: the union of both arms
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
    if (!isa<arith::SIToFPOp, arith::UIToFPOp>(op)) {
      unsigned first = isa<arith::SelectOp>(op) ? 1 : 0;
      for (unsigned i = first, e = op->getNumOperands(); i < e; ++i) {
        Value o = op->getOperand(i);
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
    else if (auto c = dyn_cast<arith::CmpFOp>(op)) {
      if (auto p = mapPred(c.getPredicate()))
        nv = arith::CmpIOp::create(b, l, *p, in[0], in[1]).getResult();
      else {
        // uno is always false and ord always true on never-NaN operands.
        bool ord = c.getPredicate() == arith::CmpFPredicate::ORD;
        nv = arith::ConstantOp::create(
                 b, l, IntegerAttr::get(b.getI1Type(), ord ? 1 : 0))
                 .getResult();
      }
    } else if (isa<arith::NegFOp>(op)) {
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
    else if (auto sel = dyn_cast<arith::SelectOp>(op))
      nv = arith::SelectOp::create(b, l, sel.getCondition(), in[0], in[1])
               .getResult();
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

  //===--------------------------------------------------------------------===//
  // Loop-carried sum reductions
  //===--------------------------------------------------------------------===//

  // A float accumulator `acc += incr` (or `-=`) carried by one iter_arg. Its
  // value stays exact when the whole-trip envelope of the running sum fits the
  // accumulator's mantissa band, so it demotes to an integer recurrence.
  struct Reduction {
    unsigned idx;
    Operation *op; // the addf/subf producing the next accumulator value
    Value incr;    // the per-iteration addend
    bool sub;      // acc - incr rather than acc + incr
    Type toTy;     // the integer carrier the envelope fits
    int64_t init;  // the constant integer the accumulator starts from
  };

  // Build the integer form of one increment-cone op, reading operands already
  // mapped to their integer clones; a float constant becomes an integer one.
  Value demoteInto(OpBuilder &b, Location l, Operation *op, Type toTy,
                   IRMapping &map) {
    auto opnd = [&](unsigned k) -> Value {
      Value o = op->getOperand(k);
      APFloat apf(0.0);
      if (matchPattern(o, m_ConstantFloat(&apf))) {
        APSInt v(toTy.getIntOrFloatBitWidth(), /*isUnsigned=*/false);
        bool exact;
        apf.convertToInteger(v, APFloat::rmNearestTiesToEven, &exact);
        return arith::ConstantOp::create(b, l, IntegerAttr::get(toTy, v))
            .getResult();
      }
      return map.lookupOrDefault(o);
    };
    if (isa<arith::SIToFPOp>(op))
      return resize(b, l, opnd(0), toTy, /*sign=*/true);
    if (isa<arith::UIToFPOp>(op))
      return resize(b, l, opnd(0), toTy, /*sign=*/false);
    if (isa<arith::NegFOp>(op)) {
      Value z = arith::ConstantOp::create(b, l, IntegerAttr::get(toTy, 0))
                    .getResult();
      return arith::SubIOp::create(b, l, z, opnd(0)).getResult();
    }
    if (isa<arith::ExtFOp, arith::TruncFOp>(op))
      return opnd(0);
    if (isa<arith::MaximumFOp, arith::MaxNumFOp>(op))
      return arith::MaxSIOp::create(b, l, opnd(0), opnd(1)).getResult();
    if (isa<arith::MinimumFOp, arith::MinNumFOp>(op))
      return arith::MinSIOp::create(b, l, opnd(0), opnd(1)).getResult();
    if (isa<arith::AddFOp>(op))
      return arith::AddIOp::create(b, l, opnd(0), opnd(1)).getResult();
    if (isa<arith::SubFOp>(op))
      return arith::SubIOp::create(b, l, opnd(0), opnd(1)).getResult();
    if (auto sel = dyn_cast<arith::SelectOp>(op))
      return arith::SelectOp::create(b, l,
                                     map.lookupOrDefault(sel.getCondition()),
                                     opnd(1), opnd(2))
          .getResult();
    return arith::MulIOp::create(b, l, opnd(0), opnd(1)).getResult();
  }

  // The integer range of the increment cone rooted at \p incr, and its ops in
  // \p cone, or nullopt when it is not a body-local integer-valued float cone.
  std::optional<ConstantRange>
  incrementCone(Value incr, affine::AffineForOp fo,
                SmallVectorImpl<Operation *> &cone) {
    Operation *root = incr.getDefiningOp();
    if (!root)
      return std::nullopt;
    roots.clear();
    ecs = {};
    seen.clear();
    converted.clear();
    walkBackwards({root});
    walkForwards();
    for (auto &kv : seen) {
      Operation *op = kv.first;
      if (kv.second.isEmptySet() || kv.second.isFullSet())
        return std::nullopt; // an operand never resolved, or the cone escaped
      if (op->getBlock() != fo.getBody() ||
          !isa<FloatType>(op->getResult(0).getType()))
        return std::nullopt;
      unsigned bits = kv.second.getMinSignedBits() + 1;
      unsigned band =
          cast<FloatType>(op->getResult(0).getType()).getFPMantissaWidth() - 1;
      if (bits > band)
        return std::nullopt;
      cone.push_back(op);
    }
    return seen.find(root)->second;
  }

  // The signed bit width [lo, hi] needs, or nullopt past \p band. lo <= hi.
  static std::optional<unsigned> hullBits(__int128 lo, __int128 hi,
                                          unsigned band) {
    auto sbits = [](__int128 v) {
      __uint128_t mag = v < 0 ? (__uint128_t)(~v) : (__uint128_t)v;
      unsigned b = 0;
      while (mag) {
        ++b;
        mag >>= 1;
      }
      return b + 1; // one sign bit
    };
    unsigned bits = std::max(sbits(lo), sbits(hi));
    return bits > band ? std::nullopt : std::optional<unsigned>(bits);
  }

  // Rebuild \p fo carrying every reduction in \p reds as an integer iter_arg:
  // the increment cones and accumulator updates rebuild in integer, everything
  // else clones, and each float result feeds its `fptosi`/`fptoui` as a resize.
  void rebuildLoop(affine::AffineForOp fo, ArrayRef<Reduction> reds,
                   DenseMap<Operation *, Type> &coneTy) {
    OpBuilder b(fo);
    Location loc = fo.getLoc();
    unsigned n = fo.getRegionIterArgs().size();
    DenseMap<unsigned, const Reduction *> byIdx;
    DenseMap<Operation *, const Reduction *> redOp;
    for (const Reduction &r : reds) {
      byIdx[r.idx] = &r;
      redOp[r.op] = &r;
    }
    SmallVector<Value> inits;
    for (unsigned i = 0; i < n; ++i)
      if (const Reduction *r = byIdx.lookup(i))
        inits.push_back(arith::ConstantOp::create(
                            b, loc, IntegerAttr::get(r->toTy, r->init))
                            .getResult());
      else
        inits.push_back(fo.getInits()[i]);

    auto yield = cast<affine::AffineYieldOp>(fo.getBody()->getTerminator());
    auto nw = affine::AffineForOp::create(
        b, loc, fo.getLowerBoundOperands(), fo.getLowerBoundMap(),
        fo.getUpperBoundOperands(), fo.getUpperBoundMap(), fo.getStepAsInt(),
        inits, [&](OpBuilder &nb, Location nloc, Value niv, ValueRange accs) {
          IRMapping map;
          map.map(fo.getInductionVar(), niv);
          for (unsigned i = 0; i < n; ++i)
            map.map(fo.getRegionIterArgs()[i], accs[i]);
          SmallVector<Value> yields(n);
          for (Operation &o : fo.getBody()->without_terminator()) {
            if (const Reduction *r = redOp.lookup(&o)) {
              Value inc = map.lookupOrDefault(r->incr);
              yields[r->idx] =
                  r->sub ? arith::SubIOp::create(nb, nloc, accs[r->idx], inc)
                               .getResult()
                         : arith::AddIOp::create(nb, nloc, accs[r->idx], inc)
                               .getResult();
            } else if (Type ty = coneTy.lookup(&o))
              map.map(o.getResult(0), demoteInto(nb, nloc, &o, ty, map));
            else
              nb.clone(o, map);
          }
          for (unsigned i = 0; i < n; ++i)
            if (!byIdx.count(i))
              yields[i] = map.lookupOrDefault(yield.getOperand(i));
          affine::AffineYieldOp::create(nb, nloc, yields);
        });
    nw->setDiscardableAttrs(fo->getDiscardableAttrDictionary());

    b.setInsertionPointAfter(nw);
    for (unsigned i = 0; i < n; ++i) {
      Value nr = nw.getResult(i);
      if (!byIdx.count(i)) {
        fo.getResult(i).replaceAllUsesWith(nr);
        continue;
      }
      for (Operation *user :
           llvm::make_early_inc_range(fo.getResult(i).getUsers())) {
        OpBuilder ub(user);
        Value rz = resize(ub, user->getLoc(), nr, user->getResult(0).getType(),
                          /*sign=*/isa<arith::FPToSIOp>(user));
        user->getResult(0).replaceAllUsesWith(rz);
        user->erase();
      }
    }
    fo.erase();
  }

  // Demote every qualifying float sum reduction of the flat loop \p fo. A loop
  // demotes only when all of its float ops belong to a demoted reduction, so no
  // float value is left dangling behind an integer carrier.
  void demoteLoop(affine::AffineForOp fo) {
    bool nested = false;
    fo.getBody()->walk([&](affine::AffineForOp) { nested = true; });
    if (nested)
      return;
    std::optional<uint64_t> trip = affine::getConstantTripCount(fo);
    if (!trip || *trip == 0 || *trip > (uint64_t(1) << 32))
      return;

    auto yield = cast<affine::AffineYieldOp>(fo.getBody()->getTerminator());
    SmallVector<Reduction> reds;
    DenseMap<Operation *, Type> coneTy;
    unsigned n = fo.getRegionIterArgs().size();
    for (unsigned idx = 0; idx < n; ++idx) {
      Value acc = fo.getRegionIterArgs()[idx];
      if (!isa<FloatType>(acc.getType()) || !acc.hasOneUse())
        continue;
      Operation *rop = yield.getOperand(idx).getDefiningOp();
      if (!rop || rop != *acc.getUsers().begin() || !rop->hasOneUse())
        continue; // the next value is the yield's alone, nothing else reads it
      bool sub = isa<arith::SubFOp>(rop);
      if (!isa<arith::AddFOp>(rop) && !sub)
        continue;
      Value incr;
      if (sub) {
        if (rop->getOperand(0) != acc)
          continue; // acc - incr only; incr - acc is not an accumulator
        incr = rop->getOperand(1);
      } else if (rop->getOperand(0) == acc)
        incr = rop->getOperand(1);
      else if (rop->getOperand(1) == acc)
        incr = rop->getOperand(0);
      else
        continue;

      APFloat apf(0.0);
      if (!matchPattern(fo.getInits()[idx], m_ConstantFloat(&apf)) ||
          !apf.isFinite() || (apf.isZero() && apf.isNegative()))
        continue;
      APFloat rounded = apf;
      if (rounded.roundToIntegral(APFloat::rmNearestTiesToEven) !=
              APFloat::opOK ||
          rounded != apf)
        continue;
      APSInt initI(kW, /*isUnsigned=*/false);
      bool exact;
      apf.convertToInteger(initI, APFloat::rmNearestTiesToEven, &exact);
      if (initI.getSignificantBits() > 63)
        continue;

      SmallVector<Operation *> cone;
      std::optional<ConstantRange> ir = incrementCone(incr, fo, cone);
      if (!ir || ir->getMinSignedBits() > 63)
        continue;
      int64_t dLo = ir->getSignedMin().getSExtValue();
      int64_t dHi = ir->getSignedMax().getSExtValue();
      if (sub)
        std::tie(dLo, dHi) = std::make_pair(-dHi, -dLo);
      __int128 base = initI.getSExtValue();
      __int128 lo = base + (__int128)*trip * std::min<__int128>(dLo, 0);
      __int128 hi = base + (__int128)*trip * std::max<__int128>(dHi, 0);
      unsigned band = cast<FloatType>(acc.getType()).getFPMantissaWidth() - 1;
      std::optional<unsigned> bits = hullBits(lo, hi, band);
      if (!bits)
        continue;

      Type toTy = IntegerType::get(&getContext(), *bits);
      bool overlap = false;
      for (Operation *op : cone)
        overlap |= coneTy.count(op) > 0;
      if (overlap)
        continue; // a cone shared with another accumulator is left alone
      for (Operation *op : cone)
        coneTy[op] = toTy;
      reds.push_back({idx, rop, incr, sub, toTy, initI.getSExtValue()});
    }
    if (reds.empty())
      return;

    DenseSet<Operation *> redOps;
    for (const Reduction &r : reds)
      redOps.insert(r.op);
    // Every float op in the body must fold into a demoted reduction, or the
    // rebuild would strand a float value on an integer carrier. A stray float
    // op leaves the loop alone.
    for (Operation &o : fo.getBody()->without_terminator())
      if ((isInterior(&o) || isRoot(&o)) && !coneTy.count(&o) &&
          !redOps.count(&o))
        return;
    // A demoted value may be read only inside its cone, never escape as a
    // second iter_arg or a store where its integer form would not fit.
    for (auto &kv : coneTy)
      for (Operation *u : kv.first->getResult(0).getUsers())
        if (!coneTy.count(u) && !redOps.count(u))
          return;
    // Each loop result must leave as a float-to-int, the only user a resize can
    // absorb.
    for (const Reduction &r : reds)
      for (Operation *user : fo.getResult(r.idx).getUsers())
        if (!isa<arith::FPToSIOp, arith::FPToUIOp>(user))
          return;

    rebuildLoop(fo, reds, coneTy);
  }

  void demoteReductions(func::FuncOp fn) {
    SmallVector<affine::AffineForOp> loops;
    fn.walk([&](affine::AffineForOp fo) { loops.push_back(fo); });
    for (affine::AffineForOp fo : loops)
      demoteLoop(fo);
  }

  void runOnOperation() override {
    roots.clear();
    ecs = {};
    seen.clear();
    converted.clear();

    findRoots(getOperation());
    walkBackwards({roots.begin(), roots.end()});
    walkForwards();
    validateAndTransform();

    // The demoted float ops are dead; erase uses before defs. Their float
    // constant operands are left for the downstream canonicalize/cse.
    for (auto &kv : llvm::reverse(converted))
      kv.first->erase();

    demoteReductions(getOperation());
  }
};

} // namespace
