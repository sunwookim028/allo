/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/AddressModel.h"

#include "allo/Scheduling/MemoryAccess.h" // asMemAccess
#include "allo/Scheduling/MemoryModel.h"  // linearizeAccessMap

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h" // getConstantIntValue
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace mlir;
using namespace mlir::allo;

AddressDelays mlir::allo::addressDelaysOf(const OperatorLibrary &lib) {
  AddressDelays d;
  d.add = lib.combMarginalDelay(OpKind::Add, AddressDelays::refWidth);
  d.mul = lib.combMarginalDelay(OpKind::Mul, AddressDelays::refWidth);
  d.div = lib.combMarginalDelay(OpKind::Div, AddressDelays::refWidth);
  return d;
}

// A carry chain's delay tracks its width, so a cone narrowed from the
// characterization width scales with it. Wiring-only forms never reach here:
// they cost no logic at any width.
static double scaled(double base, unsigned width) {
  return base * width / AddressDelays::refWidth;
}

// Nonzero digits of the non-adjacent form of \p v: the fewest signed powers of
// two that sum to it, which is how synthesis recodes a constant multiply
// (`x*15` is `(x << 4) - x`, one adder, against three for the binary form).
static unsigned nafWeight(uint64_t v) {
  unsigned w = 0;
  while (v) {
    if (v & 1) {
      // Choose the digit's sign so the next digit is forced to zero.
      v = (v & 2) ? v + 1 : v - 1;
      ++w;
    }
    v >>= 1;
  }
  return w;
}

uint64_t mlir::allo::magicMultiplier(uint64_t d, unsigned w, unsigned &shift) {
  shift = w + llvm::Log2_64_Ceil(d);
  llvm::APInt num = llvm::APInt::getOneBitSet(shift + 1, shift);
  return (num + (d - 1)).udiv(llvm::APInt(shift + 1, d)).getZExtValue();
}

// Adders and levels of the shift-add network a positive constant multiply
// recodes to, at \p width.
static std::pair<unsigned, double>
constMulCost(uint64_t k, const AddressDelays &delays, unsigned width) {
  unsigned terms = nafWeight(k);
  unsigned adds = terms ? terms - 1 : 0;
  double delay =
      adds ? std::max(1u, llvm::Log2_64_Ceil(terms)) * scaled(delays.add, width)
           : 0.0;
  return {adds, delay};
}

AddressCost mlir::allo::addressCost(AffineExpr e, const AddressDelays &delays,
                                    unsigned width) {
  // A leaf costs nothing: a constant is wiring, a dim / symbol arrives from
  // elsewhere already priced.
  if (isa<AffineConstantExpr, AffineDimExpr, AffineSymbolExpr>(e))
    return {};

  auto bin = cast<AffineBinaryOpExpr>(e);
  auto konst = dyn_cast<AffineConstantExpr>(bin.getRHS());
  // A divider is not a homomorphism modulo 2^width, so it and everything
  // feeding it are carried at the full datapath width whatever the address
  // needs; only its result may be truncated.
  bool isDiv = e.getKind() == AffineExprKind::FloorDiv ||
               e.getKind() == AffineExprKind::CeilDiv ||
               e.getKind() == AffineExprKind::Mod;
  unsigned below = isDiv ? AddressDelays::refWidth : width;
  // With one exception: `x mod 2^k` IS the low k bits, and `+`, `-` and
  // constant `*` under it are congruent modulo 2^k, so that subtree is carried
  // at k bits however wide the cone is.
  if (e.getKind() == AffineExprKind::Mod && konst && konst.getValue() > 1 &&
      llvm::isPowerOf2_64(static_cast<uint64_t>(konst.getValue())))
    below = std::min<unsigned>(
        width, llvm::Log2_64(static_cast<uint64_t>(konst.getValue())));
  AddressCost lhs = addressCost(bin.getLHS(), delays, below);
  AddressCost rhs = addressCost(bin.getRHS(), delays, below);
  AddressCost c;
  c.adders = lhs.adders + rhs.adders;
  c.multipliers = lhs.multipliers + rhs.multipliers;
  c.dividers = lhs.dividers + rhs.dividers;
  c.reciprocals = lhs.reciprocals + rhs.reciprocals;
  // Two operands converge here, so the path through this node is the LONGER of
  // them plus this node's own delay.
  double in = std::max(lhs.delay, rhs.delay);

  switch (e.getKind()) {
  case AffineExprKind::Add:
    ++c.adders;
    c.delay = in + scaled(delays.add, width);
    return c;

  case AffineExprKind::Mul: {
    // A non-constant coefficient is a genuine multiplier: unreachable from an
    // affine.load's map, but a semi-affine map is representable.
    if (!konst) {
      ++c.multipliers;
      c.delay = in + scaled(delays.mul, width);
      return c;
    }
    int64_t k = konst.getValue();
    if (k == 0)
      return {}; // the term vanishes
    // Signed-digit shift-add: the shifts are wiring, so only the summing
    // network costs. A lone digit is a bare wire, unless negative, which still
    // needs the two's-complement subtract.
    unsigned terms = nafWeight(static_cast<uint64_t>(std::abs(k)));
    unsigned adds = std::max(terms - 1, k < 0 ? 1u : 0u);
    c.adders += adds;
    c.delay = in + (adds ? std::max(1u, llvm::Log2_64_Ceil(terms)) : 0u) *
                       scaled(delays.add, width);
    return c;
  }

  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod:
    // A power-of-two divisor is a shift or a mask, which is wiring (`divConst`
    // and `modConst` lower it that way).
    if (konst && konst.getValue() > 0 &&
        llvm::isPowerOf2_64(static_cast<uint64_t>(konst.getValue()))) {
      c.delay = in;
      return c;
    }
    // Any other constant is the reciprocal multiply `divConst` builds: the
    // multiplier's shift-adds at the product width, the shift wiring. A
    // residue adds the divisor's own multiply and the subtract; a ceildiv is
    // priced as the pre-biased floordiv.
    if (konst && konst.getValue() > 1 &&
        konst.getValue() < (int64_t(1) << AddressDelays::refWidth)) {
      uint64_t d = static_cast<uint64_t>(konst.getValue());
      unsigned shift;
      uint64_t magic = magicMultiplier(d, AddressDelays::refWidth, shift);
      auto [adds, mulDelay] =
          constMulCost(magic, delays, 2 * AddressDelays::refWidth + 1);
      ++c.reciprocals;
      c.adders += adds;
      c.delay = in + mulDelay;
      if (e.getKind() == AffineExprKind::Mod) {
        auto [dAdds, dDelay] = constMulCost(d, delays, AddressDelays::refWidth);
        c.adders += dAdds + 1;
        c.delay += dDelay + scaled(delays.add, AddressDelays::refWidth);
      }
      if (e.getKind() == AffineExprKind::CeilDiv) {
        ++c.adders;
        c.delay += scaled(delays.add, AddressDelays::refWidth);
      }
      return c;
    }
    ++c.dividers;
    c.delay = in + scaled(delays.div, AddressDelays::refWidth);
    return c;

  default:
    break;
  }
  assert(false && "an affine leaf kind reached the binary-operator switch");
  return c;
}

AddressCost mlir::allo::addressCost(AffineMap map, ArrayRef<int64_t> shape,
                                    const AddressDelays &delays,
                                    unsigned width) {
  if (!map)
    return {};
  return addressCost(linearizeAccessMap(map, shape).getResult(0), delays,
                     width);
}

// The operand \p e names, in the dims-then-symbols numbering, or nullopt.
static std::optional<unsigned> operandOf(AffineExpr e, unsigned numDims) {
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return d.getPosition();
  if (auto s = dyn_cast<AffineSymbolExpr>(e))
    return numDims + s.getPosition();
  return std::nullopt;
}

// Flatten \p e into `sum(coeff * term) + constant`, accumulating into \p out
// and \p konst. Anything that is not a sum or a constant multiple is a term,
// `floordiv` and `mod` included: they are opaque here, never descended into.
static void sumTerms(AffineExpr e, int64_t scale,
                     SmallVectorImpl<std::pair<AffineExpr, int64_t>> &out,
                     int64_t &konst) {
  if (auto c = dyn_cast<AffineConstantExpr>(e)) {
    konst += scale * c.getValue();
    return;
  }
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
    if (bin.getKind() == AffineExprKind::Add) {
      sumTerms(bin.getLHS(), scale, out, konst);
      sumTerms(bin.getRHS(), scale, out, konst);
      return;
    }
    if (auto k = dyn_cast<AffineConstantExpr>(bin.getRHS());
        k && bin.getKind() == AffineExprKind::Mul) {
      sumTerms(bin.getLHS(), scale * k.getValue(), out, konst);
      return;
    }
  }
  out.push_back({e, scale});
}

// \p e as `scale * operand + offset` over exactly ONE carried operand, the
// argument a digit is taken of. The scale may be NEGATIVE: a register counts
// down as cheaply as it counts up.
static std::optional<SplitAddress::Term>
asLinearCounter(AffineExpr e, unsigned numDims, CarriedFn carried) {
  SmallVector<std::pair<AffineExpr, int64_t>> terms;
  int64_t konst = 0;
  sumTerms(e, 1, terms, konst);
  if (terms.size() != 1 || !terms[0].second)
    return std::nullopt;
  std::optional<unsigned> p = operandOf(terms[0].first, numDims);
  if (!p || !carried(*p))
    return std::nullopt;
  SplitAddress::Term t;
  t.operand = *p;
  t.coeff = 1;
  t.scale = terms[0].second;
  t.offset = konst;
  return t;
}

static std::optional<SplitAddress::Term> asDigit(AffineExpr e, unsigned numDims,
                                                 CarriedFn carried);

// \p e as `digit + c`, where the digit is a QUOTIENT with no residue over it:
// `(x floordiv D) + c` is `(x + c*D) floordiv D`, so the constant rides in the
// digit's own offset and needs no second register. A digit that already carries
// a residue cannot absorb one, since adding after a wrap is not adding before.
static std::optional<SplitAddress::Term>
asShiftedDigit(AffineExpr e, unsigned numDims, CarriedFn carried) {
  SmallVector<std::pair<AffineExpr, int64_t>> terms;
  int64_t konst = 0;
  sumTerms(e, 1, terms, konst);
  if (terms.size() != 1 || terms[0].second != 1)
    return std::nullopt;
  std::optional<SplitAddress::Term> t =
      asDigit(terms[0].first, numDims, carried);
  if (!t || t->modulus)
    return std::nullopt;
  t->offset += konst * t->divisor;
  return t;
}

// \p e as one PERIODIC term, `(scale*operand + offset) floordiv divisor mod
// modulus`, folding the nesting a delinearized nest arrives in:
//
//   `(x floordiv a) floordiv b`      is `x floordiv (a*b)`
//   `((x floordiv a) mod m) floordiv b` is `(x floordiv (a*b)) mod (m/b)`, b |
//   m
//   `((x floordiv a) mod m) mod k`   is `(x floordiv a) mod k`, k | m
//   `(x floordiv a) + c`             is `(x + c*a) floordiv a`
//
// so an arbitrarily deep chain comes out as one (divisor, modulus) pair and
// costs one register. A step that could cross two multiples of the divisor in
// one iteration is refused, since the register wraps by subtracting once; that
// test lives here so the price and the build agree on which terms reduce.
static std::optional<SplitAddress::Term> asDigit(AffineExpr e, unsigned numDims,
                                                 CarriedFn carried) {
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return std::nullopt;
  bool isDiv = bin.getKind() == AffineExprKind::FloorDiv;
  if (!isDiv && bin.getKind() != AffineExprKind::Mod)
    return std::nullopt;
  auto k = dyn_cast<AffineConstantExpr>(bin.getRHS());
  if (!k || k.getValue() <= 0)
    return std::nullopt;
  int64_t v = k.getValue();
  std::optional<SplitAddress::Term> t = asDigit(bin.getLHS(), numDims, carried);
  if (!t)
    t = asLinearCounter(bin.getLHS(), numDims, carried);
  if (!t)
    t = asShiftedDigit(bin.getLHS(), numDims, carried);
  if (!t)
    return std::nullopt;
  if (t->modulus && t->modulus % v)
    return std::nullopt; // neither identity above applies
  if (isDiv) {
    t->divisor *= v;
    t->modulus /= v; // 0 stays 0: a quotient with no residue over it
  } else {
    t->modulus = v;
  }
  // The register wraps ONCE per advance, in whichever direction, so what is
  // bounded is the magnitude: with a divisor the QUOTIENT must advance by at
  // most one, so the divisor bounds it; with none, the modulus does.
  int64_t bound = t->divisor > 1 ? t->divisor : t->modulus;
  if (std::abs(t->scale * *carried(t->operand)) > bound)
    return std::nullopt;
  return t;
}

// Whether a register can carry any part of \p e: a carried operand reached
// through sums and constant multiples alone, the two things a constant
// per-iteration difference distributes over, or a DIGIT of one (`asDigit`).
static bool reducible(AffineExpr e, unsigned numDims, CarriedFn carried) {
  if (std::optional<unsigned> p = operandOf(e, numDims))
    return carried(*p).has_value();
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return false;
  if (bin.getKind() == AffineExprKind::Add)
    return reducible(bin.getLHS(), numDims, carried) ||
           reducible(bin.getRHS(), numDims, carried);
  if (bin.getKind() == AffineExprKind::Mul)
    return isa<AffineConstantExpr>(bin.getRHS()) &&
           reducible(bin.getLHS(), numDims, carried);
  return asDigit(e, numDims, carried).has_value();
}

// Replace each maximal DIGIT inside \p e with a symbol standing for the
// register that holds it, recording them in `out.reads`. Pre-order, so the
// largest reducible subtree wins and nothing inside it is pulled out twice.
//
// A cheap operator over an expensive digit comes out cheap: evaluated whole,
// `(x mod 5) floordiv 2` is two real dividers, where over a register it is a
// shift over a wire.
static AffineExpr substituteDigits(AffineExpr e, unsigned numDims,
                                   unsigned numSymbols, CarriedFn carried,
                                   SplitAddress &out) {
  if (std::optional<SplitAddress::Term> d = asDigit(e, numDims, carried)) {
    d->coeff = 1;
    out.reads.push_back(*d);
    return getAffineSymbolExpr(numSymbols + out.reads.size() - 1,
                               e.getContext());
  }
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return e;
  return getAffineBinaryOpExpr(
      bin.getKind(),
      substituteDigits(bin.getLHS(), numDims, numSymbols, carried, out),
      substituteDigits(bin.getRHS(), numDims, numSymbols, carried, out));
}

// Accumulate `scale * e` into \p out. Descend only where something reduces, so
// a subtree that reduces nothing keeps its own shape.
static void collectSplit(AffineExpr e, int64_t scale, unsigned numDims,
                         unsigned numSymbols, CarriedFn carried,
                         SplitAddress &out) {
  auto toResidual = [&](AffineExpr sub) {
    AffineExpr r =
        substituteDigits(sub * scale, numDims, numSymbols, carried, out);
    out.residual = out.residual ? out.residual + r : r;
  };
  if (auto c = dyn_cast<AffineConstantExpr>(e)) {
    out.base += scale * c.getValue();
    return;
  }
  if (!reducible(e, numDims, carried))
    return toResidual(e);
  if (std::optional<unsigned> p = operandOf(e, numDims)) {
    out.terms.push_back({*p, scale});
    return;
  }
  // A digit is a leaf, so there is nothing under it to distribute the scale
  // into. A non-positive multiple is READ instead of summed, to keep the
  // divider off the path: the register cannot hold a signed residue.
  if (std::optional<SplitAddress::Term> d = asDigit(e, numDims, carried)) {
    if (scale <= 0)
      return toResidual(e);
    d->coeff = scale;
    out.terms.push_back(*d);
    return;
  }
  auto bin = cast<AffineBinaryOpExpr>(e);
  if (bin.getKind() == AffineExprKind::Add) {
    collectSplit(bin.getLHS(), scale, numDims, numSymbols, carried, out);
    collectSplit(bin.getRHS(), scale, numDims, numSymbols, carried, out);
    return;
  }
  collectSplit(bin.getLHS(),
               scale * cast<AffineConstantExpr>(bin.getRHS()).getValue(),
               numDims, numSymbols, carried, out);
}

SplitAddress mlir::allo::splitAddress(AffineExpr e, unsigned numDims,
                                      unsigned numSymbols, CarriedFn carried) {
  SplitAddress addr;
  if (e)
    collectSplit(e, 1, numDims, numSymbols, carried, addr);
  return addr;
}

// Mixed-radix digit extraction, rewritten to divide before it masks:
// `(x mod (a*b)) floordiv b` and `(x floordiv b) mod a` are the same digit, and
// a bank decomposition wants the second, since `a` is usually a power-of-two
// partition factor, turning the outer division into a mask. A coalesced nest
// hands over the first form, which `simplifyAffineExpr` leaves alone.
static AffineExpr divideBeforeMasking(AffineExpr e) {
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return e;
  e = getAffineBinaryOpExpr(bin.getKind(), divideBeforeMasking(bin.getLHS()),
                            divideBeforeMasking(bin.getRHS()));
  bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin || bin.getKind() != AffineExprKind::FloorDiv)
    return e;
  auto inner = dyn_cast<AffineBinaryOpExpr>(bin.getLHS());
  auto b = dyn_cast<AffineConstantExpr>(bin.getRHS());
  if (!inner || inner.getKind() != AffineExprKind::Mod || !b ||
      b.getValue() <= 0)
    return e;
  auto m = dyn_cast<AffineConstantExpr>(inner.getRHS());
  if (!m || m.getValue() <= 0 || m.getValue() % b.getValue() != 0)
    return e;
  return inner.getLHS().floorDiv(b.getValue()) % (m.getValue() / b.getValue());
}

// A residue reads its operand's coefficients modulo the divisor:
// `(a*x + c) mod k` is `((a mod k)*x + c) mod k`, so a term whose coefficient
// is a multiple of `k` contributes nothing and is dropped.
//
// Affine map composition flattens `x mod 2` into `x - (x floordiv 2)*2` before
// this file sees anything, and re-simplifying does not recover the mask. Under
// a skewed layout that lands inside the bank digit, where the coefficients are
// multiples of the factor: a coalesced 8x8 skew reads
// `(d0 floordiv 2 + d0*4 - (d0 floordiv 2)*8 + 1) mod 4`, four chained
// operators, where `(d0 floordiv 2 + 1) mod 4` is one.
static AffineExpr reduceModCoefficients(AffineExpr e) {
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return e;
  e = getAffineBinaryOpExpr(bin.getKind(), reduceModCoefficients(bin.getLHS()),
                            reduceModCoefficients(bin.getRHS()));
  bin = dyn_cast<AffineBinaryOpExpr>(e);
  auto k = bin ? dyn_cast<AffineConstantExpr>(bin.getRHS()) : nullptr;
  if (!bin || bin.getKind() != AffineExprKind::Mod || !k || k.getValue() <= 0)
    return e;
  SmallVector<std::pair<AffineExpr, int64_t>> terms;
  int64_t konst = 0;
  sumTerms(bin.getLHS(), 1, terms, konst);
  int64_t f = k.getValue();
  auto residue = [f](int64_t v) { return ((v % f) + f) % f; };
  AffineExpr sum = getAffineConstantExpr(residue(konst), e.getContext());
  for (auto &[t, coeff] : terms)
    if (int64_t r = residue(coeff))
      sum = sum + t * r;
  return sum % f;
}

// `x - (x floordiv k)*k` is `x mod k`, re-folded.
//
// `simplifyAffineExpr` FLATTENS a residue into that difference, which is a
// subtract and a shift nothing can carry, where `x mod k` is a digit of a
// counter and so a register (`asDigit`).
static AffineExpr refoldResidues(AffineExpr e) {
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return e;
  e = getAffineBinaryOpExpr(bin.getKind(), refoldResidues(bin.getLHS()),
                            refoldResidues(bin.getRHS()));
  bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin || bin.getKind() != AffineExprKind::Add)
    return e;
  SmallVector<std::pair<AffineExpr, int64_t>> terms;
  int64_t konst = 0;
  sumTerms(e, 1, terms, konst);
  bool folded = false;
  for (auto &[dividend, c] : terms) {
    if (!c)
      continue;
    for (auto &[quotient, q] : terms) {
      auto d = dyn_cast<AffineBinaryOpExpr>(quotient);
      if (!q || !d || d.getKind() != AffineExprKind::FloorDiv ||
          d.getLHS() != dividend)
        continue;
      auto k = dyn_cast<AffineConstantExpr>(d.getRHS());
      if (!k || k.getValue() <= 0 || q != -c * k.getValue())
        continue;
      dividend = dividend % k.getValue();
      q = 0;
      folded = true;
      break;
    }
  }
  if (!folded)
    return e;
  AffineExpr sum = getAffineConstantExpr(konst, e.getContext());
  for (auto &[t, c] : terms)
    if (c)
      sum = sum + t * c;
  return sum;
}

AffineExpr mlir::allo::simplifiedForHardware(AffineExpr e, unsigned numDims,
                                             unsigned numSymbols) {
  // Ratios, not nanoseconds: every layer must agree on one form, and only
  // `addressCostOf` holds a device to price against.
  static constexpr AddressDelays kRank{/*add=*/1.0, /*mul=*/4.0, /*div=*/16.0};
  AffineExpr best = e;
  double bestRank = addressCost(e, kRank, AddressDelays::refWidth).delay;
  for (AffineExpr cand :
       {simplifyAffineExpr(e, numDims, numSymbols), divideBeforeMasking(e),
        reduceModCoefficients(e), refoldResidues(e)}) {
    double rank = addressCost(cand, kRank, AddressDelays::refWidth).delay;
    if (rank < bestRank) {
      best = cand;
      bestRank = rank;
    }
  }
  return best;
}

AffineExpr mlir::allo::applyExprOf(AffineMap map) {
  assert(map.getNumResults() == 1 && "affine.apply yields one result");
  return simplifiedForHardware(map.getResult(0), map.getNumDims(),
                               map.getNumSymbols());
}

AddressExprs mlir::allo::addressExprsOf(const BankLayout &layout, AffineMap map,
                                        ArrayRef<int64_t> shape,
                                        std::optional<unsigned> assignedBank) {
  // With no partitioned axis the split's `offset` IS the row-major linear index
  // and its `bank` the constant 0, so an unbanked memref needs no branch.
  BankSplitExpr split = bankSplitOf(layout, map, shape);
  AddressExprs e;
  e.offset = split.offset;
  // A compile-time bank routes straight to its own memory and never reads the
  // digit. A skewed access holds a SLOT, whose physical bank is that slot
  // rotated at run time, so it builds the digit even when assigned.
  if (layout.numBanks > 1 && (!assignedBank || layout.skew()))
    e.bank = split.bank;
  e.width = addressWidthOf(layout.bankShape);
  return e;
}

AddressCost mlir::allo::splitAddressCost(const SplitAddress &addr,
                                         const AddressDelays &delays,
                                         unsigned width) {
  // One bare register per term, then the residual: each input past the first
  // costs one adder.
  AddressCost c;
  c.carried = addr.terms.size() + addr.reads.size();
  unsigned chain = addr.terms.size();
  unsigned inputs = chain + (addr.residual ? 1 : 0);
  c.adders = inputs ? inputs - 1 : 0;
  c.delay = chain ? (chain - 1) * scaled(delays.add, width) : 0.0;
  if (!addr.residual)
    return c;
  AddressCost r = addressCost(addr.residual, delays, width);
  c.adders += r.adders;
  c.multipliers += r.multipliers;
  c.dividers += r.dividers;
  c.reciprocals += r.reciprocals;
  c.delay =
      chain ? std::max(c.delay, r.delay) + scaled(delays.add, width) : r.delay;
  return c;
}

unsigned mlir::allo::addressWidthOf(ArrayRef<int64_t> shape) {
  int64_t elements = 1;
  for (int64_t d : shape)
    elements *= d;
  // The two spare-word rule `declaredDepth` states, so a single-element memory
  // still has a 1-bit address rather than a width `hw` cannot carry, and so
  // this and `DatapathEmitter::addrWidth` agree.
  return llvm::Log2_64_Ceil(
      static_cast<uint64_t>(std::max<int64_t>(elements, 2)));
}

// The step of \p v when it is the induction variable of a counted loop with
// constant bounds, the one shape a register can track. The register loads
// `coeff * lb` at start and adds `coeff * step`, so both must be constants; an
// `affine.for` step always is, an `scf.for` step is an operand.
//
// The emitter asks the same question of the region model it lowered to
// (`planAddressGenerators`), and both must agree on which loops carry.
static std::optional<int64_t> constantStepOf(Value v) {
  auto barg = dyn_cast<BlockArgument>(v);
  if (!barg)
    return std::nullopt;
  Operation *parent = barg.getOwner()->getParentOp();
  if (auto loop = dyn_cast<affine::AffineForOp>(parent)) {
    if (barg == loop.getInductionVar() && loop.hasConstantLowerBound())
      return loop.getStepAsInt();
    return std::nullopt;
  }
  if (auto loop = dyn_cast<scf::ForOp>(parent)) {
    std::optional<int64_t> step = getConstantIntValue(loop.getStep());
    if (barg == loop.getInductionVar() &&
        getConstantIntValue(loop.getLowerBound()))
      return step;
  }
  return std::nullopt;
}

// Merge the terms naming the same DIGIT of the same counter, \p indices giving
// each operand position the value it names. Two positions can hold one
// induction variable and the builder combines them, so pricing them apart would
// charge an adder nobody builds. Two digits of one counter do not combine.
static void mergeTermsByDigit(SplitAddress &sp, ArrayRef<Value> indices) {
  using Digit = std::tuple<Value, int64_t, int64_t, int64_t, int64_t>;
  llvm::MapVector<Digit, unsigned> group;
  SmallVector<SplitAddress::Term, 4> merged;
  for (const SplitAddress::Term &t : sp.terms) {
    Digit d{indices[t.operand], t.scale, t.offset, t.divisor, t.modulus};
    auto [it, isNew] = group.try_emplace(d, merged.size());
    if (isNew)
      merged.push_back(t);
    else
      merged[it->second].coeff += t.coeff;
  }
  llvm::erase_if(merged, [](const SplitAddress::Term &t) { return !t.coeff; });
  sp.terms = std::move(merged);
}

AddressCost mlir::allo::addressCostOf(Operation *op,
                                      const OperatorLibrary &lib) {
  std::optional<MemAccess> a = asMemAccess(op);
  if (!a || a->kind != AccessKind::Array)
    return {};
  assert(a->map && "an array access carries a map, the identity one when its "
                   "subscript is not affine");
  auto shape = cast<MemRefType>(a->root.getType()).getShape();
  AddressDelays delays = addressDelaysOf(lib);
  AddressExprs e =
      addressExprsOf(bankLayoutOf(a->root), a->map, shape, assignedBankOf(op));

  // One cone, priced as the emitter will build it.
  auto reduce = [&](AffineExpr expr, unsigned width) {
    SplitAddress sp =
        splitAddress(expr, a->map.getNumDims(), a->map.getNumSymbols(),
                     [&](unsigned p) { return constantStepOf(a->indices[p]); });
    mergeTermsByDigit(sp, a->indices);
    return splitAddressCost(sp, delays, width);
  };

  AddressCost c = reduce(e.offset, e.width);
  if (!e.bank)
    return c;
  // A second cone off the same operands, running BESIDE the offset: delay is
  // the max, operators add. Carried at the datapath width, being compared
  // against literal bank numbers rather than used as an address.
  AddressCost b = reduce(e.bank, AddressDelays::refWidth);
  c.adders += b.adders;
  c.multipliers += b.multipliers;
  c.dividers += b.dividers;
  c.reciprocals += b.reciprocals;
  c.carried += b.carried;
  c.delay = std::max(c.delay, b.delay);
  return c;
}

double mlir::allo::addressDelayOf(Operation *op, const OperatorLibrary &lib) {
  return std::round(addressCostOf(op, lib).delay * 100.0) / 100.0;
}
