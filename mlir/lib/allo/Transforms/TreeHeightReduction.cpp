/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h" // classify, measuredCombDelay
#include "allo/Transforms/Passes.h"
#include "allo/Transforms/ReductionUtils.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::allo {
#define GEN_PASS_DEF_TREEHEIGHTREDUCTIONPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {

// An associative operator family balanced in direct mode (the widened integer
// idiom is handled separately). `sub` joins `add` in the additive family as a
// sign-flipped leaf; the rest are single-op.
enum class Family { None, Additive, Mul, And, Or, Xor };

Family familyOf(Operation *op) {
  if (isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp>(op))
    return Family::Additive;
  if (isa<arith::MulFOp, arith::MulIOp>(op))
    return Family::Mul;
  if (isa<arith::AndIOp>(op))
    return Family::And;
  if (isa<arith::OrIOp>(op))
    return Family::Or;
  if (isa<arith::XOrIOp>(op))
    return Family::Xor;
  return Family::None;
}

bool isFloatFamilyOp(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op);
}

bool isSub(Operation *op) { return isa<arith::SubFOp, arith::SubIOp>(op); }

// Reassociation moves where a signed/unsigned wrap is observed, so an operator
// asserting no-overflow must not be folded into a larger tree.
bool hasIntOverflow(Operation *op) {
  auto none = arith::IntegerOverflowFlags::none;
  if (auto o = dyn_cast<arith::AddIOp>(op))
    return o.getOverflowFlags() != none;
  if (auto o = dyn_cast<arith::SubIOp>(op))
    return o.getOverflowFlags() != none;
  if (auto o = dyn_cast<arith::MulIOp>(op))
    return o.getOverflowFlags() != none;
  return false;
}

// A fresh binary arith op named `opName` over (x, y) with default attributes.
Value emitBin(OpBuilder &b, Location loc, StringRef opName, Value x, Value y) {
  OperationState state(loc, opName);
  state.addOperands({x, y});
  state.addTypes({x.getType()});
  return b.create(state)->getResult(0);
}

unsigned ceilLog2(unsigned k) { return k <= 1 ? 0 : llvm::Log2_32_Ceil(k); }

// Balancing a carried reduction trades area (a wider tree) for a shorter
// recurrence, paying back only over enough iterations; a dynamic or tiny trip
// count leaves the recurrence linear.
constexpr uint64_t kMinCarriedTrip = 4;

bool carriedTripPays(Operation *root) {
  auto loop = root->getParentOfType<affine::AffineForOp>();
  if (!loop)
    return true; // not a counted affine reduction; leave it to the old policy
  std::optional<uint64_t> trip = affine::getConstantTripCount(loop);
  return trip && *trip >= kMinCarriedTrip;
}

// A float division by a finite non-zero constant becomes a multiply by its
// reciprocal, trading a divider IP for a multiply. Inexact, so it rides the
// same fast-math gate as the reassociation.
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

// A direct-mode leaf, carrying the sign accumulated from the root (always + for
// the multiplicative and bitwise families).
struct Leaf {
  Value v;
  int sign;
};

// Whether `v`, or anything in its bounded backward cone within `block`, is a
// loop-carried value: an iter_arg, or a load of `store`'s memref from an
// earlier iteration. Such a value is folded at the tree root so the recurrence
// spans one operator; it is never pulled into the balanced subtree.
bool coneCarried(Value v, affine::AffineStoreOp store, Block *block,
                 int depth) {
  if (isLoopCarried(v) || (store && isCarriedTap(v, store)))
    return true;
  Operation *d = v.getDefiningOp();
  if (depth == 0 || !d || d->getBlock() != block)
    return false;
  for (Value o : d->getOperands())
    if (coneCarried(o, store, block, depth - 1))
      return true;
  return false;
}

struct TreeHeightReductionPass
    : public allo::impl::TreeHeightReductionPassBase<TreeHeightReductionPass> {
  using TreeHeightReductionPassBase::TreeHeightReductionPassBase;

  // The operator library, built once when a target period is given (the real
  // pipeline); absent in isolated runs, where every weight reads as 0 and the
  // tree degenerates to a plain balanced one.
  std::optional<OperatorLibrary> lib;

  // A value's arrival weight: the marginal combinational delay of its defining
  // op. Integer-gated (a float leaf must not be priced as an int add, which
  // shares OpKind::Add) and read through the non-logging measuredCombDelay.
  double weightOf(Value v) {
    Operation *d = v.getDefiningOp();
    if (!lib || !d || !isa<IntegerType>(v.getType()) || isZeroDelay(d))
      return 0.0;
    return lib->measuredCombDelay(classify(d), combParamWidth(d)).value_or(0.0);
  }

  // A combining operator's own marginal delay (0 for float / unmeasured).
  double opDelay(Operation *op) {
    if (!lib || !isa<IntegerType>(op->getResult(0).getType()))
      return 0.0;
    return lib->measuredCombDelay(classify(op), combParamWidth(op))
        .value_or(0.0);
  }

  void runOnOperation() override {
    if (enableFp)
      reciprocalizeConstDivs(getOperation());
    if (periodNs > 0.0) {
      lib = OperatorLibrary::fromModule(
          getOperation()->getParentOfType<ModuleOp>());
      lib->setSelectionPeriod((float)periodNs);
    }
    // Collect roots in program order, then process tails first so a maximal
    // tree's root is reached before its absorbed operands; a folded operand is
    // marked consumed and skipped.
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (op->getNumResults() != 1)
        return;
      bool widened = matchReductionStep(op->getResult(0)).widened();
      if (familyOf(op) == Family::None && !widened)
        return;
      if (isFloatFamilyOp(op) && !enableFp)
        return;
      candidates.push_back(op);
    });
    DenseSet<Operation *> consumed;
    for (Operation *root : llvm::reverse(candidates)) {
      if (consumed.contains(root))
        continue;
      ReductionStep tail = matchReductionStep(root->getResult(0));
      if (tail.widened())
        rewriteWidened(root, tail, consumed);
      else
        rewriteDirect(root, consumed);
    }
  }

  // --- direct mode: familyOf-based, additive signs -------------------------

  bool recurses(Operation *child, Family fam, Block *block) {
    return familyOf(child) == fam && child->getBlock() == block &&
           child->hasOneUse() && !hasIntOverflow(child);
  }

  void flatten(Operation *op, Family fam, Block *block, int sign,
               SmallVectorImpl<Leaf> &leaves,
               SmallVectorImpl<Operation *> &interior) {
    interior.push_back(op); // preorder: a parent precedes its children
    bool sub = isSub(op);
    for (unsigned k = 0; k < 2; ++k) {
      Value operand = op->getOperand(k);
      int childSign = (sub && k == 1) ? -sign : sign;
      Operation *d = operand.getDefiningOp();
      if (d && recurses(d, fam, block))
        flatten(d, fam, block, childSign, leaves, interior);
      else
        leaves.push_back({operand, childSign});
    }
  }

  // The depth of the tree the direct build would produce over `leaves`.
  unsigned directDepth(Family fam, ArrayRef<Leaf> leaves) {
    if (fam != Family::Additive)
      return ceilLog2(leaves.size());
    unsigned p = 0, q = 0;
    for (const Leaf &l : leaves)
      (l.sign > 0 ? p : q)++;
    unsigned d = std::max(ceilLog2(p), ceilLog2(q));
    return q ? d + 1 : d; // a subtract node joins the sums
  }

  Value buildDirect(OpBuilder &b, Location loc, Type ty, Family fam,
                    Operation *root, ArrayRef<Leaf> rest,
                    ArrayRef<Leaf> carried, double opw) {
    if (fam == Family::Additive) {
      bool fp = isFloatFamilyOp(root);
      StringRef add = fp ? "arith.addf" : "arith.addi";
      StringRef sub = fp ? "arith.subf" : "arith.subi";
      auto combine = [&](Value x, Value y) {
        return emitBin(b, loc, add, x, y);
      };
      SmallVector<WeightedValue> pos, neg;
      for (const Leaf &l : rest)
        (l.sign > 0 ? pos : neg).push_back({l.v, weightOf(l.v)});
      Value tp = pos.empty() ? Value() : buildWeightedTree(pos, opw, combine);
      Value tn = neg.empty() ? Value() : buildWeightedTree(neg, opw, combine);
      Value base;
      if (!tn)
        base = tp;
      else if (!tp) {
        Value zero = arith::ConstantOp::create(b, loc, b.getZeroAttr(ty));
        base = emitBin(b, loc, sub, zero, tn);
      } else
        base = emitBin(b, loc, sub, tp, tn);
      for (const Leaf &c : carried)
        base = emitBin(b, loc, c.sign > 0 ? add : sub, base, c.v);
      return base;
    }
    StringRef opName = root->getName().getStringRef();
    auto combine = [&](Value x, Value y) {
      return emitBin(b, loc, opName, x, y);
    };
    SmallVector<WeightedValue> nodes;
    for (const Leaf &l : rest)
      nodes.push_back({l.v, weightOf(l.v)});
    Value base = buildWeightedTree(nodes, opw, combine);
    for (const Leaf &c : carried)
      base = emitBin(b, loc, opName, base, c.v);
    return base;
  }

  void rewriteDirect(Operation *root, DenseSet<Operation *> &consumed) {
    Family fam = familyOf(root);
    // Index arithmetic (address computation) stays out of the datapath tree.
    if (!isFloatFamilyOp(root) &&
        !isa<IntegerType>(root->getResult(0).getType()))
      return;

    SmallVector<Leaf> leaves;
    SmallVector<Operation *> interior;
    Block *block = root->getBlock();
    flatten(root, fam, block, +1, leaves, interior);
    for (Operation *op : interior)
      consumed.insert(op);

    affine::AffineStoreOp store = closingStore(root->getResult(0));
    SmallVector<Leaf> rest, carried;
    for (const Leaf &l : leaves)
      (coneCarried(l.v, store, block, /*depth=*/4) ? carried : rest)
          .push_back(l);

    unsigned n = leaves.size();
    // A carried tree is a reduction: folding the accumulator at the root drops
    // the recurrence from n operators to 1, worth it from n>=3 when the trip
    // amortizes the wider tree. A non-carried tree only shortens its datapath,
    // so it must beat the linear chain and clear four leaves.
    bool improves = carried.empty()
                        ? (n >= 4 && directDepth(fam, leaves) < n - 1)
                        : (n >= 3 && carriedTripPays(root));
    if (!improves)
      return;
    // Leave a comparison's operand alone: it may drive control the scheduler
    // keys off, and balancing an integer predicate buys nothing.
    if (!isFloatFamilyOp(root))
      for (Operation *u : root->getUsers())
        if (isa<arith::CmpIOp>(u))
          return;
    // All leaves carried (several taps, no plain operand): balance them as the
    // tree rather than folding into nothing.
    if (rest.empty())
      std::swap(rest, carried);

    OpBuilder b(root);
    double opw = 0.0;
    for (Operation *op : interior)
      opw = std::max(opw, opDelay(op));
    Value result = buildDirect(b, root->getLoc(), root->getResult(0).getType(),
                               fam, root, rest, carried, opw);
    root->getResult(0).replaceAllUsesWith(result);
    for (Operation *op : interior)
      op->erase();
  }

  // --- widened idiom: trunc(core(ext, ext)) chains -------------------------

  void rewriteWidened(Operation *root, const ReductionStep &tail,
                      DenseSet<Operation *> &consumed) {
    ReductionChain chain;
    chain.steps.push_back(tail);
    auto [lhs, rhs] = reductionOperands(tail);
    flattenChain(lhs, tail, chain);
    flattenChain(rhs, tail, chain);
    if (chain.steps.size() < 2) // nothing absorbed: a lone step, no chain
      return;
    for (const ReductionStep &s : chain.steps) {
      consumed.insert(s.core);
      if (s.trunc)
        consumed.insert(s.trunc);
    }

    affine::AffineStoreOp store = closingStore(root->getResult(0));
    SmallVector<Value> carried, rest;
    for (Value leaf : chain.leaves)
      (isLoopCarried(leaf) || (store && isCarriedTap(leaf, store)) ? carried
                                                                   : rest)
          .push_back(leaf);

    unsigned n = chain.leaves.size();
    // The idiom marks a genuine reduction, so no loop-carried anchor is needed.
    bool improves = carried.empty() ? (llvm::Log2_32_Ceil(n) < n - 1)
                                    : (n >= 3 && carriedTripPays(root));
    if (!improves)
      return;
    if (rest.empty())
      std::swap(rest, carried);

    OpBuilder b(root);
    double opw = opDelay(tail.core);
    auto combine = [&](Value x, Value y) {
      return buildReductionStep(b, tail, x, y);
    };
    SmallVector<WeightedValue> nodes;
    for (Value v : rest)
      nodes.push_back({v, weightOf(v)});
    Value acc = buildWeightedTree(nodes, opw, combine);
    for (Value c : carried)
      acc = buildReductionStep(b, tail, acc, c);
    root->getResult(0).replaceAllUsesWith(acc);
    for (const ReductionStep &s : chain.steps)
      eraseStep(s);
  }
};

} // namespace
