/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/DependenceAnalysis.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/MemoryAccess.h"
#include "allo/Support/AffineValueMapBuilder.h"
#include "allo/Support/AliasAnalysis.h" // resolveRoot (storage identity)
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace mlir::allo::logging;
using namespace circt::analysis;

//===----------------------------------------------------------------------===//
// assume.ssa value facts
//
// Parse an `allo.assume.ssa` predicate into constant ranges on the SSA values
// it constrains: AND-ed comparisons contribute independently, tightest wins.
//===----------------------------------------------------------------------===//

namespace {
// A single-variable linear fact `c*v + k (>= | ==) 0` from one comparison.
struct Assumption {
  Value v;
  int64_t c, k;
  bool isEq; // true: == 0, false: >= 0
};

// Two accesses, orientation-independent: the polyhedral test visits a pair in
// both orders and one undecided orientation condemns the pair.
using OpPair = std::pair<Operation *, Operation *>;
} // namespace

static OpPair unorderedPair(Operation *a, Operation *b) {
  return a < b ? OpPair{a, b} : OpPair{b, a};
}

// Parse a comparison of one SSA value against a constant into `c*v + k
// (>=|==) 0`. Returns nullopt for shapes not modeled (a `ne`, or both
// operands constant or both symbolic).
static std::optional<Assumption> parseComparison(arith::CmpIOp cmp) {
  auto cL = getConstantIntValue(cmp.getLhs());
  auto cR = getConstantIntValue(cmp.getRhs());
  if (cL.has_value() == cR.has_value())
    return std::nullopt; // need exactly one constant operand

  bool isEq = false, swap = false;
  int strict = 0;
  using P = arith::CmpIPredicate;
  switch (cmp.getPredicate()) {
  case P::sge:
  case P::uge:
    break; // L - R >= 0
  case P::sgt:
  case P::ugt:
    strict = 1;
    break; // L - R - 1 >= 0
  case P::sle:
  case P::ule:
    swap = true;
    break; // R - L >= 0
  case P::slt:
  case P::ult:
    swap = true;
    strict = 1;
    break; // R - L - 1 >= 0
  case P::eq:
    isEq = true;
    break; // L - R == 0
  default:
    return std::nullopt; // ne
  }

  // Normalize to `x - y - strict`, where exactly one of x, y is the value.
  Value x = swap ? cmp.getRhs() : cmp.getLhs();
  Value y = swap ? cmp.getLhs() : cmp.getRhs();
  if (auto cx = swap ? cR : cL)
    return Assumption{y, -1, *cx - strict,
                      isEq}; // x constant: -y + (cx - strict)
  auto cy = swap ? cL : cR;
  return Assumption{x, 1, -*cy - strict,
                    isEq}; // y constant: x + (-cy - strict)
}

// Distill the parsed facts into a per-value constant range, keeping the
// tightest bound when a value is constrained more than once.
static void buildAssumedRanges(ArrayRef<Assumption> assumptions,
                               llvm::DenseMap<Value, AssumedRange> &ranges) {
  auto tighten = [&](Value v, std::optional<int64_t> lb,
                     std::optional<int64_t> ub) {
    AssumedRange &r = ranges[v];
    if (lb)
      r.lb = r.lb ? std::max(*r.lb, *lb) : lb;
    if (ub)
      r.ub = r.ub ? std::min(*r.ub, *ub) : ub;
  };
  for (const Assumption &as : assumptions) {
    // `c*v + k (>=|==) 0`  ==>  `c*v (>=|==) -k` (c is +/-1).
    if (as.isEq) {
      if ((-as.k) % as.c == 0) // exact integer solution, else vacuous
        tighten(as.v, (-as.k) / as.c, (-as.k) / as.c);
    } else if (as.c > 0) // v >= ceil(-k / c)
      tighten(as.v, llvm::divideCeilSigned(-as.k, as.c), std::nullopt);
    else // c < 0: v <= floor(-k / c)
      tighten(as.v, std::nullopt, llvm::divideFloorSigned(-as.k, as.c));
  }
}

// Collect the facts implied by an assume.ssa predicate (an `and`-tree of
// comparisons). Unrecognized shapes are not collected.
static void collectAssumptions(Value cond, SmallVectorImpl<Assumption> &out) {
  Operation *def = cond.getDefiningOp();
  if (!def)
    return;
  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    collectAssumptions(andOp.getLhs(), out);
    collectAssumptions(andOp.getRhs(), out);
  } else if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    if (auto as = parseComparison(cmp))
      out.push_back(*as);
  }
}

//===----------------------------------------------------------------------===//
// Memref dependences
//===----------------------------------------------------------------------===//

// Rewrite the components of one polyhedral result into ITERATION units, the
// unit every consumer reads them in.
static void
rescaleOnLoopStep(SmallVectorImpl<affine::DependenceComponent> &comps) {
  for (affine::DependenceComponent &c : comps) {
    int64_t step = cast<affine::AffineForOp>(c.op).getStepAsInt();
    auto rescale = [&](std::optional<int64_t> &v, int64_t open, bool isLower) {
      if (!v || *v == open) {
        v = std::nullopt;
        return;
      }
      int64_t d = isLower ? llvm::divideFloorSigned(*v, step)
                          : llvm::divideCeilSigned(*v, step);
      v = isLower && *v > 0 ? std::max<int64_t>(d, 1) : d;
    };
    rescale(c.lb, std::numeric_limits<int64_t>::min(), /*isLower=*/true);
    rescale(c.ub, std::numeric_limits<int64_t>::max(), /*isLower=*/false);
  }
}

// Bound the symbols of \p rel that an `assume.ssa` range constrains, found
// by identifier. Returns whether any bound landed.
static bool boundSymbols(presburger::IntegerRelation &rel,
                         const llvm::DenseMap<Value, AssumedRange> &ranges) {
  using presburger::Identifier;
  using presburger::VarKind;
  if (!rel.getSpace().isUsingIds())
    return false;
  bool tightened = false;
  ArrayRef<Identifier> ids = rel.getIds(VarKind::Symbol);
  unsigned off = rel.getVarKindOffset(VarKind::Symbol);
  for (const auto &[v, r] : ranges) {
    const auto *it = std::find(ids.begin(), ids.end(), Identifier(v));
    if (it == ids.end())
      continue;
    unsigned pos = off + std::distance(ids.begin(), it);
    if (r.lb) {
      rel.addBound(presburger::BoundType::LB, pos, *r.lb);
      tightened = true;
    }
    if (r.ub) {
      rel.addBound(presburger::BoundType::UB, pos, *r.ub);
      tightened = true;
    }
  }
  return tightened;
}

// Whether the `assume.ssa` value ranges prove \p srcAccess and \p dstAccess
// element-disjoint over their WHOLE iteration domains: the access relations
// (domains included), tightened by the ranges on their symbols, compose to an
// empty iteration-to-iteration relation. Built from the same public pieces
// the upstream test composes, minus its ordering constraints, so emptiness
// here is a strictly stronger fact and dropping every depth's result on it is
// sound (`A[i+n]` vs `A[i]` under `assume(n >= 64)` on a 64-trip loop).
// The upstream polyhedron cannot answer this: its composition eliminates the
// symbol and its returned system carries no value identities to bound.
static bool rangesDisjoint(const affine::MemRefAccess &srcAccess,
                           const affine::MemRefAccess &dstAccess,
                           const llvm::DenseMap<Value, AssumedRange> &ranges) {
  using presburger::IntegerRelation;
  using presburger::PresburgerSpace;
  IntegerRelation srcRel(PresburgerSpace::getRelationSpace());
  IntegerRelation dstRel(PresburgerSpace::getRelationSpace());
  if (failed(srcAccess.getAccessRelation(srcRel)) ||
      failed(dstAccess.getAccessRelation(dstRel)))
    return false;
  bool tightened = boundSymbols(srcRel, ranges);
  tightened |= boundSymbols(dstRel, ranges);
  if (!tightened)
    return false;
  dstRel.inverse();
  if (!dstRel.getSpace().isUsingIds())
    dstRel.resetIds();
  if (!srcRel.getSpace().isUsingIds())
    srcRel.resetIds();
  dstRel.mergeAndCompose(srcRel);
  return dstRel.isEmpty();
}

// Records the affine memref dependences of every ordered pair of accesses.
// `checkMemrefAccessDependence` is queried at each loop depth from 1 to
// numCommonLoops (a dependence carried by the d-th common surrounding loop)
// and at numCommonLoops + 1, the loop-independent (intra-iteration) case with
// all common loops pinned to the same iteration. At the top depth,
// `allowRAR = false` also orients the otherwise-symmetric dist-0 dependence
// by program order. A result whose polyhedron the `assume.ssa` value ranges
// empty out is dropped (see rangesEmptyDependence). Aliasing between
// distinct memrefs is not modeled: distinct SSA memrefs are ASSUMED
// disjoint.
//
// A pair with either endpoint the test cannot model (`nonPolyhedral`) is
// skipped entirely and left to the conservative path, so each pair is owned
// by exactly one analysis. A pair the test ACCEPTS but cannot decide answers
// `Failure`, which is not `NoDependence`, so it joins \p undecided and takes
// the conservative path too: an undecided pair left silently unordered would
// let two accesses that may alias share a cycle.
static void
checkMemrefDependence(ArrayRef<Operation *> memoryOps,
                      const llvm::SmallDenseSet<Operation *> &nonPolyhedral,
                      const llvm::DenseMap<Value, AssumedRange> &ranges,
                      llvm::DenseSet<OpPair> &undecided,
                      MemoryDependenceResult &results,
                      unsigned &prunedByRange) {
  for (Operation *dst : memoryOps) {
    results.try_emplace(dst); // every access gets a (possibly empty) entry
    if (nonPolyhedral.contains(dst))
      continue;
    affine::MemRefAccess dstAccess(dst);
    for (Operation *src : memoryOps) {
      if (src == dst || nonPolyhedral.contains(src))
        continue;
      affine::MemRefAccess srcAccess(src);
      unsigned numCommon = affine::getInnermostCommonLoopDepth({src, dst});
      // This pair's results, committed below unless the assumed ranges prove
      // the pair element-disjoint outright.
      SmallVector<circt::analysis::MemoryDependence, 2> found;
      for (unsigned depth = 1; depth <= numCommon + 1; ++depth) {
        // Read-read pairs get no edge at any depth (allowRAR = false): reads
        // commute, and port contention is the resource model's job. A carried
        // RAR edge is not harmless slack: composed with intra-iteration
        // chains it closes false recurrence circuits that inflate the II
        // floor. At the loop-independent depth allowRAR = false also orients
        // the dist-0 dependence by program order.
        affine::FlatAffineValueConstraints constraints;
        SmallVector<affine::DependenceComponent, 2> comps;
        auto result = affine::checkMemrefAccessDependence(
            srcAccess, dstAccess, depth, &constraints, &comps,
            /*allowRAR=*/false);
        if (result.value == affine::DependenceResult::Failure)
          undecided.insert(unorderedPair(src, dst));
        if (hasDependence(result.value)) {
          rescaleOnLoopStep(comps);
          found.emplace_back(src, result.value, comps);
        }
      }
      if (!found.empty() && !ranges.empty() &&
          rangesDisjoint(srcAccess, dstAccess, ranges)) {
        prunedByRange += found.size();
        continue;
      }
      for (circt::analysis::MemoryDependence &dep : found)
        results[dst].push_back(std::move(dep));
    }
  }
}

//===----------------------------------------------------------------------===//
// Stream dependences
//===----------------------------------------------------------------------===//

// Nearest enclosing loop (affine.for, scf.for or scf.while), skipping non-loop
// parents (e.g. affine.if / scf.if). Null if the op is not inside a loop.
static Operation *getNearestLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(parent))
      return parent;
  return nullptr;
}

// Enclosing loops (affine.for, scf.for or scf.while) of `op`, ordered outermost
// -> innermost (matching getAffineForIVs), for building dependence components.
static SmallVector<Operation *> getEnclosingLoops(Operation *op) {
  SmallVector<Operation *> inner; // innermost -> outermost as collected
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(parent))
      inner.push_back(parent);
  return llvm::to_vector(llvm::reverse(inner));
}

// Whether two same-base stream accesses may touch the same FIFO. A stream value
// is an array of FIFOs selected by its indices, so this is an affine
// disambiguation on the indices.
namespace {
enum class FifoAlias { Same, Distinct, Unknown };
} // namespace

// Whether result `k` of `m` is a function of an enclosing loop IV. The builder
// classifies loop IVs as affine DIMS and loop-invariant values (function args,
// worker-ids) as SYMBOLS, so "uses a dim" is exactly "varies across loop
// iterations".
static bool coordDependsOnIV(const affine::AffineValueMap &m, unsigned k) {
  bool usesDim = false;
  m.getAffineMap().getResult(k).walk([&](AffineExpr e) {
    if (isa<AffineDimExpr>(e))
      usesDim = true;
  });
  return usesDim;
}

static FifoAlias compareFifo(AffineValueMapBuilder &builder, Operation *a,
                             Operation *b) {
  builder.reset();
  for (Value idx : asMemAccess(a)->indices)
    if (failed(builder.importValue(idx)))
      return FifoAlias::Unknown;
  auto ma = builder.compose();

  builder.reset();
  for (Value idx : asMemAccess(b)->indices)
    if (failed(builder.importValue(idx)))
      return FifoAlias::Unknown;
  auto mb = builder.compose();

  if (ma.getNumResults() != mb.getNumResults())
    return FifoAlias::Unknown;

  affine::AffineValueMap diff;
  affine::AffineValueMap::difference(ma, mb, &diff);
  bool allZero = true;
  for (unsigned k = 0, e = diff.getAffineMap().getNumResults(); k < e; ++k) {
    auto cst = dyn_cast<AffineConstantExpr>(diff.getAffineMap().getResult(k));
    if (!cst) {
      // Symbolic offset: cannot prove same or distinct FIFO.
      allZero = false;
      continue;
    }
    if (cst.getValue() != 0) {
      // A nonzero constant offset on an IV-dependent coordinate (`put
      // fifo[i+1]` / `get fifo[i]`) overlaps FIFO ranges across iterations, so
      // serialize. An IV-independent offset selects a genuinely distinct FIFO.
      if (coordDependsOnIV(ma, k))
        return FifoAlias::Unknown;
      return FifoAlias::Distinct;
    }
  }
  return allZero ? FifoAlias::Same : FifoAlias::Unknown;
}

// Dependence components mirroring the op's enclosing loop nest, `distance` on
// the innermost loop (the only component the scheduler reads). Empty when no
// loop encloses `op`, and so unable to carry any distance.
static SmallVector<affine::DependenceComponent>
streamDepComponents(Operation *op, int64_t distance) {
  SmallVector<affine::DependenceComponent> comps;
  for (Operation *loop : getEnclosingLoops(op)) {
    affine::DependenceComponent comp;
    comp.op = loop;
    comp.lb = 0;
    comp.ub = 0;
    comps.push_back(comp);
  }
  if (!comps.empty())
    comps.back().lb = distance;
  return comps;
}

// Streams are FIFOs: every pair of accesses to the same channel must preserve
// program and iteration order regardless of direction (get-get is ordered
// too, unlike memory's RAW/WAR/WAW). Each may-aliasing pair gets a distance-0
// intra-iteration edge plus a distance-1 loop-carried back edge, closing the
// recurrence that bounds II exactly at the FIFO issue-order bound (II >= 1 +
// (t_later - t_earlier)). All pairs are serialized rather than chained in
// program order because FIFO may-aliasing is non-transitive.
static void checkStreamDependence(SmallVectorImpl<Operation *> &streamOps,
                                  AffineValueMapBuilder &builder,
                                  MemoryDependenceResult &results) {
  for (unsigned i = 0, e = streamOps.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      // `earlier` precedes `later` in program order: `walk` is a pre-order
      // traversal, so a smaller index is never scheduled after a larger one.
      Operation *earlier = streamOps[i];
      Operation *later = streamOps[j];

      // Different stream base SSA values (views peeled) are always
      // independent: SSA identity is a precise disambiguation for streams.
      if (asMemAccess(earlier)->root != asMemAccess(later)->root)
        continue;

      // Only serialize accesses sharing the same innermost loop, so both ends
      // land in one scheduling problem; no enclosing loop means the function's
      // straight-line span, which is one problem too.
      Operation *loop = getNearestLoop(earlier);
      if (loop != getNearestLoop(later))
        continue;

      // Provably-distinct FIFOs are independent; same or unknown are ordered.
      if (compareFifo(builder, earlier, later) == FifoAlias::Distinct)
        continue;

      results[later].emplace_back(earlier,
                                  affine::DependenceResult::HasDependence,
                                  streamDepComponents(later, /*distance=*/0));
      // The back edge needs a loop level to carry its distance. Outside any
      // loop there are no iterations to overtake.
      if (loop)
        results[earlier].emplace_back(
            later, affine::DependenceResult::HasDependence,
            streamDepComponents(earlier, /*distance=*/1));
    }
  }
}

//===----------------------------------------------------------------------===//
// Conservative memref dependences
//===----------------------------------------------------------------------===//

// Whether the polyhedral test can model where `op` sits: every loop enclosing
// it must be an affine.for. `getAffineForIVs` (and through it
// `getInnermostCommonLoopDepth`) silently skips every other loop form: the
// depth ladder never names an scf.for/scf.while, so the pair is reported
// loop-independent and a memory-carried recurrence is LOST. Such accesses go to
// the conservative path with the non-affine ones.
static bool inAffineNest(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (isa<LoopLikeOpInterface>(p) && !isa<affine::AffineForOp>(p))
      return false;
  return true;
}

// Dependence components mirroring the op's enclosing loop nest, `distance` on
// the innermost loop. Empty (loop-independent, distance 0) when the op is not
// in any loop.
static SmallVector<affine::DependenceComponent>
memDepComponents(Operation *op, int64_t distance) {
  SmallVector<affine::DependenceComponent> comps;
  for (Operation *loop : getEnclosingLoops(op)) {
    affine::DependenceComponent comp;
    comp.op = loop;
    comp.lb = 0;
    comp.ub = 0;
    comps.push_back(comp);
  }
  if (!comps.empty())
    comps.back().lb = distance;
  return comps;
}

// The compile-time value of \p a's subscript in dimension \p k, when it has
// one every execution: a constant map result, or a dim/symbol result whose
// operand is a constant.
static std::optional<int64_t> constantSubscript(const MemAccess &a,
                                                unsigned k) {
  if (!a.map || k >= a.map.getNumResults())
    return std::nullopt;
  AffineExpr e = a.map.getResult(k);
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return c.getValue();
  unsigned pos;
  if (auto d = dyn_cast<AffineDimExpr>(e))
    pos = d.getPosition();
  else if (auto s = dyn_cast<AffineSymbolExpr>(e))
    pos = a.map.getNumDims() + s.getPosition();
  else
    return std::nullopt;
  if (pos >= a.indices.size())
    return std::nullopt;
  return getConstantIntValue(a.indices[pos]);
}

// Whether some dimension holds unequal compile-time subscripts on both sides:
// the accessed elements then differ on every execution pair, whatever the
// other (arbitrarily dynamic) dimensions do.
static bool constantDimsDistinct(const MemAccess &a, const MemAccess &b) {
  unsigned n = std::min(a.map.getNumResults(), b.map.getNumResults());
  for (unsigned k = 0; k < n; ++k) {
    std::optional<int64_t> ca = constantSubscript(a, k);
    std::optional<int64_t> cb = constantSubscript(b, k);
    if (ca && cb && *ca != *cb)
      return true;
  }
  return false;
}

// Conservative memory dependences for pairs the polyhedral test cannot model
// (`nonPolyhedral`: a plain memref.load/store such as an indirect A[idx[i]], or
// an affine access whose loop nest is not all-affine; see inAffineNest). Any
// two accesses to the same array with at least one write are serialized in
// program order (a distance-0 forward edge), plus a distance-1 loop-carried
// back edge when they share an innermost loop (closing the recurrence that
// bounds II). Read-read pairs commute and are left independent, and so is a
// pair some dimension proves element-disjoint by unequal constant subscripts
// (`result[0][x]` vs `result[1][y]`). An `allo.assume.nodep` hint can prune a
// proven-false edge to recover II.
static void checkConservativeDependence(
    ArrayRef<Operation *> accessOps,
    const llvm::SmallDenseSet<Operation *> &nonPolyhedral,
    const llvm::DenseSet<OpPair> &undecided, MemoryDependenceResult &results,
    unsigned &prunedConstDim) {
  for (unsigned i = 0, e = accessOps.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      Operation *earlier = accessOps[i];
      Operation *later = accessOps[j];

      // Pairs the polyhedral test models are handled precisely there, unless it
      // answered `Failure` and so decided nothing.
      if (!nonPolyhedral.contains(earlier) && !nonPolyhedral.contains(later) &&
          !undecided.contains(unorderedPair(earlier, later)))
        continue;
      // Different arrays never conflict (distinct roots are distinct arrays;
      // the Allo frontend has no pointers); read-read pairs commute.
      auto ea = asMemAccess(earlier);
      auto la = asMemAccess(later);
      if (ea->root != la->root)
        continue;
      if (!ea->isWrite && !la->isWrite)
        continue;
      if (constantDimsDistinct(*ea, *la)) {
        ++prunedConstDim;
        continue;
      }

      // Forward intra-iteration edge (preserve program order).
      results[later].emplace_back(earlier,
                                  affine::DependenceResult::HasDependence,
                                  memDepComponents(later, /*distance=*/0));
      // Loop-carried back edge, closing the recurrence that bounds II.
      Operation *loop = getNearestLoop(earlier);
      if (loop && loop == getNearestLoop(later))
        results[earlier].emplace_back(
            later, affine::DependenceResult::HasDependence,
            memDepComponents(earlier, /*distance=*/1));
    }
  }
}

//===----------------------------------------------------------------------===//
// assume.nodep hint consumption
//===----------------------------------------------------------------------===//

// Direction of a dependence edge source -> dst from the read/write nature of
// its endpoints. `source` is the producer in both the forward and back-edge
// orientations, so this is orientation-independent.
static AssumeDepDirEnum edgeDirection(Operation *source, Operation *dst) {
  bool sw = asMemAccess(source)->isWrite, dw = asMemAccess(dst)->isWrite;
  if (sw && dw)
    return AssumeDepDirEnum::WAW;
  return sw ? AssumeDepDirEnum::RAW : AssumeDepDirEnum::WAR;
}

// The body block of the counted loop (affine.for or scf.for) whose induction
// variable is `iv`, or null if `iv` is not a counted-loop induction variable.
static Block *loopBodyForIV(Value iv) {
  if (auto loop = affine::getForInductionVarOwner(iv))
    return loop.getBody();
  if (auto loop = scf::getForInductionVarOwner(iv))
    return loop.getBody();
  // `loop-canonicalization` rewrites each original iv to an `affine.apply`
  // (floordiv/mod) of the surviving iv. Trace back through it so a nodep scoped
  // to a pre-coalescing iv still resolves to the coalesced loop.
  if (auto apply = iv.getDefiningOp<affine::AffineApplyOp>())
    for (Value operand : apply.getOperands())
      if (Block *body = loopBodyForIV(operand))
        return body;
  return nullptr;
}

// Prune the conservative dependence edges that an `allo.assume.nodep`
// (dependent = false) declares absent, matching by array, enclosing loop,
// inter/intra class, and, when given, direction and distance. Only conservative
// edges are removed: a proven affine dependence is NEVER dropped, so a hint
// restating something the analysis already inferred is a no-op.
static void
applyNoDepHints(ArrayRef<AssumeNoDepOp> hints,
                const llvm::SmallDenseSet<Operation *> &nonPolyhedral,
                MemoryDependenceResult &results) {
  for (AssumeNoDepOp hint : hints) {
    if (hint.getDependent())
      // Only "no dependence" assertions prune; `dependent = true` is a no-op
      // since the analysis never misses a real dependence to re-add.
      continue;
    // Resolve through views so it compares equal to the access roots.
    Value array = resolveRoot(hint.getVariable());
    Block *body = loopBodyForIV(hint.getIv());
    if (!body)
      continue;
    bool inter = hint.getDepType() == AssumeDepTypeEnum::Inter;
    auto dir = hint.getDirection();
    IntegerAttr distAttr = hint.getDistanceAttr();

    auto matches = [&](Operation *source, Operation *dst,
                       const MemoryDependence &dep) {
      // Same array, at least one endpoint outside the polyhedral test (so this
      // is a conservative edge), both accesses inside the hinted loop.
      if (asMemAccess(source)->root != array || asMemAccess(dst)->root != array)
        return false;
      if (!nonPolyhedral.contains(source) && !nonPolyhedral.contains(dst))
        return false;
      if (!body->findAncestorOpInBlock(*source) ||
          !body->findAncestorOpInBlock(*dst))
        return false;
      // inter- vs intra-iteration by the innermost distance component.
      int64_t d = dep.dependenceComponents.empty()
                      ? 0
                      : dep.dependenceComponents.back().lb.value_or(0);
      if (inter ? d < 1 : d != 0)
        return false;
      if (dir && edgeDirection(source, dst) != *dir)
        return false;
      if (distAttr && d != distAttr.getInt())
        return false;
      return true;
    };

    size_t pruned = 0;
    for (auto &entry : results)
      llvm::erase_if(entry.second, [&](const MemoryDependence &dep) {
        bool match = matches(dep.source, entry.first, dep);
        pruned += match;
        return match;
      });

    // A zero count flags a hint matching nothing, i.e. the dependence was
    // already inferred absent.
    auto note = info(Stage::Sched, hint.getOperation());
    note << "Applied dependence hint: pruned " << pruned << " conservative "
         << (inter ? "loop-carried" : "intra-iteration") << " dependence edge"
         << (pruned == 1 ? "" : "s");
    if (dir)
      note << " direction="
           << (*dir == AssumeDepDirEnum::RAW   ? "RAW"
               : *dir == AssumeDepDirEnum::WAR ? "WAR"
                                               : "WAW");
    if (distAttr)
      note << " distance=" << distAttr.getInt();
  }
}

//===----------------------------------------------------------------------===//
// DependenceAnalysis
//===----------------------------------------------------------------------===//

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Trip counts
//
// What a counted loop runs, exactly where the IR says so and as a worst case
// where only an `allo.assume.ssa` range does.
//===----------------------------------------------------------------------===//

namespace {
// An inclusive integer interval `[lo, hi]`; an open endpoint is unbounded.
using Interval = std::pair<std::optional<int64_t>, std::optional<int64_t>>;

// Bound an affine trip-count expression given each operand's known range. The
// divisor/multiplier of a mul/div/mod is always a constant in affine form, so
// each case is exact interval arithmetic; a missing operand bound propagates as
// an open endpoint.
Interval evalInterval(AffineExpr e, ArrayRef<AssumedRange> operands,
                      unsigned numDims) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return {c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return {operands[d.getPosition()].lb, operands[d.getPosition()].ub};
  if (auto s = dyn_cast<AffineSymbolExpr>(e)) {
    const AssumedRange &r = operands[numDims + s.getPosition()];
    return {r.lb, r.ub};
  }
  auto bin = cast<AffineBinaryOpExpr>(e);
  Interval l = evalInterval(bin.getLHS(), operands, numDims);
  auto constRHS = dyn_cast<AffineConstantExpr>(bin.getRHS());
  auto apply = [](std::optional<int64_t> a, std::optional<int64_t> b,
                  auto op) -> std::optional<int64_t> {
    if (a && b)
      return op(*a, *b);
    return std::nullopt;
  };
  switch (bin.getKind()) {
  case AffineExprKind::Add: {
    Interval r = evalInterval(bin.getRHS(), operands, numDims);
    auto add = [](int64_t a, int64_t b) { return a + b; };
    return {apply(l.first, r.first, add), apply(l.second, r.second, add)};
  }
  case AffineExprKind::Mul: {
    int64_t c = constRHS.getValue(); // affine: one factor is constant
    auto mul = [&](std::optional<int64_t> x) {
      return apply(x, c, std::multiplies<int64_t>());
    };
    return c >= 0 ? Interval{mul(l.first), mul(l.second)}
                  : Interval{mul(l.second), mul(l.first)};
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    int64_t c = constRHS.getValue(); // affine, positive divisor
    bool ceil = bin.getKind() == AffineExprKind::CeilDiv;
    auto div = [&](std::optional<int64_t> x) -> std::optional<int64_t> {
      if (!x)
        return std::nullopt;
      return ceil ? llvm::divideCeilSigned(*x, c)
                  : llvm::divideFloorSigned(*x, c);
    };
    return {div(l.first), div(l.second)};
  }
  case AffineExprKind::Mod:
    return {int64_t{0}, constRHS.getValue() - 1};
  default:
    return {std::nullopt, std::nullopt};
  }
}
} // namespace

// An `affine.for`: exact from the constant trip count, else bounded by the
// interval its trip-count map evaluates to over the assumed ranges.
static LoopTrip affineTrip(AffineForOp loop, const DependenceAnalysis &deps) {
  if (std::optional<uint64_t> c = getConstantTripCount(loop))
    return {static_cast<int64_t>(*c), false};

  AffineMap map;
  SmallVector<Value> operands;
  getTripCountMapAndOperands(loop, &map, &operands);
  if (!map || map.getNumResults() != 1)
    return {};

  SmallVector<AssumedRange> ranges;
  for (Value v : operands) {
    if (std::optional<int64_t> c = getConstantIntValue(v))
      ranges.push_back({*c, *c});
    else if (std::optional<AssumedRange> r = deps.getAssumedRange(v))
      ranges.push_back(*r);
    else
      ranges.push_back({});
  }
  Interval iv = evalInterval(map.getResult(0), ranges, map.getNumDims());
  if (!iv.second)
    return {};
  return {std::max<int64_t>(0, *iv.second), true};
}

// An `scf.for`: exact when lb/ub/step are all compile-time constants, else the
// worst case `ceil((max ub - min lb) / min step)`, which needs a positive step.
static LoopTrip scfTrip(scf::ForOp loop, const DependenceAnalysis &deps) {
  auto rangeOf = [&](Value v) -> AssumedRange {
    if (std::optional<int64_t> c = getConstantIntValue(v))
      return {*c, *c};
    if (std::optional<AssumedRange> r = deps.getAssumedRange(v))
      return *r;
    return {};
  };
  AssumedRange lb = rangeOf(loop.getLowerBound());
  AssumedRange ub = rangeOf(loop.getUpperBound());
  AssumedRange step = rangeOf(loop.getStep());
  auto isConst = [](const AssumedRange &r) {
    return r.lb && r.ub && *r.lb == *r.ub;
  };
  if (isConst(lb) && isConst(ub) && isConst(step)) {
    int64_t s = *step.lb;
    if (s <= 0)
      return {}; // non-positive step unsupported
    return {std::max<int64_t>(0, llvm::divideCeilSigned(*ub.lb - *lb.lb, s)),
            false};
  }
  if (ub.ub && lb.lb && step.lb && *step.lb >= 1) {
    return {
        std::max<int64_t>(0, llvm::divideCeilSigned(*ub.ub - *lb.lb, *step.lb)),
        true};
  }
  return {};
}

LoopTrip DependenceAnalysis::tripOf(Operation *loop) const {
  auto [it, fresh] = trips.try_emplace(loop);
  if (fresh)
    it->second = isa<AffineForOp>(loop)
                     ? affineTrip(cast<AffineForOp>(loop), *this)
                     : scfTrip(cast<scf::ForOp>(loop), *this);
  return it->second;
}

int64_t carriedDistanceAtLevel(ArrayRef<affine::DependenceComponent> comps,
                               unsigned level, bool &drop, bool &valid) {
  drop = false;
  valid = true;
  if (comps.empty())
    return 0; // loop-independent: same iteration at every level
  if (comps.size() < level) {
    valid = false;
    return 0;
  }
  // A `*`-direction component (lb == nullopt) is an UNKNOWN distance, not 0.
  // An OUTER level drops the edge only on a PROVEN positive distance; THIS
  // level falls back to 1 so the modulo solver never under-bounds II.
  for (unsigned k = 0; k + 1 < level; ++k) // components outer to the level
    if (comps[k].lb.value_or(0) > 0) {
      drop = true;
      return 0;
    }
  auto d = comps[level - 1].lb;
  return d.has_value() ? *d : 1;
}

bool isUnmodeledMemoryAccess(Operation *op) {
  // The complement of the access kinds the constructor's walk below collects.
  if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface,
          memref::LoadOp, memref::StoreOp, StreamGetOp, StreamPutOp,
          AssumeNoDepOp, AssumeSSAOp>(op))
    return false;
  auto mem = dyn_cast<MemoryEffectOpInterface>(op);
  return mem && (mem.hasEffect<MemoryEffects::Read>() ||
                 mem.hasEffect<MemoryEffects::Write>());
}

DependenceAnalysis::DependenceAnalysis(func::FuncOp funcOp) : func(funcOp) {
  SmallVector<Operation *> memoryOps;
  SmallVector<Operation *> streamOps;
  // All memref accesses in program (walk) order, plus the subset outside the
  // polyhedral test (a non-affine op, or a loop nest that is not all-affine),
  // which takes the conservative fallback below.
  SmallVector<Operation *> accessOps;
  SmallVector<AssumeNoDepOp> noDepHints;
  SmallVector<Assumption> assumptions;
  funcOp->walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      memoryOps.push_back(op);
      accessOps.push_back(op);
      if (!inAffineNest(op))
        nonPolyhedral.insert(op);
    } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      nonPolyhedral.insert(op);
      accessOps.push_back(op);
    } else if (isa<StreamGetOp, StreamPutOp>(op)) {
      streamOps.push_back(op);
    } else if (auto hint = dyn_cast<AssumeNoDepOp>(op)) {
      noDepHints.push_back(hint);
    } else if (auto hint = dyn_cast<AssumeSSAOp>(op)) {
      collectAssumptions(hint.getCondition(), assumptions);
    }
    // Asserted HERE, in the walk whose branches above are exactly what
    // `isUnmodeledMemoryAccess` is the complement of. `verify-rtl-legality`
    // rejects such an access before scheduling, so reaching this means the two
    // disagree and this op joined no access list: its dependences are dropped
    // and the solve would freely reorder it against what it aliases.
    assert(!isUnmodeledMemoryAccess(op) &&
           "an unmodeled memory access reached the dependence analysis");
  });

  // Distill the assume.ssa value facts into per-value constant ranges first:
  // the polyhedral test below reads them as symbol bounds.
  buildAssumedRanges(assumptions, assumedRanges);

  // Affine memref dependences over all carried depths plus the
  // loop-independent one.
  unsigned prunedByRange = 0;
  checkMemrefDependence(memoryOps, nonPolyhedral, assumedRanges, undecided,
                        results, prunedByRange);
  if (prunedByRange)
    info(Stage::Sched, funcOp)
        << "Value-range disambiguation pruned " << prunedByRange
        << " dependence result(s): the assumed ranges empty the pair's "
           "polyhedron";

  // Conservative ordering for the pairs the polyhedral test skips or cannot
  // decide.
  unsigned prunedConstDim = 0;
  checkConservativeDependence(accessOps, nonPolyhedral, undecided, results,
                              prunedConstDim);
  if (prunedConstDim)
    info(Stage::Sched, funcOp)
        << "Constant-subscript disambiguation pruned " << prunedConstDim
        << " conservative pair(s): a dimension's unequal constants keep the "
           "elements disjoint";

  AffineValueMapBuilder builder(funcOp.getContext());
  checkStreamDependence(streamOps, builder, results);

  // User hints: prune conservative edges the programmer proves absent. Applied
  // last, over the fully-built edge set.
  applyNoDepHints(noDepHints, nonPolyhedral, results);

  // Surface the distilled ranges, one line per constrained value.
  if (!assumedRanges.empty()) {
    info(Stage::Sched) << "Applied value hints: distilled "
                       << assumedRanges.size() << " value range"
                       << (assumedRanges.size() == 1 ? "" : "s");
    for (const auto &[v, r] : assumedRanges) {
      auto lb = r.lb ? std::to_string(*r.lb) : "-inf";
      auto ub = r.ub ? std::to_string(*r.ub) : "+inf";
      info(Stage::Sched) << "  " << logging::detail::describe(v.getLoc())
                         << " in [" << lb << ", " << ub << "]";
    }
  }
}

} // namespace mlir::allo
