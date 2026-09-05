/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/AddressModel.h" // addressCost
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess
#include "allo/Scheduling/RegionGraph.h"  // isSyncSubKernelCall
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "allo-c/Schedule.h" // kPipelineIIAttr
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_LOOPCANONICALIZATIONPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

//===----------------------------------------------------------------------===//
// Loop queries shared by all four rewrites.
//===----------------------------------------------------------------------===//

// A counted loop, whichever dialect spells it.
bool isForLoop(Operation *op) {
  return isa<affine::AffineForOp, scf::ForOp>(op);
}

// The single body block of a counted loop.
Block *loopBody(Operation *loop) {
  return &cast<LoopLikeOpInterface>(loop).getLoopRegions().front()->front();
}

// The OUTERMOST counted loops nested anywhere in \p loop's body; they may sit
// under an `scf.if` rather than directly in the body.
SmallVector<Operation *> nestedLoops(Operation *loop) {
  SmallVector<Operation *> out;
  loopBody(loop)->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isForLoop(op))
      return WalkResult::advance();
    out.push_back(op);
    return WalkResult::skip();
  });
  return out;
}

bool hasNestedLoop(Operation *loop) { return !nestedLoops(loop).empty(); }

// The compile-time trip count of a counted loop, or nullopt when it has none.
std::optional<int64_t> constantTrip(Operation *loop) {
  if (auto af = dyn_cast<affine::AffineForOp>(loop))
    return affine::getConstantTripCount(af);
  auto sf = cast<scf::ForOp>(loop);
  std::optional<int64_t> lb = getConstantIntValue(sf.getLowerBound()),
                         ub = getConstantIntValue(sf.getUpperBound()),
                         step = getConstantIntValue(sf.getStep());
  if (!lb || !ub || !step || !*step)
    return std::nullopt;
  int64_t span = *step > 0 ? *ub - *lb : *lb - *ub;
  if (span <= 0)
    return std::nullopt;
  return llvm::divideCeilSigned(span, std::abs(*step));
}

//===----------------------------------------------------------------------===//
// Erase the loops that never run.
//===----------------------------------------------------------------------===//

// Whether \p loop provably runs zero times.
bool isTripZero(Operation *loop) {
  if (auto af = dyn_cast<affine::AffineForOp>(loop))
    return affine::getConstantTripCount(af) == 0;
  auto sf = cast<scf::ForOp>(loop);
  if (sf.getLowerBound() == sf.getUpperBound())
    return true;
  std::optional<int64_t> lb = getConstantIntValue(sf.getLowerBound()),
                         ub = getConstantIntValue(sf.getUpperBound()),
                         step = getConstantIntValue(sf.getStep());
  if (!lb || !ub || !step)
    return false;
  return *step > 0 ? *ub <= *lb : *step < 0 && *ub >= *lb;
}

// Erase every loop under \p root that never runs, its whole body with it.
void eraseNeverTakenLoops(Operation *root) {
  SmallVector<Operation *> dead;
  root->walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface op) {
    if (!isForLoop(op) || !isTripZero(op))
      return WalkResult::advance();
    dead.push_back(op);
    return WalkResult::skip();
  });
  for (Operation *loop : dead) {
    info(Stage::Prep, loop) << "Detected a loop with a compile-time trip count "
                               "of 0; dropping it and its body";
    loop->replaceAllUsesWith(cast<LoopLikeOpInterface>(loop).getInits());
    loop->erase();
  }
}

//===----------------------------------------------------------------------===//
// Unroll the loops under a pipelined one.
//===----------------------------------------------------------------------===//

// A loop pipelined by `s.pipeline(ii != -1)`; its body must become loop-free.
bool isPipelined(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>(kPipelineIIAttr);
  return attr && attr.getInt() != -1;
}

// Every loop nested inside \p loop is a constant-trip counted loop, so the
// whole interior can be replicated away. An `scf.while` is the case this
// refuses.
bool innerLoopsUnrollable(Operation *loop) {
  WalkResult r = loopBody(loop)->walk([&](Operation *op) {
    if (!isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(op))
      return WalkResult::advance();
    return isForLoop(op) && constantTrip(op) ? WalkResult::advance()
                                             : WalkResult::interrupt();
  });
  return !r.wasInterrupted();
}

// Fully unroll every loop nested inside \p loop, innermost first: a leaf has no
// loop to replicate, so each unroll strictly shrinks what is left.
void unrollInner(Operation *loop, StringRef pipelinedName) {
  for (Operation *child : nestedLoops(loop)) {
    unrollInner(child, pipelinedName);
    // Reported BEFORE the rewrite: `loopUnrollFull` erases the loop it is given
    // and the message names it.
    info(Stage::Prep, child)
        << "Automatically fully unrolled the loop implied by pipelining on "
        << pipelinedName;
    LogicalResult r =
        isa<affine::AffineForOp>(child)
            ? affine::loopUnrollFull(cast<affine::AffineForOp>(child))
            : mlir::loopUnrollFull(cast<scf::ForOp>(child));
    assert(succeeded(r) && "a constant-trip loop must fully unroll");
    (void)r;
  }
}

//===----------------------------------------------------------------------===//
// Normalize a band to lower bound 0 and step 1.
//===----------------------------------------------------------------------===//

// Whether normalizing \p loop pays. A RUNTIME lower bound is skipped: it makes
// an address naming the induction variable name two counter registers and the
// adder joining them, and buys nothing. An induction variable with a
// NON-AFFINE use is skipped because only an affine consumer absorbs the
// `affine.apply` the rewrite leaves behind.
bool normalizable(affine::AffineForOp loop) {
  return loop.hasConstantLowerBound() &&
         llvm::all_of(loop.getInductionVar().getUsers(), [](Operation *user) {
           return isa<affine::AffineDialect>(user->getDialect());
         });
}

// Normalize every loop of \p band that pays, innermost first: normalizing an
// outer loop rewrites an inner one's bound operands, and a bound is one of the
// affine uses that absorbs the rewrite.
void normalizeBand(ArrayRef<affine::AffineForOp> band,
                   const FrozenRewritePatternSet &compose) {
  affine::AffineForOp innermostLoop = band.back();
  Block *innermost = innermostLoop.getBody();
  bool rewrote = false;
  for (affine::AffineForOp loop : llvm::reverse(band)) {
    if (!normalizable(loop))
      continue;
    Operation *before = &loop.getBody()->front();
    if (failed(affine::normalizeAffineFor(loop, /*promoteSingleIter=*/false)))
      continue;
    Operation *apply = &loop.getBody()->front();
    if (apply == before)
      continue; // already normalized: nothing was left behind
    rewrote = true;
    if (loop.getBody() != innermost)
      apply->moveBefore(&innermost->front());
  }
  if (rewrote)
    (void)applyPatternsGreedily(band.front(), compose);
}

//===----------------------------------------------------------------------===//
// Perfectize an imperfect nest by sinking its prologue and epilogue.
//===----------------------------------------------------------------------===//

// One matched imperfect nest: the inner loop and the prologue/epilogue ops
// surrounding it in the outer body (all validated sinkable).
struct Match {
  Operation *lin = nullptr;          // inner loop (affine.for / scf.for)
  SmallVector<Operation *> prologue; // before lin
  SmallVector<Operation *> epilogue; // after lin
};

Value storedMemRef(Operation *op) {
  if (auto s = dyn_cast<affine::AffineStoreOp>(op))
    return s.getMemRef();
  if (auto s = dyn_cast<memref::StoreOp>(op))
    return s.getMemRef();
  return {};
}

Value loadedMemRef(Operation *op) {
  if (auto l = dyn_cast<affine::AffineLoadOp>(op))
    return l.getMemRef();
  if (auto l = dyn_cast<memref::LoadOp>(op))
    return l.getMemRef();
  return {};
}

// Match \p outer as a sinkable imperfect nest, or return nullopt. \p reason is
// set when \p outer has the shape of an imperfect nest but an unsupported
// feature blocks it, and left empty when it is simply not one.
std::optional<Match> matchImperfect(Operation *outer, std::string &reason) {
  Match m;
  unsigned innerCount = 0;
  bool hasWhile = false;
  for (Operation &op : loopBody(outer)->without_terminator()) {
    if (isForLoop(&op)) {
      if (!m.lin)
        m.lin = &op;
      ++innerCount;
      continue;
    }
    if (isa<scf::WhileOp>(op)) {
      hasWhile = true;
      continue;
    }
    (m.lin ? m.epilogue : m.prologue).push_back(&op);
  }
  // Not an imperfect nest at all: no counted inner loop, or nothing to sink.
  if (!m.lin || (m.prologue.empty() && m.epilogue.empty()))
    return std::nullopt;

  // From here `outer` is a genuine imperfect nest, so every bail sets a reason.
  if (innerCount > 1)
    return (reason = "it has sibling inner loops"), std::nullopt;
  if (hasWhile)
    return (reason = "it contains an uncounted (scf.while) inner loop"),
           std::nullopt;
  if (outer->getNumResults() != 0)
    return (reason =
                "the outer loop carries a result (an accumulator escapes)"),
           std::nullopt;
  if (hasNestedLoop(m.lin))
    return (reason = "the inner loop is itself a nest (not innermost)"),
           std::nullopt;

  // Inner-loop guard feasibility. affine.for: a constant last-iteration IV
  // (normalized, constant trip) for the `affine.if`. scf.for: a runtime guard
  // needs a known positive step for the last-iteration test (`iv+step >= ub`).
  if (auto af = dyn_cast<affine::AffineForOp>(m.lin)) {
    if (!af.hasConstantLowerBound() || af.getConstantLowerBound() != 0 ||
        af.getStepAsInt() != 1 || !constantTrip(af))
      return (reason = "the inner loop is not a normalized constant-trip loop"),
             std::nullopt;
  } else {
    auto sf = cast<scf::ForOp>(m.lin);
    std::optional<int64_t> step = getConstantIntValue(sf.getStep());
    if (!m.epilogue.empty() && (!step || *step <= 0))
      return (reason =
                  "the inner loop has a non-constant or non-positive step"),
             std::nullopt;
  }

  // Epilogue: straight-line, no result escaping the epilogue set (so every use
  // lands inside the inner loop after sinking; stores have no results).
  DenseSet<Operation *> epiSet(m.epilogue.begin(), m.epilogue.end());
  for (Operation *op : m.epilogue) {
    if (op->getNumRegions() != 0)
      return (reason = "an epilogue op has a nested region"), std::nullopt;
    for (Value r : op->getResults())
      for (Operation *user : r.getUsers())
        if (!epiSet.contains(user))
          return (reason = "an epilogue value is used outside the nest"),
                 std::nullopt;
  }

  // Prologue: each op pure (unguarded), a store (guard first), or a load of a
  // memref not written anywhere in the nest (recompute-safe).
  DenseSet<Value> written;
  loopBody(outer)->walk([&](Operation *op) {
    if (Value mr = storedMemRef(op))
      written.insert(mr);
  });
  for (Operation *op : m.prologue) {
    if (op->getNumRegions() != 0)
      return (reason = "a prologue op has a nested region"), std::nullopt;
    if (isMemoryEffectFree(op) || storedMemRef(op))
      continue;
    Value mr = loadedMemRef(op);
    if (!mr || written.contains(mr))
      return (reason = "a prologue op has an unschedulable side effect or an "
                       "aliased load"),
             std::nullopt;
  }
  return m;
}

// Sink \p ops into a guard inserted at \p insertPt inside \p lin's body that
// fires only at the first (or last) iteration: an `affine.if` (constant IV) for
// an affine.for inner, an `scf.if` (runtime `cmpi`) for an scf.for inner.
void sinkGuarded(Operation *lin, ArrayRef<Operation *> ops, bool first,
                 Operation *insertPt) {
  OpBuilder b(insertPt);
  Location loc = lin->getLoc();
  Operation *thenTerm;
  if (auto af = dyn_cast<affine::AffineForOp>(lin)) {
    int64_t v = first ? 0 : (*constantTrip(af) - 1);
    AffineExpr d0 = b.getAffineDimExpr(0);
    IntegerSet set = IntegerSet::get(/*dimCount=*/1, /*symbolCount=*/0,
                                     {d0 - v}, /*eqFlags=*/{true});
    auto ifOp = affine::AffineIfOp::create(b, loc, set,
                                           ValueRange{af.getInductionVar()},
                                           /*withElseRegion=*/false);
    thenTerm = ifOp.getThenBlock()->getTerminator();
  } else {
    auto sf = cast<scf::ForOp>(lin);
    Value iv = sf.getInductionVar();
    Value cond;
    if (first) {
      cond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, iv,
                                   sf.getLowerBound());
    } else {
      Value next = arith::AddIOp::create(b, loc, iv, sf.getStep());
      cond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, next,
                                   sf.getUpperBound());
    }
    auto ifOp = scf::IfOp::create(b, loc, cond, /*withElseRegion=*/false);
    thenTerm = ifOp.thenBlock()->getTerminator();
  }
  for (Operation *op : ops)
    op->moveBefore(thenTerm);
}

void perfectizeNest(Match &m) {
  Operation *lin = m.lin;
  Block *body = loopBody(lin);

  // Epilogue -> last-iteration guard before the terminator; the inner loop's
  // results equal their yields at the last iteration.
  if (!m.epilogue.empty()) {
    Operation *term = body->getTerminator();
    sinkGuarded(lin, m.epilogue, /*first=*/false, term);
    for (auto [res, yv] : llvm::zip(lin->getResults(), term->getOperands()))
      res.replaceAllUsesWith(yv);
  }

  // Prologue -> body top (before the original first op): a store runs once
  // under the first-iteration guard; a pure op / safe load is recomputed
  // unguarded.
  Operation *anchor = &body->front();
  for (Operation *op : m.prologue) {
    if (storedMemRef(op))
      sinkGuarded(lin, {op}, /*first=*/true, anchor);
    else
      op->moveBefore(anchor);
  }

  info(Stage::Prep, lin) << "Perfectizing imperfect loop nest by sinking "
                         << (m.prologue.size() + m.epilogue.size())
                         << " surrounding ops into the inner loop";
}

//===----------------------------------------------------------------------===//
// Coalesce a perfect band into one loop.
//===----------------------------------------------------------------------===//

// A loop roots a perfect band unless its parent is an affine.for that perfectly
// nests it (parent body is exactly {loop, terminator}).
bool isBandRoot(affine::AffineForOp loop) {
  auto parent = dyn_cast<affine::AffineForOp>(loop->getParentOp());
  if (!parent)
    return true;
  Block &body = parent.getRegion().front();
  return &body.front() != loop.getOperation() ||
         loop->getNextNode() != body.getTerminator();
}

// A loop may be coalesced iff it is normalized (lower bound 0, step 1) with a
// constant trip count, the form that keeps the coalesced body pure-affine, and
// carries no iter_args: `coalesceLoops` merges the induction spaces but drops
// loop-carried values, silently rewriting an accumulator to its init constant.
bool isFlattenable(affine::AffineForOp loop) {
  return loop.getInits().empty() && loop.hasConstantLowerBound() &&
         loop.getConstantLowerBound() == 0 && loop.getStepAsInt() == 1 &&
         constantTrip(loop).has_value();
}

// Each original induction variable, rebuilt from the coalesced counter (dim 0),
// transcribed from `coalesceLoops` step 3 so the two cannot drift: the k-th is
// `counter floordiv (trips inside k) mod trip[k]`, no modulo on the outermost.
SmallVector<AffineExpr>
recoveredInductionVars(MutableArrayRef<affine::AffineForOp> band) {
  MLIRContext *ctx = band.front().getContext();
  SmallVector<AffineExpr> iv(band.size());
  AffineExpr running = getAffineDimExpr(0, ctx);
  for (unsigned idx = band.size(); idx > 0; --idx) {
    if (idx != band.size())
      running = running.floorDiv(*constantTrip(band[idx]));
    iv[idx - 1] = idx == 1 ? running : running % *constantTrip(band[idx - 1]);
  }
  return iv;
}

// Whether coalescing \p band leaves a real divider behind. Recovering an
// induction variable divides by a trip count. An array access may compose that
// divider away in the memref's row-major linearization, so accesses are judged
// on the composed expression; any other consumer (a value read, an if
// predicate, a bound) is left holding the recovery itself.
bool coalescingCostsADivider(MutableArrayRef<affine::AffineForOp> band) {
  MLIRContext *ctx = band.front().getContext();
  SmallVector<AffineExpr> recovered = recoveredInductionVars(band);
  for (auto [k, loop] : llvm::enumerate(band)) {
    bool escapes =
        !llvm::all_of(loop.getInductionVar().getUsers(), [](Operation *user) {
          return asMemAccess(user).has_value();
        });
    if (escapes) {
      AddressCost c =
          addressCost(recovered[k], AddressDelays{}, AddressDelays::refWidth);
      if (c.dividers || c.reciprocals)
        return true;
    }
  }
  llvm::DenseMap<Value, unsigned> level;
  for (auto [k, loop] : llvm::enumerate(band))
    level[loop.getInductionVar()] = k;

  return band.back()
      .walk([&](Operation *op) {
        std::optional<MemAccess> a = asMemAccess(op);
        if (!a || a->kind != AccessKind::Array)
          return WalkResult::advance();
        // Substitute each band induction variable for what the counter rebuilds
        // it as; every other operand keeps its slot, shifted past dim 0.
        unsigned numDims = a->map.getNumDims(), next = 1;
        SmallVector<AffineExpr> dims, syms;
        for (unsigned p = 0; p < numDims; ++p) {
          auto it = level.find(a->indices[p]);
          dims.push_back(it == level.end() ? getAffineDimExpr(next++, ctx)
                                           : recovered[it->second]);
        }
        for (unsigned p = 0, e = a->map.getNumSymbols(); p < e; ++p) {
          auto it = level.find(a->indices[numDims + p]);
          syms.push_back(it == level.end() ? getAffineSymbolExpr(p, ctx)
                                           : recovered[it->second]);
        }
        AffineMap coalesced = a->map.replaceDimsAndSymbols(
            dims, syms, next, a->map.getNumSymbols());
        // Only whether a division survives the fold decides this, not how
        // long it takes, so the delay table is empty.
        auto shape = cast<MemRefType>(a->root.getType()).getShape();
        AddressCost c = addressCost(coalesced, shape, AddressDelays{},
                                    AddressDelays::refWidth);
        if (c.dividers || c.reciprocals)
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

// Whether \p loop's body holds a synchronous sub-kernel call. Coalescing such a
// nest lands delinearization arithmetic beside the call, forcing the merged
// loop into an arithmetic sub-region plus a call sub-region run serially per
// iteration. Left uncoalesced, the inner loop stays a lone call the
// loop-over-calls controller fires directly.
bool callsSubKernel(affine::AffineForOp loop) {
  return loop
      .walk([](Operation *op) {
        return isSyncSubKernelCall(op) ? WalkResult::interrupt()
                                       : WalkResult::advance();
      })
      .wasInterrupted();
}

// Normalize the perfect band rooted at \p root and coalesce the longest prefix
// of it that pays.
void flattenBand(affine::AffineForOp root,
                 const FrozenRewritePatternSet &compose) {
  SmallVector<affine::AffineForOp> nest;
  affine::getPerfectlyNestedLoops(nest, root);
  normalizeBand(nest, compose);
  MutableArrayRef<affine::AffineForOp> band(nest);
  unsigned n = 0;
  // Banking needs no guard here: a bank digit that folded to a constant holds
  // no dimension, so substituting the delinearized counter for the induction
  // variables leaves it that constant.
  while (n < band.size() && isFlattenable(band[n]) && !callsSubKernel(band[n]))
    ++n;
  // Then give back levels until the addresses cost no divider. A level only
  // adds a divisor, so the first band that survives is the longest that does.
  while (n >= 2 && coalescingCostsADivider(band.take_front(n)))
    --n;
  if (n < 2)
    return;
  MutableArrayRef<affine::AffineForOp> loops = band.take_front(n);
  (void)affine::coalesceLoops(loops);
  // `loops.front()` is the coalesced loop; every level under it was erased.
  info(Stage::Prep, loops.front())
      << "Flattening perfect nest of " << n << " loops";
}

//===----------------------------------------------------------------------===//

struct LoopCanonicalizationPass
    : public allo::impl::LoopCanonicalizationPassBase<
          LoopCanonicalizationPass> {
  using LoopCanonicalizationPassBase::LoopCanonicalizationPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    affine::AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);
    affine::AffineLoadOp::getCanonicalizationPatterns(patterns, ctx);
    affine::AffineStoreOp::getCanonicalizationPatterns(patterns, ctx);
    FrozenRewritePatternSet compose(std::move(patterns));

    eraseNeverTakenLoops(getOperation());

    getOperation()->walk([&](affine::AffineForOp forOp) {
      (void)affine::promoteIfSingleIteration(forOp);
    });

    // The outermost counted loops; the recursion owns everything under them.
    SmallVector<Operation *> roots;
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isForLoop(op))
        return WalkResult::advance();
      roots.push_back(op);
      return WalkResult::skip();
    });
    for (Operation *loop : roots)
      canonicalize(loop, compose);
  }

  // One nest, outermost level first. Every rewrite below either leaves \p loop
  // in place or coalesces INTO it, so the caller's pointer stays valid.
  void canonicalize(Operation *loop, const FrozenRewritePatternSet &compose) {
    if (unrollUnderPipeline && isPipelined(loop) && hasNestedLoop(loop)) {
      if (innerLoopsUnrollable(loop)) {
        auto iv = cast<LoopLikeOpInterface>(loop).getSingleInductionVar();
        std::string name = logging::detail::describe(iv->getLoc());
        unrollInner(loop, name.empty() ? "<unnamed>" : name);
        return; // the interior is straight-line now: nothing below applies
      }
      warn(Stage::Prep, loop)
          << "Pipelined loop has a dynamic or uncounted inner loop; not "
             "unrolled, so it falls back to pipelining the innermost loop only";
    }

    // Deeper nests first, so this level decides on the shape the level below
    // has settled into, already normalized if it roots a band of its own.
    for (Operation *child : nestedLoops(loop))
      canonicalize(child, compose);

    if (perfectize) {
      std::string reason;
      if (std::optional<Match> m = matchImperfect(loop, reason))
        perfectizeNest(*m);
      else if (!reason.empty())
        warn(Stage::Prep, loop)
            << "Imperfect loop nest not perfectized because " << reason
            << "; the scheduler schedules its body as sequential sub-regions "
               "instead of one fused pipeline";
    }

    // The band this loop roots is as perfect as it will get.
    auto root = dyn_cast<affine::AffineForOp>(loop);
    if (root && isBandRoot(root))
      flattenBand(root, compose);
  }
};

} // namespace
