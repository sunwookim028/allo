/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/AliasAnalysis.h" // alloAliasAnalysis
#include "allo/TransformOps/AlloTransformOps.h"
#include "allo/TransformOps/Utils.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <numeric>

using namespace mlir;
using namespace mlir::allo;

///===----------------------------------------------------------------------===//
/// OutlineOp implementation
///===----------------------------------------------------------------------===//
/// Wraps `op` into an `scf.execute_region` operation. Supports operations with
/// either zero or one region.
static scf::ExecuteRegionOp wrapInExecuteRegion(RewriterBase &b,
                                                Operation *op) {
  if (op->getNumRegions() > 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      scf::ExecuteRegionOp::create(b, op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = nullptr;
    if (op->getNumRegions() == 0) {
      clonedOp = b.clone(*op);
    } else {
      clonedOp = b.cloneWithoutRegions(*op);
      Region &clonedRegion = clonedOp->getRegions().front();
      assert(clonedRegion.empty() && "expected empty region");
      b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                           clonedRegion.end());
    }
    scf::YieldOp::create(b, op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(RewriterBase &rewriter, Operation *op,
                                Region &region) {
  assert(region.hasOneBlock() && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, /*argValues=*/{});
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

static bool isFuncInherentAttr(StringRef name) {
  return name == SymbolTable::getSymbolAttrName() || name == "function_type" ||
         name == "sym_visibility" || name == "arg_attrs" || name == "res_attrs";
}

static KernelOp convertOutlinedFuncToKernel(RewriterBase &rewriter,
                                            func::FuncOp func,
                                            DenseI32ArrayAttr mapping) {
  rewriter.setInsertionPoint(func);
  auto kernel = KernelOp::create(
      rewriter, func.getLoc(), func.getSymNameAttr(),
      TypeAttr::get(func.getFunctionType()), func.getSymVisibilityAttr(),
      func.getArgAttrsAttr(), func.getResAttrsAttr(), mapping);
  for (NamedAttribute attr : func->getAttrs()) {
    if (!isFuncInherentAttr(attr.getName().getValue()))
      kernel->setAttr(attr.getName(), attr.getValue());
  }

  rewriter.inlineRegionBefore(func.getBody(), kernel.getBody(),
                              kernel.getBody().end());

  SmallVector<func::ReturnOp> returns;
  kernel.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
  for (func::ReturnOp ret : returns) {
    rewriter.setInsertionPoint(ret);
    ReturnOp::create(rewriter, ret.getLoc(), ret.getOperands());
    rewriter.eraseOp(ret);
  }

  rewriter.eraseOp(func);
  return kernel;
}

static InvokeOp convertOutlinedCallToInvoke(RewriterBase &rewriter,
                                            func::CallOp call,
                                            KernelOp kernel) {
  rewriter.setInsertionPoint(call);
  auto invoke =
      InvokeOp::create(rewriter, call.getLoc(), kernel, call.getOperands());
  if (auto argAttrs = call.getArgAttrsAttr())
    invoke->setAttr(invoke.getArgAttrsAttrName(), argAttrs);
  if (auto resAttrs = call.getResAttrsAttr())
    invoke->setAttr(invoke.getResAttrsAttrName(), resAttrs);
  rewriter.replaceOp(call, invoke.getResults());
  return invoke;
}

/// Outlines arbitrary operations with at most one region. Modified from
/// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SCF/TransformOps/SCFTransformOps.cpp
DiagnosedSilenceableFailure
transform::OutlineOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  SmallVector<Operation *, 4> kernels;
  SmallVector<Operation *, 4> calls;
  DenseMap<Operation *, SymbolTable> symbolTables;

  for (Operation *target : state.getPayloadOps(getTarget())) {
    Location loc = target->getLoc();
    if (target->getNumRegions() > 1) {
      return emitSilenceableFailure(target)
             << "expected target operation to have at most one region";
    }
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(target);
    auto exec = wrapInExecuteRegion(rewriter, target);
    if (!exec) {
      return emitSilenceableFailure(target)
             << "expected target operation to have at most one region";
    }
    func::CallOp call;
    auto outlined = outlineSingleBlockRegion(rewriter, loc, exec.getRegion(),
                                             getKernelName(), &call);
    if (failed(outlined)) {
      return emitSilenceableFailure(target)
             << "failed to outline the target operation";
    }

    if (symbolTableOp) {
      SymbolTable &symbolTable =
          symbolTables.try_emplace(symbolTableOp, symbolTableOp)
              .first->getSecond();
      symbolTable.insert(*outlined);
      call.setCalleeAttr(FlatSymbolRefAttr::get(*outlined));
    }
    // `scf.execute_region` is only an outlining container; inline it back.
    Operation *outlinedOp = *outlined;
    Operation *callOp = call;
    if (DenseI32ArrayAttr mapping = getMappingAttr()) {
      auto kernel = convertOutlinedFuncToKernel(rewriter, *outlined, mapping);
      auto invoke = convertOutlinedCallToInvoke(rewriter, call, kernel);
      outlinedOp = kernel;
      callOp = invoke;
    }
    replaceOpWithRegion(rewriter, exec, exec.getRegion());
    kernels.push_back(outlinedOp);
    calls.push_back(callOp);
  }
  results.set(cast<OpResult>(getKernel()), kernels);
  results.set(cast<OpResult>(getCall()), calls);
  return DiagnosedSilenceableFailure::success();
}

///==--------------------------------------------------------------------===//
/// ReorderOp implementation
///===-------------------------------------------------------------------===//

/// Checks if the given loops are in the same perfectly nested loop band and
/// returns the outermost loop, or null. The input loops need not be contiguous
/// or sorted by depth.
static affine::AffineForOp
inSamePerfectlyNestedLoopBand(ArrayRef<affine::AffineForOp> loops) {
  if (loops.empty())
    return {};
  if (loops.size() == 1)
    return {};
  auto tmp = llvm::to_vector(loops);
  DenseMap<affine::AffineForOp, unsigned> depthMap;
  llvm::for_each(tmp, [&depthMap](auto op) {
    unsigned depth = 0;
    Operation *curr = op;
    while ((curr = curr->getParentOp()))
      depth++;
    depthMap[op] = depth;
  });
  llvm::sort(tmp,
             [&depthMap](auto a, auto b) { return depthMap[a] < depthMap[b]; });

  for (unsigned i = 0; i < tmp.size() - 1; ++i) {
    affine::AffineForOp currLoop = tmp[i];
    affine::AffineForOp nextLoop = tmp[i + 1];
    if (!currLoop->isProperAncestor(nextLoop)) {
      return {};
    }
    Operation *ptr = currLoop;
    while (ptr != nextLoop) {
      auto loop = dyn_cast<affine::AffineForOp>(ptr);
      if (!loop) {
        return {};
      }
      Block *body = loop.getBody();
      // The body holds the nested affine.for and the terminator.
      if (body->getOperations().size() != 2) {
        return {};
      }
      ptr = &body->getOperations().front();
    }
  }
  return tmp.front();
}

DiagnosedSilenceableFailure
transform::LoopReorderOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {

  SmallVector<affine::AffineForOp> loops;
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(payload)) {
      loops.push_back(forOp);
    } else {
      std::string msg = "expected an affine.for operation.";
      if (isa<scf::ForOp>(payload)) {
        msg += " Try raise scf.for to affine.for before reordering.";
      }
      return emitSilenceableFailure(payload) << msg;
    }
  }
  if (loops.size() < 2) {
    return emitSilenceableError()
           << "at least two loops are required for reordering";
  }

  // The permutation is interpreted positionally over `loops`; duplicates would
  // make the permutation map non-bijective and break `permuteLoops`.
  DenseSet<Operation *> seenLoops;
  for (auto loop : loops) {
    if (!seenLoops.insert(loop).second)
      return emitSilenceableError() << "reorder loops must be unique";
  }

  auto outermostLoop = inSamePerfectlyNestedLoopBand(loops);
  if (!outermostLoop) {
    return emitSilenceableError()
           << "loops must be in the same perfectly nested loop band";
  }
  SmallVector<affine::AffineForOp, 4> band;
  affine::getPerfectlyNestedLoops(band, outermostLoop);

  if (getPermutation().size() != loops.size()) {
    return emitSilenceableError()
           << "the size of permutation must match the number of loops";
  }

  auto permutation = getPermutation();

  // Map selected loops to their original positions in the full perfect band.
  SmallVector<unsigned, 4> selectedOrgIndices;
  for (auto l : loops) {
    auto *it = llvm::find(band, l);
    if (it == band.end())
      return emitSilenceableError() << "selected loop is not in the loop band";
    unsigned idx = std::distance(band.begin(), it);
    selectedOrgIndices.push_back(idx);
  }

  // Build a full-band permutation map. Unselected loops keep identity mapping.
  SmallVector<unsigned, 4> permMap(band.size());
  std::iota(permMap.begin(), permMap.end(), 0u);

  for (unsigned i = 0; i < permutation.size(); ++i) {
    unsigned targetPos = selectedOrgIndices[i];
    unsigned srcPos = selectedOrgIndices[permutation[i]];
    permMap[targetPos] = srcPos;
  }

  if (!affine::isValidLoopInterchangePermutation(band, permMap)) {
    return emitSilenceableError() << "permutation violates legality "
                                     "constraints (e.g., data dependencies)";
  }
  affine::permuteLoops(band, permMap);
  return DiagnosedSilenceableFailure::success();
}

void transform::LoopReorderOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getLoopsMutable(), effects);
  modifiesPayload(effects);
}

LogicalResult transform::LoopReorderOp::verify() {
  // The number of loops is unknown at verification time, so only the
  // permutation itself is checked.
  unsigned nPerm = getPermutation().size();
  auto permutation = getPermutation();
  for (unsigned i = 0; i < nPerm; ++i) {
    if (permutation[i] < 0 || permutation[i] >= static_cast<int32_t>(nPerm)) {
      return emitOpError("permutation index out of bounds: ") << permutation[i];
    }
    for (unsigned j = i + 1; j < nPerm; ++j) {
      if (permutation[i] == permutation[j]) {
        return emitOpError("permutation contains duplicate index: ")
               << permutation[i];
      }
    }
  }
  return success();
}

///===----------------------------------------------------------------------===//
/// SplitOp implementation
///===----------------------------------------------------------------------===//

/// Sinks an index-reconstruction affine.apply into the innermost loop of its
/// block that is a proper ancestor of all its uses, restoring the perfect
/// nesting that reorder/tile/compute_at require between band loops.
static void sinkAffineApply(affine::AffineApplyOp apply) {
  while (!apply->use_empty()) {
    affine::AffineForOp target;
    for (Operation &op : *apply->getBlock()) {
      auto loop = dyn_cast<affine::AffineForOp>(&op);
      if (loop && llvm::all_of(apply->getUses(), [&](OpOperand &u) {
            return loop->isProperAncestor(u.getOwner());
          })) {
        target = loop;
        break;
      }
    }
    if (!target)
      break;
    apply->moveBefore(&target.getBody()->front());
  }
}

/// Checks if the given split factor is valid for the given loop.
/// A valid split factor should be positive and smaller than the loop trip
/// count. Only checks constant-bound loops.
static bool checkSplitFactor(affine::AffineForOp loop, int64_t factor) {
  if (!loop.hasConstantBounds()) {
    return true;
  }
  int64_t lb = loop.getConstantLowerBound();
  int64_t ub = loop.getConstantUpperBound();
  int64_t range = ub - lb;
  int64_t step = loop.getStepAsInt();
  if (range <= 0 || step <= 0)
    return false;
  int64_t tripCount = (range - 1) / step + 1;
  return factor > 0 && factor < tripCount;
}

static FailureOr<int64_t> stripCastInt(Value value) {
  Value current = value;
  while (true) {
    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      return failure();
    }
    if (isa<arith::IndexCastOp, arith::TruncIOp, arith::ExtUIOp,
            arith::ExtSIOp>(defOp)) {
      current = defOp->getOperand(0);
      continue;
    }
    IntegerAttr::ValueType cst;
    if (matchPattern(current, m_ConstantInt(&cst))) {
      return cst.getSExtValue();
    }
    return failure();
  }
}

static bool checkSplitFactor(scf::ForOp loop, int64_t factor) {
  auto lbOr = stripCastInt(loop.getLowerBound());
  auto ubOr = stripCastInt(loop.getUpperBound());
  auto stepOr = stripCastInt(loop.getStep());
  if (failed(lbOr) || failed(ubOr) || failed(stepOr)) {
    return true;
  }
  int64_t lb = *lbOr;
  int64_t ub = *ubOr;
  int64_t step = *stepOr;
  int64_t range = ub - lb;
  if (range <= 0 || step <= 0)
    return false;
  int64_t tripCount = (range - 1) / step + 1;
  return factor > 0 && factor < tripCount;
}

DiagnosedSilenceableFailure
transform::LoopSplitOp::applyToOne(transform::TransformRewriter &rewriter,
                                   Operation *target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  int64_t factor = getFactorAttr().getInt();
  if (factor <= 0) {
    return emitSilenceableFailure(getOperation())
           << "split factor must be positive";
  }
  if (auto forOp = dyn_cast<affine::AffineForOp>(target)) {
    if (!checkSplitFactor(forOp, factor)) {
      return emitSilenceableFailure(forOp)
             << "split factor is larger than or equal to the loop trip count";
    }

    // Capture the source IV location before tiling: the affine utility below
    // builds fresh loops whose induction variables would otherwise lose the
    // source NameLoc.
    Location ivLoc = forOp.getInductionVar().getLoc();

    // A single loop is always perfectly nested.
    SmallVector<affine::AffineForOp, 2> splitOps;
    if (failed(affine::tilePerfectlyNested(forOp, factor, &splitOps))) {
      return emitSilenceableFailure(forOp) << "failed to split the loop";
    }
    assert(splitOps.size() == 2 && "expected exactly two loops after tiling");

    auto outer = splitOps.front();
    auto inner = splitOps.back();
    // Both halves inherit the original loop variable's name; the emitter
    // uniquifies the duplicate.
    outer.getInductionVar().setLoc(ivLoc);
    inner.getInductionVar().setLoc(ivLoc);
    if (failed(affine::normalizeAffineFor(outer)) ||
        failed(affine::normalizeAffineFor(inner))) {
      return emitSilenceableFailure(forOp) << "failed to normalize the loop";
    }

    AffineMap innerUb = inner.getUpperBoundMap();
    if (innerUb.isConstant() && innerUb.getNumInputs() != 0) {
      // Drop the unused symbols of a constant upper bound map.
      auto cstUb = cast<AffineConstantExpr>(innerUb.getResult(0)).getValue();
      rewriter.setInsertionPoint(inner);
      inner.setUpperBound({}, rewriter.getConstantAffineMap(cstUb));
    }

    // Keep the band perfectly nested: erase the applies left dead by
    // normalization, then sink the live ones into the loop that uses them.
    bool erased = true;
    while (erased) {
      erased = false;
      SmallVector<affine::AffineApplyOp> dead;
      outer.walk([&](affine::AffineApplyOp a) {
        if (a->use_empty())
          dead.push_back(a);
      });
      for (auto a : dead) {
        rewriter.eraseOp(a);
        erased = true;
      }
    }
    SmallVector<affine::AffineApplyOp> applies;
    outer.walk([&](affine::AffineApplyOp a) { applies.push_back(a); });
    for (auto a : applies)
      sinkAffineApply(a);
    results.push_back(outer);
    results.push_back(inner);
    return DiagnosedSilenceableFailure::success();
  }
  if (auto forOp = dyn_cast<scf::ForOp>(target)) {
    if (!checkSplitFactor(forOp, factor)) {
      return emitSilenceableFailure(forOp)
             << "split factor is larger than or equal to the loop trip count";
    }
    Location ivLoc = forOp.getInductionVar().getLoc();
    rewriter.setInsertionPoint(forOp);
    Value cst =
        arith::ConstantIndexOp::create(rewriter, forOp.getLoc(), factor);
    auto loops = tilePerfectlyNested(forOp, cst);
    if (loops.size() != 1) {
      return emitSilenceableFailure(forOp) << "failed to split the loop";
    }
    // The freshly created inner loop inherits the source loop variable's name.
    loops.front().getInductionVar().setLoc(ivLoc);
    results.push_back(forOp);
    results.push_back(loops.front());
    return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableFailure(target)
         << "expected target operation to be an affine.for or scf.for loop";
}

///===----------------------------------------------------------------------===///
/// LoopTileOp implementation
///===----------------------------------------------------------------------===///
static unsigned getOperationDepth(Operation *op) {
  unsigned depth = 0;
  Operation *curr = op;
  while ((curr = curr->getParentOp()))
    depth++;
  return depth;
}

namespace {
template <typename ForOp> struct LoopWithFactor {
  ForOp loop;
  uint64_t factor;
};
} // namespace

/// Sort (loop, factor) pairs by loop depth and check they form a single
/// ancestor chain with unique loops.
template <typename ForOp>
static LogicalResult
sortAndCheckLoopFactorPairs(SmallVectorImpl<LoopWithFactor<ForOp>> &pairs) {
  DenseSet<Operation *> seenLoops;
  for (auto pair : pairs) {
    if (!seenLoops.insert(pair.loop).second)
      return failure();
  }

  llvm::sort(pairs, [](const LoopWithFactor<ForOp> &a,
                       const LoopWithFactor<ForOp> &b) {
    return getOperationDepth(a.loop) < getOperationDepth(b.loop);
  });

  for (unsigned i = 0; i < pairs.size() - 1; ++i) {
    if (!pairs[i].loop->isProperAncestor(pairs[i + 1].loop))
      return failure();
  }
  return success();
}

template <typename ForOp>
static bool isContiguousPerfectBand(SmallVectorImpl<ForOp> &loops) {
  if (loops.size() <= 1)
    return true;
  for (unsigned i = 0; i < loops.size() - 1; ++i) {
    Block *body = loops[i].getBody();
    if (body->getOperations().size() != 2)
      return false;
    if (&body->front() != loops[i + 1].getOperation())
      return false;
  }
  return true;
}

static FailureOr<SmallVector<uint64_t, 4>> parseTileFactors(ArrayAttr attr) {
  SmallVector<uint64_t, 4> factors;
  factors.reserve(attr.size());
  for (Attribute a : attr) {
    int64_t factor = cast<IntegerAttr>(a).getInt();
    if (factor <= 0)
      return failure();
    factors.push_back(static_cast<uint64_t>(factor));
  }
  return factors;
}

/// Binds tile `factors` to `loops` by handle order, then depth-sorts the pairs
/// while validating they are unique and form one ancestor chain. `sortedLoops`
/// and `sortedFactors` receive the depth-ordered loops and factors.
template <typename ForOp>
static DiagnosedSilenceableFailure
bindAndSortTileLoops(Operation *op, ArrayRef<ForOp> loops,
                     ArrayRef<uint64_t> factors,
                     SmallVectorImpl<ForOp> &sortedLoops,
                     SmallVectorImpl<uint64_t> &sortedFactors) {
  if (factors.size() != loops.size())
    return emitSilenceableFailure(op)
           << "number of tile factors must match the number of loops";

  SmallVector<LoopWithFactor<ForOp>, 4> loopFactors;
  loopFactors.reserve(loops.size());
  for (auto [loop, factor] : llvm::zip_equal(loops, factors))
    loopFactors.push_back({loop, factor});

  // Sorting pairs preserves loop-factor associations after reordering.
  if (failed(sortAndCheckLoopFactorPairs(loopFactors)))
    return emitSilenceableFailure(op)
           << "tile loops must be unique and in the same loop nest";

  sortedLoops.reserve(loopFactors.size());
  sortedFactors.reserve(loopFactors.size());
  for (const auto &it : loopFactors) {
    sortedLoops.push_back(it.loop);
    sortedFactors.push_back(it.factor);
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::LoopTileOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
                             transform::TransformState &state) {
  SmallVector<affine::AffineForOp, 4> affineLoops;
  SmallVector<scf::ForOp, 4> scfLoops;

  // Handle iteration order is the semantic order for mapping factors to loops.
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (auto affineFor = dyn_cast<affine::AffineForOp>(payload)) {
      affineLoops.push_back(affineFor);
    } else if (auto scfFor = dyn_cast<scf::ForOp>(payload)) {
      scfLoops.push_back(scfFor);
    } else {
      return emitSilenceableFailure(payload)
             << "expected an affine.for or scf.for operation";
    }
  }

  if (!affineLoops.empty() && !scfLoops.empty()) {
    return emitSilenceableError()
           << "cannot mix affine.for and scf.for loops in the same tiling";
  }
  if (affineLoops.empty() && scfLoops.empty()) {
    return emitSilenceableError() << "expected at least one loop to tile";
  }

  auto maybeFactors = parseTileFactors(getFactors());
  if (failed(maybeFactors))
    return emitSilenceableError() << "tile factors must be positive";
  const SmallVector<uint64_t, 4> &factors = *maybeFactors;

  if (!affineLoops.empty()) {
    SmallVector<affine::AffineForOp, 4> sortedLoops;
    SmallVector<uint64_t, 4> sortedFactors;
    if (DiagnosedSilenceableFailure diag =
            bindAndSortTileLoops<affine::AffineForOp>(
                getOperation(), affineLoops, factors, sortedLoops,
                sortedFactors);
        !diag.succeeded())
      return diag;

    SmallVector<Operation *, 4> tileLoops;
    SmallVector<Operation *, 4> pointLoops;

    // Capture the source IV locations before tiling rebuilds the loops, so the
    // tile and point induction variables keep the original names.
    SmallVector<Location, 4> srcIVLocs;
    for (auto loop : sortedLoops)
      srcIVLocs.push_back(loop.getInductionVar().getLoc());

    bool perfect = isContiguousPerfectBand<affine::AffineForOp>(sortedLoops);

    if (perfect) {
      // Perfect affine tiling creates [tile loops..., point loops...].
      SmallVector<unsigned, 4> uFactors;
      uFactors.reserve(sortedFactors.size());
      for (uint64_t factor : sortedFactors) {
        if (factor > std::numeric_limits<unsigned>::max()) {
          return emitSilenceableError()
                 << "tile factor exceeds supported affine tile size range";
        }
        uFactors.push_back(static_cast<unsigned>(factor));
      }
      SmallVector<affine::AffineForOp, 8> tiledNest;
      if (failed(
              affine::tilePerfectlyNested(sortedLoops, uFactors, &tiledNest)))
        return emitSilenceableFailure(sortedLoops.front())
               << "failed to tile affine perfectly nested loops";
      if (tiledNest.size() < sortedLoops.size() * 2)
        return emitSilenceableError()
               << "unexpected number of loops created by affine tiling";

      unsigned nLoops = sortedLoops.size();
      for (auto loop : tiledNest) {
        if (failed(affine::normalizeAffineFor(loop))) {
          return emitSilenceableFailure(loop)
                 << "failed to normalize tiled affine loop";
        }
      }

      // Canonicalize point-loop upper bounds
      for (auto loop : llvm::drop_begin(tiledNest, nLoops)) {
        AffineMap ubMap = loop.getUpperBoundMap();
        if (ubMap.isConstant() && ubMap.getNumInputs() != 0) {
          auto cstUb = cast<AffineConstantExpr>(ubMap.getResult(0)).getValue();
          rewriter.setInsertionPoint(loop);
          loop.setUpperBound({}, rewriter.getConstantAffineMap(cstUb));
        } else if (!ubMap.isConstant()) {
          if (ubMap.getNumResults() == 2 && ubMap.getNumInputs() == 1) {
            auto addMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                         ubMap.getResult(1));
            auto applyOp = dyn_cast_or_null<affine::AffineApplyOp>(
                loop.getUpperBoundOperands().front().getDefiningOp());
            if (!applyOp)
              continue;
            auto outerIV = applyOp.getOperand(0);
            AffineMap composed = addMap.compose(applyOp.getAffineMap());
            SmallVector<AffineExpr, 2> exprs{ubMap.getResult(0),
                                             composed.getResult(0)};
            auto finalMap = AffineMap::get(
                /*dimCount=*/1, /*symbolCount=*/0, exprs,
                rewriter.getContext());
            loop.setUpperBound(outerIV, finalMap);
          }
        }
      }
      // sink affine.apply into point loops when all uses are inside.
      for (unsigned i = 0; i < nLoops; ++i) {
        auto outer = tiledNest[i];
        auto point = tiledNest[i + nLoops];
        outer.getInductionVar().setLoc(srcIVLocs[i]);
        point.getInductionVar().setLoc(srcIVLocs[i]);
        for (auto applyOp : llvm::make_early_inc_range(
                 outer.getOps<affine::AffineApplyOp>())) {
          bool allUsesInPoint =
              llvm::all_of(applyOp->getUses(), [&](OpOperand &u) {
                return point->isProperAncestor(u.getOwner());
              });
          if (allUsesInPoint)
            applyOp->moveBefore(&point.getBody()->front());
        }
        tileLoops.push_back(outer);
        pointLoops.push_back(point);
      }
      // sink affine.apply to innermost point loops to make perfectly nested
      for (unsigned i = nLoops; i < tiledNest.size() - 1; ++i) {
        auto point = tiledNest[i];
        auto nextPoint = tiledNest[i + 1];
        for (auto applyOp : llvm::make_early_inc_range(
                 point.getOps<affine::AffineApplyOp>())) {
          bool allUsesInNextPoint =
              llvm::all_of(applyOp->getUses(), [&](OpOperand &u) {
                return nextPoint->isProperAncestor(u.getOwner());
              });
          if (allUsesInNextPoint)
            applyOp->moveBefore(&nextPoint.getBody()->front());
        }
      }
    } else {
      // Imperfect affine tiling strip-mines selected loops and sinks them
      // under the chosen target while preserving sorted loop order.
      auto point = affine::tile(sortedLoops, sortedFactors, sortedLoops.back());
      if (point.size() != sortedLoops.size()) {
        return emitSilenceableError()
               << "failed to tile affine imperfectly nested loops";
      }
      for (auto loop : sortedLoops)
        tileLoops.push_back(loop);
      for (auto [i, loop] : llvm::enumerate(point))
        loop.getInductionVar().setLoc(srcIVLocs[i]);
      for (auto loop : point)
        pointLoops.push_back(loop);
    }

    // Output handles are always reported in depth order.
    results.set(cast<OpResult>(getTileLoops()), tileLoops);
    results.set(cast<OpResult>(getPointLoops()), pointLoops);
    return DiagnosedSilenceableFailure::success();
  }

  if (!scfLoops.empty()) {
    SmallVector<scf::ForOp, 4> sortedLoops;
    SmallVector<uint64_t, 4> sortedFactors;
    if (DiagnosedSilenceableFailure diag = bindAndSortTileLoops<scf::ForOp>(
            getOperation(), scfLoops, factors, sortedLoops, sortedFactors);
        !diag.succeeded())
      return diag;

    SmallVector<Operation *, 4> tileLoops;
    SmallVector<Operation *, 4> pointLoops;

    // Capture the source IV locations so the freshly created point loops keep
    // the original names; the reused tile loops already retain theirs.
    SmallVector<Location, 4> srcIVLocs;
    for (auto loop : sortedLoops)
      srcIVLocs.push_back(loop.getInductionVar().getLoc());

    SmallVector<Value, 4> sizeVals;
    sizeVals.reserve(sortedFactors.size());
    rewriter.setInsertionPoint(sortedLoops.front());
    for (uint64_t factor : sortedFactors) {
      if (factor > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return emitSilenceableError()
               << "tile factor exceeds supported scf tile size range";
      sizeVals.push_back(
          arith::ConstantIndexOp::create(rewriter, sortedLoops.front().getLoc(),
                                         static_cast<int64_t>(factor)));
    }

    bool perfect = isContiguousPerfectBand<scf::ForOp>(sortedLoops);
    if (perfect) {
      // Perfect scf tiling returns only point loops; outer loops are updated
      // in-place and remain represented by sortedLoops.
      SmallVector<scf::ForOp, 8> point =
          tilePerfectlyNested(sortedLoops.front(), sizeVals);
      if (point.size() != sortedLoops.size()) {
        return emitSilenceableError()
               << "failed to tile scf perfectly nested loops";
      }
      for (auto loop : sortedLoops)
        tileLoops.push_back(loop);
      for (auto [i, loop] : llvm::enumerate(point))
        loop.getInductionVar().setLoc(srcIVLocs[i]);
      for (auto loop : point)
        pointLoops.push_back(loop);
    } else {
      // Imperfect scf tiling strip-mines selected loops and sinks them
      // under the chosen target while preserving sorted loop order.
      auto point = ::mlir::tile(sortedLoops, sizeVals, sortedLoops.back());
      if (point.size() != sortedLoops.size()) {
        return emitSilenceableError()
               << "failed to tile scf imperfectly nested loops";
      }
      for (auto [i, loop] : llvm::enumerate(point))
        loop.getInductionVar().setLoc(srcIVLocs[i]);
      for (auto loop : sortedLoops)
        tileLoops.push_back(loop);
      for (auto loop : point)
        pointLoops.push_back(loop);
    }

    // Output handles are always reported in depth order.
    results.set(cast<OpResult>(getTileLoops()), tileLoops);
    results.set(cast<OpResult>(getPointLoops()), pointLoops);
    return DiagnosedSilenceableFailure::success();
  }

  return emitSilenceableError() << "failed to tile loops";
}

///===----------------------------------------------------------------------===//
/// LoopFlattenOp implementation
///===----------------------------------------------------------------------===///

// modified from lib/Transforms/Utils/LoopUtils.cpp
static void coalesceLoops(MutableArrayRef<affine::AffineForOp> loops,
                          int64_t flattenedTripCount,
                          transform::TransformRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  affine::AffineForOp innermost = loops.back();
  affine::AffineForOp outermost = loops.front();
  Location loc = outermost.getLoc();

  SmallVector<int64_t, 4> ubs;
  for (auto loop : loops) {
    auto cstUb = loop.getConstantUpperBound();
    ubs.push_back(cstUb);
  }

  // The flattened trip count is validated by the caller.
  outermost.setConstantUpperBound(flattenedTripCount);

  // Remap the induction variables as
  //   iv_i = floordiv(iv_linear, product-of-loop-ranges-until-i) mod range_i,
  // computed from the innermost loop outwards as a running quotient.
  rewriter.setInsertionPointToStart(outermost.getBody());
  Value previous = outermost.getInductionVar();
  SmallVector<Operation *> opToSink;
  for (unsigned idx = loops.size(); idx > 0; --idx) {
    int64_t currUb = ubs[idx - 1];
    if (idx != loops.size()) {
      // Divide by the range of the loop processed in the previous (inner)
      // iteration, `ubs[idx]`, not this loop's own bound.
      auto quotientMap =
          AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                         rewriter.getAffineDimExpr(0).floorDiv(ubs[idx]));
      previous =
          affine::AffineApplyOp::create(rewriter, loc, quotientMap, previous);
      opToSink.push_back(previous.getDefiningOp());
    }
    Value inductionVariable;
    if (idx == 1) {
      inductionVariable = previous;
    } else {
      auto modMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                   rewriter.getAffineDimExpr(0) % currUb);
      inductionVariable =
          affine::AffineApplyOp::create(rewriter, loc, modMap, previous);
      opToSink.push_back(inductionVariable.getDefiningOp());
    }
    replaceAllUsesInRegionWith(loops[idx - 1].getInductionVar(),
                               inductionVariable, loops.back().getRegion());
  }

  // Move the innermost body above the second-outermost loop, then erase that
  // loop and the extra terminator.
  affine::AffineForOp secondOutermostLoop = loops[1];
  innermost.getBody()->back().erase();
  outermost.getBody()->getOperations().splice(
      Block::iterator(secondOutermostLoop.getOperation()),
      innermost.getBody()->getOperations());
  for (auto [iter, init] :
       llvm::zip_equal(secondOutermostLoop.getRegionIterArgs(),
                       secondOutermostLoop.getInits())) {
    iter.replaceAllUsesWith(init);
    iter.dropAllUses();
  }
  secondOutermostLoop.erase();

  // Sink the index applies into a nested loop that contains all their users.
  std::reverse(opToSink.begin(), opToSink.end());
  outermost.walk([&](affine::AffineForOp nestedLoop) {
    if (nestedLoop == outermost)
      return;
    bool canSinkAll = true;
    for (Operation *op : opToSink) {
      for (Operation *user : op->getUsers()) {
        if (!nestedLoop->isAncestor(user)) {
          canSinkAll = false;
          break;
        }
      }
      if (!canSinkAll)
        break;
    }
    if (canSinkAll) {
      Block *body = nestedLoop.getBody();
      for (Operation *op : opToSink) {
        op->moveBefore(&body->front());
      }
    }
  });
}

DiagnosedSilenceableFailure
transform::LoopFlattenOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  SmallVector<affine::AffineForOp, 4> loops;
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(payload)) {
      loops.push_back(forOp);
    } else {
      if (isa<scf::ForOp>(payload)) {
        return emitSilenceableFailure(payload)
               << "Try raise scf.for to affine.for before flattening.";
      }
      return emitSilenceableFailure(payload)
             << "expected an affine.for operation";
    }
  }
  if (loops.size() < 2) {
    return emitSilenceableError()
           << "at least two loops are required for flattening";
  }

  // Flatten supports unordered loop handles; normalize to depth order first.
  llvm::sort(loops, [](auto a, auto b) {
    return getOperationDepth(a) < getOperationDepth(b);
  });

  auto selectedOutermost = inSamePerfectlyNestedLoopBand(loops);
  if (!selectedOutermost) {
    return emitSilenceableError()
           << "loops must be in the same perfectly nested loop band";
  }

  // Flatten a contiguous band from selected outermost to selected innermost.
  SmallVector<affine::AffineForOp, 4> perfectBand;
  affine::getPerfectlyNestedLoops(perfectBand, selectedOutermost);
  auto *endIt = llvm::find(perfectBand, loops.back());
  if (endIt == perfectBand.end()) {
    return emitSilenceableError()
           << "failed to find selected innermost loop in perfect loop band";
  }
  SmallVector<affine::AffineForOp, 4> flattenBand(perfectBand.begin(),
                                                  std::next(endIt));

  // Current coalescing logic assumes normalized affine loops with constant
  // trip counts.
  int64_t flattenedTripCount = 1;
  for (auto loop : flattenBand) {
    // Coalescing erases the inner terminator and rewrites induction variables;
    // loop-carried values are not handled and would be left dangling.
    if (loop.getNumResults() != 0) {
      return emitSilenceableError()
             << "flatten does not support affine.for loops carrying iteration "
                "arguments";
    }
    if (loop.getStepAsInt() != 1 || !loop.hasConstantLowerBound() ||
        loop.getConstantLowerBound() != 0 || !loop.hasConstantUpperBound()) {
      return emitSilenceableError()
             << "flatten requires normalized affine.for loops with step=1, "
                "constant lower bound=0 and constant upper bound";
    }
    int64_t ub = loop.getConstantUpperBound();
    if (ub <= 0) {
      return emitSilenceableError()
             << "flatten requires positive constant upper bounds";
    }
    if (flattenedTripCount > std::numeric_limits<int64_t>::max() / ub) {
      return emitSilenceableError()
             << "flattened loop trip count overflows int64";
    }
    flattenedTripCount *= ub;
  }

  ::coalesceLoops(flattenBand, flattenedTripCount, rewriter);

  results.set(cast<OpResult>(getResult()), {flattenBand.front()});
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===//
/// ComputeAt implementation
///===----------------------------------------------------------------------===///

static std::optional<std::string>
tryAffineLoopFusion(affine::AffineForOp producer, affine::AffineForOp consumer,
                    unsigned targetDepth) {
  using affine::FusionResult;
  affine::ComputationSliceState sliceState;
  affine::FusionStrategy strategy(affine::FusionStrategy::ProducerConsumer);
  FusionResult test = affine::canFuseLoops(producer, consumer, targetDepth,
                                           &sliceState, strategy);
  if (test.value == FusionResult::Success) {
    affine::fuseLoops(producer, consumer, sliceState);
    producer.erase();
    return std::nullopt;
  }
  std::string reason;
  if (test.value == FusionResult::FailPrecondition) {
    reason = "failed precondition for fusion (e.g. same block)";
  } else if (test.value == FusionResult::FailBlockDependence) {
    reason = "fusion would violate another dependence in block";
  } else if (test.value == FusionResult::FailFusionDependence) {
    reason = "fusion would reverse dependences between loops";
  } else if (test.value == FusionResult::FailComputationSlice) {
    reason = "unable to compute src loop computation slice";
  } else if (test.value == FusionResult::FailIncorrectSlice) {
    reason = "slice is computed, but it is incorrect";
  }
  return reason;
}

namespace {
enum class DependenceType : uint8_t {
  NONE = 0,
  RAW = 1 << 1u,
  WAR = 1 << 2u,
  WAW = 1 << 3u,
};

DependenceType operator|(DependenceType a, DependenceType b) {
  return static_cast<DependenceType>(static_cast<uint8_t>(a) |
                                     static_cast<uint8_t>(b));
}

DependenceType operator&(DependenceType a, DependenceType b) {
  return static_cast<DependenceType>(static_cast<uint8_t>(a) &
                                     static_cast<uint8_t>(b));
}
} // namespace

// Checks dependencies between two affine.for loop nests up to `depth`, with
// `forOpA` the source and `forOpB` the sink, and returns a DependenceType mask.
static FailureOr<DependenceType> checkDependencies(affine::AffineForOp forOpA,
                                                   affine::AffineForOp forOpB,
                                                   unsigned depth) {
  SmallVector<affine::MemRefAccess, 4> accA;
  SmallVector<affine::MemRefAccess, 4> accB;
  bool hasUnsupportedAccess = false;
  // Collect only affine accesses; a non-affine memref access is conservatively
  // unsupported so the transform fails instead of mis-transforming.
  forOpA.walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      accA.emplace_back(op);
    } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasUnsupportedAccess = true;
    }
  });
  forOpB.walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      accB.emplace_back(op);
    } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasUnsupportedAccess = true;
    }
  });
  if (hasUnsupportedAccess)
    return failure();

  DependenceType ret = DependenceType::NONE;
  for (auto &a : accA) {
    for (auto &b : accB) {
      if (a.memref != b.memref) {
        continue;
      }
      if (!a.isStore() && !b.isStore()) {
        continue;
      }
      SmallVector<affine::DependenceComponent, 2> deps;
      // `checkMemrefAccessDependence` requires the probed depth to be in
      // [1, numCommonSurroundingLoops + 1]; producer and consumer are sibling
      // nests, so clamp the caller's depth into that range.
      unsigned numCommon =
          affine::getNumCommonSurroundingLoops(*a.opInst, *b.opInst);
      unsigned probeDepth = std::min(depth, numCommon + 1);
      auto depResult =
          affine::checkMemrefAccessDependence(a, b, probeDepth, nullptr, &deps);
      if (depResult.value == affine::DependenceResult::Failure) {
        return failure();
      }
      if (depResult.value == affine::DependenceResult::HasDependence) {
        if (a.isStore() && !b.isStore()) {
          ret = ret | DependenceType::RAW;
        } else if (!a.isStore() && b.isStore()) {
          ret = ret | DependenceType::WAR;
        } else if (a.isStore() && b.isStore()) {
          ret = ret | DependenceType::WAW;
        }
      }
    }
  }
  return ret;
}

namespace {
struct ConstantBoundsPrefix {
  SmallVector<int64_t, 4> lowerBounds;
  SmallVector<int64_t, 4> upperBounds;
};
} // namespace

static SmallVector<affine::AffineForOp, 4>
collectAffineLoopChain(Operation *op) {
  SmallVector<affine::AffineForOp, 4> chain;
  for (Operation *curr = op; curr; curr = curr->getParentOp()) {
    if (auto loop = dyn_cast<affine::AffineForOp>(curr))
      chain.push_back(loop);
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

static FailureOr<ConstantBoundsPrefix>
getConstantBoundsPrefix(ArrayRef<affine::AffineForOp> chain, unsigned depth) {
  if (chain.size() < depth)
    return failure();
  ConstantBoundsPrefix bounds;
  bounds.lowerBounds.reserve(depth);
  bounds.upperBounds.reserve(depth);
  for (unsigned i = 0; i < depth; ++i) {
    affine::AffineForOp loop = chain[i];
    if (!loop.hasConstantBounds())
      return failure();
    bounds.lowerBounds.push_back(loop.getConstantLowerBound());
    bounds.upperBounds.push_back(loop.getConstantUpperBound());
  }
  return bounds;
}

static bool hasSinglePathLoopPrefix(ArrayRef<affine::AffineForOp> chain,
                                    unsigned prefixDepth) {
  if (prefixDepth > chain.size())
    return false;
  for (unsigned i = 0; i + 1 < prefixDepth; ++i) {
    affine::AffineForOp current = chain[i];
    affine::AffineForOp next = chain[i + 1];
    Block *body = current.getBody();
    if (body->getOperations().size() != 2)
      return false;
    if (&body->front() != next.getOperation())
      return false;
  }
  return true;
}

static bool hasIdenticalBounds(const ConstantBoundsPrefix &a,
                               const ConstantBoundsPrefix &b) {
  return a.lowerBounds == b.lowerBounds && a.upperBounds == b.upperBounds;
}

static bool isSubsetBounds(const ConstantBoundsPrefix &producerBounds,
                           const ConstantBoundsPrefix &consumerBounds) {
  for (auto [prodLb, prodUb, consLb, consUb] : llvm::zip_equal(
           producerBounds.lowerBounds, producerBounds.upperBounds,
           consumerBounds.lowerBounds, consumerBounds.upperBounds)) {
    if (prodLb < consLb || prodUb > consUb)
      return false;
  }
  return true;
}

static IntegerSet buildSubsetGuardSet(OpBuilder &builder,
                                      const ConstantBoundsPrefix &bounds) {
  unsigned depth = bounds.lowerBounds.size();
  SmallVector<AffineExpr, 8> constraints;
  constraints.reserve(depth * 2);
  for (unsigned i = 0; i < depth; ++i) {
    AffineExpr dim = builder.getAffineDimExpr(i);
    constraints.push_back(dim - bounds.lowerBounds[i]);
    constraints.push_back((bounds.upperBounds[i] - 1) - dim);
  }
  SmallVector<bool, 8> isEq(constraints.size(), false);
  return IntegerSet::get(depth, /*symbolCount=*/0, constraints, isEq);
}

static void remapProducerIVPrefix(ArrayRef<affine::AffineForOp> producerChain,
                                  ArrayRef<affine::AffineForOp> consumerChain,
                                  unsigned depth, Region &region) {
  for (unsigned i = 0; i < depth; ++i) {
    affine::AffineForOp producer = producerChain[i];
    affine::AffineForOp consumer = consumerChain[i];
    replaceAllUsesInRegionWith(producer.getInductionVar(),
                               consumer.getInductionVar(), region);
  }
}

static bool hasBlockingSideEffectsBetween(Operation *before, Operation *after) {
  assert(before->getBlock() == after->getBlock() &&
         "expected operations in the same block");
  assert(before->isBeforeInBlock(after) &&
         "expected `before` to appear before `after`");

  for (Operation *curr = before->getNextNode(); curr && curr != after;
       curr = curr->getNextNode()) {
    if (!isMemoryEffectFree(curr))
      return true;
  }
  return false;
}

namespace {
struct ComputeAtAnalysis {
  Operation *producerOp = nullptr;
  affine::AffineForOp consumerLoop = nullptr;
  SmallVector<affine::AffineForOp, 4> producerChain;
  SmallVector<affine::AffineForOp, 4> consumerChain;
  affine::AffineForOp producerRoot = nullptr;
  affine::AffineForOp consumerRoot = nullptr;
  unsigned producerDepth = 0;
  unsigned consumerDepth = 0;
  SmallVector<Value, 4> consumerPrefixIVs;
};
} // namespace

static std::optional<std::string>
analyzeComputeAt(Operation *producerOp, affine::AffineForOp consumerLoop,
                 ComputeAtAnalysis &analysis) {
  // Collect the producer and consumer loop chains and check the structural
  // preconditions of compute_at.
  analysis.producerOp = producerOp;
  analysis.consumerLoop = consumerLoop;
  analysis.producerChain = collectAffineLoopChain(producerOp);
  if (analysis.producerChain.empty()) {
    return std::string("producer must be inside an affine.for loop nest");
  }

  analysis.consumerChain = collectAffineLoopChain(consumerLoop);
  if (analysis.consumerChain.empty()) {
    return std::string("expected consumer_loop to resolve to an affine.for");
  }

  analysis.producerDepth = analysis.producerChain.size();
  analysis.consumerDepth = analysis.consumerChain.size();
  analysis.producerRoot = analysis.producerChain.front();
  analysis.consumerRoot = analysis.consumerChain.front();

  analysis.consumerPrefixIVs.clear();
  analysis.consumerPrefixIVs.reserve(analysis.consumerDepth);
  for (affine::AffineForOp loop : analysis.consumerChain)
    analysis.consumerPrefixIVs.push_back(loop.getInductionVar());

  if (analysis.producerRoot == analysis.consumerRoot) {
    return std::string(
        "producer and consumer must belong to different root loop nests");
  }
  if (analysis.producerRoot->getBlock() != analysis.consumerRoot->getBlock()) {
    return std::string(
        "producer and consumer loop nests must be in the same block");
  }
  if (analysis.producerDepth < analysis.consumerDepth) {
    return std::string(
        "producer loop nest depth is shallower than consumer depth");
  }
  return std::nullopt;
}

static std::optional<std::string>
applyNoDependenceMove(transform::TransformRewriter &rewriter,
                      ComputeAtAnalysis &analysis) {
  unsigned consumerDepth = analysis.consumerDepth;
  unsigned producerDepth = analysis.producerDepth;

  // Only a single-path producer prefix is rewritten; imperfect control flow
  // there would make the region move and IV remap ambiguous.
  unsigned prefixDepthToValidate =
      producerDepth == consumerDepth ? producerDepth : consumerDepth + 1;
  if (!hasSinglePathLoopPrefix(analysis.producerChain, prefixDepthToValidate)) {
    return std::string(
        "producer loop prefix to be rewritten must be perfectly nested");
  }

  // No-dependence move must preserve top-level order and cannot jump over
  // side-effecting operations between producer root and consumer root.
  if (!analysis.producerRoot->isBeforeInBlock(analysis.consumerRoot)) {
    return std::string(
        "producer root loop must appear before consumer root loop");
  }
  if (hasBlockingSideEffectsBetween(analysis.producerRoot,
                                    analysis.consumerRoot)) {
    return std::string("cannot move producer across side-effecting operations "
                       "between producer and consumer roots");
  }

  // No-dependence path currently supports constant-bound prefix reasoning only;
  // subset bounds are handled by generating an affine.if guard.
  FailureOr<ConstantBoundsPrefix> producerBounds =
      getConstantBoundsPrefix(analysis.producerChain, consumerDepth);
  FailureOr<ConstantBoundsPrefix> consumerBounds =
      getConstantBoundsPrefix(analysis.consumerChain, consumerDepth);
  if (failed(producerBounds) || failed(consumerBounds)) {
    return std::string("compute_at currently supports only constant-bounds "
                       "loops for no-dependence move");
  }

  bool identicalBounds = hasIdenticalBounds(*producerBounds, *consumerBounds);
  if (!identicalBounds && !isSubsetBounds(*producerBounds, *consumerBounds)) {
    return std::string("producer loop bounds must be identical to, or a "
                       "subset of, consumer bounds");
  }

  // Move producer body/subtree under consumer loop. If bounds are subset-only,
  // first materialize an affine.if so execution stays within producer domain.
  Block *destination = analysis.consumerLoop.getBody();
  Region *ivRemapRegion = &analysis.consumerLoop.getRegion();
  if (!identicalBounds) {
    rewriter.setInsertionPointToStart(analysis.consumerLoop.getBody());
    auto ifOp = affine::AffineIfOp::create(
        rewriter, analysis.consumerLoop.getLoc(),
        buildSubsetGuardSet(rewriter, *producerBounds),
        analysis.consumerPrefixIVs,
        /*withElseRegion=*/false);
    destination = ifOp.getThenBlock();
    ivRemapRegion = &ifOp.getThenRegion();
  }

  if (producerDepth == consumerDepth) {
    Block *producerInnermostBody = analysis.producerChain.back().getBody();
    Value consumerInnermostIV = analysis.consumerChain.back().getInductionVar();
    rewriter.eraseOp(producerInnermostBody->getTerminator());
    rewriter.inlineBlockBefore(producerInnermostBody, destination,
                               destination->begin(), consumerInnermostIV);
  } else {
    Operation *producerSubtree =
        analysis.producerChain[consumerDepth].getOperation();
    rewriter.moveOpBefore(producerSubtree, destination, destination->begin());
  }

  unsigned remapDepth = consumerDepth;
  if (producerDepth == consumerDepth)
    remapDepth -= 1;
  remapProducerIVPrefix(analysis.producerChain, analysis.consumerChain,
                        remapDepth, *ivRemapRegion);
  analysis.producerRoot.erase();
  return std::nullopt;
}

static bool mayWriteAliasingMemref(Operation *op, Value memref,
                                   AliasAnalysis &aliasAnalysis) {
  if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(op))
    return !aliasAnalysis.alias(writeOp.getMemRef(), memref).isNo();

  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    iface.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        continue;
      Value effectValue = effect.getValue();
      if (!effectValue)
        return true;
      if (!aliasAnalysis.alias(effectValue, memref).isNo())
        return true;
    }
    return false;
  }

  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &nested : block) {
          if (mayWriteAliasingMemref(&nested, memref, aliasAnalysis))
            return true;
        }
      }
    }
    return false;
  }

  // Unknown ops are conservatively assumed to possibly write.
  return true;
}

static void runComputeAtPostCleanup(affine::AffineForOp consumerLoop) {
  // Forward stores to loads within the transformed loop nest only, avoiding a
  // full-function affineScalarReplace.
  Operation *scopeOp = nullptr;
  if (auto kernel = consumerLoop->getParentOfType<func::FuncOp>())
    scopeOp = kernel.getOperation();
  else
    scopeOp = consumerLoop->getParentOp();
  AliasAnalysis aliasAnalysis = alloAliasAnalysis(scopeOp);

  SmallVector<affine::AffineReadOpInterface, 16> loads;
  consumerLoop.walk(
      [&](affine::AffineReadOpInterface loadOp) { loads.push_back(loadOp); });

  SmallVector<Operation *, 16> loadsToErase;
  for (affine::AffineReadOpInterface loadOp : loads) {
    Operation *load = loadOp.getOperation();
    if (!load || !load->getBlock())
      continue;

    Value loadMemref = loadOp.getMemRef();
    affine::MemRefAccess loadAccess(load);
    Operation *forwardingStore = nullptr;

    for (Operation *curr = load->getPrevNode(); curr;
         curr = curr->getPrevNode()) {
      if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(curr)) {
        // Non-aliasing stores cannot affect this load.
        if (aliasAnalysis.alias(storeOp.getMemRef(), loadMemref).isNo())
          continue;

        affine::MemRefAccess storeAccess(curr);
        if (storeAccess == loadAccess)
          forwardingStore = curr;
        // Any aliasing store blocks the search in this block.
        break;
      }

      if (mayWriteAliasingMemref(curr, loadMemref, aliasAnalysis))
        break;
    }

    if (!forwardingStore)
      continue;
    Value storeValue =
        cast<affine::AffineWriteOpInterface>(forwardingStore).getValueToStore();
    if (storeValue.getType() != loadOp.getValue().getType())
      continue;
    loadOp.getValue().replaceAllUsesWith(storeValue);
    loadsToErase.push_back(load);
  }

  for (Operation *load : loadsToErase)
    load->erase();
}

DiagnosedSilenceableFailure
transform::ComputeAtOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
                              transform::TransformState &states) {
  auto consumerLoops = states.getPayloadOps(getConsumerLoop());
  auto producers = states.getPayloadOps(getProducer());
  if (!llvm::hasSingleElement(consumerLoops) ||
      !llvm::hasSingleElement(producers)) {
    return emitSilenceableError()
           << "expected exactly one producer and one consumer loop";
  }

  Operation *producerOp = *producers.begin();
  auto consumerLoop = dyn_cast<affine::AffineForOp>(*consumerLoops.begin());
  if (!consumerLoop) {
    return emitSilenceableError()
           << "expected consumer_loop to resolve to an affine.for";
  }

  ComputeAtAnalysis analysis;
  if (auto reason = analyzeComputeAt(producerOp, consumerLoop, analysis)) {
    return emitSilenceableError() << *reason;
  }

  // The dependence class decides between affine fusion and a manual move.
  auto depTypeOr = checkDependencies(
      analysis.producerRoot, analysis.consumerLoop, analysis.consumerDepth);
  if (failed(depTypeOr)) {
    return emitSilenceableError()
           << "dependence analysis failed; refusing compute_at";
  }
  DependenceType depType = *depTypeOr;

  if ((depType & DependenceType::RAW) != DependenceType::NONE) {
    // Fusion targets the consumer's root (its sibling in the same block) at
    // `consumerDepth`; the axis loop is reached through that depth, not by
    // passing the inner loop as the destination.
    auto reason = tryAffineLoopFusion(
        analysis.producerRoot, analysis.consumerRoot, analysis.consumerDepth);
    if (reason.has_value()) {
      return emitSilenceableError()
             << "cannot fuse producer and consumer loop nests: "
             << reason.value();
    }
  } else if (depType == DependenceType::NONE) {
    if (auto reason = applyNoDependenceMove(rewriter, analysis)) {
      return emitSilenceableError() << *reason;
    }
  } else {
    return emitSilenceableError()
           << "compute_at does not support WAR/WAW-only dependences";
  }

  runComputeAtPostCleanup(analysis.consumerLoop);

  return DiagnosedSilenceableFailure::success();
}

void transform::ComputeAtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerMutable(), effects);
  // consumer_loop is read-only and must remain reusable.
  onlyReadsHandle(getConsumerLoopMutable(), effects);
  modifiesPayload(effects);
}

///===----------------------------------------------------------------------===//
/// Shared buffer-access analysis helpers
///===----------------------------------------------------------------------===//
namespace {
struct ComposedBufferAccess {
  Operation *op = nullptr;
  AffineMap map;
  SmallVector<Value, 4> operands;
};
} // namespace

static ComposedBufferAccess composeBufferAccess(Operation *accessOp) {
  affine::MemRefAccess access(accessOp);
  affine::AffineValueMap accessValueMap;
  access.getAccessMap(&accessValueMap);
  accessValueMap.composeSimplifyAndCanonicalize();
  return {accessOp, accessValueMap.getAffineMap(),
          llvm::to_vector(accessValueMap.getOperands())};
}

static void
collectFootprintOperands(ArrayRef<ComposedBufferAccess> accesses,
                         ArrayRef<affine::AffineForOp> innerLoops,
                         ArrayRef<Value> excludedValues,
                         DenseMap<Value, unsigned> &prefixOperandPos,
                         SmallVectorImpl<Value> &prefixOperands) {
  DenseSet<Value> excluded;
  for (Value value : excludedValues)
    excluded.insert(stripCast(value));
  for (affine::AffineForOp loop : innerLoops)
    excluded.insert(stripCast(loop.getInductionVar()));

  for (const ComposedBufferAccess &access : accesses) {
    for (AffineExpr resultExpr : access.map.getResults()) {
      for (Value operand : access.operands) {
        operand = stripCast(operand);
        if (excluded.contains(operand))
          continue;
        if (!affineExprUsesValue(resultExpr, access.operands,
                                 access.map.getNumDims(), operand)) {
          continue;
        }
        if (prefixOperandPos.contains(operand))
          continue;
        prefixOperandPos[operand] = prefixOperands.size();
        prefixOperands.push_back(operand);
      }
    }
  }
}

static void populateExprReplacements(
    AffineMap accessMap, ValueRange accessOperands,
    DenseMap<Value, unsigned> &prefixOperandPos,
    ArrayRef<std::pair<Value, AffineExpr>> explicitReplacements,
    unsigned prefixDimOffset, SmallVectorImpl<AffineExpr> &dimReplacements,
    SmallVectorImpl<AffineExpr> &symReplacements) {
  dimReplacements.clear();
  symReplacements.clear();
  dimReplacements.reserve(accessMap.getNumDims());
  symReplacements.reserve(accessMap.getNumSymbols());

  auto getReplacement = [&](Value operand) {
    operand = stripCast(operand);
    for (const auto &[explicitOperand, replacement] : explicitReplacements) {
      if (operand == stripCast(explicitOperand))
        return replacement;
    }

    auto it = prefixOperandPos.find(operand);
    if (it == prefixOperandPos.end())
      return getAffineConstantExpr(0, accessMap.getContext());
    return getAffineDimExpr(prefixDimOffset + it->second,
                            accessMap.getContext());
  };

  for (unsigned i = 0; i < accessMap.getNumDims(); ++i)
    dimReplacements.push_back(getReplacement(accessOperands[i]));
  for (unsigned i = 0; i < accessMap.getNumSymbols(); ++i)
    symReplacements.push_back(
        getReplacement(accessOperands[accessMap.getNumDims() + i]));
}

static void populateExprReplacements(
    AffineMap accessMap, ArrayRef<Value> accessOperands,
    DenseMap<Value, unsigned> &prefixOperandPos,
    SmallVectorImpl<Value> & /*prefixOperands*/, Value targetLoopIV,
    std::optional<int64_t> targetLoopConstant,
    std::optional<unsigned> targetLoopDimPos, unsigned prefixDimOffset,
    SmallVectorImpl<AffineExpr> &dimReplacements,
    SmallVectorImpl<AffineExpr> &symReplacements) {
  SmallVector<std::pair<Value, AffineExpr>, 1> explicitReplacements;
  if (targetLoopIV) {
    assert(targetLoopConstant.has_value() != targetLoopDimPos.has_value() &&
           "expected exactly one loop replacement kind");
    AffineExpr replacement =
        targetLoopDimPos
            ? getAffineDimExpr(*targetLoopDimPos, accessMap.getContext())
            : getAffineConstantExpr(*targetLoopConstant,
                                    accessMap.getContext());
    explicitReplacements.emplace_back(targetLoopIV, replacement);
  }

  populateExprReplacements(accessMap, accessOperands, prefixOperandPos,
                           explicitReplacements, prefixDimOffset,
                           dimReplacements, symReplacements);
}

static FailureOr<int64_t> getLinearAffineDimCoefficient(AffineExpr expr,
                                                        unsigned numDims,
                                                        unsigned dimPos) {
  FlatLinearConstraints localVarCst(numDims, /*numSymbols=*/0);
  SmallVector<int64_t, 8> flattenedExpr;
  if (failed(getFlattenedAffineExpr(expr, numDims, /*numSymbols=*/0,
                                    &flattenedExpr, &localVarCst)))
    return failure();
  if (localVarCst.getNumLocalVars() != 0 || flattenedExpr.size() != numDims + 1)
    return failure();
  return flattenedExpr[dimPos];
}

static FailureOr<int64_t> getConstantExprDelta(AffineExpr lhs, AffineExpr rhs,
                                               unsigned numDims) {
  FlatLinearConstraints localVarCst(numDims, /*numSymbols=*/0);
  SmallVector<int64_t, 8> flattenedExpr;
  AffineExpr diff = simplifyAffineExpr(lhs - rhs, numDims, /*numSymbols=*/0);
  if (failed(getFlattenedAffineExpr(diff, numDims, /*numSymbols=*/0,
                                    &flattenedExpr, &localVarCst)))
    return failure();
  if (localVarCst.getNumLocalVars() != 0 || flattenedExpr.size() != numDims + 1)
    return failure();
  for (unsigned i = 0, e = flattenedExpr.size() - 1; i < e; ++i)
    if (flattenedExpr[i] != 0)
      return failure();
  return flattenedExpr.back();
}

///===----------------------------------------------------------------------===//
/// ReuseAt implementation
///===----------------------------------------------------------------------===///
///
/// reuse_at(target, axis) keeps a sliding window buffer of `target` across
/// iterations of `axis` and loads only the newly entering elements.
namespace {
struct LoopRoleInfo {
  DenseSet<Value> spatialIVs;
  DenseSet<Value> reductionIVs;
  DenseMap<Value, int64_t> reductionUpperBounds;
};

struct ReuseDimPlan {
  AffineMap anchorMap;
  int64_t layoutStride = 1;
  int64_t extent = 1;
  int64_t axisCoeff = 0;
  int64_t innerMinOffset = 0;
  int64_t innerMaxOffset = 0;
  bool isSliding = false;
};

struct ReuseStatePlan {
  SmallVector<ReuseDimPlan, 4> dims;
  SmallVector<Value, 4> prefixOperands;
  SmallVector<unsigned, 4> keptDims;
  SmallVector<int64_t, 4> shape;
  SmallVector<int, 4> resultToReusePos;
  unsigned slidingDim = 0;
  int64_t slidingDelta = 1;
  int64_t slidingStepAbs = 1;
  // Whether the per-access warm-up stores already cover every window slot.
  // When false, an explicit full-window warm-up fill is required so the
  // steady-state shift never reads a stale slot.
  bool accessesCoverWindow = true;
};

struct LoopNormalizationInfo {
  affine::AffineForOp loop;
  Value inductionVar;
  int64_t lowerBound = 0;
  int64_t upperBound = 0;
  int64_t step = 1;
  int64_t tripCount = 0;
};

enum class ReuseBufferStrategy {
  PhysicalShift,
  Ring,
};

struct ReuseResetBoundaryPlan {
  affine::AffineForOp rootLoop;
  affine::AffineForOp resetBoundaryLoop;
  bool canHoist = false;
};

struct ReuseAccessValidity {
  int64_t slidingLocalMin = 0;
  int64_t slidingLocalMax = 0;
  int64_t firstReusableIter = 1;
  int64_t lastReusableIter = 0;
};

struct ReuseValidityPlan {
  int64_t axisTripCount = 0;
  int64_t steadyStateStart = 1;
  int64_t updateStartIter = 1;
  int64_t updateEndIter = 0;
  SmallVector<int64_t, 4> slotFirstFillIters;
  SmallVector<int64_t, 4> slotLastUseIters;
  SmallVector<ReuseAccessValidity, 8> accesses;
};

struct ReuseExecutionPlan {
  explicit ReuseExecutionPlan(ReuseStatePlan statePlan,
                              ReuseValidityPlan validityPlan, bool enableRing)
      : statePlan(std::move(statePlan)), validityPlan(std::move(validityPlan)),
        slidingDelta(this->statePlan.slidingDelta),
        slidingStepAbs(this->statePlan.slidingStepAbs),
        steadyStateStart(this->validityPlan.steadyStateStart),
        updateStartIter(this->validityPlan.updateStartIter),
        updateEndIter(this->validityPlan.updateEndIter) {
    int slidingReusePos =
        this->statePlan.resultToReusePos[this->statePlan.slidingDim];
    int64_t slidingExtent =
        this->statePlan.dims[this->statePlan.slidingDim].extent;
    if (enableRing && slidingReusePos >= 0 && slidingExtent > 1 &&
        slidingStepAbs < slidingExtent) {
      strategy = ReuseBufferStrategy::Ring;
      ringIncrement =
          slidingDelta > 0 ? slidingStepAbs : slidingExtent - slidingStepAbs;
    }
  }

  ReuseStatePlan statePlan;
  ReuseValidityPlan validityPlan;
  ReuseBufferStrategy strategy = ReuseBufferStrategy::PhysicalShift;
  int64_t slidingDelta = 1;
  int64_t slidingStepAbs = 1;
  int64_t steadyStateStart = 1;
  int64_t updateStartIter = 1;
  int64_t updateEndIter = 0;
  int64_t ringIncrement = 0;
};

struct ReuseDimFootprint {
  AffineExpr lowerExpr;
  SmallVector<int64_t, 4> innerCoeffs;
  SmallVector<int64_t, 4> innerExtents;
  int64_t layoutStride = 1;
  int64_t extent = 1;
  int64_t axisCoeff = 0;
  int64_t innerMinOffset = 0;
  int64_t innerMaxOffset = 0;
};

// One reuse candidate: a direct affine.load of the target buffer under the
// selected axis.
struct ReuseLogicalAccess {
  Operation *anchorOp = nullptr; // the original affine.load
  Value exposedValue;            // its result value (uses are redirected)
  ComposedBufferAccess composed; // composed/simplified affine access
};

struct RingAccessCluster {
  SmallVector<unsigned, 4> accessIndices;
};

struct RingAccessPrecomputedIndices {
  SmallVector<Value, 4> logicalIndices;
  SmallVector<Value, 4> physicalIndices;
};

struct ReuseConditionalLoadResult {
  Value value;
  SmallVector<Value, 4> logicalIndices;
};

struct ReuseAccessFamilyAnalysis {
  SmallVector<ReuseLogicalAccess, 8> accesses;
  SmallVector<affine::AffineForOp, 8> innerLoops;
  ReuseExecutionPlan executionPlan;
  ReuseResetBoundaryPlan resetBoundaryPlan;

  ReuseAccessFamilyAnalysis(SmallVector<ReuseLogicalAccess, 8> accesses,
                            SmallVector<affine::AffineForOp, 8> innerLoops,
                            ReuseExecutionPlan executionPlan,
                            ReuseResetBoundaryPlan resetBoundaryPlan)
      : accesses(std::move(accesses)), innerLoops(std::move(innerLoops)),
        executionPlan(std::move(executionPlan)),
        resetBoundaryPlan(resetBoundaryPlan) {}
};
} // namespace

// Marks writes that the reuse_at pipeline itself emits into a reuse buffer
// (warmup fill, shift, refill). A chained reuse_at targeting that buffer skips
// them instead of rejecting them as user writes.
static constexpr StringLiteral kReuseMaintenanceAttr = "allo.reuse.maintenance";

static ReuseLogicalAccess makeReuseLogicalAccess(Operation *op) {
  auto readOp = cast<affine::AffineReadOpInterface>(op);
  return {op, readOp.getValue(), composeBufferAccess(op)};
}

static bool valueDependsOnTargetLoad(Value value, Value target,
                                     DenseMap<Value, bool> &cache,
                                     SmallPtrSetImpl<Value> &visiting) {
  // Trace through arithmetic and affine.if results to see whether a value
  // ultimately depends on an affine.load from the target buffer.
  if (auto it = cache.find(value); it != cache.end())
    return it->second;
  if (!visiting.insert(value).second)
    return false;

  bool depends = false;
  if (auto loadOp = value.getDefiningOp<affine::AffineLoadOp>()) {
    depends = loadOp.getMemRef() == target;
  } else if (auto result = dyn_cast<OpResult>(value)) {
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(result.getOwner())) {
      unsigned resultNumber = result.getResultNumber();
      auto yieldedDependsOnTarget = [&](Block *block) {
        if (!block)
          return false;
        auto yieldOp = dyn_cast<affine::AffineYieldOp>(block->getTerminator());
        if (!yieldOp || resultNumber >= yieldOp.getNumOperands())
          return false;
        return valueDependsOnTargetLoad(yieldOp.getOperand(resultNumber),
                                        target, cache, visiting);
      };
      depends = yieldedDependsOnTarget(ifOp.getThenBlock()) ||
                yieldedDependsOnTarget(ifOp.getElseBlock());
    } else if (auto forOp = dyn_cast<affine::AffineForOp>(result.getOwner())) {
      // The yielded value of a loop-carried result lives in the loop body and
      // is not reachable through the for-op's own operands, so trace it
      // explicitly; the operand walk still covers the inits and bounds.
      unsigned resultNumber = result.getResultNumber();
      auto yieldOp =
          dyn_cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());
      depends =
          (yieldOp && resultNumber < yieldOp.getNumOperands() &&
           valueDependsOnTargetLoad(yieldOp.getOperand(resultNumber), target,
                                    cache, visiting)) ||
          llvm::any_of(forOp.getOperands(), [&](Value operand) {
            return valueDependsOnTargetLoad(operand, target, cache, visiting);
          });
    } else {
      depends =
          llvm::any_of(result.getOwner()->getOperands(), [&](Value operand) {
            return valueDependsOnTargetLoad(operand, target, cache, visiting);
          });
    }
  } else if (Operation *defOp = value.getDefiningOp()) {
    depends = llvm::any_of(defOp->getOperands(), [&](Value operand) {
      return valueDependsOnTargetLoad(operand, target, cache, visiting);
    });
  }

  visiting.erase(value);
  cache[value] = depends;
  return depends;
}

static FailureOr<LoopNormalizationInfo>
analyzeLoopNormalization(affine::AffineForOp forOp) {
  // ReuseAt normalizes loops logically, but still requires a static affine
  // iteration space with constant bounds and a positive constant step.
  if (!forOp.hasConstantBounds() || forOp.getStepAsInt() <= 0)
    return failure();

  int64_t lb = forOp.getConstantLowerBound();
  int64_t ub = forOp.getConstantUpperBound();
  int64_t step = forOp.getStepAsInt();
  int64_t span = std::max<int64_t>(ub - lb, 0);
  int64_t tripCount = span == 0 ? 0 : (span + step - 1) / step;
  return LoopNormalizationInfo{forOp,    forOp.getInductionVar(), lb, ub, step,
                               tripCount};
}

static AffineExpr getNormalizedLoopReplacementExpr(
    MLIRContext *ctx, const LoopNormalizationInfo &info, unsigned dimPos) {
  return getAffineConstantExpr(info.lowerBound, ctx) +
         getAffineConstantExpr(info.step, ctx) * getAffineDimExpr(dimPos, ctx);
}

static Value materializeNormalizedLoopIndex(OpBuilder &builder, Location loc,
                                            const LoopNormalizationInfo &info,
                                            Value iv) {
  // Materialize the logical zero-based iteration count used by reuse_at
  // analysis without changing the payload loop bounds or step.
  if (info.lowerBound == 0 && info.step == 1)
    return iv;
  auto d0 = builder.getAffineDimExpr(0);
  auto normalizedMap = AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/0,
      (d0 - builder.getAffineConstantExpr(info.lowerBound))
          .floorDiv(info.step));
  return affine::makeComposedAffineApply(builder, loc, normalizedMap, {iv});
}

// Walk parent loops to the outermost loop of the selected axis loop.
static affine::AffineForOp getRootLoop(affine::AffineForOp loop) {
  affine::AffineForOp root = loop;
  while (auto parent = root->getParentOfType<affine::AffineForOp>())
    root = parent;
  return root;
}

// Classify each loop IV in the root nest: spatial IVs contribute to store
// indexing, reduction IVs to target-load indexing only. Also caches the
// reduction loops' upper bounds.
static LogicalResult
classifyLoopRoles(affine::AffineForOp rootForOp, Value target,
                  LoopRoleInfo &roles,
                  DenseMap<Value, LoopNormalizationInfo> &loopInfos,
                  SmallVectorImpl<affine::AffineForOp> &allLoops) {
  WalkResult walkResult = rootForOp.walk([&](affine::AffineForOp forOp) {
    auto infoOr = analyzeLoopNormalization(forOp);
    if (failed(infoOr))
      return WalkResult::interrupt();
    allLoops.push_back(forOp);
    loopInfos[forOp.getInductionVar()] = *infoOr;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  DenseSet<Value> spatialCandidates;
  DenseMap<Value, bool> loadDependenceCache;
  rootForOp.walk([&](affine::AffineStoreOp storeOp) {
    SmallPtrSet<Value, 16> visiting;
    if (!valueDependsOnTargetLoad(storeOp.getValueToStore(), target,
                                  loadDependenceCache, visiting)) {
      return WalkResult::advance();
    }
    AffineMap storeMap = storeOp.getAffineMap();
    auto storeOperands = storeOp.getMapOperands();
    for (AffineExpr resultExpr : storeMap.getResults()) {
      for (affine::AffineForOp loop : allLoops) {
        if (affineExprUsesValue(resultExpr, storeOperands,
                                storeMap.getNumDims(),
                                loop.getInductionVar())) {
          spatialCandidates.insert(loop.getInductionVar());
        }
      }
    }
    return WalkResult::advance();
  });

  DenseSet<Value> loadCandidates;
  rootForOp.walk([&](affine::AffineLoadOp loadOp) {
    if (loadOp.getMemRef() != target)
      return WalkResult::advance();
    AffineMap loadMap = loadOp.getAffineMap();
    auto loadOperands = loadOp.getMapOperands();
    for (AffineExpr resultExpr : loadMap.getResults()) {
      for (affine::AffineForOp loop : allLoops) {
        if (affineExprUsesValue(resultExpr, loadOperands, loadMap.getNumDims(),
                                loop.getInductionVar())) {
          loadCandidates.insert(loop.getInductionVar());
        }
      }
    }
    return WalkResult::advance();
  });

  if (loadCandidates.empty())
    return failure();

  roles.spatialIVs = std::move(spatialCandidates);
  for (Value iv : loadCandidates) {
    if (!roles.spatialIVs.contains(iv))
      roles.reductionIVs.insert(iv);
  }

  for (affine::AffineForOp loop : allLoops) {
    Value iv = loop.getInductionVar();
    if (!roles.reductionIVs.contains(iv))
      continue;
    roles.reductionUpperBounds[iv] = loopInfos.lookup(iv).upperBound;
  }
  return success();
}

// True if the loop IV is inferred as reduction-only in the current nest.
static bool isReductionLoop(affine::AffineForOp forOp,
                            const LoopRoleInfo &roles) {
  return roles.reductionIVs.contains(forOp.getInductionVar());
}

// True if the loop IV appears in spatial indexing (store side).
static bool isSpatialLoop(affine::AffineForOp forOp,
                          const LoopRoleInfo &roles) {
  return roles.spatialIVs.contains(forOp.getInductionVar());
}

static void
collectReuseInnerLoops(affine::AffineForOp axisLoop,
                       SmallVectorImpl<affine::AffineForOp> &innerLoops) {
  axisLoop.walk([&](affine::AffineForOp forOp) {
    if (forOp != axisLoop)
      innerLoops.push_back(forOp);
  });
}

static LogicalResult
collectReuseAccesses(affine::AffineForOp axisLoop, Value target,
                     SmallVectorImpl<ReuseLogicalAccess> &accesses,
                     SmallVectorImpl<affine::AffineForOp> &innerLoops) {
  // Every reuse access is a direct affine.load of `target`. A chained reuse_at
  // targets the previous stage's buffer, whose own maintenance writes are
  // skipped via `kReuseMaintenanceAttr`.
  collectReuseInnerLoops(axisLoop, innerLoops);

  Value targetRoot = resolveMemRefValueRoot(target);
  WalkResult walk =
      axisLoop.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        Value memref = nullptr;
        if (auto readOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
          memref = readOp.getMemRef();
          if (memref == target) {
            accesses.push_back(makeReuseLogicalAccess(op));
            return WalkResult::advance();
          }
        }

        if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
          memref = writeOp.getMemRef();
          if (resolveMemRefValueRoot(memref) == targetRoot) {
            if (op->hasAttr(kReuseMaintenanceAttr))
              return WalkResult::advance();
            auto diag =
                axisLoop.emitError()
                << "reuse_at requires the target buffer to be read-only "
                   "within the selected axis loop";
            diag.attachNote(writeOp->getLoc()) << "see write op here";
            return WalkResult::interrupt();
          }
        } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
          memref = loadOp.getMemRef();
          if (resolveMemRefValueRoot(memref) == targetRoot) {
            auto diag = emitError(targetRoot.getLoc())
                        << "reuse_at only supports affine.load accesses to the "
                           "target buffer";
            diag.attachNote(loadOp.getLoc()) << "see memref.load op here";
            return WalkResult::interrupt();
          }
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
          memref = storeOp.getMemRef();
          if (resolveMemRefValueRoot(memref) == targetRoot) {
            auto diag =
                axisLoop.emitError()
                << "reuse_at requires the target buffer to be read-only "
                   "within the selected axis loop";
            diag.attachNote(storeOp.getLoc()) << "see memref.store op here";
            return WalkResult::interrupt();
          }
        } else if (isMemRefCastOrViewLike(op)) {
          bool aliasesTarget =
              llvm::any_of(op->getResults(), [&](Value result) {
                return isa<BaseMemRefType>(result.getType()) &&
                       resolveMemRefValueRoot(result) == targetRoot;
              });
          if (aliasesTarget) {
            auto diag = axisLoop.emitError()
                        << "reuse_at does not support aliasing/view accesses "
                           "to the "
                           "target buffer within the selected axis loop";
            diag.attachNote(op->getLoc()) << "see aliasing/view op here";
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

  if (walk.wasInterrupted())
    return failure();
  if (accesses.empty()) {
    auto diag = axisLoop.emitError()
                << "no affine.load of the target buffer found within the "
                   "selected axis loop";
    return diag;
  }
  return success();
}

static FailureOr<ReuseDimFootprint>
computeRawReuseDimFootprint(const ComposedBufferAccess &access,
                            unsigned resultPos,
                            ArrayRef<affine::AffineForOp> innerLoops,
                            const LoopNormalizationInfo &axisInfo,
                            DenseMap<Value, LoopNormalizationInfo> &loopInfos,
                            DenseMap<Value, unsigned> &prefixOperandPos,
                            ArrayRef<Value> prefixOperands) {
  // Decompose one accessed buffer dimension into:
  //   lower(anchor over axis/prefix) + non-negative inner-loop coefficients.
  AffineExpr accessExpr = access.map.getResult(resultPos);
  SmallVector<affine::AffineForOp, 4> dependentLoops;
  for (affine::AffineForOp loop : innerLoops) {
    if (affineExprUsesValue(accessExpr, access.operands,
                            access.map.getNumDims(), loop.getInductionVar())) {
      dependentLoops.push_back(loop);
    }
  }

  unsigned prefixDimCount = prefixOperands.size();
  SmallVector<AffineExpr, 8> dimReplacements, symReplacements;
  SmallVector<std::pair<Value, AffineExpr>, 8> lowerReplacements;
  Value axisIV = axisInfo.inductionVar;
  lowerReplacements.emplace_back(axisIV, getNormalizedLoopReplacementExpr(
                                             access.map.getContext(), axisInfo,
                                             /*dimPos=*/0));
  for (affine::AffineForOp loop : innerLoops) {
    Value loopIV = loop.getInductionVar();
    LoopNormalizationInfo loopInfo = loopInfos.lookup(loop.getInductionVar());
    lowerReplacements.emplace_back(
        loopIV,
        getAffineConstantExpr(loopInfo.lowerBound, access.map.getContext()));
  }
  populateExprReplacements(
      access.map, access.operands, prefixOperandPos, lowerReplacements,
      /*prefixDimOffset=*/1, dimReplacements, symReplacements);
  AffineExpr lowerExpr = simplifyAffineExpr(
      accessExpr.replaceDimsAndSymbols(dimReplacements, symReplacements),
      1 + prefixDimCount, /*numSymbols=*/0);
  auto axisCoeffOr =
      getLinearAffineDimCoefficient(lowerExpr, 1 + prefixDimCount,
                                    /*dimPos=*/0);
  if (failed(axisCoeffOr))
    return failure();

  ReuseDimFootprint footprint;
  footprint.lowerExpr = lowerExpr;
  footprint.axisCoeff = *axisCoeffOr;

  if (dependentLoops.empty())
    return footprint;

  SmallVector<affine::AffineForOp, 4> activeLoops;
  SmallVector<int64_t, 4> activeLoopExtents;
  activeLoops.reserve(dependentLoops.size());
  activeLoopExtents.reserve(dependentLoops.size());
  for (affine::AffineForOp loop : dependentLoops) {
    LoopNormalizationInfo loopInfo = loopInfos.lookup(loop.getInductionVar());
    int64_t extent = loopInfo.tripCount;
    if (extent <= 0)
      return failure();
    activeLoops.push_back(loop);
    activeLoopExtents.push_back(extent);
  }

  SmallVector<std::pair<Value, AffineExpr>, 8> offsetReplacements;
  offsetReplacements.emplace_back(
      axisIV,
      getAffineConstantExpr(axisInfo.lowerBound, access.map.getContext()));
  for (affine::AffineForOp loop : innerLoops) {
    Value loopIV = loop.getInductionVar();
    LoopNormalizationInfo loopInfo = loopInfos.lookup(loop.getInductionVar());
    auto *activeIt = llvm::find(activeLoops, loop);
    if (activeIt == activeLoops.end()) {
      offsetReplacements.emplace_back(
          loopIV,
          getAffineConstantExpr(loopInfo.lowerBound, access.map.getContext()));
      continue;
    }
    unsigned idx = std::distance(activeLoops.begin(), activeIt);
    offsetReplacements.emplace_back(
        loopIV,
        getNormalizedLoopReplacementExpr(access.map.getContext(), loopInfo,
                                         /*dimPos=*/idx));
  }
  populateExprReplacements(
      access.map, access.operands, prefixOperandPos, offsetReplacements,
      /*prefixDimOffset=*/activeLoops.size(), dimReplacements, symReplacements);
  AffineExpr offsetExpr = simplifyAffineExpr(
      accessExpr.replaceDimsAndSymbols(dimReplacements, symReplacements),
      activeLoops.size() + prefixDimCount, /*numSymbols=*/0);

  SmallVector<AffineExpr, 4> expandedLowerDims;
  expandedLowerDims.reserve(1 + prefixDimCount);
  expandedLowerDims.push_back(
      getAffineConstantExpr(0, access.map.getContext()));
  for (unsigned i = 0; i < prefixDimCount; ++i)
    expandedLowerDims.push_back(
        getAffineDimExpr(activeLoops.size() + i, access.map.getContext()));
  AffineExpr expandedLowerExpr =
      simplifyAffineExpr(lowerExpr.replaceDims(expandedLowerDims),
                         activeLoops.size() + prefixDimCount, /*numSymbols=*/0);
  AffineExpr innerOffsetExpr = simplifyAffineExpr(
      offsetExpr - expandedLowerExpr, activeLoops.size() + prefixDimCount,
      /*numSymbols=*/0);

  auto innerConstOr = getConstantExprDelta(offsetExpr, expandedLowerExpr,
                                           activeLoops.size() + prefixDimCount);
  if (succeeded(innerConstOr) && *innerConstOr == 0)
    return footprint;

  FlatLinearConstraints localVarCst(activeLoops.size(), /*numSymbols=*/0);
  SmallVector<int64_t, 8> flattenedExpr;
  if (failed(getFlattenedAffineExpr(innerOffsetExpr, activeLoops.size(),
                                    /*numSymbols=*/0, &flattenedExpr,
                                    &localVarCst)))
    return failure();
  if (localVarCst.getNumLocalVars() != 0 ||
      flattenedExpr.size() != activeLoops.size() + 1)
    return failure();
  if (flattenedExpr.back() != 0)
    return failure();

  footprint.innerCoeffs.assign(flattenedExpr.begin(), flattenedExpr.end() - 1);
  footprint.innerExtents.assign(activeLoopExtents.begin(),
                                activeLoopExtents.end());
  for (int64_t coeff : footprint.innerCoeffs) {
    if (coeff < 0)
      return failure();
  }
  return footprint;
}

static void updateLayoutStrideGCD(int64_t &layoutStride, int64_t value) {
  if (value == 0)
    return;
  int64_t absValue = std::abs(value);
  layoutStride =
      layoutStride == 0 ? absValue : std::gcd(layoutStride, absValue);
}

static FailureOr<int64_t> computeDenseSlotMaxOffset(ArrayRef<int64_t> coeffs,
                                                    ArrayRef<int64_t> extents) {
  SmallVector<std::pair<int64_t, int64_t>, 4> activeTerms;
  for (auto [coeff, extent] : llvm::zip_equal(coeffs, extents)) {
    if (coeff < 0)
      return failure();
    if (coeff == 0)
      continue;
    activeTerms.emplace_back(coeff, extent);
  }
  llvm::sort(activeTerms, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  int64_t coveredMax = 0;
  for (auto [coeff, extent] : activeTerms) {
    if (coeff > coveredMax + 1)
      return failure();
    coveredMax += coeff * (extent - 1);
  }
  return coveredMax;
}

static FailureOr<ReuseDimFootprint>
projectReuseDimFootprintToSlots(const ReuseDimFootprint &rawFootprint,
                                int64_t layoutStride) {
  if (layoutStride <= 0)
    return failure();
  if (rawFootprint.axisCoeff % layoutStride != 0)
    return failure();

  ReuseDimFootprint slotFootprint = rawFootprint;
  slotFootprint.layoutStride = layoutStride;
  slotFootprint.axisCoeff = rawFootprint.axisCoeff / layoutStride;

  SmallVector<int64_t, 4> slotCoeffs;
  slotCoeffs.reserve(rawFootprint.innerCoeffs.size());
  for (auto [coeff, extent] :
       llvm::zip_equal(rawFootprint.innerCoeffs, rawFootprint.innerExtents)) {
    if (coeff % layoutStride != 0)
      return failure();
    slotCoeffs.push_back(coeff / layoutStride);
  }
  auto maxSlotOffsetOr =
      computeDenseSlotMaxOffset(slotCoeffs, rawFootprint.innerExtents);
  if (failed(maxSlotOffsetOr))
    return failure();

  slotFootprint.extent = *maxSlotOffsetOr + 1;
  slotFootprint.innerMinOffset = 0;
  slotFootprint.innerMaxOffset = *maxSlotOffsetOr;
  return slotFootprint;
}

static FailureOr<ReuseStatePlan>
analyzeReuseStatePlan(ArrayRef<ReuseLogicalAccess> accesses,
                      ArrayRef<affine::AffineForOp> innerLoops,
                      affine::AffineForOp axisLoop,
                      const LoopNormalizationInfo &axisInfo,
                      DenseMap<Value, LoopNormalizationInfo> &loopInfos,
                      unsigned bufferRank, MLIRContext *ctx) {
  // Build one logical reuse state shared by all candidate loads:
  // common sliding dim, common anchors, and one bounded local box.
  ReuseStatePlan plan;
  DenseMap<Value, unsigned> prefixOperandPos;
  SmallVector<ComposedBufferAccess, 8> composedAccesses;
  composedAccesses.reserve(accesses.size());
  for (const ReuseLogicalAccess &access : accesses)
    composedAccesses.push_back(access.composed);
  collectFootprintOperands(composedAccesses, innerLoops,
                           ArrayRef<Value>{axisLoop.getInductionVar()},
                           prefixOperandPos, plan.prefixOperands);
  unsigned prefixDimCount = plan.prefixOperands.size();

  SmallVector<bool, 4> footprintSeen(bufferRank, false);
  SmallVector<AffineExpr, 4> refLower(bufferRank);
  SmallVector<int64_t, 4> layoutStrides(bufferRank, 1);
  SmallVector<int64_t, 4> minOffset(bufferRank, 0);
  SmallVector<int64_t, 4> maxUpper(bufferRank, 0);
  SmallVector<int64_t, 4> windowMinOffset(bufferRank, 0);
  SmallVector<int64_t, 4> windowMaxOffset(bufferRank, 0);
  SmallVector<SmallVector<int64_t, 4>, 8> lowerOffsets(accesses.size());
  SmallVector<SmallVector<ReuseDimFootprint, 4>, 8> rawFootprints(
      accesses.size());
  SmallVector<SmallVector<int64_t, 4>, 8> axisCoeffs(accesses.size());
  // Per-access absolute slot range [lo, hi] in each buffer dim, used to test
  // whether the accesses already tile the window box.
  SmallVector<SmallVector<int64_t, 4>, 8> accessSlotLo(accesses.size());
  SmallVector<SmallVector<int64_t, 4>, 8> accessSlotHi(accesses.size());

  for (auto [accessIdx, access] : llvm::enumerate(accesses)) {
    if (access.composed.map.getNumResults() != bufferRank) {
      auto diag = access.anchorOp->emitError()
                  << "reuse_at requires candidate loads to match the target "
                     "buffer rank";
      diag.attachNote(access.anchorOp->getLoc())
          << "candidate access has " << access.composed.map.getNumResults()
          << " indices, but the target buffer rank is " << bufferRank;
      return diag;
    }

    lowerOffsets[accessIdx].assign(bufferRank, 0);
    rawFootprints[accessIdx].resize(bufferRank);
    for (unsigned d = 0; d < bufferRank; ++d) {
      // Derive a source-space footprint first; the common lattice stride is a
      // property of the whole access family, not of one access in isolation.
      auto footprintOr = computeRawReuseDimFootprint(
          access.composed, d, innerLoops, axisInfo, loopInfos, prefixOperandPos,
          plan.prefixOperands);
      if (failed(footprintOr)) {
        auto diag = access.anchorOp->emitError()
                    << "reuse_at requires buffer dimensions to have bounded "
                       "strided affine-lattice footprints";
        diag.attachNote(access.anchorOp->getLoc())
            << "failed to derive a bounded lattice footprint for buffer "
               "dimension "
            << d;
        return diag;
      }
      rawFootprints[accessIdx][d] = *footprintOr;

      if (!footprintSeen[d]) {
        // Seed the common coordinate system from the first candidate access.
        footprintSeen[d] = true;
        refLower[d] = footprintOr->lowerExpr;
        continue;
      }

      auto offsetOr = getConstantExprDelta(footprintOr->lowerExpr, refLower[d],
                                           1 + prefixDimCount);
      if (failed(offsetOr)) {
        auto diag =
            access.anchorOp->emitError()
            << "candidate loads do not share a common lattice coordinate "
               "system";
        diag.attachNote(access.anchorOp->getLoc())
            << "candidate access uses a non-constant local offset for "
               "buffer dimension "
            << d;
        return diag;
      }
      lowerOffsets[accessIdx][d] = *offsetOr;
    }
  }

  for (unsigned d = 0; d < bufferRank; ++d) {
    int64_t layoutStride = 0;
    for (auto [accessIdx, access] : llvm::enumerate(accesses)) {
      const ReuseDimFootprint &footprint = rawFootprints[accessIdx][d];
      updateLayoutStrideGCD(layoutStride, footprint.axisCoeff);
      updateLayoutStrideGCD(layoutStride, lowerOffsets[accessIdx][d]);
      for (int64_t coeff : footprint.innerCoeffs)
        updateLayoutStrideGCD(layoutStride, coeff);
    }
    layoutStrides[d] = layoutStride == 0 ? 1 : layoutStride;
  }

  std::optional<unsigned> detectedSlidingDim;
  std::optional<int64_t> detectedSlidingDelta;
  for (auto [accessIdx, access] : llvm::enumerate(accesses)) {
    axisCoeffs[accessIdx].resize(bufferRank, 0);
    accessSlotLo[accessIdx].resize(bufferRank, 0);
    accessSlotHi[accessIdx].resize(bufferRank, 0);
    for (unsigned d = 0; d < bufferRank; ++d) {
      auto slotFootprintOr = projectReuseDimFootprintToSlots(
          rawFootprints[accessIdx][d], layoutStrides[d]);
      if (failed(slotFootprintOr) ||
          lowerOffsets[accessIdx][d] % layoutStrides[d] != 0) {
        auto diag = access.anchorOp->emitError()
                    << "reuse_at requires buffer dimensions to have bounded "
                       "strided affine-lattice footprints";
        diag.attachNote(access.anchorOp->getLoc())
            << "failed to project buffer dimension " << d
            << " to a dense slot-space lattice";
        return diag;
      }
      const ReuseDimFootprint &footprint = *slotFootprintOr;
      int64_t slotOffset = lowerOffsets[accessIdx][d] / layoutStrides[d];
      accessSlotLo[accessIdx][d] = slotOffset + footprint.innerMinOffset;
      accessSlotHi[accessIdx][d] = slotOffset + footprint.innerMaxOffset;
      int64_t delta = footprint.axisCoeff;
      axisCoeffs[accessIdx][d] = delta;

      if (delta != 0) {
        if (detectedSlidingDim && *detectedSlidingDim != d) {
          auto diag = access.anchorOp->emitError()
                      << "candidate loads do not share a common sliding "
                         "dimension";
          diag.attachNote(access.anchorOp->getLoc())
              << "candidate access slides along buffer dimension " << d
              << ", but previous candidates slide along buffer dimension "
              << *detectedSlidingDim;
          return diag;
        }
        if (detectedSlidingDelta && *detectedSlidingDelta != delta) {
          auto diag = access.anchorOp->emitError()
                      << "candidate loads do not share a common sliding "
                         "direction";
          diag.attachNote(access.anchorOp->getLoc())
              << "candidate access uses selected-axis slot coefficient "
              << delta << " in buffer dimension " << d
              << ", but previous candidates use coefficient "
              << *detectedSlidingDelta;
          return diag;
        }
        detectedSlidingDim = d;
        detectedSlidingDelta = delta;
      }

      if (accessIdx == 0) {
        minOffset[d] = slotOffset;
        maxUpper[d] = slotOffset + footprint.extent;
        windowMinOffset[d] = slotOffset + footprint.innerMinOffset;
        windowMaxOffset[d] = slotOffset + footprint.innerMaxOffset;
        continue;
      }
      minOffset[d] = std::min(minOffset[d], slotOffset);
      maxUpper[d] = std::max(maxUpper[d], slotOffset + footprint.extent);
      windowMinOffset[d] =
          std::min(windowMinOffset[d], slotOffset + footprint.innerMinOffset);
      windowMaxOffset[d] =
          std::max(windowMaxOffset[d], slotOffset + footprint.innerMaxOffset);
    }
  }

  if (!detectedSlidingDim || !detectedSlidingDelta) {
    auto diag = axisLoop->emitError()
                << "cannot find a reusable sliding dimension for the selected "
                   "axis";
    diag.attachNote(axisLoop.getLoc())
        << "none of the candidate affine.load accesses varies with the "
           "selected axis";
    return diag;
  }

  for (auto [accessIdx, coeffs] : llvm::enumerate(axisCoeffs)) {
    // Every candidate must slide along exactly one common buffer dimension.
    if (coeffs[*detectedSlidingDim] == 0) {
      auto diag = accesses[accessIdx].anchorOp->emitError()
                  << "candidate loads do not all depend on the selected axis "
                     "through the same sliding dimension";
      diag.attachNote(accesses[accessIdx].anchorOp->getLoc())
          << "candidate access does not depend on the selected axis "
             "through buffer dimension "
          << *detectedSlidingDim;
      return diag;
    }
    for (auto [dim, coeff] : llvm::enumerate(coeffs)) {
      if (dim != *detectedSlidingDim && coeff != 0) {
        auto diag = accesses[accessIdx].anchorOp->emitError()
                    << "candidate loads do not share a common sliding "
                       "dimension";
        diag.attachNote(accesses[accessIdx].anchorOp->getLoc())
            << "candidate access also depends on the selected axis "
               "through buffer dimension "
            << dim;
        return diag;
      }
    }
  }

  plan.dims.resize(bufferRank);
  plan.resultToReusePos.assign(bufferRank, -1);
  plan.slidingDim = *detectedSlidingDim;
  plan.slidingDelta = *detectedSlidingDelta;
  plan.slidingStepAbs = std::abs(*detectedSlidingDelta);

  for (unsigned d = 0; d < bufferRank; ++d) {
    ReuseDimPlan &dimPlan = plan.dims[d];
    if (!footprintSeen[d]) {
      auto diag = axisLoop->emitError()
                  << "reuse_at failed to derive a local footprint for a "
                     "buffer dimension";
      diag.attachNote(axisLoop.getLoc())
          << "failed while materializing local state for buffer dimension "
          << d;
      return diag;
    }

    // Convert the relative offsets gathered above into a single local box.
    AffineExpr anchorExpr = simplifyAffineExpr(
        refLower[d] +
            getAffineConstantExpr(minOffset[d] * layoutStrides[d], ctx),
        1 + prefixDimCount, /*numSymbols=*/0);
    dimPlan.anchorMap =
        AffineMap::get(1 + prefixDimCount, /*symbolCount=*/0, anchorExpr, ctx);
    dimPlan.layoutStride = layoutStrides[d];
    dimPlan.extent = maxUpper[d] - minOffset[d];
    dimPlan.axisCoeff = d == *detectedSlidingDim ? *detectedSlidingDelta : 0;
    dimPlan.innerMinOffset = windowMinOffset[d] - minOffset[d];
    dimPlan.innerMaxOffset = windowMaxOffset[d] - minOffset[d];
    dimPlan.isSliding = d == *detectedSlidingDim;

    if (dimPlan.extent <= 0) {
      auto diag = axisLoop->emitError()
                  << "reuse_at derived a non-positive local state extent";
      diag.attachNote(axisLoop.getLoc()) << "derived extent " << dimPlan.extent
                                         << " for buffer dimension " << d;
      return diag;
    }

    if (dimPlan.isSliding || dimPlan.extent > 1) {
      plan.resultToReusePos[d] = static_cast<int>(plan.keptDims.size());
      plan.keptDims.push_back(d);
      plan.shape.push_back(dimPlan.extent);
    }
  }

  // Mark every window slot covered by some access: when all are covered the
  // explicit full-window warm-up fill is redundant.
  {
    unsigned nKept = plan.keptDims.size();
    int64_t totalSlots = 1;
    for (int64_t e : plan.shape)
      totalSlots *= e;
    SmallVector<bool, 16> covered(totalSlots, false);
    for (unsigned accessIdx = 0; accessIdx < accesses.size(); ++accessIdx) {
      SmallVector<int64_t, 4> lo(nKept), hi(nKept);
      bool inRange = true;
      for (unsigned k = 0; k < nKept && inRange; ++k) {
        unsigned d = plan.keptDims[k];
        lo[k] = accessSlotLo[accessIdx][d] - minOffset[d];
        hi[k] = accessSlotHi[accessIdx][d] - minOffset[d];
        inRange = lo[k] >= 0 && hi[k] < plan.shape[k] && lo[k] <= hi[k];
      }
      if (!inRange)
        continue;
      // Odometer over the access's sub-box, marking each covered slot.
      SmallVector<int64_t, 4> coord(lo);
      while (true) {
        int64_t linear = 0;
        for (unsigned k = 0; k < nKept; ++k)
          linear = linear * plan.shape[k] + coord[k];
        covered[linear] = true;
        int k = static_cast<int>(nKept) - 1;
        for (; k >= 0; --k) {
          if (++coord[k] <= hi[k])
            break;
          coord[k] = lo[k];
        }
        if (k < 0)
          break;
      }
    }
    plan.accessesCoverWindow = llvm::all_of(covered, [](bool c) { return c; });
  }

  int64_t slidingExtent = plan.dims[plan.slidingDim].extent;
  // Reuse only helps when consecutive axis iterations still overlap.
  if (plan.slidingStepAbs >= slidingExtent) {
    auto diag = axisLoop->emitError()
                << "reuse_at requires cross-iteration overlap on the sliding "
                   "dimension";
    diag.attachNote(axisLoop.getLoc())
        << "sliding step " << plan.slidingStepAbs
        << " does not leave reusable overlap within extent " << slidingExtent;
    return diag;
  }

  return plan;
}

static FailureOr<ReuseValidityPlan>
analyzeReuseValidityPlan(ArrayRef<ReuseLogicalAccess> accesses,
                         ArrayRef<affine::AffineForOp> innerLoops,
                         const LoopNormalizationInfo &axisInfo,
                         DenseMap<Value, LoopNormalizationInfo> &loopInfos,
                         const ReuseStatePlan &plan) {
  // Direct-target reuse starts with one explicit source-backed iteration, then
  // records the per-access local coverage over the sliding dimension.
  ReuseValidityPlan validityPlan;
  validityPlan.axisTripCount = axisInfo.tripCount;
  validityPlan.steadyStateStart =
      axisInfo.tripCount > 1 ? 1 : axisInfo.tripCount;
  validityPlan.updateStartIter = validityPlan.steadyStateStart;

  DenseMap<Value, unsigned> prefixOperandPos;
  for (auto [idx, operand] : llvm::enumerate(plan.prefixOperands))
    prefixOperandPos[stripCast(operand)] = idx;
  unsigned prefixDimCount = plan.prefixOperands.size();
  AffineExpr anchorExpr = plan.dims[plan.slidingDim].anchorMap.getResult(0);
  int64_t layoutStride = plan.dims[plan.slidingDim].layoutStride;
  int64_t slidingExtent = plan.dims[plan.slidingDim].extent;
  validityPlan.slotFirstFillIters.assign(slidingExtent,
                                         std::numeric_limits<int64_t>::max());
  validityPlan.slotLastUseIters.assign(slidingExtent, -1);

  validityPlan.accesses.reserve(accesses.size());
  for (const ReuseLogicalAccess &access : accesses) {
    auto footprintOr = computeRawReuseDimFootprint(
        access.composed, plan.slidingDim, innerLoops, axisInfo, loopInfos,
        prefixOperandPos, plan.prefixOperands);
    if (failed(footprintOr)) {
      auto diag = access.anchorOp->emitError()
                  << "reuse_at failed to derive sliding-dimension validity";
      diag.attachNote(access.anchorOp->getLoc())
          << "failed while computing local validity coverage";
      return diag;
    }

    auto slotFootprintOr =
        projectReuseDimFootprintToSlots(*footprintOr, layoutStride);
    auto localBaseOr = getConstantExprDelta(footprintOr->lowerExpr, anchorExpr,
                                            1 + prefixDimCount);
    if (failed(slotFootprintOr) || failed(localBaseOr) ||
        *localBaseOr % layoutStride != 0) {
      auto diag = access.anchorOp->emitError()
                  << "reuse_at requires statically-bounded validity on the "
                     "sliding dimension";
      diag.attachNote(access.anchorOp->getLoc())
          << "failed to express the access as a slot-space local offset";
      return diag;
    }

    ReuseAccessValidity accessValidity;
    int64_t slotBase = *localBaseOr / layoutStride;
    accessValidity.slidingLocalMin = slotBase + slotFootprintOr->innerMinOffset;
    accessValidity.slidingLocalMax = slotBase + slotFootprintOr->innerMaxOffset;
    accessValidity.firstReusableIter = validityPlan.steadyStateStart;
    accessValidity.lastReusableIter =
        validityPlan.axisTripCount > 0 ? validityPlan.axisTripCount - 1 : 0;

    if (accessValidity.slidingLocalMin < 0 ||
        accessValidity.slidingLocalMax >= slidingExtent) {
      auto diag = access.anchorOp->emitError()
                  << "reuse_at derived an out-of-bounds local validity range";
      diag.attachNote(access.anchorOp->getLoc())
          << "derived local sliding range [" << accessValidity.slidingLocalMin
          << ", " << accessValidity.slidingLocalMax << "] exceeds extent "
          << slidingExtent;
      return diag;
    }

    int64_t firstFillIter = accessValidity.firstReusableIter > 0
                                ? 0
                                : std::numeric_limits<int64_t>::max();
    for (int64_t slot = accessValidity.slidingLocalMin;
         slot <= accessValidity.slidingLocalMax; ++slot) {
      validityPlan.slotFirstFillIters[slot] =
          std::min(validityPlan.slotFirstFillIters[slot], firstFillIter);
      validityPlan.slotLastUseIters[slot] = std::max(
          validityPlan.slotLastUseIters[slot], accessValidity.lastReusableIter);
    }
    validityPlan.accesses.push_back(accessValidity);
  }

  validityPlan.updateStartIter =
      validityPlan.accesses.empty()
          ? validityPlan.steadyStateStart
          : llvm::min_element(validityPlan.accesses,
                              [](const ReuseAccessValidity &lhs,
                                 const ReuseAccessValidity &rhs) {
                                return lhs.firstReusableIter <
                                       rhs.firstReusableIter;
                              })
                ->firstReusableIter;
  // State maintenance (shift+refill) runs before the read in the same
  // iteration, so it must stay active through the last iteration that reads a
  // slot.
  int64_t maxLastUseIter = *llvm::max_element(validityPlan.slotLastUseIters);
  validityPlan.updateEndIter = maxLastUseIter;

  return validityPlan;
}

static SmallVector<Value, 4>
getReuseStateOperands(Value axisIV, ArrayRef<Value> prefixOperands) {
  SmallVector<Value, 4> operands;
  operands.push_back(axisIV);
  operands.append(prefixOperands.begin(), prefixOperands.end());
  return operands;
}

static ReuseResetBoundaryPlan
analyzeReuseResetBoundary(affine::AffineForOp axisLoop,
                          affine::AffineForOp rootLoop,
                          const ReuseExecutionPlan &executionPlan) {
  // Hoisting is only a placement optimization. Keep it tied to the current
  // validity model so failing the proof never rejects reuse_at.
  ReuseResetBoundaryPlan plan;
  plan.rootLoop = rootLoop;
  plan.resetBoundaryLoop = axisLoop->getParentOfType<affine::AffineForOp>();

  int64_t slidingExtent =
      executionPlan.statePlan.dims[executionPlan.statePlan.slidingDim].extent;
  SmallVector<int64_t, 4> slotFirstReusableReadIters(
      slidingExtent, std::numeric_limits<int64_t>::max());
  for (const ReuseAccessValidity &accessValidity :
       executionPlan.validityPlan.accesses) {
    for (int64_t slot = accessValidity.slidingLocalMin;
         slot <= accessValidity.slidingLocalMax; ++slot) {
      slotFirstReusableReadIters[slot] = std::min(
          slotFirstReusableReadIters[slot], accessValidity.firstReusableIter);
    }
  }

  for (auto [firstFillIter, firstReadIter] :
       llvm::zip_equal(executionPlan.validityPlan.slotFirstFillIters,
                       slotFirstReusableReadIters)) {
    if (firstReadIter == std::numeric_limits<int64_t>::max())
      continue;
    if (firstFillIter == std::numeric_limits<int64_t>::max() ||
        firstFillIter >= firstReadIter)
      return plan;
  }

  plan.canHoist = true;
  return plan;
}

static void markReuseMaintenanceWrites(Operation *scope, Value reuseBuffer) {
  auto unit = UnitAttr::get(reuseBuffer.getContext());
  scope->walk([&](Operation *nestedOp) {
    if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(nestedOp)) {
      if (writeOp.getMemRef() == reuseBuffer)
        nestedOp->setAttr(kReuseMaintenanceAttr, unit);
      return;
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(nestedOp)) {
      if (storeOp.getMemRef() == reuseBuffer)
        nestedOp->setAttr(kReuseMaintenanceAttr, unit);
    }
  });
}

static Value materializeGlobalAccessIndex(OpBuilder &builder, Location loc,
                                          const ComposedBufferAccess &access,
                                          unsigned resultDim);

static affine::AffineForOp createConstantAffineFor(OpBuilder &builder,
                                                   Location loc, int64_t lb,
                                                   int64_t ub,
                                                   ValueRange iterArgs = {}) {
  // Build a constant-bounds affine.for, optionally carrying iter_args when the
  // generated loop must thread ring state such as a rolling physical slot.
  if (iterArgs.empty()) {
    return affine::AffineForOp::create(builder, loc, lb, ub, 1);
  }
  return affine::AffineForOp::create(
      builder, loc, lb, ub, 1, iterArgs,
      [](OpBuilder &builder, Location loc, Value, ValueRange iterArgs) {
        affine::AffineYieldOp::create(builder, loc, iterArgs);
      });
}

static Value buildAnchorValue(OpBuilder &builder, Location loc,
                              const ReuseDimPlan &dimPlan,
                              ValueRange stateOperands) {
  SmallVector<OpFoldResult, 4> ofrs;
  ofrs.reserve(stateOperands.size());
  for (Value operand : stateOperands)
    ofrs.push_back(operand);
  return affine::makeComposedAffineApply(builder, loc, dimPlan.anchorMap, ofrs);
}

static Value buildOffsetValue(OpBuilder &builder, Location loc, Value base,
                              Value offset) {
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  return affine::makeComposedAffineApply(
      builder, loc, AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, d0 + d1),
      {base, offset});
}

static Value buildOffsetValue(OpBuilder &builder, Location loc, Value base,
                              int64_t offset) {
  AffineExpr d0 = builder.getAffineDimExpr(0);
  return affine::makeComposedAffineApply(
      builder, loc,
      AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                     d0 + builder.getAffineConstantExpr(offset)),
      {base});
}

static Value buildStridedOffsetValue(OpBuilder &builder, Location loc,
                                     Value base, Value offset, int64_t stride) {
  if (stride == 1)
    return buildOffsetValue(builder, loc, base, offset);
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  return affine::makeComposedAffineApply(
      builder, loc,
      AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                     d0 + d1 * builder.getAffineConstantExpr(stride)),
      {base, offset});
}

static Value buildStridedDifferenceValue(OpBuilder &builder, Location loc,
                                         Value lhs, Value rhs, int64_t stride) {
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  AffineExpr difference = d0 - d1;
  if (stride != 1)
    difference = difference.floorDiv(stride);
  return affine::makeComposedAffineApply(
      builder, loc,
      AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, difference),
      {lhs, rhs});
}

static Value buildEnteringFaceLogicalIndex(OpBuilder &builder, Location loc,
                                           Value enteringFaceIV,
                                           const ReuseExecutionPlan &plan,
                                           int64_t enteringBaseOffset) {
  return plan.slidingDelta > 0 ? buildOffsetValue(builder, loc, enteringFaceIV,
                                                  enteringBaseOffset)
                               : enteringFaceIV;
}

static void generateReuseStateShift(OpBuilder &builder, Location loc,
                                    Value reuseBuffer,
                                    const ReuseExecutionPlan &executionPlan) {
  const ReuseStatePlan &plan = executionPlan.statePlan;
  int slidingReusePos = plan.resultToReusePos[plan.slidingDim];
  int64_t slidingExtent = plan.dims[plan.slidingDim].extent;
  if (slidingReusePos < 0 || slidingExtent <= 1)
    return;

  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 4> loopIvs(plan.keptDims.size());
  for (auto [reusePos, dim] : llvm::enumerate(plan.keptDims)) {
    int64_t lb = 0;
    int64_t ub = plan.shape[reusePos];
    if (static_cast<int>(reusePos) == slidingReusePos) {
      if (executionPlan.slidingDelta > 0) {
        ub = slidingExtent - executionPlan.slidingStepAbs;
      } else {
        lb = executionPlan.slidingStepAbs;
      }
    }
    affine::AffineForOp forOp =
        createConstantAffineFor(builder, loc, /*lb=*/lb, /*ub=*/ub);
    builder.setInsertionPoint(
        forOp.getBody(), Block::iterator(forOp.getBody()->getTerminator()));
    loopIvs[reusePos] = forOp.getInductionVar();
  }

  SmallVector<Value, 4> srcIndices = loopIvs;
  SmallVector<Value, 4> dstIndices = loopIvs;
  srcIndices[slidingReusePos] = buildOffsetValue(
      builder, loc, loopIvs[slidingReusePos], executionPlan.slidingDelta);

  Value shifted =
      affine::AffineLoadOp::create(builder, loc, reuseBuffer, srcIndices);
  affine::AffineStoreOp::create(builder, loc, shifted, reuseBuffer, dstIndices);
}

static Value buildModuloOffsetValue(OpBuilder &builder, Location loc, Value lhs,
                                    Value rhs, int64_t modulus) {
  Value sum = arith::AddIOp::create(builder, loc, lhs, rhs);
  Value modulusValue = arith::ConstantIndexOp::create(builder, loc, modulus);
  return arith::RemUIOp::create(builder, loc, sum, modulusValue);
}

static Value buildWrappedIncrementValue(OpBuilder &builder, Location loc,
                                        Value current, int64_t modulus) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value last = arith::ConstantIndexOp::create(builder, loc, modulus - 1);
  Value atLast = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       current, last);
  Value next = arith::AddIOp::create(builder, loc, current, one);
  return arith::SelectOp::create(builder, loc, atLast, zero, next);
}

static IntegerSet buildReuseWarmupMissSet(OpBuilder &builder,
                                          int64_t firstReusableIter) {
  // Iterations before `firstReusableIter` miss the window and read the source.
  auto d0 = builder.getAffineDimExpr(0);
  return IntegerSet::get(/*dimCount=*/1, /*symbolCount=*/0,
                         {-d0 + (firstReusableIter - 1)},
                         /*eqFlags=*/{false});
}

static IntegerSet buildReuseReusableHitSet(OpBuilder &builder, int64_t lower,
                                           int64_t upper) {
  auto d0 = builder.getAffineDimExpr(0);
  return IntegerSet::get(/*dimCount=*/1, /*symbolCount=*/0,
                         {d0 - lower, -d0 + upper},
                         /*eqFlags=*/{false, false});
}

static std::optional<IntegerSet>
buildReuseUpdateActiveSet(OpBuilder &builder, int64_t lower, int64_t upper) {
  if (upper < lower)
    return std::nullopt;
  auto d0 = builder.getAffineDimExpr(0);
  return IntegerSet::get(/*dimCount=*/1, /*symbolCount=*/0,
                         {d0 - lower, -d0 + upper},
                         /*eqFlags=*/{false, false});
}

static ReuseConditionalLoadResult createConditionalReuseLoad(
    OpBuilder &builder, Location loc, const ReuseLogicalAccess &access,
    Value sourceBuffer, Value reuseBuffer,
    const ReuseExecutionPlan &executionPlan, ValueRange stateOperands,
    const ReuseAccessValidity &accessValidity, Value logicalAxisIV,
    Value currentIterationRingHead,
    ArrayRef<Value> precomputedLogicalIndices = {},
    ArrayRef<Value> precomputedPhysicalIndices = {});

static SmallVector<Value, 4> materializeLogicalReuseIndices(
    OpBuilder &builder, Location loc, const ReuseStatePlan &plan,
    const ComposedBufferAccess &access, ValueRange stateOperands) {
  // Rewrite one access into reuse-buffer coordinates by subtracting the
  // analyzed anchor of each kept dimension from the original global index.
  SmallVector<Value, 4> logicalIndices;
  logicalIndices.reserve(plan.keptDims.size());
  for (unsigned resultDim : plan.keptDims) {
    Value globalIndex =
        materializeGlobalAccessIndex(builder, loc, access, resultDim);
    Value anchor =
        buildAnchorValue(builder, loc, plan.dims[resultDim], stateOperands);
    logicalIndices.push_back(buildStridedDifferenceValue(
        builder, loc, globalIndex, anchor, plan.dims[resultDim].layoutStride));
  }
  return logicalIndices;
}

static SmallVector<Value, 4>
materializePhysicalReuseIndices(OpBuilder &builder, Location loc,
                                const ReuseExecutionPlan &executionPlan,
                                ArrayRef<Value> logicalIndices, Value ringHead,
                                Value physicalSlidingIndex = nullptr) {
  // Ring mode keeps logical indices stable and only remaps the sliding
  // dimension to the current physical head position.
  const ReuseStatePlan &plan = executionPlan.statePlan;
  SmallVector<Value, 4> physicalIndices(logicalIndices.begin(),
                                        logicalIndices.end());
  if (executionPlan.strategy != ReuseBufferStrategy::Ring)
    return physicalIndices;

  int slidingReusePos = plan.resultToReusePos[plan.slidingDim];
  assert(slidingReusePos >= 0 && "expected sliding dimension to be kept");
  if (physicalSlidingIndex) {
    physicalIndices[slidingReusePos] = physicalSlidingIndex;
    return physicalIndices;
  }
  assert(ringHead && "expected ring-head value for ring strategy");
  physicalIndices[slidingReusePos] = buildModuloOffsetValue(
      builder, loc, ringHead, logicalIndices[slidingReusePos],
      plan.dims[plan.slidingDim].extent);
  return physicalIndices;
}

static SmallVector<RingAccessCluster, 4>
collectRingAccessClusters(ArrayRef<ReuseLogicalAccess> accesses) {
  SmallVector<RingAccessCluster, 4> clusters;
  if (accesses.empty())
    return clusters;

  RingAccessCluster currentCluster;
  currentCluster.accessIndices.push_back(0);
  for (unsigned idx = 1, e = accesses.size(); idx < e; ++idx) {
    Operation *prev = accesses[idx - 1].anchorOp;
    Operation *current = accesses[idx].anchorOp;
    if (prev->getBlock() == current->getBlock() &&
        prev->getNextNode() == current) {
      currentCluster.accessIndices.push_back(idx);
      continue;
    }
    clusters.push_back(std::move(currentCluster));
    currentCluster = RingAccessCluster();
    currentCluster.accessIndices.push_back(idx);
  }
  clusters.push_back(std::move(currentCluster));
  return clusters;
}

static DenseMap<Operation *, RingAccessPrecomputedIndices>
precomputeRingAccessClusterIndices(OpBuilder &builder, Location loc,
                                   ArrayRef<ReuseLogicalAccess> accesses,
                                   const ReuseExecutionPlan &executionPlan,
                                   ValueRange stateOperands,
                                   Value currentIterationRingHead) {
  // Precompute per-cluster ring slots once before rewriting the loads in that
  // block, so each load wrapper reuses the same physical index materialization.
  DenseMap<Operation *, RingAccessPrecomputedIndices> precomputed;
  if (executionPlan.strategy != ReuseBufferStrategy::Ring)
    return precomputed;

  const ReuseStatePlan &plan = executionPlan.statePlan;
  int slidingReusePos = plan.resultToReusePos[plan.slidingDim];
  assert(slidingReusePos >= 0 && "expected sliding dimension to be kept");
  (void)slidingReusePos;

  for (const RingAccessCluster &cluster : collectRingAccessClusters(accesses)) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(accesses[cluster.accessIndices.front()].anchorOp);
    for (unsigned accessIdx : cluster.accessIndices) {
      const ReuseLogicalAccess &access = accesses[accessIdx];
      RingAccessPrecomputedIndices indices;
      indices.logicalIndices = materializeLogicalReuseIndices(
          builder, loc, plan, access.composed, stateOperands);
      indices.physicalIndices = materializePhysicalReuseIndices(
          builder, loc, executionPlan, indices.logicalIndices,
          currentIterationRingHead);
      precomputed.try_emplace(access.anchorOp, std::move(indices));
    }
  }

  return precomputed;
}

static void generateReuseStateRefill(OpBuilder &builder, Location loc,
                                     Value sourceBuffer, Value reuseBuffer,
                                     const ReuseExecutionPlan &executionPlan,
                                     ValueRange stateOperands,
                                     Value ringHead = nullptr) {
  // Refill only the entering face of the reuse state. Ring mode threads the
  // physical slot with an iter_arg instead of recomputing modulo per element.
  const ReuseStatePlan &plan = executionPlan.statePlan;
  int slidingReusePos = plan.resultToReusePos[plan.slidingDim];
  int64_t slidingExtent = plan.dims[plan.slidingDim].extent;
  int64_t enteringBaseOffset =
      executionPlan.slidingDelta > 0
          ? slidingExtent - executionPlan.slidingStepAbs
          : 0;

  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 4> logicalIndices(plan.keptDims.size());
  Value physicalSlidingIndex = nullptr;
  if (slidingReusePos >= 0) {
    if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
      Value firstPhysicalSlot = buildModuloOffsetValue(
          builder, loc, ringHead,
          arith::ConstantIndexOp::create(builder, loc, enteringBaseOffset),
          slidingExtent);
      affine::AffineForOp enteringFaceFor = createConstantAffineFor(
          builder, loc, /*lb=*/0, /*ub=*/executionPlan.slidingStepAbs,
          firstPhysicalSlot);
      builder.setInsertionPoint(
          enteringFaceFor.getBody(),
          Block::iterator(enteringFaceFor.getBody()->getTerminator()));
      Value enteringFaceIV = enteringFaceFor.getInductionVar();
      logicalIndices[slidingReusePos] = buildEnteringFaceLogicalIndex(
          builder, loc, enteringFaceIV, executionPlan, enteringBaseOffset);
      physicalSlidingIndex = enteringFaceFor.getRegionIterArgs().front();
      auto yieldOp = cast<affine::AffineYieldOp>(
          enteringFaceFor.getBody()->getTerminator());
      Value nextPhysicalSlot = buildWrappedIncrementValue(
          builder, loc, physicalSlidingIndex, slidingExtent);
      yieldOp->setOperand(0, nextPhysicalSlot);
    } else {
      affine::AffineForOp enteringFaceFor = createConstantAffineFor(
          builder, loc, /*lb=*/0, /*ub=*/executionPlan.slidingStepAbs);
      builder.setInsertionPoint(
          enteringFaceFor.getBody(),
          Block::iterator(enteringFaceFor.getBody()->getTerminator()));
      Value enteringFaceIV = enteringFaceFor.getInductionVar();
      logicalIndices[slidingReusePos] = buildEnteringFaceLogicalIndex(
          builder, loc, enteringFaceIV, executionPlan, enteringBaseOffset);
    }
  }

  for (auto [reusePos, dim] : llvm::enumerate(plan.keptDims)) {
    if (static_cast<int>(reusePos) == slidingReusePos)
      continue;
    affine::AffineForOp forOp =
        createConstantAffineFor(builder, loc, /*lb=*/0,
                                /*ub=*/plan.shape[reusePos]);
    builder.setInsertionPoint(
        forOp.getBody(), Block::iterator(forOp.getBody()->getTerminator()));
    logicalIndices[reusePos] = forOp.getInductionVar();
  }

  SmallVector<Value, 4> globalIndices(plan.dims.size());
  for (auto [resultDim, dimPlan] : llvm::enumerate(plan.dims)) {
    Value anchor = buildAnchorValue(builder, loc, dimPlan, stateOperands);
    int reusePos = plan.resultToReusePos[resultDim];
    if (reusePos < 0) {
      globalIndices[resultDim] = anchor;
      continue;
    }
    globalIndices[resultDim] = buildStridedOffsetValue(
        builder, loc, anchor, logicalIndices[reusePos], dimPlan.layoutStride);
  }

  Value loaded =
      affine::AffineLoadOp::create(builder, loc, sourceBuffer, globalIndices);
  if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
    SmallVector<Value, 4> physicalIndices = materializePhysicalReuseIndices(
        builder, loc, executionPlan, logicalIndices, ringHead,
        physicalSlidingIndex);
    memref::StoreOp::create(builder, loc, loaded, reuseBuffer, physicalIndices);
    return;
  }
  affine::AffineStoreOp::create(builder, loc, loaded, reuseBuffer,
                                logicalIndices);
}

// Populates the entire reuse window from source before steady state begins:
// steady-state maintenance only refills the entering face and relies on the
// rest of the window already being valid. PhysicalShift only.
static void generateReuseStateWarmupFill(
    OpBuilder &builder, Location loc, Value sourceBuffer, Value reuseBuffer,
    const ReuseExecutionPlan &executionPlan, ValueRange stateOperands) {
  const ReuseStatePlan &plan = executionPlan.statePlan;
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 4> logicalIndices(plan.keptDims.size());
  for (auto [reusePos, dim] : llvm::enumerate(plan.keptDims)) {
    affine::AffineForOp forOp = createConstantAffineFor(
        builder, loc, /*lb=*/0, /*ub=*/plan.shape[reusePos]);
    builder.setInsertionPoint(
        forOp.getBody(), Block::iterator(forOp.getBody()->getTerminator()));
    logicalIndices[reusePos] = forOp.getInductionVar();
  }

  SmallVector<Value, 4> globalIndices(plan.dims.size());
  for (auto [resultDim, dimPlan] : llvm::enumerate(plan.dims)) {
    Value anchor = buildAnchorValue(builder, loc, dimPlan, stateOperands);
    int reusePos = plan.resultToReusePos[resultDim];
    if (reusePos < 0) {
      globalIndices[resultDim] = anchor;
      continue;
    }
    globalIndices[resultDim] = buildStridedOffsetValue(
        builder, loc, anchor, logicalIndices[reusePos], dimPlan.layoutStride);
  }

  Value loaded =
      affine::AffineLoadOp::create(builder, loc, sourceBuffer, globalIndices);
  affine::AffineStoreOp::create(builder, loc, loaded, reuseBuffer,
                                logicalIndices);
}

static FailureOr<ReuseAccessFamilyAnalysis>
analyzeReuseAccessFamily(affine::AffineForOp axisLoop,
                         affine::AffineForOp rootLoop, Value target,
                         unsigned rank, const LoopNormalizationInfo &axisInfo,
                         DenseMap<Value, LoopNormalizationInfo> &loopInfos,
                         MLIRContext *ctx, bool enableRing) {
  SmallVector<ReuseLogicalAccess, 8> accesses;
  SmallVector<affine::AffineForOp, 8> innerLoops;
  if (failed(collectReuseAccesses(axisLoop, target, accesses, innerLoops))) {
    return failure();
  }

  auto stagePlanOr = analyzeReuseStatePlan(accesses, innerLoops, axisLoop,
                                           axisInfo, loopInfos, rank, ctx);
  if (failed(stagePlanOr))
    return failure();
  auto validityOr = analyzeReuseValidityPlan(accesses, innerLoops, axisInfo,
                                             loopInfos, *stagePlanOr);
  if (failed(validityOr)) {
    return failure();
  }

  ReuseExecutionPlan executionPlan(std::move(*stagePlanOr),
                                   std::move(*validityOr), enableRing);
  ReuseResetBoundaryPlan resetBoundaryPlan =
      analyzeReuseResetBoundary(axisLoop, rootLoop, executionPlan);
  return ReuseAccessFamilyAnalysis(std::move(accesses), std::move(innerLoops),
                                   std::move(executionPlan), resetBoundaryPlan);
}

static Value
emitReuseStateMaintenance(OpBuilder &builder, affine::AffineForOp axisLoop,
                          Location loc, Value target, Value reuseBuffer,
                          const ReuseExecutionPlan &executionPlan,
                          ValueRange stateOperands, Value logicalAxisIV) {
  auto updateActiveSet = buildReuseUpdateActiveSet(
      builder, executionPlan.updateStartIter, executionPlan.updateEndIter);
  if (executionPlan.strategy != ReuseBufferStrategy::Ring) {
    if (!executionPlan.statePlan.accessesCoverWindow) {
      IntegerSet warmupSet =
          buildReuseWarmupMissSet(builder, executionPlan.steadyStateStart);
      auto warmupIf =
          affine::AffineIfOp::create(builder, loc, warmupSet, logicalAxisIV,
                                     /*withElseRegion=*/false);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(
            warmupIf.getThenBlock(),
            Block::iterator(warmupIf.getThenBlock()->getTerminator()));
        generateReuseStateWarmupFill(builder, loc, target, reuseBuffer,
                                     executionPlan, stateOperands);
      }
      markReuseMaintenanceWrites(warmupIf, reuseBuffer);
    }

    if (!updateActiveSet)
      return {};
    auto updateIf = affine::AffineIfOp::create(builder, loc, *updateActiveSet,
                                               logicalAxisIV,
                                               /*withElseRegion=*/false);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(
        updateIf.getThenBlock(),
        Block::iterator(updateIf.getThenBlock()->getTerminator()));
    generateReuseStateShift(builder, loc, reuseBuffer, executionPlan);
    generateReuseStateRefill(builder, loc, target, reuseBuffer, executionPlan,
                             stateOperands);
    markReuseMaintenanceWrites(updateIf, reuseBuffer);
    return {};
  }

  Value previousIterationRingHead = axisLoop.getRegionIterArgs().back();
  int64_t slidingExtent =
      executionPlan.statePlan.dims[executionPlan.statePlan.slidingDim].extent;
  Value increment =
      arith::ConstantIndexOp::create(builder, loc, executionPlan.ringIncrement);
  Value nextHead = buildModuloOffsetValue(
      builder, loc, previousIterationRingHead, increment, slidingExtent);
  if (!updateActiveSet)
    return previousIterationRingHead;

  Value updateStart = arith::ConstantIndexOp::create(
      builder, loc, executionPlan.updateStartIter);
  Value updateEnd =
      arith::ConstantIndexOp::create(builder, loc, executionPlan.updateEndIter);
  Value atOrAfterStart = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sge, logicalAxisIV, updateStart);
  Value atOrBeforeEnd = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sle, logicalAxisIV, updateEnd);
  Value isUpdateActive =
      arith::AndIOp::create(builder, loc, atOrAfterStart, atOrBeforeEnd);
  auto updateIf =
      affine::AffineIfOp::create(builder, loc, *updateActiveSet, logicalAxisIV,
                                 /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(
        updateIf.getThenBlock(),
        Block::iterator(updateIf.getThenBlock()->getTerminator()));
    generateReuseStateRefill(builder, loc, target, reuseBuffer, executionPlan,
                             stateOperands, nextHead);
  }
  markReuseMaintenanceWrites(updateIf, reuseBuffer);
  return arith::SelectOp::create(builder, loc, isUpdateActive, nextHead,
                                 previousIterationRingHead);
}

static LogicalResult
rewriteReuseAccesses(RewriterBase &builder,
                     ArrayRef<ReuseLogicalAccess> accesses, Value target,
                     Value reuseBuffer, const ReuseExecutionPlan &executionPlan,
                     ValueRange stateOperands, Value logicalAxisIV,
                     Value currentIterationRingHead) {
  auto ringPrecomputedIndices = precomputeRingAccessClusterIndices(
      builder, reuseBuffer.getLoc(), accesses, executionPlan, stateOperands,
      currentIterationRingHead);

  for (auto [accessIdx, access] : llvm::enumerate(accesses)) {
    builder.setInsertionPoint(access.anchorOp);

    ArrayRef<Value> precomputedLogicalIndices;
    ArrayRef<Value> precomputedPhysicalIndices;
    if (auto it = ringPrecomputedIndices.find(access.anchorOp);
        it != ringPrecomputedIndices.end()) {
      precomputedLogicalIndices = it->second.logicalIndices;
      precomputedPhysicalIndices = it->second.physicalIndices;
    }

    auto rewritten = createConditionalReuseLoad(
        builder, access.anchorOp->getLoc(), access, target, reuseBuffer,
        executionPlan, stateOperands,
        executionPlan.validityPlan.accesses[accessIdx], logicalAxisIV,
        currentIterationRingHead, precomputedLogicalIndices,
        precomputedPhysicalIndices);
    Operation *rewrittenOp = rewritten.value.getDefiningOp();
    Value exposedValue = access.exposedValue;
    exposedValue.replaceUsesWithIf(rewritten.value, [&](OpOperand &use) {
      return !rewrittenOp->isAncestor(use.getOwner());
    });
    if (access.anchorOp->use_empty())
      builder.eraseOp(access.anchorOp);
  }
  return success();
}

static ReuseConditionalLoadResult createConditionalReuseLoad(
    OpBuilder &builder, Location loc, const ReuseLogicalAccess &access,
    Value sourceBuffer, Value reuseBuffer,
    const ReuseExecutionPlan &executionPlan, ValueRange stateOperands,
    const ReuseAccessValidity &accessValidity, Value logicalAxisIV,
    Value currentIterationRingHead, ArrayRef<Value> precomputedLogicalIndices,
    ArrayRef<Value> precomputedPhysicalIndices) {
  // Warm-up keeps the original load and captures it into reuse state.
  // Steady-state switches the same use to the analyzed reuse coordinates.
  const ReuseStatePlan &plan = executionPlan.statePlan;
  SmallVector<Value, 4> logicalIndices(precomputedLogicalIndices.begin(),
                                       precomputedLogicalIndices.end());
  if (logicalIndices.empty()) {
    logicalIndices = materializeLogicalReuseIndices(
        builder, loc, plan, access.composed, stateOperands);
  }

  bool coversTail = accessValidity.lastReusableIter >=
                    executionPlan.validityPlan.axisTripCount - 1;
  affine::AffineIfOp ifOp;
  if (coversTail) {
    auto warmupSet =
        buildReuseWarmupMissSet(builder, accessValidity.firstReusableIter);
    ifOp = affine::AffineIfOp::create(
        builder, loc, access.exposedValue.getType(), warmupSet, logicalAxisIV,
        /*withElseRegion=*/true);
  } else {
    auto reusableSet =
        buildReuseReusableHitSet(builder, accessValidity.firstReusableIter,
                                 accessValidity.lastReusableIter);
    ifOp = affine::AffineIfOp::create(
        builder, loc, access.exposedValue.getType(), reusableSet, logicalAxisIV,
        /*withElseRegion=*/true);
  }

  auto buildReuseHit = [&](Block *block) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(block);
    SmallVector<Value, 4> physicalIndices(precomputedPhysicalIndices.begin(),
                                          precomputedPhysicalIndices.end());
    if (physicalIndices.empty()) {
      physicalIndices = materializePhysicalReuseIndices(
          builder, loc, executionPlan, logicalIndices,
          currentIterationRingHead);
    }
    Value reused;
    if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
      reused =
          memref::LoadOp::create(builder, loc, reuseBuffer, physicalIndices);
    } else {
      reused = affine::AffineLoadOp::create(builder, loc, reuseBuffer,
                                            physicalIndices);
    }
    affine::AffineYieldOp::create(builder, loc, reused);
  };

  auto buildMiss = [&](Block *block) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(block);
    Value loaded = affine::AffineLoadOp::create(builder, loc, sourceBuffer,
                                                access.composed.map,
                                                access.composed.operands);
    affine::AffineStoreOp::create(builder, loc, loaded, reuseBuffer,
                                  logicalIndices);
    affine::AffineYieldOp::create(builder, loc, loaded);
  };

  if (coversTail) {
    buildMiss(ifOp.getThenBlock());
    buildReuseHit(ifOp.getElseBlock());
    return {ifOp.getResult(0), std::move(logicalIndices)};
  }
  buildReuseHit(ifOp.getThenBlock());
  buildMiss(ifOp.getElseBlock());
  return {ifOp.getResult(0), std::move(logicalIndices)};
}

static Value materializeGlobalAccessIndex(OpBuilder &builder, Location loc,
                                          const ComposedBufferAccess &access,
                                          unsigned resultDim) {
  AffineMap singleResultMap = access.map.getSubMap(resultDim);
  SmallVector<OpFoldResult, 4> ofrs;
  for (Value operand : access.operands)
    ofrs.push_back(operand);
  return affine::makeComposedAffineApply(builder, loc, singleResultMap, ofrs);
}

// Resolve the single-char `allo.signed` marker ('s'/'u') of a memref value so
// a buffer derived from it can inherit the marker.
static StringAttr resolveMemRefSignedMarker(Value memref) {
  if (Operation *def = memref.getDefiningOp())
    return def->getAttrOfType<StringAttr>(kAlloSignedAttr);
  auto arg = dyn_cast<BlockArgument>(memref);
  if (!arg)
    return nullptr;
  auto marker =
      arg.getOwner()->getParentOp()->getAttrOfType<StringAttr>(kAlloSignedAttr);
  if (!marker || arg.getArgNumber() >= marker.getValue().size())
    return nullptr;
  return StringAttr::get(memref.getContext(),
                         marker.getValue().substr(arg.getArgNumber(), 1));
}

DiagnosedSilenceableFailure
transform::ReuseAtOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  // Stage 0: resolve payload handles and validate structural preconditions.
  auto targets = llvm::to_vector(state.getPayloadValues(getTarget()));
  if (targets.size() != 1) {
    return emitSilenceableError()
           << "expected target handle to resolve to exactly one payload value";
  }
  Value target = targets.front();
  auto targetType = dyn_cast<MemRefType>(target.getType());
  if (!targetType)
    return emitSilenceableError()
           << "expected target to resolve to a memref value";

  auto loops = llvm::to_vector(state.getPayloadOps(getAxis()));
  if (loops.size() != 1) {
    return emitSilenceableError() << "expected axis handle to resolve to "
                                     "exactly one payload operation";
  }
  auto axisLoop = dyn_cast<affine::AffineForOp>(loops.front());
  if (!axisLoop)
    return emitSilenceableError()
           << "expected axis to resolve to exactly one affine.for loop";

  auto axisInfoOr = analyzeLoopNormalization(axisLoop);
  if (failed(axisInfoOr))
    return emitSilenceableError()
           << "reuse_at requires the selected axis loop to have constant "
              "bounds and a positive constant step";
  LoopNormalizationInfo axisInfo = *axisInfoOr;

  Operation *targetDef = target.getDefiningOp();
  if (targetDef && axisLoop->isAncestor(targetDef))
    return emitSilenceableError()
           << "expected target buffer to be defined outside the selected axis "
              "loop";

  Value axisIV = axisLoop.getInductionVar();
  unsigned rank = targetType.getRank();

  // Stage 1: analyze the loop nest and the candidate accesses under the axis.
  affine::AffineForOp rootLoop = getRootLoop(axisLoop);
  LoopRoleInfo roles;
  SmallVector<affine::AffineForOp, 8> allLoops;
  DenseMap<Value, LoopNormalizationInfo> loopInfos;
  if (failed(classifyLoopRoles(rootLoop, target, roles, loopInfos, allLoops))) {
    return emitSilenceableError()
           << "failed to classify loop roles; loops must have constant bounds "
              "with positive constant step and target must be loaded in the "
              "axis stage";
  }
  // The chosen axis must be spatial (store-indexing), not reduction-only.
  if (isReductionLoop(axisLoop, roles))
    return emitSilenceableError()
           << "selected axis loop is classified as a reduction loop";
  if (!isSpatialLoop(axisLoop, roles))
    return emitSilenceableError()
           << "selected axis loop is not classified as a spatial loop";

  auto analysisOr = analyzeReuseAccessFamily(
      axisLoop, rootLoop, target, rank, axisInfo, loopInfos,
      rewriter.getContext(), getUseRingBuffer());
  if (failed(analysisOr))
    return emitSilenceableError()
           << "failed to analyze reuse candidate accesses";
  ReuseAccessFamilyAnalysis analysis = std::move(*analysisOr);
  SmallVector<ReuseLogicalAccess, 8> &accesses = analysis.accesses;
  const ReuseExecutionPlan &executionPlan = analysis.executionPlan;
  const ReuseResetBoundaryPlan &resetBoundaryPlan = analysis.resetBoundaryPlan;
  const ReuseStatePlan &plan = executionPlan.statePlan;

  // Stage 2: materialize the reuse buffer and prepare any loop-carried state.
  rewriter.setInsertionPoint(resetBoundaryPlan.canHoist ? rootLoop : axisLoop);
  auto reuseBuffer = memref::AllocOp::create(
      rewriter, axisLoop.getLoc(),
      MemRefType::get(plan.shape, targetType.getElementType()));
  if (auto sgn = resolveMemRefSignedMarker(target))
    reuseBuffer->setAttr(kAlloSignedAttr, sgn);
  if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
    // `replaceWithAdditionalYields` builds a fresh affine.for; carry over the
    // schedule annotations so the recreated loop stays matchable downstream.
    SmallVector<NamedAttribute, 4> savedAttrs(
        axisLoop->getDiscardableAttrs().begin(),
        axisLoop->getDiscardableAttrs().end());
    Value zero = arith::ConstantIndexOp::create(rewriter, axisLoop.getLoc(), 0);
    auto newLoopOr = axisLoop.replaceWithAdditionalYields(
        rewriter, zero,
        /*replaceInitOperandUsesInLoop=*/false,
        [&](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBbArgs) {
          return SmallVector<Value, 1>{newBbArgs.front()};
        });
    if (failed(newLoopOr))
      return emitDefiniteFailure();
    axisLoop = cast<affine::AffineForOp>(*newLoopOr);
    for (NamedAttribute attr : savedAttrs)
      axisLoop->setAttr(attr.getName(), attr.getValue());
    axisIV = axisLoop.getInductionVar();
    // Ring scalarization recreates the axis loop and its IV; recompose each
    // (direct) access from its payload op so it references the new IV.
    for (ReuseLogicalAccess &access : accesses)
      access.composed = composeBufferAccess(access.anchorOp);
  }

  // Stage 3: emit per-iteration state maintenance for ring or shift mode.
  rewriter.setInsertionPointToStart(axisLoop.getBody());
  Value logicalAxisIV = materializeNormalizedLoopIndex(
      rewriter, axisLoop.getLoc(), axisInfo, axisIV);
  SmallVector<Value, 4> stateOperands =
      getReuseStateOperands(logicalAxisIV, plan.prefixOperands);
  Value currentIterationRingHead = emitReuseStateMaintenance(
      rewriter, axisLoop, axisLoop.getLoc(), target, reuseBuffer, executionPlan,
      stateOperands, logicalAxisIV);

  // Stage 4: rewrite each candidate load to the new reuse state.
  if (failed(rewriteReuseAccesses(rewriter, accesses, target, reuseBuffer,
                                  executionPlan, stateOperands, logicalAxisIV,
                                  currentIterationRingHead))) {
    return emitSilenceableError() << "failed to rewrite reuse accesses";
  }

  if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
    auto yieldOp =
        cast<affine::AffineYieldOp>(axisLoop.getBody()->getTerminator());
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp->setOperand(yieldOp.getNumOperands() - 1,
                          currentIterationRingHead);
    });
  }

  // A freshly created reuse buffer has no user stores, so mark every write
  // into it; the per-`updateIf` marking above only covers shift and refill.
  markReuseMaintenanceWrites(axisLoop, reuseBuffer);

  // Stage 5: publish the new buffer handle. No cleanup runs here: it would
  // reshape the conditional loads and strip the maintenance markers a chained
  // reuse_at relies on. The `allo-reuse-cleanup` pass simplifies the IR during
  // schedule export instead.
  results.setValues(cast<OpResult>(getResult()), {reuseBuffer});
  return DiagnosedSilenceableFailure::success();
}

void transform::ReuseAtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  if (getUseRingBuffer())
    consumesHandle(getAxisMutable(), effects);
  else
    onlyReadsHandle(getAxisMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

///===----------------------------------------------------------------------===///
/// LoopUnroll implementation
///===----------------------------------------------------------------------===///
DiagnosedSilenceableFailure transform::AlloLoopUnrollOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target || !isa<LoopLikeOpInterface>(target)) {
    return emitSilenceableError()
           << "expected target to resolve to exactly one loop-like operation";
  }
  auto factor = getFactorAttr().getInt();
  if (factor < 0) {
    return emitSilenceableError()
           << "expected unroll factor to be a non-negative "
           << "integer (0 for full unroll)";
  }

  LogicalResult result = failure();
  if (auto forOp = dyn_cast<scf::ForOp>(target)) {
    if (factor == 0) {
      result = loopUnrollFull(forOp);
    } else {
      auto unrolled = loopUnrollByFactor(forOp, factor);
      result = succeeded(unrolled) ? success() : failure();
    }
  } else if (auto forOp = dyn_cast<affine::AffineForOp>(target)) {
    result = factor == 0 ? affine::loopUnrollFull(forOp)
                         : affine::loopUnrollByFactor(forOp, factor);
  } else {
    return emitSilenceableError()
           << "failed to unroll, expected scf.for or affine.for";
  }
  if (failed(result))
    return emitSilenceableError() << "failed to unroll";
  return DiagnosedSilenceableFailure::success();
}

void transform::AlloLoopUnrollOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getLoopsMutable(), effects);
  modifiesPayload(effects);
}

///===----------------------------------------------------------------------===///
/// BufferAt implementation
///===----------------------------------------------------------------------===///

namespace {
struct BufferAtFootprint {
  // Per-instance local buffer layout derived from all accesses under the chosen
  // axis. `symbols` are the outer operands needed to materialize bounds/remaps.
  SmallVector<int64_t, 4> shape;
  SmallVector<AffineMap, 4> lowerBounds;
  SmallVector<AffineMap, 4> upperBounds;
  SmallVector<Value, 4> symbols;
  AffineMap localIndexRemap;
};
} // namespace

static DiagnosedSilenceableFailure
collectBufferAtAccesses(affine::AffineForOp axisLoop, Value buffer,
                        SmallVectorImpl<Operation *> &accessOps, bool &hasLoads,
                        bool &hasStores) {
  // Gather the affine accesses to rewrite, rejecting non-affine and
  // alias/view-based accesses: the later remap step only understands direct
  // affine accesses to the chosen memref value.
  Value bufferRoot = resolveMemRefValueRoot(buffer);
  Operation *offendingOp = nullptr;
  StringRef reason;

  auto walkResult = axisLoop.walk([&](Operation *op) {
    if (auto readOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
      if (readOp.getMemRef() == buffer) {
        accessOps.push_back(op);
        hasLoads = true;
      }
    } else if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      if (writeOp.getMemRef() == buffer) {
        accessOps.push_back(op);
        hasStores = true;
      }
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      if (resolveMemRefValueRoot(loadOp.getMemRef()) == bufferRoot) {
        offendingOp = op;
        reason = "buffer_at only supports affine.load/store accesses to the "
                 "target buffer within the selected axis loop";
        return WalkResult::interrupt();
      }
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      if (resolveMemRefValueRoot(storeOp.getMemRef()) == bufferRoot) {
        offendingOp = op;
        reason = "buffer_at only supports affine.load/store accesses to the "
                 "target buffer within the selected axis loop";
        return WalkResult::interrupt();
      }
    } else if (isMemRefCastOrViewLike(op)) {
      bool aliasesBuffer = llvm::any_of(op->getResults(), [&](Value result) {
        return isa<BaseMemRefType>(result.getType()) &&
               resolveMemRefValueRoot(result) == bufferRoot;
      });
      if (aliasesBuffer) {
        offendingOp = op;
        reason = "buffer_at does not support aliasing/view accesses to the "
                 "target buffer within the selected axis loop";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    DiagnosedSilenceableFailure diag = emitSilenceableFailure(axisLoop);
    diag << reason;
    diag.attachNote(offendingOp->getLoc()) << "see offending access here";
    return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

static FailureOr<std::pair<AffineExpr, int64_t>>
computeFootprintDim(const ComposedBufferAccess &access, unsigned resultPos,
                    ArrayRef<affine::AffineForOp> innerLoops,
                    DenseMap<Value, unsigned> &prefixOperandPos,
                    SmallVectorImpl<Value> &prefixOperands) {
  // Infer one local-buffer dimension from one access result. Only a singleton
  // point or a unit-stride interval driven by exactly one inner loop is
  // accepted, which keeps the local allocation finite and easy to reindex.
  AffineExpr accessExpr = access.map.getResult(resultPos);
  SmallVector<affine::AffineForOp, 2> dependentLoops;
  for (affine::AffineForOp loop : innerLoops) {
    if (affineExprUsesValue(accessExpr, access.operands,
                            access.map.getNumDims(), loop.getInductionVar())) {
      dependentLoops.push_back(loop);
    }
  }

  SmallVector<AffineExpr, 8> dimReplacements, symReplacements;
  if (dependentLoops.empty()) {
    populateExprReplacements(access.map, access.operands, prefixOperandPos,
                             prefixOperands, Value{}, std::nullopt,
                             std::nullopt, /*prefixDimOffset=*/0,
                             dimReplacements, symReplacements);
    AffineExpr lowerExpr = simplifyAffineExpr(
        accessExpr.replaceDimsAndSymbols(dimReplacements, symReplacements),
        prefixOperands.size(), /*numSymbols=*/0);
    return std::make_pair(lowerExpr, static_cast<int64_t>(1));
  }

  if (dependentLoops.size() != 1)
    return failure();

  affine::AffineForOp loop = dependentLoops.front();
  if (!loop.hasConstantBounds() || loop.getStepAsInt() != 1)
    return failure();

  int64_t lb = loop.getConstantLowerBound();
  int64_t ub = loop.getConstantUpperBound();
  if (ub <= lb)
    return failure();

  populateExprReplacements(access.map, access.operands, prefixOperandPos,
                           prefixOperands, loop.getInductionVar(), lb,
                           std::nullopt, /*prefixDimOffset=*/0, dimReplacements,
                           symReplacements);
  AffineExpr lowerExpr = simplifyAffineExpr(
      accessExpr.replaceDimsAndSymbols(dimReplacements, symReplacements),
      prefixOperands.size(), /*numSymbols=*/0);

  SmallVector<AffineExpr, 8> diffDims, diffSyms;
  populateExprReplacements(access.map, access.operands, prefixOperandPos,
                           prefixOperands, loop.getInductionVar(), std::nullopt,
                           /*targetLoopDimPos=*/0, /*prefixDimOffset=*/1,
                           diffDims, diffSyms);
  // Check that varying the chosen loop produces a zero-based contiguous index:
  // `(iv -> expr)` must be equivalent to `(iv - lb)` after subtracting the
  // common lower bound. This rules out gaps, permutations, and nonlinear forms.
  AffineExpr shiftedExpr =
      simplifyAffineExpr(accessExpr.replaceDimsAndSymbols(diffDims, diffSyms),
                         1 + prefixOperands.size(), /*numSymbols=*/0);

  SmallVector<AffineExpr, 4> expandedPrefixDims;
  expandedPrefixDims.reserve(prefixOperands.size());
  for (unsigned i = 0; i < prefixOperands.size(); ++i)
    expandedPrefixDims.push_back(
        getAffineDimExpr(i + 1, access.map.getContext()));
  AffineExpr expandedLowerExpr =
      simplifyAffineExpr(lowerExpr.replaceDims(expandedPrefixDims),
                         1 + prefixOperands.size(), /*numSymbols=*/0);
  AffineExpr zeroBasedExpr = simplifyAffineExpr(shiftedExpr - expandedLowerExpr,
                                                1 + prefixOperands.size(),
                                                /*numSymbols=*/0);
  AffineExpr expectedExpr =
      simplifyAffineExpr(getAffineDimExpr(0, access.map.getContext()) - lb,
                         1 + prefixOperands.size(), /*numSymbols=*/0);
  if (zeroBasedExpr != expectedExpr)
    return failure();

  return std::make_pair(lowerExpr, ub - lb);
}

static FailureOr<BufferAtFootprint>
analyzeBufferAtFootprint(ArrayRef<Operation *> accessOps,
                         ArrayRef<affine::AffineForOp> innerLoops,
                         unsigned bufferRank, MLIRContext *ctx) {
  // All accesses inside one axis instance must fit into the same local layout:
  // the per-access footprints must agree exactly.
  BufferAtFootprint footprint;
  footprint.shape.resize(bufferRank);
  footprint.lowerBounds.resize(bufferRank);
  footprint.upperBounds.resize(bufferRank);

  SmallVector<ComposedBufferAccess, 8> accesses;
  accesses.reserve(accessOps.size());
  for (Operation *accessOp : accessOps)
    accesses.push_back(composeBufferAccess(accessOp));

  DenseMap<Value, unsigned> prefixOperandPos;
  collectFootprintOperands(accesses, innerLoops, /*excludedValues=*/{},
                           prefixOperandPos, footprint.symbols);
  SmallVector<std::pair<AffineExpr, int64_t>, 4> commonFootprint;
  commonFootprint.reserve(bufferRank);

  for (const ComposedBufferAccess &access : accesses) {
    if (access.map.getNumResults() != bufferRank)
      return failure();

    SmallVector<std::pair<AffineExpr, int64_t>, 4> accessFootprint;
    accessFootprint.reserve(bufferRank);
    for (unsigned d = 0; d < bufferRank; ++d) {
      auto dimFootprint = computeFootprintDim(
          access, d, innerLoops, prefixOperandPos, footprint.symbols);
      if (failed(dimFootprint))
        return failure();
      accessFootprint.push_back(*dimFootprint);
    }

    if (commonFootprint.empty()) {
      commonFootprint = accessFootprint;
      continue;
    }
    if (commonFootprint != accessFootprint)
      return failure();
  }

  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(bufferRank);
  for (unsigned d = 0; d < bufferRank; ++d) {
    AffineExpr lowerExpr = commonFootprint[d].first;
    int64_t extent = commonFootprint[d].second;
    if (extent <= 0)
      return failure();

    footprint.shape[d] = extent;
    footprint.lowerBounds[d] =
        AffineMap::get(footprint.symbols.size(), /*symbolCount=*/0, lowerExpr);
    footprint.upperBounds[d] = AffineMap::get(
        footprint.symbols.size(), /*symbolCount=*/0, lowerExpr + extent);

    AffineExpr globalIndex =
        getAffineDimExpr(footprint.symbols.size() + d, ctx);
    remapExprs.push_back(simplifyAffineExpr(
        globalIndex - lowerExpr, footprint.symbols.size() + bufferRank,
        /*numSymbols=*/0));
  }
  footprint.localIndexRemap =
      AffineMap::get(footprint.symbols.size() + bufferRank, /*symbolCount=*/0,
                     remapExprs, ctx);
  return footprint;
}

static bool affineExprUsesDimPosition(AffineExpr expr, unsigned dimPos) {
  bool used = false;
  expr.walk([&](AffineExpr inner) {
    if (auto dim = dyn_cast<AffineDimExpr>(inner);
        dim && dim.getPosition() == dimPos)
      used = true;
  });
  return used;
}

static DiagnosedSilenceableFailure
checkBufferAtFootprintSeparability(const BufferAtFootprint &footprint,
                                   affine::AffineForOp axisLoop) {
  // A legal buffer_at needs the selected axis to separate per-instance regions.
  // This is approximated by checking whether the axis moves some footprint
  // bound far enough that adjacent iterations do not overlap on that dimension.
  auto notPrivatizable = [&](StringRef note) {
    DiagnosedSilenceableFailure diag = emitSilenceableFailure(axisLoop);
    diag << "cannot buffer_at on this axis because the target buffer cannot "
            "be made private to each iteration";
    diag.attachNote(axisLoop.getLoc()) << note;
    return diag;
  };
  StringRef axisIndependent =
      "the target-buffer access pattern does not depend on the selected axis, "
      "so every iteration would use the same region";

  Value axisIV = axisLoop.getInductionVar();
  auto *it = llvm::find(footprint.symbols, axisIV);
  if (it == footprint.symbols.end())
    return notPrivatizable(axisIndependent);

  unsigned axisPos = std::distance(footprint.symbols.begin(), it);
  uint64_t axisStep = axisLoop.getStepAsInt();
  bool foundAxisSensitiveDim = false;
  for (auto [shape, lbMap] :
       llvm::zip_equal(footprint.shape, footprint.lowerBounds)) {
    AffineExpr lowerExpr = lbMap.getResult(0);
    if (!affineExprUsesDimPosition(lowerExpr, axisPos))
      continue;

    FailureOr<int64_t> axisCoeff = getLinearAffineDimCoefficient(
        lowerExpr, footprint.symbols.size(), axisPos);
    if (failed(axisCoeff) || *axisCoeff == 0)
      continue;

    foundAxisSensitiveDim = true;
    uint64_t separatingStride =
        static_cast<uint64_t>(std::abs(*axisCoeff)) * axisStep;
    if (separatingStride >= static_cast<uint64_t>(shape))
      return DiagnosedSilenceableFailure::success();
  }

  if (!foundAxisSensitiveDim)
    return notPrivatizable(axisIndependent);
  return notPrivatizable("different iterations of the selected axis access "
                         "overlapping regions of the target buffer");
}

static void generateBufferAtCopy(OpBuilder &builder, Location loc,
                                 Value globalBuffer, Value localBuffer,
                                 const BufferAtFootprint &footprint,
                                 bool isCopyOut) {
  // Materialize copy-in/copy-out from the derived footprint maps: the
  // generated loops enumerate the global region and compute local indices by
  // subtracting each dimension's lower bound.
  unsigned rank = cast<MemRefType>(globalBuffer.getType()).getRank();
  if (rank == 0) {
    if (!isCopyOut) {
      Value globalLoad =
          affine::AffineLoadOp::create(builder, loc, globalBuffer, {});
      affine::AffineStoreOp::create(builder, loc, globalLoad, localBuffer, {});
    } else {
      Value localLoad =
          affine::AffineLoadOp::create(builder, loc, localBuffer, {});
      affine::AffineStoreOp::create(builder, loc, localLoad, globalBuffer, {});
    }
    return;
  }

  SmallVector<Value, 4> globalIndices;
  SmallVector<AffineExpr, 4> localExprs;
  SmallVector<Value, 8> localOperands;
  SmallVector<affine::AffineApplyOp, 4> maybeDeadApplys;
  globalIndices.reserve(rank);
  localExprs.reserve(rank);
  localOperands.reserve(2 * rank);

  for (unsigned d = 0; d < rank; ++d) {
    Value globalIndex;
    if (footprint.shape[d] == 1) {
      globalIndex = builder.createOrFold<affine::AffineApplyOp>(
          loc, footprint.lowerBounds[d], footprint.symbols);
    } else {
      auto forOp = affine::createCanonicalizedAffineForOp(
          builder, loc, footprint.symbols, footprint.lowerBounds[d],
          footprint.symbols, footprint.upperBounds[d], /*step=*/1);
      builder = OpBuilder::atBlockTerminator(forOp.getBody());
      globalIndex = forOp.getInductionVar();
    }

    auto offset = affine::AffineApplyOp::create(
        builder, loc, footprint.lowerBounds[d], footprint.symbols);
    maybeDeadApplys.push_back(offset);
    localOperands.push_back(offset);
    localOperands.push_back(globalIndex);
    localExprs.push_back(builder.getAffineDimExpr(2 * d + 1) -
                         builder.getAffineDimExpr(2 * d));
    globalIndices.push_back(globalIndex);
  }

  auto localMap = AffineMap::get(2 * rank, /*symbolCount=*/0, localExprs,
                                 builder.getContext());
  affine::fullyComposeAffineMapAndOperands(&localMap, &localOperands);
  localMap = simplifyAffineMap(localMap);
  affine::canonicalizeMapAndOperands(&localMap, &localOperands);
  for (affine::AffineApplyOp applyOp : maybeDeadApplys)
    if (applyOp.use_empty())
      applyOp.erase();

  if (!isCopyOut) {
    Value globalLoad =
        affine::AffineLoadOp::create(builder, loc, globalBuffer, globalIndices);
    affine::AffineStoreOp::create(builder, loc, globalLoad, localBuffer,
                                  localMap, localOperands);
    return;
  }

  Value localLoad = affine::AffineLoadOp::create(builder, loc, localBuffer,
                                                 localMap, localOperands);
  affine::AffineStoreOp::create(builder, loc, localLoad, globalBuffer,
                                globalIndices);
}

DiagnosedSilenceableFailure
transform::BufferAtOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
                             transform::TransformState &state) {
  // Precondition checks: one memref target, one affine axis, and the target
  // buffer must outlive the chosen loop instance so a local copy makes sense.
  auto buffers = llvm::to_vector(state.getPayloadValues(getTarget()));
  if (buffers.size() != 1) {
    return emitSilenceableError()
           << "expected target handle to resolve to exactly one payload value";
  }
  auto buffer = buffers.front();
  auto bufferType = dyn_cast<MemRefType>(buffer.getType());
  if (!bufferType) {
    return emitSilenceableError()
           << "expected target to resolve to a memref value";
  }
  auto loops = llvm::to_vector(state.getPayloadOps(getAxis()));
  if (loops.size() != 1) {
    return emitSilenceableError() << "expected axis handle to resolve to "
                                     "exactly one payload operation";
  }
  auto axisLoop = dyn_cast<affine::AffineForOp>(loops.front());
  if (!axisLoop) {
    return emitSilenceableError()
           << "expected axis to resolve to a affine.for loop";
  }
  Operation *bufferDef = buffer.getDefiningOp();
  if (bufferDef && axisLoop->isAncestor(bufferDef)) {
    return emitSilenceableError() << "expected target buffer to be defined "
                                     "outside the selected axis loop";
  }

  affine::AffineForOp rootLoop = getRootLoop(axisLoop);

  SmallVector<affine::AffineForOp, 4> band;
  affine::getPerfectlyNestedLoops(band, rootLoop);
  if (band.empty())
    return emitSilenceableError()
           << "cannot find contiguous nested loops for buffer_at";

  auto *axisIt = llvm::find(band, axisLoop);
  if (axisIt == band.end())
    return emitSilenceableError()
           << "selected axis is not in a contiguous loop band";
  unsigned axisIdx = std::distance(band.begin(), axisIt);

  if (axisIdx == band.size() - 1) {
    return emitSilenceableError() << "cannot buffer at innermost loop axis";
  }

  SmallVector<Operation *, 8> localAccessOps;
  bool hasLoads = false;
  bool hasStores = false;
  if (DiagnosedSilenceableFailure diag = collectBufferAtAccesses(
          axisLoop, buffer, localAccessOps, hasLoads, hasStores);
      !diag.succeeded())
    return diag;
  if (localAccessOps.empty()) {
    return emitSilenceableError()
           << "no load/store of the target buffer found within the selected "
              "axis loop";
  }

  SmallVector<affine::AffineForOp, 4> innerLoops(axisIt + 1, band.end());
  // The footprint synthesis and the separability check together are the
  // legality test for buffer_at.
  FailureOr<BufferAtFootprint> footprintOr = analyzeBufferAtFootprint(
      localAccessOps, innerLoops, bufferType.getRank(), axisLoop.getContext());
  if (failed(footprintOr)) {
    return emitSilenceableError()
           << "buffer_at requires a bounded, realizable per-instance affine "
              "footprint for the target buffer";
  }
  BufferAtFootprint footprint = std::move(*footprintOr);
  if (DiagnosedSilenceableFailure diag =
          checkBufferAtFootprintSeparability(footprint, axisLoop);
      !diag.succeeded())
    return diag;

  rewriter.setInsertionPointToStart(axisLoop.getBody());
  Location loc = buffer.getLoc();
  auto localBuffer = memref::AllocOp::create(
      rewriter, loc,
      MemRefType::get(footprint.shape, bufferType.getElementType()));
  if (auto sgn = resolveMemRefSignedMarker(buffer))
    localBuffer->setAttr(kAlloSignedAttr, sgn);

  if (hasLoads)
    generateBufferAtCopy(rewriter, loc, buffer, localBuffer, footprint,
                         /*isCopyOut=*/false);

  if (hasStores) {
    rewriter.setInsertionPoint(axisLoop.getBody()->getTerminator());
    generateBufferAtCopy(rewriter, loc, buffer, localBuffer, footprint,
                         /*isCopyOut=*/true);
  }

  for (Operation *accessOp : localAccessOps) {
    if (failed(affine::replaceAllMemRefUsesWith(
            buffer, localBuffer, accessOp,
            /*extraIndices=*/{}, footprint.localIndexRemap,
            /*extraOperands=*/footprint.symbols,
            /*symbolOperands=*/{}))) {
      return emitSilenceableError()
             << "buffer_at failed to remap accesses into the local buffer";
    }
  }

  results.setValues(cast<OpResult>(getResult()), {localBuffer});

  return DiagnosedSilenceableFailure::success();
}

void transform::BufferAtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getAxisMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}
