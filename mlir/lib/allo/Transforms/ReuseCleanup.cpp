/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_REUSECLEANUPPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

// Internal marker placed by `transform.allo.reuse_at` on every write into a
// reuse buffer. Must match the literal used in
// TransformOps/LoopTransformations.
static constexpr StringLiteral kReuseMaintenanceAttr = "allo.reuse.maintenance";

namespace {
// Merge a run of adjacent affine.if ops in a loop body that share the same
// condition (the per-access conditional reuse loads) into a single affine.if
// with one result per merged op.
struct MergeSameAffineIfsPattern
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern::OpRewritePattern;
  using AdjacentIfList = SmallVector<affine::AffineIfOp, 4>;

private:
  static bool haveSameIfStructure(affine::AffineIfOp lhs,
                                  affine::AffineIfOp rhs) {
    return lhs.getIntegerSet() == rhs.getIntegerSet() &&
           llvm::equal(lhs.getOperands(), rhs.getOperands()) &&
           lhs.hasElse() == rhs.hasElse() &&
           lhs->getNumResults() == rhs.getNumResults();
  }

  static SmallVector<AdjacentIfList, 4>
  collectAdjacentIfLists(affine::AffineForOp forOp) {
    SmallVector<AdjacentIfList, 4> ifLists;
    AdjacentIfList currentList;
    for (Operation &op : forOp.getBody()->getOperations()) {
      if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
        if (!currentList.empty() &&
            !haveSameIfStructure(currentList.back(), ifOp)) {
          if (currentList.size() > 1)
            ifLists.push_back(std::move(currentList));
          currentList.clear();
        }
        currentList.push_back(ifOp);
      } else {
        if (currentList.size() > 1)
          ifLists.push_back(std::move(currentList));
        currentList.clear();
      }
    }
    if (currentList.size() > 1)
      ifLists.push_back(std::move(currentList));
    return ifLists;
  }

  void mergeAdjacentIfs(AdjacentIfList &ifOps, affine::AffineForOp parentOp,
                        PatternRewriter &rewriter) const {
    // ifs can not use each others results
    for (auto ifOp : ifOps) {
      for (auto result : ifOp.getResults()) {
        for (Operation *user : result.getUsers()) {
          if (llvm::is_contained(ifOps, user->getParentOp()))
            return;
        }
      }
    }

    IRMapping mapping;
    auto firstIf = ifOps.front();
    if (!firstIf.hasElse() ||
        firstIf->getNumResults() != 1) // we only merge reuse_at accesses
      return;

    rewriter.setInsertionPoint(firstIf);
    Location loc = parentOp.getLoc();
    Block *thenBlock = rewriter.createBlock(parentOp.getBody());
    rewriter.setInsertionPointToStart(thenBlock);
    SmallVector<Value, 4> thenResults;
    SmallVector<Type, 4> resultTypes;
    for (auto ifOp : ifOps) {
      Block *thenBlock = ifOp.getThenBlock();
      for (Operation &op : thenBlock->without_terminator())
        rewriter.clone(op, mapping);
      auto yieldOp = cast<affine::AffineYieldOp>(thenBlock->getTerminator());
      for (auto result : yieldOp.getOperands()) {
        thenResults.push_back(mapping.lookupOrDefault(result));
        resultTypes.push_back(result.getType());
      }
    }
    affine::AffineYieldOp::create(rewriter, loc, thenResults);

    Block *elseBlock = rewriter.createBlock(parentOp.getBody());
    rewriter.setInsertionPointToStart(elseBlock);
    SmallVector<Value, 4> elseResults;
    for (auto ifOp : ifOps) {
      Block *elseBlock = ifOp.getElseBlock();
      for (Operation &op : elseBlock->without_terminator())
        rewriter.clone(op, mapping);
      auto yieldOp = cast<affine::AffineYieldOp>(elseBlock->getTerminator());
      for (auto result : yieldOp.getOperands())
        elseResults.push_back(mapping.lookupOrDefault(result));
    }
    affine::AffineYieldOp::create(rewriter, loc, elseResults);
    assert(thenResults.size() == elseResults.size() &&
           "expected same number of results from then and else blocks");

    rewriter.setInsertionPoint(firstIf);
    auto mergedIf = affine::AffineIfOp::create(rewriter, loc, resultTypes,
                                               firstIf.getCondition(),
                                               firstIf.getOperands(), true);

    rewriter.mergeBlocks(thenBlock, mergedIf.getThenBlock());
    rewriter.mergeBlocks(elseBlock, mergedIf.getElseBlock());

    unsigned resultIdx = 0;
    for (auto ifOp : ifOps) {
      for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
        rewriter.replaceAllUsesWith(result,
                                    mergedIf.getResult(resultIdx + idx));
      }
      resultIdx += ifOp.getNumResults();
    }
    for (auto ifOp : ifOps)
      rewriter.eraseOp(ifOp);
  }

public:
  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto adjacentIfLists = collectAdjacentIfLists(forOp);
    for (auto &list : adjacentIfLists) {
      mergeAdjacentIfs(list, forOp, rewriter);
    }
    return success();
  }
};

// Sink a pure value-producing slice that feeds a single store out of an
// affine.if's results and into both of its branches.
struct MergeStoreIntoAffineIfPattern : OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern<affine::AffineIfOp>::OpRewritePattern;

private:
  static FailureOr<affine::AffineStoreOp>
  collectSingleStoreSlice(affine::AffineIfOp ifOp,
                          SmallVectorImpl<Operation *> &intermediates) {
    llvm::SmallDenseSet<Operation *, 4> visited;
    SmallVector<Operation *, 8> worklist;
    for (Value v : ifOp.getResults()) {
      llvm::append_range(worklist, v.getUsers());
    }
    affine::AffineStoreOp foundStore = nullptr;

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (!visited.insert(op).second)
        continue;
      if (op->getBlock() != ifOp->getBlock())
        return failure(); // require in the same block

      if (foundStore && foundStore->isBeforeInBlock(op))
        return failure(); // reject if any uses after store

      if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
        if (foundStore && foundStore != store)
          return failure(); // multiple sinks, ignore
        foundStore = store;
        intermediates.push_back(op);
        continue;
      }

      if (!isMemoryEffectFree(op))
        return failure(); // require no mem effects

      intermediates.push_back(op);
      for (Value res : op->getResults()) {
        llvm::append_range(worklist, res.getUsers());
      }
    }
    llvm::sort(intermediates, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    if (!foundStore)
      return failure(); // require at least one store
    return foundStore;
  }

  static bool
  hasOnlyDominatingExternalOperands(affine::AffineIfOp ifOp,
                                    ArrayRef<Operation *> intermediates) {
    llvm::SmallDenseSet<Operation *, 8> sliceOps(intermediates.begin(),
                                                 intermediates.end());
    DominanceInfo dom(ifOp->getParentOp());
    for (Operation *op : intermediates) {
      for (Value operand : op->getOperands()) {
        if (llvm::is_contained(ifOp.getResults(), operand))
          continue;
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || sliceOps.contains(defOp))
          continue;
        if (!dom.dominates(defOp, ifOp.getOperation()))
          return false;
      }
    }
    return true;
  }

  static void
  cloneIntermediatesIntoBranch(SmallVectorImpl<Operation *> &intermediates,
                               Block *src, Block *dst, affine::AffineIfOp ifOp,
                               PatternRewriter &rewriter) {
    IRMapping mapping;
    rewriter.setInsertionPointToStart(dst);
    for (Operation &op : src->without_terminator())
      rewriter.clone(op, mapping);
    // Map each if result to the yielded value its branch produced, so the
    // intermediates clone against the branch's own copy.
    auto yield = cast<affine::AffineYieldOp>(src->getTerminator());
    for (OpOperand &v : yield->getOpOperands())
      mapping.map(ifOp->getResult(v.getOperandNumber()),
                  mapping.lookupOrDefault(v.get()));
    for (Operation *op : intermediates)
      rewriter.clone(*op, mapping);
  }

public:
  LogicalResult matchAndRewrite(affine::AffineIfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp.getNumResults() == 0)
      return failure();

    SmallVector<Operation *, 4> intermediates;
    auto foundStoreOr = collectSingleStoreSlice(ifOp, intermediates);
    if (failed(foundStoreOr))
      return failure();
    if (!hasOnlyDominatingExternalOperands(ifOp, intermediates))
      return failure();

    rewriter.setInsertionPoint(ifOp);
    auto newIf = affine::AffineIfOp::create(
        rewriter, ifOp->getLoc(), ifOp.getCondition(), ifOp.getOperands(),
        /*withElseRegion=*/true);

    cloneIntermediatesIntoBranch(intermediates, ifOp.getThenBlock(),
                                 newIf.getThenBlock(), ifOp, rewriter);
    cloneIntermediatesIntoBranch(intermediates, ifOp.getElseBlock(),
                                 newIf.getElseBlock(), ifOp, rewriter);

    for (Operation *op : llvm::reverse(intermediates))
      rewriter.eraseOp(op);
    rewriter.eraseOp(ifOp);

    return success();
  }
};

struct ReuseCleanupPass
    : public allo::impl::ReuseCleanupPassBase<ReuseCleanupPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // No-op unless this module actually contains reuse_at output.
    bool hasReuse = false;
    module.walk([&](Operation *op) {
      if (op->hasAttr(kReuseMaintenanceAttr)) {
        hasReuse = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasReuse)
      return;

    // CSE + canonicalize: removes the duplicate/identity affine.applys and the
    // now-dead original index computations.
    IRRewriter rewriter(context);
    DominanceInfo dom(module);
    eliminateCommonSubExpressions(rewriter, dom, module);

    {
      RewritePatternSet patterns(context);
      for (Dialect *dialect : context->getLoadedDialects())
        dialect->getCanonicalizationPatterns(patterns);
      for (RegisteredOperationName op : context->getRegisteredOperations())
        op.getCanonicalizationPatterns(patterns, context);
      (void)applyPatternsGreedily(module, std::move(patterns));
    }
    {
      // Merge the per-access conditional loads that share a condition.
      RewritePatternSet patterns(context);
      patterns.add<MergeSameAffineIfsPattern>(context);
      (void)applyPatternsGreedily(module, std::move(patterns));
    }
    {
      // Sink pure tails (e.g. the store of the reused value) into the branches.
      RewritePatternSet patterns(context);
      patterns.add<MergeStoreIntoAffineIfPattern>(context);
      (void)applyPatternsGreedily(module, std::move(patterns));
    }

    // The maintenance markers are an internal detail; drop them from the IR.
    module.walk([](Operation *op) { op->removeAttr(kReuseMaintenanceAttr); });
  }
};
} // namespace
