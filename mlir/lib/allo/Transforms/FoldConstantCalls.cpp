/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Transforms/Passes.h"

#include "allo/IR/AlloOps.h"
#include "allo/Support/AliasAnalysis.h" // alloAliasAnalysis
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::allo {
#define GEN_PASS_DEF_FOLDCONSTANTCALLSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

static FrozenRewritePatternSet buildCanonicalizers(MLIRContext *context) {
  RewritePatternSet patterns(context);
  for (Dialect *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, context);
  return FrozenRewritePatternSet(std::move(patterns));
}

namespace {
// Lower "lazy consteval" kernels: a `@consteval(lazy=True)` function enters the
// IR as an `allo.kernel` tagged `allo.lazy` and called via `allo.invoke`.
struct FoldConstantCallsPass
    : public allo::impl::FoldConstantCallsPassBase<FoldConstantCallsPass> {
  using Base::Base;

  void runOnOperation() override;

private:
  // Evaluate `callee`'s body with the given constant arguments. On success
  // fills `results` with one constant per kernel result and returns true.
  bool evaluate(IRRewriter &rewriter, KernelOp callee,
                ArrayRef<TypedAttr> constArgs,
                SmallVectorImpl<TypedAttr> &results);

  ModuleOp module;
  FrozenRewritePatternSet canonicalizers;
  unsigned tmpCounter = 0;
};
} // namespace

bool FoldConstantCallsPass::evaluate(IRRewriter &rewriter, KernelOp callee,
                                     ArrayRef<TypedAttr> constArgs,
                                     SmallVectorImpl<TypedAttr> &results) {
  // Evaluate on a throwaway `func.func` clone of the kernel body.
  OpBuilder::InsertionGuard g(rewriter);

  std::string name =
      (callee.getSymName() + ".__lazy_eval_" + Twine(tmpCounter++)).str();
  rewriter.setInsertionPointToEnd(module.getBody());
  auto fn = func::FuncOp::create(rewriter, callee.getLoc(), name,
                                 callee.getFunctionType());
  fn.setPrivate();
  IRMapping mapping;
  callee.getBody().cloneInto(&fn.getBody(), mapping);
  fn.walk([&](ReturnOp ret) {
    rewriter.setInsertionPoint(ret);
    func::ReturnOp::create(rewriter, ret.getLoc(), ret.getOperands());
    ret.erase();
  });

  Block &entry = fn.getBody().front();
  rewriter.setInsertionPointToStart(&entry);
  for (auto [arg, attr] : llvm::zip_equal(entry.getArguments(), constArgs)) {
    auto cst = arith::ConstantOp::create(rewriter, fn.getLoc(), attr);
    arg.replaceAllUsesWith(cst);
  }

  // Fixpoint: simplify the body, then fully unroll one constant-trip loop
  // (affine.for or scf.for, outermost first). Each unroll exposes the next
  // round of folds.
  bool reduced = true;
  constexpr unsigned kMaxRounds = 32;
  for (unsigned round = 0; round < kMaxRounds; ++round) {
    (void)applyPatternsGreedily(fn, canonicalizers);
    {
      DominanceInfo dom(fn);
      PostDominanceInfo pdom(fn);
      AliasAnalysis aa = alloAliasAnalysis(fn);
      affine::affineScalarReplace(fn, dom, pdom, aa);
    }
    {
      DominanceInfo dom(fn);
      eliminateCommonSubExpressions(rewriter, dom, fn);
    }

    Operation *loopOp = nullptr;
    fn.walk([&](Operation *op) {
      if (isa<affine::AffineForOp, scf::ForOp>(op)) {
        loopOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!loopOp)
      break;
    LogicalResult unrolled =
        isa<affine::AffineForOp>(loopOp)
            ? affine::loopUnrollFull(cast<affine::AffineForOp>(loopOp))
            : mlir::loopUnrollFull(cast<scf::ForOp>(loopOp));
    if (failed(unrolled)) {
      reduced = false; // non-constant trip count: cannot fully evaluate
      break;
    }
  }

  bool ok = false;
  if (reduced) {
    auto ret = cast<func::ReturnOp>(fn.getBody().front().getTerminator());
    ok = true;
    for (Value operand : ret.getOperands()) {
      Attribute attr;
      auto typed = matchPattern(operand, m_Constant(&attr))
                       ? dyn_cast<TypedAttr>(attr)
                       : nullptr;
      if (!typed) {
        ok = false;
        break;
      }
      results.push_back(typed);
    }
  }
  fn.erase();
  return ok;
}

void FoldConstantCallsPass::runOnOperation() {
  module = getOperation();
  MLIRContext *context = &getContext();
  canonicalizers = buildCanonicalizers(context);
  SymbolTableCollection symbols;

  // Collect the lazy-consteval kernels via the `allo.lazy` marker.
  DenseSet<Operation *> lazyKernels;
  for (KernelOp kernel : module.getOps<KernelOp>()) {
    if (!kernel->hasAttr(kAlloLazyAttr))
      continue;
    // The frontend cannot express a lazy consteval kernel with a non-identity
    // mapping, so it is never an SPMD grid.
    assert(
        llvm::all_of(kernel.getMapping(), [](int32_t m) { return m == 1; }) &&
        "lazy consteval kernel must have identity mapping");
    lazyKernels.insert(kernel);
  }
  if (lazyKernels.empty())
    return;

  SmallVector<InvokeOp> invokes;
  module.walk([&](InvokeOp invoke) { invokes.push_back(invoke); });

  // (callee + constant args) -> constant results, memoized.
  llvm::StringMap<SmallVector<TypedAttr>> cache;
  llvm::StringSet<> failed;
  IRRewriter rewriter(context);
  bool failedAny = false;

  for (InvokeOp invoke : invokes) {
    auto callee = symbols.lookupNearestSymbolFrom<KernelOp>(
        invoke, invoke.getCalleeAttr());
    if (!callee || !lazyKernels.contains(callee))
      continue;

    // A lazy consteval is an explicit request to fold the call away, so every
    // argument must be a compile-time constant.
    SmallVector<TypedAttr> constArgs;
    SmallString<128> key;
    llvm::raw_svector_ostream keyOS(key);
    keyOS << invoke.getCallee();
    bool allConst = true;
    for (Value operand : invoke.getArgOperands()) {
      Attribute attr;
      auto typed = matchPattern(operand, m_Constant(&attr))
                       ? dyn_cast<TypedAttr>(attr)
                       : nullptr;
      if (!typed) {
        allConst = false;
        break;
      }
      constArgs.push_back(typed);
      keyOS << '#';
      typed.print(keyOS); // print the value: a bare `<< typed` does not
    }
    if (!allConst) {
      invoke.emitError("lazy consteval kernel '")
          << callee.getSymName()
          << "' must be invoked with compile-time-constant arguments";
      failedAny = true;
      continue;
    }

    if (failed.contains(keyOS.str())) {
      invoke.emitError("could not evaluate lazy consteval kernel '")
          << callee.getSymName() << "' to a compile-time constant";
      failedAny = true;
      continue;
    }

    IRRewriter rewriter(context);
    SmallVector<TypedAttr> results;
    auto it = cache.find(keyOS.str());
    if (it != cache.end()) {
      results.assign(it->second.begin(), it->second.end());
    } else if (evaluate(rewriter, callee, constArgs, results)) {
      cache.insert({keyOS.str(), results});
    } else {
      failed.insert(keyOS.str());
      invoke.emitError("could not evaluate lazy consteval kernel '")
          << callee.getSymName() << "' to a compile-time constant";
      failedAny = true;
      continue;
    }

    assert(results.size() == invoke.getNumResults() &&
           llvm::all_of(llvm::zip_equal(invoke.getResults(), results),
                        [](auto pair) {
                          return std::get<0>(pair).getType() ==
                                 std::get<1>(pair).getType();
                        }) &&
           "evaluated results must match the invoke result signature");

    rewriter.setInsertionPoint(invoke);
    SmallVector<Value> newResults;
    for (TypedAttr attr : results)
      newResults.push_back(
          arith::ConstantOp::create(rewriter, invoke.getLoc(), attr)
              .getResult());
    rewriter.replaceOp(invoke, newResults);
  }

  // Delete the lazy kernels that are now unused.
  for (Operation *op : lazyKernels) {
    auto kernel = cast<KernelOp>(op);
    if (kernel.symbolKnownUseEmpty(module))
      kernel.erase();
  }

  if (failedAny)
    signalPassFailure();
}
