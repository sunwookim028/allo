/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#include "allo/Conversion/Passes.h"

namespace mlir::allo {
#define GEN_PASS_DEF_CONVERTALLOTOFUNCPASS
#include "allo/Conversion/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

static bool isIdentityMapping(ArrayRef<int32_t> mapping) {
  return llvm::all_of(mapping, [](int32_t x) { return x == 1; });
}

static void copyInvokeAttrs(InvokeOp invoke, func::CallOp call) {
  if (auto argAttrs = invoke.getArgAttrsAttr())
    call.setArgAttrsAttr(argAttrs);
  if (auto resAttrs = invoke.getResAttrsAttr())
    call.setResAttrsAttr(resAttrs);
  // `func.call` has no `async` field, so the async bit rides a discardable attr
  // that survives canonicalize/cse; the dataflow lowering keys on it to
  // classify the call as a spawn.
  if (invoke.getAsync())
    call->setAttr(kAlloAsyncAttr, UnitAttr::get(call.getContext()));
}

namespace {
struct ConvertKernelToFunc : OpRewritePattern<KernelOp> {
  using OpRewritePattern<KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const override {
    if (!isIdentityMapping(op.getMapping()))
      return op->emitError() << "convert-allo-to-func requires identity "
                                "(specialized) mappings; "
                                "run grid-mapping before this pass to expand @"
                             << op.getSymName();
    rewriter.setInsertionPoint(op);
    auto fn = func::FuncOp::create(
        rewriter, op->getLoc(), op.getSymName(), op.getFunctionType(),
        op.getSymVisibilityAttr(), op.getArgAttrsAttr(), op.getResAttrsAttr());
    fn->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.inlineRegionBefore(op.getRegion(), fn.getBody(),
                                fn.getBody().begin());
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
struct ConvertInvokeToFunc : OpRewritePattern<InvokeOp> {
  using OpRewritePattern<InvokeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InvokeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto call = func::CallOp::create(rewriter, op->getLoc(), op.getCalleeAttr(),
                                     op->getResultTypes(), op->getOperands());
    copyInvokeAttrs(op, call);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};
} // namespace

namespace {
struct ConvertReturnToFunc : OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto ret =
        func::ReturnOp::create(rewriter, op->getLoc(), adaptor.getOperands());
    rewriter.replaceOp(op, ret);
    return success();
  }
};
} // namespace

namespace {
struct ConvertAlloToFuncPass
    : public allo::impl::ConvertAlloToFuncPassBase<ConvertAlloToFuncPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
    target.addIllegalOp<KernelOp, InvokeOp, ReturnOp, GetWorkerIdOp,
                        GetNumWorkersOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    patterns.add<ConvertKernelToFunc>(context);
    patterns.add<ConvertReturnToFunc>(context);
    patterns.add<ConvertInvokeToFunc>(context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }
};
} // namespace
