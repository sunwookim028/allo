/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The "oracle" lowering: inline each allo.emit's instruction semantics into
// plain linalg/tensor/arith so the program can be lowered to LLVM and executed
// for differential testing.
//
// Buffers become mutable memref.globals. Unlike the experimental act oracle,
// each emit is self-contained: it freshly loads each buffer it touches from its
// global, reads operand slices, inlines the compute region, and writes the
// result slice back to the global via materialize_in_destination. No buffer
// state is threaded as a tensor SSA value across emits, so emits nested in
// scf.for / scf.if compose naturally (control-flow-safe).

#include "allo/Conversion/Passes.h"
#include "allo/IR/AlloOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "lower-instructions"

namespace mlir::allo {
#define GEN_PASS_DEF_LOWERINSTRUCTIONSPASS
#include "allo/Conversion/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

//===----------------------------------------------------------------------===//
// allo.buffer -> memref.global
//===----------------------------------------------------------------------===//

namespace {
struct ConvertDeclareBufferOpPattern
    : public OpConversionPattern<DeclareBufferOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DeclareBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    BufferTypeInterface bufferTy = op.getBufferType();
    // A buffer is its address space times its slot, so the memref is exactly
    // extents ++ slot shape. (The old rule dropped a unit slot count, which the
    // access patterns did not know about: a 1-slot vector buffer then verified
    // and failed to lower.)
    SmallVector<int64_t, 4> shape(op.getExtents());
    llvm::append_range(shape, bufferTy.getShape());
    auto memrefTy = MemRefType::get(shape, bufferTy.getElementType());
    memref::GlobalOp::create(rewriter, op.getLoc(), op.getSymNameAttr(),
                             StringAttr(), memrefTy, /*initValue=*/Attribute(),
                             /*constant=*/false, /*alignment=*/IntegerAttr());
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Compute-region inlining
//===----------------------------------------------------------------------===//

static bool isCastLikeOp(Operation *op) {
  return isa<arith::TruncFOp, arith::TruncIOp, arith::ExtFOp, arith::ExtSIOp,
             arith::ExtUIOp, arith::UIToFPOp, arith::FPToSIOp, arith::SIToFPOp,
             arith::FPToUIOp, arith::IndexCastOp, arith::BitcastOp>(op);
}

namespace {
// Clones the compute region's ops at the emit site, inferring concrete result
// types from the materialized operand slices.
struct SemanticsBuilder {
  SemanticsBuilder(RewriterBase &b, Location loc, IRMapping &mapping)
      : b(b), loc(loc), mapping(mapping) {}

  LogicalResult build(Block &block) const {
    for (Operation &op : block.without_terminator()) {
      SmallVector<Value, 4> newOperands;
      for (Value operand : op.getOperands())
        newOperands.push_back(mapping.lookupOrDefault(operand));
      if (isa<linalg::LinalgOp, linalg::SoftmaxOp, tensor::ExpandShapeOp>(op)) {
        if (failed(buildReifiedOp(cast<ReifyRankedShapedTypeOpInterface>(op),
                                  newOperands)))
          return failure();
        continue;
      }
      if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
        if (failed(buildCollapseShapeOp(collapseOp, newOperands)))
          return failure();
        continue;
      }
      // Ops whose result type is fixed by the op itself, not by its operands:
      // constants, and the splat that broadcasts a compute parameter (a scalar
      // immediate) to a tensor. Cloning them verbatim is exactly right.
      if (isa<arith::ConstantOp, tosa::ConstOp, tosa::ConstShapeOp,
              tensor::SplatOp>(op)) {
        b.clone(op, mapping);
        continue;
      }
      if (isa<arith::CmpFOp, arith::CmpIOp>(op)) {
        Type operandTy = op.getOperand(0).getType();
        Type resTy;
        if (auto shaped = dyn_cast<ShapedType>(operandTy))
          resTy = static_cast<Type>(shaped.clone(b.getI1Type()));
        else
          resTy = b.getI1Type();
        buildTrivialOp(&op, newOperands, resTy, op.getAttrs());
        continue;
      }
      if (isa<arith::ArithDialect, math::MathDialect>(op.getDialect())) {
        if (failed(buildArithLikeOp(&op, newOperands)))
          return failure();
        continue;
      }
      // Value-semantics ops (TOSA: add/mul/clamp/matmul/reshape/...) infer their
      // result types from the (re-materialized) operands + attributes — no DPS
      // init, no tensor.empty, and shape-changing ops (matmul, conv) work the
      // same as elementwise. This is the op-general inference path.
      if (auto shapeIface = dyn_cast<InferShapedTypeOpInterface>(op)) {
        if (failed(buildInferredOp(shapeIface, newOperands)))
          return failure();
        continue;
      }
      return op.emitError()
             << "unsupported operation in semantics block " << op.getName();
    }
    return success();
  }

private:
  RewriterBase &b;
  Location loc;
  IRMapping &mapping;

  void buildTrivialOp(Operation *op, ArrayRef<Value> newOperands,
                      ArrayRef<Type> newTypes,
                      ArrayRef<NamedAttribute> newAttrs) const {
    OperationState state(loc, op->getName());
    state.addOperands(newOperands);
    state.addTypes(newTypes);
    state.addAttributes(newAttrs);
    Operation *newOp = b.create(state);
    mapping.map(op, newOp);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op->getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
  }

  LogicalResult buildReifiedOp(ReifyRankedShapedTypeOpInterface op,
                               ArrayRef<Value> newOperands) const {
    for (auto [oldTy, newOperand] :
         llvm::zip(op->getOperandTypes(), newOperands)) {
      auto oldShaped = dyn_cast<ShapedType>(oldTy);
      auto newShaped = dyn_cast<ShapedType>(newOperand.getType());
      if (oldShaped && newShaped && oldShaped.getRank() != newShaped.getRank())
        return op.emitError()
               << "rank mismatch between original operand and new operand. "
               << "original: " << oldShaped << ", inferred: " << newShaped;
    }
    auto reifyOp = cast<ReifyRankedShapedTypeOpInterface>(b.clone(*op));
    b.modifyOpInPlace(reifyOp, [&]() { reifyOp->setOperands(newOperands); });
    ReifiedRankedShapedTypeDims shapes;
    if (failed(reifyOp.reifyResultShapes(b, shapes))) {
      b.eraseOp(reifyOp);
      return op.emitError() << "failed to reify result shapes";
    }
    b.eraseOp(reifyOp);

    SmallVector<Type, 4> resultTys;
    for (auto [shape, oldType] : llvm::zip(shapes, op->getResultTypes())) {
      auto oldTensor = dyn_cast<RankedTensorType>(oldType);
      if (!oldTensor)
        return op.emitError()
               << "expected ranked tensor result type, got " << oldType;
      SmallVector<int64_t, 4> dims;
      for (OpFoldResult dim : shape) {
        if (auto attr = dyn_cast<Attribute>(dim)) {
          dims.push_back(cast<IntegerAttr>(attr).getInt());
          continue;
        }
        dims.push_back(ShapedType::kDynamic);
      }
      resultTys.push_back(oldTensor.clone(dims));
    }

    OperationState state(loc, op->getName());
    state.addOperands(newOperands);
    state.addTypes(resultTys);
    state.addAttributes(op->getAttrs());
    if (op->getNumRegions() > 0)
      state.addRegion();
    Operation *newOp = b.create(state);
    mapping.map(op.getOperation(), newOp);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op->getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
    if (op->getNumRegions() == 0)
      return success();
    if (op->getNumRegions() > 1)
      return op.emitError()
             << "unexpected number of regions, expected at most 1";
    Region &newRegion = newOp->getRegion(0);
    Region &oldRegion = op->getRegion(0);
    b.cloneRegionBefore(oldRegion, newRegion, newRegion.end(), mapping);
    return success();
  }

  // Rebuild an op that infers its result types from operands+attrs via the
  // InferShapedTypeOpInterface (TOSA). Operands are the re-materialized slices,
  // so the inferred shapes reflect the actual emit-site sizes.
  LogicalResult buildInferredOp(InferShapedTypeOpInterface iface,
                                ArrayRef<Value> newOperands) const {
    Operation *op = iface.getOperation();
    SmallVector<ShapedTypeComponents> components;
    if (failed(iface.inferReturnTypeComponents(
            op->getContext(), op->getLoc(), ValueRange(newOperands),
            op->getDiscardableAttrDictionary(), op->getPropertiesStorage(),
            op->getRegions(), components)))
      return op->emitError() << "failed to infer result types";
    SmallVector<Type, 4> resultTys;
    for (auto [comp, oldTy] : llvm::zip_equal(components, op->getResultTypes())) {
      Type elt = cast<ShapedType>(oldTy).getElementType();
      if (comp.hasRank())
        resultTys.push_back(RankedTensorType::get(comp.getDims(), elt));
      else
        resultTys.push_back(oldTy);
    }
    buildTrivialOp(op, newOperands, resultTys, op->getAttrs());
    return success();
  }

  LogicalResult buildCollapseShapeOp(tensor::CollapseShapeOp op,
                                     ArrayRef<Value> newOperands) const {
    if (newOperands.size() != 1)
      return op.emitError()
             << "unexpected number of operands for tensor.collapse_shape";
    auto srcTy = dyn_cast<RankedTensorType>(newOperands.front().getType());
    if (!srcTy)
      return op.emitError() << "expected ranked tensor source type, got "
                            << newOperands.front().getType();
    RankedTensorType resultTy = tensor::CollapseShapeOp::inferCollapsedType(
        srcTy, op.getReassociationIndices());
    if (!resultTy)
      return op.emitError() << "failed to infer collapsed result type";
    buildTrivialOp(op, newOperands, resultTy, op->getAttrs());
    return success();
  }

  LogicalResult buildArithLikeOp(Operation *op,
                                 ArrayRef<Value> newOperands) const {
    if (op->getNumResults() != 1 ||
        (op->getNumOperands() != 1 && op->getNumOperands() != 2))
      return op->emitError()
             << "unexpected number of operands/results for arithmetic op";
    Type operandTy = newOperands.front().getType();
    Type resultTy = op->getResult(0).getType();
    if (isa<ShapedType>(operandTy)) {
      return op->emitError()
             << "use linalg operations for shaped types, arith operations "
                "should only be used for scalar types, got "
             << operandTy;
    }
    Type newTy = isCastLikeOp(op) ? resultTy : operandTy;
    buildTrivialOp(op, newOperands, newTy, op->getAttrs());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// allo.emit -> self-contained inlined semantics over memref globals
//===----------------------------------------------------------------------===//

namespace {
struct ConvertEmitOpPattern : public OpConversionPattern<EmitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(EmitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto declOp = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
        op, op.getInstructionAttr());
    if (!declOp)
      return op.emitError() << "referenced instruction '@"
                            << op.getInstruction() << "' not found";

    Block &accessBlock = declOp.getAccessBlock();
    Block &semBlock = declOp.getSemanticsBlock();
    rewriter.setInsertionPoint(op);

    // Ordered buffer names: sources then destinations (matches addr yield).
    SmallVector<StringAttr, 4> srcNames, dstNames, allNames;
    for (auto s : declOp.getSources().getAsRange<FlatSymbolRefAttr>())
      srcNames.push_back(s.getAttr());
    for (auto d : declOp.getDestinations().getAsRange<FlatSymbolRefAttr>())
      dstNames.push_back(d.getAttr());
    allNames.append(srcNames);
    allNames.append(dstNames);

    // Freshly load each distinct buffer this emit touches from its global.
    DenseMap<StringAttr, std::pair<Value, Value>>
        handles; // name -> {memref,tensor}
    for (StringAttr name : allNames) {
      if (handles.count(name))
        continue;
      auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          op, FlatSymbolRefAttr::get(name));
      if (!global)
        return op.emitError() << "buffer '@" << name.getValue()
                              << "' has no global declaration";
      auto memrefTy = global.getType();
      Value mem = memref::GetGlobalOp::create(rewriter, op.getLoc(), memrefTy,
                                              global.getSymName());
      auto tensorTy =
          RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
      Value tensor = bufferization::ToTensorOp::create(
          rewriter, op.getLoc(), tensorTy, mem, /*restrict=*/true,
          /*writable=*/true);
      handles[name] = {mem, tensor};
    }

    auto accessOps = getBufferAccessOps(rewriter, accessBlock, op, adaptor);

    // Materialize read slices for every buffer operand (incl. DPS-out dests).
    SmallVector<Value, 4> slices;
    for (auto [name, accessOp] : llvm::zip_equal(allNames, accessOps)) {
      auto sliceOr =
          accessOp.materialize(rewriter, op.getLoc(), handles[name].second);
      if (failed(sliceOr))
        return accessOp.emitError()
               << "failed to materialize buffer access for buffer '@"
               << name.getValue() << "'";
      slices.push_back(*sliceOr);
    }
    auto computeArgs = generateExtraComputeArgs(rewriter, op, declOp);
    slices.append(computeArgs);

    // Inline the compute region.
    IRMapping mapping;
    for (auto [arg, slice] : llvm::zip_equal(semBlock.getArguments(), slices))
      mapping.map(arg, slice);
    SemanticsBuilder builder(rewriter, op.getLoc(), mapping);
    if (failed(builder.build(semBlock)))
      return failure();

    // Write results back into the destination buffer tensors, then store each
    // distinct destination buffer back to its global.
    Operation *yieldOp = semBlock.getTerminator();
    SmallVector<Value, 4> valuesToWrite;
    for (Value operand : yieldOp->getOperands())
      valuesToWrite.push_back(mapping.lookupOrDefault(operand));
    unsigned nSrc = srcNames.size();
    for (unsigned i = 0; i < dstNames.size(); ++i) {
      auto updatedOr = accessOps[i + nSrc].materialize(
          rewriter, op.getLoc(), valuesToWrite[i], handles[dstNames[i]].second);
      if (failed(updatedOr))
        return accessOps[i + nSrc].emitError()
               << "failed to materialize write-back for buffer '@"
               << dstNames[i].getValue() << "'";
      handles[dstNames[i]].second = *updatedOr;
    }
    DenseSet<StringAttr> stored;
    for (StringAttr name : dstNames) {
      if (!stored.insert(name).second)
        continue;
      bufferization::MaterializeInDestinationOp::create(
          rewriter, op.getLoc(), TypeRange(), handles[name].second,
          handles[name].first, /*restrict=*/true, /*writable=*/true);
    }

    // Erase the cloned access-pattern ops (walking relayout chains) and the
    // emit.
    for (auto accessOp : accessOps) {
      SmallVector<Operation *, 4> toErase;
      Operation *curr = accessOp.getOperation();
      while (curr) {
        toErase.push_back(curr);
        if (auto relayout = dyn_cast<BufferRelayoutOpInterface>(curr))
          curr = relayout.getSource().getDefiningOp();
        else
          break;
      }
      for (auto *e : toErase)
        rewriter.eraseOp(e);
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  // Clone the addr region at the emit site (mapping addr params) and collect
  // the yielded buffer-access pattern ops.
  static SmallVector<BufferAccessOpInterface, 4>
  getBufferAccessOps(RewriterBase &b, Block &accessBlock, EmitOp op,
                     OpAdaptor adaptor) {
    SmallVector<BufferAccessOpInterface, 4> accessOps;
    IRMapping mapping;
    unsigned dynamicIdx = 0;
    auto dynamicParams = adaptor.getAddrParams();
    for (auto [blockArg, staticParam] : llvm::zip_equal(
             accessBlock.getArguments(), op.getStaticAddrParams())) {
      if (ShapedType::isDynamic(staticParam))
        mapping.map(blockArg, dynamicParams[dynamicIdx++]);
      else {
        auto cst = arith::ConstantIndexOp::create(b, op.getLoc(), staticParam);
        mapping.map(blockArg, cst);
      }
    }
    Operation *yieldClone = nullptr;
    for (Operation &nested : accessBlock.getOperations()) {
      Operation *cloned = b.clone(nested, mapping);
      if (isa<YieldOp>(cloned))
        yieldClone = cloned;
    }
    assert(yieldClone && "access block must have a yield terminator");
    for (auto operand : yieldClone->getOperands()) {
      auto accessOp = operand.getDefiningOp<BufferAccessOpInterface>();
      assert(accessOp &&
             "terminator operands should be defined by buffer access ops");
      accessOps.push_back(accessOp);
    }
    b.eraseOp(yieldClone);
    return accessOps;
  }

  static SmallVector<Value, 4>
  generateExtraComputeArgs(RewriterBase &b, EmitOp op, DefineOp declOp) {
    unsigned dynamicIdx = 0;
    SmallVector<Value, 4> computeArgs;
    auto dynamicParams = op.getComputeParams();
    for (auto [staticParam, blockArg] :
         llvm::zip(op.getStaticComputeParams(), declOp.getExtraComputeArgs())) {
      if (ShapedType::isDynamic(staticParam))
        computeArgs.push_back(dynamicParams[dynamicIdx++]);
      else {
        Value cst;
        if (isa<IntegerType>(blockArg.getType()))
          cst = arith::ConstantIntOp::create(b, op.getLoc(), blockArg.getType(),
                                             staticParam);
        else if (isa<IndexType>(blockArg.getType()))
          cst = arith::ConstantIndexOp::create(b, op.getLoc(), staticParam);
        else
          llvm_unreachable("unsupported compute parameter type");
        computeArgs.push_back(cst);
      }
    }
    return computeArgs;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerInstructionsPass
    : public allo::impl::LowerInstructionsPassBase<LowerInstructionsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Step 1: buffers -> memref.global
    {
      ConversionTarget target(*ctx);
      target.addIllegalOp<DeclareBufferOp>();
      target.addLegalOp<memref::GlobalOp>();
      RewritePatternSet patterns(ctx);
      patterns.add<ConvertDeclareBufferOpPattern>(ctx);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }

    // Step 2: emits -> inlined semantics
    {
      ConversionTarget target(*ctx);
      target.addLegalOp<memref::GetGlobalOp, memref::SubViewOp>();
      target.addLegalOp<bufferization::ToTensorOp,
                        bufferization::MaterializeInDestinationOp>();
      target.addLegalOp<tensor::ExtractSliceOp, tensor::InsertSliceOp,
                        tensor::ExpandShapeOp, tensor::CollapseShapeOp,
                        tensor::EmptyOp, tensor::DimOp, tensor::SplatOp>();
      target.addLegalOp<math::Exp2Op, math::Log2Op, math::ExpOp, math::LogOp,
                        math::AbsFOp, math::AbsIOp, math::FloorOp, math::SqrtOp,
                        math::RsqrtOp, math::CeilOp, math::TruncOp>();
      target.addLegalOp<linalg::GenericOp, linalg::YieldOp, linalg::MapOp,
                        linalg::ReduceOp, linalg::TransposeOp, linalg::FillOp,
                        linalg::ContractOp, linalg::SoftmaxOp, linalg::MatmulOp,
                        linalg::BatchMatmulOp, linalg::AddOp, linalg::SubOp,
                        linalg::MulOp, linalg::BroadcastOp>();
      target.addLegalDialect<arith::ArithDialect, tosa::TosaDialect>();
      target.addIllegalOp<EmitOp>();
      RewritePatternSet patterns(ctx);
      patterns.add<ConvertEmitOpPattern>(ctx);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }

    // Step 3: drop instruction definitions
    SmallVector<DefineOp> defineOps;
    getOperation()->walk([&](DefineOp op) { defineOps.push_back(op); });
    for (auto op : defineOps)
      op.erase();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    memref::MemRefDialect, tensor::TensorDialect,
                    tosa::TosaDialect>();
  }
};
} // namespace
