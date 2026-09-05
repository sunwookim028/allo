/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Conversion/Passes.h"
#include "allo/IR/AlloOps.h"
#include "allo/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::allo;

namespace {
struct StripSchedulingHintsPass
    : public PassWrapper<StripSchedulingHintsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripSchedulingHintsPass)
  StringRef getArgument() const final { return "allo-strip-scheduling-hints"; }

  void runOnOperation() override {
    // Collect first: erasing mid-walk invalidates the iteration.
    SmallVector<Operation *> hints;
    getOperation()->walk([&](Operation *op) {
      if (isa<AssumeNoDepOp, AssumeSSAOp, VolatileOp>(op))
        hints.push_back(op);
    });
    for (Operation *hint : hints)
      hint->erase();
  }
};
} // namespace

void allo::populateLowerToLLVMPipeline(OpPassManager &pm, bool enableTensor) {
  pm.addPass(std::make_unique<StripSchedulingHintsPass>());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createGridMappingPass());
  pm.addPass(createLowerDataflowPass());

  if (enableTensor) {
    pm.addPass(createConvertTensorToLinalgPass());
    bufferization::OneShotBufferizePassOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.bufferAlignment = 64;
    options.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    options.allowReturnAllocsFromLoops = false;
    pm.addPass(bufferization::createOneShotBufferizePass(options));
    pm.addPass(bufferization::createDropEquivalentBufferResultsPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    bufferization::buildBufferDeallocationPipeline(pm);
    pm.addPass(createConvertBufferizationToMemRefPass());
  }

  pm.addPass(createConvertAlloToFuncPass());
  // No global C-wrapper request: only the top kernel needs the C interface ABI,
  // marked with `llvm.emit_c_interface` and preserved by ConvertAlloToFunc. A
  // global request would prefix the dataflow runtime symbols `_mlir_ciface_`.
  auto &nestedPM = pm.nest<func::FuncOp>();
  nestedPM.addPass(createConvertLinalgToAffineLoopsPass());
  nestedPM.addPass(affine::createAffineScalarReplacementPass());
  nestedPM.addPass(createLoopInvariantCodeMotionPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createDataflowSpawnPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

namespace {
struct AlloLowerToLLVMPipelineOptions
    : public PassPipelineOptions<AlloLowerToLLVMPipelineOptions> {
  Option<bool> enableTensor{
      *this, "enable-tensor",
      llvm::cl::desc(
          "Run tensor->linalg + one-shot bufferization before lowering"),
      llvm::cl::init(true)};
};
} // namespace

void allo::registerAlloLLVMLoweringPipeline() {
  PassPipelineRegistration<AlloLowerToLLVMPipelineOptions>(
      "lower-to-llvm", "Lower allo/canonical-form IR to the LLVM dialect",
      [](OpPassManager &pm, const AlloLowerToLLVMPipelineOptions &opts) {
        populateLowerToLLVMPipeline(pm, opts.enableTensor);
      });
}
