/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/InitAllPasses.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "allo/Conversion/Passes.h"
#include "allo/Transforms/Passes.h"

void mlir::allo::registerAllPasses() {
  mlir::registerTransformsPasses();

  // Conversion passes
  registerArithToLLVMConversionPass();
  registerConvertBufferizationToMemRefPass();
  registerConvertControlFlowToLLVMPass();
  registerConvertFuncToLLVMPass();
  registerConvertIndexToLLVMPass();
  registerConvertMathToLLVMPass();
  registerConvertLinalgToStandardPass();
  registerConvertTensorToLinalgPass();
  registerConvertToLLVMPass();
  registerFinalizeMemRefToLLVMConversionPass();
  registerLowerAffinePass();
  registerReconcileUnrealizedCastsPass();
  registerSCFToControlFlowPass();
  registerTosaToArithPass();
  registerTosaToLinalg();
  registerTosaToLinalgNamed();
  registerTosaToTensorPass();

  allo::registerConversionPasses();
  allo::registerTransformsPasses();
  allo::registerAlloLLVMLoweringPipeline();

  // Dialect passes
  affine::registerAffinePasses();
  arith::registerArithPasses();
  bufferization::registerBufferizationPasses();
  func::registerFuncPasses();
  registerLinalgPasses();
  LLVM::registerLLVMPasses();
  LLVM::registerTargetLLVMIRTransformsPasses();
  math::registerMathPasses();
  memref::registerMemRefPasses();
  registerSCFPasses();
  tensor::registerTensorPasses();
  tosa::registerTosaPasses();
  transform::registerTransformPasses();
  vector::registerVectorPasses();

  // Dialect pipelines
  bufferization::registerBufferizationPipelines();
  tosa::registerTosaToLinalgPipelines();
}
