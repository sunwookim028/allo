# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(IR)
add_subdirectory(CAPI)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Microarch)
add_subdirectory(Scheduling)
add_subdirectory(Transforms)
add_subdirectory(TransformOps)
add_subdirectory(Translation)

add_mlir_library(MLIRAlloRegisterEverything
  InitAllDialects.cpp
  InitAllPasses.cpp
  InitAllExtensions.cpp

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRVectorDialect
  MLIRTransformDialect
  CIRCTHW
  CIRCTComb
  CIRCTSeq

  MLIRArithToLLVM
  MLIRBufferizationToMemRef
  MLIRControlFlowToLLVM
  MLIRIndexToLLVM
  MLIRFuncToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRConvertToLLVMPass
  MLIRLinalgToStandard
  MLIRTensorToLinalg
  MLIRTosaToArith
  MLIRTosaToLinalg
  MLIRTosaToTensor
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  MLIRBufferizationPipelines
  MLIRReconcileUnrealizedCasts

  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  MLIRLinalgTransforms
  MLIRLLVMIRTransforms
  MLIRMathTransforms
  MLIRMemRefTransforms
  MLIRSCFTransforms
  MLIRTensorTransforms
  MLIRVectorTransforms
  MLIRTosaTransforms
  MLIRTargetLLVMIRTransforms
  MLIRTransformDialectTransforms

  MLIRBufferizationTransformOps
  MLIRAffineTransformOps
  MLIRFuncTransformOps
  MLIRLinalgTransformOps
  MLIRMemRefTransformOps
  MLIRSCFTransformOps
  MLIRTensorTransformOps
  MLIRVectorTransformOps
  MLIRTransformDialect
  MLIRTransformLoopExtension

  MLIRAllo
  MLIRAlloConversion
  MLIRAlloSupport
  MLIRAlloMicroarch
  MLIRAlloScheduling
  MLIRAlloTransforms
  MLIRAlloTransformOps
  MLIRAlloTranslation
)
