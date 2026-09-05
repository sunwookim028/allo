# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_conversion_library(MLIRAlloConversion
    ConvertAlloToFunc.cpp
    LowerInstructions.cpp
    LowerDataflow.cpp
    LowerHLMem.cpp
    LowerToLLVM.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    CIRCTLoopSchedule
    CIRCTScheduling
    CIRCTHW
    CIRCTSV
    CIRCTSeq
    MLIRIR
    MLIRAllo
    MLIRAlloSupport
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRBufferizationPipelines
    MLIRAffineToStandard
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRMathToLLVM
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRIndexToLLVM
    MLIRArithToLLVM
    MLIRTensorDialect
    MLIRTensorToLinalg
    MLIRTosaDialect
    MLIRTransformUtils
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRBufferizationToMemRef
    MLIRReconcileUnrealizedCasts
)
