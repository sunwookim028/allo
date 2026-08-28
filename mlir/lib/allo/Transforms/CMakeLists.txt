# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloTransforms
    MaterializeTopology.cpp
    GridMapping.cpp
    DataflowSpawn.cpp
    ReuseCleanup.cpp
    FoldConstantCalls.cpp
    GenerateApintWrapper.cpp
    FoldIfStatements.cpp
    LoopCanonicalization.cpp
    DropTrivialFunc.cpp
    AssignBanks.cpp
    ResolveBanking.cpp
    ReconcileArrayDirectives.cpp
    OutlineLooseProcesses.cpp
    RaiseCountedWhile.cpp
    ElideDeadInit.cpp
    RaiseToAffine.cpp
    ScalarizeMemory.cpp
    HoistInvariantReads.cpp
    RaiseMemoryReductions.cpp
    RotateReductions.cpp
    TreeHeightReduction.cpp
    NarrowDemandedBits.cpp
    FloatToInt.cpp
    LegalizeArith.cpp
    ExpandRegionBounds.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAlloSupport
    MLIRAnalysis
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRArithTransforms
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRTransforms
    MLIRTransformUtils
    MLIRPass

    LINK_COMPONENTS
    Core
)
