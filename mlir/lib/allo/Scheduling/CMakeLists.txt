# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(ALLO_SCHEDULING_LIBS
    CIRCTScheduling
    MLIRAllo
    MLIRAlloSupport
    MLIRAffineDialect
    MLIRAffineAnalysis
    MLIRAffineUtils
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRIR
    MLIRTransformUtils
    ortools::ortools
)
# OR-Tools' interface definitions shape its headers, and the object library
# compiling this source does not inherit them from the link interface.
set_source_files_properties(CPSATScheduler.cpp ModuloOracle.cpp PROPERTIES
    COMPILE_DEFINITIONS
    "$<TARGET_PROPERTY:ortools::ortools,INTERFACE_COMPILE_DEFINITIONS>")

add_mlir_library(MLIRAlloScheduling
    AddressModel.cpp
    CPSATScheduler.cpp
    ModuloOracle.cpp
    DependenceAnalysis.cpp
    Footprint.cpp
    LatencyModel.cpp
    MemoryAccess.cpp
    MemoryModel.cpp
    OperatorLibrary.cpp
    ProblemBuilder.cpp
    RegionGraph.cpp
    ScheduleModel.cpp
    Scheduler.cpp
    PreVerification.cpp
    SDC.cpp
    PostConversion.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    ${ALLO_SCHEDULING_LIBS}
)
