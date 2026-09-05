# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloSupport
    AffineValueMapBuilder.cpp
    AffineRaising.cpp
    AliasAnalysis.cpp
    BitAnalysis.cpp
    TopologyGraph.cpp
    Logging.cpp
    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PRIVATE
    MLIRAllo
    MLIRIR
    MLIRAnalysis
    MLIRAffineDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    spdlog::spdlog_header_only
)

add_mlir_library(AlloDataflowRuntime SHARED
    DataflowRuntime.cpp

    LINK_LIBS PRIVATE
    marl
)

set_target_properties(AlloDataflowRuntime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/allo/_mlir/_mlir_libs"
)

install(TARGETS AlloDataflowRuntime
    LIBRARY DESTINATION "allo/_mlir/_mlir_libs")
