# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloSupport
    TopologyGraph.cpp
    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAllo
    MLIRIR
)

add_mlir_library(AlloDataflowRuntime SHARED
    DataflowRuntime.cpp

    LINK_LIBS PRIVATE
    marl
)

# PROJECT_SOURCE_DIR is the repo root (top-level project(Allo))
set_target_properties(AlloDataflowRuntime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/allo/_mlir/_mlir_libs"
)

install(TARGETS AlloDataflowRuntime
    LIBRARY DESTINATION "allo/_mlir/_mlir_libs")
