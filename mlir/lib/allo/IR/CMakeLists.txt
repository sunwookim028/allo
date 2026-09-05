# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(MLIRAllo
    AlloOps.cpp
    AlloISATypes.cpp
    AlloISAOps.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRAffineDialect
)
