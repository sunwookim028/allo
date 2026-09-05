# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloMicroarch
    BindingPolicy.cpp
    Datapath.cpp
    DatapathBuilder.cpp
    Memory.cpp
    ControlEmitter.cpp
    DatapathEmitter.cpp
    MemoryEmitter.cpp
    HWEmitter.cpp
    EmitDriver.cpp
    Primitives.cpp
    Naming.cpp
    Report.cpp
    Reservation.cpp
    Interface.cpp
    Verification.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAllo
    MLIRAlloScheduling
    MLIRAffineDialect
    MLIRAffineAnalysis
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRIR
    CIRCTHW
    CIRCTComb
    CIRCTSV
    CIRCTSeq
    CIRCTSupport
)
