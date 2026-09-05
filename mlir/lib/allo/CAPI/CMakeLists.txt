# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_public_c_api_library(MLIRAlloCAPI
    Registration.cpp
    Passes.cpp
    Schedule.cpp
    IRUtils.cpp
    AlloTypes.cpp
    AlloAttrs.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRAlloRegisterEverything
)
