# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_llvm_tool(allo-opt allo-opt.cpp)

llvm_update_compile_flags(allo-opt)

add_dependencies(allo-opt MLIRAlloIncGen)

target_link_libraries(allo-opt
    PRIVATE
    MLIRIR
    MLIROptLib
    MLIRAlloRegisterEverything
)
