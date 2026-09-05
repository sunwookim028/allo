# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_llvm_tool(allo-lsp-server allo-lsp-server.cpp)

llvm_update_compile_flags(allo-lsp-server)

add_dependencies(allo-lsp-server MLIRAlloIncGen)

target_link_libraries(allo-lsp-server
    PRIVATE
    MLIRIR
    MLIRLspServerLib
    MLIRAlloRegisterEverything
)
