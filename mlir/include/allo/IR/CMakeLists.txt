# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS AlloOps.td)
mlir_tablegen(AlloOps.h.inc -gen-op-decls)
mlir_tablegen(AlloOps.cpp.inc -gen-op-defs)
mlir_tablegen(AlloOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AlloOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(AlloDialect.h.inc -gen-dialect-decls -dialect=allo)
mlir_tablegen(AlloDialect.cpp.inc -gen-dialect-defs -dialect=allo)
add_public_tablegen_target(MLIRAlloOpsIncGen)

set(LLVM_TARGET_DEFINITIONS AlloTypes.td)
mlir_tablegen(AlloTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AlloTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRAlloTypesIncGen)

set(LLVM_TARGET_DEFINITIONS AlloAttrs.td)
mlir_tablegen(AlloAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AlloAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAlloAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS AlloAttrs.td)
mlir_tablegen(AlloEnums.h.inc -gen-enum-decls)
mlir_tablegen(AlloEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAlloEnumsIncGen)

# ACT ISA operations and their interfaces remain a compiler extension of the
# common Allo dialect.  Keep their generated interfaces distinct from the
# scheduling/datapath interfaces declared by AlloOps.td.
set(LLVM_TARGET_DEFINITIONS AlloISAOps.td)
mlir_tablegen(AlloISAOps.h.inc -gen-op-decls)
mlir_tablegen(AlloISAOps.cpp.inc -gen-op-defs)
mlir_tablegen(AlloISAOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AlloISAOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAlloISAOpsIncGen)

set(LLVM_TARGET_DEFINITIONS AlloISATypes.td)
mlir_tablegen(AlloISATypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(AlloISATypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRAlloISATypesIncGen)

# used to generate all the above targets at once
add_custom_target(MLIRAlloIncGen
    DEPENDS
    MLIRAlloOpsIncGen
    MLIRAlloTypesIncGen
    MLIRAlloAttrsIncGen
    MLIRAlloEnumsIncGen
    MLIRAlloISAOpsIncGen
    MLIRAlloISATypesIncGen
)

add_mlir_doc(AlloDialect AlloDialect Allo/ -gen-dialect-doc)
add_mlir_doc(AlloOps AlloOps Allo/ -gen-op-doc)
