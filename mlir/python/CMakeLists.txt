# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "allo")

include(AddMLIRPython)

# Vendor our own relocatable MLIR instance under the `allo._mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=allo._mlir.")

set(ALLO_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/allo")
set(ALLO_MLIR_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/mlir/lib/allo/Bindings")
set(ALLO_MLIR_PYTHON_PACKAGE_DIR "${PROJECT_SOURCE_DIR}/allo/_mlir")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(AlloMLIRPythonSources)

declare_mlir_python_sources(AlloMLIRPythonSources.Core
  ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT AlloMLIRPythonSources
  SOURCES
    schedule.py
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AlloMLIRPythonSources
  ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AlloOps.td
  SOURCES
    dialects/allo.py
  DIALECT_NAME allo
)

declare_mlir_dialect_extension_python_bindings(
  ADD_TO_PARENT AlloMLIRPythonSources
  ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AlloTransformOps.td
  SOURCES
    dialects/transform/allo.py
  DIALECT_NAME transform
  EXTENSION_NAME allo_transform
)

################################################################################
# Extension: the slim CAPI driver module `_allo`
################################################################################

declare_mlir_python_extension(AlloMLIRPythonSources.Extension
  MODULE_NAME _allo
  ADD_TO_PARENT AlloMLIRPythonSources
  ROOT_DIR "${ALLO_MLIR_PYTHON_SOURCE_DIR}"
  SOURCES
    AlloModule.cpp
    AlloTypes.cpp
    AlloAttrs.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPITransforms
    MLIRAlloCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Common CAPI aggregate (single shared MLIR instance)
################################################################################

set(_allo_declared_sources
  AlloMLIRPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.ExecutionEngine
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.affine
  MLIRPythonSources.Dialects.bufferization
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.complex
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.index
  MLIRPythonSources.Dialects.linalg
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.tosa
  MLIRPythonSources.Dialects.vector
  MLIRPythonSources.Dialects.transform
  MLIRPythonSources.Dialects.transform.extras
  MLIRPythonSources.Dialects.transform.interpreter
  MLIRPythonSources.Dialects.structured_transform
  MLIRPythonSources.Dialects.ub
)

add_mlir_python_common_capi_library(AlloMLIRAggregateCAPI
  INSTALL_COMPONENT AlloMLIRPythonModules
  INSTALL_DESTINATION "allo/_mlir/_mlir_libs"
  OUTPUT_DIRECTORY "${ALLO_MLIR_PYTHON_PACKAGE_DIR}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES ${_allo_declared_sources}
)

################################################################################
# Instantiate the package
################################################################################

add_mlir_python_modules(AlloMLIRPythonModules
  ROOT_PREFIX "${ALLO_MLIR_PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX "allo/_mlir"
  DECLARED_SOURCES ${_allo_declared_sources}
  COMMON_CAPI_LINK_LIBS
    AlloMLIRAggregateCAPI
)
