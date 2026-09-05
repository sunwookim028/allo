# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Locate the bundled LLVM/MLIR build. Honor an explicit -DLLVM_DIR/-DMLIR_DIR,
# else derive both from LLVM_BASE_DIR (the $LLVM_BASE_DIR env var, falling back
# to the in-tree externals build).
if(NOT DEFINED LLVM_DIR OR NOT DEFINED MLIR_DIR)
  set(LLVM_BASE_DIR "${PROJECT_SOURCE_DIR}/externals/llvm-project/build")
  if(NOT EXISTS "${LLVM_BASE_DIR}/lib/cmake/mlir")
    message(FATAL_ERROR "MLIR cmake package not found under '${LLVM_BASE_DIR}'. "
                        "Build externals/llvm-project or set -DLLVM_DIR/-DMLIR_DIR to the cmake package locations.")
  endif()
  set(LLVM_DIR "${LLVM_BASE_DIR}/lib/cmake/llvm")
  set(MLIR_DIR "${LLVM_BASE_DIR}/lib/cmake/mlir")
endif()

if(NOT DEFINED CIRCT_DIR)
  set(CIRCT_DIR "${PROJECT_SOURCE_DIR}/externals/circt/build/lib/cmake/circt")
  if(NOT EXISTS ${CIRCT_DIR})
    message(FATAL_ERROR "CIRCT not found under '${CIRCT_DIR}'. "
                        "Build externals/circt or set -DCIRCT_DIR / $CIRCT_DIR.")
  endif()
endif()

find_package(MLIR CONFIG REQUIRED)

find_package(ortools CONFIG QUIET)
if(NOT ortools_FOUND)
  message(FATAL_ERROR
    "No OR-Tools cmake package was found. Point CMAKE_PREFIX_PATH at the "
    "install prefix holding lib/cmake/ortools (its config also resolves "
    "absl/protobuf from there).")
endif()

find_package(CIRCT CONFIG REQUIRED)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")
message(STATUS "Using ortoolsConfig.cmake in: ${ortools_DIR}")
# A prebuilt OR-Tools ships shared libraries, found at runtime through the
# link path.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# Keep the Python wheel runnable without requiring callers to reproduce the
# build tree's LD_LIBRARY_PATH.  The CIRCT bootstrap installs OR-Tools as a
# shared library, while its remaining dependencies are linked into that DSO.
get_target_property(ALLO_ORTOOLS_LIBRARY
                    ortools::ortools IMPORTED_LOCATION_RELEASE)
if(NOT ALLO_ORTOOLS_LIBRARY)
  get_target_property(ALLO_ORTOOLS_LIBRARY ortools::ortools IMPORTED_LOCATION)
endif()
if(NOT ALLO_ORTOOLS_LIBRARY)
  message(FATAL_ERROR "Could not resolve the OR-Tools shared library")
endif()
install(FILES "${ALLO_ORTOOLS_LIBRARY}"
        DESTINATION "allo/_mlir/_mlir_libs"
        RENAME "libortools.so.9")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddCIRCT)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Third-party headers must be SYSTEM includes
include_directories(SYSTEM ${spdlog_SOURCE_DIR}/include)
get_target_property(ALLO_ORTOOLS_INCLUDE_DIRS
                    ortools::ortools INTERFACE_INCLUDE_DIRECTORIES)
include_directories(SYSTEM ${ALLO_ORTOOLS_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})

message(STATUS "Using Python binding")
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

option(ALLO_STRIP_RELEASE_LIBS "Strip symbol tables from Release shared libs" ON)
if(ALLO_STRIP_RELEASE_LIBS AND
   (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel"))
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-s")
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)
