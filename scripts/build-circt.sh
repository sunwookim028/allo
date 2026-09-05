#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

CIRCT_DIR=$(realpath "${1:-externals/circt}")
LLVM_DIR=$(realpath "${2:-externals/llvm-project/build}")
BUILD_TYPE=${3:-"Release"}
CC=${4:-"clang"}
CXX=${5:-"clang++"}
EXTRA_ARGS=("${@:6}")
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}

mkdir -p "$CIRCT_DIR/build"

cd "$CIRCT_DIR"
# OR-Tools 9.5 fetches deps (zlib among them) that still declare
# cmake_minimum_required(VERSION <3.5), which CMake >= 4 refuses outright. The
# shim applies to every nested configure; without it ortools never installs and
# the Allo build then fails with "No OR-Tools cmake package was found".
export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"
./utils/get-or-tools.sh

cd "$CIRCT_DIR/build"
cmake -G Ninja ../ \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir" \
  -DLLVM_USE_LINKER=lld \
  -DCIRCT_INCLUDE_TESTS=OFF \
  -DCIRCT_INCLUDE_DOCS=OFF \
  -DCIRCT_INCLUDE_INTEGRATION_TESTS=OFF \
  -DCIRCT_BINDINGS_PYTHON_ENABLED=OFF \
  "${EXTRA_ARGS[@]}"

ninja -j "$BUILD_JOBS"
