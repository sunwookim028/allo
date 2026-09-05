#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

LLVM_DIR=${1:-"externals/llvm-project"}
BUILD_TYPE=${2:-"Release"}
CC=${3:-"clang"}
CXX=${4:-"clang++"}
EXTRA_ARGS=("${@:5}")
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}

mkdir -p "$LLVM_DIR/build"
cd "$LLVM_DIR/build"
cmake ../llvm \
  -G Ninja \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DMLIR_INCLUDE_DOCS=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(which python3)" \
  "${EXTRA_ARGS[@]}"

ninja -j "$BUILD_JOBS"
