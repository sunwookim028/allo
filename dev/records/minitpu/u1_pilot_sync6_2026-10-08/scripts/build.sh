#!/bin/bash
# usage: build.sh <worktree> <tag>   (sync5's build.sh, parameterised: both sides get their own mlir/build)
WT=$1 TAG=$2 O=/work/shared/users/phd/sk3463/scratch/sync6_out
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate allo
source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/tmp/claude-1772902/sync6/tmpb; mkdir -p $TMPDIR
mkdir -p $WT/mlir/build && cd $WT/mlir/build
s=$(date +%s)
cmake -G Ninja .. -DCMAKE_C_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/cc -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++ \
  -DCMAKE_MAKE_PROGRAM=$CONDA_PREFIX/bin/ninja \
  -DLLVM_DIR=/work/shared/common/llvm-project-main/build-rhel8/lib/cmake/llvm \
  -DMLIR_DIR=/work/shared/common/llvm-project-main/build-rhel8/lib/cmake/mlir \
  -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python > $O/cmake_$TAG.log 2>&1 || { echo "CMAKE FAIL $TAG"; exit 1; }
nice ninja -j16 > $O/build_$TAG.log 2>&1
echo "BUILD $TAG EXIT $? $(( $(date +%s)-s ))s"
