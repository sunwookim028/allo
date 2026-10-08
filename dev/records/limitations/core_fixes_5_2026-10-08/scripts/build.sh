#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate allo
source /opt/rh/gcc-toolset-13/enable
O=/work/shared/users/phd/sk3463/scratch/fix5
mkdir -p /work/shared/users/phd/sk3463/scratch/wt-fix5/mlir/build && cd /work/shared/users/phd/sk3463/scratch/wt-fix5/mlir/build
s=$(date +%s)
if [ ! -f build.ninja ]; then
cmake -G Ninja .. -DCMAKE_C_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/cc -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++ \
  -DCMAKE_MAKE_PROGRAM=$CONDA_PREFIX/bin/ninja \
  -DLLVM_DIR=/work/shared/common/llvm-project-main/build-rhel8/lib/cmake/llvm \
  -DMLIR_DIR=/work/shared/common/llvm-project-main/build-rhel8/lib/cmake/mlir \
  -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python > $O/cmake.log 2>&1 || { echo CMAKE FAIL; exit 1; }
fi
nice ninja -j16 > $O/build.log 2>&1
echo "BUILD EXIT $? $(( $(date +%s)-s ))s"
