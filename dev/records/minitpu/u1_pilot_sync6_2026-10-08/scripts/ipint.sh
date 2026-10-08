#!/bin/bash
# usage: ipint.sh <worktree> <tag>. tests/ip_integration as its own invocation, plain allo env, VERILATOR set, gcc-toolset-13 (sync5 s.3)
WT=$1 TAG=$2 O=/work/shared/users/phd/sk3463/scratch/sync6_out; L=$O/gates_$TAG
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate allo >/dev/null 2>&1
export OMP_NUM_THREADS=8 TMPDIR=/tmp/claude-1772902/sync6/tmpi_$TAG; mkdir -p $TMPDIR
PY=$CONDA_PREFIX/bin/python
export PATH=/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH VERILATOR=/work/shared/users/phd/sk3463/tools/verilator/bin/verilator
source /opt/rh/gcc-toolset-13/enable
cd -P $WT; export PYTHONPATH=$(pwd -P)
s=$(date +%s); $PY -m pytest -p no:cacheprovider -q tests/ip_integration tests/utils/test_backend_utils.py -k "not csynth and not lib_gemm" --junitxml=$O/ipint_$TAG.xml > $L/ip_integration.log 2>&1
echo "ip_integration EXIT $? $(( $(date +%s)-s ))s" | tee -a $L/summary.txt
