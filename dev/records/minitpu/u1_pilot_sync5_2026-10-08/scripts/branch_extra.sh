#!/bin/bash
# branch only: reproduce.sh --no-cosim, ip_integration pytest, M-R1 reproduce --all, mlp-cosim
WT=/work/shared/users/phd/sk3463/scratch/wt-reg5 O=/work/shared/users/phd/sk3463/scratch/sync5_out; L=$O/gates_branch
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate allo >/dev/null 2>&1
export OMP_NUM_THREADS=8 TMPDIR=/tmp/claude-1772902/sync5/tmpx; mkdir -p $TMPDIR
PY=$CONDA_PREFIX/bin/python
run() { n=$1; shift; s=$(date +%s); ( "$@" ) > $L/$n.log 2>&1; echo "$n EXIT $? $(( $(date +%s)-s ))s" | tee -a $L/summary.txt; }
cd -P $WT; run ip_integration env PYTHONPATH=$WT $PY -m pytest -p no:cacheprovider -q tests/ip_integration tests/utils/test_backend_utils.py -k "not csynth and not lib_gemm"
( export PATH=/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH; source /opt/rh/gcc-toolset-13/enable
  run rtl_reproduce_all examples/minitpu/rtl/reproduce.sh --all )
cd -P $WT/examples/tinytpu; run mlp_cosim env PYTHONPATH=$WT make mlp-cosim MODEL=mlp_small PYTHON=$PY
echo "EXTRA DONE $(date -Is)" >> $L/summary.txt
