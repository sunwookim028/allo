#!/bin/bash
# branch only: M-R1 reproduce --all (Verilator), then (after the branch pytest) reproduce.sh --no-cosim and mlp-cosim
WT=/work/shared/users/phd/sk3463/scratch/wt-reg6 O=/work/shared/users/phd/sk3463/scratch/sync6_out; L=$O/gates_branch
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate allo >/dev/null 2>&1
export OMP_NUM_THREADS=8 TMPDIR=/tmp/claude-1772902/sync6/tmpx; mkdir -p $TMPDIR
PY=$CONDA_PREFIX/bin/python
run() { n=$1; shift; s=$(date +%s); ( "$@" ) > $L/$n.log 2>&1; echo "$n EXIT $? $(( $(date +%s)-s ))s" | tee -a $L/summary.txt; }
cd -P $WT
case "$1" in
rtl) ( export PATH=/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH VERILATOR=/work/shared/users/phd/sk3463/tools/verilator/bin/verilator
       source /opt/rh/gcc-toolset-13/enable; run rtl_reproduce_all examples/minitpu/rtl/reproduce.sh --all ) ;;
late) run tinytpu_reproduce_nocosim env PYTHONPATH=$WT examples/tinytpu/reproduce.sh --no-cosim
      cd -P $WT/examples/tinytpu; run mlp_cosim env PYTHONPATH=$WT make mlp-cosim MODEL=mlp_small PYTHON=$PY ;;
esac
