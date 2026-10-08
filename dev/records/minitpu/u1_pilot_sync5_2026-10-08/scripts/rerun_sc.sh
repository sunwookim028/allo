#!/bin/bash
# systemc_csim and eva need Catapult's SystemC (MGC_HOME): rerun under env-zhang21.sh, conda python by path (sync4's env for these)
WT=$1 TAG=$2 O=/work/shared/users/phd/sk3463/scratch/sync5_out; L=$O/gates_$TAG
source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export TMPDIR=/tmp/claude-1772902/sync5/tmpr_$TAG; mkdir -p $TMPDIR
PY=$ALLO_PYTHON; cd -P $WT; WT=$(pwd -P); export PYTHONPATH=$WT
run() { n=$1; shift; s=$(date +%s); ( "$@" ) > $L/$n.log 2>&1; echo "$n RERUN EXIT $? $(( $(date +%s)-s ))s" | tee -a $L/summary.txt; }
cd -P $WT/examples/tinytpu; run systemc_csim $PY systemc_csim.py 3
cd -P $WT/examples/eva; run eva $PY cosim_eva_systemc.py
cd -P $WT && git checkout -- examples/eva/generated 2>/dev/null; git clean -fdq -- examples/eva/generated 2>/dev/null
echo "RERUN DONE $(date -Is)" >> $L/summary.txt
