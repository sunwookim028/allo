#!/bin/bash
# usage: checks.sh <worktree> <logdir>: MiniTPU harness checks touched by the fixes (both backends)
WT=$1 L=$2; mkdir -p $L
cd $WT; source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export PYTHONPATH=$WT TMPDIR=/tmp/claude-1772902/fix5tmp OMP_NUM_THREADS=8
P=/work/shared/users/phd/sk3463/scratch/fix5/prj_checks; mkdir -p $P
run() { n=$1; shift; s=$(date +%s); "$@" > $L/$n.log 2>&1; echo "$n EXIT $? $(( $(date +%s)-s ))s $(grep -E '^(UNIT|CONTRACT)-' $L/$n.log | cut -c1-140 | tr '\n' ' ')" | tee -a $L/summary.txt; }
for b in simulator systemc; do
  for i in iram fq fq_a8; do
    run fetch_f1_${i}_$b $ALLO_PYTHON -m examples.minitpu.harness.check fetch --variant f1 --inst $i --backend $b --project $P/f1_${i}_$b
  done
  run fetch_f1_d12_$b $ALLO_PYTHON -m examples.minitpu.harness.check fetch --variant f1_d12 --backend $b --project $P/f1d12_$b
  run loop_ctrl_l1_d12_$b $ALLO_PYTHON -m examples.minitpu.harness.check loop_ctrl --variant l1_d12 --backend $b --project $P/l1d12_$b
done
