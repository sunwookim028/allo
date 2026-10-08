#!/bin/bash
# branch only: the U4 / core_fixes_4 check verdicts on simulator + systemc (env-zhang21.sh, conda python by path)
WT=/work/shared/users/phd/sk3463/scratch/wt-reg6 O=/work/shared/users/phd/sk3463/scratch/sync6_out; L=$O/gates_branch
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate allo >/dev/null 2>&1
PY=$CONDA_PREFIX/bin/python; export OMP_NUM_THREADS=8
source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export TMPDIR=/tmp/claude-1772902/sync6/tmpc; mkdir -p $TMPDIR
cd -P $WT; export PYTHONPATH=$(pwd -P)
run() { n=$1; shift; s=$(date +%s); ( "$@" ) > $L/$n.log 2>&1; echo "$n EXIT $? $(( $(date +%s)-s ))s" | tee -a $L/summary.txt; }
B="--backend simulator --backend systemc"
run c_sequencer $PY -m examples.minitpu.harness.check sequencer $B --project $O/prjc_sequencer
run c_seq_issue $PY -m examples.minitpu.harness.check seq_issue $B --project $O/prjc_seq_issue
run c_fetch_iram $PY -m examples.minitpu.harness.check fetch --inst iram $B --project $O/prjc_fetch
for i in core o1 o4 bw; do
  run c_dma_$i $PY -m examples.minitpu.harness.check dma --inst $i --variant bits --variant bits_reset $B --project $O/prjc_dma_$i
done
for i in core o1 o4; do
  run c_dmavmem_$i $PY -m examples.minitpu.harness.check dma_vmem --inst $i --variant d12_reset_r256 $B --project $O/prjc_dmavmem_$i
done
echo "CHECKS DONE $(date -Is)" >> $L/summary.txt
