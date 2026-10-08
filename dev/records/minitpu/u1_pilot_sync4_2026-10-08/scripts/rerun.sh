#!/bin/bash
# branch reruns: check.py via -m (script form shadowed by harness/calendar.py), reproduce.sh under the plain allo env
WT=/work/shared/users/phd/sk3463/scratch/wt-reg; TAG=$1; O=/work/shared/users/phd/sk3463/scratch/sync4_out; L=$O/gates_$TAG
[ "$TAG" = base ] && WT=/work/shared/users/phd/sk3463/allo
run() { n=$1; shift; s=$(date +%s); ( "$@" ) > $L/$n.log 2>&1; echo "$n RERUN EXIT $? $(( $(date +%s)-s ))s" | tee -a $L/summary.txt; }
( source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
  export TMPDIR=/tmp/claude-1772902/sync4_tmpr_$TAG; mkdir -p $TMPDIR; PY=$ALLO_PYTHON; cd -P $WT
  for u in bf16_add vpu_regfile sfu mxu_pe; do
    run m_check_$u $PY -m examples.minitpu.harness.check $u --backend systemc --project $O/prjm_${TAG}_$u
  done
  run m_check_vpu_word_array_narrow $PY -m examples.minitpu.harness.check vpu_word_array --backend systemc --inst narrow --project $O/prjm_${TAG}_vwa )
if [ "$TAG" = branch ]; then
 ( source $(conda info --base)/etc/profile.d/conda.sh; conda activate allo >/dev/null 2>&1; export OMP_NUM_THREADS=8
   export TMPDIR=/tmp/claude-1772902/sync4_tmpr_$TAG; cd -P $WT/examples/tinytpu; run reproduce_nocosim ./reproduce.sh --no-cosim )
fi
echo "RERUN DONE $(date -Is)" >> $L/summary.txt
