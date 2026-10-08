#!/bin/bash
# usage: gates.sh <worktree> <tag>   (sync5's gates.sh; systemc_csim/eva run directly under env-zhang21.sh, as sync5's rerun_sc.sh did;
#   run_u3e also by path where template/calendar.py is gone. Origin: sync4's gates.sh: plain allo env, python by path, PYTHONPATH=checkout;
#   check.py via -m under env-zhang21.sh with the conda python by path; stress also at TPU_MAXDIM=16 as reproduce.sh)
WT=$1 TAG=$2 O=/work/shared/users/phd/sk3463/scratch/sync6_out
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate allo >/dev/null 2>&1
export OMP_NUM_THREADS=8
export TMPDIR=/tmp/claude-1772902/sync6/tmpg_$TAG; mkdir -p $TMPDIR
PY=$CONDA_PREFIX/bin/python; L=$O/gates_$TAG; mkdir -p $L
cd -P $WT; WT=$(pwd -P); export PYTHONPATH=$WT
run() { n=$1; shift; s=$(date +%s); ( "$@" ) > $L/$n.log 2>&1; echo "$n EXIT $? $(( $(date +%s)-s ))s" | tee -a $L/summary.txt; }
echo "HEAD $(git -C $WT rev-parse HEAD) start $(date -Is)" >> $L/summary.txt
run allo_where $PY -c "import allo,os;print(os.path.realpath(allo.__file__))"
run hashes $PY -c "
import hashlib
from allo.dataflow import customize
from examples.tinytpu.microarch_isa import tinytpu_isa, schedule
for T in ('vhls','catapult','systemc'):
    s=customize(tinytpu_isa); schedule(s); t=str(s.build(target=T))
    open('$L/emit_'+T+'.txt','w').write(t)
    print('HASH',T,hashlib.sha256(t.encode()).hexdigest(),len(t))
"
cd -P $WT/examples/tinytpu
run gen_isa $PY gen_isa.py --check
run lift_units $PY lift_units.py --check
run bench_isa $PY bench_isa.py
run stress_isa $PY stress_isa.py
run stress_isa_md16 env TPU_MAXDIM=16 $PY stress_isa.py
run act_gate $PY act_compile.py --gate
run mutate_none $PY mutate.py none
( source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1; export PYTHONPATH=$WT TMPDIR; run systemc_csim $PY systemc_csim.py 3 )
cd -P $WT
run df_unit $PY tests/dataflow/test_df_unit.py
run region_stateful $PY -m pytest -p no:cacheprovider -q tests/dataflow/test_region_stateful.py
cd -P $WT/examples/eva
( source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1; export PYTHONPATH=$WT TMPDIR; run eva $PY cosim_eva_systemc.py )
cd -P $WT && git checkout -- examples/eva/generated 2>/dev/null; git clean -fdq -- examples/eva/generated 2>/dev/null
run minitpu_quick $PY examples/minitpu/run.py --quick
run u3e $PY -m examples.minitpu.template.run_u3e
[ -f $WT/examples/minitpu/template/calendar.py ] || run u3e_path $PY examples/minitpu/template/run_u3e.py
( source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1; export PYTHONPATH=$WT TMPDIR
  for u in bf16_add vpu_regfile sfu mxu_pe; do
    run m_check_$u $PY -m examples.minitpu.harness.check $u --backend systemc --project $O/prjm_${TAG}_$u
  done
  for i in core o1 o4 bw; do
    run m_check_dma_$i $PY -m examples.minitpu.harness.check dma --inst $i --backend simulator --project $O/prjm_${TAG}_dma_$i
  done
  if [ -f $WT/examples/minitpu/units/seq_issue.py ]; then
    run m_check_seq_issue $PY -m examples.minitpu.harness.check seq_issue --backend simulator --project $O/prjm_${TAG}_seq_issue
  fi )
echo "DONE $(date -Is)" >> $L/summary.txt
