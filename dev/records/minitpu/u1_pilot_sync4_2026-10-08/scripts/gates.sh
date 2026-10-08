#!/bin/bash
# usage: gates.sh <worktree> <tag>   (sync3's gates.sh + mutate none, df tests, reproduce.sh on branch)
WT=$1 TAG=$2 O=/work/shared/users/phd/sk3463/scratch/sync4_out
source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export TMPDIR=/tmp/claude-1772902/sync4_tmpg_$TAG; mkdir -p $TMPDIR
PY=$ALLO_PYTHON; L=$O/gates_$TAG; mkdir -p $L
cd -P $WT; WT=$(pwd -P)
run() { n=$1; shift; s=$(date +%s); ( "$@" ) > $L/$n.log 2>&1; echo "$n EXIT $? $(( $(date +%s)-s ))s" | tee -a $L/summary.txt; }
echo "HEAD $(git -C $WT rev-parse HEAD) start $(date -Is)" >> $L/summary.txt
run allo_where $PY -c "import allo,os;print(os.path.realpath(allo.__file__))"
run hashes $PY -c "
import hashlib
from allo.dataflow import customize
from examples.tinytpu.microarch_isa import tinytpu_isa, schedule
for T in ('vhls','catapult','systemc'):
    s=customize(tinytpu_isa); schedule(s); t=str(s.build(target=T))
    print('HASH',T,hashlib.sha256(t.encode()).hexdigest(),len(t))
"
cd -P $WT/examples/tinytpu
run gen_isa $PY gen_isa.py --check
run lift_units $PY lift_units.py --check
run bench_isa $PY bench_isa.py
run stress_isa $PY stress_isa.py
run act_gate $PY act_compile.py --gate
run mutate_none $PY mutate.py none
run systemc_csim $PY systemc_csim.py 3
cd -P $WT
run df_unit $PY tests/dataflow/test_df_unit.py
run region_stateful $PY tests/dataflow/test_region_stateful.py
cd -P $WT/examples/eva
run eva env PYTHONPATH=$WT $PY cosim_eva_systemc.py
cd -P $WT && git checkout -- examples/eva/generated 2>/dev/null; git clean -fdq -- examples/eva/generated 2>/dev/null
run minitpu_quick $PY examples/minitpu/run.py --quick
run u3e $PY -m examples.minitpu.template.run_u3e
for u in bf16_add vpu_regfile sfu mxu_pe; do
  run check_$u $PY examples/minitpu/harness/check.py $u --backend systemc --project $O/prj_${TAG}_$u
done
run check_vpu_word_array_narrow $PY examples/minitpu/harness/check.py vpu_word_array --backend systemc --inst narrow --project $O/prj_${TAG}_vwa
if [ "$TAG" = branch ]; then
  cd -P $WT/examples/tinytpu; run reproduce_nocosim ./reproduce.sh --no-cosim; cd -P $WT
fi
echo "DONE $(date -Is)" >> $L/summary.txt
