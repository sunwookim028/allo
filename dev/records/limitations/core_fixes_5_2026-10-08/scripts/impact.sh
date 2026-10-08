#!/bin/bash
# usage: impact.sh <worktree> <logdir>: the core-fixes-5 impact check (the task's list)
WT=$1 L=$2; mkdir -p $L; F=/work/shared/users/phd/sk3463/scratch/fix5
cd $WT; source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export PYTHONPATH=$WT TMPDIR=/tmp/claude-1772902/fix5tmp_impact OMP_NUM_THREADS=8; mkdir -p $TMPDIR
PY=$ALLO_PYTHON
run() { n=$1; shift; s=$(date +%s); ( "$@" ) > $L/$n.log 2>&1; echo "$n EXIT $? $(( $(date +%s)-s ))s" | tee -a $L/summary.txt; }
echo "HEAD $(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --short | wc -l) start $(date -Is)" >> $L/summary.txt
run allo_where $PY -c "import allo,os;print(os.path.realpath(allo.__file__))"
run hashes $F/hashes.sh $WT $L/emit
run df_unit $PY tests/dataflow/test_df_unit.py
run pytest_list $PY -m pytest -p no:cacheprovider -q tests/dataflow/test_region_stateful.py tests/dataflow/test_systemc_*.py tests/test_compose_*.py tests/dataflow/test_compose_*.py tests/dataflow/test_stream_ports*.py tests/dataflow/test_stream_flush.py tests/test_wide_literal.py --junitxml=$L/pytest_list.xml
run limits_repro $PY tests/limits/new_wide_literal_shift.py
( cd examples/tinytpu
  run gen_isa $PY gen_isa.py --check
  run stress_isa_md16 env TPU_MAXDIM=16 $PY stress_isa.py )
run u3e $PY -m examples.minitpu.template.run_u3e
run checks $F/checks.sh $WT $L/checks
run rtl_reproduce examples/minitpu/rtl/reproduce.sh --all
echo "DONE $(date -Is)" >> $L/summary.txt
