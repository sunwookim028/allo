#!/bin/bash
# usage: pytest_suite.sh <worktree> <logdir>: the SystemC/compose suites (csim included) + df_unit
WT=$1 L=$2; mkdir -p $L
cd $WT; source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export PYTHONPATH=$WT TMPDIR=/tmp/claude-1772902/fix5tmp OMP_NUM_THREADS=8
s=$(date +%s)
$ALLO_PYTHON tests/dataflow/test_df_unit.py > $L/df_unit.log 2>&1; echo "df_unit EXIT $?" | tee $L/summary.txt
$ALLO_PYTHON -m pytest -p no:cacheprovider -q tests/dataflow/test_region_stateful.py tests/dataflow/test_systemc_*.py \
  tests/test_compose_*.py tests/dataflow/test_compose_*.py tests/dataflow/test_stream_ports*.py \
  tests/dataflow/test_stream_flush.py tests/systemc/test_emit.py --junitxml=$L/junit.xml > $L/pytest.log 2>&1
echo "pytest EXIT $? $(tail -1 $L/pytest.log) $(( $(date +%s)-s ))s" | tee -a $L/summary.txt
