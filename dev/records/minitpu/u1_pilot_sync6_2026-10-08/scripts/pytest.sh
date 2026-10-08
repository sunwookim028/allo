#!/bin/bash
# usage: pytest.sh <worktree> <tag>. sync5's selection in its run-2 form on BOTH sides: tests/ip_integration's two
# Verilator files are ignored here (they abort the interpreter under env-zhang21.sh, sync5 s.3) and run in their
# own invocation in the plain env with VERILATOR set (ipint.sh)
WT=$1 TAG=$2 O=/work/shared/users/phd/sk3463/scratch/sync6_out
source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export TMPDIR=/tmp/claude-1772902/sync6/tmp_$TAG; mkdir -p $TMPDIR
cd -P $WT; export PYTHONPATH=$(pwd -P)
start=$(date +%s)
$ALLO_PYTHON -m pytest -p no:cacheprovider -q tests/dataflow --ignore=tests/dataflow/aie tests/act tests/test_*.py tests/utils/test_backend_utils.py tests/ip_integration \
  --ignore=tests/ip_integration/test_rtl.py --ignore=tests/ip_integration/test_rtl_adapter.py --junitxml=$O/pytest_$TAG.xml > $O/pytest_$TAG.log 2>&1
echo "PYTEST_EXIT $? WALL $(( $(date +%s)-start ))s" >> $O/pytest_$TAG.log
