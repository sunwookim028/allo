#!/bin/bash
# sync4's selection, plus tests/utils/test_backend_utils.py and (where present) tests/ip_integration, last
WT=$1 TAG=$2 O=/work/shared/users/phd/sk3463/scratch/sync5_out
source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export TMPDIR=/tmp/claude-1772902/sync5/tmp_$TAG; mkdir -p $TMPDIR
cd -P $WT; export PYTHONPATH=$(pwd -P)
EXTRA="tests/utils/test_backend_utils.py"; [ -d tests/ip_integration ] && EXTRA="$EXTRA tests/ip_integration"
start=$(date +%s)
$ALLO_PYTHON -m pytest -p no:cacheprovider -q tests/dataflow --ignore=tests/dataflow/aie tests/act tests/test_*.py $EXTRA --junitxml=$O/pytest_$TAG.xml > $O/pytest_$TAG.log 2>&1
echo "PYTEST_EXIT $? WALL $(( $(date +%s)-start ))s" >> $O/pytest_$TAG.log
