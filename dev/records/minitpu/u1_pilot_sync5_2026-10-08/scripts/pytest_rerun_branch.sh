#!/bin/bash
# branch full suite, second run: the first aborted (SIGABRT, exit 134) in tests/ip_integration/test_rtl.py
# under env-zhang21.sh (libstdc++ conflict, s.3); the two Verilator files are run separately (plain env + VERILATOR)
WT=/work/shared/users/phd/sk3463/scratch/wt-reg5 TAG=branch2 O=/work/shared/users/phd/sk3463/scratch/sync5_out
source $WT/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export TMPDIR=/tmp/claude-1772902/sync5/tmp_$TAG; mkdir -p $TMPDIR
cd -P $WT; export PYTHONPATH=$(pwd -P)
start=$(date +%s)
$ALLO_PYTHON -m pytest -p no:cacheprovider -q tests/dataflow --ignore=tests/dataflow/aie tests/act tests/test_*.py tests/utils/test_backend_utils.py tests/ip_integration \
  --ignore=tests/ip_integration/test_rtl.py --ignore=tests/ip_integration/test_rtl_adapter.py --junitxml=$O/pytest_$TAG.xml > $O/pytest_$TAG.log 2>&1
echo "PYTEST_EXIT $? WALL $(( $(date +%s)-start ))s" >> $O/pytest_$TAG.log
