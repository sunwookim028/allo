#!/bin/bash
# usage: broad.sh <tree> <tag>: every top-level tests/*.py and tests/dataflow/ (csim included), junit out
T=$1 TAG=$2 L=/work/shared/users/phd/sk3463/scratch/fix5/logs/broad_$TAG; mkdir -p $L
cd $T; source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export PYTHONPATH=$T TMPDIR=/tmp/claude-1772902/fix5tmp_$TAG OMP_NUM_THREADS=8; mkdir -p $TMPDIR
$ALLO_PYTHON -c "import allo,os;print(os.path.realpath(allo.__file__))" > $L/allo_where.txt 2>&1
s=$(date +%s)
timeout 4h $ALLO_PYTHON -m pytest -p no:cacheprovider -q tests/*.py tests/dataflow --ignore=tests/dataflow/aie --junitxml=$L/junit.xml > $L/pytest.log 2>&1
echo "EXIT $? $(tail -1 $L/pytest.log) $(( $(date +%s)-s ))s" > $L/summary.txt
