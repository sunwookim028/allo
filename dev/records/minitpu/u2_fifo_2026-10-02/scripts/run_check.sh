#!/bin/bash
# usage: run_check.sh <backend> <inst> <tag> [--n N] variant...
cd /work/shared/users/phd/sk3463/scratch/wt-u2ff
source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export MINITPU_HARNESS_CACHE=/work/shared/users/phd/sk3463/scratch/u2ff/hcache
be=$1 inst=$2 tag=$3; shift 3
N=""
if [ "$1" = "--n" ]; then N="--n $2"; shift 2; fi
L=/work/shared/users/phd/sk3463/scratch/u2ff/logs
for v in "$@"; do
  timeout 3000 $ALLO_PYTHON -X faulthandler -m examples.minitpu.harness.check vpu_fifo --inst $inst --backend $be --variant $v $N --project /work/shared/users/phd/sk3463/scratch/u2ff/prj > $L/${be}_${inst}_${v}${tag}.log 2>&1
  rc=$?
  echo "== $be $inst $v$tag exit $rc"
  grep -E "^(UNIT|RTL|    )|Segmentation fault|Aborted|Error|error" $L/${be}_${inst}_${v}${tag}.log | head -8 | cut -c1-700
done
