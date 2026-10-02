#!/bin/bash
# usage: run_check.sh <backend> <inst> <tag> variant...
cd /work/shared/users/phd/sk3463/scratch/wt-u2rf
source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export MINITPU_HARNESS_CACHE=/work/shared/users/phd/sk3463/scratch/u2rf/hcache
be=$1 inst=$2 tag=$3; shift 3
L=/work/shared/users/phd/sk3463/scratch/u2rf/logs
for v in "$@"; do
  timeout 3000 $ALLO_PYTHON -X faulthandler -m examples.minitpu.harness.check vpu_regfile --inst $inst --backend $be --variant $v --project /work/shared/users/phd/sk3463/scratch/u2rf/prj > $L/${be}_${inst}_${v}${tag}.log 2>&1
  rc=$?
  echo "== $be $inst $v$tag exit $rc"
  grep -E "^(UNIT|    )|Segmentation fault|Aborted" $L/${be}_${inst}_${v}${tag}.log | head -8 | cut -c1-600
done
