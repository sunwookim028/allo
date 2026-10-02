#!/bin/bash
# usage: run_check.sh <backend> <inst> <tag> variant...   (zhang-21; logs to scratch/u2wa/logs)
cd /work/shared/users/phd/sk3463/scratch/wt-u2wa
source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export MINITPU_HARNESS_CACHE=/work/shared/users/phd/sk3463/scratch/u2wa/hcache
be=$1 inst=$2 tag=$3; shift 3
L=/work/shared/users/phd/sk3463/scratch/u2wa/logs
for v in "$@"; do
  timeout 3000 $ALLO_PYTHON -X faulthandler -m examples.minitpu.harness.check vpu_word_array --inst $inst --backend $be --variant $v ${CHECK_N:+--n $CHECK_N} --project /work/shared/users/phd/sk3463/scratch/u2wa/prj > $L/${be}_${inst}_${v}${tag}.log 2>&1
  rc=$?
  echo "== $be $inst $v$tag exit $rc"
  grep -E "^(UNIT|RTL|    )|Segmentation fault|Aborted" $L/${be}_${inst}_${v}${tag}.log | head -8 | cut -c1-700
done
