#!/bin/bash
# usage: run_cmp.sh <out> <name> <unit> [cmp_rtl.py args] ...  (one comparison per line on stdin:
#        "<name> <unit> [args]"); appends cmp_rtl.py's output to <out>.
W=/work/shared/users/phd/sk3463/scratch/wt-u1-cat2
S=/work/shared/users/phd/sk3463/scratch/u1_cat2
R=$W/dev/records/minitpu/u1_catapult_units_2026-10-02/scripts
cd $W && source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export MINITPU_HARNESS_CACHE=$S/hcache
while read name unit args; do
  [ -z "$name" ] && continue
  case $name in \#*) continue;; esac
  $ALLO_PYTHON $R/cmp_rtl.py $unit $S/$name.prj $args >> $1 2>&1 || echo "CMP-FAILED $name $unit $args" >> $1
done
echo CMP_DONE >> $1
