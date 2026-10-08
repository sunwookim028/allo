#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: u4d_checks.sh <list>: per line "<name> <unit> <variant> [u4d_check.py args]": waits for the
# Catapult run <name> to finish (RUN_EXIT in <S>/<name>.log), then runs u4d_check.py on it once
# (output <S>/<name>.check). Sequential: one Verilator build at a time.
S=${U4D_SCRATCH:-/work/shared/users/phd/sk3463/scratch/u4d}
D=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$PWD
grep -v '^#' "$1" | grep . | while read name unit var rest; do
  until grep -q "^RUN_EXIT" $S/$name.log 2>/dev/null; do sleep 20; done
  [ -s $S/$name.check ] && continue
  if ls $S/$name.prj/build/Catapult/*.v1/concat_sim_rtl.v >/dev/null 2>&1; then
    $ALLO_PYTHON $D/u4d_check.py $unit $var $S/$name.prj $rest 2>&1 | grep -v -i "warn" > $S/$name.check
  else
    echo "NO-RTL $name: $(grep -h '^BUILD' $S/$name.log | cut -c1-600)" > $S/$name.check
  fi
  echo "$name: $(grep -h '^UNIT\|^NO-RTL' $S/$name.check | cut -c1-330)"
done
