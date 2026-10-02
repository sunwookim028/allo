#!/bin/bash
# usage: run_csyn.sh <prj>   -- apply hand-patch P1 (once), run Catapult from <prj>/build
prj=$1
source /work/shared/users/phd/sk3463/scratch/wt-u1-cat/examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
cd $prj
[ -f kernel.emitted.cpp ] || { \cp -f kernel.cpp kernel.emitted.cpp; python3 /work/shared/users/phd/sk3463/scratch/u1_cat/patch_kernel.py kernel.cpp $PATCH_ARGS; }
mkdir -p build && cd build && rm -rf Catapult* catapult.log
start=$(date +%s)
catapult -shell -f ../run.tcl > ../csyn.log 2>&1; rc=$?
echo "CATAPULT_EXIT $rc WALL $(( $(date +%s) - start ))s" >> ../csyn.log
