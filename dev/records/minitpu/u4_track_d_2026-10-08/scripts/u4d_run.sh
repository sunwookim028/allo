#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: u4d_run.sh <name> <unit> <variant> [u4d_build.py args...]
# One Catapult job, capped: ulimit -v 32 GB (virtual, per process) and a 90-minute wall timeout;
# /usr/bin/time -v records wall time and peak RSS. Project <S>/<name>.prj, log <S>/<name>.log
# (S = $U4D_SCRATCH, default scratch/u4d). Run from the worktree root after sourcing
# examples/minitpu/harness/env-zhang21.sh.
name=$1; shift
S=${U4D_SCRATCH:-/work/shared/users/phd/sk3463/scratch/u4d}
D=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$PWD
(
  flock 9   # one Catapult job at a time (scratch/u4d/catapult.lock)
  ulimit -v 32000000
  /usr/bin/time -v timeout 90m nice $ALLO_PYTHON $D/u4d_build.py "$1" "$2" $S/$name.prj "${@:3}"
) 9>$S/catapult.lock > $S/$name.log 2>&1
rc=$?
echo "RUN_EXIT $rc" >> $S/$name.log
grep -h "^BUILD\|^\[latency\]\|Elapsed (wall\|Maximum resident" $S/$name.log | cut -c1-600 | sed "s/^/$name: /"
echo "$name: RUN_EXIT $rc"
