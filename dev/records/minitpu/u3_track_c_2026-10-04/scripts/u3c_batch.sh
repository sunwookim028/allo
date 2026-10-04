#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: u3c_batch.sh <list> <scratch> [P]: one u3c_build.py per line "<name> <unit> <variant> [args]",
# P in parallel; project <scratch>/<name>.prj, log <scratch>/<name>.log. Run from the worktree root
# after `source examples/minitpu/harness/env-zhang21.sh`.
L=$1; S=$2; P=${3:-6}
D=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$PWD
grep -v '^#' "$L" | grep . | xargs -P "$P" -L 1 bash -c 'n=$0; u=$1; v=$2; shift 2; eval "$ALLO_PYTHON '"$D"'/u3c_build.py $u $v '"$S"'/$n.prj $*" > '"$S"'/$n.log 2>&1; grep -h "^BUILD\|^\[latency\]" '"$S"'/$n.log | cut -c1-400 | sed "s/^/$n: /"'
