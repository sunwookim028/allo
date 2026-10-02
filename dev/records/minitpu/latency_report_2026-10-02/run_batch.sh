#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: run_batch.sh <list> <scratch> [P]: one build_unit.py per line "<name> <unit> <variant> [args]",
# P in parallel, project <scratch>/<name>.prj, output <scratch>/<name>.log. Run from the worktree root.
L=$1; S=$2; P=${3:-6}
source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
D=$(dirname "$0")
grep -v '^#' "$L" | grep . | xargs -P "$P" -L 1 bash -c 'n=$0; u=$1; v=$2; shift 2; eval "$ALLO_PYTHON '"$D"'/build_unit.py $u $v '"$S"'/$n.prj $*" > '"$S"'/$n.log 2>&1; grep -h "^BUILD\|^\[latency\]" '"$S"'/$n.log | sed "s/^/$n: /"' 
