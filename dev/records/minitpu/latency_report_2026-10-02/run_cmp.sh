#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: run_cmp.sh <list> <scratch>: cmp_rtl.py per line "<name> <unit> [args]" on <scratch>/<name>.prj.
# Run from the worktree root. MINITPU_HARNESS_CACHE defaults to <scratch>/hcache.
L=$1; S=$2
source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export MINITPU_HARNESS_CACHE=${MINITPU_HARNESS_CACHE:-$S/hcache}
grep -v '^#' "$L" | grep . | while read name unit args; do
  $ALLO_PYTHON $(dirname "$0")/cmp_rtl.py $unit $S/$name.prj $args 2>&1 || echo "CMP-FAILED $name"
done
