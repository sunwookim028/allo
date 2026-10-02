#!/bin/bash
# usage: run_csyn.sh <tag> <variant> <width> <clock> [extra emit_csyn args...]
# Catapult csyn of one FIFO variant (whole region: producer_0 + Stream + consumer_0)
# into scratch/u3fc/cat/<tag>.prj, n = 64 cycles, both kernels pipelined at II=1.
cd /work/shared/users/phd/sk3463/scratch/wt-u2fc
source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
tag=$1 var=$2 width=$3 clock=$4; shift 4
S=/work/shared/users/phd/sk3463/scratch/u3fc/cat; mkdir -p $S
$ALLO_PYTHON dev/records/minitpu/u3_fifo_composed_2026-10-02/scripts/emit_csyn.py vpu_fifo $var $S/$tag.prj \
  --n 64 --width $width --clock $clock --pipeline producer_0:t --pipeline consumer_0:t "$@" 2>&1 | tail -3
grep -E "^# (Error|Warning: .*SCHD)|CATAPULT_EXIT|Total Area Score|Error" $S/$tag.prj/csyn.log | head -5
