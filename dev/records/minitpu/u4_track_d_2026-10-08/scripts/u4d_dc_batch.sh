#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: u4d_dc_batch.sh <list>: one u4d_dc.sh per line "<env|-> <name> <top> <clk|none> <period> <src>...",
# <env> = PARAMS=a=1,b=2 or DEFINES=X (or -); $S in a line = scratch/u4d. Sequential (u4d_dc.sh locks).
T=$(cd "$(dirname "$0")" && pwd); S=${U4D_SCRATCH:-/work/shared/users/phd/sk3463/scratch/u4d}
grep -v '^#' "$1" | grep . | sed "s#\\\$S#$S#g" | while read env rest; do
  ( unset PARAMS DEFINES; [ "$env" != "-" ] && export "$env"; $T/u4d_dc.sh $rest | sed "s/^/$(echo $rest | cut -d' ' -f1): /" )
done
