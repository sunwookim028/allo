#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: u4d_dc.sh <name> <top> <clk|none> <period> <src>...   (env DEFINES, PARAMS optional)
# U3 track C's u3c_dc.sh (U1's dc_u1.tcl flow + DEFINES/PARAMS, dc_u3c.tcl unchanged), one DC job at a
# time (flock), wall time and peak RSS by /usr/bin/time; output $U4D_DC (default scratch/u4d/dc).
name=$1 top=$2 clk=$3 period=$4; shift 4
source /etc/profile.d/modules.sh; module load synopsys-dc-W-2024.09 >/dev/null 2>&1
export SNPSLMD_LICENSE_FILE=27020@en-license-05.coecis.cornell.edu
export ADK=/scratch/users/sk3463/build_T4_MAXDIM16_shipped_baseline/1-freepdk-45nm/view-standard
D=${U4D_DC:-/work/shared/users/phd/sk3463/scratch/u4d/dc}
T=$(cd "$(dirname "$0")" && pwd)
export DESIGN=$top CLK=$clk PERIOD=$period SRCS="$*" OUT=$D/out/$name
mkdir -p $D/work/$name && cd $D/work/$name
(
  flock 9
  /usr/bin/time -v dc_shell -f $T/dc_u3c.tcl
) 9>$D/dc.lock > $D/out_$name.log 2>&1; rc=$?
echo "DC_EXIT $rc $(grep -h 'Elapsed (wall' $D/out_$name.log | sed 's/.*: //') peak $(grep -h 'Maximum resident' $D/out_$name.log | sed 's/.*: //') kB" | tee -a $D/out_$name.log
