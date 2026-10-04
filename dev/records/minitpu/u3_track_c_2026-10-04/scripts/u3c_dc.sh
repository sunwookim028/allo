#!/bin/bash
# usage: u3c_dc.sh <name> <top> <clk|none> <period> <src>...   (env DEFINES, PARAMS optional)
# Same flow as U1's run_dc.sh (dc_u1.tcl), output to $U3C_DC (default scratch/u3c/dc).
name=$1 top=$2 clk=$3 period=$4; shift 4
source /etc/profile.d/modules.sh; module load synopsys-dc-W-2024.09 >/dev/null 2>&1
export SNPSLMD_LICENSE_FILE=27020@en-license-05.coecis.cornell.edu
export ADK=/scratch/users/sk3463/build_T4_MAXDIM16_shipped_baseline/1-freepdk-45nm/view-standard
D=${U3C_DC:-/work/shared/users/phd/sk3463/scratch/u3c/dc}
T=$(cd "$(dirname "$0")" && pwd)
export DESIGN=$top CLK=$clk PERIOD=$period SRCS="$*" OUT=$D/out/$name
mkdir -p $D/work/$name && cd $D/work/$name
start=$(date +%s)
dc_shell -f $T/dc_u3c.tcl > $D/out_$name.log 2>&1; rc=$?
echo "DC_EXIT $rc WALL $(( $(date +%s) - start ))s" | tee -a $D/out_$name.log
