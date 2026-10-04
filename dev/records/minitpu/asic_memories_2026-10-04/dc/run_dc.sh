#!/bin/bash
# usage: [MACRO_DB=<.db>] [DEFINES="A=1 B=2"] run_dc.sh <name> <top> <clk|none> <period> <src>...
# The U2 pilot's run_dc.sh with dc_sram.tcl (this directory) and paths under scratch/asicmem_dc.
name=$1 top=$2 clk=$3 period=$4; shift 4
source /etc/profile.d/modules.sh; module load synopsys-dc-W-2024.09 >/dev/null 2>&1
export SNPSLMD_LICENSE_FILE=27020@en-license-05.coecis.cornell.edu
export ADK=/scratch/users/sk3463/build_T4_MAXDIM16_shipped_baseline/1-freepdk-45nm/view-standard
D=/work/shared/users/phd/sk3463/scratch/asicmem_dc
T=$(cd "$(dirname "$0")" && pwd)
export DESIGN=$top CLK=$clk PERIOD=$period SRCS="$*" OUT=$D/out/$name
mkdir -p $D/work/$name && cd $D/work/$name
start=$(date +%s)
dc_shell -f $T/dc_sram.tcl > $D/out_$name.log 2>&1; rc=$?
echo "DC_EXIT $rc WALL $(( $(date +%s) - start ))s" | tee -a $D/out_$name.log
