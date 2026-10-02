#!/bin/bash
# usage: run_dc.sh <name> <top> <clk|none> <period> [-G PARAM=V ...] <src>...
# Same flow as the U1/U2 records (dc_u1.tcl byte-identical). Parameters for the
# MiniTPU instances are set by elaborating a one-line wrapper (see fifo_*.sv).
name=$1 top=$2 clk=$3 period=$4; shift 4
source /etc/profile.d/modules.sh; module load synopsys-dc-W-2024.09 >/dev/null 2>&1
export SNPSLMD_LICENSE_FILE=27020@en-license-05.coecis.cornell.edu
export ADK=/scratch/users/sk3463/build_T4_MAXDIM16_shipped_baseline/1-freepdk-45nm/view-standard
D=/work/shared/users/phd/sk3463/scratch/u2ff/dc
T=/work/shared/users/phd/sk3463/scratch/wt-u2ff/dev/records/minitpu/u2_fifo_2026-10-02/scripts/dc
export DESIGN=$top CLK=$clk PERIOD=$period SRCS="$*" OUT=$D/out/$name
mkdir -p $D/work/$name && cd $D/work/$name
start=$(date +%s)
dc_shell -f $T/dc_u1.tcl > $D/out_$name.log 2>&1; rc=$?
echo "DC_EXIT $rc WALL $(( $(date +%s) - start ))s" | tee -a $D/out_$name.log | tee $D/out/$name/wall.txt
