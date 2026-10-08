#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: board_build.sh <minitpu-clone> <stage: synth-check|synth|qor|all> <out-dir> [cpus, default 56-63]
# Runs the clone's own fpga/run_board_build.sh zcu104 <stage> <out-dir> on zhang-21: Vivado 2023.2
# (/opt/xilinx/Vivado/2023.2, bootgen from Vitis 2023.2), a python3 >= 3.7 for stamp_build_id.py (the allo
# env's; the host's is 3.6), and the whole build pinned to 8 CPUs with taskset (shared host).
# PL_CLK_MHZ passes through (the clone's default is 200).
set -eo pipefail
C=$1; STAGE=$2; OUT=$3; CPUS=${4:-56-63}
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate allo
export PATH=/opt/xilinx/Vivado/2023.2/bin:/opt/xilinx/Vitis/2023.2/bin:$CONDA_PREFIX/bin:$PATH
unset LD_PRELOAD; set -u
mkdir -p "$OUT"
echo "board_build: clone $C HEAD $(git -C "$C" rev-parse HEAD) stage $STAGE out $OUT cpus $CPUS PL_CLK_MHZ=${PL_CLK_MHZ:-200} start $(date -Is)"
cd "$C"
t0=$(date +%s)
set +e
taskset -c "$CPUS" fpga/run_board_build.sh zcu104 "$STAGE" "$OUT"
rc=$?
set -e
echo "board_build: rc=$rc wall $(( $(date +%s) - t0 ))s end $(date -Is)"
exit $rc
