#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: run_tb.sh <minitpu-scratch-tree> <prj> <tb-name> <build-dir> [extra verilator args]
# Builds tb/<tb-name>.sv of the scratch MiniTPU tree with mxu_allo.sv (generated from <prj>) in place
# of src/core/mxu/mxu.sv, the Catapult RTL (concat_sim_rtl.v) as the core; the unit suite's flags.
set -euo pipefail
T=$1; P=$2; TB=$3; B=$4; shift 4
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
TOP=$(grep -o "mxu_wide_dim[0-9]*" "$P/kernel.cpp" | head -1)
python3 "$D/gen_mxu_allo.py" "$P" "$B/mxu_allo.sv"
RTL="$P/build/Catapult/$TOP.v1/concat_sim_rtl.v"
test -f "$RTL" || RTL="$P/build/Catapult/$TOP.v1/concat_rtl.v"
cd "$T"
verilator --build-jobs 8 --binary --timing -Wno-fatal --top-module "$TB" \
  src/core/vpu/vpu_pkg.sv "$B/mxu_allo.sv" "$RTL" "tb/$TB.sv" --Mdir "$B/obj_$TB" "$@" > "$B/verilator_$TB.log" 2>&1 \
  || { tail -30 "$B/verilator_$TB.log"; exit 1; }
timeout "${TB_TIMEOUT:-600}" "$B/obj_$TB/V$TB"
