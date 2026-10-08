#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: run_vpu_tb.sh <minitpu-tree> <prj|-> <tb-name> <build-dir> [DOC_NAME=value ...]
# One of MiniTPU's vpu-level testbenches (tb/<tb-name>.sv), built the way tb/run_verilator_bundle_suite.sh builds
# it (sequencer.f, vpu.f, tb/isa_latency_pkg.sv, the tb; verilator --binary --timing -Wno-fatal), with two changes:
#  * <prj> given: vpu.f's src/core/mxu/mxu.sv is replaced by mxu_allo.sv (gen_mxu_vhls.py from <prj>) and the
#    Vitis HLS RTL is added; "-" builds MiniTPU's own mxu.sv (the reference);
#  * each DOC_NAME=value overrides that localparam in a copy of tb/isa_latency_pkg.sv (an ISA version's numbers).
# The MiniTPU tree is only read.
set -euo pipefail
T=$(readlink -f "$1"); P=$2; TB=$3; B=$(mkdir -p "$4" && readlink -f "$4"); shift 4
D=$(cd "$(dirname "$0")" && pwd)
cp "$T/tb/isa_latency_pkg.sv" "$B/isa_latency_pkg.sv"
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  grep -q "localparam int unsigned $k = " "$B/isa_latency_pkg.sv" || { echo "no $k in isa_latency_pkg"; exit 2; }
  sed -i "s/localparam int unsigned $k = [0-9]*;/localparam int unsigned $k = $v;  \/\/ overridden by run_vpu_tb.sh/" "$B/isa_latency_pkg.sv"
done
sed "s#^#$T/#" "$T/src/core/vpu.f" > "$B/vpu.f"
if [[ "$P" != "-" ]]; then
  P=$(readlink -f "$P")
  python3 "$D/gen_mxu_vhls.py" "$P" "$B/mxu_allo.sv" ${FIFO_DEPTH:+--fifo-depth $FIFO_DEPTH} >/dev/null
  V=$P/out.prj/solution1/syn/verilog
  grep -v "src/core/mxu/mxu.sv$" "$B/vpu.f" > "$B/vpu.f.tmp"
  { cat "$B/vpu.f.tmp"; echo "$B/mxu_allo.sv"; ls "$V"/*.v; } > "$B/vpu.f"; rm "$B/vpu.f.tmp"
fi
sed "s#^\([^+-]\)#$T/\1#" "$T/src/core/sequencer/sequencer.f" > "$B/sequencer.f"
cd "$T"
verilator --build-jobs 8 --binary --timing -Wno-fatal --top-module "$TB" -f "$B/sequencer.f" -f "$B/vpu.f" \
  "$B/isa_latency_pkg.sv" "tb/$TB.sv" --Mdir "$B/obj_$TB" > "$B/verilator_$TB.log" 2>&1 || { tail -20 "$B/verilator_$TB.log"; exit 1; }
timeout "${TB_TIMEOUT:-1200}" "$B/obj_$TB/V$TB"
