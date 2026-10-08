#!/usr/bin/env bash
# MiniTPU's fetch and loop testbenches (U4 Phase 0, track F1), built unchanged out of the read-only clone.
set -uo pipefail
M=/work/shared/users/phd/sk3463/minitpu
B=/work/shared/users/phd/sk3463/scratch/u4p0_tbs/front
mkdir -p "$B"
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
cd "$M"
verilator --version
git -C "$M" rev-parse HEAD
run() {
  local tb=$1; shift
  echo "== $tb"
  verilator --build-jobs 8 --binary --timing -Wno-fatal --top-module "$tb" "$@" "tb/$tb.sv" --Mdir "$B/$tb" > "$B/$tb.build.log" 2>&1 || { echo "BUILD FAIL $tb"; tail -20 "$B/$tb.build.log"; return; }
  ( cd "$M" && timeout 600 "$B/$tb/V$tb" ) 2>&1 | tail -8
}
run tb_fetch_queue_shift src/pkg/minitpu_config_pkg.sv src/core/vpu/vpu_pkg.sv \
  src/core/sequencer/sequencer_pkg.sv src/core/sequencer/sequencer_fetch_queue.sv
for tb in tb_bundle_loop tb_loop_begin_r tb_loop_buffer_cap; do
  run $tb -f src/core/sequencer/sequencer.f -f src/core/vpu.f tb/isa_latency_pkg.sv
done
