#!/usr/bin/env bash
# MiniTPU's issue/writeback testbenches (sequencer + vpu), built unchanged out of the read-only clone.
set -uo pipefail
M=/work/shared/users/phd/sk3463/minitpu
B=/work/shared/users/phd/sk3463/scratch/u4p0_tbs/issue
mkdir -p "$B"
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
cd "$M"
verilator --version
run() {
  local tb=$1; shift
  echo "== $tb"
  verilator --build-jobs 8 --binary --timing -Wno-fatal --top-module "$tb" "$@" "tb/$tb.sv" --Mdir "$B/$tb" > "$B/$tb.build.log" 2>&1 || { echo "BUILD FAIL $tb"; tail -20 "$B/$tb.build.log"; return; }
  ( cd "$M" && timeout 900 "$B/$tb/V$tb" ) 2>&1 | tail -25
}
for tb in tb_vpu_latency_probe tb_bundle_delay tb_bundle_vadd_loop tb_bundle_interlocks tb_vpu_reduction_writeback; do
  run $tb -f src/core/sequencer/sequencer.f -f src/core/vpu.f tb/isa_latency_pkg.sv
done
