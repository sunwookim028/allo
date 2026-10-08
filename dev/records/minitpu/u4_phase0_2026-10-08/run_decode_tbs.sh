#!/usr/bin/env bash
# MiniTPU's decode/address testbenches (U4 Phase 0, track F2), built unchanged out of the read-only clone.
set -uo pipefail
M=/work/shared/users/phd/sk3463/minitpu
B=/work/shared/users/phd/sk3463/scratch/u4p0_tbs/decode
mkdir -p "$B"
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
cd "$M"
verilator --version
git -C "$M" rev-parse HEAD
run() {
  local tb=$1; shift
  echo "== $tb"
  verilator --build-jobs 8 --binary --timing -Wno-fatal --top-module "$tb" "$@" "tb/$tb.sv" --Mdir "$B/$tb" > "$B/$tb.build.log" 2>&1 || { echo "BUILD FAIL $tb"; tail -20 "$B/$tb.build.log"; return; }
  ( cd "$M" && timeout 600 "$B/$tb/V$tb" ) 2>&1 | grep -E "PASS tb_|FAIL|MISMATCH|Error|error|cases|round trip" | tail -12
}
for tb in tb_bundle_encoding tb_agu_resolve_width tb_bundle_scalar_agu tb_bundle_vpu_adapter tb_isa_conformance; do
  run $tb -f src/core/sequencer/sequencer.f -f src/core/vpu.f tb/isa_latency_pkg.sv
done
