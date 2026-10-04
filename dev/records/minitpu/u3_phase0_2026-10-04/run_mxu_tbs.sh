#!/usr/bin/env bash
# MiniTPU's MXU-contract testbenches, built unchanged out of the read-only clone.
set -uo pipefail
M=/work/shared/users/phd/sk3463/minitpu
B=/work/shared/users/phd/sk3463/scratch/u3p0_tbs/mxu
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
cd "$M"
verilator --version
run() {
  local tb=$1; shift
  echo "== $tb"
  verilator --build-jobs 8 --binary --timing -Wno-fatal --top-module "$tb" "$@" "tb/$tb.sv" --Mdir "$B/$tb" > "$B/$tb.build.log" 2>&1 || { echo "BUILD FAIL $tb"; tail -20 "$B/$tb.build.log"; return; }
  ( cd "$M" && timeout 600 "$B/$tb/V$tb" ) 2>&1 | tail -15
}
run tb_mxu_single_port src/core/vpu/vpu_pkg.sv src/core/vpu/vpu_fifo.sv \
  src/core/mxu/mxu_acc24_add_pipe.sv src/core/vpu/vpu_bf16_add.sv \
  src/core/mxu/mxu_bf16_mul_acc24.sv src/core/mxu/mxu_pe.sv \
  src/core/mxu/mxu_systolic_array.sv src/core/mxu/mxu.sv
run tb_acc24_special_pipe src/core/vpu/vpu_pkg.sv src/core/mxu/mxu_acc24_add_pipe.sv
for tb in tb_matrix_full_vreg tb_matrix_command_ii tb_matrix_weight_pipelining tb_matrix_weight_banks; do
  run $tb -f src/core/sequencer/sequencer.f -f src/core/vpu.f tb/isa_latency_pkg.sv
done
