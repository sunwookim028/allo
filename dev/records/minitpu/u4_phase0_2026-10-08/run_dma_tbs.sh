#!/usr/bin/env bash
# MiniTPU's DMA / host-path testbenches, built unchanged out of the read-only clone (file lists from tb/run_verilator_*.sh).
set -uo pipefail
M=/work/shared/users/phd/sk3463/minitpu
B=/work/shared/users/phd/sk3463/scratch/u4p0_tbs/dma
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
mkdir -p "$B"
cd "$M"
verilator --version
run() {
  local tb=$1; shift
  echo "== $tb"
  local t0=$SECONDS
  verilator --build-jobs 8 --binary --timing -Wno-fatal --top-module "$tb" "$@" "tb/$tb.sv" --Mdir "$B/$tb" > "$B/$tb.build.log" 2>&1 || { echo "BUILD FAIL $tb"; tail -20 "$B/$tb.build.log"; return; }
  ( cd "$M" && timeout 900 "$B/$tb/V$tb" ) 2>&1 | tail -12
  echo "   rc=${PIPESTATUS[0]} wall=$((SECONDS - t0))s"
}
run tb_dma_bandwidth -f src/core/sequencer/sequencer.f -f src/core/vpu.f src/core/dma/dma.sv src/core/dma/dma_addr_gen.sv
run tb_dm_axi_bridge_narrow src/pkg/minitpu_config_pkg.sv src/ddr/dma_landing_fifo.sv src/ddr/dm_axi_bridge.sv
run tb_iram_loader +incdir+src/pkg src/pkg/minitpu_config_pkg.sv src/core/sequencer/iram_loader.sv
run tb_copy_abi_v9 src/pkg/minitpu_modes_pkg.sv src/host/command_unit.sv
run tb_axi_cdma_addr_v9 src/pkg/minitpu_modes_pkg.sv src/host/axi_cdma_ctrl.sv
# tb_cdma_fabric_axi_mem is not a testbench: the AXI slave model of tb_cdma_fabric_e2e, which needs the vendor cdma_0/ddr4_0 IP under xsim (fpga/microbench/cdma_fabric_sim/run.sh); n/a here.
run tb_cdma_fabric_axi_mem -f src/minitpu.f   # elaborates as a top with no initial block: "end at 0s", no PASS
run tb_launch_path_e2e -f src/minitpu.f
