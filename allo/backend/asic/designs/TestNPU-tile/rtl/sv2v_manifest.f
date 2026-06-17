# Packages
npu/npu_config_pkg.sv
npu/npu_isa_pkg.sv

# SRAM macro wrappers / black-boxes
# Replace these with OpenRAM-generated Verilog views once the SRAM node is wired.
../sim/wrappers/bram_spad.sv
../sim/wrappers/bram_iram.sv

# FPU leaves
compute_tile/fpu/fp16_mul_fp32.sv
compute_tile/fpu/bf16_mul_fp32.sv
compute_tile/fpu/fp8e4m3_mul_fp32.sv
compute_tile/fpu/fp16_to_fp32.sv
compute_tile/fpu/fp32_to_fp16.sv
compute_tile/fpu/fp_add.sv
compute_tile/fpu/fp_mul.sv
compute_tile/fpu/fp_max.sv
compute_tile/fpu/fp_exp2_bram.sv
compute_tile/fpu/fp_rsqrt_bram.sv
compute_tile/fpu/fp_recip_bram.sv
compute_tile/fpu/fp_reduce_tree.sv

# Compute tile modules
compute_tile/pc.sv
compute_tile/decoder.sv
compute_tile/vec_regfile.sv
compute_tile/pe.sv
compute_tile/systolic.sv
compute_tile/mxu.sv
compute_tile/vpu_lane.sv
compute_tile/vpu_ctrl.sv
compute_tile/sequencer.sv
compute_tile/spad.sv

# Top
compute_tile/compute_tile.sv