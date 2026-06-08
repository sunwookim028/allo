export DESIGN_NAME = compute_tile
export PLATFORM    = nangate45

# Full compute_tile source closure.
export VERILOG_FILES = \
  ./designs/src/$(DESIGN_NAME)/npu_config_pkg.sv \
  ./designs/src/$(DESIGN_NAME)/npu_isa_pkg.sv \
  ./designs/src/$(DESIGN_NAME)/vpu_pkg.sv \
  ./designs/src/$(DESIGN_NAME)/compute_tile.sv \
  ./designs/src/$(DESIGN_NAME)/sequencer.sv \
  ./designs/src/$(DESIGN_NAME)/l1.sv \
  ./designs/src/$(DESIGN_NAME)/mxu.sv \
  ./designs/src/$(DESIGN_NAME)/systolic.sv \
  ./designs/src/$(DESIGN_NAME)/pe.sv \
  ./designs/src/$(DESIGN_NAME)/vpu.sv \
  ./designs/src/$(DESIGN_NAME)/vpu_op.sv \
  ./designs/src/$(DESIGN_NAME)/vec_regfile.sv \
  ./designs/src/$(DESIGN_NAME)/pc.sv \
  ./designs/src/$(DESIGN_NAME)/decoder.sv \
  ./designs/src/$(DESIGN_NAME)/fp16_to_fp32.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_to_fp16.sv \
  ./designs/src/$(DESIGN_NAME)/fp16_mul_fp32.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_add.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_mul.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_max.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_exp2.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_rsqrt.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_recip.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_exp2_bram.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_rsqrt_bram.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_recip_bram.sv \
  ./designs/src/$(DESIGN_NAME)/bram_l1_data.sv \
  ./designs/src/$(DESIGN_NAME)/bram_iram.sv

export SDC_FILE      = ./designs/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc
export ABC_AREA      = 1

# Keep default adder mapping for initial compute_tile bring-up.
export ADDER_MAP_FILE :=

export CORE_UTILIZATION ?= 55
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT        = 100
export REMOVE_CELLS_FOR_EQY   = TAPCELL*

export SYNTH_ARGS = -noshare
export SYNTH_MEMORY_MAX_BITS = 70000
export SYNTH_MOCK_LARGE_MEMORIES = 1
