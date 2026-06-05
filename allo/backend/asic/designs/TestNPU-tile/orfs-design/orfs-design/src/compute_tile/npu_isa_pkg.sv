// GENERATED — do not edit directly.
// Regenerate with: make -C npu generate-sv-pkgs
// Source: mininpu/isa.py

/* verilator lint_off UNUSEDPARAM */
package npu_isa_pkg;

  // OP_*
  localparam logic [7:0] OP_MATMUL_TILE = 8'd1;
  localparam logic [7:0] OP_VEC_ROWMAX = 8'd16;
  localparam logic [7:0] OP_VEC_ROWSUM = 8'd17;
  localparam logic [7:0] OP_VEC_EXP2 = 8'd18;
  localparam logic [7:0] OP_VEC_SCALE_BCAST = 8'd19;
  localparam logic [7:0] OP_VEC_SCALE_IMM = 8'd20;
  localparam logic [7:0] OP_VEC_SUB_BCAST = 8'd21;
  localparam logic [7:0] OP_VEC_DIV_BCAST = 8'd22;
  localparam logic [7:0] OP_VEC_ZERO_ACC = 8'd23;
  localparam logic [7:0] OP_VEC_SILU = 8'd24;
  localparam logic [7:0] OP_VEC_MUL_ACC = 8'd25;
  localparam logic [7:0] OP_VEC_ADD_ACC = 8'd26;
  localparam logic [7:0] OP_VEC_SUB_ACC = 8'd27;
  localparam logic [7:0] OP_VEC_SCALE_COL_BCAST = 8'd28;
  localparam logic [7:0] OP_VEC_GELU = 8'd29;
  localparam logic [7:0] OP_VEC_MAXIMUM = 8'd32;
  localparam logic [7:0] OP_VEC_FMA = 8'd33;
  localparam logic [7:0] OP_VEC_EXP2_VREG = 8'd34;
  localparam logic [7:0] OP_VEC_SUB_VREG = 8'd35;
  localparam logic [7:0] OP_VEC_MOVI = 8'd36;
  localparam logic [7:0] OP_VEC_COPY_VREG = 8'd37;
  localparam logic [7:0] OP_VEC_RSQRT = 8'd38;
  localparam logic [7:0] OP_VEC_FMA_IMM = 8'd39;
  localparam logic [7:0] OP_VEC_ADD_VREG = 8'd40;
  localparam logic [7:0] OP_VEC_MUL_VREG = 8'd41;
  localparam logic [7:0] OP_VEC_LOAD_IMM = 8'd42;
  localparam logic [7:0] OP_CAST_F32_F16 = 8'd48;
  localparam logic [7:0] OP_CAST_F16_F32 = 8'd49;
  localparam logic [7:0] OP_CAST_F16_F32_VREG = 8'd50;
  localparam logic [7:0] OP_DMA_LOAD_SPM = 8'd64;
  localparam logic [7:0] OP_DMA_LOAD_ACC = 8'd65;
  localparam logic [7:0] OP_DMA_STORE_SPM = 8'd66;
  localparam logic [7:0] OP_DMA_STORE_ACC = 8'd67;
  localparam logic [7:0] OP_LOOP_BEGIN = 8'd80;
  localparam logic [7:0] OP_LOOP_END = 8'd81;
  localparam logic [7:0] OP_FLUSH = 8'd82;
  localparam logic [7:0] OP_FLUSH_SLOT = 8'd83;
  localparam logic [7:0] OP_MATMUL_TILE_I8 = 8'd2;
  localparam logic [7:0] OP_MATMUL_TILE_DQ = 8'd3;
  localparam logic [7:0] OP_DMA_LOAD_SPM_I4 = 8'd68;
  localparam logic [7:0] OP_VEC_DEQUANT_I4 = 8'd69;
  localparam logic [7:0] OP_DMA_LOAD_SPM_I8 = 8'd70;
  localparam logic [7:0] OP_DMA_STORE_SPM_I8 = 8'd71;
  localparam logic [7:0] OP_DMA_LOAD_SPM_MI = 8'd72;
  localparam logic [7:0] OP_DMA_LOAD_ACC_MI = 8'd73;
  localparam logic [7:0] OP_DMA_LOAD_SPM_I4_MI = 8'd74;
  localparam logic [7:0] OP_DMA_LOAD_SPM_I8_MI = 8'd75;
  localparam logic [7:0] OP_DMA_STORE_SPM_MI = 8'd76;
  localparam logic [7:0] OP_DMA_STORE_ACC_MI = 8'd77;
  localparam logic [7:0] OP_DMA_STORE_SPM_I8_MI = 8'd78;
  localparam logic [7:0] OP_BARRIER = 8'd84;
  localparam logic [7:0] OP_LOOP_BEGIN_CSR = 8'd85;

  // FLAG_DTYPE_*
  localparam logic [2:0] FLAG_DTYPE_LSB = 3'd6;
  localparam logic [2:0] FLAG_DTYPE_WIDTH = 3'd2;

  // FLAG_ASYNC_*
  localparam logic [2:0] FLAG_ASYNC_BIT = 3'd5;

  // FLAG_IMM_*
  localparam logic [2:0] FLAG_IMM_OP1_BIT = 3'd4;
  localparam logic [2:0] FLAG_IMM_OP0_BIT = 3'd3;

  // FLAG_MATMUL_*
  localparam logic [1:0] FLAG_MATMUL_CAUSAL_BIT = 2'd2;
  localparam logic [1:0] FLAG_MATMUL_ACCUMULATE_BIT = 2'd1;
  localparam logic [1:0] FLAG_MATMUL_TRANS_B_BIT = 2'd0;

  // DTYPE_*
  localparam logic [1:0] DTYPE_FP16 = 2'd0;
  localparam logic [1:0] DTYPE_INT8 = 2'd1;
  localparam logic [1:0] DTYPE_BF16 = 2'd2;
  localparam logic [1:0] DTYPE_FP8 = 2'd3;

  // FLUSH_*
  localparam logic [63:0] FLUSH_INSTR = 64'h5200000000000000;

  // MAT_DIM_*
  localparam logic [7:0] MAT_DIM_MAX = 8'd128;

  // DRAM_STRIDE_*
  localparam logic [16:0] DRAM_STRIDE_MAX = 17'd65536;

  // BANKING_*
  localparam logic [3:0] BANKING = 4'd8;

  // VREG_ID_*
  localparam logic [1:0] VREG_ID_BITS = 2'd3;

  // MODE_*
  localparam logic [1:0] MODE_VPU = 2'd0;
  localparam logic [1:0] MODE_SYSTOLIC = 2'd1;
  localparam logic [1:0] MODE_TMA = 2'd2;
  localparam logic [1:0] MODE_HALT = 2'd3;

  // VPU_TYPE_*
  localparam logic [2:0] VPU_TYPE_SCALAR = 3'd0;
  localparam logic [2:0] VPU_TYPE_VLOAD = 3'd1;
  localparam logic [2:0] VPU_TYPE_VSTORE = 3'd2;
  localparam logic [2:0] VPU_TYPE_VCOMPUTE = 3'd3;

  // VPU_OP_*
  localparam logic [2:0] VPU_OP_ADD = 3'd0;
  localparam logic [2:0] VPU_OP_SUB = 3'd1;
  localparam logic [2:0] VPU_OP_RELU = 3'd2;
  localparam logic [2:0] VPU_OP_MUL = 3'd3;
  localparam logic [2:0] VPU_OP_D_RELU = 3'd4;

  // TMA_DIR_*
  localparam logic [0:0] TMA_DIR_DM_TO_L2 = 1'd0;
  localparam logic [0:0] TMA_DIR_L2_TO_DM = 1'd1;

  // HALT_*
  localparam logic [63:0] HALT_INSTR = 64'hC000000000000000;

  // FIELDS_INSTR bit positions
  localparam int FIELDS_INSTR_OP_LSB = 56;
  localparam int FIELDS_INSTR_OP_WIDTH = 8;
  localparam int FIELDS_INSTR_FLAGS_LSB = 48;
  localparam int FIELDS_INSTR_FLAGS_WIDTH = 8;
  localparam int FIELDS_INSTR_OP0_LSB = 32;
  localparam int FIELDS_INSTR_OP0_WIDTH = 16;
  localparam int FIELDS_INSTR_OP1_LSB = 16;
  localparam int FIELDS_INSTR_OP1_WIDTH = 16;
  localparam int FIELDS_INSTR_OP2_LSB = 0;
  localparam int FIELDS_INSTR_OP2_WIDTH = 16;

  // FIELDS_VPU bit positions
  localparam int FIELDS_VPU_MODE_LSB = 62;
  localparam int FIELDS_VPU_MODE_WIDTH = 2;
  localparam int FIELDS_VPU_ADDR_A_LSB = 49;
  localparam int FIELDS_VPU_ADDR_A_WIDTH = 13;
  localparam int FIELDS_VPU_ADDR_B_LSB = 36;
  localparam int FIELDS_VPU_ADDR_B_WIDTH = 13;
  localparam int FIELDS_VPU_ADDR_OUT_LSB = 23;
  localparam int FIELDS_VPU_ADDR_OUT_WIDTH = 13;
  localparam int FIELDS_VPU_VPU_TYPE_LSB = 20;
  localparam int FIELDS_VPU_VPU_TYPE_WIDTH = 3;
  localparam int FIELDS_VPU_VREG_DST_LSB = 17;
  localparam int FIELDS_VPU_VREG_DST_WIDTH = 3;
  localparam int FIELDS_VPU_VREG_A_LSB = 14;
  localparam int FIELDS_VPU_VREG_A_WIDTH = 3;
  localparam int FIELDS_VPU_VREG_B_LSB = 11;
  localparam int FIELDS_VPU_VREG_B_WIDTH = 3;
  localparam int FIELDS_VPU_VPU_OPCODE_LSB = 4;
  localparam int FIELDS_VPU_VPU_OPCODE_WIDTH = 3;
  localparam int FIELDS_VPU_SCALAR_B_LSB = 3;
  localparam int FIELDS_VPU_SCALAR_B_WIDTH = 1;

  // FIELDS_TMA bit positions
  localparam int FIELDS_TMA_MODE_LSB = 62;
  localparam int FIELDS_TMA_MODE_WIDTH = 2;
  localparam int FIELDS_TMA_DIR_LSB = 61;
  localparam int FIELDS_TMA_DIR_WIDTH = 1;
  localparam int FIELDS_TMA_DM_BASE_LSB = 45;
  localparam int FIELDS_TMA_DM_BASE_WIDTH = 16;
  localparam int FIELDS_TMA_L2_BASE_LSB = 30;
  localparam int FIELDS_TMA_L2_BASE_WIDTH = 15;
  localparam int FIELDS_TMA_LENGTH_LSB = 14;
  localparam int FIELDS_TMA_LENGTH_WIDTH = 16;

  // FIELDS_SYSTOLIC bit positions
  localparam int FIELDS_SYSTOLIC_MODE_LSB = 62;
  localparam int FIELDS_SYSTOLIC_MODE_WIDTH = 2;
  localparam int FIELDS_SYSTOLIC_ADDR_A_LSB = 49;
  localparam int FIELDS_SYSTOLIC_ADDR_A_WIDTH = 13;
  localparam int FIELDS_SYSTOLIC_ADDR_B_LSB = 36;
  localparam int FIELDS_SYSTOLIC_ADDR_B_WIDTH = 13;
  localparam int FIELDS_SYSTOLIC_ADDR_OUT_LSB = 23;
  localparam int FIELDS_SYSTOLIC_ADDR_OUT_WIDTH = 13;
  localparam int FIELDS_SYSTOLIC_LENGTH_LSB = 0;
  localparam int FIELDS_SYSTOLIC_LENGTH_WIDTH = 23;

endpackage
/* verilator lint_on UNUSEDPARAM */
