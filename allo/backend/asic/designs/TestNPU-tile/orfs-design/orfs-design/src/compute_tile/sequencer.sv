`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: sequencer
// Description: Centralized compute controller. Owns PC, instruction BRAM, and
//              decoder; dispatches each fetched instruction by opcode.
//
// Stage 2 of the NPU-ISA roadmap: adds all vec.* opcodes (0x10–0x1C, 0x1E–0x29)
// on top of Stage 1 (matmul.tile, flush, DMA, cast).  0x1D (vec.gelu) stays
// RESERVED.
//
// Stage 3 of the NPU-ISA roadmap: adds loop.begin (0x50) and loop.end (0x51)
// hardware loop control with 4 independent induction-variable (IV) registers.
//
// ISA v2 / Gap-1: MultiLinearAddr (.mi) DMA family (0x48–0x4E).
//   Adds 3 extra pool-fetch states (POOL2/3/4) after POOL0/1 for .mi opcodes.
//   Address = dram_addr + iv_reg[iv_a_id]*iv_stride_a + iv_reg[iv_b_id]*iv_stride_b
//   Pool layout (5 entries, force-appended adjacent):
//     pool[op1+0] = dram_addr
//     pool[op1+1] = row_step  (bytes per row-loop step)
//     pool[op1+2] = iv_stride_a
//     pool[op1+3] = iv_id_pair  (low16=iv_a_id, high16=iv_b_id)
//     pool[op1+4] = iv_stride_b (u32, sign-extended for address arithmetic)
//   IV address resolution is purely combinational (behavioural model, LATENCY=0).
//   TODO: For timing closure on FPGA at > 200 MHz, pipeline the two
//         multiply-adds (iv_a*stride_a + iv_b*stride_b) into a 1-cycle
//         registered stage between POOL4_WAIT and DMA_ROW_REQ.
//
// vec.load.imm (0x2A): loads BANKING fp32 values from the literal pool directly
// into a VREG register; uses states VREG_LOAD_REQ / VREG_LOAD_WAIT → VEC_ACC_VREG_WRITE.
//
// FSM:
//   IDLE         → FETCH_1 (on `start`)
//   FETCH_1/2/3  → multi-cycle pipeline before decode is stable
//   EXEC_DISPATCH:
//       OP_MATMUL_TILE      → start MXU, go to WAIT_MXU
//       OP_DMA_LOAD_*/STORE_* → latch operands, pool dereference (addr+stride adjacent), row loop
//       OP_DMA_*_MI         → same, plus 3 more pool fetches (POOL2/3/4) for IV operands
//       OP_CAST_*           → latch operands, CAST_REQ, CAST_WAIT
//       OP_FLUSH            → HALT_STATE
//       OP_VEC_*            → see Stage 2 states below
//       (other)             → HALT_STATE (undefined-opcode safe stop)
//   WAIT_MXU     → wait for systolic_done, return to FETCH_1
//   DMA_POOL0_REQ/WAIT → fetch pool[op1]   (dram_addr)
//   DMA_POOL1_REQ/WAIT → fetch pool[op1+1] (row_step for .mi; dram_stride for classic)
//   DMA_POOL2_REQ/WAIT → fetch pool[op1+2] (iv_stride_a; .mi only)
//   DMA_POOL3_REQ/WAIT → fetch pool[op1+3] (iv_id_pair;  .mi only)
//   DMA_POOL4_REQ/WAIT → fetch pool[op1+4] (iv_stride_b; .mi only)
//   DMA_ROW_REQ/WAIT   → DMA row transfer loop (load: DM→BRAM; store: BRAM→DM)
//   CAST_REQ     → issue BRAM read at cast_src_addr (1-cycle latency)
//   CAST_WAIT    → BRAM data valid; write converted data to dest BRAM or VREG
//   HALT_STATE   → done=1, return to IDLE
//
// Stage 2 FSM states:
//   VEC_VREG_ISSUE      — latch VREG rd addresses (async read → data next cycle)
//   VEC_VREG_EXEC       — VREG data valid; compute; write back to VREG (or start ACC loop)
//   VEC_FMA_LATCH       — fma only: latch a*b products; redirect rd_b to VREG[d]
//   VEC_ACC_VREG_REQ    — issue BRAM read for ACC row (rowmax/rowsum reduction)
//   VEC_ACC_VREG_WAIT   — BRAM data valid; reduce one lane; advance row counter
//   VEC_ACC_VREG_WRITE  — write assembled VREG register (all BANKING rows done)
//   VEC_ACC_RD_REQ      — issue BRAM read for single-src ACC row
//   VEC_ACC_RD_WAIT     — data valid; compute element-wise; write modified row to ACC dst
//   VEC_ACC2_RD_A_REQ   — two-src ACC ops: read acc_a row
//   VEC_ACC2_RD_A_WAIT  — latch acc_a; issue acc_b read same row
//   VEC_ACC2_RD_B_WAIT  — acc_b valid; compute; write acc_dst row
//   VEC_ZERO_ACC_WRITE  — write zero row; loop over `rows`
//
// Literal pool layout: each pool entry is stored as the lower 32 bits of one
// 64-bit IRAM word.  The pool starts at IRAM word address `pool_base`.
// Entry k is at IRAM word address (pool_base + k).
//
// DMA address arithmetic:
//   row_byte_addr  = dram_addr_reg + dma_row_r × dram_stride_reg
//   dm_word_addr   = row_byte_addr >> DM_BYTE_SHIFT   (DM is DM_DATA_WIDTH-bit wide)
//
// Cast: 2-state pipeline (REQ → WAIT) uses BRAM Port B with 1-cycle latency.
//   cast.f32.f16       : acc_src BRAM row (8×fp32) → 8×fp16 → spm_dst BRAM row (lower 128b)
//   cast.f16.f32       : spm_src BRAM row (8×fp16 in lower 128b) → 8×fp32 → acc_dst BRAM row
//   cast.f16.f32.vreg  : spm_src BRAM row (8×fp16 in lower 128b) → 8×fp32 → VREG[vreg_dst]
//
// Pool disambiguation:
//   vec.scale.imm  → DMA_POOL0_REQ (pool entry = fp32 imm)   → VEC_ACC_RD_REQ
//   vec.fma        → DMA_POOL0_REQ (pool entry = vreg_d idx)  → VEC_VREG_ISSUE
//   vec.movi       → DMA_POOL0_REQ (pool entry = fp32 imm)    → VEC_VREG_EXEC
//   vec.fma.imm    → DMA_POOL0_REQ (scale) → DMA_POOL1_REQ (bias) → VEC_VREG_ISSUE
//   (DMA)          → DMA_POOL0_REQ / DMA_POOL1_REQ → DMA_ROW_REQ (unchanged)
//
// vec_opcode_reg disambiguates at DMA_POOL0/1_WAIT; initialised to 0 at reset
// so the default (DMA) branch fires for DMA instructions.
//////////////////////////////////////////////////////////////////////////////////

module sequencer #(
    parameter N               = 8,
    parameter DATA_WIDTH      = 32,
    parameter L1_ADDR_WIDTH   = 10,
    parameter L1_DATA_WIDTH   = 256,
    parameter DM_ADDR_WIDTH   = 16,
    parameter DM_DATA_WIDTH   = 256,
    parameter MEM_LATENCY     = 2,
    parameter PE_LATENCY      = 0,
    parameter IRAM_ADDR_WIDTH = 12
) (
    input logic clk,
    input logic rst_n,

    // High-level control
    input  logic start,
    output logic done,
    output logic illegal_op_o,

    // I-RAM DMA interface (Port A of I_bram)
    input logic                       instr_write_en,
    input logic [IRAM_ADDR_WIDTH-1:0] iram_addr,
    input logic [               63:0] dma_iram_din,

    // BRAM Port B Interface (for data access)
    output logic [L1_ADDR_WIDTH-1:0] bram_addr_b,
    output logic [L1_DATA_WIDTH-1:0] bram_din_b,
    input  logic [L1_DATA_WIDTH-1:0] bram_dout_b,
    output logic                     bram_en_b,
    output logic                     bram_we_b,

    // Literal pool base: IRAM word address of the first pool entry.
    // Set by the host before asserting start; stable throughout execution.
    input logic [IRAM_ADDR_WIDTH-1:0] pool_base,

    // ABI v2 §D: program registry (Gap 3)
    // program_id_csr[3:0]: program slot selector; entry PC = program_id * PROGRAM_SLOT_SIZE
    input logic [ 31:0] program_id_csr,
    // kernel_arg_csr[127:0]: four 32-bit per-kernel arguments packed as
    // {arg3, arg2, arg1, arg0} with arg0 in bits[31:0].
    input logic [127:0] kernel_arg_csr,

    // Device-memory DMA interface (256-bit wide, one word per 32-byte chunk).
    // dm_addr is a word address: byte_addr >> log2(DM_DATA_WIDTH/8).
    output logic [DM_ADDR_WIDTH-1:0] dm_addr,
    output logic [DM_DATA_WIDTH-1:0] dm_din,
    input  logic [DM_DATA_WIDTH-1:0] dm_dout,
    output logic                     dm_en,
    output logic                     dm_we,

    // === TMA instruction port — Stage 1: tied off (superseded by inline DMA)
    output logic        tma_req,
    output logic        tma_dir,
    output logic [15:0] tma_dm_base,
    output logic [14:0] tma_l2_base,
    output logic [15:0] tma_len,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        tma_done
    /* verilator lint_on UNUSEDSIGNAL */
);

  //---------------------------------------------
  // Local constants
  //---------------------------------------------
  localparam [7:0] OP_MATMUL_TILE = 8'd1;
  localparam [7:0] OP_VEC_ROWMAX = 8'd16;
  localparam [7:0] OP_VEC_ROWSUM = 8'd17;
  localparam [7:0] OP_VEC_EXP2 = 8'd18;
  localparam [7:0] OP_VEC_SCALE_BCAST = 8'd19;
  localparam [7:0] OP_VEC_SCALE_IMM = 8'd20;
  localparam [7:0] OP_VEC_SUB_BCAST = 8'd21;
  localparam [7:0] OP_VEC_DIV_BCAST = 8'd22;
  localparam [7:0] OP_VEC_ZERO_ACC = 8'd23;
  localparam [7:0] OP_VEC_SILU = 8'd24;
  localparam [7:0] OP_VEC_MUL_ACC = 8'd25;
  localparam [7:0] OP_VEC_ADD_ACC = 8'd26;
  localparam [7:0] OP_VEC_SUB_ACC = 8'd27;
  localparam [7:0] OP_VEC_SCALE_COL_BCAST = 8'd28;
  localparam [7:0] OP_VEC_MAXIMUM = 8'd32;
  localparam [7:0] OP_VEC_FMA = 8'd33;
  localparam [7:0] OP_VEC_EXP2_VREG = 8'd34;
  localparam [7:0] OP_VEC_SUB_VREG = 8'd35;
  localparam [7:0] OP_VEC_MOVI = 8'd36;
  localparam [7:0] OP_VEC_COPY_VREG = 8'd37;
  localparam [7:0] OP_VEC_RSQRT = 8'd38;
  localparam [7:0] OP_VEC_FMA_IMM = 8'd39;
  localparam [7:0] OP_VEC_ADD_VREG = 8'd40;
  localparam [7:0] OP_VEC_MUL_VREG = 8'd41;
  localparam [7:0] OP_VEC_LOAD_IMM = 8'd42;
  localparam [7:0] OP_CAST_F32_F16 = 8'd48;
  localparam [7:0] OP_CAST_F16_F32 = 8'd49;
  localparam [7:0] OP_CAST_F16_F32_VREG = 8'd50;
  localparam [7:0] OP_DMA_LOAD_SPM = 8'd64;
  localparam [7:0] OP_DMA_LOAD_ACC = 8'd65;
  localparam [7:0] OP_DMA_STORE_SPM = 8'd66;
  localparam [7:0] OP_DMA_STORE_ACC = 8'd67;
  localparam [7:0] OP_LOOP_BEGIN = 8'd80;
  localparam [7:0] OP_LOOP_END = 8'd81;
  localparam [7:0] OP_FLUSH = 8'd82;
  localparam [7:0] OP_FLUSH_SLOT = 8'd83;
  localparam [7:0] OP_MATMUL_TILE_I8 = 8'd2;
  localparam [7:0] OP_MATMUL_TILE_DQ = 8'd3;
  localparam [7:0] OP_DMA_LOAD_SPM_I4 = 8'd68;
  localparam [7:0] OP_DMA_LOAD_SPM_MI = 8'd72;
  localparam [7:0] OP_DMA_LOAD_ACC_MI = 8'd73;
  localparam [7:0] OP_DMA_LOAD_SPM_I4_MI = 8'd74;
  localparam [7:0] OP_DMA_LOAD_SPM_I8_MI = 8'd75;
  localparam [7:0] OP_DMA_STORE_SPM_MI = 8'd76;
  localparam [7:0] OP_DMA_STORE_ACC_MI = 8'd77;
  localparam [7:0] OP_DMA_STORE_SPM_I8_MI = 8'd78;
  localparam [7:0] OP_BARRIER = 8'd84;
  localparam [7:0] OP_LOOP_BEGIN_CSR = 8'd85;
  localparam [2:0] FLAG_ASYNC_BIT = 3'd5;

  // DM bus is DM_DATA_WIDTH bits = DM_DATA_WIDTH/8 bytes wide per word.
  localparam DM_BYTES_PER_WORD = DM_DATA_WIDTH / 8;
  // Right-shift applied to a byte address to obtain a DM word address.
  localparam DM_BYTE_SHIFT = $clog2(DM_BYTES_PER_WORD);
  // Number of fp16/fp32 lanes per 256-bit BRAM row.
  localparam BANKING = L1_DATA_WIDTH / DATA_WIDTH;  // = 8
  // Gap 3: Program registry slot size.
  // Each program occupies PROGRAM_SLOT_SIZE IRAM words.
  // 16 programs × 256 words/program = 4096 = 2^IRAM_ADDR_WIDTH.
  localparam PROGRAM_SLOT_SIZE = 256;

  //---------------------------------------------
  // Internal Signals
  //---------------------------------------------

  // PC signals
  logic [IRAM_ADDR_WIDTH-1:0] pc_val;
  logic pc_enable;
  logic pc_load;
  logic [IRAM_ADDR_WIDTH-1:0] pc_load_val;

  // Decoder signals
  logic [63:0] current_instr;
  logic [7:0] op;
  // flags: decoded from instruction — bits[1] (accumulate) and [2] (causal) wired to MXU.
  // bits[0] (transB) and [7:3] (reserved/DMA) are intentionally unused in Stage 1.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [7:0] flags;
  /* verilator lint_on UNUSEDSIGNAL */
  logic [7:0] flags_reg;  // flags latched at EXEC_DISPATCH; valid throughout pool-fetch states
  logic mxu_accumulate;
  logic mxu_causal;
  assign mxu_accumulate = flags[1];
  assign mxu_causal     = flags[2];
  /* verilator lint_off UNUSEDSIGNAL */
  // op[15:L1_ADDR_WIDTH] unused at Stage 1; reserved for wider L1 in Stage 2+.
  logic [15:0] op0;
  logic [15:0] op1;
  logic [15:0] op2;
  /* verilator lint_on UNUSEDSIGNAL */

  // Unit control
  logic start_systolic;
  logic systolic_done;

  // BRAM signals from MXU
  logic [L1_ADDR_WIDTH-1:0] systolic_addr;
  logic [L1_DATA_WIDTH-1:0] systolic_din_b;
  logic [L1_DATA_WIDTH-1:0] systolic_dout_b;
  logic systolic_en_b;
  logic systolic_we_b;

  // IRAM port B address MUX control
  logic pool_fetch_active;  // redirect IRAM B to pool addr
  logic [IRAM_ADDR_WIDTH-1:0] pool_fetch_addr;
  logic [IRAM_ADDR_WIDTH-1:0] iram_b_addr;

  // DMA operand registers (latched from decoded instruction at EXEC_DISPATCH)
  logic [7:0] dma_op_reg;  // latched opcode
  logic [L1_ADDR_WIDTH-1:0] dma_spm_base;  // SPM start row (op0 truncated)
  logic [7:0] dma_mat_rows;  // rows to transfer:
                             //   load: op2[15:8]+1; store: op2[15:8]+1
  logic [7:0] dma_row_r;  // current row counter
  logic [31:0] dram_addr_reg;  // latched from pool[pf_addr_base] (step 0)
  logic [31:0] dram_stride_reg;  // latched from pool[dma_stride_pool_idx]

  // DMA address computation (combinational)
  logic [31:0] dma_byte_addr;  // byte address for current row

  // Cast operand registers
  logic [L1_ADDR_WIDTH-1:0] cast_src_addr;  // source BRAM row
  logic [L1_ADDR_WIDTH-1:0] cast_dst_addr;  // dest BRAM row (unused for vreg variant)
  logic [2:0] cast_vreg_dst;  // dest VREG register (vreg variant)
  logic cast_is_f32_to_f16;  // selects fp32→fp16 path
  logic cast_to_vreg;  // selects VREG output

  // fp16↔fp32 converter lane arrays (instantiated below via generate)
  logic [BANKING-1:0][15:0] cast_fp16_in;
  logic [BANKING-1:0][31:0] cast_fp32_from_fp16;
  logic [BANKING-1:0][31:0] cast_fp32_in;
  logic [BANKING-1:0][15:0] cast_fp16_from_fp32;

  // VREG interface
  logic [2:0] vreg_rd_addr_a;
  logic [2:0] vreg_rd_addr_b;
  logic [2:0] vreg_wr_addr;
  logic [BANKING-1:0][DATA_WIDTH-1:0] vreg_rd_data_a;
  logic [BANKING-1:0][DATA_WIDTH-1:0] vreg_rd_data_b;
  logic [BANKING-1:0][DATA_WIDTH-1:0] vreg_wr_data;
  logic vreg_wr_en;

  // Stage 2 VEC operand registers
  logic [L1_ADDR_WIDTH-1:0] vec_acc_src;  // ACC source base row
  logic [L1_ADDR_WIDTH-1:0] vec_acc_dst;  // ACC destination base row
  logic [L1_ADDR_WIDTH-1:0] vec_acc_b;  // ACC second source (2-src ops)
  logic [2:0] vec_vreg_dst;  // VREG destination register
  logic [2:0] vec_vreg_a;  // VREG source a register
  logic [2:0] vec_vreg_b;  // VREG source b register
  logic [2:0] vec_vreg_d;  // VREG d register (fma only)
  logic [7:0] vec_opcode_reg;  // latched op for inner states
  logic [7:0] vec_row_r;  // row counter 0..BANKING-1
  logic [31:0] vec_imm_reg;  // fp32 immediate from pool
  logic [31:0] vec_bias_reg;  // fp32 bias (fma.imm)
  logic [7:0] vec_zero_rows;  // row count for vec.zero.acc / movi
  logic [BANKING-1:0][DATA_WIDTH-1:0] vreg_data_reg;  // assembled VREG (rowmax/rowsum)
  logic [BANKING-1:0][DATA_WIDTH-1:0] vec_prod_reg;  // partial products (fma)
  logic [L1_DATA_WIDTH-1:0] vec_acc_a_row_reg;  // latched acc_a row (2-src ops)
  (* DONT_TOUCH = "true" *)
  logic [L1_DATA_WIDTH-1:0] bram_dout_b_a2_reg;  // registered bram_dout_b B-side for 2-src ACC
  (* DONT_TOUCH = "true" *)
  logic [L1_DATA_WIDTH-1:0] acc2_result_reg;  // registered fp32 result for 2-src ACC
  (* DONT_TOUCH = "true" *)
  logic [L1_DATA_WIDTH-1:0] vec_computed_row_reg;  // registered single-src result (breaks BRAM→fp32→BRAM)
  logic [L1_ADDR_WIDTH-1:0] acc2_dst_addr_reg;  // registered write address for 2-src ACC
  logic [BANKING-1:0][31:0] acc_exp2_row_reg;  // registered exp2_bram result (breaks BRAM→BRAM path)
  logic [BANKING-1:0][DATA_WIDTH-1:0] vec_vreg_latch;  // latched VREG[a] for bcast loop

  // Stage 3: Loop control — IV registers and frame state (4 independent IVs)
  logic [3:0][IRAM_ADDR_WIDTH-1:0] iv_body_start;  // IRAM address of first body instruction
  logic [3:0][31:0] iv_hi;  // loop upper bound (exclusive)
  logic [3:0][7:0] iv_step;  // increment per iteration
  logic [3:0][31:0] iv_reg;  // current IV value
  logic [1:0] loop_iv_id;  // latched iv_id for current loop op
  logic [15:0] loop_hi_pool_idx;  // latched pool index of hi value
  logic [7:0] loop_lo;  // latched lo value (initial IV)
  // Combinational: current IV + step (used in LOOP_END_EXEC and pc_enable)
  logic [31:0] iv_reg_next;

  // Stage 4: int4 DMA nibble-unpack state
  logic [2:0] i4_subrow_r;  // which of 8 subrows to write in DMA_I4_WRITE
  logic [L1_DATA_WIDTH-1:0] i4_dm_latch;  // 256-bit DM word latched at DMA_ROW_WAIT for i4
  logic [31:0] i4_subrow_nibbles;  // combinational: 32-bit slice of i4_dm_latch

  // Stage 4: matmul.tile.i4 operand latches (op0/op1/op2 change during pool fetch)
  logic [L1_ADDR_WIDTH-1:0] mxu_acc_dst_reg;
  logic [L1_ADDR_WIDTH-1:0] mxu_spm_a_reg;
  logic [L1_ADDR_WIDTH-1:0] mxu_spm_b_i4_reg;
  logic [7:0] mxu_zero_point_reg;
  logic mxu_i4_mode_reg;
  logic mxu_dq_mode_reg;  // ISA v1.4: 1 when matmul.tile.dq is executing (per-column scale)

  // ISA v2 / Gap-1: MultiLinearAddr (.mi) DMA registers
  // dma_is_mi_mode: set when the latched opcode is in the 0x48–0x4E range.
  logic dma_is_mi_mode;
  assign dma_is_mi_mode = (dma_op_reg >= 8'h48 && dma_op_reg <= 8'h4E);
  logic [31:0] dma_iv_stride_a_reg;  // bytes per iv_a increment (pool[op1+2])
  logic [31:0] dma_iv_id_pair_reg;  // {iv_b_id[15:0], iv_a_id[15:0]} (pool[op1+3])
  logic [31:0] dma_iv_stride_b_reg;  // bytes per iv_b increment, u32 bit-pattern (pool[op1+4])
  // Combinational IV offset: iv_reg[iv_a_id]*iv_stride_a + iv_reg[iv_b_id]*iv_stride_b
  logic [31:0] dma_iv_offset;
  logic [ 1:0] dma_iv_a_id;
  logic [ 1:0] dma_iv_b_id;

  // ISA v2 / Gap-2: async DMA slot tracking
  logic [ 0:0] dma_async_slot_id;  // latched slot ID from pool (0 or 1)
  logic [ 1:0] dma_slot_done;  // per-slot completion sticky bits; cleared on flush.slot exit
  logic [15:0] flush_slot_mask;  // op0 mask for OP_FLUSH_SLOT

  // Unified pool-fetch sub-FSM context registers (POOL_FETCH_REQ / POOL_FETCH_WAIT).
  // Every opcode that previously jumped to DMA_POOL0_REQ, LOOP_BEGIN_POOL_REQ,
  // VREG_LOAD_REQ, or DMA_POOL_SLOT_REQ now sets these four fields at EXEC_DISPATCH
  // and transitions to POOL_FETCH_REQ.  POOL_FETCH_WAIT latches current_instr[31:0]
  // into the appropriate register based on (vec_opcode_reg, pf_step), increments
  // pf_step, and loops back to POOL_FETCH_REQ until pf_step == pf_max_step, at which
  // point it routes to the state named by pf_next_state_tag.
  //
  // Pool address (combinational): pool_base + pf_addr_base + pf_step
  //   — equals POOL0 addr at step 0, POOL1 at step 1, etc. for all opcode families.
  logic [ 3:0] pf_step;  // current fetch step (0-based)
  logic [ 3:0] pf_max_step;  // inclusive last step
  logic [15:0] pf_addr_base;  // pool index for step 0 (= dma_pool_idx for DMA/vec,
                              //   loop_hi_pool_idx for loop.begin)
  logic [ 3:0] pf_next_state_tag;  // routing tag — which state to enter after the last step

  // Pool-fetch routing tags (pf_next_state_tag values)
  localparam [3:0] PF_TO_DMA_ROW_REQ = 4'd0;
  localparam [3:0] PF_TO_VEC_ACC_RD_REQ = 4'd1;
  localparam [3:0] PF_TO_VEC_VREG_ISSUE = 4'd2;
  localparam [3:0] PF_TO_VEC_VREG_EXEC = 4'd3;
  localparam [3:0] PF_TO_WAIT_MXU = 4'd4;
  localparam [3:0] PF_TO_FETCH_1 = 4'd5;  // loop.begin: latch IV frame, advance PC
  localparam [3:0] PF_TO_VEC_ACC_VREG_WRITE = 4'd6;  // vec.load.imm: VREG assembled, write it

  // Gap 3: combinational mux for kernel_arg_csr slice selected by loop_hi_pool_idx[1:0].
  // Used in LOOP_CSR_LATCH to latch the hi bound from the CSR without a pool fetch.
  logic [31:0] csr_loop_hi_val;

  //---------------------------------------------
  // FSM for Instruction Orchestration
  //---------------------------------------------
  typedef enum logic [5:0] {
    IDLE = 6'd0,
    FETCH_1 = 6'd1,
    FETCH_2 = 6'd2,
    FETCH_3 = 6'd3,
    EXEC_DISPATCH = 6'd4,
    WAIT_MXU = 6'd5,
    HALT_STATE = 6'd6,
    // Unified pool-fetch sub-FSM: replaces DMA_POOL0-4_REQ/WAIT, LOOP_BEGIN_POOL_REQ/WAIT,
    // VREG_LOAD_REQ/WAIT, DMA_POOL_SLOT_REQ/WAIT (16 states → 2 states).
    // Address = pool_base + pf_addr_base + pf_step; step count and routing via pf_* regs.
    POOL_FETCH_REQ = 6'd7,  // IRAM presents pool_base+pf_addr_base+pf_step on addrb
    POOL_FETCH_WAIT = 6'd8,  // current_instr = pool[step]; latch; loop or route to tag
    DMA_ROW_REQ = 6'd9,  // issue memory read (DM for load; BRAM for store)
    DMA_ROW_WAIT = 6'd10,  // data valid; write to other memory; advance row
    CAST_REQ = 6'd11,  // issue BRAM read at cast_src_addr
    CAST_WAIT = 6'd12,  // BRAM data valid; write converted result
    VEC_VREG_ISSUE = 6'd13,
    VEC_VREG_EXEC = 6'd14,
    VEC_ACC_VREG_REQ = 6'd15,
    VEC_ACC_VREG_WAIT = 6'd16,
    VEC_ACC_VREG_WRITE = 6'd17,
    VEC_ACC_RD_REQ = 6'd18,
    VEC_ACC_RD_WAIT = 6'd19,
    VEC_ACC2_RD_A_REQ = 6'd20,
    VEC_ACC2_RD_A_WAIT = 6'd21,
    VEC_ACC2_RD_B_WAIT = 6'd22,
    VEC_ZERO_ACC_WRITE = 6'd23,
    VEC_ACC2_COMPUTE = 6'd36,  // fp32 runs on registered B-side data; latch result
    VEC_ACC2_WR = 6'd37,  // write registered result to acc_dst BRAM
    VEC_ACC_EXP2_WR = 6'd38,  // write registered exp2 result to acc_dst BRAM
    VEC_ACC_RD_WR = 6'd39,  // write registered single-src result to acc_dst BRAM
    VEC_ACC_EXP2_ADDR = 6'd40,  // bram_dout_b_exp2_reg → exp2_bram addr registered; wait 1 cycle
    VEC_FMA_LATCH = 6'd24,
    LOOP_END_EXEC = 6'd25,  // IV += step; jump or fall through
    DMA_I4_WRITE = 6'd26,  // Stage 4: unpack nibbles into SPM_I8 (8 rows per DM word)
    BARRIER_WAIT = 6'd27,  // Gap 3: OP_BARRIER synchronization wait state
    LOOP_CSR_LATCH = 6'd28,  // Gap 3: OP_LOOP_BEGIN_CSR — latch hi from kernel_arg_csr
    FLUSH_SLOT_WAIT = 6'd29,  // ISA v2/Gap-2: spin until (dma_slot_done & mask) == mask
    // BRAM-LUT NL pipeline states — 1-cycle latency for exp2/rsqrt/recip BRAM modules.
    // These replace the purely-combinational fp32_exp2/rsqrt/recip chains (~90K LUT savings).
    VEC_ACC_NL_EXP2_LATCH  = 6'd30,  // exp2_bram output valid; write EXP2 result or proceed to SILU recip
    VEC_ACC_NL_RECIP_LATCH = 6'd31,  // recip_bram output valid; write SILU result
    VEC_VREG_NL_LATCH = 6'd32,  // exp2/rsqrt BRAM output valid; write to VREG
    // Reduction tree pipeline: 4-stage pipelined to break the ~11.5 ns fp32_add chain.
    // PIPE0: BRAM data → latch rsum/rmax_l0_reg; PIPE1: drive u_rsum_l1 (LATENCY=1 stage 1);
    // PIPE2: latch rsum/rmax_l1_reg (LATENCY=1 result ready); WAIT: final output.
    VEC_ACC_VREG_PIPE0 = 6'd33,  // latch L0 reduction outputs
    VEC_ACC_VREG_PIPE1 = 6'd34,  // drive u_rsum_l1_{0,1} inputs (LATENCY=1 stage-1 FF captures)
    VEC_ACC_VREG_PIPE2 = 6'd46,  // latch L1 reduction outputs (u_rsum_l1 LATENCY=1 result ready)
    // SILU NLE multiply pipeline: fp32_mul(x, NEG_LOG2E) is ~10 ns — register result before
    // driving exp2_bram address to break the BRAM→DSP→BRAM-addr 19.9 ns path.
    VEC_ACC_SILU_MUL = 6'd35,  // silu_neg_log2e_reg valid; exp2_bram addr registered this cycle
    // Fix 11: fp32_exp2_bram is now 2-cycle latency; add one extra wait state per caller.
    // _ADDR states: registered input drives exp2_bram → internal _r2 registers capture this cycle.
    // _LATCH states: _r2 regs valid → BRAM row captures this cycle; NL_EXP2_LATCH reads result.
    VEC_VREG_EXP2_ADDR = 6'd41,  // latch_a stable → exp2_bram; _r2 regs capture (cycle 0)
    VEC_VREG_EXP2_LATCH = 6'd42,  // _r2 valid; BRAM captures; proceed to VEC_VREG_NL_LATCH
    VEC_ACC_EXP2_LATCH = 6'd43,  // bram_dout_b_exp2_reg path: _r2 valid; BRAM captures
    VEC_ACC_SILU_LATCH = 6'd44,  // silu_neg_log2e_reg path: _r2 valid → f_fp32_r captures
    // Fix 12: fp32_exp2_bram is now 3-cycle; one extra wait for EXP2 and SILU ACC callers.
    // EXP2_VREG absorbs it naturally via VEC_VREG_EXP2_ADDR/LATCH (no new state needed there).
    VEC_ACC_EXP2_LATCH2 = 6'd45,  // f_fp32_r valid → BRAM addr; BRAM captures rom_out_r
    // Fix 13: u_sub_b uses fp32_add LATENCY=1; need one extra cycle after VEC_ACC_RD_WAIT.
    VEC_ACC_RD_WAIT2 = 6'd47  // u_sub_b LATENCY=1 result valid; latch vec_computed_row_reg
  } seq_state_t;

  seq_state_t state;

  //---------------------------------------------
  // Pool fetch address MUX
  //---------------------------------------------
  // Unified pool-fetch address — a single formula for all opcodes.
  // pf_addr_base = dma_pool_idx for DMA/vec; loop_hi_pool_idx for loop.begin.
  // Step 0 → pf_addr_base+0 (= POOL0 / hi-entry), step 1 → +1 (= POOL1 / bias / stride), …
  assign pool_fetch_active = (state == POOL_FETCH_REQ || state == POOL_FETCH_WAIT);
  assign pool_fetch_addr = pool_base + (pf_addr_base) + (pf_step);
  assign iram_b_addr = pool_fetch_active ? pool_fetch_addr : pc_val;

  //---------------------------------------------
  // DMA byte address (combinational)
  //---------------------------------------------
  // Classic: Row r byte address = dram_addr_reg + r × dram_stride_reg
  // .mi:     Row r byte address = (dram_addr_reg + iv_offset) + r × dram_stride_reg
  //          where iv_offset = iv_reg[iv_a_id]*iv_stride_a + iv_reg[iv_b_id]*iv_stride_b
  //
  // IV address resolution is purely combinational (behavioural model, LATENCY=0).
  // TODO: For timing closure on FPGA at > 200 MHz, pipeline the two
  //       multiply-adds into a 1-cycle registered stage inserted before DMA_ROW_REQ.
  always @* begin : iv_offset_comb
    dma_iv_a_id = dma_iv_id_pair_reg[1:0];  // bits[15:0] → low 2 bits (iv_id is 2-bit)
    dma_iv_b_id = dma_iv_id_pair_reg[17:16];  // bits[31:16] → low 2 bits
    dma_iv_offset = iv_reg[dma_iv_a_id] * dma_iv_stride_a_reg + iv_reg[dma_iv_b_id] * dma_iv_stride_b_reg;
  end
  assign dma_byte_addr = dram_addr_reg
                        + ({24'h0, dma_row_r} * dram_stride_reg)
                        + (dma_is_mi_mode ? dma_iv_offset : 32'h0);

  //---------------------------------------------
  // Stage 3: IV next-value (combinational)
  //---------------------------------------------
  // iv_reg_next = IV[loop_iv_id] + zero_extend(iv_step[loop_iv_id])
  // Used in LOOP_END_EXEC and pc_enable to decide jump vs. fall-through.
  assign iv_reg_next = iv_reg[loop_iv_id] + {24'h0, iv_step[loop_iv_id]};

  //---------------------------------------------
  // Gap 3: CSR loop hi — combinational mux over kernel_arg_csr
  //---------------------------------------------
  // Selects one of four 32-bit kernel_arg slices based on loop_hi_pool_idx[1:0]
  // (which carries the csr_idx latched from op1 at EXEC_DISPATCH).
  always @* begin : csr_loop_hi_mux
    case (loop_hi_pool_idx[1:0])
      2'd0:    csr_loop_hi_val = kernel_arg_csr[31:0];
      2'd1:    csr_loop_hi_val = kernel_arg_csr[63:32];
      2'd2:    csr_loop_hi_val = kernel_arg_csr[95:64];
      default: csr_loop_hi_val = kernel_arg_csr[127:96];  // 2'd3
    endcase
  end

  //---------------------------------------------
  // Stage 4: i4_subrow_nibbles — 32-bit slice of i4_dm_latch for current subrow
  //---------------------------------------------
  always @* begin : i4_nibble_mux
    case (i4_subrow_r)
      3'd0:    i4_subrow_nibbles = i4_dm_latch[ 31:  0];
      3'd1:    i4_subrow_nibbles = i4_dm_latch[ 63: 32];
      3'd2:    i4_subrow_nibbles = i4_dm_latch[ 95: 64];
      3'd3:    i4_subrow_nibbles = i4_dm_latch[127: 96];
      3'd4:    i4_subrow_nibbles = i4_dm_latch[159:128];
      3'd5:    i4_subrow_nibbles = i4_dm_latch[191:160];
      3'd6:    i4_subrow_nibbles = i4_dm_latch[223:192];
      default: i4_subrow_nibbles = i4_dm_latch[255:224];  // 3'd7
    endcase
  end

  //---------------------------------------------
  // Cast converter input wiring (combinational from bram_dout_b)
  //---------------------------------------------
  generate
    for (genvar i = 0; i < BANKING; i++) begin : cast_input_mux
      assign cast_fp16_in[i] = bram_dout_b[i*16+:16];
      assign cast_fp32_in[i] = bram_dout_b[i*32+:32];
    end
  endgenerate

  //---------------------------------------------
  // PC advance — fires at the end of each executed instruction
  //---------------------------------------------
  assign pc_enable =
    (state == WAIT_MXU             && systolic_done)                                          ||
    (state == DMA_ROW_WAIT         && (dma_row_r == dma_mat_rows - 8'd1)                     &&
                                      dma_op_reg != OP_DMA_LOAD_SPM_I4)                      ||
    (state == DMA_I4_WRITE         && (i4_subrow_r == 3'd7)                                  &&
                                      (dma_row_r == dma_mat_rows - 8'd1))                    ||
    (state == CAST_WAIT)                                                                       ||
    (state == VEC_ACC_VREG_WRITE)                                                              ||
      // VEC_ACC_RD_WR fires for non-NL ops (scale/sub/div/imm) after writing registered result.
      // EXP2/SILU complete one or two cycles later in the NL pipeline states below.
      (state == VEC_ACC_RD_WR && (vec_row_r == (BANKING - 1))) ||
      // EXP2 completes in VEC_ACC_EXP2_WR (registered exp2 output written to BRAM).
      (state == VEC_ACC_EXP2_WR        && (vec_row_r == (BANKING - 1))                       &&
                                        vec_opcode_reg == OP_VEC_EXP2)                        ||
      // SILU completes in NL_RECIP_LATCH (2 extra BRAM cycles: exp2 + recip).
      (state == VEC_ACC_NL_RECIP_LATCH && (vec_row_r == (BANKING - 1))                      &&
                                         vec_opcode_reg == OP_VEC_SILU)                       ||
    (state == VEC_ACC2_WR          && (vec_row_r == (BANKING - 1)))                        ||
    (state == VEC_ZERO_ACC_WRITE   && (vec_row_r == vec_zero_rows - 8'd1))                   ||
      // VEC_VREG_EXEC fires for instant VREG ops; EXP2_VREG and RSQRT wait one more
      // cycle for the BRAM module to produce its result.
      (state == VEC_VREG_EXEC        && vec_opcode_reg != OP_VEC_SCALE_BCAST    &&
                                      vec_opcode_reg != OP_VEC_SCALE_COL_BCAST &&
                                      vec_opcode_reg != OP_VEC_SUB_BCAST       &&
                                      vec_opcode_reg != OP_VEC_DIV_BCAST       &&
                                      vec_opcode_reg != OP_VEC_EXP2_VREG       &&
                                      vec_opcode_reg != OP_VEC_RSQRT)                        ||
      // NL VREG ops (exp2.vreg, rsqrt) complete in VEC_VREG_NL_LATCH.
      (state == VEC_VREG_NL_LATCH) ||
      // loop.begin: PC must advance past the loop.begin instruction so FETCH_1
      // loads the first body instruction (body_start = pc_val + 1 captured in POOL_FETCH_WAIT).
      // Fires on the last pool-fetch step when the tag is PF_TO_FETCH_1 (loop.begin only).
      (state == POOL_FETCH_WAIT && pf_next_state_tag == PF_TO_FETCH_1 && pf_step == pf_max_step) ||
      // loop.begin.csr: same PC-advance requirement, no pool fetch needed.
      (state == LOOP_CSR_LATCH) ||
      // loop.end fall-through: IV >= hi, so advance PC past loop.end.
      (state == LOOP_END_EXEC && iv_reg_next >= iv_hi[loop_iv_id]) ||
      // Gap-2: flush.slot exits FLUSH_SLOT_WAIT to FETCH_1 when all slots are done.
      // PC must advance past the flush.slot instruction on the exit cycle.
      (state == FLUSH_SLOT_WAIT &&
       (dma_slot_done & flush_slot_mask[1:0]) == flush_slot_mask[1:0]) ||
      // Gap-3: barrier exits BARRIER_WAIT to FETCH_1 on re-start.
      // PC must advance past the barrier instruction on the resume cycle.
      (state == BARRIER_WAIT && start);

  //---------------------------------------------
  // Stage 2 combinational VPU datapath — synthesizable.
  // All vec.* operations are implemented as combinational FP32 module
  // instances.  Results are sampled by the always_ff FSM at the appropriate
  // state boundary.  vec_computed_row / vec_computed_row_2src are muxed
  // inline inside the BRAM-routing always @* below to avoid iverilog
  // NBA-ordering races (same block as the consumers of those signals).
  //---------------------------------------------

  logic [DATA_WIDTH-1:0] vec_reduce_result;
  logic [L1_DATA_WIDTH-1:0] vec_computed_row;
  logic [L1_DATA_WIDTH-1:0] vec_computed_row_2src;
  logic [L1_DATA_WIDTH-1:0] vec_computed_row_mux;
  logic [L1_DATA_WIDTH-1:0] vec_computed_row_2src_mux;

  // Per-lane wires — VREG-to-VREG operations
  logic [BANKING-1:0][31:0] lane_fma_prod;  // a * b (latched into vec_prod_reg)
  logic [BANKING-1:0][31:0] lane_fma_sum;  // vec_prod_reg + d
  logic [BANKING-1:0][31:0] lane_fmaimm_prod;  // a * imm (combinational)
  logic [BANKING-1:0][31:0] lane_fmaimm_prod_reg;  // registered in VEC_FMA_LATCH — breaks mul+add chain
  logic [BANKING-1:0][31:0] lane_fmaimm_sum;  // (a*imm_reg) + bias
  logic [BANKING-1:0][31:0] lane_maximum;  // max(a, b)
  logic [BANKING-1:0][31:0] lane_exp2_vreg;  // 2^a  — alias for shared_exp2_out
  logic [BANKING-1:0][31:0] lane_sub_vreg;  // a − b
  logic [BANKING-1:0][31:0] lane_add_vreg;  // a + b
  logic [BANKING-1:0][31:0] lane_mul_vreg;  // a * b
  logic [BANKING-1:0][31:0] lane_rsqrt;  // 1/sqrt(a)

  // Per-lane shared fp32_exp2 resource: muxed across exp2.vreg / vec.exp2 / silu uses.
  // All three are mutually exclusive (single opcode per execution), so one instance
  // per lane covers all three rather than instantiating three separate exp2 units.
  logic [BANKING-1:0][31:0] shared_exp2_in;  // muxed input to the single fp32_exp2
  logic [BANKING-1:0][31:0] shared_exp2_out;  // output — aliased to all three consumers

  // Per-lane wires — single-src ACC element-wise operations
  logic [BANKING-1:0][31:0] acc_exp2_row;  // alias for shared_exp2_out
  logic [BANKING-1:0][31:0] acc_silu_row;
  logic [BANKING-1:0][31:0] acc_scale_bcast_row;
  logic [BANKING-1:0][31:0] acc_scale_col_row;
  logic [BANKING-1:0][31:0] acc_sub_bcast_row;
  logic [BANKING-1:0][31:0] acc_div_bcast_row;
  logic [BANKING-1:0][31:0] acc_scale_imm_row;

  // Per-lane wires — two-src ACC operations
  logic [BANKING-1:0][31:0] acc2_mul_row;
  logic [BANKING-1:0][31:0] acc2_add_row;
  logic [BANKING-1:0][31:0] acc2_sub_row;

  // SILU intermediate wires — silu_exp2_val is an alias for shared_exp2_out
  logic [BANKING-1:0][31:0] silu_neg_log2e, silu_exp2_val, silu_denom, silu_recip_val;
  // silu_neg_log2e_reg: registered output of fp32_mul(x, NEG_LOG2E), latched in VEC_ACC_RD_WAIT.
  // Breaks the BRAM→DSP→exp2_bram-addr 19.9 ns path into two ≤10 ns stages via VEC_ACC_SILU_MUL.
  logic [BANKING-1:0][31:0] silu_neg_log2e_reg;
  // SILU x-latch: original ACC row element latched in VEC_ACC_RD_WAIT so it is
  // still available two cycles later in VEC_ACC_NL_RECIP_LATCH (after the two
  // BRAM-LUT pipeline stages for exp2 and recip).
  logic [BANKING-1:0][31:0] silu_x_latch;

  // Scalar operand for row-broadcast ops (same value replicated across all lanes)
  logic [31:0] bcast_scalar, bcast_recip;
  assign bcast_scalar = vec_vreg_latch[vec_row_r];
  // Registered version: latched in VEC_ACC_RD_REQ so that u_sub_b and u_scl_b start
  // from a stable FF in VEC_ACC_RD_WAIT — breaks the high-fanout VREG→fp32 path.
  logic [31:0] bcast_scalar_reg;
  // Fix 9: register bram_dout_b for OP_VEC_EXP2 to break L1-BRAM→fp32_sub→exp2-BRAM addr path.
  logic [BANKING-1:0][31:0] bram_dout_b_exp2_reg;
  // Fix 12b: latch vreg_rd_data_a/b at VEC_VREG_ISSUE to break VREG-mux→FP-op path (~12 ns).
  // VEC_VREG_ISSUE is already a dedicated wait state; latching here adds 0 cycles.
  // Exception: u_fma_sum .b stays live — FMA redirects rd_addr_b→VREG[d] in VEC_FMA_LATCH.
  logic [BANKING-1:0][DATA_WIDTH-1:0] vreg_rd_data_a_latch;
  logic [BANKING-1:0][DATA_WIDTH-1:0] vreg_rd_data_b_latch;

  // BRAM-LUT 1/x: breaks the 74ns combinational NR chain that violates timing at 100 MHz.
  // bcast_scalar is valid in VEC_ACC_RD_REQ; BRAM registers the address at that posedge
  // and bcast_recip is valid one cycle later in VEC_ACC_RD_WAIT — same cycle as bram_dout_b.
  fp32_recip_bram u_bcast_recip (
      .clk(clk),
      .x(bcast_scalar),
      .result(bcast_recip)
  );

  // Row-reduce trees (8 → 4 → 2 → 1)
  logic [31:0] rsum_l0[4], rsum_l1[2], rowsum_result;
  logic [31:0] rmax_l0[4], rmax_l1[2], rowmax_result;

  fp32_add u_rsum_l0_0 (
      .clk(clk),
      .rst_n(rst_n),
      .a(bram_dout_b[0+:32]),
      .b(bram_dout_b[32+:32]),
      .result(rsum_l0[0])
  );
  fp32_add u_rsum_l0_1 (
      .clk(clk),
      .rst_n(rst_n),
      .a(bram_dout_b[64+:32]),
      .b(bram_dout_b[96+:32]),
      .result(rsum_l0[1])
  );
  fp32_add u_rsum_l0_2 (
      .clk(clk),
      .rst_n(rst_n),
      .a(bram_dout_b[128+:32]),
      .b(bram_dout_b[160+:32]),
      .result(rsum_l0[2])
  );
  fp32_add u_rsum_l0_3 (
      .clk(clk),
      .rst_n(rst_n),
      .a(bram_dout_b[192+:32]),
      .b(bram_dout_b[224+:32]),
      .result(rsum_l0[3])
  );
  // Pipeline registers: break the combinational chain.
  // L0 latched in PIPE0; L1 uses LATENCY=1 u_rsum_l1 (FF inside module) so latched in PIPE2.
  logic [31:0] rsum_l0_reg[4], rmax_l0_reg[4];
  logic [31:0] rsum_l1_reg[2], rmax_l1_reg[2];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 4; i++) begin
        rsum_l0_reg[i] <= '0;
        rmax_l0_reg[i] <= '0;
      end
      for (int i = 0; i < 2; i++) begin
        rsum_l1_reg[i] <= '0;
        rmax_l1_reg[i] <= '0;
      end
    end else begin
      if (state == VEC_ACC_VREG_PIPE0) begin
        for (int i = 0; i < 4; i++) begin
          rsum_l0_reg[i] <= rsum_l0[i];
          rmax_l0_reg[i] <= rmax_l0[i];
        end
      end
      if (state == VEC_ACC_VREG_PIPE2) begin
        for (int i = 0; i < 2; i++) begin
          rsum_l1_reg[i] <= rsum_l1[i];
          rmax_l1_reg[i] <= rmax_l1[i];
        end
      end
    end
  end

  fp32_add #(
      .LATENCY(1)
  ) u_rsum_l1_0 (
      .clk(clk),
      .rst_n(rst_n),
      .a(rsum_l0_reg[0]),
      .b(rsum_l0_reg[1]),
      .result(rsum_l1[0])
  );
  fp32_add #(
      .LATENCY(1)
  ) u_rsum_l1_1 (
      .clk(clk),
      .rst_n(rst_n),
      .a(rsum_l0_reg[2]),
      .b(rsum_l0_reg[3]),
      .result(rsum_l1[1])
  );
  fp32_add u_rsum_l2 (
      .clk(clk),
      .rst_n(rst_n),
      .a(rsum_l1_reg[0]),
      .b(rsum_l1_reg[1]),
      .result(rowsum_result)
  );

  fp32_max u_rmax_l0_0 (
      .a(bram_dout_b[0+:32]),
      .b(bram_dout_b[32+:32]),
      .result(rmax_l0[0])
  );
  fp32_max u_rmax_l0_1 (
      .a(bram_dout_b[64+:32]),
      .b(bram_dout_b[96+:32]),
      .result(rmax_l0[1])
  );
  fp32_max u_rmax_l0_2 (
      .a(bram_dout_b[128+:32]),
      .b(bram_dout_b[160+:32]),
      .result(rmax_l0[2])
  );
  fp32_max u_rmax_l0_3 (
      .a(bram_dout_b[192+:32]),
      .b(bram_dout_b[224+:32]),
      .result(rmax_l0[3])
  );
  fp32_max u_rmax_l1_0 (
      .a(rmax_l0_reg[0]),
      .b(rmax_l0_reg[1]),
      .result(rmax_l1[0])
  );
  fp32_max u_rmax_l1_1 (
      .a(rmax_l0_reg[2]),
      .b(rmax_l0_reg[3]),
      .result(rmax_l1[1])
  );
  fp32_max u_rmax_l2 (
      .a(rmax_l1_reg[0]),
      .b(rmax_l1_reg[1]),
      .result(rowmax_result)
  );

  assign vec_reduce_result = (vec_opcode_reg == OP_VEC_ROWSUM) ? rowsum_result : rowmax_result;

  localparam [31:0] NEG_LOG2E = 32'hBFB8AA3B;  // −log2(e) ≈ −1.44269504
  localparam [31:0] ONE_F32 = 32'h3F800000;

  generate
    for (genvar gi = 0; gi < BANKING; gi++) begin : g_vpu_lanes
      // VREG-to-VREG
      fp32_mul u_fma_prod (
          .clk(clk),
          .rst_n(rst_n),
          .a(vreg_rd_data_a_latch[gi]),
          .b(vreg_rd_data_b_latch[gi]),
          .result(lane_fma_prod[gi])
      );
      fp32_add u_fma_sum (
          .clk(clk),
          .rst_n(rst_n),
          .a(vec_prod_reg[gi]),
          .b(vreg_rd_data_b[gi]),  // live: FMA_LATCH redirects rd_addr_b→VREG[d]
          .result(lane_fma_sum[gi])
      );
      fp32_mul u_fmaimm_p (
          .clk(clk),
          .rst_n(rst_n),
          .a(vreg_rd_data_a_latch[gi]),
          .b(vec_imm_reg),
          .result(lane_fmaimm_prod[gi])
      );
      fp32_add u_fmaimm_s (
          .clk(clk),
          .rst_n(rst_n),
          .a(lane_fmaimm_prod_reg[gi]),  // registered — breaks VREG_mux+mul+add in one cycle
          .b(vec_bias_reg),
          .result(lane_fmaimm_sum[gi])
      );
      fp32_max u_maximum (
          .a(vreg_rd_data_a_latch[gi]),
          .b(vreg_rd_data_b_latch[gi]),
          .result(lane_maximum[gi])
      );

      // Shared fp32_exp2_bram: one BRAM-LUT instance per lane covers exp2.vreg,
      // vec.exp2, and silu (all mutually exclusive).  1-cycle registered output:
      // result valid the cycle AFTER the input is presented.  This replaces the
      // 13-FP-unit combinational chain (~38–43 K LUTs at BANKING=8).
      assign shared_exp2_in[gi] =
          (vec_opcode_reg == OP_VEC_EXP2_VREG) ? vreg_rd_data_a_latch[gi] :  // Fix 12b: latch
          (vec_opcode_reg == OP_VEC_EXP2) ? bram_dout_b_exp2_reg[gi] :  // Fix 9: registered
          silu_neg_log2e_reg[gi];  // OP_VEC_SILU (registered)
      fp32_exp2_bram u_exp2_bram (
          .clk(clk),
          .x(shared_exp2_in[gi]),
          .result(shared_exp2_out[gi])
      );
      assign lane_exp2_vreg[gi] = shared_exp2_out[gi];
      assign acc_exp2_row[gi]   = shared_exp2_out[gi];
      assign silu_exp2_val[gi]  = shared_exp2_out[gi];

      fp32_add u_sub_v (
          .clk(clk),
          .rst_n(rst_n),
          .a(vreg_rd_data_a_latch[gi]),
          .b({~vreg_rd_data_b_latch[gi][31], vreg_rd_data_b_latch[gi][30:0]}),
          .result(lane_sub_vreg[gi])
      );
      fp32_add u_add_v (
          .clk(clk),
          .rst_n(rst_n),
          .a(vreg_rd_data_a_latch[gi]),
          .b(vreg_rd_data_b_latch[gi]),
          .result(lane_add_vreg[gi])
      );
      fp32_mul u_mul_v (
          .clk(clk),
          .rst_n(rst_n),
          .a(vreg_rd_data_a_latch[gi]),
          .b(vreg_rd_data_b_latch[gi]),
          .result(lane_mul_vreg[gi])
      );
      // fp32_rsqrt_bram: 9-FP-unit combinational chain replaced by BRAM-LUT.
      // 1-cycle registered output (valid the cycle after VEC_VREG_EXEC).
      fp32_rsqrt_bram u_rsqrt_bram (
          .clk(clk),
          .x(vreg_rd_data_a_latch[gi]),
          .result(lane_rsqrt[gi])
      );

      // Single-src ACC element-wise
      // (acc_exp2_row[gi] driven by shared_exp2_out above)

      // SILU(x) = x / (1 + 2^(−x·log2e))
      fp32_mul u_silu_nle (
          .clk(clk),
          .rst_n(rst_n),
          .a(bram_dout_b[gi*32+:32]),
          .b(NEG_LOG2E),
          .result(silu_neg_log2e[gi])
      );
      // silu_exp2_val[gi] driven by shared_exp2_out above (input = silu_neg_log2e[gi] when opcode==SILU)
      fp32_add u_silu_den (
          .clk(clk),
          .rst_n(rst_n),
          .a(ONE_F32),
          .b(silu_exp2_val[gi]),
          .result(silu_denom[gi])
      );
      // fp32_recip_bram: 9-FP-unit combinational chain replaced by BRAM-LUT.
      // 1-cycle registered output (valid the cycle after VEC_ACC_NL_EXP2_LATCH).
      fp32_recip_bram u_silu_rec_bram (
          .clk(clk),
          .x(silu_denom[gi]),
          .result(silu_recip_val[gi])
      );
      // u_silu_out uses silu_x_latch (original x, latched in VEC_ACC_RD_WAIT)
      // so the multiplier gets the correct ACC element even though bram_dout_b
      // may have changed by the time silu_recip_val is valid.
      fp32_mul u_silu_out (
          .clk(clk),
          .rst_n(rst_n),
          .a(silu_x_latch[gi]),
          .b(silu_recip_val[gi]),
          .result(acc_silu_row[gi])
      );

      fp32_mul u_scl_b (
          .clk(clk),
          .rst_n(rst_n),
          .a(bcast_scalar_reg),
          .b(bram_dout_b[gi*32+:32]),
          .result(acc_scale_bcast_row[gi])
      );
      fp32_mul u_scl_c (
          .clk(clk),
          .rst_n(rst_n),
          .a(vec_vreg_latch[gi]),
          .b(bram_dout_b[gi*32+:32]),
          .result(acc_scale_col_row[gi])
      );
      fp32_add #(
          .LATENCY(1)
      ) u_sub_b (
          .clk(clk),
          .rst_n(rst_n),
          .a(bram_dout_b[gi*32+:32]),
          .b({~bcast_scalar_reg[31], bcast_scalar_reg[30:0]}),
          .result(acc_sub_bcast_row[gi])
      );
      fp32_mul u_div_b (
          .clk(clk),
          .rst_n(rst_n),
          .a(bram_dout_b[gi*32+:32]),
          .b(bcast_recip),
          .result(acc_div_bcast_row[gi])
      );
      fp32_mul u_scl_imm (
          .clk(clk),
          .rst_n(rst_n),
          .a(vec_imm_reg),
          .b(bram_dout_b[gi*32+:32]),
          .result(acc_scale_imm_row[gi])
      );

      // Two-src ACC — inputs use registered B-side data (bram_dout_b_a2_reg) to
      // break the BRAM-output → fp32 → BRAM-input combinational path.
      fp32_mul u_a2_mul (
          .clk(clk),
          .rst_n(rst_n),
          .a(vec_acc_a_row_reg[gi*32+:32]),
          .b(bram_dout_b_a2_reg[gi*32+:32]),
          .result(acc2_mul_row[gi])
      );
      fp32_add u_a2_add (
          .clk(clk),
          .rst_n(rst_n),
          .a(vec_acc_a_row_reg[gi*32+:32]),
          .b(bram_dout_b_a2_reg[gi*32+:32]),
          .result(acc2_add_row[gi])
      );
      fp32_add u_a2_sub (
          .clk(clk),
          .rst_n(rst_n),
          .a(vec_acc_a_row_reg[gi*32+:32]),
          .b({~bram_dout_b_a2_reg[gi*32+31], bram_dout_b_a2_reg[gi*32+:31]}),
          .result(acc2_sub_row[gi])
      );
    end
  endgenerate

  //---------------------------------------------
  // FSM sequential logic
  //---------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state                <= IDLE;
      done                 <= 1'b0;
      start_systolic       <= 1'b0;
      pc_load              <= 1'b0;
      pc_load_val          <= '0;
      dma_op_reg           <= '0;
      dma_spm_base         <= '0;
      dma_mat_rows         <= '0;
      dma_row_r            <= '0;
      dram_addr_reg        <= '0;
      dram_stride_reg      <= '0;
      cast_src_addr        <= '0;
      cast_dst_addr        <= '0;
      cast_vreg_dst        <= '0;
      cast_is_f32_to_f16   <= 1'b0;
      cast_to_vreg         <= 1'b0;
      vreg_wr_en           <= 1'b0;
      vreg_wr_addr         <= '0;
      vreg_wr_data         <= '0;
      vreg_rd_addr_a       <= '0;
      vreg_rd_addr_b       <= '0;
      vec_acc_src          <= '0;
      vec_acc_dst          <= '0;
      vec_acc_b            <= '0;
      vec_vreg_dst         <= '0;
      vec_vreg_a           <= '0;
      vec_vreg_b           <= '0;
      vec_vreg_d           <= '0;
      vec_opcode_reg       <= '0;
      vec_row_r            <= '0;
      vec_imm_reg          <= '0;
      vec_bias_reg         <= '0;
      vec_zero_rows        <= '0;
      vreg_data_reg        <= '0;
      vec_prod_reg         <= '0;
      lane_fmaimm_prod_reg <= '0;
      vec_acc_a_row_reg    <= '0;
      bram_dout_b_a2_reg   <= '0;
      acc2_result_reg      <= '0;
      vec_computed_row_reg <= '0;
      bcast_scalar_reg     <= '0;
      bram_dout_b_exp2_reg <= '0;
      vreg_rd_data_a_latch <= '0;
      vreg_rd_data_b_latch <= '0;
      acc2_dst_addr_reg    <= '0;
      acc_exp2_row_reg     <= '0;
      vec_vreg_latch       <= '0;
      // Stage 3: loop-control registers
      iv_body_start        <= '0;
      iv_hi                <= '0;
      iv_step              <= '0;
      iv_reg               <= '0;
      loop_iv_id           <= '0;
      loop_hi_pool_idx     <= '0;
      loop_lo              <= '0;
      // Stage 4: int4 registers
      i4_subrow_r          <= '0;
      i4_dm_latch          <= '0;
      mxu_acc_dst_reg      <= '0;
      mxu_spm_a_reg        <= '0;
      mxu_spm_b_i4_reg     <= '0;
      mxu_zero_point_reg   <= '0;
      mxu_i4_mode_reg      <= 1'b0;
      mxu_dq_mode_reg      <= 1'b0;
      // ISA v2 / Gap-1: .mi DMA registers
      dma_iv_stride_a_reg  <= '0;
      dma_iv_id_pair_reg   <= '0;
      dma_iv_stride_b_reg  <= '0;
      // ISA v2 / Gap-2: async DMA + flush.slot registers
      dma_async_slot_id    <= '0;
      dma_slot_done        <= 2'b00;
      flush_slot_mask      <= '0;
      flags_reg            <= '0;
      // Unified pool-fetch sub-FSM context
      pf_step              <= 4'd0;
      pf_max_step          <= 4'd0;
      pf_addr_base         <= '0;
      pf_next_state_tag    <= PF_TO_DMA_ROW_REQ;
      // BRAM-LUT NL pipeline
      silu_x_latch         <= '0;
      silu_neg_log2e_reg   <= '0;
    end else begin
      start_systolic <= 1'b0;
      pc_load        <= 1'b0;
      done           <= 1'b0;
      vreg_wr_en     <= 1'b0;  // default: no write

      case (state)
        IDLE: begin
          if (start) begin
            // Gap 3: PC starts at program_id * PROGRAM_SLOT_SIZE (= << 8).
            // program_id is slv_reg13[3:0]; bits [3:0] select 1 of 16 slots.
            pc_load     <= 1'b1;
            pc_load_val <= (program_id_csr[3:0]) << 8;
            state       <= FETCH_1;
          end
        end

        FETCH_1: state <= FETCH_2;
        FETCH_2: state <= FETCH_3;
        FETCH_3: state <= EXEC_DISPATCH;

        EXEC_DISPATCH: begin
          flags_reg <= flags;  // latch flags; valid through all subsequent pool-fetch states
          case (op)
            OP_MATMUL_TILE: begin
              start_systolic  <= 1'b1;
              mxu_i4_mode_reg <= 1'b0;  // ensure fp16 mode
              state           <= WAIT_MXU;
            end
            OP_DMA_LOAD_SPM, OP_DMA_LOAD_ACC: begin
              // Load: pool[op1]=dram_addr, pool[op1+1]=dram_stride
              //       Gap-2 async: pool[op1+2]=slot_id (pf_max_step=2)
              dma_op_reg        <= op;
              dma_spm_base      <= op0[L1_ADDR_WIDTH-1:0];
              dma_mat_rows      <= op2[15:8] + 8'd1;
              dma_row_r         <= 8'd0;
              vec_opcode_reg    <= '0;
              pf_addr_base      <= op1;
              pf_step           <= 4'd0;
              pf_max_step       <= flags[FLAG_ASYNC_BIT] ? 4'd2 : 4'd1;
              pf_next_state_tag <= PF_TO_DMA_ROW_REQ;
              state             <= POOL_FETCH_REQ;
            end
            OP_DMA_STORE_SPM, OP_DMA_STORE_ACC: begin
              // Store: pool[op1]=dram_addr, pool[op1+1]=dram_stride
              //        Gap-2 async: pool[op1+2]=slot_id (pf_max_step=2)
              dma_op_reg        <= op;
              dma_spm_base      <= op0[L1_ADDR_WIDTH-1:0];
              dma_mat_rows      <= op2[15:8] + 8'd1;
              dma_row_r         <= 8'd0;
              vec_opcode_reg    <= '0;
              pf_addr_base      <= op1;
              pf_step           <= 4'd0;
              pf_max_step       <= flags[FLAG_ASYNC_BIT] ? 4'd2 : 4'd1;
              pf_next_state_tag <= PF_TO_DMA_ROW_REQ;
              state             <= POOL_FETCH_REQ;
            end
            OP_CAST_F32_F16, OP_CAST_F16_F32, OP_CAST_F16_F32_VREG: begin
              cast_src_addr      <= op1[L1_ADDR_WIDTH-1:0];
              cast_dst_addr      <= op0[L1_ADDR_WIDTH-1:0];
              cast_vreg_dst      <= op0[2:0];
              cast_is_f32_to_f16 <= (op == OP_CAST_F32_F16);
              cast_to_vreg       <= (op == OP_CAST_F16_F32_VREG);
              state              <= CAST_REQ;
            end

            // ---- ACC→VREG reductions ----
            OP_VEC_ROWMAX, OP_VEC_ROWSUM: begin
              vec_vreg_dst   <= op0[2:0];
              vec_acc_src    <= op1[L1_ADDR_WIDTH-1:0];
              vec_opcode_reg <= op;
              vec_row_r      <= 8'd0;
              vreg_data_reg  <= '0;
              state          <= VEC_ACC_VREG_REQ;
            end

            // ---- Single-src ACC→ACC without VREG (exp2, silu) ----
            OP_VEC_EXP2, OP_VEC_SILU: begin
              vec_acc_dst    <= op0[L1_ADDR_WIDTH-1:0];
              vec_acc_src    <= op1[L1_ADDR_WIDTH-1:0];
              vec_opcode_reg <= op;
              vec_row_r      <= 8'd0;
              state          <= VEC_ACC_RD_REQ;
            end

            // ---- Single-src ACC→ACC with VREG broadcast ----
            OP_VEC_SCALE_BCAST, OP_VEC_SCALE_COL_BCAST, OP_VEC_SUB_BCAST, OP_VEC_DIV_BCAST: begin
              vec_acc_dst    <= op0[L1_ADDR_WIDTH-1:0];
              vec_acc_src    <= op1[L1_ADDR_WIDTH-1:0];
              vec_vreg_a     <= op2[2:0];
              vec_opcode_reg <= op;
              vec_row_r      <= 8'd0;
              // Read VREG[a] first; issue address now, data in VEC_VREG_ISSUE
              vreg_rd_addr_a <= op2[2:0];
              vreg_rd_addr_b <= '0;
              state          <= VEC_VREG_ISSUE;
            end

            // ---- Scale by fp32 immediate (pool dereference) ----
            OP_VEC_SCALE_IMM: begin
              vec_acc_dst       <= op0[L1_ADDR_WIDTH-1:0];
              vec_acc_src       <= op1[L1_ADDR_WIDTH-1:0];
              vec_opcode_reg    <= op;
              vec_row_r         <= 8'd0;
              pf_addr_base      <= op2;
              pf_step           <= 4'd0;
              pf_max_step       <= 4'd0;
              pf_next_state_tag <= PF_TO_VEC_ACC_RD_REQ;
              state             <= POOL_FETCH_REQ;
            end

            // ---- Two-src ACC element-wise ops ----
            OP_VEC_MUL_ACC, OP_VEC_ADD_ACC, OP_VEC_SUB_ACC: begin
              vec_acc_dst    <= op0[L1_ADDR_WIDTH-1:0];
              vec_acc_src    <= op1[L1_ADDR_WIDTH-1:0];
              vec_acc_b      <= op2[L1_ADDR_WIDTH-1:0];
              vec_opcode_reg <= op;
              vec_row_r      <= 8'd0;
              state          <= VEC_ACC2_RD_A_REQ;
            end

            // ---- vec.zero.acc ----
            OP_VEC_ZERO_ACC: begin
              vec_acc_dst    <= op0[L1_ADDR_WIDTH-1:0];
              vec_zero_rows  <= op1[7:0];
              vec_row_r      <= 8'd0;
              vec_opcode_reg <= op;
              state          <= VEC_ZERO_ACC_WRITE;
            end

            // ---- VREG-only ops without pool ----
            OP_VEC_MAXIMUM, OP_VEC_SUB_VREG, OP_VEC_ADD_VREG, OP_VEC_MUL_VREG,
            OP_VEC_EXP2_VREG, OP_VEC_RSQRT, OP_VEC_COPY_VREG: begin
              vec_vreg_dst   <= op0[2:0];
              vec_vreg_a     <= op1[2:0];
              vec_vreg_b     <= op2[2:0];
              vec_opcode_reg <= op;
              vreg_rd_addr_a <= op1[2:0];
              vreg_rd_addr_b <= op2[2:0];
              state          <= VEC_VREG_ISSUE;
            end

            // ---- vec.fma: vreg_d/vreg_b indices from pool[op2] ----
            OP_VEC_FMA: begin
              vec_vreg_dst      <= op0[2:0];
              vec_vreg_a        <= op1[2:0];
              vec_opcode_reg    <= op;
              pf_addr_base      <= op2;
              pf_step           <= 4'd0;
              pf_max_step       <= 4'd0;
              pf_next_state_tag <= PF_TO_VEC_VREG_ISSUE;
              state             <= POOL_FETCH_REQ;
            end

            // ---- vec.movi: fp32 imm from pool[op2] ----
            OP_VEC_MOVI: begin
              vec_vreg_dst      <= op0[2:0];
              vec_zero_rows     <= op1[7:0];
              vec_opcode_reg    <= op;
              pf_addr_base      <= op2;
              pf_step           <= 4'd0;
              pf_max_step       <= 4'd0;
              pf_next_state_tag <= PF_TO_VEC_VREG_EXEC;
              state             <= POOL_FETCH_REQ;
            end

            // ---- vec.fma.imm: scale=pool[op2], bias=pool[op2+1] ----
            OP_VEC_FMA_IMM: begin
              vec_vreg_dst      <= op0[2:0];
              vec_vreg_a        <= op1[2:0];
              vec_opcode_reg    <= op;
              pf_addr_base      <= op2;
              pf_step           <= 4'd0;
              pf_max_step       <= 4'd1;  // 2 steps: scale (0) + bias (1)
              pf_next_state_tag <= PF_TO_VEC_VREG_ISSUE;
              state             <= POOL_FETCH_REQ;
            end

            // ---- vec.load.imm: load BANKING fp32 values from pool[op1..op1+7] ----
            // pf_step doubles as the row index; vreg_data_reg[pf_step] latched per step.
            OP_VEC_LOAD_IMM: begin
              vec_vreg_dst      <= op0[2:0];
              vreg_data_reg     <= '0;
              pf_addr_base      <= op1;
              pf_step           <= 4'd0;
              pf_max_step       <= (BANKING - 1);  // BANKING steps (0..BANKING-1)
              pf_next_state_tag <= PF_TO_VEC_ACC_VREG_WRITE;
              state             <= POOL_FETCH_REQ;
            end

            // ---- Stage 3: loop control ----
            OP_LOOP_BEGIN: begin
              // op0[1:0]=iv_id; op1=pool index of hi; op2[15:8]=lo; op2[7:0]=step
              // pool[op1] = hi bound; body_start latched in POOL_FETCH_WAIT.
              loop_iv_id        <= op0[1:0];
              loop_hi_pool_idx  <= op1;
              loop_lo           <= op2[15:8];
              iv_step[op0[1:0]] <= op2[7:0];
              pf_addr_base      <= op1;  // pool_base + op1 + 0 = hi entry
              pf_step           <= 4'd0;
              pf_max_step       <= 4'd0;
              pf_next_state_tag <= PF_TO_FETCH_1;
              state             <= POOL_FETCH_REQ;
            end

            OP_LOOP_END: begin
              loop_iv_id <= op0[1:0];
              state      <= LOOP_END_EXEC;
            end

            // ---- Stage 4: int4 DMA nibble unpack (sync only — no async flag) ----
            OP_DMA_LOAD_SPM_I4: begin
              dma_op_reg        <= op;
              dma_spm_base      <= op0[L1_ADDR_WIDTH-1:0];
              dma_mat_rows      <= op2[15:8] + 8'd1;
              dma_row_r         <= 8'd0;
              vec_opcode_reg    <= '0;
              pf_addr_base      <= op1;
              pf_step           <= 4'd0;
              pf_max_step       <= 4'd1;  // pool[op1]=addr, pool[op1+1]=stride
              pf_next_state_tag <= PF_TO_DMA_ROW_REQ;
              state             <= POOL_FETCH_REQ;
            end

            // ---- ISA v2 / Gap-1: MultiLinearAddr (.mi) DMA family (0x48–0x4E) ----
            // Pool layout (sync 5): [dram_addr, row_step, iv_stride_a, iv_id_pair, iv_stride_b]
            // Gap-2 async: slot_id appended at pool[op1+5] (pf_max_step=5)
            OP_DMA_LOAD_SPM_MI, OP_DMA_LOAD_ACC_MI, OP_DMA_LOAD_SPM_I4_MI,
            OP_DMA_LOAD_SPM_I8_MI, OP_DMA_STORE_SPM_MI,
            OP_DMA_STORE_ACC_MI, OP_DMA_STORE_SPM_I8_MI: begin
              dma_op_reg        <= op;
              dma_spm_base      <= op0[L1_ADDR_WIDTH-1:0];
              dma_mat_rows      <= op2[15:8] + 8'd1;
              dma_row_r         <= 8'd0;
              vec_opcode_reg    <= '0;
              pf_addr_base      <= op1;
              pf_step           <= 4'd0;
              pf_max_step       <= flags[FLAG_ASYNC_BIT] ? 4'd5 : 4'd4;
              pf_next_state_tag <= PF_TO_DMA_ROW_REQ;
              state             <= POOL_FETCH_REQ;
            end

            // ---- Stage 4: fused int4 matmul: pool[op2]=spm_b, pool[op2+1]=zero_point ----
            OP_MATMUL_TILE_I8: begin
              mxu_acc_dst_reg   <= op0[L1_ADDR_WIDTH-1:0];
              mxu_spm_a_reg     <= op1[L1_ADDR_WIDTH-1:0];
              mxu_i4_mode_reg   <= 1'b0;  // set to 1 at POOL_FETCH_WAIT step 1
              vec_opcode_reg    <= OP_MATMUL_TILE_I8;
              pf_addr_base      <= op2;
              pf_step           <= 4'd0;
              pf_max_step       <= 4'd1;  // pool[op2]=spm_b, pool[op2+1]=zero_point
              pf_next_state_tag <= PF_TO_WAIT_MXU;
              state             <= POOL_FETCH_REQ;
            end

            // ---- ISA v1.4: fused per-column dequant+scale matmul ----
            // Pool layout: pool[op2]=spm_b, pool[op2+1]=zero_point, pool[op2+2]=vreg_scale_idx
            // The weights in spm_b are nibble-unpacked INT4 (loaded via dma.load.spm.i4).
            // scale_vec is read from VREG[vreg_scale_idx] (async, 8 fp32 elements).
            OP_MATMUL_TILE_DQ: begin
              mxu_acc_dst_reg   <= op0[L1_ADDR_WIDTH-1:0];
              mxu_spm_a_reg     <= op1[L1_ADDR_WIDTH-1:0];
              mxu_i4_mode_reg   <= 1'b0;  // set to 1 at POOL_FETCH_WAIT step 2
              mxu_dq_mode_reg   <= 1'b0;  // set to 1 at POOL_FETCH_WAIT step 2
              vec_opcode_reg    <= OP_MATMUL_TILE_DQ;
              pf_addr_base      <= op2;
              pf_step           <= 4'd0;
              pf_max_step       <= 4'd2;  // 3 steps: spm_b(0), zero_point(1), vreg_scale_idx(2)
              pf_next_state_tag <= PF_TO_WAIT_MXU;
              state             <= POOL_FETCH_REQ;
            end

            // ---- Gap 3: OP_BARRIER (0x54) — synchronization point ----
            // Asserts done (1-cycle pulse, same as HALT_STATE) so the host sees
            // compute_done and instr_ready goes back to 1.  The sequencer then
            // stalls in BARRIER_WAIT until the host re-asserts start (a second
            // compute doorbell).  On re-start, execution resumes at the
            // instruction after the barrier (PC already advanced by decode pipeline).
            //
            // IMPORTANT: done is only pulsed HERE (at EXEC_DISPATCH), not
            // continuously in BARRIER_WAIT.  Pulsing every cycle in BARRIER_WAIT
            // caused compute_ctrl to see done=1 (stale) when it entered CC_RUNNING
            // on the second execute, firing compute_done prematurely before the
            // post-barrier program even started executing.
            OP_BARRIER: begin
              done  <= 1'b1;  // 1-cycle done pulse — mirrors HALT_STATE semantics
              state <= BARRIER_WAIT;
            end

            // ---- Gap 3: OP_LOOP_BEGIN_CSR (0x55) — CSR-sourced trip count ----
            // Like loop.begin (0x50) but the hi bound comes from kernel_arg_csr[op1]
            // instead of the literal pool. No pool fetch needed.
            // op0[1:0] = iv_id; op1[1:0] = kernel_arg index (0-3);
            // op2[15:8] = lo; op2[7:0] = step.
            // loop_hi_pool_idx is reused here to carry the CSR index (op1[1:0])
            // into LOOP_CSR_LATCH.
            OP_LOOP_BEGIN_CSR: begin
              loop_iv_id        <= op0[1:0];
              loop_hi_pool_idx  <= op1;  // repurposed: carries csr_idx for LOOP_CSR_LATCH
              loop_lo           <= op2[15:8];
              iv_step[op0[1:0]] <= op2[7:0];
              state             <= LOOP_CSR_LATCH;
            end

            // ---- ISA v2 / Gap-2: flush.slot (0x53) — masked async DMA fence ----
            // op0[15:0] = slot_mask; stall until dma_slot_done[i]=1 for all i in mask.
            // Clears the matched done bits on exit so they can be reused.
            OP_FLUSH_SLOT: begin
              flush_slot_mask <= op0;
              state           <= FLUSH_SLOT_WAIT;
            end

            OP_FLUSH: state <= HALT_STATE;
            default:  state <= HALT_STATE;
          endcase
        end

        WAIT_MXU: begin
          if (systolic_done) begin
            mxu_i4_mode_reg <= 1'b0;
            mxu_dq_mode_reg <= 1'b0;
            state           <= FETCH_1;
          end
        end

        HALT_STATE: begin
          done  <= 1'b1;
          state <= IDLE;
        end

        // ======================================================================
        // Unified pool-fetch sub-FSM.
        //
        // POOL_FETCH_REQ: IRAM addrb is driven combinationally to
        //   pool_base + pf_addr_base + pf_step this cycle; BRAM registers the
        //   address and the read data arrives at the next posedge.
        //
        // POOL_FETCH_WAIT: current_instr[31:0] holds pool[pf_addr_base + pf_step].
        //   Latch the appropriate register (keyed on vec_opcode_reg + pf_step),
        //   then either loop back (pf_step < pf_max_step) or route to the state
        //   named by pf_next_state_tag (pf_step == pf_max_step).
        //
        // Pool address formula (pool_fetch_addr assign above):
        //   pool_base + pf_addr_base + pf_step
        //
        // Step map (pf_max_step → last step inclusive):
        //   DMA classic sync  : max=1 (addr=0, stride=1)
        //   DMA classic async : max=2 (addr=0, stride=1, slot_id=2)
        //   DMA .mi sync      : max=4 (addr=0, stride=1, iv_stride_a=2, iv_id_pair=3, iv_stride_b=4)
        //   DMA .mi async     : max=5 (… + slot_id=5)
        //   DMA_LOAD_SPM_I4   : max=1 (no async variant)
        //   VEC_SCALE_IMM     : max=0 (imm=0)
        //   VEC_FMA           : max=0 (vreg_d/b packed=0)
        //   VEC_MOVI          : max=0 (imm=0)
        //   VEC_FMA_IMM       : max=1 (scale=0, bias=1)
        //   VEC_LOAD_IMM      : max=BANKING-1 (8 pool entries → vreg_data_reg[0..7])
        //   MATMUL_TILE_I8    : max=1 (spm_b=0, zero_point=1)
        //   MATMUL_TILE_DQ    : max=2 (spm_b=0, zero_point=1, vreg_scale_idx=2)
        //   LOOP_BEGIN        : max=0 (hi bound=0 → latch IV frame + pc_enable)
        // ======================================================================
        POOL_FETCH_REQ: state <= POOL_FETCH_WAIT;

        POOL_FETCH_WAIT: begin
          // ------------------------------------------------------------------
          // Per-step latch table.
          // Non-DMA opcodes (vec_opcode_reg != 0) dispatch on step.
          // DMA path (vec_opcode_reg == 0) dispatches on pf_step.
          // LOOP_BEGIN and VREG_LOAD (vec_opcode_reg == 0 too) are discriminated
          // by pf_next_state_tag before falling through to the DMA step table.
          // ------------------------------------------------------------------
          if (vec_opcode_reg == OP_VEC_SCALE_IMM) begin
            // step 0: fp32 immediate
            vec_imm_reg <= current_instr[31:0];

          end else if (vec_opcode_reg == OP_VEC_FMA) begin
            // step 0: packed (vreg_b[2:0] << 3 | vreg_d[2:0])
            vec_vreg_d     <= current_instr[2:0];
            vec_vreg_b     <= current_instr[5:3];
            vreg_rd_addr_a <= vec_vreg_a;
            vreg_rd_addr_b <= current_instr[5:3];

          end else if (vec_opcode_reg == OP_VEC_MOVI) begin
            // step 0: fp32 immediate
            vec_imm_reg <= current_instr[31:0];

          end else if (vec_opcode_reg == OP_VEC_FMA_IMM) begin
            if (pf_step == 4'd0) vec_imm_reg <= current_instr[31:0];  // scale
            else begin  // step 1: bias; also set VREG read addresses for ISSUE
              vec_bias_reg   <= current_instr[31:0];
              vreg_rd_addr_a <= vec_vreg_a;
              vreg_rd_addr_b <= '0;
            end

          end else if (vec_opcode_reg == OP_MATMUL_TILE_I8) begin
            if (pf_step == 4'd0) begin
              mxu_spm_b_i4_reg <= current_instr[L1_ADDR_WIDTH-1:0];  // spm_b row
            end else begin  // step 1: zero_point; i4 mode set; MXU starts at PF_TO_WAIT_MXU
              mxu_zero_point_reg <= current_instr[7:0];
              mxu_i4_mode_reg    <= 1'b1;
            end

          end else if (vec_opcode_reg == OP_MATMUL_TILE_DQ) begin
            // Step 0: spm_b SPM row index (nibble-packed weights, from dma.load.spm.i4)
            // Step 1: zero_point scalar (uniform u8, same as i4 mode)
            // Step 2: vreg_scale_idx — set VREG read address; arm modes; start_systolic
            //         fires this same posedge via PF_TO_WAIT_MXU routing below.
            //         MXU S_IDLE latches scale_vec_reg the NEXT posedge (T+1), by which
            //         time vreg_rd_addr_a NBA from T has propagated → vreg_rd_data_a valid.
            if (pf_step == 4'd0) begin
              mxu_spm_b_i4_reg <= current_instr[L1_ADDR_WIDTH-1:0];
            end else if (pf_step == 4'd1) begin
              mxu_zero_point_reg <= current_instr[7:0];
            end else begin  // step 2: vreg_scale_idx; arm dq+i4 modes; MXU starts at PF_TO_WAIT_MXU
              vreg_rd_addr_a  <= current_instr[2:0];  // async: scale_vec valid combinationally
              mxu_i4_mode_reg <= 1'b1;  // enable nibble-unpack address mux
              mxu_dq_mode_reg <= 1'b1;  // enable per-column scale path in MXU
            end

          end else if (pf_next_state_tag == PF_TO_FETCH_1) begin
            // LOOP_BEGIN: latch IV frame from the single pool entry (hi bound).
            // pc_enable fires this cycle via the assign above (POOL_FETCH_WAIT + tag + last step).
            iv_hi[loop_iv_id]         <= current_instr[31:0];
            iv_reg[loop_iv_id]        <= {24'h0, loop_lo};
            iv_body_start[loop_iv_id] <= pc_val + (1);

          end else if (pf_next_state_tag == PF_TO_VEC_ACC_VREG_WRITE) begin
            // VREG_LOAD (vec.load.imm): latch one fp32 per step.
            // pf_step serves as the row index; no separate vec_row_r needed here.
            vreg_data_reg[pf_step] <= current_instr[31:0];

          end else begin
            // DMA path (vec_opcode_reg == 0, all DMA opcodes including .mi + async)
            case (pf_step)
              4'd0:    dram_addr_reg <= current_instr[31:0];
              4'd1:    dram_stride_reg <= current_instr[31:0];
              4'd2: begin
                // .mi step 2 = iv_stride_a; classic async step 2 = slot_id
                if (dma_is_mi_mode) dma_iv_stride_a_reg <= current_instr[31:0];
                else dma_async_slot_id <= current_instr[0];
              end
              4'd3:    dma_iv_id_pair_reg <= current_instr[31:0];  // .mi only
              4'd4:    dma_iv_stride_b_reg <= current_instr[31:0];  // .mi only
              4'd5:    dma_async_slot_id <= current_instr[0];  // .mi async slot_id
              default: ;
            endcase
          end

          // ------------------------------------------------------------------
          // Step counter + routing
          // ------------------------------------------------------------------
          if (pf_step == pf_max_step) begin
            pf_step <= 4'd0;  // reset for next invocation
            case (pf_next_state_tag)
              PF_TO_DMA_ROW_REQ:        state <= DMA_ROW_REQ;
              PF_TO_VEC_ACC_RD_REQ:     state <= VEC_ACC_RD_REQ;
              PF_TO_VEC_VREG_ISSUE:     state <= VEC_VREG_ISSUE;
              PF_TO_VEC_VREG_EXEC:      state <= VEC_VREG_EXEC;
              PF_TO_WAIT_MXU: begin
                start_systolic <= 1'b1;
                state          <= WAIT_MXU;
              end
              PF_TO_FETCH_1:            state <= FETCH_1;
              PF_TO_VEC_ACC_VREG_WRITE: state <= VEC_ACC_VREG_WRITE;
              default:                  state <= FETCH_1;
            endcase
          end else begin
            pf_step <= pf_step + 4'd1;
            state   <= POOL_FETCH_REQ;
          end
        end  // POOL_FETCH_WAIT

        // DMA row loop.  DMA_ROW_REQ: issue the first memory read.
        // DMA_ROW_WAIT: capture the response and issue the second memory write.
        DMA_ROW_REQ: state <= DMA_ROW_WAIT;

        DMA_ROW_WAIT: begin
          if (dma_op_reg == OP_DMA_LOAD_SPM_I4 || dma_op_reg == OP_DMA_LOAD_SPM_I4_MI) begin
            // i4 / i4.mi: latch 256-bit DM word; subrow loop runs in DMA_I4_WRITE.
            i4_dm_latch <= dm_dout;
            i4_subrow_r <= 3'd0;
            state       <= DMA_I4_WRITE;
          end else begin
            dma_row_r <= dma_row_r + 8'd1;
            if (dma_row_r == dma_mat_rows - 8'd1) begin
              // Gap-2: on async DMA completion, mark slot done
              if (flags_reg[FLAG_ASYNC_BIT]) dma_slot_done[dma_async_slot_id] <= 1'b1;
              state <= FETCH_1;  // pc_enable also fires this cycle via assign above
            end else state <= DMA_ROW_REQ;
          end
        end

        // DMA_I4_WRITE: write one SPM_I8 row per cycle (8 subrows per DM word).
        // BRAM write is driven in the always @* routing block below.
        // When the last subrow of the last DM word is written, pc_enable fires and
        // we return to FETCH_1 (or continue to DMA_ROW_REQ for the next DM word).
        DMA_I4_WRITE: begin
          i4_subrow_r <= i4_subrow_r + 3'd1;
          if (i4_subrow_r == 3'd7) begin
            dma_row_r <= dma_row_r + 8'd1;
            if (dma_row_r == dma_mat_rows - 8'd1) begin
              if (flags_reg[FLAG_ASYNC_BIT]) dma_slot_done[dma_async_slot_id] <= 1'b1;
              state <= FETCH_1;  // pc_enable fires via DMA_I4_WRITE branch in assign
            end else state <= DMA_ROW_REQ;
          end
        end

        // Cast: issue BRAM read at cast_src_addr; data arrives in CAST_WAIT.
        CAST_REQ: state <= CAST_WAIT;

        // Cast: BRAM data (bram_dout_b) is valid; write converted result.
        CAST_WAIT: begin
          if (cast_to_vreg) begin
            // cast.f16.f32.vreg: fp32 results go to VREG
            vreg_wr_en   <= 1'b1;
            vreg_wr_addr <= cast_vreg_dst;
            for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= cast_fp32_from_fp16[i];
          end
          // BRAM write for non-vreg cast is handled in always @* below.
          state <= FETCH_1;
        end

        // ---- Stage 2 VEC states ----

        // Issue VREG read addresses; async read → data valid combinationally
        // and captured at next posedge (VEC_VREG_EXEC).
        VEC_VREG_ISSUE: begin
          // Fix 12b: latch VREG async read outputs here — breaks VREG-mux→FP-op path.
          // rd_addr_a/b were set at EXEC_DISPATCH (or pool-fetch); async VREG data valid
          // combinationally.  Latching here adds 0 cycles (VEC_VREG_ISSUE was already a
          // dedicated wait).  FMA_LATCH redirects rd_addr_b→VREG[d] after this latch, so
          // u_fma_sum .b must use live vreg_rd_data_b in VEC_VREG_EXEC (see generate block).
          for (int i = 0; i < BANKING; i++) vreg_rd_data_a_latch[i] <= vreg_rd_data_a[i];
          for (int i = 0; i < BANKING; i++) vreg_rd_data_b_latch[i] <= vreg_rd_data_b[i];
          if (vec_opcode_reg == OP_VEC_FMA || vec_opcode_reg == OP_VEC_FMA_IMM)
            state <= VEC_FMA_LATCH;
          else state <= VEC_VREG_EXEC;
        end

        // FMA/FMA_IMM: latch multiply results; redirect rd_addr_b → VREG[d] for FMA only.
        VEC_FMA_LATCH: begin
          for (int i = 0; i < BANKING; i++) vec_prod_reg[i] <= lane_fma_prod[i];
          for (int i = 0; i < BANKING; i++) lane_fmaimm_prod_reg[i] <= lane_fmaimm_prod[i];
          if (vec_opcode_reg == OP_VEC_FMA) vreg_rd_addr_b <= vec_vreg_d;
          state <= VEC_VREG_EXEC;
        end

        // VREG data valid; compute result and write back.
        VEC_VREG_EXEC: begin
          case (vec_opcode_reg)
            // --- bcast ops: latch VREG[a] then start ACC loop ---
            OP_VEC_SCALE_BCAST, OP_VEC_SCALE_COL_BCAST, OP_VEC_SUB_BCAST, OP_VEC_DIV_BCAST: begin
              for (int i = 0; i < BANKING; i++) vec_vreg_latch[i] <= vreg_rd_data_a_latch[i];
              vreg_wr_en <= 1'b0;
              state      <= VEC_ACC_RD_REQ;
            end

            // --- fma: acc product_reg + VREG[d] (vreg_rd_data_b now valid) ---
            OP_VEC_FMA: begin
              vreg_wr_en   <= 1'b1;
              vreg_wr_addr <= vec_vreg_dst;
              for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= lane_fma_sum[i];
              state <= FETCH_1;
            end

            // --- fma.imm: VREG[src]*scale + bias ---
            OP_VEC_FMA_IMM: begin
              vreg_wr_en   <= 1'b1;
              vreg_wr_addr <= vec_vreg_dst;
              for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= lane_fmaimm_sum[i];
              state <= FETCH_1;
            end

            // --- movi: fill [0:zero_rows) with imm ---
            OP_VEC_MOVI: begin
              vreg_wr_en   <= 1'b1;
              vreg_wr_addr <= vec_vreg_dst;
              for (int i = 0; i < BANKING; i++) begin
                if (i < (vec_zero_rows)) vreg_wr_data[i] <= vec_imm_reg;
                else vreg_wr_data[i] <= '0;
              end
              state <= FETCH_1;
            end

            // --- vec.maximum ---
            OP_VEC_MAXIMUM: begin
              vreg_wr_en   <= 1'b1;
              vreg_wr_addr <= vec_vreg_dst;
              for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= lane_maximum[i];
              state <= FETCH_1;
            end

            // --- vec.exp2.vreg ---
            // Fix 12b: vreg_rd_data_a_latch (captured at VEC_VREG_ISSUE) drives exp2_bram.x
            // via shared_exp2_in; no extra capture needed here.  exp2_bram is now 3-cycle —
            // VEC_VREG_EXP2_ADDR/LATCH absorb cycles 1 and 2 before VEC_VREG_NL_LATCH.
            OP_VEC_EXP2_VREG: begin
              vreg_wr_en <= 1'b0;
              state      <= VEC_VREG_EXP2_ADDR;
            end

            // --- vec.sub.vreg ---
            OP_VEC_SUB_VREG: begin
              vreg_wr_en   <= 1'b1;
              vreg_wr_addr <= vec_vreg_dst;
              for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= lane_sub_vreg[i];
              state <= FETCH_1;
            end

            // --- vec.add.vreg ---
            OP_VEC_ADD_VREG: begin
              vreg_wr_en   <= 1'b1;
              vreg_wr_addr <= vec_vreg_dst;
              for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= lane_add_vreg[i];
              state <= FETCH_1;
            end

            // --- vec.mul.vreg ---
            OP_VEC_MUL_VREG: begin
              vreg_wr_en   <= 1'b1;
              vreg_wr_addr <= vec_vreg_dst;
              for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= lane_mul_vreg[i];
              state <= FETCH_1;
            end

            // --- vec.rsqrt ---
            // BRAM-LUT rsqrt: input presented this cycle; result valid next cycle.
            OP_VEC_RSQRT: begin
              vreg_wr_en <= 1'b0;
              state      <= VEC_VREG_NL_LATCH;
            end

            // --- vec.copy.vreg ---
            OP_VEC_COPY_VREG: begin
              vreg_wr_en   <= 1'b1;
              vreg_wr_addr <= vec_vreg_dst;
              for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= vreg_rd_data_a_latch[i];
              state <= FETCH_1;
            end

            default: state <= FETCH_1;
          endcase
        end  // VEC_VREG_EXEC

        // ---- exp2.vreg pipeline states (3-cycle exp2_bram, Fix 11+12) ----
        // VEC_VREG_EXEC:       latch_a drives exp2_bram.x; _r2 regs capture (cycle 0).
        // VEC_VREG_EXP2_ADDR:  _r2 → u_f_sub; f_fp32_r captures (cycle 1).
        // VEC_VREG_EXP2_LATCH: f_fp32_r → BRAM addr; BRAM captures rom_out_r (cycle 2).
        // VEC_VREG_NL_LATCH:   result valid (cycle 3); write to VREG.
        VEC_VREG_EXP2_ADDR:  state <= VEC_VREG_EXP2_LATCH;
        VEC_VREG_EXP2_LATCH: state <= VEC_VREG_NL_LATCH;

        // ---- NL VREG latch: exp2_bram / rsqrt_bram output valid ----
        // For OP_VEC_EXP2_VREG: entered from VEC_VREG_EXP2_LATCH (Fix 11, 2-cycle exp2_bram).
        // For OP_VEC_RSQRT:     entered from VEC_VREG_EXEC (unchanged, 1-cycle rsqrt_bram).
        VEC_VREG_NL_LATCH: begin
          vreg_wr_en   <= 1'b1;
          vreg_wr_addr <= vec_vreg_dst;
          if (vec_opcode_reg == OP_VEC_EXP2_VREG) begin
            for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= lane_exp2_vreg[i];
          end else begin  // OP_VEC_RSQRT
            for (int i = 0; i < BANKING; i++) vreg_wr_data[i] <= lane_rsqrt[i];
          end
          state <= FETCH_1;
        end

        // ---- ACC→VREG reduction (rowmax / rowsum) ----
        // 4-stage pipeline: REQ → PIPE0 (L0 latch) → PIPE1 (u_rsum_l1 stage-1 FF)
        //                       → PIPE2 (L1 latch, LATENCY=1 result ready) → WAIT (final).
        // u_rsum_l1_{0,1} use fp32_add LATENCY=1 to break the 11.5 ns combinational chain.
        VEC_ACC_VREG_REQ: state <= VEC_ACC_VREG_PIPE0;

        VEC_ACC_VREG_PIPE0: state <= VEC_ACC_VREG_PIPE1;  // rsum/rmax_l0_reg latched by always_ff

        VEC_ACC_VREG_PIPE1: state <= VEC_ACC_VREG_PIPE2;  // u_rsum_l1 LATENCY=1 stage-1 FF captures

        VEC_ACC_VREG_PIPE2: state <= VEC_ACC_VREG_WAIT;  // rsum/rmax_l1_reg latched by always_ff

        VEC_ACC_VREG_WAIT: begin
          vreg_data_reg[vec_row_r] <= vec_reduce_result;
          vec_row_r                <= vec_row_r + 8'd1;
          if (vec_row_r == (BANKING - 1)) state <= VEC_ACC_VREG_WRITE;
          else state <= VEC_ACC_VREG_REQ;
        end

        VEC_ACC_VREG_WRITE: begin
          vreg_wr_en   <= 1'b1;
          vreg_wr_addr <= vec_vreg_dst;
          vreg_wr_data <= vreg_data_reg;
          state        <= FETCH_1;
        end

        // ---- Single-src ACC element-wise (exp2, silu, scale.bcast, etc.) ----
        VEC_ACC_RD_REQ: begin
          bcast_scalar_reg <= bcast_scalar;  // registered for VEC_ACC_RD_WAIT fp32 computation
          state <= VEC_ACC_RD_WAIT;
        end

        VEC_ACC_RD_WAIT: begin
          if (vec_opcode_reg == OP_VEC_EXP2) begin
            // Fix 9: latch bram_dout_b so exp2_bram addr is driven from a FF (not directly
            // from L1 BRAM output).  exp2_bram samples bram_dout_b_exp2_reg in EXP2_ADDR.
            for (int gi2 = 0; gi2 < BANKING; gi2++)
            bram_dout_b_exp2_reg[gi2] <= bram_dout_b[gi2*32+:32];
            state <= VEC_ACC_EXP2_ADDR;
          end else if (vec_opcode_reg == OP_VEC_SILU) begin
            // SILU: latch x for later multiply and register silu_neg_log2e (fp32_mul output).
            // fp32_mul is ~10 ns — must not chain into exp2_bram addr (another ~10 ns) in one cycle.
            // VEC_ACC_SILU_MUL presents silu_neg_log2e_reg to exp2_bram addr register (≤2 ns).
            for (int gi2 = 0; gi2 < BANKING; gi2++) begin
              silu_x_latch[gi2]       <= bram_dout_b[gi2*32+:32];
              silu_neg_log2e_reg[gi2] <= silu_neg_log2e[gi2];
            end
            state <= VEC_ACC_SILU_MUL;
          end else if (vec_opcode_reg == OP_VEC_SUB_BCAST) begin
            // Fix 13: u_sub_b uses fp32_add LATENCY=1; result not valid until next cycle.
            // u_vcr_mux activates acc_sub_bcast_row in VEC_ACC_RD_WAIT2.
            state <= VEC_ACC_RD_WAIT2;
          end else begin
            // Non-NL ops (scale/div etc.): latch combinational result into register.
            // Write to BRAM in VEC_ACC_RD_WR — breaks bram_dout_b→fp32→bram_din_b path.
            vec_computed_row_reg <= vec_computed_row;
            state <= VEC_ACC_RD_WR;
          end
        end

        VEC_ACC_RD_WAIT2: begin
          // u_sub_b LATENCY=1 result valid; latch and proceed to write.
          vec_computed_row_reg <= vec_computed_row;
          state <= VEC_ACC_RD_WR;
        end

        VEC_ACC_RD_WR: begin
          // Write registered single-src result; advance row.
          vec_row_r <= vec_row_r + 8'd1;
          if (vec_row_r == (BANKING - 1)) state <= FETCH_1;
          else state <= VEC_ACC_RD_REQ;
        end

        // ---- SILU NLE multiply pipeline (3-cycle exp2_bram, Fix 11+12) ----
        // VEC_ACC_SILU_MUL:    silu_neg_log2e_reg → exp2_bram; _r2 regs capture (cycle 0).
        // VEC_ACC_SILU_LATCH:  _r2 → u_f_sub; f_fp32_r captures (cycle 1).
        // VEC_ACC_EXP2_LATCH2: f_fp32_r → BRAM; rom_out_r captures (cycle 2, shared w/ EXP2).
        // VEC_ACC_NL_EXP2_LATCH: result valid.
        VEC_ACC_SILU_MUL:   state <= VEC_ACC_SILU_LATCH;
        VEC_ACC_SILU_LATCH: state <= VEC_ACC_EXP2_LATCH2;

        // ---- EXP2 ACC pipeline (3-cycle exp2_bram, Fix 9+11+12) ----
        // VEC_ACC_EXP2_ADDR:   bram_dout_b_exp2_reg → exp2_bram; _r2 regs capture (cycle 0).
        // VEC_ACC_EXP2_LATCH:  _r2 → u_f_sub; f_fp32_r captures (cycle 1).
        // VEC_ACC_EXP2_LATCH2: f_fp32_r → BRAM; rom_out_r captures (cycle 2, shared w/ SILU).
        // VEC_ACC_NL_EXP2_LATCH: result valid.
        VEC_ACC_EXP2_ADDR:  state <= VEC_ACC_EXP2_LATCH;
        VEC_ACC_EXP2_LATCH: state <= VEC_ACC_EXP2_LATCH2;

        // ---- Fix 12: shared 3rd-cycle wait for EXP2 + SILU ACC paths ----
        VEC_ACC_EXP2_LATCH2: state <= VEC_ACC_NL_EXP2_LATCH;

        // ---- NL pipeline: exp2_bram output valid ----
        // For OP_VEC_EXP2: write result to BRAM and advance to next row.
        // For OP_VEC_SILU:  silu_denom (1 + exp2) drives recip_bram this cycle;
        //                   result will be valid in VEC_ACC_NL_RECIP_LATCH.
        VEC_ACC_NL_EXP2_LATCH: begin
          if (vec_opcode_reg == OP_VEC_EXP2) begin
            for (int i = 0; i < BANKING; i++) acc_exp2_row_reg[i] <= acc_exp2_row[i];
            state <= VEC_ACC_EXP2_WR;
          end else begin  // OP_VEC_SILU
            state <= VEC_ACC_NL_RECIP_LATCH;
          end
        end

        VEC_ACC_EXP2_WR: begin
          vec_row_r <= vec_row_r + 8'd1;
          if (vec_row_r == (BANKING - 1)) state <= FETCH_1;
          else state <= VEC_ACC_RD_REQ;
        end

        // ---- NL pipeline: recip_bram output valid (SILU only) ----
        // acc_silu_row = silu_x_latch * recip_bram_out is combinational; write to BRAM.
        VEC_ACC_NL_RECIP_LATCH: begin
          vec_row_r <= vec_row_r + 8'd1;
          if (vec_row_r == (BANKING - 1)) state <= FETCH_1;
          else state <= VEC_ACC_RD_REQ;
        end

        // ---- Two-src ACC element-wise (mul/add/sub.acc) ----
        // Pipeline: RD_A_REQ → RD_A_WAIT (latch A) → RD_B_WAIT (latch B reg) →
        //           COMPUTE (fp32 runs on registered inputs, latch result) →
        //           WR (write registered result to BRAM)
        VEC_ACC2_RD_A_REQ: state <= VEC_ACC2_RD_A_WAIT;

        VEC_ACC2_RD_A_WAIT: begin
          vec_acc_a_row_reg <= bram_dout_b;
          state             <= VEC_ACC2_RD_B_WAIT;
        end

        VEC_ACC2_RD_B_WAIT: begin
          bram_dout_b_a2_reg <= bram_dout_b;
          acc2_dst_addr_reg  <= vec_acc_dst + (vec_row_r);
          state              <= VEC_ACC2_COMPUTE;
        end

        VEC_ACC2_COMPUTE: begin
          acc2_result_reg <= vec_computed_row_2src;
          state           <= VEC_ACC2_WR;
        end

        VEC_ACC2_WR: begin
          vec_row_r <= vec_row_r + 8'd1;
          if (vec_row_r == (BANKING - 1)) state <= FETCH_1;
          else state <= VEC_ACC2_RD_A_REQ;
        end

        // ---- vec.zero.acc ----
        VEC_ZERO_ACC_WRITE: begin
          vec_row_r <= vec_row_r + 8'd1;
          if (vec_row_r == vec_zero_rows - 8'd1) state <= FETCH_1;
          else state <= VEC_ZERO_ACC_WRITE;
        end

        // ---- Stage 3: loop.end ----
        LOOP_END_EXEC: begin
          // iv_reg_next = IV[loop_iv_id] + step  (combinational assign above)
          iv_reg[loop_iv_id] <= iv_reg_next;
          if (iv_reg_next < iv_hi[loop_iv_id]) begin
            // Jump back to body_start
            pc_load     <= 1'b1;
            pc_load_val <= iv_body_start[loop_iv_id];
          end
          // else: fall through — pc_enable fires via assign to advance past loop.end
          state <= FETCH_1;
        end

        // ---- Gap 3: OP_BARRIER — stall until host re-starts ----
        // done was pulsed once at EXEC_DISPATCH (OP_BARRIER case); it is NOT
        // re-asserted here.  The host's wait_for_flag sees instr_ready=1 (from
        // that single pulse) and knows the barrier has been reached.
        // On re-start (host issues a second compute doorbell), execution resumes
        // at the instruction after the barrier.
        // pc_load is NOT asserted here; the PC retains its post-dispatch value.
        BARRIER_WAIT: begin
          // done stays 0 (pulse default); sequencer idles until start fires.
          if (start) begin
            state <= FETCH_1;
          end
        end

        // ---- Gap 3: OP_LOOP_BEGIN_CSR — latch hi from kernel_arg_csr ----
        // CSR-sourced hi value: csr_loop_hi_val (combinational mux over kernel_arg_csr
        // driven by loop_hi_pool_idx[1:0], which carries csr_idx from EXEC_DISPATCH).
        LOOP_CSR_LATCH: begin
          iv_hi[loop_iv_id]         <= csr_loop_hi_val;  // hi from kernel_arg_csr[csr_idx]
          iv_reg[loop_iv_id]        <= {24'h0, loop_lo};
          iv_body_start[loop_iv_id] <= pc_val + (1);
          // pc_enable fires via LOOP_CSR_LATCH condition in assign above
          state                     <= FETCH_1;
        end

        // ---- ISA v2 / Gap-2: flush.slot (OP_FLUSH_SLOT = 0x53) ----
        // Spin until (dma_slot_done & flush_slot_mask[1:0]) == flush_slot_mask[1:0].
        // In the behavioral model, async DMA completes synchronously so this
        // exits immediately after the preceding async DMA completes its rows.
        // Once all targeted slots report done, clear those bits and proceed.
        FLUSH_SLOT_WAIT: begin
          if ((dma_slot_done & flush_slot_mask[1:0]) == flush_slot_mask[1:0]) begin
            // All requested slots are done; clear their done bits.
            dma_slot_done <= dma_slot_done & ~flush_slot_mask[1:0];
            state         <= FETCH_1;
          end
          // else: spin (stay in FLUSH_SLOT_WAIT)
        end

        default: state <= IDLE;
      endcase
    end
  end

  //---------------------------------------------
  // TMA — Stage 1: tied off (superseded by inline DMA above)
  //---------------------------------------------
  assign tma_req     = 1'b0;
  assign tma_dir     = 1'b0;
  assign tma_dm_base = 16'd0;
  assign tma_l2_base = 15'd0;
  assign tma_len     = 16'd0;

  //---------------------------------------------
  // vec_computed_row / vec_computed_row_2src
  //
  // Separate always @* so that Vivado does not place acc_sub_bcast_row and
  // similar fp32 results in the same combinational cone as bram_din_b.
  // Keeping them inline in the bram_din_b block (the old iverilog workaround)
  // created a 14 ns VREG→fp32→bram_din_b path that violated timing.
  // The simulator re-evaluates multiple always @* blocks correctly.
  //---------------------------------------------
  always @* begin : u_vcr_mux
    vec_computed_row_mux = bram_dout_b;
    if (state == VEC_ACC_RD_WAIT) begin
      for (int j2 = 0; j2 < BANKING; j2++) begin
        case (vec_opcode_reg)
          OP_VEC_EXP2:            vec_computed_row_mux[j2*32+:32] = acc_exp2_row[j2];
          OP_VEC_SILU:            vec_computed_row_mux[j2*32+:32] = acc_silu_row[j2];
          OP_VEC_SCALE_BCAST:     vec_computed_row_mux[j2*32+:32] = acc_scale_bcast_row[j2];
          OP_VEC_SCALE_COL_BCAST: vec_computed_row_mux[j2*32+:32] = acc_scale_col_row[j2];
          // OP_VEC_SUB_BCAST: handled in VEC_ACC_RD_WAIT2 (u_sub_b LATENCY=1)
          OP_VEC_DIV_BCAST:       vec_computed_row_mux[j2*32+:32] = acc_div_bcast_row[j2];
          OP_VEC_SCALE_IMM:       vec_computed_row_mux[j2*32+:32] = acc_scale_imm_row[j2];
          default:                vec_computed_row_mux[j2*32+:32] = bram_dout_b[j2*32+:32];
        endcase
      end
    end else if (state == VEC_ACC_RD_WAIT2) begin
      // Fix 13: u_sub_b LATENCY=1 result valid one cycle after VEC_ACC_RD_WAIT.
      for (int j2 = 0; j2 < BANKING; j2++) vec_computed_row_mux[j2*32+:32] = acc_sub_bcast_row[j2];
    end
    vec_computed_row = vec_computed_row_mux;

    vec_computed_row_2src_mux = bram_dout_b;
    if (state == VEC_ACC2_COMPUTE) begin
      for (int j2 = 0; j2 < BANKING; j2++) begin
        case (vec_opcode_reg)
          OP_VEC_MUL_ACC: vec_computed_row_2src_mux[j2*32+:32] = acc2_mul_row[j2];
          OP_VEC_ADD_ACC: vec_computed_row_2src_mux[j2*32+:32] = acc2_add_row[j2];
          OP_VEC_SUB_ACC: vec_computed_row_2src_mux[j2*32+:32] = acc2_sub_row[j2];
          default:        vec_computed_row_2src_mux[j2*32+:32] = vec_acc_a_row_reg[j2*32+:32];
        endcase
      end
    end
    vec_computed_row_2src = vec_computed_row_2src_mux;
  end

  //---------------------------------------------
  // BRAM Port B Routing
  //
  // Priority: DMA_ROW (load write / store read) > MXU > idle.
  // current_instr may hold pool data during pool-dereference states, so the
  // MXU arm is gated on state==WAIT_MXU rather than on the combinational `op`
  // field, which prevents spurious BRAM enables during pool reads.
  //---------------------------------------------
  always @* begin
    bram_addr_b     = '0;
    bram_din_b      = '0;
    bram_en_b       = 1'b0;
    bram_we_b       = 1'b0;
    systolic_dout_b = bram_dout_b;

    if (state == WAIT_MXU) begin
      bram_addr_b = systolic_addr;
      bram_din_b  = systolic_din_b;
      bram_en_b   = systolic_en_b | systolic_we_b;
      bram_we_b   = systolic_we_b;
    end else if (state == DMA_ROW_REQ &&
                 (dma_op_reg == OP_DMA_STORE_SPM || dma_op_reg == OP_DMA_STORE_ACC ||
                  dma_op_reg == OP_DMA_STORE_SPM_MI || dma_op_reg == OP_DMA_STORE_ACC_MI ||
                  dma_op_reg == OP_DMA_STORE_SPM_I8_MI)) begin
      // Store: read row from BRAM so it can be written to DM in DMA_ROW_WAIT
      bram_addr_b = dma_spm_base + (dma_row_r);
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b0;
    end else if (state == DMA_ROW_WAIT &&
                 (dma_op_reg == OP_DMA_LOAD_SPM || dma_op_reg == OP_DMA_LOAD_ACC ||
                  dma_op_reg == OP_DMA_LOAD_SPM_MI || dma_op_reg == OP_DMA_LOAD_ACC_MI ||
                  dma_op_reg == OP_DMA_LOAD_SPM_I8_MI)) begin
      // Load: write DM data (now valid) into BRAM
      bram_addr_b = dma_spm_base + (dma_row_r);
      bram_din_b  = dm_dout;
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b1;
    end else if (state == CAST_REQ) begin
      // Issue BRAM read at source address; data valid in CAST_WAIT.
      bram_addr_b = cast_src_addr;
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b0;
    end else if (state == CAST_WAIT && !cast_to_vreg) begin
      // Write converted data to destination BRAM row.
      bram_addr_b = cast_dst_addr;
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b1;
      if (cast_is_f32_to_f16) begin
        // cast.f32.f16: pack 8 fp16 into lower 128 bits; upper 128 bits = 0
        for (int i = 0; i < BANKING; i++) bram_din_b[i*16+:16] = cast_fp16_from_fp32[i];
        bram_din_b[255:128] = '0;
      end else begin
        // cast.f16.f32: pack 8 fp32 into full 256 bits
        for (int i = 0; i < BANKING; i++) bram_din_b[i*32+:32] = cast_fp32_from_fp16[i];
      end
      // ---- Stage 2 BRAM routing ----
    end else if (state == VEC_ACC_VREG_REQ) begin
      bram_addr_b = vec_acc_src + (vec_row_r);
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b0;
    end else if (state == VEC_ACC_RD_REQ) begin
      bram_addr_b = vec_acc_src + (vec_row_r);
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b0;
    end else if (state == VEC_ACC_RD_WAIT) begin
      // EXP2/SILU: BRAM module needs more cycles; no write-back this state.
      // Non-NL ops: result latched into vec_computed_row_reg this cycle; written in VEC_ACC_RD_WR.
    end else if (state == VEC_ACC_RD_WR) begin
      // Write registered single-src result — breaks bram_dout_b→fp32→bram_din_b path.
      bram_addr_b = vec_acc_dst + (vec_row_r);
      bram_din_b  = vec_computed_row_reg;
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b1;
    end else if (state == VEC_ACC_NL_EXP2_LATCH) begin
      // exp2_bram output valid; latch to acc_exp2_row_reg this cycle (FSM), write next cycle.
      // For SILU, silu_denom drives recip_bram — no write yet.
    end else if (state == VEC_ACC_EXP2_WR) begin
      // Write registered exp2 result — breaks BRAM→assembly→BRAM path.
      bram_addr_b = vec_acc_dst + (vec_row_r);
      for (int j2 = 0; j2 < BANKING; j2++) bram_din_b[j2*32+:32] = acc_exp2_row_reg[j2];
      bram_en_b = 1'b1;
      bram_we_b = 1'b1;
    end else if (state == VEC_ACC_NL_RECIP_LATCH) begin
      // recip_bram output valid; acc_silu_row = silu_x_latch * recip computed
      // combinationally by u_silu_out; write SILU result.
      if (vec_opcode_reg == OP_VEC_SILU) begin
        bram_addr_b = vec_acc_dst + (vec_row_r);
        for (int j2 = 0; j2 < BANKING; j2++) bram_din_b[j2*32+:32] = acc_silu_row[j2];
        bram_en_b = 1'b1;
        bram_we_b = 1'b1;
      end
    end else if (state == VEC_ACC2_RD_A_REQ) begin
      bram_addr_b = vec_acc_src + (vec_row_r);
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b0;
    end else if (state == VEC_ACC2_RD_A_WAIT) begin
      // Immediately issue acc_b read (data from acc_a captured in FSM this cycle)
      bram_addr_b = vec_acc_b + (vec_row_r);
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b0;
    end else if (state == VEC_ACC2_WR) begin
      bram_addr_b = acc2_dst_addr_reg;
      bram_din_b  = acc2_result_reg;
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b1;
    end else if (state == VEC_ZERO_ACC_WRITE) begin
      bram_addr_b = vec_acc_dst + (vec_row_r);
      bram_din_b  = '0;
      bram_en_b   = 1'b1;
      bram_we_b   = 1'b1;
    end else if (state == DMA_I4_WRITE) begin
      // Unpack 8 nibbles from i4_subrow_nibbles → 8 uint8 in bits[63:0], upper zero.
      // BRAM row = spm_base + (dm_word_row × 8) + subrow
      bram_addr_b = dma_spm_base + ({dma_row_r, 3'b0}) + (i4_subrow_r);
      bram_din_b  = '0;
      for (int j = 0; j < N; j++) begin
        bram_din_b[j*8+:8] = {4'b0, i4_subrow_nibbles[j*4+:4]};
      end
      bram_en_b = 1'b1;
      bram_we_b = 1'b1;
    end
  end

  //---------------------------------------------
  // Device-memory control (combinational)
  //---------------------------------------------
  always @* begin
    dm_addr = '0;
    dm_din  = '0;
    dm_en   = 1'b0;
    dm_we   = 1'b0;

    if (state == DMA_ROW_REQ &&
        (dma_op_reg == OP_DMA_LOAD_SPM || dma_op_reg == OP_DMA_LOAD_ACC ||
         dma_op_reg == OP_DMA_LOAD_SPM_I4 ||
         dma_op_reg == OP_DMA_LOAD_SPM_MI    || dma_op_reg == OP_DMA_LOAD_ACC_MI    ||
         dma_op_reg == OP_DMA_LOAD_SPM_I4_MI || dma_op_reg == OP_DMA_LOAD_SPM_I8_MI)) begin
      // Load: issue DM read; response arrives in DMA_ROW_WAIT (1-cycle DM latency)
      dm_addr = dma_byte_addr[DM_BYTE_SHIFT+:DM_ADDR_WIDTH];
      dm_en   = 1'b1;
      dm_we   = 1'b0;
    end else if (state == DMA_ROW_WAIT &&
                 (dma_op_reg == OP_DMA_STORE_SPM || dma_op_reg == OP_DMA_STORE_ACC ||
                  dma_op_reg == OP_DMA_STORE_SPM_MI || dma_op_reg == OP_DMA_STORE_ACC_MI ||
                  dma_op_reg == OP_DMA_STORE_SPM_I8_MI)) begin
      // Store: BRAM response (bram_dout_b) is now valid; write it to DM
      dm_addr = dma_byte_addr[DM_BYTE_SHIFT+:DM_ADDR_WIDTH];
      dm_din  = bram_dout_b;
      dm_en   = 1'b1;
      dm_we   = 1'b1;
    end
  end

  //---------------------------------------------
  // Illegal opcode detection
  //
  // Registered 1-cycle pulse: fires the cycle after EXEC_DISPATCH encounters
  // an opcode not in the dispatch table.  npu.sv latches it into a sticky
  // error-status register (slv_reg9[0]) readable by the host via AXI-Lite.
  //---------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) illegal_op_o <= 1'b0;
    else
      illegal_op_o <= (state == EXEC_DISPATCH) &&
                         !(op == OP_MATMUL_TILE         || op == OP_MATMUL_TILE_I8      ||
                           op == OP_MATMUL_TILE_DQ      ||
                           op == OP_DMA_LOAD_SPM        || op == OP_DMA_LOAD_ACC        ||
                           op == OP_DMA_STORE_SPM       || op == OP_DMA_STORE_ACC       ||
                           op == OP_DMA_LOAD_SPM_I4     ||
                           op == OP_CAST_F32_F16        ||
                           op == OP_CAST_F16_F32        ||
                           op == OP_CAST_F16_F32_VREG   ||
                           op == OP_VEC_ROWMAX          || op == OP_VEC_ROWSUM          ||
                           op == OP_VEC_EXP2            || op == OP_VEC_SCALE_BCAST     ||
                           op == OP_VEC_SCALE_IMM       || op == OP_VEC_SUB_BCAST       ||
                           op == OP_VEC_DIV_BCAST       || op == OP_VEC_ZERO_ACC        ||
                           op == OP_VEC_SILU            || op == OP_VEC_MUL_ACC         ||
                           op == OP_VEC_ADD_ACC         || op == OP_VEC_SUB_ACC         ||
                           op == OP_VEC_SCALE_COL_BCAST ||
                           op == OP_VEC_MAXIMUM         || op == OP_VEC_FMA             ||
                           op == OP_VEC_EXP2_VREG       || op == OP_VEC_SUB_VREG        ||
                           op == OP_VEC_MOVI            || op == OP_VEC_COPY_VREG       ||
                           op == OP_VEC_RSQRT           || op == OP_VEC_FMA_IMM         ||
                           op == OP_VEC_ADD_VREG        || op == OP_VEC_MUL_VREG        ||
                           op == OP_VEC_LOAD_IMM        ||
                           op == OP_LOOP_BEGIN          || op == OP_LOOP_END            ||
                           op == OP_BARRIER             || op == OP_LOOP_BEGIN_CSR      ||
                           op == OP_FLUSH               ||
      // ISA v2 / Gap-1: .mi DMA family
      op == OP_DMA_LOAD_SPM_MI     || op == OP_DMA_LOAD_ACC_MI     ||
                           op == OP_DMA_LOAD_SPM_I4_MI  || op == OP_DMA_LOAD_SPM_I8_MI ||
                           op == OP_DMA_STORE_SPM_MI    || op == OP_DMA_STORE_ACC_MI    ||
                           op == OP_DMA_STORE_SPM_I8_MI ||
      // ISA v2 / Gap-2: flush.slot
      op == OP_FLUSH_SLOT);
  end

  //---------------------------------------------
  // Submodule Instantiations
  //---------------------------------------------

  // PC: Program Counter
  pc #(
      .PC_WIDTH(IRAM_ADDR_WIDTH)
  ) u_pc (
      .clk(clk),
      .rst_n(rst_n),
      .PC_enable(pc_enable),
      .PC_load(pc_load),
      .PC_load_val(pc_load_val),
      .PC(pc_val)
  );

  // I_bram: Instruction BRAM (Port A is write-only; douta unused)
  /* verilator lint_off PINCONNECTEMPTY */
  blk_mem_gen_1 I_bram (
      .clka (clk),
      .ena  (1'b1),
      .wea  (instr_write_en),
      .addra(iram_addr),
      .dina (dma_iram_din),
      .douta(),

      .clkb (clk),
      .enb  (1'b1),
      .web  (1'b0),
      .addrb(iram_b_addr),
      .dinb (64'b0),
      .doutb(current_instr)
  );
  /* verilator lint_on PINCONNECTEMPTY */

  // Decoder
  decoder u_decoder (
      .instr_decode(current_instr),
      .op_decode   (op),
      .flags_decode(flags),
      .op0_decode  (op0),
      .op1_decode  (op1),
      .op2_decode  (op2)
  );

  // MXU: Matrix Unit — computes Z = X @ W^T.
  // Stage 1: accumulate (flags[1]) and causal (flags[2]) wired; transB (flags[0]) reserved.
  // Operand mapping: op0=acc_dst, op1=spm_a (X), op2=spm_b (W).
  // Stage 4: for OP_MATMUL_TILE_I8, pool fetch overwrites op0/op1/op2 before MXU starts,
  // so we latch operands in mxu_*_reg and mux here.
  logic [L1_ADDR_WIDTH-1:0] mxu_base_addr_w, mxu_base_addr_x, mxu_base_addr_out;
  // dq_mode: pass scale_vec from VREG read port A. vreg_rd_addr_a is set at
  // POOL_FETCH_WAIT step 2 (one cycle before start_systolic fires), so
  // vreg_rd_data_a is valid combinationally when MXU latches on start.
  logic [N-1:0][31:0] mxu_scale_vec;
  assign mxu_scale_vec = vreg_rd_data_a;
  assign mxu_base_addr_w = mxu_i4_mode_reg ? mxu_spm_b_i4_reg : op2[L1_ADDR_WIDTH-1:0];
  assign mxu_base_addr_x = mxu_i4_mode_reg ? mxu_spm_a_reg : op1[L1_ADDR_WIDTH-1:0];
  assign mxu_base_addr_out = mxu_i4_mode_reg ? mxu_acc_dst_reg : op0[L1_ADDR_WIDTH-1:0];

  mxu u_mxu (
      .clk          (clk),
      .rst_n        (rst_n),
      .start        (start_systolic),
      .done         (systolic_done),
      .accumulate   (mxu_accumulate),
      .causal       (mxu_causal),
      .base_addr_w  (mxu_base_addr_w),
      .base_addr_x  (mxu_base_addr_x),
      .base_addr_out(mxu_base_addr_out),
      .mem_req_addr (systolic_addr),
      .mem_req_data (systolic_din_b),
      .mem_resp_data(systolic_dout_b),
      .mem_read_en  (systolic_en_b),
      .mem_write_en (systolic_we_b),
      .i4_mode      (mxu_i4_mode_reg),
      .i4_zero_point(mxu_zero_point_reg),
      .scale_vec    (mxu_scale_vec),
      .dq_mode      (mxu_dq_mode_reg)
  );

  // VREG: Vector register file (8 regs × 8 fp32 elements)
  // Stage 2: read ports driven from FSM (no longer tied to 0)
  vec_regfile #(
      .NUM_REGS  (8),
      .ELEM_WIDTH(DATA_WIDTH),
      .NUM_ELEMS (BANKING)
  ) u_vreg (
      .clk      (clk),
      .rst_n    (rst_n),
      .rd_addr_a(vreg_rd_addr_a),
      .rd_data_a(vreg_rd_data_a),
      .rd_addr_b(vreg_rd_addr_b),
      .rd_data_b(vreg_rd_data_b),
      .wr_en    (vreg_wr_en),
      .wr_addr  (vreg_wr_addr),
      .wr_data  (vreg_wr_data)
  );

  // fp16→fp32 and fp32→fp16 converter lanes (8 lanes, one per BANKING slot)
  generate
    for (genvar i = 0; i < BANKING; i++) begin : cast_lanes
      fp16_to_fp32 u_fp16_to_fp32 (
          .fp16_in (cast_fp16_in[i]),
          .fp32_out(cast_fp32_from_fp16[i])
      );
      fp32_to_fp16 u_fp32_to_fp16 (
          .fp32_in (cast_fp32_in[i]),
          .fp16_out(cast_fp16_from_fp32[i])
      );
    end
  endgenerate

endmodule
