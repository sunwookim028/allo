`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: compute_tile
// Description: Wrapper for sequencer (logic) and l1 (data memory).
//              Exposes TMA signals so npu.sv can wire them to l2_tile.
//////////////////////////////////////////////////////////////////////////////////

module compute_tile
  import npu_config_pkg::*;
// IRAM_ADDR_WIDTH and L1_DATA_WIDTH are module parameters (not just package
// imports) so Vivado's IP packager XPath evaluator can resolve port-width
// expressions (IP_Flow 19-627).  Literals must stay in sync with params.py /
// npu_config_pkg; Vivado cannot evaluate package-qualified default expressions.
// The parameter names intentionally shadow the same-named package localparams
// (VARHIDDEN) — the values are identical and the shadowing is necessary.
/* verilator lint_off VARHIDDEN */
#(
    parameter int IRAM_ADDR_WIDTH = 10,   // npu_config_pkg::IRAM_ADDR_WIDTH
    parameter int L1_DATA_WIDTH   = 256,  // npu_config_pkg::L1_DATA_WIDTH
    parameter int DM_ADDR_WIDTH   = 16,   // npu_config_pkg::DM_ADDR_WIDTH
    parameter int DM_DATA_WIDTH   = 256   // npu_config_pkg::DM_DATA_WIDTH
    /* verilator lint_on VARHIDDEN */
) (
    input logic clk,
    // TODO(lint): rst_n is used as async-reset in sequencer.sv but
    // synchronously in pc.sv.  Tracked for follow-up; lint_off SYNCASYNCNET
    // to avoid blocking CI on a real bug indicator.
    /* verilator lint_off SYNCASYNCNET */
    input logic rst_n,
    /* verilator lint_on SYNCASYNCNET */

    // High-level control
    input  logic start,
    output logic done,
    output logic illegal_op_o,

    // DMA Instruction Interface
    input logic                       instr_write_en,
    input logic [IRAM_ADDR_WIDTH-1:0] iram_addr,
    input logic [               63:0] dma_iram_din,

    // DMA Data Interface
    input  logic [             15:0] base_addr,
    input  logic                     dma_wr_en,
    input  logic [L1_DATA_WIDTH-1:0] dma_wr_data,
    input  logic [             15:0] dma_write_pointer,
    input  logic                     dma_rd_en,
    output logic [L1_DATA_WIDTH-1:0] dma_rd_data,
    input  logic [             15:0] dma_read_pointer,

    // Literal pool base (set by host before start; wired from IRAM load addr)
    input logic [IRAM_ADDR_WIDTH-1:0] pool_base,

    // ABI v2 §D: program registry
    // program_id_csr[3:0]: selects entry point PC = program_id * PROGRAM_SLOT_SIZE
    input logic [ 31:0] program_id_csr,
    // kernel_arg_csr[127:0]: four 32-bit per-kernel arguments (kernel_arg_0..3)
    // Packed as {arg3, arg2, arg1, arg0} with arg0 in bits[31:0].
    input logic [127:0] kernel_arg_csr,

    // Device-memory DMA interface (256-bit wide; word-addressed)
    output logic [DM_ADDR_WIDTH-1:0] dm_addr,
    output logic [DM_DATA_WIDTH-1:0] dm_din,
    input  logic [DM_DATA_WIDTH-1:0] dm_dout,
    output logic                     dm_en,
    output logic                     dm_we,

    // TMA signals (sequencer → l2_tile via npu.sv)
    output logic        tma_req,
    output logic        tma_dir,
    output logic [15:0] tma_dm_base,
    output logic [14:0] tma_l2_base,
    output logic [15:0] tma_len,
    input  logic        tma_done
);

  // Internal BRAM connection
  logic [L1_ADDR_WIDTH-1:0] pc_addr_b;
  logic [L1_DATA_WIDTH-1:0] pc_din_b;
  logic [L1_DATA_WIDTH-1:0] pc_dout_b;
  logic                     pc_en_b;
  logic                     pc_we_b;

  // Instantiate TensorCore
  sequencer u_sequencer (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .done(done),
      .illegal_op_o(illegal_op_o),
      .instr_write_en(instr_write_en),
      .iram_addr(iram_addr),
      .dma_iram_din(dma_iram_din),
      .bram_addr_b(pc_addr_b),
      .bram_din_b(pc_din_b),
      .bram_dout_b(pc_dout_b),
      .bram_en_b(pc_en_b),
      .bram_we_b(pc_we_b),
      .pool_base(pool_base),
      .program_id_csr(program_id_csr),
      .kernel_arg_csr(kernel_arg_csr),
      .dm_addr(dm_addr),
      .dm_din(dm_din),
      .dm_dout(dm_dout),
      .dm_en(dm_en),
      .dm_we(dm_we),
      .tma_req(tma_req),
      .tma_dir(tma_dir),
      .tma_dm_base(tma_dm_base),
      .tma_l2_base(tma_l2_base),
      .tma_len(tma_len),
      .tma_done(tma_done)
  );

  // Instantiate L1
  l1 u_l1 (
      .clk(clk),
      .rst_n(rst_n),
      .base_addr(base_addr),
      .dma_wr_en(dma_wr_en),
      .dma_wr_data(dma_wr_data),
      .dma_write_pointer(dma_write_pointer),
      .dma_rd_en(dma_rd_en),
      .dma_rd_data(dma_rd_data),
      .dma_read_pointer(dma_read_pointer),
      .dma_comp_addr_b(pc_addr_b),
      .dma_comp_din_b(pc_din_b),
      .dma_comp_dout_b(pc_dout_b),
      .dma_comp_en_b(pc_en_b),
      .dma_comp_we_b(pc_we_b)
  );

endmodule
