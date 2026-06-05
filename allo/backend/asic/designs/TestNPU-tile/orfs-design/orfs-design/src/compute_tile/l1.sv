`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// l1.sv
//
// 8-Bank interleaved L1 Data Memory.
// Port A (32-bit): Multiplexed DMA access
// Port B (256-bit): Parallel access for Compute units
//////////////////////////////////////////////////////////////////////////////////

// rst_n / dma_rd_en kept in port list for interface compatibility.
module l1
  import npu_config_pkg::*;
(
    input logic clk,
    /* verilator lint_off UNUSEDSIGNAL */
    input logic rst_n,
    /* verilator lint_on UNUSEDSIGNAL */

    // Control interface
    input logic [15:0] base_addr,  // Kept for compatibility, but offset logic simplified

    // Write interface (from Slave Stream)
    input logic                     dma_wr_en,
    input logic [L1_DATA_WIDTH-1:0] dma_wr_data,
    input logic [             15:0] dma_write_pointer,

    // Read interface (to Master Stream)
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                     dma_rd_en,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic [L1_DATA_WIDTH-1:0] dma_rd_data,
    input  logic [             15:0] dma_read_pointer,

    // Compute-side BRAM port (Port B)
    input  logic [L1_ADDR_WIDTH-1:0] dma_comp_addr_b,
    input  logic [L1_DATA_WIDTH-1:0] dma_comp_din_b,
    output logic [L1_DATA_WIDTH-1:0] dma_comp_dout_b,
    input  logic                     dma_comp_en_b,
    input  logic                     dma_comp_we_b
);

  localparam int BRAM_ADDRB_W = 13;  // physical BRAM port width; L1_ADDR_WIDTH bits are used

  // Internal signals
  // Offset is now directly in 256-bit words.
  // Upper bits [15:10] of dma_addr are reserved (256-bit word L1 spans only
  // 1024 entries today).
  /* verilator lint_off UNUSEDSIGNAL */
  wire [15:0] dma_addr;
  /* verilator lint_on UNUSEDSIGNAL */
  assign dma_addr = base_addr + (dma_wr_en ? dma_write_pointer : dma_read_pointer);

  wire [L1_ADDR_WIDTH-1:0] dma_bram_addr = dma_addr[L1_ADDR_WIDTH-1:0];

  //-----------------------------------------------
  // BRAM instantiation (true dual-port)
  //-----------------------------------------------
  genvar i;
  generate
    for (i = 0; i < 8; i = i + 1) begin : bank
      blk_mem_gen_0 u_bram (
          // Port A - DMA side (now aligned to 8x32 bits)
          .clka (clk),
          .ena  (1'b1),
          .wea  (dma_wr_en),
          // Pad L1_ADDR_WIDTH (10) to BRAM_ADDRB_W to match BRAM's addra port.
          .addra({{(BRAM_ADDRB_W - L1_ADDR_WIDTH) {1'b0}}, dma_bram_addr}),
          .dina (dma_wr_data[i*32+:32]),
          .douta(dma_rd_data[i*32+:32]),

          // Port B - Compute side
          .clkb (clk),
          .enb  (dma_comp_en_b),
          .web  (dma_comp_we_b),
          // Pad L1_ADDR_WIDTH (10) to BRAM_ADDRB_W to match BRAM's addrb port.
          .addrb({{(BRAM_ADDRB_W - L1_ADDR_WIDTH) {1'b0}}, dma_comp_addr_b}),
          .dinb (dma_comp_din_b[i*32+:32]),
          .doutb(dma_comp_dout_b[i*32+:32])
      );
    end
  endgenerate

endmodule

