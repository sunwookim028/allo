// MiniTPU's vpu_word_array.sv (b3ba0a4d) with its `ifdef SYNTHESIS branch (the Xilinx
// xpm_memory_tdpram) removed, so DC maps the simulation model's array to flops: the
// flop-mapped MiniTPU VMEM number (asic_memories_2026-10-04.rst). Nothing else changed.
// Copyright MiniTPU Authors
// SPDX-License-Identifier: Apache-2.0
`timescale 1ns/1ps

// VMEM storage: one array a whole VREG wide (128 B at 16x4); both ports read and write whole words only.

module vpu_word_array
  import vpu_pkg::*;
#(
  parameter int unsigned READ_LATENCY = vpu_pkg::VMEM_READ_LATENCY,
  parameter int unsigned DMA_READ_LATENCY = vpu_pkg::VMEM_DMA_READ_LATENCY
) (
  input logic clk_i,
  // Unused on purpose: the RAM has no reset, and clearing it would block RAM inference.
  /* verilator lint_off UNUSEDSIGNAL */
  input logic rst_ni,
  /* verilator lint_on UNUSEDSIGNAL */

  // -- compute port: one whole word -----------------------------------------
  input logic compute_en_i,
  input logic compute_we_i,
  input logic [VMEM_ADDR_W-1:0] compute_addr_i,
  input logic [NUM_SUBLANES-1:0][VMEM_STRIPE_W-1:0] compute_wdata_i,
  output logic [NUM_SUBLANES-1:0][VMEM_STRIPE_W-1:0] compute_rdata_o,

  // -- DMA port: also one whole word ----------------------------------------
  input logic dma_en_i,
  input logic dma_we_i,
  input logic [VMEM_ADDR_W-1:0] dma_addr_i,
  input logic [NUM_SUBLANES-1:0][VMEM_STRIPE_W-1:0] dma_wdata_i,
  output logic [NUM_SUBLANES-1:0][VMEM_STRIPE_W-1:0] dma_rdata_o
);

  localparam int unsigned WORD_W = NUM_SUBLANES * VMEM_STRIPE_W;

  // Generate-scope $error so a bad latency fails the build, not only simulation.
  if (READ_LATENCY < 1) begin : g_chk_read_latency
    $error("vpu_word_array: READ_LATENCY must be at least one cycle");
  end
  if (DMA_READ_LATENCY < 1) begin : g_chk_dma_read_latency
    $error("vpu_word_array: DMA_READ_LATENCY must be at least one cycle");
  end

  wire [WORD_W-1:0] compute_wdata_word = compute_wdata_i;
  wire [WORD_W-1:0] dma_wdata_word = dma_wdata_i;
  wire [WORD_W-1:0] compute_rdata_word;
  wire [WORD_W-1:0] dma_rdata_word;


  // Simulation model: same registered read; same-word collisions across the two ports are undefined, as on the FPGA.
  logic [WORD_W-1:0] mem [VMEM_WORDS];
  logic [WORD_W-1:0] compute_read_pipe [READ_LATENCY];
  logic [WORD_W-1:0] dma_read_pipe [DMA_READ_LATENCY];

  always_ff @(posedge clk_i) begin
    if (compute_en_i) begin
      if (compute_we_i) mem[compute_addr_i] <= compute_wdata_word;
      compute_read_pipe[0] <= mem[compute_addr_i];
    end
    for (int unsigned stage = 1; stage < READ_LATENCY; stage++)
      compute_read_pipe[stage] <= compute_read_pipe[stage-1];
    // (the two always_ff blocks of the model merged into one: DC refuses two
    // processes driving one array, ELAB-366; same-word cross-port collisions
    // are undefined in both forms)
    if (dma_en_i) begin
      if (dma_we_i) mem[dma_addr_i] <= dma_wdata_word;
      dma_read_pipe[0] <= mem[dma_addr_i];
    end
    for (int unsigned stage = 1; stage < DMA_READ_LATENCY; stage++)
      dma_read_pipe[stage] <= dma_read_pipe[stage-1];
  end
  assign compute_rdata_word = compute_read_pipe[READ_LATENCY-1];
  assign dma_rdata_word = dma_read_pipe[DMA_READ_LATENCY-1];


  assign compute_rdata_o = compute_rdata_word;
  assign dma_rdata_o = dma_rdata_word;

endmodule : vpu_word_array
