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

// [DC copy: the `ifdef SYNTHESIS (xpm_memory_tdpram) branch removed; the simulation model is synthesized]
  // Simulation model: same registered read; same-word collisions across the two ports are undefined, as on the FPGA.
  logic [WORD_W-1:0] mem [VMEM_WORDS];
  logic [WORD_W-1:0] compute_read_pipe [READ_LATENCY];
  logic [WORD_W-1:0] dma_read_pipe [DMA_READ_LATENCY];

  // [DC copy] The model's two always_ff blocks both write mem, which DC refuses
  // (ELAB-366, multiple drivers): merged into one block, DMA write last, which
  // is the model's block order (the DMA write wins a write-write collision).
  always_ff @(posedge clk_i) begin
    if (compute_en_i) begin
      if (compute_we_i) mem[compute_addr_i] <= compute_wdata_word;
      compute_read_pipe[0] <= mem[compute_addr_i];
    end
    for (int unsigned stage = 1; stage < READ_LATENCY; stage++)
      compute_read_pipe[stage] <= compute_read_pipe[stage-1];
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
