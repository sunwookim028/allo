// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// U4 harness wrapper: dma_desc_adapter with d_slot_t flattened. Built with
// +define+SYNTHESIS: the adapter's sim-only S_LAT check reads its sibling
// u_scalar_agu by hierarchical reference, which exists only inside sequencer.sv.
`timescale 1ns/1ps
module u4_dma_desc_adapter
  import minitpu_config_pkg::*;
  import sequencer_pkg::*;
(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start_i,
    input  logic [$bits(d_slot_t)-1:0]   d_i,
    input  logic [31:0]                  sreg_rd_base_i,
    input  logic [31:0]                  sreg_rd_stride_i,
    input  logic                         desc_accept_i,
    output logic                         desc_valid_o,
    output logic                         desc_is_store_o,
    output logic [DMA_CHANNEL_SEL_W-1:0] desc_channel_o,
    output logic [vpu_pkg::VMEM_BEAT_ADDR_W-1:0] desc_vmem_row_o,
    output logic [DESC_BEAT_ROWS_W-1:0]  desc_rows_o,
    output logic [6:0]                   desc_cols_o,
    output logic [31:0]                  desc_base_o,
    output logic [31:0]                  desc_stride_o,
    output logic                         done_o
);
  dma_desc_adapter u (
      .clk, .rst_n, .start_i, .d_i(d_slot_t'(d_i)), .sreg_rd_base_i, .sreg_rd_stride_i,
      .desc_valid_o, .desc_is_store_o, .desc_channel_o, .desc_vmem_row_o, .desc_rows_o,
      .desc_cols_o, .desc_base_o, .desc_stride_o, .desc_accept_i, .done_o);
endmodule
