// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// U4 harness wrapper (track C, D2): dma.sv closed on MiniTPU's VMEM DMA port.
// No logic of its own: the dma -> VMEM glue is minitpu_core.sv:211-230 as it
// stands (vmem_gnt tied 1, the write beat outranks the read, the 16-bit beat
// pointer cut to VMEM_BEAT_ADDR_W, dma_flush_i tied 0), and the VMEM is
// vpu_vmem_simd (vpu_dma_group + vpu_word_array, DMA read latency 2) with the
// compute port idle (compute_req_i = '0: never valid).
`timescale 1ns/1ps
module u4_dma_vmem
  import minitpu_config_pkg::*;
  import vpu_pkg::lane_stripe_t;
  import vpu_pkg::vpu_req_t;
  import vpu_pkg::NUM_SUBLANES;
  import vpu_pkg::VREG_ADDR_W;
#(
    parameter int OUTSTANDING    = 2,
    parameter int TIMEOUT_CYCLES = 1024,
    parameter int LANDING_DEPTH  = 8
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        desc_valid,
    input  logic        desc_is_store,
    input  logic        desc_channel,
    input  logic [13:0] desc_vmem_row,
    input  logic [13:0] desc_rows,
    input  logic [6:0]  desc_cols,
    input  logic [31:0] desc_base,
    input  logic [31:0] desc_stride,
    input  logic [1:0]  clear_channel_done,
    input  logic        dm_req_ready,
    input  logic        dm_rsp_valid,
    input  logic [255:0] dm_rsp_data,
    input  logic        dm_rsp_last,
    input  logic [1:0]  dm_rsp_resp,
    output logic        desc_accept,
    output logic [1:0]  dma_channel_done,
    output logic        dma_idle,
    output logic        dm_req_valid,
    output logic        dm_req_we,
    output logic [28:0] dm_req_addr,
    output logic [7:0]  dm_req_len,
    output logic [255:0] dm_req_wdata,
    output logic [31:0] dm_req_wstrb,
    output logic        dm_rsp_ready,
    output logic        vmem_wr_en,
    output logic [255:0] vmem_wr_data,
    output logic [15:0] vmem_wr_ptr,
    output logic        vmem_rd_en,
    output logic [15:0] vmem_rd_ptr,
    output logic        dma_err,
    output logic        dma_err_channel,
    output logic [1:0]  dma_err_cause,
    output logic [1:0]  dma_err_resp,
    output logic        dma_rsp_seen,
    output logic [31:0] beat_count_o,
    output logic [31:0] overlap_stat_o
);
  logic [255:0] dma_vmem_rd_data;

  dma #(
      .DMA_CHANNEL_COUNT(2), .OUTSTANDING(OUTSTANDING), .TIMEOUT_CYCLES(TIMEOUT_CYCLES),
      .LANDING_DEPTH(LANDING_DEPTH), .M_AXI_DATA_WIDTH(256)
  ) u_dma (
      .clk, .rst_n, .desc_valid, .desc_is_store, .desc_channel, .desc_vmem_row, .desc_rows,
      .desc_cols, .desc_base, .desc_stride, .desc_accept, .dma_channel_done, .clear_channel_done,
      .dma_idle, .dm_req_valid, .dm_req_ready, .dm_req_we, .dm_req_addr, .dm_req_len,
      .dm_req_wdata, .dm_req_wstrb, .dm_rsp_valid, .dm_rsp_ready, .dm_rsp_data, .dm_rsp_last,
      .dm_rsp_resp, .vmem_wr_en, .vmem_wr_data, .vmem_wr_ptr, .vmem_rd_en,
      .vmem_rd_data(dma_vmem_rd_data), .vmem_rd_ptr, .vmem_gnt(1'b1), .dma_err, .dma_err_channel,
      .dma_err_cause, .dma_err_resp, .dma_rsp_seen, .beat_count_o, .overlap_stat_o);

  // minitpu_core.sv:216-230, verbatim in effect
  /* verilator lint_off UNUSEDSIGNAL */
  logic [15:0] selected_ptr;
  /* verilator lint_on UNUSEDSIGNAL */
  logic vpu_dma_en, vpu_dma_we;
  logic [vpu_pkg::VMEM_BEAT_ADDR_W-1:0] vpu_dma_addr;
  lane_stripe_t vpu_dma_wdata, vpu_dma_rdata;
  always_comb begin
    selected_ptr  = vmem_wr_en ? vmem_wr_ptr : vmem_rd_ptr;
    vpu_dma_en    = vmem_wr_en || vmem_rd_en;
    vpu_dma_we    = vpu_dma_en && vmem_wr_en;
    vpu_dma_addr  = vpu_pkg::VMEM_BEAT_ADDR_W'(selected_ptr);
    vpu_dma_wdata = lane_stripe_t'(vmem_wr_data);
  end
  assign dma_vmem_rd_data = 256'(vpu_dma_rdata);

  /* verilator lint_off UNUSEDSIGNAL */
  logic load_valid;
  logic [VREG_ADDR_W-1:0] load_vreg_idx;
  lane_stripe_t load_data [NUM_SUBLANES];
  /* verilator lint_on UNUSEDSIGNAL */
  lane_stripe_t store_data [NUM_SUBLANES];
  always_comb for (int s = 0; s < NUM_SUBLANES; s++) store_data[s] = '0;

  vpu_vmem_simd u_vmem (
      .clk_i(clk), .rst_ni(rst_n),
      .compute_req_i(vpu_req_t'('0)), .store_data_i(store_data),
      .load_valid_o(load_valid), .load_vreg_idx_o(load_vreg_idx), .load_data_o(load_data),
      .dma_en_i(vpu_dma_en), .dma_we_i(vpu_dma_we), .dma_addr_i(vpu_dma_addr),
      .dma_wdata_i(vpu_dma_wdata), .dma_flush_i(1'b0), .dma_rdata_o(vpu_dma_rdata));
endmodule
