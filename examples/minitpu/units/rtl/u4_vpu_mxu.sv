// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
`timescale 1ns/1ps

// U4 track B harness wrapper (not MiniTPU RTL; u4_vpu_wb plus the MXU port, by hierarchical
// reference, to measure result_latency.vmatpush from the issue edge): MiniTPU's vpu.sv unchanged, with
// vpu_ctrl_t presented as one flat vector and the shared writeback observed by
// hierarchical reference, so the VREG write-port calendar can be measured.
module u4_vpu_mxu
  import vpu_pkg::*;
(
  input  logic clk_i,
  input  logic rst_ni,
  input  logic [$bits(vpu_ctrl_t)-1:0] ctrl_i,
  input  logic dma_en_i,
  input  logic dma_we_i,
  input  logic [VMEM_BEAT_ADDR_W-1:0] dma_addr_i,
  input  logic [NUM_LANES*DATA_WIDTH-1:0] dma_wdata_i,
  output logic [7:0] ctrl_w_o,          // $bits(vpu_ctrl_t), a layout check
  output logic [5:0] wb_src_o,          // vpu.sv wb_source_valid: {tx, matrix, reduce, sfu, alu, load}
  output logic wb_stage_valid_o,        // first writeback register
  output logic [NUM_SUBLANES-1:0] wb_local_valid_o,  // the VREG write enables
  output logic [VREG_ADDR_W-1:0] wb_local_addr_o,    // sublane 0's write address
  output logic [NUM_SUBLANES*NUM_LANES*DATA_WIDTH-1:0] rdata_a_o,  // VREG port A, all sublanes
  output logic matrix_busy_o,
  output logic unsupported_o,
  output logic [NUM_LANES*DATA_WIDTH-1:0] dma_rdata_o,
  output logic mxu_push_o,       // mxu.input_push_i (after the serializer)
  output logic mxu_accept_o,     // mxu.input_accept_o
  output logic mxu_valid_o,      // mxu.output_valid_o: a result group is waiting
  output logic mxu_pop_o,        // mxu.output_pop_i (the pop engine takes it)
  output logic mpop_wb_o         // the matrix source at the writeback mux (wb_source_valid[4])
);
  vpu_ctrl_t ctrl;
  assign ctrl = vpu_ctrl_t'(ctrl_i);
  assign ctrl_w_o = 8'($bits(vpu_ctrl_t));
  lane_stripe_t dma_rdata;

  vpu u_vpu (
    .clk_i, .rst_ni, .ctrl_i(ctrl), .unsupported_o, .matrix_busy_o,
    .dma_en_i, .dma_we_i, .dma_addr_i, .dma_wdata_i(lane_stripe_t'(dma_wdata_i)),
    .dma_flush_i(1'b0), .dma_rdata_o(dma_rdata)
  );
  assign dma_rdata_o = dma_rdata;

  assign wb_src_o = u_vpu.wb_source_valid;
  assign mxu_push_o = u_vpu.mxu_serialized_push;
  assign mxu_accept_o = u_vpu.mxu_input_accept;
  assign mxu_valid_o = u_vpu.mxu_output_valid;
  assign mxu_pop_o = u_vpu.mxu_output_pop;
  assign mpop_wb_o = u_vpu.wb_source_valid[4];
  assign wb_stage_valid_o = u_vpu.wb_stage_valid_q;
  assign wb_local_valid_o = u_vpu.wb_local_valid_q;
  assign wb_local_addr_o = u_vpu.wb_local_addr_q[0];
  for (genvar s = 0; s < NUM_SUBLANES; s++) begin : g_rd
    assign rdata_a_o[s*NUM_LANES*DATA_WIDTH +: NUM_LANES*DATA_WIDTH] = u_vpu.vreg_rdata_a[s];
  end
endmodule
