// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
`timescale 1ns/1ps

// U4 sequencer-loop harness wrapper (not MiniTPU RTL): u4_seq_cmd.sv plus the scalar
// AGU's four committed SREGs read by hierarchical reference (sreg_o, sreg i at
// [32*i +: 32], as u4_scalar_agu.sv). From u4_seq_cmd.sv: MiniTPU's sequencer.sv unchanged,
// with vpu_ctrl_t presented as its three slot commands (README D-23) -- V (vector),
// X (memory), M (matrix) -- each a valid vector and a payload vector, so a check
// can mask a slot's payload in the cycles its valid is low. Flattening only: every
// output bit is one bit of vpu_ctrl_o or a sequencer port, in struct order.
module u4_seq_loop
  import minitpu_config_pkg::*;
  import vpu_pkg::*;
  import sequencer_pkg::*;
(
  input  logic clk,
  input  logic rst_n,
  input  logic start,
  input  logic instr_write_en,
  input  logic [INSTR_ADDR_W-1:0] iram_addr,
  input  logic [BUNDLE_WIDTH-1:0] dma_iram_din,
  input  logic [31:0] program_id_csr,
  input  logic [127:0] kernel_arg_csr,
  input  logic [1:0] dma_channel_done,
  input  logic dma_idle,
  input  logic dma_desc_accept,
  input  logic matrix_busy_i,
  output logic bundle_issued_o,
  output logic done,
  output logic [1:0] clear_channel_done,
  output logic state_is_idle_o,
  output logic state_is_run_o,
  output logic [VMEM_ADDR_W-1:0] x_issue_address,
  output logic dma_desc_valid,
  output logic dma_desc_is_store,
  output logic [DMA_CHANNEL_SEL_W-1:0] dma_desc_channel,
  output logic [vpu_pkg::VMEM_BEAT_ADDR_W-1:0] dma_desc_vmem_row,
  output logic [DESC_BEAT_ROWS_W-1:0] dma_desc_rows,
  output logic [6:0] dma_desc_cols,
  output logic [31:0] dma_desc_base,
  output logic [31:0] dma_desc_stride,
  // V: {alu, txin, txout, sfu, reduce} valid; payload {raddr_a, raddr_b, txin_index,
  //    txout_index, txout_vd, alu_op, alu_vd, sfu_op, sfu_vd, reduce_op, reduce_lane, reduce_vd}
  output logic [4:0] v_valid_o,
  output logic [41:0] v_pay_o,
  // X: vmem.valid; payload {vmem.op, vmem.vreg_idx, vmem.vmem_address, vmem_store_read_hint}
  output logic x_valid_o,
  output logic [18:0] x_pay_o,
  // M: {vmatload, vmatpush, vmatpop} valid; payload {vmatload_base, vmatpush_vs, vmatpop_vd}
  output logic [2:0] m_valid_o,
  output logic [14:0] m_pay_o,
  // the scalar state (sequencer_scalar_agu.sv sreg_q), pre-edge
  output logic [127:0] sreg_o
);
  vpu_ctrl_t c;
  /* verilator lint_off UNUSEDSIGNAL */
  logic v_issue_valid, m_issue_valid, x_issue_valid, x_issue_op;
  v_slot_t v_issue;
  m_slot_t m_issue;
  logic [VREG_ADDR_W-1:0] x_issue_vreg_idx;
  /* verilator lint_on UNUSEDSIGNAL */

  sequencer u_seq (
    .clk, .rst_n, .start, .x_issue_address, .done,
    .instr_write_en, .iram_addr, .dma_iram_din, .program_id_csr, .kernel_arg_csr,
    .dma_channel_done, .dma_idle, .clear_channel_done, .dma_desc_accept,
    .dma_desc_valid, .dma_desc_is_store, .dma_desc_channel, .dma_desc_vmem_row, .dma_desc_rows,
    .dma_desc_cols, .dma_desc_base, .dma_desc_stride,
    .v_issue_valid, .v_issue, .m_issue_valid, .m_issue,
    .x_issue_valid, .x_issue_op, .x_issue_vreg_idx,
    .vpu_ctrl_o(c), .matrix_busy_i, .bundle_issued_o, .state_is_idle_o, .state_is_run_o
  );

  assign v_valid_o = {c.alu_valid, c.txin_valid, c.txout_valid, c.sfu_valid, c.reduce_valid};
  assign v_pay_o = {c.raddr_a, c.raddr_b, c.txin_index, c.txout_index, c.txout_vd, 4'(c.alu_op), c.alu_vd,
                    2'(c.sfu_op), c.sfu_vd, 1'(c.reduce_op), c.reduce_lane, c.reduce_vd};
  assign x_valid_o = c.vmem.valid;
  assign x_pay_o = {c.vmem.op, c.vmem.vreg_idx, c.vmem.vmem_address, c.vmem_store_read_hint};
  assign m_valid_o = {c.vmatload_valid, c.vmatpush_valid, c.vmatpop_valid};
  assign m_pay_o = {c.vmatload_base, c.vmatpush_vs, c.vmatpop_vd};
  assign sreg_o = {u_seq.u_scalar_agu.sreg_q[3], u_seq.u_scalar_agu.sreg_q[2],
                   u_seq.u_scalar_agu.sreg_q[1], u_seq.u_scalar_agu.sreg_q[0]};
endmodule
