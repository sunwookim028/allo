// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// U4 harness wrapper: sequencer_scalar_agu, iv_by_level flattened, s_op as
// bits, and the four SREGs exposed read-only by hierarchical reference
// (sreg_o, sreg i at [32*i +: 32]) so every cycle's state is observable.
`timescale 1ns/1ps
module u4_scalar_agu
  import sequencer_pkg::*;
#(
    parameter int unsigned S_LAT = sequencer_pkg::S_LAT
) (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       s_valid_i,
    input  logic [2:0]                 s_op_i,
    input  logic [SREG_ADDR_W-1:0]     s_rd_i,
    input  logic [SREG_ADDR_W-1:0]     s_rs_i,
    input  logic                       s_use_iv_i,
    input  logic [LEVEL_SEL_W-1:0]     s_level_i,
    input  logic [IMM_W-1:0]           s_imm_i,
    input  logic [127:0]               kernel_arg_csr_i,
    input  logic [32*STACK_DEPTH-1:0]  iv_flat_i,
    input  logic [SREG_ADDR_W-1:0]     rd_sel_d_base_i,
    input  logic [SREG_ADDR_W-1:0]     rd_sel_d_stride_i,
    input  logic                       rd_loop_bound_from_arg_i,
    input  logic [SREG_ADDR_W-1:0]     rd_sel_loop_bound_i,
    output logic [31:0]                rd_data_d_base_o,
    output logic [31:0]                rd_data_d_stride_o,
    output logic [L_HI_W-1:0]          rd_data_loop_bound_o,
    output logic [3:0]                 sreg_written_o,
    output logic [127:0]               sreg_o
);
  logic [31:0] iv[STACK_DEPTH];
  always_comb for (int i = 0; i < STACK_DEPTH; i++) iv[i] = iv_flat_i[32*i +: 32];
  sequencer_scalar_agu #(.S_LAT(S_LAT)) u (
      .clk, .rst_n, .s_valid_i, .s_op_i(s_op_e'(s_op_i)), .s_rd_i, .s_rs_i, .s_use_iv_i,
      .s_level_i, .s_imm_i, .kernel_arg_csr_i, .iv_by_level_i(iv),
      .rd_sel_d_base_i, .rd_data_d_base_o, .rd_sel_d_stride_i, .rd_data_d_stride_o,
      .rd_loop_bound_from_arg_i, .rd_sel_loop_bound_i, .rd_data_loop_bound_o,
      .sreg_written_o);
  assign sreg_o = {u.sreg_q[3], u.sreg_q[2], u.sreg_q[1], u.sreg_q[0]};
endmodule
