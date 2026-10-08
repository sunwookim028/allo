// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// U4 harness wrapper (not MiniTPU RTL): sequencer_loop_ctrl + sequencer_loop_buffer
// wired exactly as sequencer.sv:141-179 wires them (capture_valid_i is the
// issued pulse, capture_data_i the bundle the fetch queue offers). The only
// change is that the unpacked iv_by_level_o array is flattened, level k at
// bits [32k +: 32].
`timescale 1ns/1ps
module u4_loop_ctrl
  import minitpu_config_pkg::*;
  import sequencer_pkg::*;
(
    input  logic clk,
    input  logic rst_n,
    input  logic                    loop_begin_valid,
    input  logic [INSTR_ADDR_W-1:0] body_start,
    input  logic [7:0]              lo,
    input  logic [15:0]             hi,
    input  logic [7:0]              step,
    input  logic                    may_skip,
    input  logic [INSTR_ADDR_W-1:0] skip,
    input  logic                    loop_end_valid,
    input  logic                    bundle_issued,
    input  logic [BUNDLE_WIDTH-1:0] capture_data,
    output logic                    branch_taken,
    output logic [INSTR_ADDR_W-1:0] branch_target,
    output logic [STACK_DEPTH*32-1:0] iv_by_level,
    output logic [31:0]             iv_tos,
    output logic [LEVEL_SEL_W-1:0]  level_tos,
    output logic                    lb_reset,
    output logic                    lb_capture_en,
    output logic                    lb_replay_en,
    output logic [$clog2(LB_CAP)-1:0] lb_replay_idx,
    output logic                    lb_capture_overflow,
    output logic [BUNDLE_WIDTH-1:0] replay_data,
    output logic [$clog2(STACK_DEPTH + 1)-1:0] depth,
    output logic                    overflow,
    output logic                    underflow
);
  logic [31:0] iv_arr[STACK_DEPTH];
  always_comb for (int k = 0; k < STACK_DEPTH; k++) iv_by_level[32*k +: 32] = iv_arr[k];
  sequencer_loop_ctrl u_loop_ctrl (
      .clk(clk), .rst_n(rst_n),
      .loop_begin_valid_i(loop_begin_valid), .loop_body_start_i(body_start),
      .loop_lo_i(lo), .loop_hi_i(hi), .loop_step_i(step),
      .loop_may_skip_i(may_skip), .loop_skip_i(skip),
      .loop_end_valid_i(loop_end_valid), .bundle_issued_i(bundle_issued),
      .branch_taken_o(branch_taken), .branch_target_o(branch_target),
      .iv_tos_o(iv_tos), .iv_by_level_o(iv_arr), .level_tos_o(level_tos),
      .lb_reset_o(lb_reset), .lb_capture_en_o(lb_capture_en), .lb_replay_en_o(lb_replay_en),
      .lb_replay_idx_o(lb_replay_idx), .lb_capture_overflow_i(lb_capture_overflow),
      .loop_depth_o(depth), .loop_overflow_o(overflow), .loop_underflow_o(underflow));
  sequencer_loop_buffer u_loop_buffer (
      .clk(clk), .rst_n(rst_n),
      .capture_en_i(lb_capture_en), .capture_valid_i(bundle_issued),
      .capture_data_i(capture_data), .capture_overflow_o(lb_capture_overflow),
      .replay_en_i(lb_replay_en), .replay_idx_i(lb_replay_idx), .replay_data_o(replay_data),
      .loop_reset_i(lb_reset));
endmodule
