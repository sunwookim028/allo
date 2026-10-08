// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// U4 harness wrapper: sequencer_agu_resolve (combinational), dummy clock,
// iv_by_level[STACK_DEPTH] flattened (level i at [32*i +: 32]). No logic.
`timescale 1ns/1ps
/* verilator lint_off UNUSEDSIGNAL */
module u4_agu_resolve
  import vpu_pkg::*;
  import sequencer_pkg::*;
(
    input  logic                         clk_i,
    input  logic [32*STACK_DEPTH-1:0]    iv_flat_i,
    input  logic [VMEM_ADDR_W-1:0]       x_literal_i,
    input  logic                         x_agu_valid_i,
    input  logic [LEVEL_SEL_W-1:0]       x_agu_level_i,
    input  logic [X_SHIFT_W-1:0]         x_agu_shift_i,
    output logic [VMEM_ADDR_W-1:0]       x_resolved_addr_o
);
  logic [31:0] iv[STACK_DEPTH];
  always_comb for (int i = 0; i < STACK_DEPTH; i++) iv[i] = iv_flat_i[32*i +: 32];
  sequencer_agu_resolve u_agu (
      .iv_by_level(iv), .x_literal_i, .x_agu_valid_i, .x_agu_level_i, .x_agu_shift_i,
      .x_resolved_addr_o);
endmodule
