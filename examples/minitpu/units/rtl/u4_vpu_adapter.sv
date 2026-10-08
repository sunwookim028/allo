// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// U4 harness wrapper: sequencer_vpu_adapter (combinational), dummy clock, the
// v/m/x slots and vpu_ctrl_t flattened (packed-struct layout, MSB first).
`timescale 1ns/1ps
/* verilator lint_off UNUSEDSIGNAL */
module u4_vpu_adapter
  import vpu_pkg::*;
  import sequencer_pkg::*;
(
    input  logic                         clk_i,
    input  logic [$bits(v_slot_t)-1:0]   v_i,
    input  logic [$bits(m_slot_t)-1:0]   m_i,
    input  logic [$bits(x_slot_t)-1:0]   x_i,
    input  logic                         issue_i,
    input  logic [VMEM_ADDR_W-1:0]       x_resolved_row_i,
    output logic [$bits(vpu_ctrl_t)-1:0] vpu_ctrl_o
);
  vpu_ctrl_t c;
  sequencer_vpu_adapter u (
      .v_i(v_slot_t'(v_i)), .m_i(m_slot_t'(m_i)), .x_i(x_slot_t'(x_i)), .issue_i,
      .x_resolved_row_i, .vpu_ctrl_o(c));
  assign vpu_ctrl_o = c;
endmodule
