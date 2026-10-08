// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// U4 harness wrapper: sequencer_decoder (combinational) with a dummy clock for
// the trace driver, and bundle_fields_t flattened to one vector (MSB first, the
// packed struct's own layout). No logic.
`timescale 1ns/1ps
/* verilator lint_off UNUSEDSIGNAL */
module u4_seq_decoder
  import sequencer_pkg::*;
(
    input  logic                              clk_i,
    input  logic [BUNDLE_WIDTH-1:0]           bundle_i,
    output logic [$bits(bundle_fields_t)-1:0] fields_o
);
  bundle_fields_t f;
  sequencer_decoder u_dec (.bundle_i(bundle_i), .fields_o(f));
  assign fields_o = f;
endmodule
