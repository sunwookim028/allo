// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// A deliberately non-trivial streaming vector adder:
//  * ready/valid handshake on both inputs and the output
//  * a 3-stage internal pipeline, so results lag inputs by several cycles
//  * ap_start / ap_done block-level control
// Nothing here is combinational w.r.t. the caller: the transactor MUST run a
// real cycle loop to get the data out.
module vadd_rtl #(
    parameter integer W = 32
) (
    input  wire          ap_clk,
    input  wire          ap_rst,
    input  wire          ap_start,
    output wire          ap_done,
    // input stream A
    input  wire [W-1:0]  a_tdata,
    input  wire          a_tvalid,
    output wire          a_tready,
    // input stream B
    input  wire [W-1:0]  b_tdata,
    input  wire          b_tvalid,
    output wire          b_tready,
    // output stream C
    output reg  [W-1:0]  c_tdata,
    output reg           c_tvalid,
    input  wire          c_tready
);
    // Fire only when both inputs are available and the output can accept.
    wire fire = ap_start && a_tvalid && b_tvalid && (!c_tvalid || c_tready);
    assign a_tready = fire;
    assign b_tready = fire;

    // 3-stage pipeline: stage0 latches, stage1/2 just delay, to prove the
    // transactor is genuinely stepping cycles rather than reading a wire.
    reg [W-1:0] s0, s1, s2;
    reg         v0, v1, v2;

    always @(posedge ap_clk) begin
        if (ap_rst) begin
            s0 <= 0; s1 <= 0; s2 <= 0;
            v0 <= 0; v1 <= 0; v2 <= 0;
            c_tdata <= 0; c_tvalid <= 0;
        end else begin
            if (!c_tvalid || c_tready) begin
                s0 <= a_tdata + b_tdata;  v0 <= fire;
                s1 <= s0;                 v1 <= v0;
                s2 <= s1;                 v2 <= v1;
                c_tdata  <= s2;
                c_tvalid <= v2;
            end
        end
    end

    // Idle once nothing is in flight.
    assign ap_done = ap_start && !v0 && !v1 && !v2 && !c_tvalid;
endmodule
