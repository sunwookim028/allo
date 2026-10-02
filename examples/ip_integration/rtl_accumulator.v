// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Supplied single-transaction FIFO wrapper; state persists until reset.
`timescale 1ns / 1ps
module accumulator (
    input wire ap_clk, ap_rst, ap_ce, ap_start, ap_continue,
    output wire ap_ready, ap_idle, ap_done,
    input wire [31:0] a_dout,
    input wire a_empty_n,
    output wire a_read,
    output wire [31:0] c_din,
    output wire c_write,
    input wire c_full_n
);
    reg [1:0] state = 0;
    reg [31:0] total = 0;
    assign ap_idle = state == 0;
    assign ap_ready = ap_ce && state == 0;
    assign ap_done = state == 3;
    assign a_read = ap_ce && state == 1 && a_empty_n;
    assign c_write = ap_ce && state == 2 && c_full_n;
    assign c_din = total;
    always @(posedge ap_clk) begin
        if (ap_rst) begin state <= 0; total <= 0; end
        else if (ap_ce) begin
            case (state)
                0: if (ap_start) state <= 1;
                1: if (a_empty_n) begin total <= total + a_dout; state <= 2; end
                2: if (c_full_n) state <= 3;
                3: if (ap_continue) state <= 0;
            endcase
        end
    end
endmodule
