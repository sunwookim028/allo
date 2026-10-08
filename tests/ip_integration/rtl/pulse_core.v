// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Deliberately finishes several cycles after its final output transfer.
`timescale 1ns / 1ps
module pulse_core (
    input wire clk, rst, ce, go,
    output reg done,
    input wire [31:0] a_data,
    input wire a_valid,
    output wire a_ready,
    output reg [31:0] c_data,
    output reg c_valid,
    input wire c_ready
);
    reg [1:0] state;
    reg [2:0] delay_count;
    assign a_ready = ce && state == 1 && !c_valid;
    always @(posedge clk) begin
        if (rst) begin
            state <= 0; done <= 0; c_valid <= 0; c_data <= 0; delay_count <= 0;
        end else if (ce) begin
            done <= 0;
            case (state)
                0: if (go) state <= 1;
                1: begin
                    if (a_valid && a_ready) begin c_data <= a_data + 1; c_valid <= 1; end
                    if (c_valid && c_ready) begin c_valid <= 0; state <= 2; delay_count <= 3; end
                end
                2: if (delay_count == 0) begin done <= 1; state <= 0; end
                   else delay_count <= delay_count - 1'b1;
                default: state <= 0;
            endcase
        end
    end
endmodule
