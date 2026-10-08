// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// An ordinary ready/valid IP: no HLS control, active-low synchronous reset.
`timescale 1ns / 1ps
module ready_valid_accumulator #(
    parameter integer W = 32
) (
    input wire clk, rst_n, ce,
    input wire [W-1:0] a_data, b_data,
    input wire a_valid, b_valid,
    output wire a_ready, b_ready,
    output reg [W-1:0] c_data,
    output reg c_valid,
    input wire c_ready
);
    reg [W-1:0] total;
    wire room = !c_valid || c_ready;
    assign a_ready = ce && room && b_valid;
    assign b_ready = ce && room && a_valid;
    always @(posedge clk) begin
        if (!rst_n) begin
            total <= 0;
            c_data <= 0;
            c_valid <= 0;
        end else if (ce && room) begin
            c_valid <= a_valid && b_valid;
            if (a_valid && b_valid) begin
                total <= total + a_data + b_data;
                c_data <= total + a_data + b_data;
            end
        end
    end
endmodule
