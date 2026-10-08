// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
module memory (
    input wire ap_clk, ap_rst, ap_start,
    output wire ap_done,
    output wire [1:0] addr,
    output wire ce, we,
    input wire [31:0] q,
    output wire [31:0] d
);
    reg [2:0] state = 0;
    assign addr = 0;
    assign ce = state == 1 || state == 2;
    assign we = state == 2;
    assign d = q + 1;
    assign ap_done = state == 3;
    always @(posedge ap_clk) begin
        if (ap_rst) state <= 0;
        else case (state)
            0: if (ap_start) state <= 1;
            1: state <= 2;
            2: state <= 3;
            3: if (!ap_start) state <= 0;
            default: state <= 0;
        endcase
    end
endmodule
