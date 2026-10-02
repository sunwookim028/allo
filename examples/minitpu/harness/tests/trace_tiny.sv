// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// A tiny stateful module to test harness/rtl.py's trace shape: a 4-entry,
// 100-bit RAM with one asynchronous and one registered (latency 2) read, a
// never-reset register, and a reset counter with an assertion.
module trace_tiny (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         we_i,
  input  logic [1:0]   waddr_i,
  input  logic [99:0]  wdata_i,
  input  logic [1:0]   raddr_i,
  output logic [99:0]  async_o,
  output logic [99:0]  reg_o,
  output logic [7:0]   count_o,
  output logic [31:0]  junk_o
);
  logic [99:0] mem [4];
  logic [99:0] pipe_q [2];
  logic [31:0] junk_q;  // never written, never reset
  assign async_o = mem[raddr_i];
  assign reg_o = pipe_q[1];
  assign junk_o = junk_q;
  always_ff @(posedge clk_i) begin
    if (we_i) mem[waddr_i] <= wdata_i;
    pipe_q[0] <= mem[raddr_i];
    pipe_q[1] <= pipe_q[0];
  end
  always_ff @(posedge clk_i) begin
    if (!rst_ni) count_o <= '0;
    else begin
      assert (count_o != 8'd5) else $error("count reached 5");
      count_o <= count_o + 8'(we_i);
    end
  end
endmodule
