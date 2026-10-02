// vpu_fifo at WIDTH=64, DEPTH=16 (MiniTPU's instance), and the same behind output registers.
module mtpu_fifo_64x16 (input logic clk_i, rst_ni, push_i, input logic [63:0] push_data_i, input logic pop_i,
  output logic [63:0] pop_data_o, output logic empty_o, full_o);
  vpu_fifo #(.WIDTH(64), .DEPTH(16)) u (.*);
endmodule
module mtpu_fifo_64x16_oreg (input logic clk, rst_n, push, input logic [63:0] push_data, input logic pop,
  output logic [63:0] pop_data, output logic empty, full);
  mtpu_fifo_oreg #(.WIDTH(64), .DEPTH(16)) u (.*);
endmodule
