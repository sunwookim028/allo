// vpu_fifo at WIDTH=32, DEPTH=4 (MiniTPU's instance), and the same behind output registers.
module mtpu_fifo_32x4 (input logic clk_i, rst_ni, push_i, input logic [31:0] push_data_i, input logic pop_i,
  output logic [31:0] pop_data_o, output logic empty_o, full_o);
  vpu_fifo #(.WIDTH(32), .DEPTH(4)) u (.*);
endmodule
module mtpu_fifo_32x4_oreg (input logic clk, rst_n, push, input logic [31:0] push_data, input logic pop,
  output logic [31:0] pop_data, output logic empty, full);
  mtpu_fifo_oreg #(.WIDTH(32), .DEPTH(4)) u (.*);
endmodule
