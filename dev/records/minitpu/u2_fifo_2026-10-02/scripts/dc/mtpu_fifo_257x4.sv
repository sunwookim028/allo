// vpu_fifo at WIDTH=257, DEPTH=4 (MiniTPU's instance), and the same behind output registers.
module mtpu_fifo_257x4 (input logic clk_i, rst_ni, push_i, input logic [256:0] push_data_i, input logic pop_i,
  output logic [256:0] pop_data_o, output logic empty_o, full_o);
  vpu_fifo #(.WIDTH(257), .DEPTH(4)) u (.*);
endmodule
module mtpu_fifo_257x4_oreg (input logic clk, rst_n, push, input logic [256:0] push_data, input logic pop,
  output logic [256:0] pop_data, output logic empty, full);
  mtpu_fifo_oreg #(.WIDTH(257), .DEPTH(4)) u (.*);
endmodule
