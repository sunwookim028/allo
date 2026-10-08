`timescale 1ns/1ps
module tb_mxu_probe;
  import vpu_pkg::*;
  localparam int DIM = `MRTL_DIM;
  logic clk = 0, rst_n = 0;
  logic input_push = 0, input_ready, input_accept, weight_commit = 0;
  mxu_input_kind_e input_kind = MXU_INPUT_LHS;
  logic [DIM-1:0][DATA_WIDTH-1:0] input_data = '0;
  logic output_pop = 0, output_valid;
  logic [NUM_SUBLANES-1:0][DIM-1:0][DATA_WIDTH-1:0] output_data;
  always #5 clk = ~clk;
  mxu #(.DIM(DIM)) dut (.clk_i(clk), .rst_ni(rst_n), .input_push_i(input_push), .input_kind_i(input_kind),
    .input_data_i(input_data), .input_ready_o(input_ready), .input_accept_o(input_accept),
    .weight_commit_i(weight_commit), .output_pop_i(output_pop), .output_valid_o(output_valid), .output_data_o(output_data));
  initial begin
    repeat (2) @(posedge clk); rst_n <= 1;
    for (int k = 0; k < 40; k++) begin
      repeat (25) @(posedge clk);
      $display("cyc %0d: in(rst) %0d out(rst) %0d | acc_out %0d vld_out %0d od_out %0d | lag %0d acc_lag %0d", (k+1)*25,
        dut.n_rst_in, dut.n_rst_out, dut.n_acc_out, dut.n_vld_out, dut.n_od_out, dut.lag_o, dut.acc_lag_o);
    end
    $finish;
  end
endmodule
