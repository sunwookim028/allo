// MiniTPU's combinational mxu_bf16_mul_acc24 with its output registered: the
// port shape of Catapult's Wire-port RTL at latency 1 (the PE registers the
// product the same way, vpu_pkg MXU_PE_LATENCY = 1 + MXU_ACC_ADD_LATENCY).
module mtpu_macc_oreg (
  input  logic        clk,
  input  logic [15:0] a,
  input  logic [15:0] b,
  output logic [23:0] r
);
  logic [23:0] r_d;
  mxu_bf16_mul_acc24 u (.a_i(a), .b_i(b), .result_o(r_d));
  always_ff @(posedge clk) r <= r_d;
endmodule
