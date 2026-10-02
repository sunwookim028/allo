// MiniTPU's combinational vpu_bf16_mul with its output registered: the port
// shape of Catapult's Wire-port RTL at latency 1 (inputs -> multiplier -> flop),
// as the pilot's mtpu_add_oreg.sv.
module mtpu_mul_oreg (
  input  logic        clk,
  input  logic [15:0] a,
  input  logic [15:0] b,
  output logic [15:0] r
);
  logic [15:0] r_d;
  vpu_bf16_mul u (.a_i(a), .b_i(b), .result_o(r_d));
  always_ff @(posedge clk) r <= r_d;
endmodule
