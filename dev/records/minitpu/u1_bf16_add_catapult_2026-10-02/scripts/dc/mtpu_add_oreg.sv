// MiniTPU's vpu_bf16_add with its output registered only: the same port shape
// as Catapult's RTL for the Wire-port Allo unit at 3.33 ns (inputs -> adder -> flop).
module mtpu_add_oreg (
  input  logic        clk,
  input  logic [15:0] a,
  input  logic [15:0] b,
  output logic [15:0] r
);
  logic [15:0] r_d;
  vpu_bf16_add u (.a_i(a), .b_i(b), .result_o(r_d));
  always_ff @(posedge clk) r <= r_d;
endmodule
