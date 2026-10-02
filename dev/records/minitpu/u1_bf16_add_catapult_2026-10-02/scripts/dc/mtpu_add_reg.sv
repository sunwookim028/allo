// Register wrapper for MiniTPU's combinational vpu_bf16_add, so it can be
// timed reg-to-reg like Catapult's clocked RTL: inputs and output registered.
module mtpu_add_reg (
  input  logic        clk,
  input  logic [15:0] a,
  input  logic [15:0] b,
  output logic [15:0] r
);
  logic [15:0] a_q, b_q, r_d;
  vpu_bf16_add u (.a_i(a_q), .b_i(b_q), .result_o(r_d));
  always_ff @(posedge clk) begin
    a_q <= a;
    b_q <= b;
    r   <= r_d;
  end
endmodule
