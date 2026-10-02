// The 256-bit instance (vpu_vreg_stripe.sv:26) as a DC top: bare, and with
// its three read ports registered (mtpu_rf_oreg.sv's shape).
module mtpu_rf256 (
  input  logic clk, input logic [4:0] ra, rb, rc, wa, input logic [255:0] wd, input logic we,
  output logic [255:0] qa, qb, qc);
  vpu_regfile #(.WIDTH(256)) u (
    .clk_i(clk), .rst_ni(1'b1), .raddr_a_i(ra), .rdata_a_o(qa), .raddr_b_i(rb), .rdata_b_o(qb),
    .raddr_c_i(rc), .rdata_c_o(qc), .waddr_i(wa), .wdata_i(wd), .we_i(we));
endmodule
module mtpu_rf256_oreg (
  input  logic clk, input logic [4:0] ra, rb, rc, wa, input logic [255:0] wd, input logic we,
  output logic [255:0] qa, qb, qc);
  mtpu_rf_oreg #(.WIDTH(256)) u (.*);
endmodule
