// MiniTPU's vpu_regfile (asynchronous reads) with its three read ports
// registered: the port shape of Catapult's Wire-port RTL at read latency 1,
// as the U1 records' mtpu_*_oreg.sv wrappers.
module mtpu_rf_oreg #(parameter int unsigned WIDTH = 16) (
  input  logic             clk,
  input  logic [4:0]       ra, rb, rc, wa,
  input  logic [WIDTH-1:0] wd,
  input  logic             we,
  output logic [WIDTH-1:0] qa, qb, qc
);
  logic [WIDTH-1:0] da, db, dc;
  vpu_regfile #(.WIDTH(WIDTH)) u (
    .clk_i(clk), .rst_ni(1'b1),
    .raddr_a_i(ra), .rdata_a_o(da), .raddr_b_i(rb), .rdata_b_o(db),
    .raddr_c_i(rc), .rdata_c_o(dc), .waddr_i(wa), .wdata_i(wd), .we_i(we));
  always_ff @(posedge clk) begin
    qa <= da; qb <= db; qc <= dc;
  end
endmodule
