// MiniTPU's vpu_fifo (combinational pop_data/empty/full from registered state)
// with its three outputs registered: the port shape of Catapult's Wire-port
// RTL at output latency 1, as the U1/U2 records' mtpu_*_oreg.sv wrappers.
module mtpu_fifo_oreg #(parameter int unsigned WIDTH = 32, parameter int unsigned DEPTH = 4) (
  input  logic             clk,
  input  logic             rst_n,
  input  logic             push,
  input  logic [WIDTH-1:0] push_data,
  input  logic             pop,
  output logic [WIDTH-1:0] pop_data,
  output logic             empty,
  output logic             full
);
  logic [WIDTH-1:0] d;
  logic e, f;
  vpu_fifo #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u (
    .clk_i(clk), .rst_ni(rst_n), .push_i(push), .push_data_i(push_data), .pop_i(pop),
    .pop_data_o(d), .empty_o(e), .full_o(f));
  always_ff @(posedge clk) begin
    pop_data <= d; empty <= e; full <= f;
  end
endmodule
