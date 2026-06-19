module sram_probe
(
  input  logic        clk,
  input  logic        csb0,
  input  logic        web0,
  input  logic [4:0]  addr0,
  input  logic [31:0] din0,
  output logic [31:0] dout0,
  input  logic        csb1,
  input  logic        web1,
  input  logic [4:0]  addr1,
  input  logic [31:0] din1,
  output logic [31:0] dout1
);

  sram_probe_2rw_32x32_wpr4 sram
  (
    .clk0  (clk),
    .csb0  (csb0),
    .web0  (web0),
    .addr0 (addr0),
    .din0  (din0),
    .dout0 (dout0),
    .clk1  (clk),
    .csb1  (csb1),
    .web1  (web1),
    .addr1 (addr1),
    .din1  (din1),
    .dout1 (dout1)
  );

endmodule

