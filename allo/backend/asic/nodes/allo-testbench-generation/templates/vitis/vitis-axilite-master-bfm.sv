`timescale 1ns/1ps

module vitis_axilite_master_bfm #(
  parameter integer ADDR_WIDTH = 6,
  parameter integer DATA_WIDTH = 32
) (
  input  logic clk,
  input  logic reset_n,
  output logic awvalid,
  input  logic awready,
  output logic [ADDR_WIDTH-1:0] awaddr,
  output logic wvalid,
  input  logic wready,
  output logic [DATA_WIDTH-1:0] wdata,
  output logic [DATA_WIDTH/8-1:0] wstrb,
  output logic arvalid,
  input  logic arready,
  output logic [ADDR_WIDTH-1:0] araddr,
  input  logic rvalid,
  output logic rready,
  input  logic [DATA_WIDTH-1:0] rdata,
  input  logic [1:0] rresp,
  input  logic bvalid,
  output logic bready,
  input  logic [1:0] bresp
);
  real input_delay_ns;

  initial begin
    if (!$value$plusargs("ALLO_BAGL_INPUT_DELAY_NS=%f", input_delay_ns))
      input_delay_ns = 0.0;
    awvalid = 1'b0;
    awaddr = '0;
    wvalid = 1'b0;
    wdata = '0;
    wstrb = '1;
    arvalid = 1'b0;
    araddr = '0;
    rready = 1'b0;
    bready = 1'b0;
  end

  task automatic write(
    input logic [ADDR_WIDTH-1:0] address,
    input logic [DATA_WIDTH-1:0] value
  );
    begin
      // Drive requests on the falling edge so the DUT sees stable values at
      // the following rising edge. Blocking assignments avoid an NBA race
      // between this procedural BFM and a gate-level AXI-Lite slave.
      @(negedge clk);
      #(input_delay_ns);
      awaddr = address;
      awvalid = 1'b1;
      do @(posedge clk); while (awready !== 1'b1);
      @(negedge clk);
      #(input_delay_ns);
      awvalid = 1'b0;

      wdata = value;
      wstrb = '1;
      wvalid = 1'b1;
      do @(posedge clk); while (wready !== 1'b1);
      @(negedge clk);
      #(input_delay_ns);
      wvalid = 1'b0;

      bready = 1'b1;
      do @(posedge clk); while (bvalid !== 1'b1);
      if (bresp !== 2'b00)
        $fatal(1, "AXI-Lite write to 0x%0h returned response %0b", address, bresp);
      @(negedge clk);
      bready = 1'b0;
    end
  endtask
endmodule
