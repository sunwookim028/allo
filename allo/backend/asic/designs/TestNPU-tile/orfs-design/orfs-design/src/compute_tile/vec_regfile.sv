`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: vec_regfile
// Description: Vector register file for SIMD VPU
//              8 registers (V0-V7), each holding 8 × 32-bit FP32 elements
//              Dual read ports, single write port
//              Total size: 8 × 256 bits = 2048 bits = 256 bytes
//////////////////////////////////////////////////////////////////////////////////

module vec_regfile #(
    parameter int NUM_REGS   = 8,
    parameter int ELEM_WIDTH = 32,
    parameter int NUM_ELEMS  = 8
) (
    input logic clk,
    input logic rst_n,

    // Read Port A
    input logic [2:0] rd_addr_a,
    output logic [NUM_ELEMS-1:0][ELEM_WIDTH-1:0] rd_data_a,

    // Read Port B
    input logic [2:0] rd_addr_b,
    output logic [NUM_ELEMS-1:0][ELEM_WIDTH-1:0] rd_data_b,

    // Write Port
    input logic wr_en,
    input logic [2:0] wr_addr,
    input logic [NUM_ELEMS-1:0][ELEM_WIDTH-1:0] wr_data
);

  // Register array: 8 registers × 8 elements × 32 bits
  logic [NUM_ELEMS-1:0][ELEM_WIDTH-1:0] regs[NUM_REGS-1:0];

  // Asynchronous reads
  assign rd_data_a = regs[rd_addr_a];
  assign rd_data_b = regs[rd_addr_b];

  // Synchronous write
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < NUM_REGS; i++) begin
        regs[i] <= '0;
      end
    end else if (wr_en) begin
      regs[wr_addr] <= wr_data;
    end
  end

endmodule
