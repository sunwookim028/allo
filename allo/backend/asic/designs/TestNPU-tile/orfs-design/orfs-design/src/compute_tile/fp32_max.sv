// fp32_max — IEEE 754 FP32 max(a, b).
//
// SYNTHESIS + SIMULATION path: synthesisable behavioral model.  Replaces
// cvfpu fpnew_noncomp MINMAX.
// NaN semantics follow IEEE 754-2008 maxNum: if one operand is NaN the other
// is returned; +0 > −0.  LATENCY is ignored — output is always combinational.
`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
module fp32_max #(
    parameter int unsigned LATENCY = 0
) (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] result
);
  /* verilator lint_on UNUSEDPARAM */

  // IEEE 754-2008 maxNum: NaN propagation, sign-aware comparison.
  logic a_nan, b_nan;
  assign a_nan = (a[30:23] == 8'hFF) && (a[22:0] != 0);
  assign b_nan = (b[30:23] == 8'hFF) && (b[22:0] != 0);

  always @(*) begin
    if (a_nan && b_nan) begin
      result = 32'h7FC00000;  // canonical qNaN
    end else if (a_nan) begin
      result = b;
    end else if (b_nan) begin
      result = a;
    end else if (a[31] != b[31]) begin
      result = a[31] ? b : a;  // positive > negative
    end else if (!a[31]) begin
      result = (a[30:0] >= b[30:0]) ? a : b;  // both positive
    end else begin
      result = (a[30:0] <= b[30:0]) ? a : b;  // both negative (flipped)
    end
  end

endmodule
