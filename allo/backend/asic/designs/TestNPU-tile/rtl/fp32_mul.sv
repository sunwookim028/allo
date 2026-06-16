// fp32_mul — IEEE 754 FP32 multiply.
//
// Scalar VPU / sequencer use only (pe.sv uses fp16_mul_fp32 instead).
//
// SYNTHESIS path: synthesisable behavioral IEEE 754 model.  Vivado infers
// DSP48E2 slices for the 24×24-bit mantissa multiply.  Replaces fpnew_fma
// FP32 MUL mode (which cost ~2000 LUTs/instance from the full FMA datapath).
//
// SIMULATION path: DPI-C (Verilator) or hand-rolled IEEE 754 (iverilog).
// LATENCY is ignored in both paths — output is always combinational.
`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
module fp32_mul #(
    parameter LATENCY   = 0,
    parameter              FORMAT    = "FP32",  // legacy; unused
    parameter          INT_BITS  = 16,      // legacy; unused
    parameter          FRAC_BITS = 16,      // legacy; unused
    parameter          WIDTH     = 32       // legacy; unused
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        clk,
    input  logic        rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] result
);
  /* verilator lint_on UNUSEDPARAM */
`ifdef VERILATOR
  import "DPI-C" pure function int dpi_fp32_mul(
    input int a_in,
    input int b_in
  );
`endif

`ifdef VERILATOR
  always @(*) result = dpi_fp32_mul(a, b);
`else
  localparam FP32_EXP_BIAS = 127;

  logic [23:0] a_m, b_m, z_m;
  logic [9:0] a_e, b_e, z_e;
  logic a_s, b_s, z_s;
  logic [49:0] product;
  logic guard_bit, round_bit, sticky;

  always @(*) begin
    result    = 32'h0;
    z_s       = 1'b0;
    z_e       = 10'd0;
    product   = 50'd0;
    z_m       = 24'd0;
    guard_bit = 1'b0;
    round_bit = 1'b0;
    sticky    = 1'b0;

    a_m       = {1'b0, a[22:0]};
    b_m       = {1'b0, b[22:0]};
    a_e       = {2'b00, a[30:23]} - 10'd127;
    b_e       = {2'b00, b[30:23]} - 10'd127;
    a_s       = a[31];
    b_s       = b[31];

    if ((a_e == 128 && a_m != 0) || (b_e == 128 && b_m != 0)) begin
      result = {1'b1, 8'hFF, 1'b1, 22'h0};  // NaN
    end else if (a_e == 128) begin
      if (($signed(b_e) == -127) && (b_m == 0))
        result = {1'b1, 8'hFF, 1'b1, 22'h0};  // Inf × 0 → NaN
      else result = {a_s ^ b_s, 8'hFF, 23'h0};  // Inf × finite
    end else if (b_e == 128) begin
      if (($signed(a_e) == -127) && (a_m == 0))
        result = {1'b1, 8'hFF, 1'b1, 22'h0};  // 0 × Inf → NaN
      else result = {a_s ^ b_s, 8'hFF, 23'h0};
    end else if (($signed(a_e) == -127) && (a_m == 0)) begin
      result = {a_s ^ b_s, 8'h0, 23'h0};  // 0 × anything
    end else if (($signed(b_e) == -127) && (b_m == 0)) begin
      result = {a_s ^ b_s, 8'h0, 23'h0};
    end else begin
      if ($signed(a_e) == -127) begin
        a_e = -126;
      end else begin
        a_m[23] = 1'b1;
      end
      if ($signed(b_e) == -127) begin
        b_e = -126;
      end else begin
        b_m[23] = 1'b1;
      end

      // Normalize subnormal inputs
      if (~a_m[23]) begin
        a_m = a_m << 1;
        a_e = a_e - 10'd1;
      end
      if (~b_m[23]) begin
        b_m = b_m << 1;
        b_e = b_e - 10'd1;
      end

      z_s       = a_s ^ b_s;
      z_e       = a_e + b_e + 10'd1;
      product   = a_m * b_m * 4;  // ×4 aligns 48-bit product into [49:26]

      z_m       = product[49:26];
      guard_bit = product[25];
      round_bit = product[24];
      sticky    = (product[23:0] != 0);

      if ($signed(z_e) < -126) begin
        z_e       = z_e + (-126 - $signed(z_e));
        z_m       = z_m >> (-126 - $signed(z_e));
        guard_bit = z_m[0];
        round_bit = guard_bit;
        sticky    = sticky | round_bit;
      end else if (z_m[23] == 0) begin
        z_e       = z_e - 10'd1;
        z_m       = z_m << 1;
        z_m[0]    = guard_bit;
        guard_bit = round_bit;
        round_bit = 1'b0;
      end else if (guard_bit && (round_bit | sticky | z_m[0])) begin
        z_m = z_m + 24'd1;
        if (z_m == 24'hffffff) z_e = z_e + 10'd1;
      end

      result[22:0]  = z_m[22:0];
      result[30:23] = (z_e[7:0] + (FP32_EXP_BIAS));
      result[31]    = z_s;

      if ($signed(z_e) == -126 && z_m[23] == 0) result[30:23] = 8'h0;

      if ($signed(z_e) > $signed((FP32_EXP_BIAS))) begin
        result[22:0]  = 23'h0;
        result[30:23] = 8'hFF;
        result[31]    = z_s;
      end
    end
  end
`endif

endmodule
