// fp16_mul_fp32 — FP16 × FP16 → FP32 multiply.
//
// SYNTHESIS: (* use_dsp = "yes" *) on the 11×11 mantissa product forces
// Vivado to map the multiply to a single DSP48E2 rather than LUT chains.
// At DIM=16 (256 PEs) this saves ~500 K LUTs vs. the fpnew_fma FP32 MUL path.
//
// SIMULATION: identical arithmetic, no DSP attribute; works with iverilog ≥12
// and Verilator.
//
// Special cases: NaN propagation, Inf×0→NaN, Inf, zero.
// Subnormal FP16 inputs are flushed to zero (conservative; subnormals do not
// arise in normal INT4/FP16 GEMM workloads).
// Truncation (no round-to-nearest): the 2–3 LSBs dropped from the product are
// zero when both mantissas are normal FP16, so truncation is exact for the
// normal×normal path.
`timescale 1ns / 1ps
module fp16_mul_fp32 (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [31:0] result
);

  // FP16 fields: [15]=sign, [14:10]=exp (bias 15), [9:0]=mantissa
  // exp_a/exp_b are 8-bit (zero-extended from 5) so FP32 exponent arithmetic
  // stays within the same width and avoids WIDTHEXPAND lint warnings.
  logic sign_a, sign_b, sign_r;
  logic [7:0] exp_a, exp_b;
  assign sign_a = a[15];
  assign exp_a  = {3'b0, a[14:10]};
  assign sign_b = b[15];
  assign exp_b  = {3'b0, b[14:10]};
  assign sign_r = sign_a ^ sign_b;

  // Hidden-1 prepended; subnormal operands (exp==0) keep hidden-0.
  logic [10:0] mant_a, mant_b;
  assign mant_a = (exp_a != 8'h0) ? {1'b1, a[9:0]} : {1'b0, a[9:0]};
  assign mant_b = (exp_b != 8'h0) ? {1'b1, b[9:0]} : {1'b0, b[9:0]};

  // 11 × 11 mantissa multiply (22-bit result).
  // Synthesis: (* use_dsp = "yes" *) tells Vivado to infer DSP48E2 here.
`ifdef SYNTHESIS
  (* use_dsp = "yes" *) logic [21:0] mant_prod;
`else
  logic [21:0] mant_prod;
`endif
  assign mant_prod = mant_a * mant_b;

  // Special-case flags
  logic zero_a, zero_b, inf_a, inf_b, nan_a, nan_b;
  assign zero_a = (exp_a == 8'h00) && (a[9:0] == 10'h0);
  assign zero_b = (exp_b == 8'h00) && (b[9:0] == 10'h0);
  assign inf_a  = (exp_a == 8'h1F) && (a[9:0] == 10'h0);
  assign inf_b  = (exp_b == 8'h1F) && (b[9:0] == 10'h0);
  assign nan_a  = (exp_a == 8'h1F) && (a[9:0] != 10'h0);
  assign nan_b  = (exp_b == 8'h1F) && (b[9:0] != 10'h0);

  // FP32 exponent for normal×normal:
  //   (e_a − 15) + (e_b − 15) + 127  =  e_a + e_b + 97
  //   +1 when mant_prod[21]=1 (product ≥ 2.0, needs renormalisation → +98).
  logic [7:0] exp_sum;

  always @* begin
    result  = 32'h0;
    exp_sum = 8'h0;

    if (nan_a || nan_b) begin
      result = 32'h7FC00000;
    end else if ((inf_a && zero_b) || (inf_b && zero_a)) begin
      result = 32'h7FC00000;  // Inf × 0 → NaN
    end else if (inf_a || inf_b) begin
      result = {sign_r, 8'hFF, 23'h0};
    end else if (zero_a || zero_b || (exp_a == 8'h0) || (exp_b == 8'h0)) begin
      result = {sign_r, 8'h00, 23'h0};  // zero or subnormal → flush to zero
    end else begin
      if (mant_prod[21]) begin
        // Leading 1 at bit 21: product ∈ [2^21, 2^22).
        // Normalised mantissa fraction = mant_prod[20:0] (21 bits).
        // FP32 mantissa [22:0] = {mant_prod[20:0], 2'b00}.
        exp_sum = exp_a + exp_b + 8'd98;
        result  = (exp_sum >= 8'hFF) ? {sign_r, 8'hFF, 23'h0}
                                     : {sign_r, exp_sum, mant_prod[20:0], 2'b00};
      end else begin
        // Leading 1 at bit 20: product ∈ [2^20, 2^21).
        // Normalised mantissa fraction = mant_prod[19:0] (20 bits).
        // FP32 mantissa [22:0] = {mant_prod[19:0], 3'b000}.
        exp_sum = exp_a + exp_b + 8'd97;
        result  = (exp_sum >= 8'hFF) ? {sign_r, 8'hFF, 23'h0}
                                     : {sign_r, exp_sum, mant_prod[19:0], 3'b000};
      end
    end
  end

endmodule
