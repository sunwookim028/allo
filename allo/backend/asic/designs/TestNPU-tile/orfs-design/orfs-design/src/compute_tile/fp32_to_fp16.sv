`timescale 1ns / 1ps
// IEEE 754 single-precision to half-precision conversion — combinational.
// Round-to-nearest-even (RNE).
//
// Input:  fp32_in  [31:0]  — sign:1 exp:8 mantissa:23
// Output: fp16_out [15:0]  — sign:1 exp:5 mantissa:10
//
// Overflow  (fp32_exp > 142) → fp16 ±infinity
// Underflow (fp32_exp < 103) → fp16 ±zero (flush-to-zero)
// NaN                        → fp16 quiet NaN
// fp32 subnormals (exp=0)    → fp16 zero (flush-to-zero)
module fp32_to_fp16 (
    input  logic [31:0] fp32_in,
    output logic [15:0] fp16_out
);

  logic        sign;
  logic [ 7:0] exp32;
  logic [22:0] mant32;

  // Exponent conversion constants — all derived from IEEE 754 field widths.
  localparam FP32_EXP_BIAS = 127;
  localparam FP16_EXP_BIAS = 15;
  localparam FP16_MANT_BITS = 10;
  // exp16 = exp32 - (FP32_EXP_BIAS - FP16_EXP_BIAS); difference = 112
  localparam EXP_BIAS_DIFF = FP32_EXP_BIAS - FP16_EXP_BIAS;
  // fp16 max normal biased exponent = 2^5 - 2 = 30 (0x1E); exp=31 is reserved for inf/NaN
  localparam FP16_MAX_EXP = (1 << 5) - 2;
  // fp32_exp > FP16_MAX_EXP + EXP_BIAS_DIFF = 142 → fp16 overflow
  localparam FP32_OVF_EXP = FP16_MAX_EXP + EXP_BIAS_DIFF;
  // fp32_exp < FP32_EXP_BIAS - (FP16_EXP_BIAS-1) - FP16_MANT_BITS = 103 → flush to fp16 zero
  // (smallest fp16 subnormal = 2^{-(FP16_EXP_BIAS-1+FP16_MANT_BITS)} = 2^{-24})
  localparam FP32_UNF_EXP = FP32_EXP_BIAS - (FP16_EXP_BIAS - 1) - FP16_MANT_BITS;

  assign sign   = fp32_in[31];
  assign exp32  = fp32_in[30:23];
  assign mant32 = fp32_in[22:0];

  // Round-to-nearest-even helpers (declared at module scope to avoid latches)
  logic [ 4:0] exp16_base;  // fp16 biased exponent before rounding
  logic [ 9:0] mant16_base;  // top 10 bits of fp32 mantissa
  logic        round_bit;  // fp32_mant[12]
  logic        sticky;  // OR of fp32_mant[11:0]
  logic        lsb;  // fp16 mantissa LSB = fp32_mant[13]
  logic        round_up;  // RNE round-up decision
  logic [10:0] mant16_rounded;  // 11-bit after rounding (may carry into exp)

  assign exp16_base     = (exp32 - (EXP_BIAS_DIFF));
  assign mant16_base    = mant32[22:13];
  assign round_bit      = mant32[12];
  assign sticky         = (mant32[11:0] != 12'h0);
  assign lsb            = mant32[13];
  assign round_up       = round_bit & (sticky | lsb);
  assign mant16_rounded = {1'b0, mant16_base} + (round_up ? 11'h1 : 11'h0);

  always @* begin
    if (exp32 == 8'hFF) begin
      // NaN or Infinity
      if (mant32 == 23'h0) fp16_out = {sign, 5'h1F, 10'h0};  // Infinity
      else fp16_out = {sign, 5'h1F, 1'b1, 9'h0};  // quiet NaN
    end else if (exp32 == 8'h00) begin
      // fp32 zero or subnormal — flush to fp16 zero
      fp16_out = {sign, 15'h0};
    end else if (exp32 > (FP32_OVF_EXP)) begin
      // Overflow → fp16 infinity
      fp16_out = {sign, 5'h1F, 10'h0};
    end else if (exp32 < (FP32_UNF_EXP)) begin
      // Underflow → fp16 zero (flush-to-zero)
      fp16_out = {sign, 15'h0};
    end else begin
      // Normal range: exp32 in [FP32_UNF_EXP, FP32_OVF_EXP] → exp16 in [1, FP16_MAX_EXP]
      if (mant16_rounded[10]) begin
        // Rounding caused mantissa overflow — increment exponent
        if (exp16_base == (FP16_MAX_EXP))
          fp16_out = {sign, 5'h1F, 10'h0};  // rounds up to infinity
        else fp16_out = {sign, (exp16_base + 5'h1), 10'h0};
      end else begin
        fp16_out = {sign, exp16_base, mant16_rounded[9:0]};
      end
    end
  end

endmodule
