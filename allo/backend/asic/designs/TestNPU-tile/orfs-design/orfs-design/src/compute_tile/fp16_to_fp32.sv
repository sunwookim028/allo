`timescale 1ns / 1ps
// IEEE 754 half-precision to single-precision conversion — combinational.
//
// Input:  fp16_in  [15:0]  — sign:1 exp:5 mantissa:10
// Output: fp32_out [31:0]  — sign:1 exp:8 mantissa:23
//
// Handles: zero, infinity, NaN, subnormals, normals.
// Subnormals are normalised (leading-1 found by priority scan; max shift = 10).
module fp16_to_fp32 (
    input  logic [15:0] fp16_in,
    output logic [31:0] fp32_out
);

  logic       sign;
  logic [4:0] exp16;
  logic [9:0] mant16;

  // Exponent bias constants — both derived from IEEE 754 field widths.
  localparam int unsigned FP32_EXP_BIAS = 127;
  localparam int unsigned FP16_EXP_BIAS = 15;
  // exp16 + (FP32_EXP_BIAS - FP16_EXP_BIAS) = exp16 + 112 gives the fp32 biased exponent.
  localparam int unsigned EXP_BIAS_DIFF = FP32_EXP_BIAS - FP16_EXP_BIAS;
  // fp32 mantissa pads 13 zero bits below the fp16 mantissa: fp32(23b) - fp16(10b) = 13.
  localparam int unsigned FP32_MANT_PAD = 23 - 10;

  assign sign   = fp16_in[15];
  assign exp16  = fp16_in[14:10];
  assign mant16 = fp16_in[9:0];

  // Subnormal normalisation helpers (declared at module scope to avoid latches)
  logic [3:0] lz;  // leading-zero count in mant16[9:0]
  logic [9:0] norm_mant;  // mantissa after shifting out the leading 1
  logic [7:0] sub_exp;  // fp32 exponent for the normalised subnormal

  // Leading-zero count via priority encoder
  always_comb begin
    if (mant16[9]) lz = 4'd0;
    else if (mant16[8]) lz = 4'd1;
    else if (mant16[7]) lz = 4'd2;
    else if (mant16[6]) lz = 4'd3;
    else if (mant16[5]) lz = 4'd4;
    else if (mant16[4]) lz = 4'd5;
    else if (mant16[3]) lz = 4'd6;
    else if (mant16[2]) lz = 4'd7;
    else if (mant16[1]) lz = 4'd8;
    else lz = 4'd9;  // mant16[0] is the only remaining candidate leading 1
  end

  // Shift mantissa left by (lz+1) to remove the leading 1 (it becomes implicit)
  assign norm_mant = mant16 << (lz + 4'd1);
  // fp32 exp: fp16 subnormal value = 2^(-14) × mant16/1024.
  // Leading 1 is at bit (9-lz), so value = 2^(-14) × 2^(9-lz) / 1024 × (1+frac)
  //                                       = 2^(-15-lz) × (1+frac).
  // fp32 biased exp = (-15 - lz) + 127 = EXP_BIAS_DIFF - lz.
  assign sub_exp   = 8'(EXP_BIAS_DIFF) - {4'h0, lz};

  always_comb begin
    if (exp16 == 5'h1F) begin
      // Infinity or NaN
      if (mant16 == 10'h0) fp32_out = {sign, 8'hFF, 23'h0};  // Infinity
      else fp32_out = {sign, 8'hFF, 1'b1, 22'h0};  // quiet NaN
    end else if (exp16 == 5'h00) begin
      if (mant16 == 10'h0) fp32_out = {sign, 31'h0};  // ±zero
      else
        // Normalised subnormal: fp32 = sign | sub_exp | norm_mant | FP32_MANT_PAD zeros
        fp32_out = {
          sign, sub_exp, norm_mant, {FP32_MANT_PAD{1'b0}}
        };
    end else begin
      // Normal: fp32_exp = fp16_exp + EXP_BIAS_DIFF  (127 - 15 = 112)
      fp32_out = {sign, (8'(exp16) + 8'(EXP_BIAS_DIFF)), mant16, {FP32_MANT_PAD{1'b0}}};
    end
  end

endmodule
