// fp32_exp2 — combinational IEEE 754 FP32 2^x.
//
// Algorithm:
//   Split x = n + f  where  n = floor(x) (integer), f in [0, 1).
//   2^x = 2^n * 2^f.
//   2^n:  adjust result biased exponent by n.
//   2^f:  4-term Horner polynomial (Minimax for f in [0,1), ~23-bit accuracy):
//         p = c0 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
//   Result = {0, n+127, p[22:0]}  (p in [1,2) so p[30:23]==127).
//
// Special cases:
//   NaN in      → quiet NaN out
//   +Inf        → +Inf
//   -Inf / x < -149  → +0
//   x > 127.999      → +Inf  (biased_exp would overflow)
//
// Instantiates fp32_add and fp32_mul (combinational, no clock).

module fp32_exp2 (
    input  logic [31:0] x,
    output logic [31:0] result
);
  // ------------------------------------------------------------------ //
  // FP32 field extraction
  // ------------------------------------------------------------------ //
  logic        xs;
  logic [ 7:0] xe;
  logic [22:0] xm;
  assign xs = x[31];
  assign xe = x[30:23];
  assign xm = x[22:0];

  logic x_nan, x_pinf, x_ninf, x_zero;
  assign x_nan  = (&xe) & (|xm);
  assign x_pinf = (&xe) & ~(|xm) & ~xs;
  assign x_ninf = (&xe) & ~(|xm) & xs;
  assign x_zero = (xe == 8'h00);  // subnormals treated as zero

  // True exponent E = xe − 127 (signed 9-bit)
  logic signed [8:0] xE;
  assign xE = {1'b0, xe} - 9'sd127;

  // ------------------------------------------------------------------ //
  // Overflow / underflow detection
  //   Overflow : x > 127  →  xe≥135 (E≥8) or xe=134 and xm nonzero (>128)
  //              or xe=134 and xm=0 (x=128); but 2^128 is Inf in fp32 so
  //              conservatively overflow when x >= 128 (xe >= 134+1? no)
  //              x=127.999... → xe=133 (E=6, 64≤|x|<128, OK).
  //              x=128.0      → xe=134, xs=0, xm=0 → biased result exp = 255 = Inf.
  //   Underflow : x < −149  →  xs=1 and xe≥135 (|x|≥128 negative) or
  //              special small value.  Conservatively underflow when x ≤ −150.
  // ------------------------------------------------------------------ //
  logic ovf, unf;
  // overflow: positive x with E >= 7 (x >= 128)
  assign ovf = !xs & !x_nan & (xE >= 9'sd7);
  // underflow: negative x with E >= 7 (|x| >= 128, result < 2^-128 ≈ 0)
  //            or x is -Inf
  assign unf = (xs & !x_nan & (xE >= 9'sd7)) | x_ninf;

  // ------------------------------------------------------------------ //
  // Integer part n = floor(x), represented as signed 8-bit.
  // Range clamped to [-127, 127] for valid FP32 outputs.
  // ------------------------------------------------------------------ //
  logic signed [ 8:0] n_int;  // floor(x)
  logic        [22:0] frac_mant;  // fractional mantissa bits of |x|, left-aligned in 23 bits
  logic               has_frac;  // x is not an exact integer

  // full_mant = {implicit 1, xm} — 24-bit value
  logic        [23:0] full_mant;
  assign full_mant = {1'b1, xm};

  // Shift amount to expose fractional bits: shift full_mant left by (E+1) to
  // move all integer bits off the top; remaining bits are fractional, left-aligned.
  // Valid only for 0 ≤ E < 23.
  logic [4:0] frac_shift;
  always @* begin
    if (xE < 9'sd0) frac_shift = 5'd0;
    else if (xE >= 9'sd23) frac_shift = 5'd23;
    else frac_shift = xE[4:0] + 5'd1;
  end

  // frac_mant: the fractional bits of |x|, left-aligned into 23 bits.
  // For xE < 0: all of full_mant is fractional (but we represent |x| differently below).
  // For 0 ≤ xE < 23: shift out integer bits.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [23:0] shifted_fm;
  /* verilator lint_on UNUSEDSIGNAL */
  assign shifted_fm = full_mant << frac_shift;
  assign frac_mant  = shifted_fm[22:0];
  assign has_frac   = (|frac_mant) & !x_zero;

  // Integer magnitude |n| for positive x or ceil(|x|) for negative non-integer.
  logic [7:0] n_abs;
  always @* begin
    if (x_zero || xE < 9'sd0) n_abs = 8'd0;
    else if (xE >= 9'sd7) n_abs = 8'd127;  // clamped (handled by ovf/unf)
    else begin
      // integer bits = top (xE+1) bits of full_mant → shift right by (23-xE)
      n_abs = (full_mant >> (5'd23 - xE[4:0]));
    end
  end

  // floor(x):
  //   positive non-zero : n = +n_abs
  //   positive zero / xE<0 : n = 0
  //   negative integer   : n = −n_abs
  //   negative non-integer: n = −(n_abs + 1)  (floor rounds toward −∞)
  always @* begin
    if (x_zero) n_int = 9'sd0;
    else if (!xs) begin  // positive
      n_int = {1'b0, n_abs};
    end else begin  // negative
      if (xE < 9'sd0)  // |x| < 1 → floor = −1
        n_int = -9'sd1;
      else if (!has_frac)  // exact integer
        n_int = -{1'b0, n_abs};
      else  // floor rounds down
        n_int = -{1'b0, n_abs} - 9'sd1;
    end
  end

  // ------------------------------------------------------------------ //
  // float(n): convert n_int to FP32 for f = x − float(n)
  // ------------------------------------------------------------------ //
  logic        fn_sign;
  logic [ 7:0] fn_abs;
  logic [ 2:0] fn_msb;  // floor(log2(fn_abs))
  logic [ 7:0] fn_exp;
  logic [22:0] fn_mant;
  logic [31:0] float_n;

  assign fn_sign = n_int[8];  // sign bit of 9-bit signed
  assign fn_abs  = fn_sign ? (-n_int) : (n_int);

  always @* begin
    casez (fn_abs)
      8'b1???????: fn_msb = 3'd7;
      8'b01??????: fn_msb = 3'd6;
      8'b001?????: fn_msb = 3'd5;
      8'b0001????: fn_msb = 3'd4;
      8'b00001???: fn_msb = 3'd3;
      8'b000001??: fn_msb = 3'd2;
      8'b0000001?: fn_msb = 3'd1;
      default:     fn_msb = 3'd0;
    endcase
  end

  logic [7:0] fn_mask;
  assign fn_mask = (8'h01 << fn_msb) - 8'h01;  // bits below leading 1
  assign fn_mant = (fn_abs & fn_mask) << (5'd23 - {2'b0, fn_msb});
  assign fn_exp  = 8'd127 + {5'd0, fn_msb};
  assign float_n = (n_int == 9'sd0) ? 32'h0 : {fn_sign, fn_exp, fn_mant};

  // ------------------------------------------------------------------ //
  // f = x − float(n)  via fp32_add(x, −float_n)
  // f is in [0, 1) for valid inputs.
  // ------------------------------------------------------------------ //
  logic [31:0] neg_float_n, f_fp32;
  assign neg_float_n = {~float_n[31], float_n[30:0]};

  fp32_add u_f_sub (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(x),
      .b(neg_float_n),
      .result(f_fp32)
  );

  // ------------------------------------------------------------------ //
  // Polynomial: p = c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*c6)))))
  // Taylor / minimax coefficients for 2^f, f in [0,1) — degree 6:
  //   c0 = 1.0             32'h3F800000
  //   c1 = ln2             32'h3F317218
  //   c2 = ln2^2/2         32'h3E75FDF0
  //   c3 = ln2^3/6         32'h3D634247
  //   c4 = ln2^4/24        32'h3C1D9E88
  //   c5 = ln2^5/120       32'h3AAEC42B  (≈ 1.333e-3)
  //   c6 = ln2^6/720       32'h39216A85  (≈ 1.540e-4)
  // Max polynomial truncation error < 1.5e-5, well below rtol=1e-4.
  // Horner inner-to-outer: start from c6.
  // ------------------------------------------------------------------ //
  localparam [31:0] C0 = 32'h3F800000;  // 1.0
  localparam [31:0] C1 = 32'h3F317218;  // 0.693147
  localparam [31:0] C2 = 32'h3E75FDF0;  // 0.240227
  localparam [31:0] C3 = 32'h3D634247;  // 0.055504
  localparam [31:0] C4 = 32'h3C1D9E88;  // 0.009618
  localparam [31:0] C5 = 32'h3AAEC42B;  // 0.001333
  localparam [31:0] C6 = 32'h39216A85;  // 0.000154

  // h5 = c6*f + c5  (innermost, new)
  logic [31:0] h5_mul, h5;
  fp32_mul u_h5m (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(C6),
      .b(f_fp32),
      .result(h5_mul)
  );
  fp32_add u_h5a (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h5_mul),
      .b(C5),
      .result(h5)
  );

  // h4 = h5*f + c4  (new)
  logic [31:0] h4_mul, h4;
  fp32_mul u_h4m (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h5),
      .b(f_fp32),
      .result(h4_mul)
  );
  fp32_add u_h4a (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h4_mul),
      .b(C4),
      .result(h4)
  );

  // h3 = h4*f + c3
  logic [31:0] h3_mul, h3;
  fp32_mul u_h3m (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h4),
      .b(f_fp32),
      .result(h3_mul)
  );
  fp32_add u_h3a (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h3_mul),
      .b(C3),
      .result(h3)
  );

  // h2 = h3*f + c2
  logic [31:0] h2_mul, h2;
  fp32_mul u_h2m (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h3),
      .b(f_fp32),
      .result(h2_mul)
  );
  fp32_add u_h2a (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h2_mul),
      .b(C2),
      .result(h2)
  );

  // h1 = h2*f + c1
  logic [31:0] h1_mul, h1;
  fp32_mul u_h1m (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h2),
      .b(f_fp32),
      .result(h1_mul)
  );
  fp32_add u_h1a (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h1_mul),
      .b(C1),
      .result(h1)
  );

  // p = h1*f + c0  (p in [1, 2)); p[31] (sign) is always 0 in normal range
  /* verilator lint_off UNUSEDSIGNAL */
  logic [31:0] p_mul, p;
  /* verilator lint_on UNUSEDSIGNAL */
  fp32_mul u_pm (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(h1),
      .b(f_fp32),
      .result(p_mul)
  );
  fp32_add u_pa (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(p_mul),
      .b(C0),
      .result(p)
  );

  // ------------------------------------------------------------------ //
  // Assemble result
  // biased exponent of result = n_int + 127; mantissa from p[22:0].
  // p should be in [1,2) so p[30:23]=127; if polynomial rounds to 2.0 (p[30:23]=128)
  // add 1 to the result exponent and use zero mantissa.
  // ------------------------------------------------------------------ //
  logic [7:0] res_exp_raw;
  logic       p_rounded_up;
  assign res_exp_raw  = (n_int + 9'sd127);
  assign p_rounded_up = (p[30:23] == 8'd128);  // 2.0 due to rounding

  logic [7:0] res_exp;
  assign res_exp = p_rounded_up ? res_exp_raw + 8'd1 : res_exp_raw;

  logic [22:0] res_mant;
  assign res_mant = p_rounded_up ? 23'h0 : p[22:0];

  // Final mux with special cases
  always @* begin
    if (x_nan) result = {1'b0, 8'hFF, 1'b1, xm[21:0]};  // quiet NaN
    else if (x_pinf) result = 32'h7F800000;  // +Inf
    else if (ovf) result = 32'h7F800000;  // overflow → +Inf
    else if (unf) result = 32'h00000000;  // underflow → +0
    else result = {1'b0, res_exp, res_mant};
  end

endmodule
