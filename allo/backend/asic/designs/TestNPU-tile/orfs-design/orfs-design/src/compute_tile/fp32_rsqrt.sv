// fp32_rsqrt — combinational IEEE 754 FP32 1/sqrt(x).
//
// Algorithm: Quake magic-constant initial estimate + 2 Newton-Raphson iterations.
//   y0 = bit_cast(0x5F3759DF − (bit_cast_int(x) >> 1))   (~7 bits accurate)
//   y1 = y0 * (1.5 − 0.5*x*y0*y0)                        (~14 bits)
//   y2 = y1 * (1.5 − 0.5*x*y1*y1)                        (~23 bits, full FP32)
//
// Special cases:
//   x ≤ 0 (negative, −0, +0) → quiet NaN  (undefined)
//   +Inf                      → +0
//   NaN                       → quiet NaN
//
// Instantiates fp32_mul and fp32_add (combinational).

module fp32_rsqrt (
    input  logic [31:0] x,
    output logic [31:0] result
);
  localparam logic [31:0] HALF = 32'h3F000000;  // 0.5
  localparam logic [31:0] ONEHALF = 32'h3FC00000;  // 1.5
  localparam logic [31:0] QNAN = 32'hFFC00000;  // quiet NaN

  logic x_nan, x_inf, x_neg_or_zero;
  assign x_nan         = (&x[30:23]) & (|x[22:0]);
  assign x_inf         = (&x[30:23]) & ~(|x[22:0]);
  assign x_neg_or_zero = x[31] | (x[30:0] == 31'h0);

  // ------ Initial estimate y0 ------
  logic [31:0] y0;
  assign y0 = 32'h5F3759DF - {1'b0, x[30:1]};  // integer subtraction on raw bits

  // ------ Newton-Raphson iteration: y_new = y * (1.5 − 0.5*x*y*y) ------
  // First iteration (y0 → y1)
  logic [31:0] hx, y0sq, hxy0sq, nr0_inner, y1;
  fp32_mul u_hx (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(HALF),
      .b(x),
      .result(hx)
  );
  fp32_mul u_y0sq (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(y0),
      .b(y0),
      .result(y0sq)
  );
  fp32_mul u_hxy0 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(hx),
      .b(y0sq),
      .result(hxy0sq)
  );
  fp32_add u_nr0 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(ONEHALF),
      .b({~hxy0sq[31], hxy0sq[30:0]}),
      .result(nr0_inner)
  );
  fp32_mul u_y1 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(y0),
      .b(nr0_inner),
      .result(y1)
  );

  // Second iteration (y1 → y2)
  logic [31:0] y1sq, hxy1sq, nr1_inner, y2;
  fp32_mul u_y1sq (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(y1),
      .b(y1),
      .result(y1sq)
  );
  fp32_mul u_hxy1 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(hx),
      .b(y1sq),
      .result(hxy1sq)
  );
  fp32_add u_nr1 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(ONEHALF),
      .b({~hxy1sq[31], hxy1sq[30:0]}),
      .result(nr1_inner)
  );
  fp32_mul u_y2 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(y1),
      .b(nr1_inner),
      .result(y2)
  );

  always_comb begin
    if (x_nan || x_neg_or_zero) result = QNAN;
    else if (x_inf) result = 32'h00000000;  // 1/sqrt(Inf) = 0
    else result = y2;
  end

endmodule
