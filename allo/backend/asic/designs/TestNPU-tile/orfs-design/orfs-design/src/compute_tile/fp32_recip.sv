// fp32_recip — combinational IEEE 754 FP32 1/x.
//
// Algorithm: integer-subtraction initial estimate + 3 Newton-Raphson iterations.
//   y0 = bit_cast_float(0x7F000000 − bit_cast_int(|x|)), sign carried from x.
//        This gives |x·y0 − 1| ≤ 0.125 for all normal FP32 inputs.
//   NR: y_new = y * (2 − x * y)
//   Three iterations: max relative error ≤ 0.125^8 ≈ 6e-8 (< 1e-5 rtol).
//
// Special cases:
//   NaN  → quiet NaN
//   ±Inf → ±0
//   ±0   → ±Inf
//   normal / subnormal handled by NR from initial estimate

module fp32_recip (
    input  logic [31:0] x,
    output logic [31:0] result
);
  localparam [31:0] TWO = 32'h40000000;
  localparam [31:0] QNAN = 32'hFFC00000;

  logic x_nan, x_inf, x_zero;
  assign x_nan  = (&x[30:23]) & (|x[22:0]);
  assign x_inf  = (&x[30:23]) & ~(|x[22:0]);
  assign x_zero = (x[30:0] == 31'h0);

  // Initial estimate: 0x7F000000 − |x_bits| (integer subtraction on magnitude).
  // Gives y0 ∈ [2^(−k−1), 2^(−k)] for x ∈ [2^k, 2^(k+1)), with |x·y0−1| ≤ 0.125.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [31:0] y0_abs_bits, y0;
  /* verilator lint_on UNUSEDSIGNAL */
  assign y0_abs_bits = 32'h7F000000 - {1'b0, x[30:0]};
  assign y0 = {x[31], y0_abs_bits[30:0]};

  // First NR iteration: y1 = y0 * (2 − x*y0)
  logic [31:0] xy0, two_minus_xy0, y1;
  fp32_mul u_xy0 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(x),
      .b(y0),
      .result(xy0)
  );
  fp32_add u_nr0 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(TWO),
      .b({~xy0[31], xy0[30:0]}),
      .result(two_minus_xy0)
  );
  fp32_mul u_y1 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(y0),
      .b(two_minus_xy0),
      .result(y1)
  );

  // Second NR iteration: y2 = y1 * (2 − x*y1)
  logic [31:0] xy1, two_minus_xy1, y2;
  fp32_mul u_xy1 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(x),
      .b(y1),
      .result(xy1)
  );
  fp32_add u_nr1 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(TWO),
      .b({~xy1[31], xy1[30:0]}),
      .result(two_minus_xy1)
  );
  fp32_mul u_y2 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(y1),
      .b(two_minus_xy1),
      .result(y2)
  );

  // Third NR iteration: y3 = y2 * (2 − x*y2)
  logic [31:0] xy2, two_minus_xy2, y3;
  fp32_mul u_xy2 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(x),
      .b(y2),
      .result(xy2)
  );
  fp32_add u_nr2 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(TWO),
      .b({~xy2[31], xy2[30:0]}),
      .result(two_minus_xy2)
  );
  fp32_mul u_y3 (
      .clk(1'b0),
      .rst_n(1'b1),
      .a(y2),
      .b(two_minus_xy2),
      .result(y3)
  );

  always @* begin
    if (x_nan) result = QNAN;
    else if (x_inf) result = {x[31], 31'h0};  // 1/±Inf = ±0
    else if (x_zero) result = {x[31], 8'hFF, 23'h0};  // 1/±0 = ±Inf
    else result = y3;
  end

endmodule
