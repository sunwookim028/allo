// fp32_add — IEEE 754 FP32 add.
//
// SYNTHESIS path: synthesisable behavioral IEEE 754 model.  Replaces
// fpnew_fma ADD mode (~1000 LUTs/instance from the full FMA datapath).
// At DIM=16 (512 fp32_add instances across PE + MXU acc), this saves ~350 K
// LUTs.  LATENCY=0: purely combinational.  LATENCY=1: two-stage pipeline
// split after mantissa add (before normalization) — breaks the ~36-level
// combinational chain (~11.5 ns at 97 MHz) into two ~18-level halves.
//
// SIMULATION path: DPI-C (Verilator) or same IEEE 754 model (iverilog).
// LATENCY=1 in Verilator: DPI result is still combinational; the caller's
// FSM adds one extra wait state so correctness is preserved.
//
// Fixed-point fallback (FORMAT != "FP32") is a saturating integer adder and
// is identical in both paths.
`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
module fp32_add #(
    parameter LATENCY   = 0,
    parameter              FORMAT    = "FP32",
    parameter          INT_BITS  = 16,
    parameter          FRAC_BITS = 16,
    parameter          WIDTH     = 32
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic             clk,
    input  logic             rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] result
);
  /* verilator lint_on UNUSEDPARAM */
`ifdef VERILATOR
  import "DPI-C" pure function int dpi_fp32_add(
    input int a_in,
    input int b_in
  );
`endif

  generate
    if (FORMAT == "FP32") begin : fp32_mode

`ifdef VERILATOR
      // LATENCY=1 in Verilator: DPI result is combinational; caller FSM adds the extra wait.
      always @(*) result = dpi_fp32_add(a, b);
`else
      // IEEE 754 FP32 adder — synthesisable for Vivado, behavioral for iverilog.
      // Input extraction shared by LATENCY=0 and LATENCY=1 paths.
      logic a_sign, b_sign;
      logic [7:0] a_exp, b_exp;
      logic [22:0] a_mant, b_mant;
      logic a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;

      assign a_sign = a[31];
      assign a_exp  = a[30:23];
      assign a_mant = a[22:0];
      assign b_sign = b[31];
      assign b_exp  = b[30:23];
      assign b_mant = b[22:0];

      assign a_nan  = (a_exp == 8'hFF) && (a_mant != 0);
      assign b_nan  = (b_exp == 8'hFF) && (b_mant != 0);
      assign a_inf  = (a_exp == 8'hFF) && (a_mant == 0);
      assign b_inf  = (b_exp == 8'hFF) && (b_mant == 0);
      assign a_zero = (a_exp == 8'h00) && (a_mant == 0);
      assign b_zero = (b_exp == 8'h00) && (b_mant == 0);

      if (LATENCY == 0) begin : lat0
        // Purely combinational — alignment + mantissa add + normalization in one always block.
        logic result_sign;
        logic [23:0] a_mant_ext, b_mant_ext;
        logic [24:0] sum_mant;
        integer larger_exp_i, exp_diff_i, result_exp_i;
        logic          normalize_done;
        integer        shift;
        /* verilator lint_off UNUSEDSIGNAL */
        logic   [24:0] mant_sub;
        /* verilator lint_on UNUSEDSIGNAL */
        integer        i;

        always @(*) begin
          result         = '0;
          result_sign    = 1'b0;
          a_mant_ext     = '0;
          b_mant_ext     = '0;
          sum_mant       = '0;
          larger_exp_i   = 0;
          exp_diff_i     = 0;
          result_exp_i   = 0;
          normalize_done = 1'b0;
          shift          = 0;
          mant_sub       = '0;

          if (a_nan || b_nan) begin
            result = 32'h7FC00000;
          end else if (a_inf && b_inf) begin
            result = (a_sign == b_sign) ? {a_sign, 8'hFF, 23'h0} : 32'h7FC00000;
          end else if (a_inf) begin
            result = {a_sign, 8'hFF, 23'h0};
          end else if (b_inf) begin
            result = {b_sign, 8'hFF, 23'h0};
          end else if (a_zero && b_zero) begin
            result = {a_sign & b_sign, 8'h00, 23'h0};
          end else if (a_zero) begin
            result = b;
          end else if (b_zero) begin
            result = a;
          end else begin
            a_mant_ext = (a_exp == 8'h00) ? {1'b0, a_mant} : {1'b1, a_mant};
            b_mant_ext = (b_exp == 8'h00) ? {1'b0, b_mant} : {1'b1, b_mant};

            if (a_exp >= b_exp) begin
              larger_exp_i = (a_exp);
              exp_diff_i   = (a_exp) - (b_exp);
              b_mant_ext   = b_mant_ext >> exp_diff_i;
            end else begin
              larger_exp_i = (b_exp);
              exp_diff_i   = (b_exp) - (a_exp);
              a_mant_ext   = a_mant_ext >> exp_diff_i;
            end

            if (a_sign == b_sign) begin
              sum_mant    = a_mant_ext + b_mant_ext;
              result_sign = a_sign;
            end else begin
              if (a_mant_ext >= b_mant_ext) begin
                sum_mant    = a_mant_ext - b_mant_ext;
                result_sign = a_sign;
              end else begin
                sum_mant    = b_mant_ext - a_mant_ext;
                result_sign = b_sign;
              end
            end

            if (larger_exp_i == 0) begin
              if (sum_mant == 0) begin
                result = 32'h00000000;
              end else if (sum_mant[23]) begin
                result = {result_sign, 8'h01, sum_mant[22:0]};
              end else begin
                result = {result_sign, 8'h00, sum_mant[22:0]};
              end
            end else begin
              result_exp_i   = larger_exp_i;
              normalize_done = 1'b0;

              if (sum_mant[24]) begin
                sum_mant       = sum_mant >> 1;
                result_exp_i   = result_exp_i + 1;
                normalize_done = 1'b1;
              end else if (sum_mant[23]) begin
                normalize_done = 1'b1;
              end

              if (!normalize_done) begin
                for (i = 22; i >= 0; i = i - 1) begin
                  if (sum_mant[i] && !normalize_done) begin
                    sum_mant       = sum_mant << (23 - i);
                    result_exp_i   = result_exp_i - (23 - i);
                    normalize_done = 1'b1;
                  end
                end
              end

              if (sum_mant == 0) begin
                result = 32'h00000000;
              end else if (result_exp_i >= 255) begin
                result = {result_sign, 8'hFF, 23'h0};
              end else if (result_exp_i <= 0) begin
                shift = 1 - result_exp_i;
                if (shift >= 25) begin
                  result = {result_sign, 8'h00, 23'h0};
                end else begin
                  mant_sub = sum_mant >> shift;
                  result   = {result_sign, 8'h00, mant_sub[22:0]};
                end
              end else begin
                result = {result_sign, result_exp_i[7:0], sum_mant[22:0]};
              end
            end
          end
        end

      end else begin : lat1
        // Two-stage pipeline.
        // Stage 1: special cases + exponent alignment + mantissa add → pipeline FF.
        // Stage 2: normalization → result.
        // Splits the ~36-level combinational chain into two ~18-level halves.

        // Stage 1 combinational outputs (→ pipeline FF)
        logic        s1_is_special;
        logic [31:0] s1_special_result;
        logic        s1_result_sign;
        logic [ 7:0] s1_larger_exp;
        logic [24:0] s1_sum_mant;

        // Stage 1 local temporaries
        logic [23:0] s1_a_mant_ext, s1_b_mant_ext;
        integer        s1_exp_diff;

        // Pipeline registers
        logic          p1_is_special;
        logic   [31:0] p1_special_result;
        logic          p1_result_sign;
        logic   [ 7:0] p1_larger_exp;
        logic   [24:0] p1_sum_mant;

        // Stage 2 local temporaries
        logic          s2_normalize_done;
        integer s2_result_exp, s2_shift, s2_i;
        logic [24:0] s2_sum_mant;
        /* verilator lint_off UNUSEDSIGNAL */
        logic [24:0] s2_mant_sub;
        /* verilator lint_on UNUSEDSIGNAL */

        // Stage 1: alignment + mantissa add
        always @(*) begin
          s1_is_special     = 1'b0;
          s1_special_result = '0;
          s1_result_sign    = 1'b0;
          s1_larger_exp     = '0;
          s1_sum_mant       = '0;
          s1_a_mant_ext     = '0;
          s1_b_mant_ext     = '0;
          s1_exp_diff       = 0;

          if (a_nan || b_nan) begin
            s1_is_special = 1'b1;
            s1_special_result = 32'h7FC00000;
          end else if (a_inf && b_inf) begin
            s1_is_special     = 1'b1;
            s1_special_result = (a_sign == b_sign) ? {a_sign, 8'hFF, 23'h0} : 32'h7FC00000;
          end else if (a_inf) begin
            s1_is_special = 1'b1;
            s1_special_result = {a_sign, 8'hFF, 23'h0};
          end else if (b_inf) begin
            s1_is_special = 1'b1;
            s1_special_result = {b_sign, 8'hFF, 23'h0};
          end else if (a_zero && b_zero) begin
            s1_is_special = 1'b1;
            s1_special_result = {a_sign & b_sign, 8'h00, 23'h0};
          end else if (a_zero) begin
            s1_is_special = 1'b1;
            s1_special_result = b;
          end else if (b_zero) begin
            s1_is_special = 1'b1;
            s1_special_result = a;
          end else begin
            s1_a_mant_ext = (a_exp == 8'h00) ? {1'b0, a_mant} : {1'b1, a_mant};
            s1_b_mant_ext = (b_exp == 8'h00) ? {1'b0, b_mant} : {1'b1, b_mant};

            if (a_exp >= b_exp) begin
              s1_larger_exp = a_exp;
              s1_exp_diff   = (a_exp) - (b_exp);
              s1_b_mant_ext = s1_b_mant_ext >> s1_exp_diff;
            end else begin
              s1_larger_exp = b_exp;
              s1_exp_diff   = (b_exp) - (a_exp);
              s1_a_mant_ext = s1_a_mant_ext >> s1_exp_diff;
            end

            if (a_sign == b_sign) begin
              s1_sum_mant    = s1_a_mant_ext + s1_b_mant_ext;
              s1_result_sign = a_sign;
            end else begin
              if (s1_a_mant_ext >= s1_b_mant_ext) begin
                s1_sum_mant    = s1_a_mant_ext - s1_b_mant_ext;
                s1_result_sign = a_sign;
              end else begin
                s1_sum_mant    = s1_b_mant_ext - s1_a_mant_ext;
                s1_result_sign = b_sign;
              end
            end
          end
        end

        // Pipeline FF
        always @(posedge clk) begin
          p1_is_special     <= s1_is_special;
          p1_special_result <= s1_special_result;
          p1_result_sign    <= s1_result_sign;
          p1_larger_exp     <= s1_larger_exp;
          p1_sum_mant       <= s1_sum_mant;
        end

        // Stage 2: normalization
        always @(*) begin
          result            = '0;
          s2_result_exp     = 0;
          s2_normalize_done = 1'b0;
          s2_shift          = 0;
          s2_sum_mant       = '0;
          s2_mant_sub       = '0;
          s2_i              = 0;

          if (p1_is_special) begin
            result = p1_special_result;
          end else begin
            s2_sum_mant = p1_sum_mant;

            if (p1_larger_exp == 8'h00) begin
              if (s2_sum_mant == 0) begin
                result = 32'h00000000;
              end else if (s2_sum_mant[23]) begin
                result = {p1_result_sign, 8'h01, s2_sum_mant[22:0]};
              end else begin
                result = {p1_result_sign, 8'h00, s2_sum_mant[22:0]};
              end
            end else begin
              s2_result_exp     = (p1_larger_exp);
              s2_normalize_done = 1'b0;

              if (s2_sum_mant[24]) begin
                s2_sum_mant       = s2_sum_mant >> 1;
                s2_result_exp     = s2_result_exp + 1;
                s2_normalize_done = 1'b1;
              end else if (s2_sum_mant[23]) begin
                s2_normalize_done = 1'b1;
              end

              if (!s2_normalize_done) begin
                for (s2_i = 22; s2_i >= 0; s2_i = s2_i - 1) begin
                  if (s2_sum_mant[s2_i] && !s2_normalize_done) begin
                    s2_sum_mant       = s2_sum_mant << (23 - s2_i);
                    s2_result_exp     = s2_result_exp - (23 - s2_i);
                    s2_normalize_done = 1'b1;
                  end
                end
              end

              if (s2_sum_mant == 0) begin
                result = 32'h00000000;
              end else if (s2_result_exp >= 255) begin
                result = {p1_result_sign, 8'hFF, 23'h0};
              end else if (s2_result_exp <= 0) begin
                s2_shift = 1 - s2_result_exp;
                if (s2_shift >= 25) begin
                  result = {p1_result_sign, 8'h00, 23'h0};
                end else begin
                  s2_mant_sub = s2_sum_mant >> s2_shift;
                  result      = {p1_result_sign, 8'h00, s2_mant_sub[22:0]};
                end
              end else begin
                result = {p1_result_sign, s2_result_exp[7:0], s2_sum_mant[22:0]};
              end
            end
          end
        end

      end  // lat1
`endif

    end else begin : fixed_point_mode
      // Saturating integer adder — identical for synthesis and simulation.
      logic [WIDTH-1:0] sum;
      logic overflow;
      assign sum      = a + b;
      assign overflow = (a[WIDTH-1] == b[WIDTH-1]) && (sum[WIDTH-1] != a[WIDTH-1]);
      always @* begin
        if (overflow) result = a[WIDTH-1] ? {1'b1, {WIDTH - 1{1'b0}}} : {1'b0, {WIDTH - 1{1'b1}}};
        else result = sum;
      end
    end
  endgenerate

endmodule
