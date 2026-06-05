`timescale 1ns / 1ps

module pe #(
    parameter int          DATA_WIDTH = 32,
    parameter int unsigned LATENCY    = 0
) (
    input logic clk,
    input logic rst_n,

    // North wires of PE (partial sum + weight input)
    input logic [DATA_WIDTH-1:0] pe_psum_in,
    input logic [          15:0] pe_weight_in,   // fp16 input
    input logic                  pe_accept_w_in,

    // West wires of PE (activation + control)
    input logic [15:0] pe_input_in,   // fp16 input
    input logic        pe_valid_in,
    input logic        pe_switch_in,
    input logic        pe_enabled,

    // South wires of the PE (partial sum + weight output)
    output logic [DATA_WIDTH-1:0] pe_psum_out,
    output logic [          15:0] pe_weight_out, // fp16 pass-through

    // East wires of the PE (activation + control out)
    output logic [15:0] pe_input_out,  // fp16 pass-through
    output logic        pe_valid_out,
    output logic        pe_switch_out
);

  // Internal data paths (FP32 arithmetic)
  logic [DATA_WIDTH-1:0] mult_out;
  (* DONT_TOUCH = "true" *)logic [DATA_WIDTH-1:0] mult_out_reg;  // Fix 10: pipeline stage between DSP and CARRY8
  logic                  mult_valid_reg;
  logic [DATA_WIDTH-1:0] mac_out;
  logic [          15:0] weight_reg_active;  // foreground weight register (fp16)
  logic [          15:0] weight_reg_inactive;  // background weight register (fp16)

  // ------------------------------------------------------------------------
  // FP16 × FP16 → FP32 multiply: mult_out = pe_input_in × weight_reg_active
  // One DSP48E2 per PE (synthesis) vs. ~2000 LUTs from fpnew_fma FP32 MUL.
  // ------------------------------------------------------------------------
  fp16_mul_fp32 u_fp16_mul (
      .a     (pe_input_in),
      .b     (weight_reg_active),
      .result(mult_out)
  );

  // ------------------------------------------------------------------------
  // FP32 add: mac_out = mult_out + pe_psum_in
  // ------------------------------------------------------------------------
  fp32_add #(
      .LATENCY  (LATENCY),
      .FORMAT   ("FP32"),
      .INT_BITS (16),
      .FRAC_BITS(16),
      .WIDTH    (DATA_WIDTH)
  ) adder (
      .clk   (clk),
      .rst_n (rst_n),
      .a     (mult_out_reg),
      .b     (pe_psum_in),
      .result(mac_out)
  );

  // ------------------------------------------------------------------------
  // Control-signal delay lines (2*LATENCY stages).
  // fp16_mul_fp32 is combinational (0 cycles); fp32_add adds LATENCY cycles.
  // mult_out_reg adds 1 explicit pipeline stage (Fix 10); pe_valid_out is
  // one cycle ahead of pe_psum_out by design (mxu.sv PE_LAT += 1 accounts).
  // ------------------------------------------------------------------------
  localparam int SR_DEPTH = 2 * LATENCY;

  logic        valid_sr [SR_DEPTH+1];
  logic        switch_sr[SR_DEPTH+1];
  logic [15:0] input_sr [SR_DEPTH+1];

  assign valid_sr[0]  = pe_valid_in;
  assign switch_sr[0] = pe_switch_in;
  assign input_sr[0]  = pe_input_in;

  generate
    if (SR_DEPTH > 0) begin : gen_sr_block
      for (genvar i = 1; i <= SR_DEPTH; i++) begin : gen_sr
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            valid_sr[i]  <= 1'b0;
            switch_sr[i] <= 1'b0;
            input_sr[i]  <= '0;
          end else begin
            valid_sr[i]  <= valid_sr[i-1];
            switch_sr[i] <= switch_sr[i-1];
            input_sr[i]  <= input_sr[i-1];
          end
        end
      end
    end
  endgenerate

  // ------------------------------------------------------------------------
  // Sequential control + register updates
  // ASIC FIX: Separated async reset from sync enable (pe_enabled)
  // ------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Async reset only
      pe_input_out        <= '0;
      pe_psum_out         <= '0;
      pe_weight_out       <= '0;
      pe_valid_out        <= 1'b0;
      pe_switch_out       <= 1'b0;
      weight_reg_active   <= '0;
      weight_reg_inactive <= '0;
      mult_out_reg        <= '0;
      mult_valid_reg      <= 1'b0;
    end else if (!pe_enabled) begin
      // Sync clear when disabled
      pe_input_out        <= '0;
      pe_psum_out         <= '0;
      pe_weight_out       <= '0;
      pe_valid_out        <= 1'b0;
      pe_switch_out       <= 1'b0;
      weight_reg_active   <= '0;
      weight_reg_inactive <= '0;
      mult_out_reg        <= '0;
      mult_valid_reg      <= 1'b0;
    end else begin
      // Pass-through control signals (delayed to match mac_out pipeline depth)
      pe_valid_out   <= valid_sr[SR_DEPTH];
      pe_switch_out  <= switch_sr[SR_DEPTH];

      // Fix 10: latch multiply result one cycle before fp32_add (cuts DSP→CARRY8 path)
      mult_out_reg   <= mult_out;
      mult_valid_reg <= valid_sr[SR_DEPTH];

      // Weight register updates
      if (pe_accept_w_in) begin
        // Load new weight into inactive register and forward it south
        weight_reg_inactive <= pe_weight_in;
        pe_weight_out       <= pe_weight_in;
      end else begin
        pe_weight_out <= '0;
      end

      // Swap active weight when switch is asserted
      if (pe_switch_in) begin
        if (pe_accept_w_in) begin
          // Directly load the new weight as active this cycle
          weight_reg_active <= pe_weight_in;
        end else begin
          // Promote previously loaded inactive weight
          weight_reg_active <= weight_reg_inactive;
        end
      end
      // else: retain weight_reg_active

      // pe_input_out: 1-cycle latency (valid_sr[SR_DEPTH])
      if (valid_sr[SR_DEPTH]) begin
        pe_input_out <= input_sr[SR_DEPTH];
      end

      // pe_psum_out: 2-cycle latency (mult_valid_reg, one cycle after mult_out_reg)
      if (mult_valid_reg) begin
        pe_psum_out <= mac_out;  // FP32 MAC result (mult_out_reg + pe_psum_in)
      end else begin
        pe_psum_out <= '0;
      end
    end
  end

endmodule
