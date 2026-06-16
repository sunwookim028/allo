`timescale 1ns / 1ps

// =============================================================================
// systolic.sv — Parameterized DIM×DIM weight-stationary systolic array
//
// Port interface uses PACKED arrays so iverilog can correctly elaborate
// genvar-indexed accesses in generate blocks and module instantiation
// connections (unpacked array port connections are broken in iverilog 12).
//
// Signal flow:
//   Activations: west → east, one entry per row  (sys_data_in[r])
//   Weights:     north → south, one per column   (sys_weight_in[c])
//   Partial sums: north → south, accumulated; bottom-row drives outputs
//   Switch:      pe[0][0] → right along row 0, then south down each column
//   Valid:       follows activations west → east
//   Accept_w:    per-column broadcast (sys_accept_w[c])
// =============================================================================
module systolic #(
    parameter DIM        = 8,
    parameter LATENCY    = 0,
    parameter DATA_WIDTH = 32
) (
    input logic clk,
    input logic rst_n,

    // Activation stream: packed [DIM-1:0][15:0], one fp16 per row from west
    input logic [DIM-1:0][15:0] sys_data_in,
    input logic [DIM-1:0]       sys_start,

    // Output: packed fp32 partial sums from bottom row, valid flags
    output logic [DIM-1:0][DATA_WIDTH-1:0] sys_data_out,
    output logic [DIM-1:0]                 sys_valid_out,

    // Weight stream: packed [DIM-1:0][15:0], one fp16 per column from north
    input logic [DIM-1:0][15:0] sys_weight_in,
    input logic [DIM-1:0]       sys_accept_w,

    // Switch: propagates top-left to bottom-right
    input logic sys_switch_in
);

  // ---------------------------------------------------------------------------
  // Internal PE wires — flat 1D arrays indexed by (r*DIM + c)
  // ---------------------------------------------------------------------------
  localparam TOTAL = DIM * DIM;

  /* verilator lint_off UNUSEDSIGNAL */
  logic [          15:0] pe_input_out_w [TOTAL];
  logic [DATA_WIDTH-1:0] pe_psum_out_w  [TOTAL];
  logic [          15:0] pe_weight_out_w[TOTAL];
  logic                  pe_valid_out_w [TOTAL];
  logic                  pe_switch_out_w[TOTAL];
  logic [          15:0] pe_input_in_w  [TOTAL];
  logic                  pe_valid_in_w  [TOTAL];
  logic [DATA_WIDTH-1:0] pe_psum_in_w   [TOTAL];
  logic [          15:0] pe_weight_in_w [TOTAL];
  logic                  pe_switch_in_w [TOTAL];
  /* verilator lint_on UNUSEDSIGNAL */

  // ---------------------------------------------------------------------------
  // Generate DIM × DIM PE grid
  // ---------------------------------------------------------------------------
  genvar r, c;
  generate
    for (r = 0; r < DIM; r++) begin : gen_row
      for (c = 0; c < DIM; c++) begin : gen_col

        if (c == 0) begin : g_west_edge
          assign pe_input_in_w[r*DIM+c] = sys_data_in[r];
          assign pe_valid_in_w[r*DIM+c] = sys_start[r];
        end else begin : g_west_internal
          assign pe_input_in_w[r*DIM+c] = pe_input_out_w[r*DIM+c-1];
          assign pe_valid_in_w[r*DIM+c] = pe_valid_out_w[r*DIM+c-1];
        end

        if (r == 0) begin : g_north_edge
          assign pe_psum_in_w[r*DIM+c]   = '0;
          assign pe_weight_in_w[r*DIM+c] = sys_weight_in[c];
        end else begin : g_north_internal
          assign pe_psum_in_w[r*DIM+c]   = pe_psum_out_w[(r-1)*DIM+c];
          assign pe_weight_in_w[r*DIM+c] = pe_weight_out_w[(r-1)*DIM+c];
        end

        // Switch routing: origin at pe[0][0]; right along row 0; then south
        // down each column.
        if (r == 0 && c == 0) begin : g_switch_origin
          assign pe_switch_in_w[r*DIM+c] = sys_switch_in;
        end else if (r == 0) begin : g_switch_row0
          assign pe_switch_in_w[r*DIM+c] = pe_switch_out_w[c-1];
        end else begin : g_switch_col
          assign pe_switch_in_w[r*DIM+c] = pe_switch_out_w[(r-1)*DIM+c];
        end

        pe #(
            .DATA_WIDTH(DATA_WIDTH),
            .LATENCY   (LATENCY)
        ) u_pe (
            .clk       (clk),
            .rst_n     (rst_n),
            .pe_enabled(1'b1),

            .pe_input_in (pe_input_in_w[r*DIM+c]),
            .pe_valid_in (pe_valid_in_w[r*DIM+c]),
            .pe_switch_in(pe_switch_in_w[r*DIM+c]),

            .pe_psum_in    (pe_psum_in_w[r*DIM+c]),
            .pe_weight_in  (pe_weight_in_w[r*DIM+c]),
            .pe_accept_w_in(sys_accept_w[c]),

            .pe_input_out (pe_input_out_w[r*DIM+c]),
            .pe_valid_out (pe_valid_out_w[r*DIM+c]),
            .pe_switch_out(pe_switch_out_w[r*DIM+c]),
            .pe_psum_out  (pe_psum_out_w[r*DIM+c]),
            .pe_weight_out(pe_weight_out_w[r*DIM+c])
        );
      end
    end
  endgenerate

  // ---------------------------------------------------------------------------
  // Bottom-row → packed output arrays
  // ---------------------------------------------------------------------------
  generate
    for (genvar oc = 0; oc < DIM; oc++) begin : gen_out
      assign sys_data_out[oc]  = pe_psum_out_w[(DIM-1)*DIM+oc];
      assign sys_valid_out[oc] = pe_valid_out_w[(DIM-1)*DIM+oc];
    end
  endgenerate

endmodule
