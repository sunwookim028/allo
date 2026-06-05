`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: vpu
// Description: SIMD Vector Processing Unit with register file
//              - 8-lane SIMD execution (8 parallel vpu_op ALU instances)
//              - Vector register file (V0-V7)
//              - Supports VLOAD, VSTORE, VCOMPUTE operations
//              - Scalar broadcast support for vector-scalar operations
//////////////////////////////////////////////////////////////////////////////////

module vpu
  import npu_config_pkg::*, vpu_pkg::*;
(
    input logic clk,
    input logic rst_n,

    // Instruction fields from decoder
    input logic [12:0] addr_a,
    input logic [12:0] addr_b,
    input logic [12:0] addr_out,
    input logic [2:0] vpu_type,
    input logic [2:0] vreg_dst,
    input logic [2:0] vreg_a,
    input logic [2:0] vreg_b,
    input logic [2:0] vpu_opcode,
    input logic scalar_b,
    input logic start,

    // BRAM interface
    output logic [L1_ADDR_WIDTH-1:0] bram_addr,
    output logic [L1_DATA_WIDTH-1:0] bram_din,
    input logic [L1_DATA_WIDTH-1:0] bram_dout,
    output logic bram_en,
    output logic bram_we,

    output logic done
);
  localparam int NUM_LANES = L1_DATA_WIDTH / DATA_WIDTH;

  // SIMD internal signals
  logic start_simd;
  logic done_simd;
  logic [L1_ADDR_WIDTH-1:0] bram_addr_simd;
  logic [L1_DATA_WIDTH-1:0] bram_din_simd;
  logic bram_en_simd;
  logic bram_we_simd;

  // Start FSM on any VPU instruction
  assign start_simd = start;

  // Route output signals
  assign bram_addr = bram_addr_simd;
  assign bram_din = bram_din_simd;
  assign bram_en = bram_en_simd;
  assign bram_we = bram_we_simd;
  assign done = done_simd;


  // FSM states
  typedef enum logic [4:0] {
    IDLE,
    VLOAD_REQ,
    VLOAD_WAIT1,
    VLOAD_CAPTURE,
    VSTORE_REQ,
    VCOMPUTE_READ,
    VCOMPUTE_EXEC,
    SCALAR_READ_A,
    SCALAR_WAIT_A,
    SCALAR_LATCH_A,
    SCALAR_WAIT_B,
    SCALAR_COMPUTE,
    DONE_STATE
  } state_t;

  state_t state;

  // Vector register file
  logic [2:0] rf_rd_addr_a, rf_rd_addr_b, rf_wr_addr;
  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] rf_rd_data_a, rf_rd_data_b, rf_wr_data;
  logic rf_wr_en;

  vec_regfile #(
      .NUM_REGS  (8),
      .ELEM_WIDTH(DATA_WIDTH),
      .NUM_ELEMS (NUM_LANES)
  ) regfile (
      .clk(clk),
      .rst_n(rst_n),
      .rd_addr_a(rf_rd_addr_a),
      .rd_data_a(rf_rd_data_a),
      .rd_addr_b(rf_rd_addr_b),
      .rd_data_b(rf_rd_data_b),
      .wr_en(rf_wr_en),
      .wr_addr(rf_wr_addr),
      .wr_data(rf_wr_data)
  );

  // 8 parallel ALU instances
  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] alu_result;

  generate
    for (genvar i = 0; i < NUM_LANES; i++) begin : alu_lanes
      logic [DATA_WIDTH-1:0] operand_b_lane;

      // Scalar broadcast: use element 0 for all lanes
      assign operand_b_lane = scalar_b ? rf_rd_data_b[0] : rf_rd_data_b[i];

      vpu_op #(
          .DATA_W(DATA_WIDTH),
          .OP_W  (VPU_OPCODE_WIDTH)
      ) alu (
          .operand0(rf_rd_data_a[i]),
          .operand1(operand_b_lane),
          .opcode(vpu_opcode),
          .result_out(alu_result[i])
      );
    end
  endgenerate

  // Saved instruction fields for multi-cycle operations.
  // Address fields are 13-bit from the decoder; only L1_ADDR_WIDTH bits are
  // forwarded to BRAM, upper bits are reserved for future expansion.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [12:0] saved_addr_a;
  logic [12:0] saved_addr_b;
  logic [12:0] saved_addr_out;
  /* verilator lint_on UNUSEDSIGNAL */
  /* verilator lint_off UNUSEDSIGNAL */
  // saved_vreg_a is latched for symmetry but not currently consumed.
  logic [2:0] saved_vreg_dst, saved_vreg_a;
  /* verilator lint_on UNUSEDSIGNAL */

  // Scalar path: operand latches and dedicated ALU instance
  logic [DATA_WIDTH-1:0] scalar_a_reg, scalar_b_reg;
  logic [DATA_WIDTH-1:0] scalar_result;
  logic [DATA_WIDTH-1:0] scalar_op_b;

  // During SCALAR_COMPUTE, use bram_dout directly as operand B so we avoid
  // an extra pipeline cycle.  At all other times scalar_b_reg is used (it
  // holds the last latched value and keeps the combinational logic stable).
  assign scalar_op_b = (state == SCALAR_COMPUTE) ? bram_dout[DATA_WIDTH-1:0] : scalar_b_reg;

  vpu_op #(
      .DATA_W(DATA_WIDTH),
      .OP_W  (3)
  ) scalar_alu (
      .operand0(scalar_a_reg),
      .operand1(scalar_op_b),
      .opcode(vpu_opcode),
      .result_out(scalar_result)
  );

  // FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      bram_en_simd <= 1'b0;
      bram_we_simd <= 1'b0;
      bram_addr_simd <= '0;
      bram_din_simd <= '0;
      rf_wr_en <= 1'b0;
      rf_wr_addr <= '0;
      rf_wr_data <= '0;
      rf_rd_addr_a <= '0;
      rf_rd_addr_b <= '0;
      done_simd <= 1'b0;
      saved_addr_a <= '0;
      saved_addr_b <= '0;
      saved_addr_out <= '0;
      saved_vreg_dst <= '0;
      saved_vreg_a <= '0;
      scalar_a_reg <= '0;
      scalar_b_reg <= '0;
    end else begin
      // Defaults
      done_simd <= 1'b0;
      bram_we_simd <= 1'b0;
      bram_en_simd <= 1'b0;
      rf_wr_en <= 1'b0;

      case (state)
        IDLE: begin
          if (start_simd) begin
            // Save instruction fields
            saved_addr_a   <= addr_a;
            saved_addr_b   <= addr_b;
            saved_addr_out <= addr_out;
            saved_vreg_dst <= vreg_dst;
            saved_vreg_a   <= vreg_a;

            case (vpu_type)
              VPU_TYPE_SCALAR: begin  // SCALAR
                state <= SCALAR_READ_A;
              end
              VPU_TYPE_VLOAD: begin  // VLOAD
                state <= VLOAD_REQ;
              end
              VPU_TYPE_VSTORE: begin  // VSTORE
                rf_rd_addr_a <= vreg_a;
                state <= VSTORE_REQ;
              end
              VPU_TYPE_VCOMPUTE: begin  // VCOMPUTE
                rf_rd_addr_a <= vreg_a;
                rf_rd_addr_b <= vreg_b;
                state <= VCOMPUTE_READ;
              end
              default: state <= DONE_STATE;
            endcase
          end
        end

        // VLOAD: Read 8 elements from BRAM concurrently
        VLOAD_REQ: begin
          bram_en_simd <= 1'b1;
          bram_we_simd <= 1'b0;
          bram_addr_simd <= saved_addr_a[L1_ADDR_WIDTH-1:0];
          state <= VLOAD_WAIT1;
        end

        VLOAD_WAIT1: begin
          bram_en_simd <= 1'b0;  // BRAM address/en already issued in REQ
          state <= VLOAD_CAPTURE;
        end

        VLOAD_CAPTURE: begin
          // Capture row and write to regfile
          rf_wr_en <= 1'b1;
          rf_wr_addr <= saved_vreg_dst;
          rf_wr_data <= bram_dout;
          state <= DONE_STATE;
        end

        // VSTORE: Write 8 elements to BRAM concurrently
        VSTORE_REQ: begin
          bram_en_simd <= 1'b1;
          bram_we_simd <= 1'b1;
          bram_addr_simd <= saved_addr_out[L1_ADDR_WIDTH-1:0];
          bram_din_simd <= rf_rd_data_a;  // 256 bits

          state <= DONE_STATE;
        end

        // VCOMPUTE: Wait one cycle for register read
        VCOMPUTE_READ: begin
          state <= VCOMPUTE_EXEC;
        end

        // VCOMPUTE: Parallel execution (1 cycle)
        VCOMPUTE_EXEC: begin
          rf_wr_en <= 1'b1;
          rf_wr_addr <= saved_vreg_dst;
          rf_wr_data <= alu_result;
          state <= DONE_STATE;
        end

        // SCALAR: issue read for operand A (addr presented; keep en high)
        SCALAR_READ_A: begin
          bram_en_simd   <= 1'b1;
          bram_we_simd   <= 1'b0;
          bram_addr_simd <= saved_addr_a[L1_ADDR_WIDTH-1:0];
          state          <= SCALAR_WAIT_A;
        end

        // SCALAR: extra wait — bram_dout for A becomes valid next cycle
        SCALAR_WAIT_A: begin
          bram_en_simd <= 1'b0;
          state        <= SCALAR_LATCH_A;
        end

        // SCALAR: bram_dout carries A; latch and issue read for B
        SCALAR_LATCH_A: begin
          scalar_a_reg   <= bram_dout[DATA_WIDTH-1:0];
          bram_en_simd   <= 1'b1;
          bram_we_simd   <= 1'b0;
          bram_addr_simd <= saved_addr_b[L1_ADDR_WIDTH-1:0];
          state          <= SCALAR_WAIT_B;
        end

        // SCALAR: extra wait — bram_dout for B becomes valid next cycle
        SCALAR_WAIT_B: begin
          bram_en_simd <= 1'b0;
          state        <= SCALAR_COMPUTE;
        end

        // SCALAR: bram_dout carries B (fed combinationally into scalar_alu);
        // write result to addr_out in the same cycle.
        SCALAR_COMPUTE: begin
          bram_en_simd   <= 1'b1;
          bram_we_simd   <= 1'b1;
          bram_addr_simd <= saved_addr_out[L1_ADDR_WIDTH-1:0];
          bram_din_simd  <= {{(L1_DATA_WIDTH - DATA_WIDTH) {1'b0}}, scalar_result};
          state          <= DONE_STATE;
        end

        DONE_STATE: begin
          done_simd <= 1'b1;
          bram_en_simd <= 1'b0;
          state <= IDLE;
        end
        default: begin
          state <= IDLE;
        end
      endcase
    end
  end

endmodule
