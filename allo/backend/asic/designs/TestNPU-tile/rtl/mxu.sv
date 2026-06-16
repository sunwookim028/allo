// -----------------------------------------------------------------------------
// Matrix Unit (MXU) - Systolic Array Wrapper
// -----------------------------------------------------------------------------
//
// Parameters:
//  - N: The size of the systolic array (N×N PEs). Imported from npu_config_pkg.
//  - DATA_WIDTH: The size of each matrix element in bits.
//  - L1_ADDR_WIDTH: The size in bits of each memory address.
//  - MEM_LATENCY: The response latency of memory. Assume a fixed latency.
//  - L1_DATA_WIDTH: Compute bus width (8 × DATA_WIDTH = 256 bits)
//
// Memory layout (L1 row addresses, each row is L1_DATA_WIDTH wide):
//   base_addr_w + [0..N-1]   : weight rows (fp16, N elements per row)
//   base_addr_x + [0..N-1]   : activation rows (fp16, N elements per row)
//   base_addr_out + [0..N-1] : output rows (fp32, N elements per row)
// -----------------------------------------------------------------------------

/* verilator lint_off WIDTHEXPAND */
module mxu #(
    parameter N             = 8,
    parameter DATA_WIDTH    = 32,
    parameter L1_ADDR_WIDTH = 10,
    parameter MEM_LATENCY   = 2,
    parameter L1_DATA_WIDTH = 256,
    parameter PE_LATENCY    = 0
) (
    // --- Control Signals (Standard) ---
    input logic clk,   // Clock signal
    input logic rst_n, // Reset signal (active low)

    // --- To/from control ---
    input  logic start,  // Pulse high to begin mxu operation
    output logic done,   // Pulsed high by mxu when storing is complete

    // ISA flags (latched when start is driven high):
    input logic accumulate, // flags[1]: add to acc_dst instead of overwriting
    input logic causal,     // flags[2]: mask upper triangle with -INF after multiply

    // Base addresses (latched when start is driven high):
    input logic [L1_ADDR_WIDTH-1:0] base_addr_w,   // Base address of weight matrix
    input logic [L1_ADDR_WIDTH-1:0] base_addr_x,   // Base address of x matrix
    input logic [L1_ADDR_WIDTH-1:0] base_addr_out, // Base address to store output matrix

    // --- To/from memory ---
    output logic [L1_ADDR_WIDTH-1:0] mem_req_addr,
    output logic [L1_DATA_WIDTH-1:0] mem_req_data,
    input logic [L1_DATA_WIDTH-1:0] mem_resp_data,
    output logic mem_read_en,
    output logic mem_write_en,
    input logic i4_mode,
    input logic [7:0] i4_zero_point,
    input logic [N-1:0][31:0] scale_vec,  // fp32 per-column scales (dq_mode only)
    input logic dq_mode  // 1 = apply per-column scale after dequant
);
  localparam BANKING_FACTOR = L1_DATA_WIDTH / DATA_WIDTH;
  localparam FP16_BITS = 16;

`ifndef SYNTHESIS
  // Integer-to-FP16 case table for int4 dequant.  Both val and zp are in
  // [0, 15], so diff = val - zp ∈ [-15, 15] and abs_diff ∈ [0, 15].
  // Avoids $itor/$realtobits which are unsupported in Verilator functions.
  function automatic logic [15:0] dequant_to_fp16(input logic [7:0] val, input logic [7:0] zp);
    logic signed [ 8:0] diff;
    logic               sign_bit;
    logic        [ 7:0] abs_diff;
    logic        [14:0] abs_fp16_body;
    begin
      diff     = {1'b0, val} - {1'b0, zp};
      sign_bit = diff[8];
      abs_diff = sign_bit ? (8'd0 - diff[7:0]) : diff[7:0];
      case (abs_diff[3:0])
        4'd0: abs_fp16_body = 15'h0000;  // 0.0
        4'd1: abs_fp16_body = 15'h3C00;  // 1.0
        4'd2: abs_fp16_body = 15'h4000;  // 2.0
        4'd3: abs_fp16_body = 15'h4200;  // 3.0
        4'd4: abs_fp16_body = 15'h4400;  // 4.0
        4'd5: abs_fp16_body = 15'h4500;  // 5.0
        4'd6: abs_fp16_body = 15'h4600;  // 6.0
        4'd7: abs_fp16_body = 15'h4700;  // 7.0
        4'd8: abs_fp16_body = 15'h4800;  // 8.0
        4'd9: abs_fp16_body = 15'h4880;  // 9.0
        4'd10: abs_fp16_body = 15'h4900;  // 10.0
        4'd11: abs_fp16_body = 15'h4980;  // 11.0
        4'd12: abs_fp16_body = 15'h4A00;  // 12.0
        4'd13: abs_fp16_body = 15'h4A80;  // 13.0
        4'd14: abs_fp16_body = 15'h4B00;  // 14.0
        4'd15: abs_fp16_body = 15'h4B80;  // 15.0
        default: abs_fp16_body = 15'h0000;
      endcase
      dequant_to_fp16 = {sign_bit, abs_fp16_body};
    end
  endfunction

  // Dequant nibble with per-column zero-point and scale → fp16.
  // val_u8          : nibble value [0..15]
  // zero_u8         : zero-point scalar [0..15]
  // scale_f32_bits  : IEEE-754 fp32 scale factor (per column)
  // Behavioral only: uses real arithmetic + manual fp32 decode/encode.
  // Avoids $bitstoshortreal/$shortrealtobits (Verilator promotes shortreal → real,
  // corrupting 32-bit IEEE-754 layout).  Uses $bitstoreal with a 32-bit mask instead.
  // TODO: replace with synthesisable model for FPGA target.
  function automatic logic [15:0] dequant_scale_to_fp16(
      input logic [7:0] val_u8, input logic [7:0] zero_u8, input logic [31:0] scale_f32_bits);
    real scale_r, result_r;
    logic               s_sign;
    logic        [ 7:0] s_exp8;
    logic        [22:0] s_mant23;
    logic signed [ 8:0] diff_i;
    logic        [31:0] res_bits;
    logic               sign_b;
    logic        [ 7:0] exp32;
    logic        [22:0] mant32;
    logic        [ 4:0] exp16;
    logic        [ 9:0] mant16;
    integer             exp_unbiased;
    logic        [10:0] exp64;
    logic        [10:0] dexp;
    logic        [63:0] double_bits;
    begin
      // --- Decode scale_f32_bits to real (double) via manual fp32→double conversion ---
      s_sign   = scale_f32_bits[31];
      s_exp8   = scale_f32_bits[30:23];
      s_mant23 = scale_f32_bits[22:0];
      if (s_exp8 == 8'hFF) begin
        // NaN or Inf — treat as 0 scale
        scale_r = 0.0;
      end else if (s_exp8 == 8'd0) begin
        // Denormal or zero — close enough to 0 for our range
        scale_r = 0.0;
      end else begin
        // Normal: value = (-1)^sign * 2^(exp-127) * (1 + mant/2^23)
        // Build IEEE-754 double: sign | exp+896 (rebias 127→1023) | mant<<29
        exp_unbiased = (s_exp8) - 127;
        /* verilator lint_off WIDTHTRUNC */
        exp64 = (exp_unbiased + 1023);
        /* verilator lint_on WIDTHTRUNC */
        double_bits = {s_sign, (exp64), {s_mant23, 29'b0}};
        scale_r = $bitstoreal(double_bits);
      end

      // --- Compute diff * scale as real ---
      diff_i = $signed({1'b0, val_u8}) - $signed({1'b0, zero_u8});
      result_r = (diff_i) * scale_r;

      // --- Encode result (real/double) back to fp32 bits via double→fp32 ---
      double_bits = $realtobits(result_r);
      sign_b = double_bits[63];
      // double exp: bits [62:52] biased by 1023; fp32 exp biased by 127 → diff = 896
      dexp = double_bits[62:52];
      if (dexp == 11'h7FF) begin  // NaN/Inf
        exp32  = 8'hFF;
        mant32 = double_bits[51:29];
      end else if (dexp == 11'd0 || dexp < 11'd897) begin  // zero or underflow to fp32 denorm
        exp32  = 8'd0;
        mant32 = 23'd0;
      end else if (dexp > 11'd1150) begin  // overflow → Inf in fp32
        exp32  = 8'hFF;
        mant32 = 23'd0;
      end else begin
        exp32  = (dexp - 11'd896);  // rebias: 1023-127=896
        mant32 = double_bits[51:29];  // top 23 mantissa bits
      end
      res_bits = {sign_b, exp32, mant32};

      // --- Encode fp32 result to fp16 ---
      sign_b   = res_bits[31];
      exp32    = res_bits[30:23];
      mant32   = res_bits[22:0];
      if (exp32 == 8'hFF) begin  // NaN/Inf
        exp16  = 5'h1F;
        mant16 = mant32[22:13] | {9'b0, |mant32};
      end else if (exp32 == 8'd0 || exp32 < 8'd103) begin  // zero or underflow
        exp16  = 5'd0;
        mant16 = 10'd0;
      end else if (exp32 > 8'd142) begin  // overflow → Inf
        exp16  = 5'h1F;
        mant16 = 10'd0;
      end else begin
        exp16  = (exp32 - 8'd112);  // rebias: exp32_unbiased + 15
        mant16 = mant32[22:13];  // top 10 mantissa bits (truncate)
      end
      dequant_scale_to_fp16 = {sign_b, exp16, mant16};
    end
  endfunction
`endif
  // Total extra cycles added by the PE arithmetic pipeline: each of fp32_mul
  // and fp32_add contributes PE_LATENCY stages, plus 1 for mult_out_reg (Fix 10).
  localparam PE_LAT = 2 * PE_LATENCY + 1;

  // Output, accumulate, and weight/x buffers
  logic signed [DATA_WIDTH-1:0] out_matrix[N*N-1:0];
  logic [15:0] weight_matrix[N*N-1:0];
  logic [15:0] x_matrix[N*N-1:0];
  logic [DATA_WIDTH-1:0] acc_buf[N*N-1:0];

  // Combinational accumulate sum: acc_buf[i] + out_matrix[i]
  logic [DATA_WIDTH-1:0] acc_sum[N*N-1:0];
  generate
    for (genvar gi = 0; gi < N * N; gi++) begin : ACC_ADD
      fp32_add #(
          .LATENCY(0)
      ) u_acc_add (
          .clk   (clk),
          .rst_n (rst_n),
          .a     (acc_buf[gi]),
          .b     (out_matrix[gi]),
          .result(acc_sum[gi])
      );
    end
  endgenerate

  // Registered flag inputs (latched on start pulse)
  logic accumulate_reg;
  logic causal_reg;
`ifndef SYNTHESIS
  logic               i4_mode_reg;
  logic [  7:0]       i4_zero_point_reg;
  logic [N-1:0][31:0] scale_vec_reg;
  logic               dq_mode_reg;
`endif

  // Registers to latch matrix addresses
  logic [L1_ADDR_WIDTH-1:0] base_addr_x_reg, base_addr_w_reg, base_addr_out_reg;

  // Index for load/store progress (indexes loading rows)
  logic [5:0] load_idx;
  wire [L1_ADDR_WIDTH-1:0] load_idx_addr = {{(L1_ADDR_WIDTH - $bits(load_idx)) {1'b0}}, load_idx};

  // Timer for memory fixed latency
  logic [$clog2(MEM_LATENCY+1)-1:0] mem_latency_timer;

  // Systolic array interface — packed arrays (iverilog requires packed for
  // module-to-module port connections; unpacked array ports are silently broken)
  logic [N-1:0][15:0] sys_weight_in;
  logic [N-1:0][15:0] sys_data_in;
  logic [N-1:0] sys_accept_w;
  logic [N-1:0] sys_start;
  logic [N-1:0][DATA_WIDTH-1:0] sys_data_out;
  logic [N-1:0] sys_valid_out;
  logic [N-1:0] sys_valid_out_d;  // Fix 10: pe_psum_out is 1 cycle behind pe_valid_out; delay valid for capture
  logic sys_switch_in;

  systolic array (
      .clk          (clk),
      .rst_n        (rst_n),
      .sys_data_in  (sys_data_in),
      .sys_start    (sys_start),
      .sys_data_out (sys_data_out),
      .sys_valid_out(sys_valid_out),
      .sys_weight_in(sys_weight_in),
      .sys_accept_w (sys_accept_w),
      .sys_switch_in(sys_switch_in)
  );

  // -------------------------------------------------------------------------
  // Finite State Machine (FSM)
  // -------------------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE,
    S_LOAD_ACC,
    S_LOAD_ACC_WAIT,
    S_LOAD_W_REQ,
    S_LOAD_W_WAIT,
    S_LOAD_X_REQ,
    S_LOAD_X_WAIT,
    S_RUN,
    S_CAPTURE,
    S_CAUSAL,
    S_STORE_REQ,
    S_STORE_WAIT,
    S_DONE
  } state_t;

  state_t state;
  // phase_counter tracks progress through S_RUN and S_CAPTURE.
  // Width covers the maximum count: 5*N + PE_LAT (S_RUN) or 4*N + PE_LAT (S_CAPTURE).
  logic [$clog2(8*N + PE_LAT + 1):0] phase_counter;
  // row_ptr[c]: how many output elements column c has captured into out_matrix.
  logic [3:0] row_ptr[N];

  // -------------------------------------------------------------------------
  // Fix 10: sys_valid_out is 1 cycle ahead of sys_data_out (pe_psum_out has
  // 2-cycle latency from pe_valid_in; pe_valid_out has 1-cycle latency).
  // Delay sys_valid_out by 1 cycle so captures fire when data is valid.
  // -------------------------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) sys_valid_out_d <= '0;
    else sys_valid_out_d <= sys_valid_out;
  end

  // -------------------------------------------------------------------------
  // Output capture + acc_buf management
  // -------------------------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < N; i++) row_ptr[i] <= 0;
      for (int r = 0; r < N; r++) for (int c = 0; c < N; c++) out_matrix[(r*N+c)] <= '0;
      for (int i = 0; i < N * N; i++) acc_buf[i] <= '0;
    end else begin
      if (start) begin
        // Clear row_ptr, out_matrix, and acc_buf on each new computation.
        // acc_buf will be overwritten in S_LOAD_ACC_WAIT if accumulate_reg=1.
        for (int i = 0; i < N; i++) row_ptr[i] <= 0;
        for (int r = 0; r < N; r++) for (int c = 0; c < N; c++) out_matrix[(r*N+c)] <= '0;
        for (int i = 0; i < N * N; i++) acc_buf[i] <= '0;
      end else if (state == S_LOAD_ACC_WAIT && mem_latency_timer >= (MEM_LATENCY - 1) && accumulate_reg) begin
        // Load one row of acc_buf from SPM (fp32, packed BANKING per 256-bit word)
        for (int i = 0; i < N; i++) begin
          acc_buf[load_idx*N+i] <= mem_resp_data[i*DATA_WIDTH+:DATA_WIDTH];
        end
      end else if (state == S_CAUSAL) begin
        // Apply causal mask (-INF) to upper triangle, then fold in accumulate
        for (int r = 0; r < N; r++) begin
          for (int c = 0; c < N; c++) begin
            if (causal_reg && c > r) begin
              out_matrix[r*N+c] <= 32'hFF800000;
            end else begin
              out_matrix[r*N+c] <= acc_sum[r*N+c];
            end
          end
        end
      end else begin
        // Capture systolic outputs; use sys_valid_out_d (1-cycle delayed) so
        // sys_data_out has settled (pe_psum_out is 1 cycle behind pe_valid_out).
        for (int col = 0; col < N; col++) begin
          if (sys_valid_out_d[col] && row_ptr[col] < N) begin
            out_matrix[row_ptr[col]*N+col] <= sys_data_out[col];
            row_ptr[col] <= row_ptr[col] + 1;
          end
        end
      end
    end
  end

  // -------------------------------------------------------------------------
  // FSM Logic
  // -------------------------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state             <= S_IDLE;
      phase_counter     <= '0;
      mem_req_addr      <= '0;
      mem_req_data      <= '0;
      mem_read_en       <= 0;
      mem_write_en      <= 0;
      mem_latency_timer <= '0;
      load_idx          <= 0;
      done              <= 0;
      base_addr_x_reg   <= '0;
      base_addr_w_reg   <= '0;
      base_addr_out_reg <= '0;
      accumulate_reg    <= 1'b0;
      causal_reg        <= 1'b0;
`ifndef SYNTHESIS
      i4_mode_reg       <= 1'b0;
      i4_zero_point_reg <= 8'd0;
      scale_vec_reg     <= '0;
      dq_mode_reg       <= 1'b0;
`endif
      for (int i = 0; i < N * N; i++) begin
        weight_matrix[i] <= '0;
        x_matrix[i] <= '0;
      end
      for (int i = 0; i < N; i++) begin
        sys_accept_w[i] <= 0;
        sys_start[i]    <= 0;
        sys_weight_in[i] <= '0;
        sys_data_in[i]   <= '0;
      end
      sys_switch_in <= 0;
    end else begin
      // Defaults
      done <= 0;
      for (int i = 0; i < N; i++) begin
        sys_accept_w[i] <= 0;
        sys_start[i]    <= 0;
        sys_weight_in[i] <= '0;
        sys_data_in[i]   <= '0;
      end
      sys_switch_in <= 0;
      mem_req_addr  <= '0;
      mem_req_data  <= '0;
      mem_read_en   <= 0;
      mem_write_en  <= 0;

      case (state)
        S_IDLE: begin
          if (start) begin
            base_addr_x_reg   <= base_addr_x;
            base_addr_w_reg   <= base_addr_w;
            base_addr_out_reg <= base_addr_out;
            accumulate_reg    <= accumulate;
            causal_reg        <= causal;
`ifndef SYNTHESIS
            i4_mode_reg       <= i4_mode;
            i4_zero_point_reg <= i4_zero_point;
            scale_vec_reg     <= scale_vec;
            dq_mode_reg       <= dq_mode;
`endif
            state <= S_LOAD_ACC;
          end
        end

        S_LOAD_ACC: begin
          if (accumulate_reg) begin
            mem_req_addr <= base_addr_out_reg + load_idx_addr;
            mem_read_en <= 1;
            mem_latency_timer <= '0;
            state <= S_LOAD_ACC_WAIT;
          end else begin
            state <= S_LOAD_W_REQ;
          end
        end

        S_LOAD_ACC_WAIT: begin
          if (mem_latency_timer >= (MEM_LATENCY - 1)) begin
            if (load_idx >= N - 1) begin
              load_idx <= '0;
              state <= S_LOAD_W_REQ;
            end else begin
              load_idx <= load_idx + 1;
              state <= S_LOAD_ACC;
            end
          end else begin
            mem_latency_timer <= mem_latency_timer + 1;
          end
        end

        S_LOAD_W_REQ: begin
          mem_req_addr <= base_addr_w_reg + load_idx_addr;
          mem_read_en <= 1;
          mem_latency_timer <= '0;
          state <= S_LOAD_W_WAIT;
        end

        S_LOAD_W_WAIT: begin
          if (mem_latency_timer >= (MEM_LATENCY - 1)) begin
            // Load full row of W: fp16 normally; dequant uint8 nibble in i4 mode.
            for (int i = 0; i < N; i++) begin
`ifndef SYNTHESIS
              if (dq_mode_reg) begin
                // SV sim quirk: calling a real-arithmetic function directly in an NBA
                // expression can silently yield stale/zero values in some simulators.
                // Use an explicit blocking intermediate to force eager evaluation.
                logic [15:0] tmp_dq;
                tmp_dq = dequant_scale_to_fp16(mem_resp_data[i*8+:8], i4_zero_point_reg,
                                               scale_vec_reg[i]);
                weight_matrix[load_idx*N+i] <= tmp_dq;
              end else if (i4_mode_reg)
                weight_matrix[load_idx*N+i] <= dequant_to_fp16(
                    mem_resp_data[i*8+:8], i4_zero_point_reg
                );
              else weight_matrix[load_idx*N+i] <= mem_resp_data[i*FP16_BITS+:FP16_BITS];
`else
              weight_matrix[load_idx*N+i] <= mem_resp_data[i*FP16_BITS+:FP16_BITS];
`endif
            end
            if (load_idx >= N - 1) begin
              load_idx <= '0;
              state <= S_LOAD_X_REQ;
            end else begin
              load_idx <= load_idx + 1;
              state <= S_LOAD_W_REQ;
            end
          end else begin
            mem_latency_timer <= mem_latency_timer + 1;
          end
        end

        S_LOAD_X_REQ: begin
          mem_req_addr <= base_addr_x_reg + load_idx_addr;
          mem_read_en <= 1;
          mem_latency_timer <= '0;
          state <= S_LOAD_X_WAIT;
        end

        S_LOAD_X_WAIT: begin
          if (mem_latency_timer >= (MEM_LATENCY - 1)) begin
            // Load full row of X: BRAM[addr] contains X[load_idx][0:N-1] as fp16
            for (int i = 0; i < N; i++) begin
              x_matrix[load_idx*N+i] <= mem_resp_data[i*FP16_BITS+:FP16_BITS];
            end
            if (load_idx >= N - 1) begin
              load_idx <= '0;
              phase_counter <= '0;
              state <= S_RUN;
            end else begin
              load_idx <= load_idx + 1;
              state <= S_LOAD_X_REQ;
            end
          end else begin
            mem_latency_timer <= mem_latency_timer + 1;
          end
        end

        S_RUN: begin
          phase_counter <= phase_counter + 1;

          // ---- Weight pipeline ----
          // Column k: load N weights over phases k..k+N-1, in reverse order
          // (weight[k][N-1] first so PE[0][k] ends up holding weight[k][0]).
          // Switch fires one cycle after the last column finishes loading,
          // at phase 2*N-1.
          for (int col = 0; col < N; col++) begin
            if ((phase_counter) >= col && (phase_counter) < N + col) begin
              sys_weight_in[col] <= weight_matrix[col*N+(N-1-((phase_counter)-col))];
              sys_accept_w[col]  <= 1;
            end
          end

          if (phase_counter == 2 * N - 1) sys_switch_in <= 1;

          // ---- X input stream ----
          // Row r of the systolic array receives column r of X.
          // Row r starts at phase 2*N + r (one cycle after switch, staggered by row).
          for (int row = 0; row < N; row++) begin
            if ((phase_counter) >= 2 * N + row && (phase_counter) < 3 * N + row) begin
              sys_start[row]   <= 1;
              sys_data_in[row] <= x_matrix[((phase_counter)-(2*N+row))*N+row];
            end
          end

          // Stop after 5*N + PE_LAT phases: PE pipeline shifts valid by PE_LAT.
          if (phase_counter >= 5 * N + PE_LAT) begin
            phase_counter <= '0;
            state <= S_CAPTURE;
          end
        end

        S_CAPTURE: begin
          bit all_done;
          all_done = 1;
          for (int i = 0; i < N; i++) all_done &= (row_ptr[i] == N);

          phase_counter <= phase_counter + 1;

          // Wait until all N columns have N captured outputs, or watchdog expires.
          // Watchdog: 4*N + PE_LAT accounts for shifted valid timing.
          if (all_done || phase_counter >= 4 * N + PE_LAT) begin
            phase_counter <= 0;
            state <= S_CAUSAL;
          end
        end

        S_CAUSAL: begin
          state <= S_STORE_REQ;
        end

        S_STORE_REQ: begin
          mem_req_addr <= base_addr_out_reg + load_idx_addr;
          // Write out_matrix[load_idx][0:N-1] packed into the 256-bit row
          for (int i = 0; i < N; i++) begin
            mem_req_data[i*DATA_WIDTH+:DATA_WIDTH] <= out_matrix[load_idx*N+i];
          end
          // Zero-fill unused upper slots (when N < BANKING_FACTOR)
          for (int i = N; i < BANKING_FACTOR; i++) begin
            mem_req_data[i*DATA_WIDTH+:DATA_WIDTH] <= '0;
          end
          mem_write_en <= 1;
          mem_latency_timer <= '0;
          state <= S_STORE_WAIT;
        end

        S_STORE_WAIT: begin
          if (mem_latency_timer >= (MEM_LATENCY - 1)) begin
            if (load_idx >= N - 1) begin
              state <= S_DONE;
            end else begin
              load_idx <= load_idx + 1;
              state <= S_STORE_REQ;
            end
          end else begin
            mem_latency_timer <= mem_latency_timer + 1;
          end
        end

        S_DONE: begin
          done <= 1;
          load_idx <= 0;
          state <= S_IDLE;
        end
        default: begin
          done <= 0;
          mem_read_en <= 0;
          mem_write_en <= 0;
          state <= S_IDLE;
        end

      endcase
    end
  end

  // Debug: expose out_matrix elements for waveform probing.
  // Consumed by dv/compute_tile/test_mxu.py (per-element probes).
  /* verilator lint_off UNUSEDSIGNAL */
  generate
    for (genvar i = 0; i < N * N; i++) begin : OUT_DEBUG
      logic signed [DATA_WIDTH-1:0] out_elem;
      assign out_elem = out_matrix[i];
    end
  endgenerate
  /* verilator lint_on UNUSEDSIGNAL */

endmodule
/* verilator lint_on WIDTHEXPAND */
