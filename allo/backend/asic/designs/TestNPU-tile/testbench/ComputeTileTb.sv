`timescale 1ns/1ps

`define CLK_PERIOD 20
`define ASSIGNMENT_DELAY 5
`define FINISH_TIME 5000000

`ifndef NUM_PROGRAM_WORDS
`define NUM_PROGRAM_WORDS 11
`endif

`ifndef NUM_L1_INIT_ROWS
`define NUM_L1_INIT_ROWS 16
`endif

`ifndef NUM_EXPECTED_ROWS
`define NUM_EXPECTED_ROWS 8
`endif

module ComputeTileTb;

  localparam VECTOR_WIDTH = 256;
  // Matches the post-route compute_tile wrapper (MiniNPU 0e901466).
  localparam IRAM_ADDR_WIDTH = 12;
  localparam L1_DATA_WIDTH = 256;
  localparam DM_ADDR_WIDTH = 27;
  localparam DM_DATA_WIDTH = 256;
  localparam TEST_ROWS = `NUM_PROGRAM_WORDS + `NUM_L1_INIT_ROWS + `NUM_EXPECTED_ROWS;
  localparam OUTPUT_BASE = 16;
  // Post-route SDF simulation is substantially slower in cycle count than RTL
  // around the SRAM-backed LSU/MXU sequence.  The previous 10k limit expired
  // just as IRAM returned the final HALT word, before the sequencer could
  // decode it and pulse done.  The independent 5 ms global watchdog below
  // still catches a genuinely stuck design.
  localparam TIMEOUT_CYCLES = 100000;

  reg clk;
  reg rst_n;

  reg start;
  wire done;
  wire illegal_op_o;

  reg instr_write_en;
  reg [IRAM_ADDR_WIDTH-1:0] iram_addr;
  reg [63:0] dma_iram_din;

  reg [15:0] base_addr;
  reg dma_wr_en;
  reg [L1_DATA_WIDTH-1:0] dma_wr_data;
  reg [15:0] dma_write_pointer;
  reg dma_rd_en;
  wire [L1_DATA_WIDTH-1:0] dma_rd_data;
  reg [15:0] dma_read_pointer;

  reg [IRAM_ADDR_WIDTH-1:0] pool_base;
  reg [31:0] program_id_csr;
  reg [127:0] kernel_arg_csr;

  wire [DM_ADDR_WIDTH-1:0] dm_addr;
  wire [DM_DATA_WIDTH-1:0] dm_din;
  reg [DM_DATA_WIDTH-1:0] dm_dout;
  wire dm_en;
  wire dm_we;
  reg dm_ack;

  wire tma_req;
  wire tma_dir;
  wire [15:0] tma_dm_base;
  wire [14:0] tma_mt_base;
  wire [15:0] tma_len;
  reg tma_done;

  wire [31:0] perf_cnt_cycles_o;
  wire [31:0] perf_cnt_instrs_o;
  wire [1:0] dma_slot_done_o;

  reg [VECTOR_WIDTH-1:0] test_vectors [0:TEST_ROWS-1];

  compute_tile compute_tile_inst (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .done(done),
      .illegal_op_o(illegal_op_o),
      .instr_write_en(instr_write_en),
      .iram_addr(iram_addr),
      .dma_iram_din(dma_iram_din),
      .base_addr(base_addr),
      .dma_wr_en(dma_wr_en),
      .dma_wr_data(dma_wr_data),
      .dma_write_pointer(dma_write_pointer),
      .dma_rd_en(dma_rd_en),
      .dma_rd_data(dma_rd_data),
      .dma_read_pointer(dma_read_pointer),
      .pool_base(pool_base),
      .program_id_csr(program_id_csr),
      .kernel_arg_csr(kernel_arg_csr),
      .dm_addr(dm_addr),
      .dm_din(dm_din),
      .dm_dout(dm_dout),
      .dm_en(dm_en),
      .dm_we(dm_we),
      .dm_ack(dm_ack),
      .tma_req(tma_req),
      .tma_dir(tma_dir),
      .tma_dm_base(tma_dm_base),
      .tma_mt_base(tma_mt_base),
      .tma_len(tma_len),
      .tma_done(tma_done),
      .perf_cnt_cycles_o(perf_cnt_cycles_o),
      .perf_cnt_instrs_o(perf_cnt_instrs_o),
      .dma_slot_done_o(dma_slot_done_o)
  );

  always #(`CLK_PERIOD/2) clk = ~clk;

  task reset_dut;
    begin
      rst_n = 1'b0;
      start = 1'b0;
      instr_write_en = 1'b0;
      iram_addr = '0;
      dma_iram_din = '0;
      base_addr = '0;
      dma_wr_en = 1'b0;
      dma_wr_data = '0;
      dma_write_pointer = '0;
      dma_rd_en = 1'b0;
      dma_read_pointer = '0;
      pool_base = `NUM_PROGRAM_WORDS;
      program_id_csr = 32'd0;
      kernel_arg_csr = 128'd0;
      dm_dout = '0;
      dm_ack = 1'b0;
      tma_done = 1'b0;
      repeat (6) @(posedge clk);
      rst_n <= #`ASSIGNMENT_DELAY 1'b1;
      repeat (4) @(posedge clk);
    end
  endtask

  task load_program;
    integer i;
    begin
      for (i = 0; i < `NUM_PROGRAM_WORDS; i = i + 1) begin
        instr_write_en <= #`ASSIGNMENT_DELAY 1'b1;
        iram_addr <= #`ASSIGNMENT_DELAY i;
        dma_iram_din <= #`ASSIGNMENT_DELAY test_vectors[i][63:0];
        @(posedge clk);
      end
      instr_write_en <= #`ASSIGNMENT_DELAY 1'b0;
      dma_iram_din <= #`ASSIGNMENT_DELAY 64'd0;
      @(posedge clk);
    end
  endtask

  task load_l1_rows;
    integer i;
    begin
      base_addr <= #`ASSIGNMENT_DELAY 16'd0;
      for (i = 0; i < `NUM_L1_INIT_ROWS; i = i + 1) begin
        dma_wr_en <= #`ASSIGNMENT_DELAY 1'b1;
        dma_write_pointer <= #`ASSIGNMENT_DELAY i;
        dma_wr_data <= #`ASSIGNMENT_DELAY test_vectors[`NUM_PROGRAM_WORDS + i];
        @(posedge clk);
      end
      dma_wr_en <= #`ASSIGNMENT_DELAY 1'b0;
      dma_wr_data <= #`ASSIGNMENT_DELAY {L1_DATA_WIDTH{1'b0}};
      @(posedge clk);
    end
  endtask

  task run_compute;
    integer cycle;
    reg saw_done;
    begin
      saw_done = 1'b0;
      start <= #`ASSIGNMENT_DELAY 1'b1;
      @(posedge clk);
      start <= #`ASSIGNMENT_DELAY 1'b0;

      for (cycle = 0; cycle < TIMEOUT_CYCLES && !saw_done; cycle = cycle + 1) begin
        @(posedge clk);
        #1;
        if (illegal_op_o) begin
          $error("compute_tile raised illegal_op_o");
          $finish(2);
        end
        if (done) begin
          $display("compute_tile done after %0d cycles (perf=%0d, instrs=%0d)",
                   cycle, perf_cnt_cycles_o, perf_cnt_instrs_o);
          saw_done = 1'b1;
        end
      end

      if (!saw_done) begin
        $error("compute_tile timed out waiting for done");
        $finish(2);
      end

      repeat (2) @(posedge clk);
    end
  endtask

  task check_outputs;
    integer i;
    reg [L1_DATA_WIDTH-1:0] expected;
    begin
      dma_rd_en <= #`ASSIGNMENT_DELAY 1'b1;
      for (i = 0; i < `NUM_EXPECTED_ROWS; i = i + 1) begin
        expected = test_vectors[`NUM_PROGRAM_WORDS + `NUM_L1_INIT_ROWS + i];
        dma_read_pointer <= #`ASSIGNMENT_DELAY (OUTPUT_BASE + i);
        @(posedge clk);
        #1;
        $display("row %0d got=%h expected=%h", OUTPUT_BASE + i, dma_rd_data, expected);
        assert (dma_rd_data === expected)
          else begin
            $error("Mismatch at output row %0d", OUTPUT_BASE + i);
            $finish(2);
          end
      end
      dma_rd_en <= #`ASSIGNMENT_DELAY 1'b0;
      dma_read_pointer <= #`ASSIGNMENT_DELAY 16'd0;
      @(posedge clk);
    end
  endtask

  initial begin
    $readmemh("inputs/test_vectors.txt", test_vectors);

    clk = 1'b0;
    reset_dut();
    load_program();
    load_l1_rows();
    run_compute();
    check_outputs();

    $display("ComputeTileTb PASS");
    $finish;
  end

  initial begin
    $vcdplusfile("dump.vcd");
    $vcdplusmemon();
    $vcdpluson(0, ComputeTileTb);
    #(`FINISH_TIME);
    $error("ComputeTileTb timed out");
    $finish(2);
  end

endmodule
