// Copyright MiniTPU Authors
// SPDX-License-Identifier: Apache-2.0
`timescale 1ns/1ps
// A copy of tb_mxu_single_port.sv (MiniTPU b3ba0a4d) that (1) takes DIM from `MRTL_DIM (default 2; identity
// weights at any DIM), (2) measures the result latency (last LHS push -> output_valid), the pop-recovery window
// (pop -> output_valid low) and a second tile, (3) prints the wrapper's token lag. The scenario of the
// original is kept: RHS rows bottom-up, commit 3 cycles after the last, NUM_SUBLANES LHS rows, wait, check, pop.
// minitpu_fpga_mxu_2026-10-08: copied from minitpu_rtl_mxu_2026-10-08/tb/ with one change, `MRTL_MARGIN (default
// the original's 6 * DIM + 16): the wait before the first pop, so that tile B's result has landed when the pop
// interval is measured on a core whose push->valid exceeds the original margin.
module tb_mxu_single_port_lat;
  import vpu_pkg::*;
`ifndef MRTL_DIM
  `define MRTL_DIM 2
`endif
  localparam int DIM = `MRTL_DIM;
  logic clk = 0;
  logic rst_n = 0;
  logic input_push, input_ready, input_accept, weight_commit;
  mxu_input_kind_e input_kind;
  logic [DIM-1:0][DATA_WIDTH-1:0] input_data;
  logic output_pop, output_valid;
  logic [NUM_SUBLANES-1:0][DIM-1:0][DATA_WIDTH-1:0] output_data;
  always #5 clk = ~clk;
  int unsigned cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  mxu #(.DIM(DIM)) dut (
    .clk_i(clk), .rst_ni(rst_n),
    .input_push_i(input_push), .input_kind_i(input_kind),
    .input_data_i(input_data), .input_ready_o(input_ready),
    .input_accept_o(input_accept),
    .weight_commit_i(weight_commit),
    .output_pop_i(output_pop), .output_valid_o(output_valid),
    .output_data_o(output_data)
  );

  task automatic push_row(input mxu_input_kind_e kind, input logic [DIM-1:0][15:0] row);
    begin
      @(negedge clk);
      if (!input_ready) $fatal(1, "MXU input unexpectedly full");
      input_kind = kind;
      input_data = row;
      input_push = 1;
      @(negedge clk);
      input_push = 0;
    end
  endtask

  int unsigned last_push_cyc, valid_cyc, pop_cyc, next_cyc, low_cyc, errors = 0;
  logic [DIM-1:0][15:0] row;

  function automatic logic [15:0] expect_val(input int unsigned base, input int unsigned r, input int unsigned c);
    return 16'(base + 16'h40 * c + r);
  endfunction

  task automatic push_tile(input int unsigned base);
    begin
      for (int unsigned r = 0; r < NUM_SUBLANES; r++) begin
        for (int unsigned c = 0; c < DIM; c++) row[c] = expect_val(base, r, c);
        push_row(MXU_INPUT_LHS, row);
      end
      last_push_cyc = cyc;
    end
  endtask

  task automatic check_tile(input int unsigned base, input string tag);
    begin
      for (int unsigned r = 0; r < NUM_SUBLANES; r++)
        for (int unsigned c = 0; c < DIM; c++)
          if (output_data[r][c] != expect_val(base, r, c)) begin
            errors++;
            $display("FAIL %s sublane %0d lane %0d got %04x expected %04x", tag, r, c, output_data[r][c], expect_val(base, r, c));
          end
      if (errors == 0) $display("CHECK %s tile data exact", tag);
    end
  endtask

  task automatic lag_report(input string tag);
    begin
`ifdef MRTL_ALLO
      $display("MEASURE %s token_lag %0d acc_lag %0d", tag, dut.lag_o, dut.acc_lag_o);
`endif
    end
  endtask

  initial begin
    input_push = 0; input_kind = MXU_INPUT_LHS; input_data = '0;
    weight_commit = 0; output_pop = 0;
    repeat (2) @(posedge clk);
    rst_n <= 1;
    // identity weights, bottom row first: row k of the tile is e_{DIM-1-k}
    for (int unsigned k = 0; k < DIM; k++) begin
      for (int unsigned c = 0; c < DIM; c++) row[c] = (c == DIM - 1 - k) ? 16'h3f80 : 16'h0000;
      push_row(MXU_INPUT_RHS, row);
    end
    repeat (3) @(negedge clk);
    weight_commit = 1;
    @(negedge clk); weight_commit = 0;
    // tile A: push -> valid
    push_tile(16'h4040);
    wait (output_valid);
    @(negedge clk);
    valid_cyc = cyc;
    $display("MEASURE push_to_valid %0d (last push cycle %0d, valid cycle %0d)", valid_cyc - last_push_cyc, last_push_cyc, valid_cyc);
    lag_report("A");
    check_tile(16'h4040, "A");
    // tile B queued behind A, then pop A: pop -> the next group valid (the pop interval)
    push_tile(16'h4140);
    // let B's result land before the pop (its own push->valid plus margin), so pop -> next valid is the pop interval
`ifndef MRTL_MARGIN
  `define MRTL_MARGIN (6 * DIM + 16)
`endif
    repeat (`MRTL_MARGIN) @(negedge clk);
    if (!output_valid) begin errors++; $display("FAIL valid dropped while A was unpopped"); end
    output_pop = 1;
    @(negedge clk); output_pop = 0;
    pop_cyc = cyc;
    while (!output_valid) @(negedge clk);
    next_cyc = cyc;
    $display("MEASURE pop_to_next_valid %0d (pop cycle %0d, next valid cycle %0d)", next_cyc - pop_cyc, pop_cyc, next_cyc);
    lag_report("B");
    check_tile(16'h4140, "B");
    output_pop = 1;
    @(negedge clk); output_pop = 0;
    pop_cyc = cyc;
    while (output_valid) @(negedge clk);
    low_cyc = cyc;
    $display("MEASURE pop_to_valid_low %0d (pop cycle %0d, low cycle %0d)", low_cyc - pop_cyc, pop_cyc, low_cyc);
    repeat (64) @(negedge clk);
    if (output_valid) begin errors++; $display("FAIL valid rose with nothing queued"); end
    // the token rate: lag growth over idle cycles (a lag that grows is a core below one token per cycle)
    lag_report("idle+0");
    repeat (1000) @(negedge clk);
    lag_report("idle+1000");
    repeat (1000) @(negedge clk);
    lag_report("idle+2000");
`ifdef MRTL_ALLO
    if (dut.lockstep_violation_o) begin errors++; $display("FAIL lockstep violation flagged by the wrapper"); end
`endif
    if (errors) $fatal(1, "FAIL tb_mxu_single_port_lat DIM=%0d errors=%0d", DIM, errors);
    $display("PASS tb_mxu_single_port_lat DIM=%0d", DIM);
    $finish;
  end
endmodule
