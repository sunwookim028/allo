`timescale 1ns/1ps

module allo_generated_testbench;
@SIGNAL_DECLARATIONS@
  real allo_bagl_clock_compensation_ns;
  real allo_bagl_input_delay_ns;
  real allo_bagl_output_delay_ns;
  integer allo_bagl_num_reset_cycles;

  initial begin
    if (!$value$plusargs("ALLO_BAGL_CLK_INS_SRC_LAT_NS=%f", allo_bagl_clock_compensation_ns))
      allo_bagl_clock_compensation_ns = 0.0;
    if (!$value$plusargs("ALLO_BAGL_INPUT_DELAY_NS=%f", allo_bagl_input_delay_ns))
      allo_bagl_input_delay_ns = 0.0;
    if (!$value$plusargs("ALLO_BAGL_OUTPUT_DELAY_NS=%f", allo_bagl_output_delay_ns))
      allo_bagl_output_delay_ns = 0.0;
    if (!$value$plusargs("ALLO_BAGL_NUM_RESET_CYCLES=%d", allo_bagl_num_reset_cycles))
      allo_bagl_num_reset_cycles = 8;
  end

  initial ap_clk = 1'b0;
  always #(@CLOCK_HALF_PERIOD@) ap_clk = ~ap_clk;

  initial begin
    if ($test$plusargs("ALLO_DUMP_VCD")) begin
      $dumpfile("outputs/run.vcd");
      $dumpvars(0, allo_generated_testbench);
    end
  end

  @TOP_MODULE@ dut (
@DUT_CONNECTIONS@
  );

@BFM_INSTANTIATIONS@

  task automatic wait_for_done(input integer timeout_cycles);
    integer elapsed;
    begin
      elapsed = 0;
      while (ap_done !== 1'b1 && elapsed < timeout_cycles) begin
        @(posedge ap_clk);
        #(allo_bagl_clock_compensation_ns + allo_bagl_output_delay_ns);
        if ($isunknown(ap_done)) begin
          $fatal(1, "ap_done became unknown after %0d cycles", elapsed);
        end
        elapsed = elapsed + 1;
      end
      if (elapsed >= timeout_cycles)
        $fatal(1, "timeout waiting for ap_done after %0d cycles", timeout_cycles);
    end
  endtask

  initial begin
    ap_rst_n = 1'b0;
    ap_start = 1'b0;
    repeat (2) @(negedge ap_clk);
@WORKLOAD_SEQUENCE@
    $display("ALLO_TEST_PASS");
    $finish;
  end
endmodule
