`timescale 1ns/1ps

module allo_generated_testbench;
@SIGNAL_DECLARATIONS@
  real allo_bagl_clock_compensation_ns;
  real allo_bagl_input_delay_ns;
  real allo_bagl_output_delay_ns;
  integer allo_bagl_num_reset_cycles;
  logic clear_completion;
  logic [@OUTPUT_COUNT@-1:0] completion_seen;

  initial begin
    if (!$value$plusargs("ALLO_BAGL_CLK_INS_SRC_LAT_NS=%f", allo_bagl_clock_compensation_ns))
      allo_bagl_clock_compensation_ns = 0.0;
    if (!$value$plusargs("ALLO_BAGL_INPUT_DELAY_NS=%f", allo_bagl_input_delay_ns))
      allo_bagl_input_delay_ns = 0.0;
    if (!$value$plusargs("ALLO_BAGL_OUTPUT_DELAY_NS=%f", allo_bagl_output_delay_ns))
      allo_bagl_output_delay_ns = 0.0;
    if (!$value$plusargs("ALLO_BAGL_NUM_RESET_CYCLES=%d", allo_bagl_num_reset_cycles))
      allo_bagl_num_reset_cycles = @DEFAULT_RESET_CYCLES@;
  end

  initial @CLOCK@ = 1'b0;
  always #(@CLOCK_HALF_PERIOD@) @CLOCK@ = ~@CLOCK@;

  initial begin
    if ($test$plusargs("ALLO_DUMP_VCD")) begin
      $dumpfile("outputs/run.vcd");
      $dumpvars(0, allo_generated_testbench);
    end
  end

  @TOP_MODULE@ dut (
@DUT_CONNECTIONS@
  );

@ARGUMENT_MODELS@

  always @(posedge @CLOCK@) begin
    if (@RESET@ || clear_completion)
      completion_seen <= '0;
    else begin
@COMPLETION_UPDATES@
    end
  end

  task automatic wait_for_completion(
    input logic [@OUTPUT_COUNT@-1:0] required,
    input integer timeout_cycles
  );
    integer elapsed;
    begin
      elapsed = 0;
      while ((completion_seen & required) !== required && elapsed < timeout_cycles) begin
        @(posedge @CLOCK@);
        #(allo_bagl_clock_compensation_ns + allo_bagl_output_delay_ns);
        if ($isunknown(completion_seen & required))
          $fatal(1, "Catapult completion status became unknown after %0d cycles", elapsed);
        elapsed = elapsed + 1;
      end
      if (elapsed >= timeout_cycles)
        $fatal(1, "timeout waiting for Catapult outputs after %0d cycles", timeout_cycles);
    end
  endtask

@ARGUMENT_TASKS@

  initial begin
    @RESET@ = 1'b1;
    clear_completion = 1'b1;
@WORKLOAD_SEQUENCE@
    $display("ALLO_TEST_PASS");
    $finish;
  end
endmodule
