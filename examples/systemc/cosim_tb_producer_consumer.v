// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// One-off RTL cosim testbench for the producer/consumer SystemC design
// (tests/dataflow/test_systemc_backend.py :: _producer_consumer, B = A + 1).
//
// This verifies the *generated Verilog* is functionally correct at the register
// level — a stronger check than SystemC csim. It is hand-written (no emitter
// changes / no SCVerify); SCVerify would need the emitter to emit CCS_DESIGN /
// CCS_MAIN, which it does not.
//
// HOW TO PRODUCE THE RTL + RUN (zhang-21):
//   1. synthesize {top}:
//        df.build(top, target="systemc", mode="csyn", project="pc.prj"); mod()
//      -> pc.prj/Catapult/top.v1/concat_sim_rtl.v   (self-contained RTL)
//   2. run cosim (Xcelium; VCS also available on PATH):
//        unset LD_PRELOAD; export LD_LIBRARY_PATH=""
//        xrun -q -sv -timescale 1ns/1ps \
//             pc.prj/Catapult/top.v1/concat_sim_rtl.v \
//             examples/systemc/cosim_tb_producer_consumer.v
//   expect: "sent A=0 got B=1 ... A=7 B=8" then ">>> RTL COSIM PASS".
//
// GOTCHAS (see memory allo-interconnect-extension-research):
//   * RTL has no `timescale`  -> pass -timescale to the simulator.
//   * reset is ACTIVE-LOW (async_reset_signal_is(rst,false)): rst=0 resets, rst=1 runs.
//   * Connections valid/ready must be sampled #1 AFTER the posedge, or the
//     handshake races the clock edge and stimulus is off-by-one.
//   * DUT top ports: v11 = A input  (TB drives v11_vld/v11_dat, DUT drives v11_rdy)
//                    v12 = B output (DUT drives v12_vld/v12_dat, TB drives v12_rdy)

`timescale 1ns/1ps
module tb;
  reg clk = 0, rst = 0, v11_vld = 0, v12_rdy = 0;
  reg [31:0] v11_dat = 0;
  wire v11_rdy, v12_vld;
  wire [31:0] v12_dat;
  integer i = 0, o = 0, errors = 0;

  localparam N = 8;

  top dut(.clk(clk), .rst(rst),
          .v11_vld(v11_vld), .v11_rdy(v11_rdy), .v11_dat(v11_dat),
          .v12_vld(v12_vld), .v12_rdy(v12_rdy), .v12_dat(v12_dat));

  always #5 clk = ~clk;                          // 10 ns clock

  initial begin                                  // active-low reset, then run
    rst = 0; v12_rdy = 1;                         // always ready to accept B
    repeat (5) @(posedge clk);
    rst = 1;
  end

  // source: present A = 0..N-1, advance only on an accepted cycle (vld && rdy)
  initial begin
    wait (rst == 1);
    v11_dat = 0; v11_vld = 1;
    forever begin
      @(posedge clk); #1;                        // settle before sampling
      if (v11_vld && v11_rdy) begin
        $display("t=%0t sent A=%0d", $time, v11_dat);
        i = i + 1;
        if (i < N) v11_dat = i; else v11_vld = 0;
      end
    end
  end

  // sink: capture B on every accepted cycle, check B == A + 1
  initial begin
    wait (rst == 1);
    forever begin
      @(posedge clk); #1;
      if (v12_vld && v12_rdy) begin
        $display("t=%0t got  B=%0d (exp %0d)", $time, v12_dat, o + 1);
        if (v12_dat !== o + 1) errors = errors + 1;
        o = o + 1;
        if (o == N) begin
          if (errors == 0) $display(">>> RTL COSIM PASS: all %0d outputs = input+1", N);
          else             $display(">>> RTL COSIM FAIL: %0d errors", errors);
          $finish;
        end
      end
    end
  end

  initial begin #200000 $display(">>> TIMEOUT (o=%0d) — possible deadlock", o); $finish; end
endmodule
