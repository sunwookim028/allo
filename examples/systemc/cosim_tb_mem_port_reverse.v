// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// RTL cosim testbench for the memory-port design mem_port_reverse
// (b[i] = a[N-1-i] + 1, N=8). Unlike the stream design, A is a RANDOM-ACCESS
// INPUT routed to an internal AlloMem (no external port), so the testbench
// PRELOADS the memory via a Verilog hierarchical `force` into the synthesized
// register array -- the RTL analogue of the C++ tb's `dut.<mem>.mem[f]=..` poke.
// B is a normal stream output port (v8).
//
// ⚠️ STATUS: this cosim currently DEADLOCKS (kernel never pops the rsp). It is a
// REAL csim-vs-RTL divergence, not a TB bug: Catapult pipelines the LOAD loop
// (issues req[i+1] before popping rsp[i]) over the zero-buffer Connections::
// Combinational req/rsp channels, while AlloMem has multi-cycle latency ->
// AlloMem can't take req[1] (holding rsp[0]) and the kernel won't pop rsp[0]
// (offering req[1]) -> latency-insensitive deadlock. csim + csynth are fine; the
// RTL hangs. Fix needs buffered req/rsp (blocked by the Catapult Fifo bug), a
// no-pipeline directive on the loop, or a same-cycle combinational AlloMem.
// The preload/handshake harness below is correct and reusable once that lands.
//
// RTL memory mapping (Catapult): each mem[i] (int32) is split into two regs
//   mem_<i>_sva_rsp_0 [3:0]  = value[31:28]   (high 4 bits)
//   mem_<i>_sva_rsp_1 [27:0] = value[27:0]    (low 28 bits)
// under top.mp0_0_mem.AlloMem_int32_t_8_3_32_run_inst.  For A=[0..7] (values<16)
// -> rsp_0=0, rsp_1=i.  Expected B[k] = A[7-k]+1 = 8-k.
//
// PRODUCE RTL:  df.build(top, target="systemc", mode="csyn", project="mpr.prj"); mod()
//   -> mpr.prj/Catapult/top.v1/concat_sim_rtl.v
// RUN (Xcelium; keep CDS_LIC_FILE, drop the login LD_PRELOAD):
//   env -i HOME=$HOME PATH=/opt/cadence/XCELIUM2403/tools.lnx86/bin:/usr/bin:/bin \
//       CDS_LIC_FILE=5280@en-license-05.coecis.cornell.edu \
//       LD_LIBRARY_PATH=/opt/cadence/XCELIUM2403/tools.lnx86/lib \
//       xrun -q -sv -timescale 1ns/1ps <rtl> cosim_tb_mem_port_reverse.v

`define MEM dut.mp0_0_mem.AlloMem_int32_t_8_3_32_run_inst

module tb;
  reg clk = 0, rst = 0, v8_rdy = 0;
  wire v8_vld;
  wire [31:0] v8_dat;
  integer i, k;
  reg [31:0] B [0:7];
  reg ok;

  top dut(.clk(clk), .rst(rst), .v8_vld(v8_vld), .v8_rdy(v8_rdy), .v8_dat(v8_dat));

  always #5 clk = ~clk;   // 10 ns period

  // preload internal memory A = [0,1,..,7]  (read-only -> force holds it)
  initial begin
    rst = 0; v8_rdy = 0;
    #23;
    force `MEM.mem_0_sva_rsp_0 = 4'd0;  force `MEM.mem_0_sva_rsp_1 = 28'd0;
    force `MEM.mem_1_sva_rsp_0 = 4'd0;  force `MEM.mem_1_sva_rsp_1 = 28'd1;
    force `MEM.mem_2_sva_rsp_0 = 4'd0;  force `MEM.mem_2_sva_rsp_1 = 28'd2;
    force `MEM.mem_3_sva_rsp_0 = 4'd0;  force `MEM.mem_3_sva_rsp_1 = 28'd3;
    force `MEM.mem_4_sva_rsp_0 = 4'd0;  force `MEM.mem_4_sva_rsp_1 = 28'd4;
    force `MEM.mem_5_sva_rsp_0 = 4'd0;  force `MEM.mem_5_sva_rsp_1 = 28'd5;
    force `MEM.mem_6_sva_rsp_0 = 4'd0;  force `MEM.mem_6_sva_rsp_1 = 28'd6;
    force `MEM.mem_7_sva_rsp_0 = 4'd0;  force `MEM.mem_7_sva_rsp_1 = 28'd7;
    #7;
    rst = 1;         // release active-low reset
    v8_rdy = 1;      // ready to accept B
  end

  // collect the streamed B outputs (sample #1 after posedge to dodge races)
  initial k = 0;
  always @(posedge clk) begin
    #1;
    if (v8_vld && v8_rdy) begin
      B[k] = v8_dat;
      $display("t=%0t got B[%0d] = %0d (exp %0d)", $time, k, v8_dat, 8 - k);
      k = k + 1;
      if (k == 8) begin
        ok = 1;
        for (i = 0; i < 8; i = i + 1) if (B[i] !== (8 - i)) ok = 0;
        if (ok) $display(">>> MEM-PORT RTL COSIM PASS: B = A[::-1]+1");
        else    $display(">>> MEM-PORT RTL COSIM FAIL");
        $finish;
      end
    end
  end

  initial begin #100000; $display(">>> TIMEOUT"); $finish; end
endmodule
