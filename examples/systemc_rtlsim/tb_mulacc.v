`timescale 1ns/1ps
// ---------------------------------------------------------------------------
// Isolated RTL test of the pe_split mul->[boundary]->acc pair, straight out of
// Catapult's netlist.  No AlloMem, no feed/sink: the testbench itself drives the
// two input Connections streams (fa, fb) and sinks the result stream (res), so
// every handshake is under our control.
//
// Connections protocol (from EmitSystemC.cpp:1594-1597):
//   Push: do { val=1; msg=m; wait(); } while(!rdy); val=0;
//   Pop : do { rdy=1;        wait(); } while(!val); rdy=0;
// i.e. an ordinary synchronous valid/ready: transfer at the posedge where both high.
//
// -d BOUNDARY_WIRE     : mul.v8 -> plain wire -> acc.v22        (pe_wire)
// -d BOUNDARY_HS       : mul.v8_* -> vld/rdy/dat -> acc.v22_*   (pe_channel)
// -d BOUNDARY_FIFO     : ... through AlloFifo_ac_int_32_true_2  (pe_stream)
// -d STALL_IN=<n>      : feed a new A/B pair only every n-th cycle (n>=1)
// -d STALL_OUT=<n>     : sink accepts a result only every n-th cycle (n>=1)
// -d BREAK_WIRE        : deliberate breakage -- register the wire (1 cycle late)
// ---------------------------------------------------------------------------
`ifndef STALL_IN
 `define STALL_IN 1
`endif
`ifndef STALL_OUT
 `define STALL_OUT 1
`endif

module tb;
  reg clk = 0, rst = 0;
`ifndef ACC_RST_DELAY
 `define ACC_RST_DELAY 0
`endif
  // acc_0 may be released from reset later than mul_0, to line its loop up with the
  // first product appearing on the wire.
  reg [7:0] rstcnt = 0;
  wire rst_acc = rst && (rstcnt >= `ACC_RST_DELAY);
  always @(posedge clk or negedge rst)
    if (!rst) rstcnt <= 0; else if (rstcnt < 8'hFF) rstcnt <= rstcnt + 1;
  integer cyc = 0;
  always #1 clk = ~clk;
  always @(posedge clk) if (rst) cyc <= cyc + 1;

  reg [31:0] A [0:7];
  reg [31:0] B [0:7];
  reg [31:0] want [0:7];
  integer i;
  initial begin
    for (i = 0; i < 8; i = i + 1) begin
      A[i] = i + 1;
      B[i] = 2 * (i + 1);
    end
    want[0]=32'd2;   want[1]=32'd10;  want[2]=32'd28;  want[3]=32'd60;
    want[4]=32'd110; want[5]=32'd182; want[6]=32'd280; want[7]=32'd408;
  end

  // ---- fa / fb drivers -----------------------------------------------------
  integer ia = 0, ib = 0;
  reg  fa_vld, fb_vld;
  wire fa_rdy, fb_rdy;
  wire [31:0] fa_dat = A[ia & 3'h7];
  wire [31:0] fb_dat = B[ib & 3'h7];
  always @(*) begin
    fa_vld = rst && (ia < 8) && ((cyc % `STALL_IN) == 0);
    fb_vld = rst && (ib < 8) && ((cyc % `STALL_IN) == 0);
  end
  always @(posedge clk) begin
    if (!rst) begin ia <= 0; ib <= 0; end
    else begin
      if (fa_vld && fa_rdy) ia <= ia + 1;
      if (fb_vld && fb_rdy) ib <= ib + 1;
    end
  end

  // ---- result sink ---------------------------------------------------------
  wire res_vld;
  reg  res_rdy;
  wire [31:0] res_dat;
  integer n = 0, errs = 0, k;
  reg [31:0] got [0:7];
`ifdef LOCKSTEP
  // POSITIVE CONTROL for the wire: externally impose the lockstep the Wire boundary
  // does not provide.  mul_0 latches a new product at the edge where its internal
  // v8_and_cse is high; let acc_0 take exactly one step on the following edge.
  reg newprod_d;
  always @(posedge clk or negedge rst)
    if (!rst) newprod_d <= 1'b0;
    else      newprod_d <= tb.u_mul.mul_0_run_inst.v8_and_cse;
  always @(*) res_rdy = rst && (newprod_d || tb.u_mul.done); // mdone: drain the last push
`else
  always @(*) res_rdy = rst && ((cyc % `STALL_OUT) == 0);
`endif
  always @(posedge clk) begin
    if (rst && res_vld && res_rdy && n < 8) begin
      got[n] = res_dat;
      n = n + 1;
    end
  end

  // ---- DUT: mul -> boundary -> acc ----------------------------------------
  wire mdone, adone;
`ifdef BOUNDARY_WIRE
  wire [31:0] prod_raw;
  reg  [31:0] prod_reg;
  always @(posedge clk or negedge rst)
    if (!rst) prod_reg <= 32'd0; else prod_reg <= prod_raw;
 `ifdef BREAK_WIRE
  wire [31:0] prod = prod_reg;          // BREAKAGE: one cycle of wire latency
 `else
  `ifdef BREAK_DATA
  wire [31:0] prod = prod_raw ^ 32'd1;  // BREAKAGE: corrupt the boundary datum
  `else
  wire [31:0] prod = prod_raw;          // the real non-handshaked wire
  `endif
 `endif
  mul_0 u_mul (.clk(clk), .rst(rst), .done(mdone),
               .v6_vld(fa_vld), .v6_rdy(fa_rdy), .v6_dat(fa_dat),
               .v7_vld(fb_vld), .v7_rdy(fb_rdy), .v7_dat(fb_dat),
               .v8(prod_raw));
  acc_0 u_acc (.clk(clk), .rst(rst_acc), .done(adone),
               .v22(prod),
               .v23_vld(res_vld), .v23_rdy(res_rdy), .v23_dat(res_dat));
`else
  wire p_vld, p_rdy;  wire [31:0] p_dat;
 `ifdef BOUNDARY_FIFO
  wire q_vld, q_rdy;  wire [31:0] q_dat;
  AlloFifo_ac_int_32_true_2 u_fifo (.clk(clk), .rst(rst),
`ifdef BREAK_DATA
      .in_vld(p_vld), .in_rdy(p_rdy), .in_dat(p_dat ^ 32'd1),
`else
      .in_vld(p_vld), .in_rdy(p_rdy), .in_dat(p_dat),
`endif
      .out_vld(q_vld), .out_rdy(q_rdy), .out_dat(q_dat));
 `else
  wire q_vld;
  wire q_rdy;
  wire [31:0] q_dat;
  assign q_vld = p_vld;
 `ifdef BREAK_DATA
  assign q_dat = p_dat ^ 32'd1;         // BREAKAGE: corrupt the boundary datum
 `else
  assign q_dat = p_dat;
 `endif
  assign p_rdy = q_rdy;
 `endif
  mul_0 u_mul (.clk(clk), .rst(rst), .done(mdone),
               .v6_vld(fa_vld), .v6_rdy(fa_rdy), .v6_dat(fa_dat),
               .v7_vld(fb_vld), .v7_rdy(fb_rdy), .v7_dat(fb_dat),
               .v8_vld(p_vld), .v8_rdy(p_rdy), .v8_dat(p_dat));
  acc_0 u_acc (.clk(clk), .rst(rst), .done(adone),
               .v22_vld(q_vld), .v22_rdy(q_rdy), .v22_dat(q_dat),
               .v23_vld(res_vld), .v23_rdy(res_rdy), .v23_dat(res_dat));
`endif

  // ---- run -----------------------------------------------------------------
  initial begin
    rst = 0;
    repeat (6) @(negedge clk);
    rst = 1;
    while (n < 8 && cyc < 500) @(posedge clk);
    repeat (4) @(posedge clk);
    $display("== %s  stall_in=%0d stall_out=%0d  captured=%0d/8  cycles=%0d",
             `TAG, `STALL_IN, `STALL_OUT, n, cyc);
    for (k = 0; k < 8; k = k + 1) begin
      if (k < n) begin
        if (got[k] !== want[k]) errs = errs + 1;
        $display("   C[%0d] got=%0d want=%0d %s", k, $signed(got[k]), $signed(want[k]),
                 (got[k] === want[k]) ? "" : "<== MISMATCH");
      end else begin
        errs = errs + 1;
        $display("   C[%0d] got=<none> want=%0d <== MISSING", k, $signed(want[k]));
      end
    end
    if (errs == 0) $display("RESULT: PASS  %s", `TAG);
    else           $display("RESULT: FAIL  %s  (%0d/8 wrong)", `TAG, errs);
    $finish;
  end
endmodule
