`timescale 1ns/1ps
// Generic testbench for the archived Catapult RTL of pe_split's four-kernel PE.
//   AB[2][8] lives in the internal AlloMem (flat index r*8+c); results leave on v40.
// Preload is a hierarchical poke of the AlloMem flops -- exactly what the SystemC
// testbench does in csim ("AlloMem.mem[], done before reset is released").
`define MEMPATH dut.mp0_0_mem.AlloMem_ac_int_32_true_16_4_32_run_inst
`define POKE(N,VAL) begin \
  `MEMPATH.mem_``N``_sva_rsp_0 = (VAL) >> 27; \
  `MEMPATH.mem_``N``_sva_rsp_1 = (VAL) & 32'h07FFFFFF; end

module tb;
  reg clk = 0;
  reg rst = 0;
  reg v40_rdy = 1;
  wire done, v40_vld;
  wire [31:0] v40_dat;

  integer n, errs, cyc;
  reg [31:0] got  [0:7];
  reg [31:0] want [0:7];

  `TOPMOD dut (.clk(clk), .rst(rst), .done(done),
               .v40_vld(v40_vld), .v40_rdy(v40_rdy), .v40_dat(v40_dat));

  always #1 clk = ~clk;

  // golden: C[i] = sum_{j<=i} A[j]*B[j], A=1..8, B=2,4,...,16
  initial begin
    want[0]=32'd2;   want[1]=32'd10;  want[2]=32'd28;  want[3]=32'd60;
    want[4]=32'd110; want[5]=32'd182; want[6]=32'd280; want[7]=32'd408;
  end

  initial begin
    n = 0; errs = 0; cyc = 0;
    rst = 0;
`ifndef RSTN
 `define RSTN 6
`endif
`ifdef RST_POS
    repeat (`RSTN) @(posedge clk);
    #0.2 rst = 1;
`else
    repeat (`RSTN) @(negedge clk);
    rst = 1;
`endif
    // A at flat 0..7, B at flat 8..15
    `POKE(0,  32'd1)  `POKE(1,  32'd2)  `POKE(2,  32'd3)  `POKE(3,  32'd4)
    `POKE(4,  32'd5)  `POKE(5,  32'd6)  `POKE(6,  32'd7)  `POKE(7,  32'd8)
    `POKE(8,  32'd2)  `POKE(9,  32'd4)  `POKE(10, 32'd6)  `POKE(11, 32'd8)
    `POKE(12, 32'd10) `POKE(13, 32'd12) `POKE(14, 32'd14) `POKE(15, 32'd16)
`ifdef BREAK_MEM
    `POKE(3,  32'd99)   // deliberate-breakage control: corrupt one input word
`endif
  end

  // capture the output stream
  always @(posedge clk) begin
    if (rst) begin
      cyc <= cyc + 1;
      if (v40_vld && v40_rdy && n < 8) begin
        got[n] = v40_dat;
        n = n + 1;
      end
    end
  end

  integer k;
  initial begin
    wait (rst);
    while (n < 8 && cyc < 2000) @(posedge clk);
    repeat (4) @(posedge clk);
    $display("TOP=%s  captured=%0d  cycles=%0d  done=%b", `TOPNAME, n, cyc, done);
    for (k = 0; k < 8; k = k + 1) begin
      if (k < n) begin
        if (got[k] !== want[k]) errs = errs + 1;
        $display("  C[%0d] got=%0d want=%0d %s", k, $signed(got[k]), $signed(want[k]),
                 (got[k] === want[k]) ? "ok" : "<== MISMATCH");
      end else begin
        errs = errs + 1;
        $display("  C[%0d] got=<none>  want=%0d <== MISSING", k, $signed(want[k]));
      end
    end
    if (errs == 0) $display("RESULT: PASS  (%s)", `TOPNAME);
    else           $display("RESULT: FAIL  (%s)  %0d/8 wrong", `TOPNAME, errs);
    $finish;
  end
endmodule
