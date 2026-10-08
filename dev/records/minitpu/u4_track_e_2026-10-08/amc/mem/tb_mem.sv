// Behaviour of AMC's lowered D-12 memories (amctool --memory-only), held to the
// D-12 declarations: read latency (address -> data, in edges), write visible one
// edge later (visible=1), same-cycle read of a row being written (D-12 declares
// "refuse" for IRAM and the loop buffer and "obligation" for VMEM: AMC's choice is
// recorded, not judged), and two writes to one row in one cycle (VMEM, H11).
// run: (scl gcc-toolset-13) verilator --binary --timing -Wno-fatal -MAKEFLAGS "CXX=g++ LINK=g++ AR=ar" tb_mem.sv iram.sv loopbuf.sv vmem_split.sv vmem_rw11.sv
`timescale 1ns/1ps
module tb_mem;
  logic clk = 0, rst = 1;
  always #5 clk = ~clk;
  int errors = 0;
  task automatic check(string what, logic [127:0] got, logic [127:0] want);
    if (got !== want) begin errors++; $display("CHECK-FAIL %s: got %h want %h", what, got, want); end
    else $display("CHECK-OK   %s", what);
  endtask
  // ---- IRAM: w(1) + r(1), 4096 x 128 ----
  logic [11:0] ia0, ia1; logic [127:0] iwd, ird; logic iwe = 0, ire = 0, id0, id1;
  iram_impl iram(.clk(clk), .rst(rst), .p0_addr(ia0), .p0_wr_data(iwd), .p0_wr_en(iwe),
                 .p1_addr(ia1), .p1_rd_en(ire), .p0_done(id0), .p1_rd_data(ird), .p1_done(id1));
  // ---- loop buffer: w(1) + r(0), 24 x 128 ----
  logic [4:0] la0, la1; logic [127:0] lwd, lrd; logic lwe = 0, lre = 0, ld0, ld1;
  loopbuf_impl lb(.clk(clk), .rst(rst), .p0_addr(la0), .p0_wr_data(lwd), .p0_wr_en(lwe),
                  .p1_addr(la1), .p1_rd_en(lre), .p0_done(ld0), .p1_rd_data(lrd), .p1_done(ld1));
  // ---- VMEM workaround: c = r(3) + w(1), d = r(2) + w(1) (E-A3) ----
  logic [11:0] va[4]; logic [1023:0] vwd1, vwd3, vrd0, vrd2; logic vre0 = 0, vwe1 = 0, vre2 = 0, vwe3 = 0, vd0, vd2;
  vmem_split_impl vm(.clk(clk), .rst(rst), .p0_addr(va[0]), .p0_rd_en(vre0), .p1_addr(va[1]), .p1_wr_data(vwd1),
                     .p1_wr_en(vwe1), .p2_addr(va[2]), .p2_rd_en(vre2), .p3_addr(va[3]), .p3_wr_data(vwd3),
                     .p3_wr_en(vwe3), .p0_rd_data(vrd0), .p0_done(vd0), .p2_rd_data(vrd2), .p2_done(vd2));
  // ---- VMEM at rw(1, 1) x 2 ----
  logic [11:0] wa0, wa1; logic [1023:0] wwd0, wwd1, wrd0, wrd1; logic wre0 = 0, wwe0 = 0, wre1 = 0, wwe1 = 0, wd0, wd1;
  vmem_rw11_impl vr(.clk(clk), .rst(rst), .p0_addr(wa0), .p0_rd_en(wre0), .p0_wr_data(wwd0), .p0_wr_en(wwe0),
                    .p1_addr(wa1), .p1_rd_en(wre1), .p1_wr_data(wwd1), .p1_wr_en(wwe1),
                    .p0_rd_data(wrd0), .p0_done(wd0), .p1_rd_data(wrd1), .p1_done(wd1));
  localparam logic [127:0] A = 128'hA5A5_0000_1111_2222_3333_4444_5555_6666, B = 128'h0123_4567_89AB_CDEF_0F0F_F0F0_1234_5678;
  initial begin
    // The simulator (Verilator) is two-state (an unreset word reads 0, not X): put a known OLD word in
    // the rows the collision checks read, by hierarchical reference.
    iram.mem0[5] = B; lb.mem0[3] = A; lb.mem0[4] = A;
    @(negedge clk); rst = 0;
    // IRAM: write row 5 = A (edge 1); same cycle read row 5 -> old (unwritten: X)
    iwe = 1; ia0 = 5; iwd = A; ire = 1; ia1 = 5;
    @(negedge clk); iwe = 0; ia1 = 5;                      // edge 1 done: read issued at edge 1 saw old
    check("iram r(1): same-cycle read of the row being written = the OLD word", ird, B);
    @(negedge clk);                                        // read of row 5 issued in the 2nd cycle, data after edge 2
    check("iram r(1): write visible at the next cycle, data 1 edge after the address", ird, A);
    ire = 0; ia1 = 7; @(negedge clk);
    check("iram r(1): rd_en low holds the last word (rd_hold)", ird, A);
    // loop buffer: async read
    lwe = 1; la0 = 3; lwd = B; lre = 1; la1 = 3; #1;
    check("loopbuf r(0): same-cycle read is combinational and the OLD word", lrd, A);
    @(negedge clk); lwe = 0; #1;
    check("loopbuf r(0): write visible the next cycle, read combinational", lrd, B);
    la1 = 4; #1; check("loopbuf r(0): an address change is seen without an edge", lrd, A);
    // VMEM split: write row 9 through d (p3) = {8{B}}, read it through c (p0, r(3)) and d (p2, r(2))
    vwe3 = 1; va[3] = 9; vwd3 = {8{B}}; @(negedge clk); vwe3 = 0;
    vre0 = 1; va[0] = 9; vre2 = 1; va[2] = 9; @(negedge clk); vre0 = 0; vre2 = 0;
    check("vmem c r(3): 1 edge after the address: not yet", vrd0[127:0] === B, 0);
    @(negedge clk); check("vmem d r(2): data 2 edges after the address", vrd2[127:0], B);
    check("vmem c r(3): 2 edges: not yet", vrd0[127:0] === B, 0);
    @(negedge clk); check("vmem c r(3): data 3 edges after the address", vrd0[127:0], B);
    // VMEM rw11: both ports write row 2 in one cycle
    wwe0 = 1; wa0 = 2; wwd0 = {8{A}}; wwe1 = 1; wa1 = 2; wwd1 = {8{B}}; @(negedge clk); wwe0 = 0; wwe1 = 0;
    wre0 = 1; wa0 = 2; @(negedge clk); wre0 = 0;
    $display("NOTE vmem rw(1,1) x2: two writes to row 2 in one cycle leave %s (no collision check, no X)",
             wrd0[127:0] === A ? "port 0's word (p0 wins)" : (wrd0[127:0] === B ? "port 1's word (p1 wins)" : "neither"));
    if (errors == 0) $display("TB-MEM PASS"); else $display("TB-MEM FAIL %0d", errors);
    $finish;
  end
endmodule
