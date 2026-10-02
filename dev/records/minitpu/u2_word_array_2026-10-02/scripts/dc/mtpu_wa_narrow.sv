// MiniTPU's vpu_word_array at the harness's ``narrow`` geometry (8 words of
// 64 b: NUM_LANES = 1, VMEM_ENTRIES_PER_LANE = 16), for the DC same-flow
// comparison. The defines must precede vpu_pkg.sv in one ``analyze`` call;
// the wrapper renames nothing: its ports are the unit's. DC predefines
// SYNTHESIS (and refuses `undef, VER-402), which selects the XPM primitive:
// vpu_word_array_sim.sv beside this file is the unit with that branch
// removed, so the simulation model (flops + read pipes) is what DC
// synthesizes -- the like-for-like of Catapult's register map in ASIC.
`define MINITPU_NUM_LANES 1
`define MINITPU_VMEM_ENTRIES_PER_LANE 16
module mtpu_wa_narrow (
  input  logic        clk_i,
  input  logic        compute_en_i, compute_we_i,
  input  logic [2:0]  compute_addr_i,
  input  logic [63:0] compute_wdata_i,
  output logic [63:0] compute_rdata_o,
  input  logic        dma_en_i, dma_we_i,
  input  logic [2:0]  dma_addr_i,
  input  logic [63:0] dma_wdata_i,
  output logic [63:0] dma_rdata_o
);
  vpu_word_array u (
    .clk_i, .rst_ni(1'b1),
    .compute_en_i, .compute_we_i, .compute_addr_i, .compute_wdata_i, .compute_rdata_o,
    .dma_en_i, .dma_we_i, .dma_addr_i, .dma_wdata_i, .dma_rdata_o);
endmodule
