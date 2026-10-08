// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// U4 harness wrapper (not MiniTPU RTL): sequencer_iram + sequencer_fetch_queue
// wired exactly as sequencer.sv:99-124 wires them, with every port the
// sequencer drives brought out (pause is a port here; sequencer.sv drives it
// from a register that is always 0).
`timescale 1ns/1ps
module u4_fetch
  import minitpu_config_pkg::*;
  import sequencer_pkg::*;
(
    input  logic clk,
    input  logic rst_n,
    input  logic                    instr_write_en,
    input  logic [INSTR_ADDR_W-1:0] iram_addr,
    input  logic [BUNDLE_WIDTH-1:0] dma_iram_din,
    input  logic                    pause,
    input  logic                    pc_flush,
    input  logic [INSTR_ADDR_W-1:0] restart_addr,
    input  logic                    bundle_pop,
    output logic [INSTR_ADDR_W-1:0] rd_addr,
    output logic                    rd_addr_valid,
    output logic                    bundle_valid,
    output logic [BUNDLE_WIDTH-1:0] bundle_data,
    output logic [INSTR_ADDR_W-1:0] bundle_addr,
    output logic                    empty,
    output logic                    full
);
  logic [BUNDLE_WIDTH-1:0] iram_rd_data;
  sequencer_iram u_iram (
      .clk(clk), .fetch_addr(rd_addr), .rd_data_o(iram_rd_data),
      .instr_write_en(instr_write_en), .iram_addr(iram_addr), .dma_iram_din(dma_iram_din));
  sequencer_fetch_queue u_fetch_queue (
      .clk(clk), .rst_n(rst_n),
      .rd_addr_o(rd_addr), .rd_addr_valid_o(rd_addr_valid),
      .bram_data_i(iram_rd_data), .pause_i(pause), .pc_flush_i(pc_flush),
      .pc_restart_addr_i(restart_addr),
      .bundle_valid_o(bundle_valid), .bundle_data_o(bundle_data), .bundle_addr_o(bundle_addr),
      .bundle_pop_i(bundle_pop), .empty_o(empty), .full_o(full));
endmodule
