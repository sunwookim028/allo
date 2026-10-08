# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Generates the shims that put MiniTPU's ``minitpu_core`` behind pins PR #48's RTLModule carries.

The shims are Allo's RTL, not MiniTPU's. Each instantiates ``minitpu_core`` (seam K of
``dev/records/minitpu/minitpu_rtl_plan_2026-10-08.rst`` section 2) unmodified and adapts each of its port
groups to <= 32-bit stream or RAM pins:

* ``cmd``  (32-bit ready/valid stream in): the host program. IRAM bundles as four 32-bit words each
  (little-endian, word 0 = bundle bits [31:0], exactly the bytes of an ``asm.py`` image), CSR writes
  (``kernel_arg_csr`` 0..3, ``program_id_csr``), a cycle limit, ``START``, ``END``; see ``CMD``.
* ``st``   (32-bit ready/valid stream out): ``NSTAT`` status words after each run; see ``STATUS``.
* ``mw0..mw7`` / ``mr0..mr7`` (sixteen #48 ``MemPort``\ s, 32-bit words, read latency 1; ``mwk`` and ``mrk``
  are bound to the same Allo array, write port first, so a read sees a same-cycle write): the device memory,
  word-interleaved so that one 256-bit DM word is one access on each bank in one cycle. Bank k holds bytes
  [4k, 4k+4) of every 32-byte DM word.
* ``done``: high after ``END`` -- the end of the RTLModule call.

The core's device-memory credit pipe (``dm_req_*``/``dm_rsp_*``; the contract of ``u4_track_c_2026-10-08.rst``
section 10) is the seam in both forms; what answers it differs:

* ``bridge`` (``rtl/minitpu_core_shim.sv``, the default): MiniTPU's own ``src/ddr/uncore_io_tile.sv``
  (``dm_axi_bridge``, two requests in flight, landing FIFO), unmodified, then an AXI4 slave on the banks that
  answers as the testbench's ``axi4_mem_model`` at ``ROUND_TRIP_CYCLES=0``. The memory system of
  ``tb_kernel_image``, so the cycle counts are comparable (M-R2b).
* ``direct`` (``rtl/minitpu_core_shim_direct.sv``): M-R1's memory, the credit pipe answered straight from the
  banks, one request in flight; a request outside the 32 MiB window is answered DECERR (``2'b11``).

    python examples/minitpu-rtl/gen_shim.py            # rewrites both shims
    python examples/minitpu-rtl/gen_shim.py --check    # fails if a checked-in file differs
"""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
MODULES = {"bridge": "minitpu_core_shim", "direct": "minitpu_core_shim_direct"}
OUTS = {m: HERE / "rtl" / f"{name}.sv" for m, name in MODULES.items()}
MEMORY_NOTE = {
    "bridge": "through MiniTPU's own uncore_io_tile (dm_axi_bridge, OUTSTANDING 2) and an AXI4 slave modelled on the "
              "testbench's axi4_mem_model at ROUND_TRIP_CYCLES=0 (M-R2b)",
    "direct": "answered straight from the banks, one request in flight (M-R1's memory)",
}

BANKS = 8                      # 256-bit DM word / 32-bit RAM word
BANK_ADDR_W = 20               # 2**20 DM words = 32 MiB, tb_kernel_image's MEM_SIZE
MEM_BASE = 0x8000_0000         # tb_kernel_image's MEM_BASE (byte address)
DM_WORD_BYTES = 32
NSTAT = 16

# Command words: opcode in [31:28].
CMD = {
    "NOP": 0x0,      # nothing
    "IRAM": 0x1,     # [27:16] first IRAM address, [15:0] n; then 4n words, bundle by bundle
    "CSR": 0x2,      # [3:0] k: 0..3 kernel_arg_csr[32k+:32], 4 program_id_csr; then 1 value word
    "START": 0x3,    # pulse start, wait for done (or the cycle limit), then emit NSTAT status words
    "MAXCYC": 0x4,   # then 1 word: cycle limit for the next run (0 = none)
    "END": 0xF,      # raise done (end of the RTLModule call); the last word of the program
}

# Status words, in order.
STATUS = [
    ("flags", "{25'd0, bridge_to_seen, bad_cmd_q, timeout_q, core_illegal_seen_q, dma_rsp_seen, dma_err, done_seen_q}"),
    ("perf_cnt_cycles", "perf_cnt_cycles"),
    ("perf_cnt_instrs", "perf_cnt_instrs"),
    ("dma_err", "{dma_err, 7'd0, dma_err_channel, 6'd0, dma_err_cause, 6'd0, dma_err_resp}"),
    ("beat_count", "beat_count"),
    ("overlap_stat", "overlap_stat"),
    ("dma_channel_done", "32'(dma_channel_done)"),
    ("shim_run_cycles", "run_cyc_q"),
    ("shim_load_words", "rd_words_q"),
    ("shim_store_words", "wr_words_q"),
    ("shim_mem_errors", "mem_err_q"),
    ("shim_partial_wstrb", "wstrb_err_q"),
    ("shim_bundles_loaded", "bundles_q"),
    ("shim_load_requests", "rd_reqs_q"),
    ("shim_store_requests", "wr_reqs_q"),
    ("magic", "32'h4D52_3101"),
]
assert len(STATUS) == NSTAT


def direct_side() -> str:
    """M-R1's memory: the credit pipe answered straight from the banks, one request in flight."""
    base_word = MEM_BASE // DM_WORD_BYTES
    depth = 1 << BANK_ADDR_W
    bank_assign = []
    for k in range(BANKS):
        bank_assign += [f"  assign mw{k}_addr = wr_idx;", f"  assign mw{k}_ce   = mem_wr;", f"  assign mw{k}_we   = mem_wr;",
                        f"  assign mw{k}_d    = dm_req_wdata[{32 * k + 31}:{32 * k}];",
                        f"  assign mr{k}_addr = rd_idx_q;", f"  assign mr{k}_ce   = mem_rd;"]
    q_concat = ", ".join(f"mr{k}_q" for k in reversed(range(BANKS)))
    return f"""  // ---------------------------------------------------------------- memory side: the credit pipe on {BANKS} RAMs
  wire bridge_to_seen = 1'b0;   // no bridge in this form
  localparam logic [1:0] M_IDLE = 2'd0, M_READ = 2'd1, M_WRITE = 2'd2, M_WRESP = 2'd3;
  logic [1:0]  ms_q;
  logic [{BANK_ADDR_W - 1}:0] rd_idx_q, wr_idx_q, wr_idx;
  logic [8:0]  rd_issue_left_q, rd_ret_left_q, wr_left_q;
  logic        pend_q, err_q;
  logic [31:0] rd_words_q, wr_words_q, mem_err_q, wstrb_err_q, rd_reqs_q, wr_reqs_q;

  // Range check of a new request: [addr, addr + len] must lie in [BASE_WORD, BASE_WORD + DEPTH).
  wire [29:0] req_off = {{1'b0, dm_req_addr}} - BASE_WORD;
  wire        req_ok  = ({{1'b0, dm_req_addr}} >= BASE_WORD) && (req_off + {{22'd0, dm_req_len}} < DEPTH);

  assign dm_req_ready = (ms_q == M_IDLE) || (ms_q == M_WRITE);
  wire req_fire  = dm_req_valid & dm_req_ready;
  wire first_beat = req_fire && (ms_q == M_IDLE);
  wire rsp_fire  = dm_rsp_valid & dm_rsp_ready;

  // Store beats write all {BANKS} banks in the cycle they are taken.
  assign wr_idx = (ms_q == M_IDLE) ? req_off[{BANK_ADDR_W - 1}:0] : wr_idx_q;
  wire wr_ok  = (ms_q == M_IDLE) ? req_ok : !err_q;
  wire mem_wr = req_fire && dm_req_we && wr_ok;
  // Loads issue one word per cycle once the previous one is (being) taken; the RAM answers a cycle later.
  wire rd_issue = (ms_q == M_READ) && (rd_issue_left_q != 9'd0) && (!pend_q || rsp_fire);
  wire mem_rd   = rd_issue && !err_q;

{chr(10).join(bank_assign)}

  assign dm_rsp_valid = ((ms_q == M_READ) && pend_q) || (ms_q == M_WRESP);
  assign dm_rsp_data  = ((ms_q == M_READ) && !err_q) ? {{{q_concat}}} : '0;
  assign dm_rsp_last  = (ms_q == M_WRESP) || (rd_ret_left_q == 9'd1);
  assign dm_rsp_resp  = err_q ? 2'b11 : 2'b00;   // DECERR, as tb's axi4_mem_model answers outside its window

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ms_q <= M_IDLE; rd_idx_q <= '0; wr_idx_q <= '0; rd_issue_left_q <= '0; rd_ret_left_q <= '0;
      wr_left_q <= '0; pend_q <= 1'b0; err_q <= 1'b0;
      rd_words_q <= '0; wr_words_q <= '0; mem_err_q <= '0; wstrb_err_q <= '0; rd_reqs_q <= '0; wr_reqs_q <= '0;
    end else begin
      if (req_fire && dm_req_we && dm_req_wstrb != 32'hFFFF_FFFF) wstrb_err_q <= wstrb_err_q + 32'd1;
      if (mem_wr) wr_words_q <= wr_words_q + 32'd1;
      unique case (ms_q)
        M_IDLE: if (first_beat) begin
          err_q <= !req_ok;
          if (!req_ok) mem_err_q <= mem_err_q + 32'd1;
          if (dm_req_we) begin
            wr_reqs_q <= wr_reqs_q + 32'd1;
            wr_idx_q <= req_off[{BANK_ADDR_W - 1}:0] + {BANK_ADDR_W}'d1;
            wr_left_q <= {{1'b0, dm_req_len}};
            ms_q <= (dm_req_len == 8'd0) ? M_WRESP : M_WRITE;
          end else begin
            rd_reqs_q <= rd_reqs_q + 32'd1;
            rd_idx_q <= req_off[{BANK_ADDR_W - 1}:0];
            rd_issue_left_q <= {{1'b0, dm_req_len}} + 9'd1;
            rd_ret_left_q <= {{1'b0, dm_req_len}} + 9'd1;
            pend_q <= 1'b0;
            ms_q <= M_READ;
          end
        end
        M_WRITE: if (req_fire) begin
          wr_idx_q <= wr_idx_q + {BANK_ADDR_W}'d1;
          wr_left_q <= wr_left_q - 9'd1;
          if (wr_left_q == 9'd1) ms_q <= M_WRESP;
        end
        M_WRESP: if (rsp_fire) begin err_q <= 1'b0; ms_q <= M_IDLE; end
        default: begin  // M_READ
          if (rd_issue) begin
            rd_idx_q <= rd_idx_q + {BANK_ADDR_W}'d1;
            rd_issue_left_q <= rd_issue_left_q - 9'd1;
          end
          pend_q <= rd_issue ? 1'b1 : (rsp_fire ? 1'b0 : pend_q);
          if (rsp_fire) begin
            rd_words_q <= rd_words_q + 32'd1;
            rd_ret_left_q <= rd_ret_left_q - 9'd1;
            if (rd_ret_left_q == 9'd1) begin err_q <= 1'b0; pend_q <= 1'b0; ms_q <= M_IDLE; end
          end
        end
      endcase
    end
  end
"""


def bridge_side() -> str:
    """M-R2b's memory: MiniTPU's own ``uncore_io_tile`` (``dm_axi_bridge`` + landing FIFO, unmodified) on the
    core's credit pipe, and below its AXI4 master a slave that answers as ``tb/minitpu_axi_mem_model.svh``'s
    ``axi4_mem_model`` does at ``ROUND_TRIP_CYCLES=0``: one read burst at a time (``arready`` only with none
    pending), its first beat the cycle after the address, a beat per cycle; a single-outstanding AW/W/B
    machine with ``bvalid`` the cycle after ``wlast``; DECERR (``2'b11``) and zero data outside the window."""
    bank_assign = []
    for k in range(BANKS):
        bank_assign += [f"  assign mw{k}_addr = aw_idx_q;", f"  assign mw{k}_ce   = mem_wr;", f"  assign mw{k}_we   = mem_wr;",
                        f"  assign mw{k}_d    = axi_wdata[{32 * k + 31}:{32 * k}];",
                        f"  assign mr{k}_addr = rd_first ? ar_idx : r_idx_q + {BANK_ADDR_W}'d1;",
                        f"  assign mr{k}_ce   = mem_rd;"]
    q_concat = ", ".join(f"mr{k}_q" for k in reversed(range(BANKS)))
    return f"""  // ---------------------------------------------------------------- memory side: MiniTPU's bridge, then the TB's memory
  // The credit pipe (u4_track_c section 10) is the core's port and stays the seam: the bridge sits BELOW it,
  // between the core's DMA and an AXI4 slave, exactly as src/minitpu.sv wires it once the IRAM loader is idle.
  localparam int AW = minitpu_config_pkg::DRAM_BYTE_ADDR_W;
  localparam logic [32:0] MEM_LO = 33'h{MEM_BASE:X};
  localparam logic [32:0] MEM_HI = 33'h{MEM_BASE + (1 << BANK_ADDR_W) * DM_WORD_BYTES:X};   // BASE + SIZE

  logic [AW-1:0]  axi_araddr, axi_awaddr;
  logic [7:0]     axi_arlen, axi_awlen;
  logic [2:0]     axi_arsize, axi_awsize;
  logic [1:0]     axi_arburst, axi_awburst, axi_rresp, axi_bresp;
  logic [0:0]     axi_arid, axi_awid;
  logic           axi_arvalid, axi_arready, axi_rvalid, axi_rready, axi_rlast;
  logic           axi_awvalid, axi_awready, axi_wvalid, axi_wready, axi_wlast, axi_bvalid, axi_bready;
  logic [255:0]   axi_rdata, axi_wdata;
  logic [31:0]    axi_wstrb;
  logic           bridge_to_seen, axi_ar_seen, axi_r_seen, axi_aw_seen, axi_b_seen;

  uncore_io_tile #(.M_AXI_ID_WIDTH(1), .M_AXI_DATA_WIDTH(256)) u_uncore_io_tile (
    .clk, .rst_n,
    .dm_req_valid, .dm_req_ready, .dm_req_we, .dm_req_addr, .dm_req_len, .dm_req_wdata, .dm_req_wstrb,
    .dm_rsp_valid, .dm_rsp_ready, .dm_rsp_data, .dm_rsp_last, .dm_rsp_resp,
    .bridge_to_seen, .axi_ar_seen, .axi_r_seen, .axi_aw_seen, .axi_b_seen,
    .dm_axi_arid(axi_arid), .dm_axi_araddr(axi_araddr), .dm_axi_arlen(axi_arlen), .dm_axi_arsize(axi_arsize),
    .dm_axi_arburst(axi_arburst), .dm_axi_arvalid(axi_arvalid), .dm_axi_arready(axi_arready),
    .dm_axi_rid(1'b0), .dm_axi_rdata(axi_rdata), .dm_axi_rresp(axi_rresp), .dm_axi_rvalid(axi_rvalid),
    .dm_axi_rlast(axi_rlast), .dm_axi_rready(axi_rready),
    .dm_axi_awid(axi_awid), .dm_axi_awaddr(axi_awaddr), .dm_axi_awlen(axi_awlen), .dm_axi_awsize(axi_awsize),
    .dm_axi_awburst(axi_awburst), .dm_axi_awvalid(axi_awvalid), .dm_axi_awready(axi_awready),
    .dm_axi_wdata(axi_wdata), .dm_axi_wstrb(axi_wstrb), .dm_axi_wlast(axi_wlast), .dm_axi_wvalid(axi_wvalid),
    .dm_axi_wready(axi_wready),
    .dm_axi_bid(1'b0), .dm_axi_bresp(axi_bresp), .dm_axi_bvalid(axi_bvalid), .dm_axi_bready(axi_bready)
  );
  // The TB's AXI port is 32 bits wide (m_araddr): the model sees the address's low 32 bits, as here.
  wire [31:0] ar_a = axi_araddr[31:0];
  wire [31:0] aw_a = axi_awaddr[31:0];
  // axi4_mem_model's in_range(addr, (len + 1) << size), plus what the banks can serve: whole 32 B beats.
  wire [32:0] ar_bytes = ({{25'd0, axi_arlen}} + 33'd1) << axi_arsize;
  wire [32:0] aw_bytes = ({{25'd0, axi_awlen}} + 33'd1) << axi_awsize;
  wire ar_ok = ({{1'b0, ar_a}} >= MEM_LO) && ({{1'b0, ar_a}} + ar_bytes <= MEM_HI);
  wire aw_ok = ({{1'b0, aw_a}} >= MEM_LO) && ({{1'b0, aw_a}} + aw_bytes <= MEM_HI);
  wire ar_shape = (axi_arsize == 3'd5) && (ar_a[4:0] == 5'd0);
  wire aw_shape = (axi_awsize == 3'd5) && (aw_a[4:0] == 5'd0);
  wire [{BANK_ADDR_W - 1}:0] ar_idx = 20'((ar_a - 32'h{MEM_BASE:X}) >> 5);
  wire [{BANK_ADDR_W - 1}:0] aw_idx = 20'((aw_a - 32'h{MEM_BASE:X}) >> 5);

  // Read channel: reads.size() == 0 is arready at round trip 0; beat k shows the cycle after it is fetched.
  logic       rd_busy_q, ar_err_q;
  logic [7:0] ar_len_q, r_beat_q;
  logic [{BANK_ADDR_W - 1}:0] r_idx_q;            // bank index of the beat on the R channel
  assign axi_arready = !rd_busy_q;
  wire ar_fire = axi_arvalid & axi_arready;
  wire r_fire  = axi_rvalid & axi_rready;
  wire r_last_beat = (r_beat_q == ar_len_q);
  wire rd_first = ar_fire && ar_ok && ar_shape;
  wire mem_rd   = rd_first || (r_fire && !r_last_beat && !ar_err_q);
  assign axi_rvalid = rd_busy_q;
  assign axi_rlast  = rd_busy_q && r_last_beat;
  assign axi_rresp  = ar_err_q ? 2'b11 : 2'b00;
  assign axi_rdata  = ar_err_q ? '0 : {{{q_concat}}};

  // Write channel: W_IDLE -> W_DATA on AW, -> W_RESP on wlast, bvalid in W_RESP's first cycle.
  localparam logic [1:0] W_IDLE = 2'd0, W_DATA = 2'd1, W_RESP = 2'd2;
  logic [1:0] ws_q;
  logic       aw_err_q;
  logic [{BANK_ADDR_W - 1}:0] aw_idx_q;
  assign axi_awready = (ws_q == W_IDLE);
  assign axi_wready  = (ws_q == W_DATA);
  wire aw_fire = axi_awvalid & axi_awready;
  wire w_fire  = axi_wvalid & axi_wready;
  wire mem_wr  = w_fire && !aw_err_q;
  assign axi_bvalid = (ws_q == W_RESP);
  assign axi_bresp  = aw_err_q ? 2'b11 : 2'b00;

{chr(10).join(bank_assign)}

  logic [31:0] rd_words_q, wr_words_q, mem_err_q, wstrb_err_q, rd_reqs_q, wr_reqs_q;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_busy_q <= 1'b0; ar_err_q <= 1'b0; ar_len_q <= '0; r_beat_q <= '0; r_idx_q <= '0;
      ws_q <= W_IDLE; aw_err_q <= 1'b0; aw_idx_q <= '0;
      rd_words_q <= '0; wr_words_q <= '0; mem_err_q <= '0; wstrb_err_q <= '0; rd_reqs_q <= '0; wr_reqs_q <= '0;
    end else begin
      if (ar_fire) begin
        rd_busy_q <= 1'b1; ar_len_q <= axi_arlen; r_beat_q <= '0; r_idx_q <= ar_idx;
        ar_err_q <= !(ar_ok && ar_shape);
        rd_reqs_q <= rd_reqs_q + 32'd1;
        if (!(ar_ok && ar_shape)) mem_err_q <= mem_err_q + 32'd1;
      end else if (r_fire) begin
        rd_words_q <= rd_words_q + 32'd1;
        if (r_last_beat) rd_busy_q <= 1'b0;
        else begin r_beat_q <= r_beat_q + 8'd1; r_idx_q <= r_idx_q + {BANK_ADDR_W}'d1; end
      end
      unique case (ws_q)
        W_IDLE: if (aw_fire) begin
          aw_idx_q <= aw_idx; aw_err_q <= !(aw_ok && aw_shape); ws_q <= W_DATA;
          wr_reqs_q <= wr_reqs_q + 32'd1;
          if (!(aw_ok && aw_shape)) mem_err_q <= mem_err_q + 32'd1;
        end
        W_DATA: if (w_fire) begin
          wr_words_q <= wr_words_q + 32'd1;
          if (axi_wstrb != 32'hFFFF_FFFF) wstrb_err_q <= wstrb_err_q + 32'd1;
          aw_idx_q <= aw_idx_q + {BANK_ADDR_W}'d1;
          if (axi_wlast) ws_q <= W_RESP;
        end
        W_RESP: if (axi_bvalid && axi_bready) ws_q <= W_IDLE;
        default: ws_q <= W_IDLE;
      endcase
    end
  end
"""



def emit(memory: str = "bridge") -> str:
    base_word = MEM_BASE // DM_WORD_BYTES
    depth = 1 << BANK_ADDR_W
    bank_ports = []
    for k in range(BANKS):
        bank_ports += [f"  output logic [{BANK_ADDR_W - 1}:0] mw{k}_addr,", f"  output logic        mw{k}_ce,",
                       f"  output logic        mw{k}_we,", f"  output logic [31:0] mw{k}_d,",
                       f"  output logic [{BANK_ADDR_W - 1}:0] mr{k}_addr,", f"  output logic        mr{k}_ce,",
                       f"  input  logic [31:0] mr{k}_q,"]
    status_mux = "\n".join(f"      {i:2d}: st_word = {expr};  // {name}" for i, (name, expr) in enumerate(STATUS))
    c = {k: f"4'h{v:X}" for k, v in CMD.items()}
    memory_side = (bridge_side if memory == "bridge" else direct_side)()
    module = MODULES[memory]
    return f"""// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// GENERATED by examples/minitpu-rtl/gen_shim.py -- edit the generator, not this file.
//
// Allo's shim around MiniTPU's unmodified minitpu_core (seam K): a command stream in, a status stream out,
// and {BANKS} word-interleaved RAMs ({BANKS} x 2^{BANK_ADDR_W} x 32 bit = 32 MiB at byte 0x{MEM_BASE:08X}), each with a write
// and a read MemPort on the same array, serving the core's device-memory credit pipe -- {MEMORY_NOTE[memory]}.
// Every pin is <= 32 bits, as PR #48's RTLModule requires.
`timescale 1ns/1ps

module {module} (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] cmd_data,
  input  logic        cmd_valid,
  output logic        cmd_ready,
  output logic [31:0] st_data,
  output logic        st_valid,
  input  logic        st_ready,
{chr(10).join(bank_ports)}
  output logic        done
);
  localparam int NSTAT = {NSTAT};
  localparam logic [29:0] BASE_WORD = 30'h{base_word:X};   // MEM_BASE / 32
  localparam logic [29:0] DEPTH = 30'h{depth:X};

  // ---------------------------------------------------------------- the core, unmodified
  logic        core_start, core_done, core_illegal;
  logic        iram_we;
  logic [11:0] iram_addr;
  logic [127:0] iram_din;
  logic [31:0] program_id;
  logic [127:0] kernel_arg;
  logic        dm_req_valid, dm_req_ready, dm_req_we;
  logic [28:0] dm_req_addr;
  logic [7:0]  dm_req_len;
  logic [255:0] dm_req_wdata;
  logic [31:0] dm_req_wstrb;
  logic        dm_rsp_valid, dm_rsp_ready, dm_rsp_last;
  logic [255:0] dm_rsp_data;
  logic [1:0]  dm_rsp_resp;
  logic        dma_err, dma_rsp_seen;
  logic [7:0]  dma_err_channel;
  logic [1:0]  dma_err_cause, dma_err_resp;
  logic [31:0] beat_count, overlap_stat, perf_cnt_cycles, perf_cnt_instrs;
  logic [1:0]  dma_channel_done;

  minitpu_core u_core (
    .clk, .rst_n,
    .start(core_start), .done(core_done), .illegal_op_o(core_illegal),
    .instr_write_en(iram_we), .iram_addr, .dma_iram_din(iram_din),
    .program_id_csr(program_id), .kernel_arg_csr(kernel_arg),
    .dm_req_valid, .dm_req_ready, .dm_req_we, .dm_req_addr, .dm_req_len, .dm_req_wdata, .dm_req_wstrb,
    .dm_rsp_valid, .dm_rsp_ready, .dm_rsp_data, .dm_rsp_last, .dm_rsp_resp,
    .dma_err_o(dma_err), .dma_err_channel_o(dma_err_channel), .dma_err_cause_o(dma_err_cause),
    .dma_err_resp_o(dma_err_resp), .dma_rsp_seen_o(dma_rsp_seen), .beat_count_o(beat_count),
    .overlap_stat_o(overlap_stat), .perf_cnt_cycles_o(perf_cnt_cycles), .perf_cnt_instrs_o(perf_cnt_instrs),
    .dma_channel_done_o(dma_channel_done)
  );

  // ---------------------------------------------------------------- command side
  localparam logic [2:0] C_IDLE = 3'd0, C_IRAM = 3'd1, C_VALUE = 3'd2, C_RUN = 3'd3, C_STAT = 3'd4, C_END = 3'd5;
  logic [2:0]  cs_q;
  logic [3:0]  value_sel_q;          // 0..3 kernel_arg, 4 program_id, 8 cycle limit
  logic [1:0]  word_q;               // word within the bundle being loaded
  logic [95:0] bundle_lo_q;
  logic [15:0] bundles_left_q;
  logic [11:0] iram_ptr_q;
  logic [31:0] maxcyc_q, run_cyc_q, bundles_q;
  logic [4:0]  st_idx_q;
  logic        start_q, done_seen_q, timeout_q, core_illegal_seen_q, bad_cmd_q;
  logic        iram_we_q;
  logic [11:0] iram_addr_q;
  logic [127:0] iram_din_q;

  wire cmd_fire = cmd_valid & cmd_ready;
  wire [3:0] op = cmd_data[31:28];
  assign cmd_ready  = (cs_q == C_IDLE) || (cs_q == C_IRAM) || (cs_q == C_VALUE);
  assign core_start = start_q;
  assign iram_we    = iram_we_q;
  assign iram_addr  = iram_addr_q;
  assign iram_din   = iram_din_q;
  assign done       = (cs_q == C_END);

  logic [31:0] st_word;
  always_comb begin
    unique case (st_idx_q)
{status_mux}
      default: st_word = 32'd0;
    endcase
  end
  assign st_valid = (cs_q == C_STAT);
  assign st_data  = st_word;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cs_q <= C_IDLE; value_sel_q <= '0; word_q <= '0; bundle_lo_q <= '0; bundles_left_q <= '0;
      iram_ptr_q <= '0; maxcyc_q <= '0; run_cyc_q <= '0; bundles_q <= '0; st_idx_q <= '0;
      start_q <= 1'b0; done_seen_q <= 1'b0; timeout_q <= 1'b0; core_illegal_seen_q <= 1'b0; bad_cmd_q <= 1'b0;
      iram_we_q <= 1'b0; iram_addr_q <= '0; iram_din_q <= '0; program_id <= '0; kernel_arg <= '0;
    end else begin
      start_q   <= 1'b0;
      iram_we_q <= 1'b0;
      if (core_illegal) core_illegal_seen_q <= 1'b1;
      unique case (cs_q)
        C_IDLE: if (cmd_fire) begin
          unique case (op)
            {c["NOP"]}: ;
            {c["IRAM"]}: begin
              iram_ptr_q <= cmd_data[27:16];
              bundles_left_q <= cmd_data[15:0];
              word_q <= '0;
              if (cmd_data[15:0] != 16'd0) cs_q <= C_IRAM;
            end
            {c["CSR"]}: begin value_sel_q <= cmd_data[3:0]; cs_q <= C_VALUE; end
            {c["MAXCYC"]}: begin value_sel_q <= 4'd8; cs_q <= C_VALUE; end
            {c["START"]}: begin
              start_q <= 1'b1; run_cyc_q <= '0; done_seen_q <= 1'b0; timeout_q <= 1'b0; cs_q <= C_RUN;
            end
            {c["END"]}: cs_q <= C_END;
            default: bad_cmd_q <= 1'b1;
          endcase
        end
        C_IRAM: if (cmd_fire) begin
          word_q <= word_q + 2'd1;
          unique case (word_q)
            2'd0: bundle_lo_q[31:0]  <= cmd_data;
            2'd1: bundle_lo_q[63:32] <= cmd_data;
            2'd2: bundle_lo_q[95:64] <= cmd_data;
            default: begin
              iram_we_q <= 1'b1;
              iram_addr_q <= iram_ptr_q;
              iram_din_q <= {{cmd_data, bundle_lo_q}};
              iram_ptr_q <= iram_ptr_q + 12'd1;
              bundles_q <= bundles_q + 32'd1;
              bundles_left_q <= bundles_left_q - 16'd1;
              if (bundles_left_q == 16'd1) cs_q <= C_IDLE;
            end
          endcase
        end
        C_VALUE: if (cmd_fire) begin
          unique case (value_sel_q)
            4'd0: kernel_arg[31:0]   <= cmd_data;
            4'd1: kernel_arg[63:32]  <= cmd_data;
            4'd2: kernel_arg[95:64]  <= cmd_data;
            4'd3: kernel_arg[127:96] <= cmd_data;
            4'd4: program_id <= cmd_data;
            4'd8: maxcyc_q <= cmd_data;
            default: bad_cmd_q <= 1'b1;
          endcase
          cs_q <= C_IDLE;
        end
        C_RUN: begin
          run_cyc_q <= run_cyc_q + 32'd1;
          if (core_done) begin done_seen_q <= 1'b1; st_idx_q <= '0; cs_q <= C_STAT; end
          else if (maxcyc_q != 32'd0 && run_cyc_q >= maxcyc_q) begin
            timeout_q <= 1'b1; st_idx_q <= '0; cs_q <= C_STAT;
          end
        end
        C_STAT: if (st_ready) begin
          st_idx_q <= st_idx_q + 5'd1;
          if (st_idx_q == 5'(NSTAT - 1)) cs_q <= C_IDLE;
        end
        default: ;  // C_END holds done
      endcase
    end
  end

{memory_side}endmodule
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a checked-in shim differs")
    args = parser.parse_args()
    stale = []
    for memory, out in OUTS.items():
        text = emit(memory)
        if args.check:
            if not out.is_file() or out.read_text() != text:
                stale.append(out.name)
        else:
            out.write_text(text)
            print(f"wrote {out}")
    if args.check:
        if stale:
            print(f"{stale} differ from the generator's output; run {pathlib.Path(__file__).name}", file=sys.stderr)
            return 1
        print(f"{', '.join(o.name for o in OUTS.values())}: up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
