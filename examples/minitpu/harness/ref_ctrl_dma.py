# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: references for MiniTPU's DMA (``src/core/dma/``).

``dma_addr_gen`` is one line of arithmetic: ``word_addr = base + row*stride``
truncated to 32 bits (``row`` 14 bits, zero-extended).

``dma`` is a **cycle model** of ``dma.sv``'s observable behaviour: every
register of the per-channel descriptor FSMs, the issue engine (``I_IDLE`` /
``I_LD_REQ`` / ``I_ST_RD`` / ``I_ST_RDW`` / ``I_ST_REQ``), the credit counter,
the in-order order FIFO, the landing FIFO, the watchdog and the sticky error and
statistics registers, transcribed statement by statement with the
non-blocking last-assignment-wins order of ``dma.sv``'s single ``always_ff``.
Its outputs are combinational functions of that state and the cycle's inputs
(``rtl.py``'s ``"pre"`` sampling); ``rst_n`` is asynchronous (low in a row
resets the state before the row is sampled).

What the RTL leaves undefined: the landing FIFO's payload RAM is not reset
(``lnd_mem``), so ``vmem_wr_data`` / ``vmem_wr_ptr`` are masked while the FIFO
is empty (reason ``landing empty``); everything else is reset and defined.
Row 0 is masked (``power-up``): ``rst_n`` starts low, so the asynchronous
reset has seen no edge of ``rst_n`` or ``clk`` before row 0 is sampled.

Events counted: requests, response beats (and dropped stale ones), VMEM
writes/reads, completions, errors, timeouts, and ``desc_stall`` (a descriptor
held against a non-IDLE channel: ``dma.sv``'s sim-only assertion fires on its
4th consecutive cycle).
"""

M32 = (1 << 32) - 1
IDLE, PENDING, ACTIVE, DONE = 0, 1, 2, 3
I_IDLE, I_LD_REQ, I_ST_RD, I_ST_RDW, I_ST_REQ = 0, 1, 2, 3, 4

ROW_BITS = 14
DRAM_BEAT_ADDR_W = 29
VMEM_DMA_READ_LATENCY = 2  # vpu_pkg.sv
PAGE_WORDS = 4096 // 32  # 128 DM words per 4 KiB page

INPUTS = [("rst_n", 1), ("desc_valid", 1), ("desc_is_store", 1), ("desc_channel", 1),
          ("desc_vmem_row", 14), ("desc_rows", ROW_BITS), ("desc_cols", 7), ("desc_base", 32),
          ("desc_stride", 32), ("clear_channel_done", 2), ("dm_req_ready", 1),
          ("dm_rsp_valid", 1), ("dm_rsp_data", 256), ("dm_rsp_last", 1), ("dm_rsp_resp", 2),
          ("vmem_rd_data", 256), ("vmem_gnt", 1)]
OUTPUTS = [("desc_accept", 1), ("dma_channel_done", 2), ("dma_idle", 1), ("dm_req_valid", 1),
           ("dm_req_we", 1), ("dm_req_addr", DRAM_BEAT_ADDR_W), ("dm_req_len", 8),
           ("dm_req_wdata", 256), ("dm_req_wstrb", 32), ("dm_rsp_ready", 1), ("vmem_wr_en", 1),
           ("vmem_wr_data", 256), ("vmem_wr_ptr", 16), ("vmem_rd_en", 1), ("vmem_rd_ptr", 16),
           ("dma_err", 1), ("dma_err_channel", 1), ("dma_err_cause", 2), ("dma_err_resp", 2),
           ("dma_rsp_seen", 1), ("beat_count_o", 32), ("overlap_stat_o", 32)]


def addr_gen(base, row, stride):
    """``dma_addr_gen.sv``: ``base + {18'h0, row} * stride``, 32 bits."""
    return (int(base) + (int(row) & 0x3FFF) * int(stride)) & M32


def clog2(x):
    return max(0, (int(x) - 1).bit_length())


class DmaModel:
    """``dma.sv`` at one parameter set (module docstring)."""

    def __init__(self, channels=2, outstanding=2, timeout=1024, landing=8):
        assert channels == 2, "the trace ports are sized for two channels"
        self.N, self.OUT, self.TO, self.LND = channels, outstanding, timeout, landing
        self.CRED_M = (1 << clog2(outstanding + 1)) - 1
        self.OUTC_M = (1 << clog2(outstanding + 1)) - 1
        self.OPTR = max(1, clog2(outstanding))
        self.LNDC_M = (1 << clog2(landing + 1)) - 1
        self.TO_M = (1 << (1 if timeout < 2 else clog2(timeout))) - 1
        self.lnd_mem = [None] * landing  # unreset payload: None = never written
        self.reset()
        self.stall = 0

    def reset(self):
        N = self.N
        self.st = [IDLE] * N
        self.store = [0] * N
        self.vmem = [0] * N
        self.last_row = [0] * N
        self.base = [0] * N
        self.stride = [0] * N
        self.next_row = [0] * N
        self.all_issued = [0] * N
        self.bif = [0] * N
        self.done = 0
        self.eng, self.iss, self.rr = I_IDLE, 0, 0
        self.st_beat, self.st_data, self.rd_lat = 0, 0, 0
        self.credit = self.OUT
        self.ord_ch = [0] * self.OUT
        self.ord_vmem = [0] * self.OUT
        self.ord_store = [0] * self.OUT
        self.ord_head = self.ord_tail = self.ord_count = 0
        self.lnd_head = self.lnd_tail = self.lnd_count = 0
        self.rsp_beat = 0
        self.to_cnt = 0
        self.err = self.err_ch = self.err_cause = self.err_resp = 0
        self.rsp_seen = 0
        self.beats = 0
        self.ovl_max = self.ovl_cyc = 0
        self.stall = 0

    # ---- combinational view -------------------------------------------------
    def _issue_view(self):
        i = self.iss
        cur = self.next_row[i]
        agu = addr_gen(self.base[i], cur, self.stride[i])
        logical = agu & ((1 << DRAM_BEAT_ADDR_W) - 1)
        contiguous = self.stride[i] == 1
        left_m1 = (self.last_row[i] - cur) & ((1 << ROW_BITS) - 1)
        page_off = logical & (PAGE_WORDS - 1)
        to_bound_m1 = (PAGE_WORDS - 1) - page_off
        left_sat = 0xFF if left_m1 >> 8 else left_m1 & 0xFF
        blen = min(left_sat, to_bound_m1) if contiguous else 0
        last = blen == left_m1
        vbase = (self.vmem[i] + cur) & 0xFFFF
        return cur, logical, blen, last, vbase

    def _lnd_head(self):
        e = self.lnd_mem[self.lnd_head]
        return e if e is not None else (0, 0, 0, 0, 0, 0)

    def comb(self, x):
        """Outputs for this cycle's inputs ``x`` (a dict), plus internals the edge needs."""
        cur, logical, blen, last, vbase = self._issue_view()
        o = {}
        if self.eng == I_LD_REQ:
            req_valid = int(self.credit != 0)
        elif self.eng == I_ST_REQ:
            req_valid = int(self.st_beat != 0 or self.credit != 0)
        else:
            req_valid = 0
        o["dm_req_valid"] = req_valid
        o["dm_req_we"] = int(self.eng == I_ST_REQ)
        o["dm_req_addr"] = logical
        o["dm_req_len"] = blen
        o["dm_req_wdata"] = self.st_data
        o["dm_req_wstrb"] = (1 << 32) - 1
        rsp_ready = 1 if self.ord_count == 0 else int(self.lnd_count < self.LND)
        o["dm_rsp_ready"] = rsp_ready
        lnd_ne = self.lnd_count != 0
        hch, hstore, hdest, hdata, hr, hlast = self._lnd_head()
        wr_en = int(lnd_ne and not hstore and hr == 0)
        o["vmem_wr_en"] = wr_en
        o["vmem_wr_ptr"] = hdest
        o["vmem_wr_data"] = hdata
        rd_en = int(self.eng == I_ST_RD and not wr_en)
        o["vmem_rd_en"] = rd_en
        o["vmem_rd_ptr"] = (self.vmem[self.iss] + cur + self.st_beat) & 0xFFFF
        o["desc_accept"] = int(x["desc_valid"] and self.st[x["desc_channel"]] == IDLE)
        o["dma_channel_done"] = self.done
        idle = self.ord_count == 0 and self.lnd_count == 0
        if any(s in (PENDING, ACTIVE) for s in self.st):
            idle = False
        o["dma_idle"] = int(idle)
        o["dma_err"] = self.err
        o["dma_err_channel"] = self.err_ch
        o["dma_err_cause"] = self.err_cause
        o["dma_err_resp"] = self.err_resp
        o["dma_rsp_seen"] = self.rsp_seen
        o["beat_count_o"] = self.beats
        o["overlap_stat_o"] = (self.ovl_cyc << 8) | self.ovl_max
        internal = dict(cur=cur, blen=blen, last=last, vbase=vbase, lnd_ne=lnd_ne,
                        head=(hch, hstore, hdest, hdata, hr, hlast))
        return o, internal

    # ---- the clock edge -----------------------------------------------------
    def edge(self, x, o, it, ev):
        """Advance one rising edge (``rst_n`` high), ``dma.sv``'s always_ff in order."""
        N = self.N
        # snapshot of every old value the block reads
        st0 = list(self.st)
        store0 = list(self.store)
        all_issued0 = list(self.all_issued)
        bif0 = list(self.bif)
        err0 = self.err
        cur, blen, last, vbase = it["cur"], it["blen"], it["last"], it["vbase"]
        hch, hstore, hdest, hdata, hr, hlast = it["head"]
        req_fire = o["dm_req_valid"] and x["dm_req_ready"]
        rsp_fire = x["dm_rsp_valid"] and o["dm_rsp_ready"]
        real_rsp = rsp_fire and self.ord_count != 0
        lnd_pop = it["lnd_ne"] and (hstore or hr != 0 or (hr == 0 and x["vmem_gnt"]))
        issue_push, issue_ch = False, self.iss
        ord_pop = False
        lnd_push = False
        drain_last = lnd_pop and hlast
        drain_ch = hch
        timeout = False

        new = {}  # NBA targets: (name, index) -> value; last write wins

        def nba(name, val, idx=None):
            new[(name, idx)] = val

        # stats / events
        if req_fire:
            ev["dm_req"] = ev.get("dm_req", 0) + 1
        if rsp_fire and not real_rsp:
            ev["rsp_dropped"] = ev.get("rsp_dropped", 0) + 1
        if o["vmem_wr_en"] and x["vmem_gnt"]:
            ev["vmem_write"] = ev.get("vmem_write", 0) + 1
        if o["vmem_rd_en"]:
            ev["vmem_read"] = ev.get("vmem_read", 0) + 1

        # 1) descriptor accept
        dc = x["desc_channel"]
        if x["desc_valid"] and st0[dc] == IDLE:
            nba("st", PENDING, dc)
            nba("store", x["desc_is_store"], dc)
            nba("vmem", x["desc_vmem_row"], dc)
            nba("last_row", x["desc_rows"], dc)
            nba("base", x["desc_base"], dc)
            nba("stride", x["desc_stride"], dc)
            nba("next_row", 0, dc)
            nba("all_issued", 0, dc)
            ev["desc"] = ev.get("desc", 0) + 1
        # 2) W1C clear
        done = self.done
        for s in range(N):
            if (x["clear_channel_done"] >> s) & 1:
                done &= ~(1 << s)
                if st0[s] == DONE:
                    nba("st", IDLE, s)
        # 3) issue engine
        if self.eng == I_IDLE:
            pick, any_p = 0, False
            for i in range(N - 1, -1, -1):
                idx = (self.rr + i) % N
                if st0[idx] == PENDING:
                    any_p, pick = True, idx
            if any_p:
                nba("iss", pick)
                nba("st", ACTIVE, pick)
                nba("rr", (pick + 1) % N)
                nba("st_beat", 0)
                nba("eng", I_ST_RD if store0[pick] else I_LD_REQ)
        elif self.eng == I_LD_REQ:
            if req_fire:
                issue_push = True
                if last:
                    nba("all_issued", 1, self.iss)
                    nba("eng", I_IDLE)
                else:
                    nba("next_row", (cur + blen + 1) & 0x3FFF, self.iss)
                    nba("eng", I_LD_REQ)
        elif self.eng == I_ST_RD:
            if o["vmem_rd_en"] and x["vmem_gnt"]:
                nba("rd_lat", 0)
                nba("eng", I_ST_RDW)
        elif self.eng == I_ST_RDW:
            if self.rd_lat == VMEM_DMA_READ_LATENCY - 1:
                nba("st_data", x["vmem_rd_data"])
                nba("eng", I_ST_REQ)
            else:
                nba("rd_lat", (self.rd_lat + 1) & 3)
        elif self.eng == I_ST_REQ:
            if req_fire:
                if self.st_beat == 0:
                    issue_push = True
                if self.st_beat == blen:
                    if last:
                        nba("all_issued", 1, self.iss)
                        nba("eng", I_IDLE)
                    else:
                        nba("next_row", (cur + blen + 1) & 0x3FFF, self.iss)
                        nba("st_beat", 0)
                        nba("eng", I_ST_RD)
                else:
                    nba("st_beat", (self.st_beat + 1) & 0xFF)
                    nba("eng", I_ST_RD)
        # 4) response accept
        if x["dm_rsp_valid"]:
            nba("rsp_seen", 1)
        if real_rsp:
            nba("lnd_tail", 0 if self.lnd_tail == self.LND - 1 else self.lnd_tail + 1)
            lnd_push = True
            if x["dm_rsp_last"]:
                ord_pop = True
                nba("ord_head", 0 if self.ord_head == self.OUT - 1 else self.ord_head + 1)
                nba("rsp_beat", 0)
            else:
                nba("rsp_beat", (self.rsp_beat + 1) & 0xFF)
            ev["rsp"] = ev.get("rsp", 0) + 1
            # lnd_mem write (separate always_ff, rst_n && real_rsp)
            lnd_entry = (self.ord_ch[self.ord_head], self.ord_store[self.ord_head],
                         (self.ord_vmem[self.ord_head] + self.rsp_beat) & 0xFFFF,
                         x["dm_rsp_data"], x["dm_rsp_resp"], x["dm_rsp_last"])
            lnd_slot = self.lnd_tail
        # 5) order FIFO push
        if issue_push:
            nba("ord_ch", issue_ch, self.ord_tail)
            nba("ord_vmem", vbase, self.ord_tail)
            nba("ord_store", store0[self.iss], self.ord_tail)
            nba("ord_tail", 0 if self.ord_tail == self.OUT - 1 else self.ord_tail + 1)
        nba("ord_count", (self.ord_count + int(issue_push) - int(ord_pop)) & self.OUTC_M)
        # 6) landing FIFO pop
        if lnd_pop:
            nba("lnd_head", 0 if self.lnd_head == self.LND - 1 else self.lnd_head + 1)
            if hr != 0:
                nba("err", 1)
                if not err0:
                    nba("err_ch", hch)
                    nba("err_cause", 1)
                    nba("err_resp", hr)
                ev["resp_error"] = ev.get("resp_error", 0) + 1
        nba("lnd_count", (self.lnd_count + int(lnd_push) - int(lnd_pop)) & self.LNDC_M)
        bc_load = lnd_pop and not hstore and hr == 0
        bc_store = req_fire and o["dm_req_we"]
        nba("beats", (self.beats + int(bc_load) + int(bc_store)) & M32)
        # 6c) overlap
        nonidle = sum(1 for s in st0 if s in (PENDING, ACTIVE))
        if nonidle > self.ovl_max:
            nba("ovl_max", nonidle & 0xFF)
        if nonidle >= 2 and self.ovl_cyc != (1 << 24) - 1:
            nba("ovl_cyc", self.ovl_cyc + 1)
        # 7) credit
        nba("credit", (self.credit - int(issue_push) + int(ord_pop)) & self.CRED_M)
        # 8) per-channel in flight + completion
        for s in range(N):
            inc = issue_push and issue_ch == s
            dec = drain_last and drain_ch == s
            nba("bif", (bif0[s] + int(inc) - int(dec)) & self.OUTC_M, s)
            if dec and not inc and bif0[s] == 1 and all_issued0[s]:
                done |= 1 << s
                nba("st", DONE, s)
                ev["channel_done"] = ev.get("channel_done", 0) + 1
        # 9) watchdog
        if self.ord_count == 0 or x["dm_rsp_valid"] or req_fire:
            nba("to_cnt", 0)
        else:
            if self.TO != 0 and self.to_cnt == ((self.TO - 1) & self.TO_M):
                timeout = True
            else:
                nba("to_cnt", (self.to_cnt + 1) & self.TO_M)
        # 10) watchdog flush
        if timeout:
            ev["timeout"] = ev.get("timeout", 0) + 1
            nba("err", 1)
            if not err0:
                nba("err_ch", self.ord_ch[self.ord_head])
                nba("err_cause", 2)
                nba("err_resp", 0)
            for s in range(N):
                nba("bif", 0, s)
                if st0[s] in (PENDING, ACTIVE):
                    done |= 1 << s
                    nba("st", DONE, s)
            nba("credit", self.OUT)
            for k in ("ord_head", "ord_tail", "ord_count", "lnd_head", "lnd_tail", "lnd_count",
                      "rsp_beat", "st_beat", "to_cnt"):
                nba(k, 0)
            nba("eng", I_IDLE)
        # the sim-only descriptor-stall check (its own always_ff)
        if x["desc_valid"] and not o["desc_accept"] and st0[dc] != IDLE:
            if self.stall == 3:
                ev["desc_stall_assert"] = ev.get("desc_stall_assert", 0) + 1
            if self.stall != 4:
                self.stall += 1
        else:
            self.stall = 0
        # commit
        self.done = done & 3
        for (name, idx), val in new.items():
            if idx is None:
                setattr(self, name, val)
            else:
                getattr(self, name)[idx] = val
        if real_rsp:
            self.lnd_mem[lnd_slot] = lnd_entry


def dma_trace(cmd, n, params):
    """Run the cycle model over a command trace (``{port: list of ints}``).

    Returns ``({port: [int]}, {port: [reason]}, events)``.
    """
    m = DmaModel(**params)
    names = [p for p, _ in INPUTS]
    out = {p: [] for p, _ in OUTPUTS}
    why = {p: [] for p, _ in OUTPUTS}
    ev = {}
    for t in range(n):
        x = {p: cmd[p][t] for p in names}
        if not x["rst_n"]:
            m.reset()
        o, it = m.comb(x)
        empty = m.lnd_count == 0
        for p, _ in OUTPUTS:
            out[p].append(o[p])
            r = ""
            if t == 0:
                # an asynchronous reset acts on an edge of rst_n or clk; row 0 has had neither
                r = "power-up"
            elif p in ("vmem_wr_data", "vmem_wr_ptr") and (empty or m.lnd_mem[m.lnd_head] is None):
                r = "landing empty"
            why[p].append(r)
        if x["rst_n"]:
            m.edge(x, o, it, ev)
    return out, why, ev
