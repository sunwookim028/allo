# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``sequencer.sv`` as one unit -- the bundle-issue reference.

A cycle model of the whole sequencer composed from the Phase 0 references of
its parts, so the composite is held to the same models its units are:
``ref_ctrl_front`` (IRAM + fetch queue, loop control + loop buffer) and
``ref_ctrl_decode`` (decoder, address resolve, scalar AGU, descriptor
adapter, VPU adapter), plus the top FSM of ``sequencer.sv:254-550`` (IDLE,
RUN, D_WAIT, FLUSH_WAIT, HALT_DRAIN; the ``delay`` hold; ``run_accept``).

What it predicts per cycle: ``vpu_ctrl_o`` (the 85-bit command to the VPU),
the resolved DMA descriptor, ``done``, ``clear_channel_done``,
``bundle_issued_o``, the state flags and ``x_issue_address``; and, as
events, the sim-only schedule checks of ``sequencer.sv:407-432`` and
``:561-575`` through the same write-port calendar ``ref_ctrl_wb`` measured
on ``vpu.sv`` (so a fired assertion is predicted, not just counted).
"""

import numpy as np

from examples.minitpu.harness import ref_ctrl_decode as D
from examples.minitpu.harness import ref_ctrl_wb as WB
from examples.minitpu.harness import rtl
from examples.minitpu.harness.ref_ctrl_front import LoopCtrl

INSTR_ADDR_W = 12
AM = (1 << INSTR_ADDR_W) - 1
S_IDLE, S_RUN, S_D_WAIT, S_FLUSH_WAIT, S_HALT_DRAIN = range(5)
C_LOOP_END, C_WAIT_CHANNEL, C_HALT = 1, 2, 3
UNINIT = "uninit IRAM"

OUTS = [("vpu_ctrl_o", 85), ("bundle_issued_o", 1), ("done", 1), ("clear_channel_done", 2),
        ("state_is_idle_o", 1), ("state_is_run_o", 1), ("x_issue_address", 12),
        ("dma_desc_valid", 1), ("dma_desc_is_store", 1), ("dma_desc_channel", 1),
        ("dma_desc_vmem_row", 14), ("dma_desc_rows", 14), ("dma_desc_cols", 7),
        ("dma_desc_base", 32), ("dma_desc_stride", 32)]
PAYLOAD = ("vpu_ctrl_o", "x_issue_address")  # follow the fetch head even when nothing issues

# the per-class write-port claim, as sequencer.sv:327-335 books it
_W_OF = {"alu": WB.W["alu"], "sfu": WB.W["sfu"], "reduce": WB.W["reduce"],
         "lane_reduce": WB.W["lane_reduce"], "load": WB.W["load"]}


class Fetch:
    """sequencer_iram + sequencer_fetch_queue (as ``ref_ctrl_front``, stepped)."""

    def __init__(self, depth=4):
        self.depth = depth
        self.iram = {}
        self.rd_data = None
        self.reset()
        self.mem = [None] * depth

    def reset(self):
        self.next_addr = self.req_pending = self.req_addr = self.count = 0

    def head(self):
        h = self.mem[0] if self.count else None
        return (self.count != 0, h[1] if h else None, h[0] if h else 0)

    def edge(self, r, flush, restart, pop, rst):
        rd_valid = self.count < self.depth - 1  # pause_i is tied low (sequencer.sv:503)
        rd_addr = self.next_addr
        bram = self.rd_data
        self.rd_data = self.iram.get(rd_addr)
        if r["instr_write_en"]:
            self.iram[r["iram_addr"]] = r["dma_iram_din"]
        if rst:
            self.reset()
            return
        push = bool(self.req_pending) and not flush
        pop = pop and self.count != 0
        entry = (self.req_addr, bram)
        if flush:
            self.next_addr, self.req_pending, self.count = restart & AM, 0, 0
            return
        if pop:
            self.mem = self.mem[1:] + [self.mem[-1]]
            if push:
                self.mem[(self.count - 1) & (self.depth - 1)] = entry
        elif push:
            self.mem[self.count & (self.depth - 1)] = entry
        self.req_pending, self.req_addr = int(rd_valid), rd_addr
        if rd_valid:
            self.next_addr = (rd_addr + 1) & AM
        self.count += int(push) - int(pop)


class Sequencer:
    def __init__(self):
        self.fetch = Fetch()
        self.loop = LoopCtrl()
        self.sagu = D.ScalarAgu()
        self.desc = D.DescAdapter()
        self.ev = {}
        self.reset()

    def reset(self):
        self.state, self.done, self.flush_mask, self.clear, self.delay = S_IDLE, 0, 0, 0, 0
        # sim-only calendar (sequencer.sv:280-405)
        self.reserved = 0  # bit k: the write port is booked k cycles from now
        self.release = [None] * (WB.W["reduce"] + 1)  # wb_release_{valid,addr}_q
        self.pending = set()

    def event(self, k):
        self.ev[k] = self.ev.get(k, 0) + 1

    def step(self, r):
        rst = r["rst_n"] == 0
        if rst:
            self.reset()
            self.fetch.reset()
            self.sagu.reset()
            self.desc.reset()
        lp = self.loop
        replay_en = lp.sp != 0 and bool(lp.warm)
        fq_valid, fq_data, fq_addr = self.fetch.head()
        if replay_en:
            word = lp.lbmem[lp.ridx] if lp.ridx < len(lp.lbmem) else None
            bundle_valid = True
        else:
            word, bundle_valid = fq_data, fq_valid
        undefined = word is None
        f = D.decode(word or 0)
        delay_hold = self.delay != 0
        run_accept = self.state == S_RUN and bundle_valid and not delay_hold
        if run_accept and undefined:
            self.event("issued an unwritten IRAM word")
        run_is_dma = run_accept and bool(f["d.valid"])
        # scalar AGU read ports (pre-edge registers)
        kargs = r["kernel_arg_csr"]
        srow = {"kernel_arg_csr_i": kargs, "rd_sel_d_base_i": f["d.base_sreg"],
                "rd_sel_d_stride_i": f["d.stride_sreg"], "rd_loop_bound_from_arg_i": f["l.hi_from_arg"],
                "rd_sel_loop_bound_i": f["l.hi_idx"]}
        so = self.sagu.outputs(srow)
        # loop control (+ its edge)
        lrow = {"rst_n": r["rst_n"], "loop_begin_valid": int(run_accept and bool(f["l.valid"])),
                "body_start": (fq_addr + 1) & AM, "lo": f["l.lo"], "step": f["l.step"],
                "hi": so["rd_data_loop_bound_o"] if f["l.hi_from_reg"] else f["l.hi"],
                "may_skip": f["l.hi_from_reg"], "skip": f["l.skip"] & AM,
                "loop_end_valid": int(run_accept and not run_is_dma and f["c.subop"] == C_LOOP_END),
                "bundle_issued": int(run_accept), "capture_data": fq_data}
        lo, _ = lp.step(lrow)
        iv = D.ivs_of(lo["iv_by_level"])
        x_row = D.agu_resolve(iv, f["x.literal"], f["x.agu_valid"], f["x.agu_level"], f["x.agu_shift"])
        sub = lambda pre, lay: D.pack(lay, {n: f[f"{pre}.{n}"] for n, _ in lay})  # noqa: E731
        ctrl = D.vpu_adapter(sub("v", D.V_SLOT), sub("m", D.M_SLOT), sub("x", D.X_SLOT),
                             int(run_accept), x_row)
        do = self.desc.outputs({"desc_accept_i": r["dma_desc_accept"]})
        entry = (self.state == S_IDLE) and bool(r["start"])
        flush = bool(lo["branch_taken"]) or entry
        restart = ((r["program_id_csr"] & 0xF) << 8) & AM if entry else lo["branch_target"]
        o = {"vpu_ctrl_o": ctrl, "bundle_issued_o": int(run_accept), "done": self.done,
             "clear_channel_done": self.clear, "state_is_idle_o": int(self.state == S_IDLE),
             "state_is_run_o": int(self.state == S_RUN), "x_issue_address": x_row,
             "dma_desc_valid": do["desc_valid_o"], "dma_desc_is_store": do["desc_is_store_o"],
             "dma_desc_channel": do["desc_channel_o"], "dma_desc_vmem_row": do["desc_vmem_row_o"],
             "dma_desc_rows": do["desc_rows_o"], "dma_desc_cols": 0, "dma_desc_base": do["desc_base_o"],
             "dma_desc_stride": do["desc_stride_o"]}
        why = {p: ("reset" if rst else "") for p, _ in OUTS}
        if not rst and undefined:
            for p in PAYLOAD:
                why[p] = UNINIT
            if run_accept:
                for p, _ in OUTS:
                    why[p] = UNINIT
        # ---- rising edge ------------------------------------------------
        self.fetch.edge(r, flush, restart, run_accept and not replay_en, rst)
        if rst:
            return o, why
        self._calendar(f, r, run_accept and not undefined)
        self.sagu.edge({"s_valid_i": int(run_accept and bool(f["s.valid"])), "s_op_i": f["s.op"],
                        "s_rd_i": f["s.rd"], "s_rs_i": f["s.rs"], "s_use_iv_i": f["s.use_iv"],
                        "s_level_i": f["s.level"], "s_imm_i": f["s.imm"], "kernel_arg_csr_i": kargs,
                        "iv_flat_i": lo["iv_by_level"]})
        self.desc.edge({"start_i": int(run_is_dma), "d_i": sub("d", D.D_SLOT),
                        "sreg_rd_base_i": so["rd_data_d_base_o"], "sreg_rd_stride_i": so["rd_data_d_stride_o"],
                        "desc_accept_i": r["dma_desc_accept"]})
        d_done = do["done_o"]
        st = self.state
        if st == S_IDLE:
            self.delay = 0
        elif run_accept:
            self.delay = f["delay"]
        elif st == S_RUN and delay_hold:
            self.delay -= 1
        self.done, self.clear = 0, 0
        if st == S_IDLE:
            if r["start"]:
                self.state = S_RUN
        elif st == S_RUN:
            if run_accept:
                if run_is_dma:
                    self.state = S_D_WAIT
                elif f["c.subop"] == C_WAIT_CHANNEL:
                    self.flush_mask = f["c.operand"] & 3
                    self.state = S_FLUSH_WAIT
                elif f["c.subop"] == C_HALT:
                    self.state = S_HALT_DRAIN
        elif st == S_D_WAIT:
            if d_done:
                self.state = S_RUN
        elif st == S_FLUSH_WAIT:
            if (r["dma_channel_done"] & self.flush_mask) == self.flush_mask:
                self.clear = self.flush_mask
                self.state = S_RUN
        elif st == S_HALT_DRAIN:
            if r["dma_idle"]:
                self.done, self.state = 1, S_IDLE
        return o, why

    def _calendar(self, f, r, issue):
        """The sim-only checker of ``sequencer.sv:277-432, 561-575``, register
        for register: the write-port reservation shift register, the release
        tag pipeline (one address per slot: a second release landing in the
        same slot overwrites the first, so that VREG stays pending), and the
        pending set. Checks on the issue cycle, then the edge."""
        e_cls, cand = None, 0
        writes, reads = [], set()
        if f["v.alu_valid"]:
            e_cls, e_vd = "alu", f["v.alu_vd"]
        elif f["v.sfu_valid"]:
            e_cls, e_vd = "sfu", f["v.sfu_vd"]
        elif f["v.reduce_valid"]:
            e_cls, e_vd = ("lane_reduce" if f["v.reduce_lane"] else "reduce"), f["v.reduce_vd"]
        claims = []
        if e_cls:
            reads.add(f["v.raddr_a"])
            if e_cls == "alu" and f["v.alu_op"] != 3:
                reads.add(f["v.raddr_b"])
            writes.append((e_vd, _W_OF[e_cls]))
            claims.append(_W_OF[e_cls])
        if f["x.valid"] and f["x.op"]:
            reads.add(f["x.vreg_idx"])
        if f["x.valid"] and not f["x.op"]:
            writes.append((f["x.vreg_idx"], _W_OF["load"]))
            claims.append(_W_OF["load"])
        if f["m.subop"] == 1:
            reads |= {(f["m.reg_idx"] + k) & 31 for k in range(4)}
        elif f["m.subop"] == 2:
            reads.add(f["m.reg_idx"])
        elif f["m.subop"] == 3:
            writes.append((f["m.reg_idx"], WB.W["mpop"]))
            claims.append(WB.W["mpop"])
        for w in claims:
            cand |= 1 << w
        if issue:
            if reads & self.pending:
                self.event("assert: RAW on a VREG in flight")
            if cand & self.reserved:
                self.event("assert: write-port collision")
            if r["matrix_busy_i"] and f["m.subop"] != 0:
                self.event("assert: M command while its engine is busy")
            if f["c.subop"] == C_HALT and self.pending:
                self.event("assert: halt with VREG(s) in flight")
            if len(claims) != len(set(claims)):
                self.event("assert: same-bundle write-port collision")
            if {v for v, _ in writes} & self.pending:
                self.event("assert: WAW on a VREG in flight")
        # ---- edge ----
        rel0 = self.release[0]
        self.release = self.release[1:] + [None]
        self.reserved >>= 1
        if rel0 is not None:
            self.pending.discard(rel0)
        if issue:
            self.reserved |= cand >> 1
            self.pending |= {v for v, _ in writes}
            for v, w in writes:  # E, then X, then M: a later one overwrites a shared slot
                self.release[w - 1] = v

def issue_trace(cmd):
    """Run the model over a ``sequencer`` command trace."""
    cols = {p: rtl.unpack(c) for p, c in cmd.items()}
    n = len(cols["rst_n"])
    m = Sequencer()
    out = {p: [] for p, _ in OUTS}
    reason = {p: np.array([""] * n, dtype=object) for p, _ in OUTS}
    for t in range(n):
        o, why = m.step({p: cols[p][t] for p in cols})
        for p, _ in OUTS:
            out[p].append(o[p])
            reason[p][t] = why[p]
    ev = dict(m.ev)
    for k, v in m.loop.ev.items():
        ev[f"loop: {k}"] = v
    return {p: rtl.pack(out[p], w) for p, w in OUTS}, reason, ev
