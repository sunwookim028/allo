# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 references for the sequencer's front: fetch and loop control.

Cycle models (``rtl.py`` ``trace`` convention, every output ``"pre"``) of

* ``sequencer_fetch_queue`` alone (``fetch_queue_trace``), and behind
  ``sequencer_iram`` as ``sequencer.sv`` wires them (``iram=True``);
* ``sequencer_loop_ctrl`` + ``sequencer_loop_buffer`` (``loop_ctrl_trace``).

Each takes ``{port: uint64[n, nwords]}`` and returns ``(resp, reason,
events)`` as the U2/U3 trace references do. Unreset storage carries a taint:
the fetch queue's ``fifo_mem``, the IRAM array and its read register, the loop
buffer's ``mem``; a slot that shows such a value before it was written is
masked ``uninit``. Rows with ``rst_n = 0`` are masked ``reset`` (the async
reset may or may not see an edge on the first row, by the random initial
value of ``rst_n``).

Both models are *implementations*, transcribed register by register; what
of them is a contract is said in ``units/fetch.py`` and ``units/loop_ctrl.py``.
"""

import numpy as np

from examples.minitpu.harness import rtl

UNINIT = "uninit"
BUNDLE_W = 128
INSTR_ADDR_W = 12
STACK_DEPTH = 8
LEVEL_SEL_W = 3
LB_CAP = 24


def _cols(cmd):
    return {p: rtl.unpack(v) for p, v in cmd.items()}


def _finish(out, why, widths):
    resp = {p: rtl.pack(out[p], widths[p]) for p in out}
    reason = {p: np.array(why[p], dtype=object) for p in why}
    return resp, reason


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

FQ_PORTS = dict(pause="pause_i", flush="pc_flush_i", restart="pc_restart_addr_i", pop="bundle_pop_i",
                rd_addr="rd_addr_o", rd_valid="rd_addr_valid_o", valid="bundle_valid_o",
                data="bundle_data_o", addr="bundle_addr_o", empty="empty_o", full="full_o")
FETCH_PORTS = dict(pause="pause", flush="pc_flush", restart="restart_addr", pop="bundle_pop",
                   rd_addr="rd_addr", rd_valid="rd_addr_valid", valid="bundle_valid",
                   data="bundle_data", addr="bundle_addr", empty="empty", full="full")


def fetch_queue_trace(cmd, addr_w=INSTR_ADDR_W, depth=4, iram=False):
    """``sequencer_fetch_queue`` (and, with ``iram``, ``sequencer_iram`` in
    front of it, the ``u4_fetch`` wrapper)."""
    c = _cols(cmd)
    n = len(c["rst_n"])
    am = (1 << addr_w) - 1
    cm = (1 << (depth + 1).bit_length()) - 1  # $clog2(DEPTH + 1) bits
    im = depth - 1  # IDX_W'(...) for a power-of-two DEPTH
    P = FETCH_PORTS if iram else FQ_PORTS
    widths = {P["rd_addr"]: addr_w, P["rd_valid"]: 1, P["valid"]: 1, P["data"]: BUNDLE_W,
              P["addr"]: addr_w, P["empty"]: 1, P["full"]: 1}
    out = {p: [0] * n for p in widths}
    why = {p: [""] * n for p in widths}
    ev = {}
    next_addr = req_pending = req_addr = count = 0
    mem = [None] * depth  # (addr, data|None) or None: fifo_mem is unreset
    iram_mem = {}
    rd_data = None  # sequencer_iram's read register (unreset)

    def event(k):
        ev[k] = ev.get(k, 0) + 1

    for t in range(n):
        rst = c["rst_n"][t] == 0
        if rst:
            next_addr = req_pending = req_addr = count = 0
        empty = count == 0
        room = count < depth - 1
        rd_valid = int((not c[P["pause"]][t]) and room)
        rd_addr = next_addr
        head = mem[0]
        o = {P["rd_addr"]: rd_addr, P["rd_valid"]: rd_valid, P["valid"]: int(not empty),
             P["empty"]: int(empty), P["full"]: int(count == depth),
             P["data"]: (head[1] or 0) if head else 0, P["addr"]: head[0] if head else 0}
        for p, v in o.items():
            out[p][t] = v
            if rst:
                why[p][t] = "reset"
        if not rst and (head is None or head[1] is None):
            why[P["data"]][t] = UNINIT
        if not rst and head is None:
            why[P["addr"]][t] = UNINIT
        # ---- rising edge ------------------------------------------------
        bram = rd_data if iram else c["bram_data_i"][t]
        if iram:  # the read register takes the old contents, then this edge's write lands
            rd_data = iram_mem.get(rd_addr)
            if c["instr_write_en"][t]:
                iram_mem[c["iram_addr"][t]] = c["dma_iram_din"][t]
        if rst:
            continue
        flush = bool(c[P["flush"]][t])
        push = bool(req_pending) and not flush
        pop = bool(c[P["pop"]][t]) and not empty
        if c[P["pop"]][t] and empty:
            event("pop empty")
        if flush:
            event("flush")
        if push and iram and bram is None:
            event("fetch of unwritten IRAM")
        entry = (req_addr, bram)
        if flush:
            next_addr, req_pending = c[P["restart"]][t] & am, 0
            count = 0
        else:
            if pop:
                mem = mem[1:] + [mem[-1]]
                if push:
                    mem[(count - 1) & im] = entry
            elif push:
                mem[count & im] = entry
            req_pending, req_addr = rd_valid, rd_addr
            if rd_valid:
                next_addr = (rd_addr + 1) & am
            count = (count + int(push) - int(pop)) & cm
    resp, reason = _finish(out, why, widths)
    return resp, reason, ev


# ---------------------------------------------------------------------------
# loop control + loop buffer
# ---------------------------------------------------------------------------

LOOP_OUT_W = dict(branch_taken=1, branch_target=INSTR_ADDR_W, iv_by_level=32 * STACK_DEPTH,
                  iv_tos=32, level_tos=LEVEL_SEL_W, lb_reset=1, lb_capture_en=1, lb_replay_en=1,
                  lb_replay_idx=5, lb_capture_overflow=1, replay_data=BUNDLE_W, depth=4,
                  overflow=1, underflow=1)


class LoopCtrl:
    """One ``u4_loop_ctrl`` (loop control + loop buffer), stepped a cycle at a
    time: ``step(row)`` takes one row of inputs (Python ints), returns that
    row's ``"pre"`` outputs and their reasons, then takes the rising edge."""

    def __init__(self):
        self.ev = {}
        self._reset()
        self.lbmem = [None] * LB_CAP  # unreset LUTRAM

    def _reset(self):
        self.fr_bs, self.fr_iv, self.fr_hi, self.fr_step = ([0] * STACK_DEPTH for _ in range(4))
        self.sp = self.warm = self.cap_count = self.body_len = self.ridx = self.invalid = self.wr_ptr = 0

    def event(self, k):
        self.ev[k] = self.ev.get(k, 0) + 1

    def step(self, r):
        M32, AM = (1 << 32) - 1, (1 << INSTR_ADDR_W) - 1
        rst = r["rst_n"] == 0
        if rst:
            self._reset()
        fr_bs, fr_iv, fr_hi, fr_step = self.fr_bs, self.fr_iv, self.fr_hi, self.fr_step
        sp = self.sp
        bv, endv, iss = bool(r["loop_begin_valid"]), bool(r["loop_end_valid"]), bool(r["bundle_issued"])
        lo, hi, step = r["lo"], r["hi"], r["step"]
        tos = 0 if sp == 0 else sp - 1
        iv_next = (fr_iv[tos] + fr_step[tos]) & M32
        cont = endv and sp != 0 and iv_next < fr_hi[tos]
        exit_ = endv and sp != 0 and not cont
        skip = bv and bool(r["may_skip"]) and hi <= lo
        push = bv and not skip
        cap_en = sp != 0 and not self.warm and not self.invalid
        rep_en = sp != 0 and bool(self.warm)
        cap_ovf = cap_en and iss and self.wr_ptr == LB_CAP
        completes = cap_en and endv and not cap_ovf
        cold_cont = cont and not self.warm
        warm_exit = exit_ and bool(self.warm)
        taken = cold_cont or warm_exit or skip
        if skip:
            target = r["body_start"] + r["skip"]
        elif cold_cont:
            target = fr_bs[tos]
        else:
            target = fr_bs[tos] + self.body_len + 1
        ridx = self.ridx
        rd = self.lbmem[ridx] if ridx < LB_CAP else None
        o = dict(branch_taken=int(taken), branch_target=target & AM,
                 iv_by_level=sum(v << (32 * k) for k, v in enumerate(fr_iv)), iv_tos=fr_iv[tos],
                 level_tos=tos & 7, lb_reset=int(push or exit_ or skip), lb_capture_en=int(cap_en),
                 lb_replay_en=int(rep_en), lb_replay_idx=ridx, lb_capture_overflow=int(cap_ovf),
                 replay_data=rd or 0, depth=sp, overflow=int(push and sp == STACK_DEPTH),
                 underflow=int(endv and sp == 0))
        why = {p: ("reset" if rst else "") for p in o}
        if not rst and rd is None:
            why["replay_data"] = "idx>=CAP" if ridx >= LB_CAP else UNINIT
        for flag, name in ((bv and endv, "begin+end together"), (o["overflow"], "stack overflow"),
                           (o["underflow"], "stack underflow"), (cap_ovf, "capture overflow"),
                           (skip, "skip")):
            if flag and not rst:
                self.event(name)
        if rst:
            return o, why
        # ---- rising edge ------------------------------------------------
        if push and sp != STACK_DEPTH:
            fr_bs[sp], fr_iv[sp], fr_hi[sp], fr_step[sp] = r["body_start"], lo, hi, step
            self.sp = sp + 1
        elif endv and sp != 0:
            if cont:
                fr_iv[tos] = iv_next
            else:
                self.sp = sp - 1
        if push:
            self.warm = self.cap_count = self.ridx = self.invalid = 0
        elif exit_ or skip:
            self.warm, self.cap_count, self.ridx, self.invalid = 0, 0, 0, 1
        else:
            nc = self.cap_count
            if cap_en and iss:
                nc = (self.cap_count + 1) & 31
            if completes:
                self.body_len, self.warm = self.cap_count, 1
            self.cap_count = nc
            if rep_en and iss:
                self.ridx = 0 if endv else (ridx + 1) & 31
        lb_reset = push or exit_ or skip
        # sequencer_loop_buffer (capture_valid_i = bundle_issued)
        if not lb_reset and cap_en and iss and self.wr_ptr < LB_CAP:
            self.lbmem[self.wr_ptr] = r["capture_data"]
        if lb_reset:
            self.wr_ptr = 0
        elif cap_en and iss and self.wr_ptr < LB_CAP:
            self.wr_ptr += 1
        return o, why


def loop_ctrl_trace(cmd):
    """``sequencer_loop_ctrl`` + ``sequencer_loop_buffer`` (``u4_loop_ctrl``)."""
    c = _cols(cmd)
    n = len(c["rst_n"])
    out = {p: [0] * n for p in LOOP_OUT_W}
    why = {p: [""] * n for p in LOOP_OUT_W}
    m = LoopCtrl()
    for t in range(n):
        o, w = m.step({p: c[p][t] for p in c})
        for p in LOOP_OUT_W:
            out[p][t], why[p][t] = o[p], w[p]
    resp, reason = _finish(out, why, LOOP_OUT_W)
    return resp, reason, m.ev
