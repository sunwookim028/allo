# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: the sequencer loop closed -- five stateful units composed, cycle-locked.

The first multi-unit control composition of the MiniTPU ladder (README D-23,
D-24; ``dev/records/minitpu/u4_seqloop_2026-10-08.rst``). Track B's issue unit
(``units/seq_issue.py`` ``locked``) took the fetch head, loop control's
``iv_by_level`` and the scalar AGU's read ports as side columns replayed from
Phase 0's references; here they come from track A's units, composed:

``loader``  the IRAM's host port (``sequencer_iram.sv`` port A)
``fq``      ``sequencer_fetch_queue.sv`` + the IRAM read register (F1, the
            ``fetch_d12`` body), owner of ``iram.fetch``
``lcap``    ``sequencer_loop_buffer.sv``'s write side, owner of ``lb.cap``
``lctl``    ``sequencer_loop_ctrl.sv`` (L1, the ``loop_ctrl_d12`` body), owner
            of ``lb.replay``
``sagu``    ``sequencer_scalar_agu.sv`` (S1): four SREGs, the ``S_LAT`` pipe,
            three read ports answered from committed state (no bypass)
``issue``   ``sequencer.sv``'s top FSM + A1 (``dma_desc_adapter.sv``) + C1
            (``decode_*``, ``resolve``, ``adapt_*``), as ``seq_issue.locked``
``vrx`` ``xrx`` ``mrx``  the VPU side of D-23's three slot commands (harness
            receivers: they record the tokens in order)

The IRAM (4,096 x 128) and the loop buffer (``LB_CAP`` x 128) are D-12
memories, unreset (D-14), one owner per port. Every geometry number is read
from the D-20 record (``control_geometry.SHIPPED``).

**One iteration of every unit is one clock cycle** (D-24). The RTL's
combinational paths between the parts become one token per channel per
iteration, in the order the RTL's combinational cone runs:

1. state first: ``fq`` puts its head (``f_head``, ``f_data``, ``f_cap``);
   ``lctl`` puts its registered state (``l_repen``, the replay word ``l_rep``,
   ``iv_by_level`` on ``l_iv``/``l_ivs``); ``lcap`` puts ``c_atcap``;
2. ``issue`` gets the head, selects the replay word or the queue head,
   decodes, and sends the scalar AGU's read selects with the S op (``i_sa``);
3. ``sagu`` answers from committed SREGs (``s_b``, ``s_s``, ``s_lb``) -- the
   one same-cycle ROUND TRIP of the sequencer (issue -> AGU -> issue);
4. ``issue`` sends loop control its begin/end/issued and the bound
   (``i_lc``), the fetch queue its pop and the entry flush (``i_fq``), and the
   VPU its V/X/M commands (``cv``/``cx``/``cm``, D-23: one token per valid
   command, in its issue cycle);
5. ``lctl`` decides the branch and sends ``l_fq`` (flush, target) and
   ``l_cap`` (buffer reset, capture enable, issued);
6. every unit takes its edge.

No unit gets a token that depends on its own put of the same iteration
except through that order, so the per-iteration graph is acyclic and depth 1
would already be deadlock-free; the exchanges are declared depth 2 and the
commands ``QD`` (D-23: >= 2, F-B3). In the RTL every exchange of steps 1-5
is a WIRE (combinational, same cycle); in this composition it is a Stream on
both backends -- the finding the record classifies (S-1): the region's
links are Streams because a Wire unit port is not wired (``stream_ports.rst``)
and a csim Wire is not cycle-locked (limitation 22).

The fetch queue's flush is the RTL's synchronous clear inside ``fq`` (it owns
its FIFO as state, F1): no Stream is flushed, so D-25 is not needed here and
no count-based drop either (the record says why).
"""

from __future__ import annotations

import numpy as np

from allo.compose import Architecture, Channel, Memory, Port, unit
from allo.ir.types import UInt, int32, uint1  # noqa: F401  (names the bodies use)

from examples.minitpu.harness import ref_ctrl_decode as D
from examples.minitpu.template.control_geometry import SHIPPED as G
from examples.minitpu.units.agu_resolve import resolve  # noqa: F401  (C1, called by issue)
from examples.minitpu.units.scalar_agu import scalar_result  # noqa: F401  (S1's result mux)
from examples.minitpu.units.seq_decoder import (  # noqa: F401  (C1, called by issue)
    decode_c, decode_d, decode_l, decode_m, decode_s, decode_v, decode_x)
from examples.minitpu.units.vpu_adapter import adapt_m, adapt_v, adapt_x  # noqa: F401

# ---------------------------------------------------------------------------------
# Literal bit positions in the bodies (README D-17: a slice bound may not be a
# parameter). They are the sequencer_pkg layouts; held to them here at import.
# ---------------------------------------------------------------------------------


def _offsets(layout):
    out, o = {}, sum(w for _, w in layout)
    for n, w in layout:
        o -= w
        out[n] = (o, o + w)
    return out


_LITERALS = {
    "D_SLOT": {"valid": (56, 57), "is_store": (55, 56), "vmem_address": (41, 53), "rows": (29, 41),
               "has_disp": (28, 29), "disp": (4, 28), "base_sreg": (2, 4), "stride_sreg": (0, 2)},
    "C_SLOT": {"subop": (4, 7), "operand": (0, 4)},
    "L_SLOT": {"valid": (44, 45), "hi_from_reg": (43, 44), "hi_from_arg": (42, 43), "hi_idx": (40, 42),
               "lo": (36, 40), "step": (32, 36), "hi": (16, 32), "skip": (0, 16)},
    "S_SLOT": {"valid": (35, 36), "op": (32, 35), "rd": (30, 32), "rs": (28, 30), "use_iv": (27, 28),
               "level": (24, 27), "imm": (0, 24)},
    "X_SLOT": {"agu_shift": (0, 4), "agu_level": (4, 7), "agu_valid": (7, 8), "literal": (8, 20)},
}
for _slot, _fields in _LITERALS.items():
    _o = _offsets(getattr(D, _slot))
    for _f, _b in _fields.items():
        assert _o[_f] == _b, f"{_slot}.{_f}: the bodies slice {_b}, sequencer_pkg says {_o[_f]}"
assert D.D_SLOT[2] == ("channel_sel", 2), "DMA_CHANNEL_SEL_W: the body keeps bit 53 (one channel bit)"


# ---------------------------------------------------------------------------------
# The units
# ---------------------------------------------------------------------------------


@unit(memories=("WE", "WA", "WD", "iram.host"), parameters=("N",))
def loader(we: uint1[N], wa: UInt(12)[N], wd: UInt(32)[N, 4], mem):
    """``sequencer_iram.sv`` port A: the host write (``dma_iram_din``)."""
    for t in range(N):
        d: UInt(128) = 0
        d[0:32] = wd[t, 0]
        d[32:64] = wd[t, 1]
        d[64:96] = wd[t, 2]
        d[96:128] = wd[t, 3]
        a: int32 = wa[t]
        if we[t]:
            mem[a] = d


@unit(memories=("RST_F", "iram.fetch"), reads=("i_fq", "l_fq"), writes=("f_head", "f_data", "f_cap"),
      parameters=("N", "AW"))
def fq(rst: uint1[N], mem):
    """``sequencer_fetch_queue.sv`` behind the IRAM's read register (F1).
    Head first (state), then this cycle's pop and flush, then the edge. The
    flush (a taken branch or the entry) is the queue's own synchronous clear."""
    rd_reg: UInt(128) = 0  # sequencer_iram's read register (T-2: on Stream links the port's L=1 is this)
    fq_addr: UInt(AW)[4] = 0
    fq_data: UInt(128)[4] = 0
    next_addr: UInt(AW) = 0
    req_pending: uint1 = 0
    req_addr: UInt(AW) = 0
    count: UInt(3) = 0
    for t in range(N):
        live: uint1 = rst[t]
        if live == 0:
            next_addr = 0
            req_pending = 0
            req_addr = 0
            count = 0
        empty: uint1 = count == 0
        rd_valid: uint1 = 0
        if count < 3:  # pause_i tied low (sequencer.sv:503)
            rd_valid = 1
        head: UInt(16) = fq_addr[0]
        if empty == 0:
            head[12] = 1
        f_head.put(head)
        f_data.put(fq_data[0])
        f_cap.put(fq_data[0])
        # the edge: port B reads the fetch address every cycle
        bram: UInt(128) = rd_reg
        ir: int32 = next_addr
        rd_reg = mem[ir]
        ci: UInt(16) = i_fq.get()
        cl: UInt(16) = l_fq.get()
        if live:
            entry: uint1 = ci[12]
            taken: uint1 = cl[12]
            if entry or taken:
                if entry:
                    next_addr = ci[0:12]
                else:
                    next_addr = cl[0:12]
                req_pending = 0
                count = 0
            else:
                push: uint1 = req_pending
                pop: uint1 = ci[13] & (1 - empty)
                if pop:
                    for j in range(3):
                        fq_addr[j] = fq_addr[j + 1]
                        fq_data[j] = fq_data[j + 1]
                if push:
                    slot: UInt(2) = count
                    if pop:
                        slot = count - 1
                    si: int32 = slot
                    fq_addr[si] = req_addr
                    fq_data[si] = bram
                count = count + push - pop
                req_pending = rd_valid
                req_addr = next_addr
                if rd_valid:
                    next_addr = next_addr + 1


@unit(memories=("RST_C", "lb.cap"), reads=("f_cap", "l_cap"), writes=("c_atcap",), parameters=("N", "CAP", "CW"))
def lcap(rst: uint1[N], mem):
    """``sequencer_loop_buffer.sv``'s write side: the first pass's capture of
    the fetch head. ``at_cap`` is state, offered before the controls."""
    wr_ptr: UInt(CW) = 0
    for t in range(N):
        live: uint1 = rst[t]
        if live == 0:
            wr_ptr = 0
        at_cap: uint1 = wr_ptr == CAP
        c_atcap.put(at_cap)
        k: UInt(4) = l_cap.get()
        d: UInt(128) = f_cap.get()
        lb_reset: uint1 = k[2]
        cap_en: uint1 = k[1]
        iss: uint1 = k[0]
        we: uint1 = 0
        if live and lb_reset == 0 and cap_en and iss and wr_ptr < CAP:
            we = 1
        iw: int32 = 0
        if wr_ptr < CAP:
            iw = wr_ptr
        if we:
            mem[iw] = d
        if live:
            if lb_reset:
                wr_ptr = 0
            elif we:
                wr_ptr = wr_ptr + 1


@unit(memories=("RST_L", "lb.replay"), reads=("i_lc", "c_atcap"),
      writes=("l_repen", "l_rep", "l_iv", "l_ivs", "l_fq", "l_cap"),
      parameters=("N", "AW", "SD", "CAP", "CW", "IW", "SPW"))
def lctl(rst: uint1[N], mem):
    """``sequencer_loop_ctrl.sv`` (L1): the frame stack, the replay read, and
    the branch -- combinational in this cycle's begin/end from ``issue``."""
    fr_bs: UInt(AW)[SD] = 0
    fr_iv: UInt(32)[SD] = 0
    fr_hi: UInt(16)[SD] = 0
    fr_step: UInt(8)[SD] = 0
    sp: UInt(SPW) = 0
    warm: uint1 = 0
    cap_count: UInt(CW) = 0
    body_len: UInt(CW) = 0
    ridx: UInt(IW) = 0
    invalid: uint1 = 0
    for t in range(N):
        live: uint1 = rst[t]
        if live == 0:
            for k in range(SD):
                fr_bs[k] = 0
                fr_iv[k] = 0
                fr_hi[k] = 0
                fr_step[k] = 0
            sp = 0
            warm = 0
            cap_count = 0
            body_len = 0
            ridx = 0
            invalid = 0
        # ---- registered outputs (state): replay, iv_by_level ----
        rep_en: uint1 = 0
        if sp != 0 and warm == 1:
            rep_en = 1
        ra: int32 = 0
        if ridx < CAP:
            ra = ridx
        l_repen.put(rep_en)
        l_rep.put(mem[ra])
        for k in range(SD):
            l_iv.put(fr_iv[k])
        for k in range(SD):
            l_ivs.put(fr_iv[k])
        # ---- this cycle's controls from issue (combinational in the RTL) ----
        c: UInt(64) = i_lc.get()
        bs: UInt(AW) = c[0:12]
        lo: UInt(8) = c[12:16]
        step: UInt(8) = c[16:20]
        hi: UInt(16) = c[20:36]
        skip_i: UInt(AW) = c[36:48]
        bv: uint1 = c[48]
        endv: uint1 = c[49]
        iss: uint1 = c[50]
        ms: uint1 = c[51]
        at_cap: uint1 = c_atcap.get()
        tos: UInt(SPW) = 0
        if sp != 0:
            tos = sp - 1
        it: int32 = tos
        step32: UInt(32) = fr_step[it]
        iv_next: UInt(32) = fr_iv[it] + step32
        hi32: UInt(32) = fr_hi[it]
        on: uint1 = endv & (sp != 0)
        cont: uint1 = 0
        if on and iv_next < hi32:
            cont = 1
        exit_: uint1 = on & (1 - cont)
        lo16: UInt(16) = lo
        skip: uint1 = 0
        if bv and ms and hi <= lo16:
            skip = 1
        push: uint1 = bv & (1 - skip)
        lb_reset: uint1 = push | exit_ | skip
        cap_en: uint1 = 0
        if sp != 0 and warm == 0 and invalid == 0:
            cap_en = 1
        cap_ovf: uint1 = cap_en & iss & at_cap
        completes: uint1 = cap_en & endv & (1 - cap_ovf)
        cold_cont: uint1 = cont & (1 - warm)
        warm_exit: uint1 = exit_ & warm
        target: UInt(AW) = fr_bs[it] + body_len + 1
        if skip:
            target = bs + skip_i
        elif cold_cont:
            target = fr_bs[it]
        br: UInt(16) = target
        if cold_cont or warm_exit or skip:
            br[12] = 1
        l_fq.put(br)
        kc: UInt(4) = 0
        kc[2] = lb_reset
        kc[1] = cap_en
        kc[0] = iss
        l_cap.put(kc)
        # ---- the edge ----
        if live:
            isp: int32 = sp
            if push and sp != SD:
                fr_bs[isp] = bs
                fr_iv[isp] = lo
                fr_hi[isp] = hi
                fr_step[isp] = step
                sp = sp + 1
            elif endv and sp != 0:
                if cont:
                    fr_iv[it] = iv_next
                else:
                    sp = sp - 1
            if push:
                warm = 0
                cap_count = 0
                ridx = 0
                invalid = 0
            elif exit_ or skip:
                warm = 0
                cap_count = 0
                ridx = 0
                invalid = 1
            else:
                old_count: UInt(CW) = cap_count
                if cap_en and iss:
                    cap_count = cap_count + 1
                if completes:
                    body_len = old_count
                    warm = 1
                if rep_en and iss:
                    if endv:
                        ridx = 0
                    else:
                        ridx = ridx + 1


@unit(memories=("RST_S", "KA", "SREG"), reads=("i_sa", "l_ivs"), writes=("s_b", "s_s", "s_lb"),
      parameters=("N", "SD", "SL"), calls=("scalar_result",))
def sagu(rst: uint1[N], ka: UInt(32)[N, 4], sreg_o: UInt(32)[N, 4]):
    """``sequencer_scalar_agu.sv`` (S1): the read ports answer this cycle's
    selects from committed SREGs (no bypass); a write lands ``S_LAT`` edges
    after its issue through the pipe written as data."""
    sreg: UInt(32)[4] = 0
    pv: uint1[SL] = 0
    prd: UInt(2)[SL] = 0
    pdata: UInt(32)[SL] = 0
    pprod: UInt(32)[SL] = 0
    pmac: uint1[SL] = 0
    for t in range(N):
        live: uint1 = rst[t]
        if live == 0:
            for k in range(4):
                sreg[k] = 0
            for k in range(SL):
                pv[k] = 0
                prd[k] = 0
                pdata[k] = 0
                pprod[k] = 0
                pmac[k] = 0
        for k in range(4):
            sreg_o[t, k] = sreg[k]  # A5: the 2-D output stored first
        iv: UInt(32)[SD] = 0
        for k in range(SD):
            iv[k] = l_ivs.get()
        q: UInt(64) = i_sa.get()
        # ---- read ports (combinational in the RTL: issue -> here -> issue) ----
        ib: int32 = q[36:38]
        i_s: int32 = q[38:40]
        il: int32 = q[41:43]
        s_b.put(sreg[ib])
        s_s.put(sreg[i_s])
        raw: UInt(32) = sreg[il]
        if q[40]:
            raw = ka[t, il]
        bound: UInt(32) = raw[0:16]
        if raw[16:32] != 0:
            bound = 0xFFFF
        s_lb.put(bound)
        # ---- the edge ----
        if live:
            sv: uint1 = q[35]
            sop: UInt(3) = q[32:35]
            irs: int32 = q[28:30]
            ilv: int32 = q[24:27]
            imm: UInt(24) = q[0:24]
            operand: UInt(32) = sreg[irs]
            if q[27]:
                operand = iv[ilv]
            prod: UInt(32) = operand * imm
            pre: UInt(32) = scalar_result(sop, sreg[irs], imm, ka[t, irs])
            for j in range(SL - 1):
                kk: int32 = SL - 1 - j
                pv[kk] = pv[kk - 1]
                prd[kk] = prd[kk - 1]
                pdata[kk] = pdata[kk - 1]
                pprod[kk] = pprod[kk - 1]
                pmac[kk] = pmac[kk - 1]
            pv[0] = sv
            prd[0] = q[30:32]
            pdata[0] = pre
            pprod[0] = prod
            pmac[0] = sop == 3
            if pv[SL - 1]:
                fin: UInt(32) = pdata[SL - 1]
                if pmac[SL - 1]:
                    fin = fin + pprod[SL - 1]
                iw: int32 = prd[SL - 1]
                sreg[iw] = fin


@unit(memories=("RST_I", "START", "CHD", "DIDLE", "ACC", "PID", "ISS", "DONE", "CLR", "SIDLE", "SRUN", "XADDR",
                "DV", "DST", "DCH", "DROW", "DROWS", "DBASE", "DSTR", "VV", "XV", "MV"),
      reads=("f_head", "f_data", "l_repen", "l_rep", "l_iv", "s_b", "s_s", "s_lb"),
      writes=("i_fq", "i_lc", "i_sa", "cv", "cx", "cm"),
      parameters=("N", "SD"),
      calls=("decode_v", "decode_m", "decode_x", "decode_d", "decode_c", "decode_l", "decode_s", "resolve",
             "adapt_v", "adapt_x", "adapt_m"))
def issue(rst: uint1[N], start: uint1[N], chd: UInt(8)[N], didle: uint1[N], acc: uint1[N], pid: UInt(8)[N],
          iss: uint1[N], done: uint1[N], clr: UInt(8)[N], sidle: uint1[N], srun: uint1[N], xaddr: UInt(16)[N],
          dv: uint1[N], dst: uint1[N], dch: uint1[N], drow: UInt(16)[N], drows: UInt(16)[N],
          dbase: UInt(32)[N], dstr: UInt(32)[N], vv: UInt(8)[N], xv: uint1[N], mv: UInt(8)[N]):
    """``sequencer.sv:254-550`` (I1) + ``dma_desc_adapter.sv`` (A1) + C1, as
    ``seq_issue.locked``; the head, ``iv_by_level`` and the SREG reads now
    arrive from ``fq``/``lctl``/``sagu`` instead of side columns."""
    st: UInt(8) = 0  # IDLE 0, RUN 1, D_WAIT 2, FLUSH_WAIT 3, HALT_DRAIN 4
    dq: UInt(8) = 0  # the delay counter
    done_q: uint1 = 0
    clr_q: UInt(8) = 0
    fmask: UInt(8) = 0
    di: uint1 = 0  # A1: dma_desc_adapter's registers
    d_st: uint1 = 0
    d_ch: uint1 = 0
    d_va: UInt(16) = 0
    d_rows: UInt(16) = 0
    d_base: UInt(32) = 0
    d_str: UInt(32) = 0
    kv: UInt(32) = 0
    kx: UInt(32) = 0
    km: UInt(32) = 0
    for t in range(N):
        r: uint1 = rst[t]
        if r == 0:  # asynchronous reset: the state is reset before the outputs
            st = 0
            dq = 0
            done_q = 0
            clr_q = 0
            fmask = 0
            di = 0
            d_st = 0
            d_ch = 0
            d_va = 0
            d_rows = 0
            d_base = 0
            d_str = 0
        # ---- what the parts hold this cycle ----
        hd: UInt(16) = f_head.get()
        fw: UInt(128) = f_data.get()
        rep: uint1 = l_repen.get()
        rw: UInt(128) = l_rep.get()
        ivs: UInt(32)[SD] = 0
        for k in range(SD):
            ivs[k] = l_iv.get()
        w: UInt(128) = fw
        h: uint1 = hd[12]
        if rep:  # sequencer.sv: the loop buffer's replay outranks the queue head
            w = rw
            h = 1
        v: UInt(46) = decode_v(w)
        m: UInt(8) = decode_m(w)
        x: UInt(27) = decode_x(w)
        d: UInt(57) = decode_d(w)
        c: UInt(7) = decode_c(w)
        lf: UInt(45) = decode_l(w)
        sf: UInt(36) = decode_s(w)
        dl: UInt(8) = w[15:22]
        ra: uint1 = 0
        if st == 1 and h != 0 and dq == 0:
            ra = 1
        d_valid: uint1 = d[56]
        isdma: uint1 = ra & d_valid
        csub: UInt(3) = c[4:7]
        # ---- the scalar AGU's read ports: selects out, data back (same cycle) ----
        qa: UInt(64) = 0
        qa[0:36] = sf
        qa[35] = sf[35] & ra  # s_valid_i = run_accept & s.valid
        qa[36:38] = d[2:4]  # rd_sel_d_base_i
        qa[38:40] = d[0:2]  # rd_sel_d_stride_i
        qa[40] = lf[42]  # rd_loop_bound_from_arg_i
        qa[41:43] = lf[40:42]  # rd_sel_loop_bound_i
        i_sa.put(qa)
        b: UInt(32) = s_b.get()
        sr: UInt(32) = s_s.get()
        lb: UInt(32) = s_lb.get()
        # ---- loop control's inputs (same cycle) ----
        ql: UInt(64) = 0
        bs: UInt(12) = hd[0:12]
        ql[0:12] = bs + 1  # body_start: the head's address + 1
        ql[12:16] = lf[36:40]  # lo
        ql[16:20] = lf[32:36]  # step
        if lf[43]:  # loop.begin.r: the bound from the scalar AGU's read port
            ql[20:36] = lb[0:16]
        else:
            ql[20:36] = lf[16:32]
        ql[36:48] = lf[0:12]  # skip
        ql[48] = lf[44] & ra  # loop_begin_valid
        if ra == 1 and isdma == 0 and csub == 1:
            ql[49] = 1  # loop_end_valid
        ql[50] = ra  # bundle_issued
        ql[51] = lf[43]  # may_skip
        i_lc.put(ql)
        # ---- the fetch queue's pop and the entry flush (same cycle) ----
        go: uint1 = start[t]
        qf: UInt(16) = 0
        if st == 0 and go == 1:
            qf[12] = 1  # entry: flush to the program's slot
        p4: UInt(8) = pid[t]
        qf[8:12] = p4[0:4]
        qf[13] = ra & (1 - rep)  # bundle_pop: a replayed bundle is not popped
        i_fq.put(qf)
        cd: UInt(8) = chd[t]
        idl: uint1 = didle[t]
        ac: uint1 = acc[t]
        # ---- outputs (before the edge) ----
        iss[t] = ra
        done[t] = done_q
        clr[t] = clr_q
        sidle[t] = st == 0
        srun[t] = st == 1
        lev: UInt(3) = x[4:7]
        lit: UInt(12) = x[8:20]
        agv: uint1 = x[7]
        shf: UInt(4) = x[0:4]
        xa: UInt(12) = resolve(ivs, lit, agv, lev, shf)
        xaddr[t] = xa
        vc: UInt(47) = adapt_v(v, ra)
        xc: UInt(20) = adapt_x(x, ra, xa)
        mc: UInt(18) = adapt_m(m, ra)
        vval: UInt(8) = 0
        vval[4] = vc[36]  # alu_valid
        vval[3] = vc[35]  # txin_valid
        vval[2] = vc[32]  # txout_valid
        vval[1] = vc[15]  # sfu_valid
        vval[0] = vc[7]  # reduce_valid
        vv[t] = vval
        if vval != 0:  # D-23: a V command is a token, sent in its issue cycle
            vtok: UInt(64) = 0
            vtok[42:47] = vval[0:5]
            vtok[0:7] = vc[0:7]
            vtok[7:14] = vc[8:15]
            vtok[14:23] = vc[16:25]
            vtok[23:30] = vc[25:32]
            vtok[30:32] = vc[33:35]
            vtok[32:42] = vc[37:47]
            cv.put(vtok)
            kv = kv + 1
        xval: uint1 = xc[19]
        xv[t] = xval
        if xval != 0:
            xtok: UInt(32) = 0
            xtok[19] = 1
            xtok[0:19] = xc[0:19]
            cx.put(xtok)
            kx = kx + 1
        mval: UInt(8) = 0
        mval[2] = mc[17]  # vmatload_valid
        mval[1] = mc[11]  # vmatpush_valid
        mval[0] = mc[5]  # vmatpop_valid
        mv[t] = mval
        if mval != 0:
            mtok: UInt(32) = 0
            mtok[15:18] = mval[0:3]
            mtok[10:15] = mc[12:17]
            mtok[5:10] = mc[6:11]
            mtok[0:5] = mc[0:5]
            cm.put(mtok)
            km = km + 1
        dv[t] = di  # A1 outputs
        dst[t] = d_st
        dch[t] = d_ch
        drow[t] = d_va << 2
        drows[t] = (d_rows << 2) | 3
        dbase[t] = d_base
        dstr[t] = d_str
        # ---- the edge ----
        if r != 0:
            d_done: uint1 = di & ac
            if di == 0:
                if isdma != 0:
                    d_st = d[55]
                    d_ch = d[53]
                    d_va = d[41:53]
                    d_rows = d[29:41]
                    d_str = sr
                    disp: UInt(32) = d[4:28]
                    if d[28] != 0:
                        if disp >= 0x800000:
                            disp = disp | 0xFF000000
                        d_base = b + disp
                    else:
                        d_base = b
                    di = 1
            elif ac != 0:
                di = 0
            if st == 0:
                dq = 0
            elif ra != 0:
                dq = dl
            elif st == 1 and dq != 0:
                dq = dq - 1
            done_q = 0
            clr_q = 0
            if st == 0:
                if go != 0:
                    st = 1
            elif st == 1:
                if ra != 0:
                    if isdma != 0:
                        st = 2
                    elif csub == 2:
                        fmask = c[0:2]
                        st = 3
                    elif csub == 3:
                        st = 4
            elif st == 2:
                if d_done != 0:
                    st = 1
            elif st == 3:
                if (cd & fmask) == fmask:
                    clr_q = fmask
                    st = 1
            elif st == 4:
                if idl != 0:
                    done_q = 1
                    st = 0
    # end of trace (harness only): pad every command stream to N tokens so the
    # receivers' trip count is static; a pad token has no valid bit
    for j in range(N):
        jj: UInt(32) = j
        if jj >= kv:
            cv.put(0)
        if jj >= kx:
            cx.put(0)
        if jj >= km:
            cm.put(0)


@unit(memories=("RV",), reads=("cv",), parameters=("N",))
def vrx(rv: UInt(64)[N]):
    """The V command's receiver (the VPU's side of D-23), harness: in order."""
    for k in range(N):
        rv[k] = cv.get()


@unit(memories=("RX",), reads=("cx",), parameters=("N",))
def xrx(rx: UInt(32)[N]):
    for k in range(N):
        rx[k] = cx.get()


@unit(memories=("RM",), reads=("cm",), parameters=("N",))
def mrx(rm: UInt(32)[N]):
    for k in range(N):
        rm[k] = cm.get()


# ---------------------------------------------------------------------------------
# The architecture
# ---------------------------------------------------------------------------------

#: D-12 memories (README D-12, D-14): unreset, one owner per port.
IRAM = Memory("iram", "UInt(128)", rows=str(G.IRAM_ROWS),
              ports=(Port("host", "w", visible=1), Port("fetch", "r", latency=G.IRAM_MEM_LATENCY)),
              collision="refuse", reset=False)
# The replay port is declared FIRST: the D-12 server serves its ports in
# declaration order each iteration, and with ``cap`` first it waits for the
# capture's address before answering the replay read -- which loop control
# needs before issue can decide what lcap's write is: a deadlock by
# construction (record finding S-2).
LB = Memory("lb", "UInt(128)", rows="CAP",
            ports=(Port("replay", "r", latency=0), Port("cap", "w", visible=1)),
            collision="refuse", reset=False)

#: (boundary array, Allo type, numpy dtype, source): the region's arguments in order.
#: ``source`` is the trace port an input reads (None: an output).
BOUNDARY = [
    ("WE", "uint1[N]", np.uint8, "instr_write_en"), ("WA", "UInt(12)[N]", np.uint16, "iram_addr"),
    ("WD", "UInt(32)[N, 4]", np.uint32, "dma_iram_din"),
    ("RST_F", "uint1[N]", np.uint8, "rst_n"), ("RST_C", "uint1[N]", np.uint8, "rst_n"),
    ("RST_L", "uint1[N]", np.uint8, "rst_n"), ("RST_S", "uint1[N]", np.uint8, "rst_n"),
    ("KA", "UInt(32)[N, 4]", np.uint32, "kernel_arg_csr"), ("SREG", "UInt(32)[N, 4]", np.uint32, None),
    ("RST_I", "uint1[N]", np.uint8, "rst_n"), ("START", "uint1[N]", np.uint8, "start"),
    ("CHD", "UInt(8)[N]", np.uint8, "dma_channel_done"), ("DIDLE", "uint1[N]", np.uint8, "dma_idle"),
    ("ACC", "uint1[N]", np.uint8, "dma_desc_accept"), ("PID", "UInt(8)[N]", np.uint8, "program_id_csr"),
    ("ISS", "uint1[N]", np.uint8, None), ("DONE", "uint1[N]", np.uint8, None), ("CLR", "UInt(8)[N]", np.uint8, None),
    ("SIDLE", "uint1[N]", np.uint8, None), ("SRUN", "uint1[N]", np.uint8, None),
    ("XADDR", "UInt(16)[N]", np.uint16, None), ("DV", "uint1[N]", np.uint8, None),
    ("DST", "uint1[N]", np.uint8, None), ("DCH", "uint1[N]", np.uint8, None),
    ("DROW", "UInt(16)[N]", np.uint16, None), ("DROWS", "UInt(16)[N]", np.uint16, None),
    ("DBASE", "UInt(32)[N]", np.uint32, None), ("DSTR", "UInt(32)[N]", np.uint32, None),
    ("VV", "UInt(8)[N]", np.uint8, None), ("XV", "uint1[N]", np.uint8, None), ("MV", "UInt(8)[N]", np.uint8, None),
    ("RV", "UInt(64)[N]", np.uint64, None), ("RX", "UInt(32)[N]", np.uint32, None),
    ("RM", "UInt(32)[N]", np.uint32, None),
]

#: The exchanges inside one cycle (all Wires in the RTL) and the D-23 commands.
EXCHANGES = [
    ("f_head", "UInt(16)", "fq -> issue: {valid, head address}"),
    ("f_data", "UInt(128)", "fq -> issue: the head bundle"),
    ("f_cap", "UInt(128)", "fq -> lcap: capture_data (the queue head)"),
    ("l_repen", "uint1", "lctl -> issue: replay_en (registered)"),
    ("l_rep", "UInt(128)", "lctl -> issue: the replay word (async read of lb at ridx)"),
    ("i_sa", "UInt(64)", "issue -> sagu: three read selects + the S op"),
    ("s_b", "UInt(32)", "sagu -> issue: rd_data_d_base_o"),
    ("s_s", "UInt(32)", "sagu -> issue: rd_data_d_stride_o"),
    ("s_lb", "UInt(32)", "sagu -> issue: rd_data_loop_bound_o"),
    ("i_lc", "UInt(64)", "issue -> lctl: begin/end/issued, lo, step, hi, skip, body_start"),
    ("i_fq", "UInt(16)", "issue -> fq: bundle_pop, the entry flush and its address"),
    ("l_fq", "UInt(16)", "lctl -> fq: branch_taken, branch_target"),
    ("l_cap", "UInt(4)", "lctl -> lcap: lb_reset, capture_en, bundle_issued"),
    ("c_atcap", "uint1", "lcap -> lctl: wr_ptr == LB_CAP (the capture overflow's input)"),
]
VECTORS = [("l_iv", "lctl -> issue: iv_by_level (agu_resolve)"),
           ("l_ivs", "lctl -> sagu: iv_by_level (SMAC's operand)")]
COMMANDS = [("cv", "UInt(64)", "issue -> V (vector): 5 valids + 42 payload bits"),
            ("cx", "UInt(32)", "issue -> X (memory): valid + 19"),
            ("cm", "UInt(32)", "issue -> M (matrix): 3 valids + 15")]


def architecture(n, qd=2, xd=2):
    """The composition at trace length ``n``; ``qd``: the command Streams'
    depth (D-23 legality, >= 2 stall-free in csim, F-B3); ``xd``: the
    exchanges' depth."""
    params = {"N": n, "AW": G.INSTR_ADDR_W, "SD": G.STACK_DEPTH, "CAP": G.LB_CAP, "CW": G.LB_COUNT_W,
              "IW": G.LB_IDX_W, "SPW": G.SP_W, "SL": G.S_LAT, "QD": qd, "XD": xd}
    ch = [Channel(c, t, "XD", carries=why) for c, t, why in EXCHANGES]
    ch += [Channel(c, "UInt(32)", "SD", carries=why) for c, why in VECTORS]
    ch += [Channel(c, t, "QD", carries=why) for c, t, why in COMMANDS]
    mems = tuple(Memory(name, t) for name, t, _, _ in BOUNDARY) + (IRAM, LB)
    return Architecture(name="seqloop", parameters=params, memories=mems, channels=tuple(ch),
                        units=(loader, fq, lcap, lctl, sagu, issue, vrx, xrx, mrx))


def make(n, w=16, inst="loop", qd=2):
    """The region (every link a Stream on both backends; see the module note)."""
    return architecture(n, qd).region("simulator", {"iram": "registers", "lb": "registers"})


# ---------------------------------------------------------------------------------
# The runner: trace in, the sequencer's outputs (D-23 slot ports, sreg_o) out
# ---------------------------------------------------------------------------------

_OUT_OF = {"ISS": "bundle_issued_o", "DONE": "done", "CLR": "clear_channel_done", "SIDLE": "state_is_idle_o",
           "SRUN": "state_is_run_o", "XADDR": "x_issue_address", "DV": "dma_desc_valid",
           "DST": "dma_desc_is_store", "DCH": "dma_desc_channel", "DROW": "dma_desc_vmem_row",
           "DROWS": "dma_desc_rows", "DBASE": "dma_desc_base", "DSTR": "dma_desc_stride",
           "VV": "v_valid_o", "XV": "x_valid_o", "MV": "m_valid_o"}


def args_of(cmd, n):
    from examples.minitpu.units.ctrl_lanes import split

    out = {}
    for name, t, dt, src in BOUNDARY:
        two_d = ", 4]" in t
        if src is None:
            out[name] = np.zeros((n, 4) if two_d else n, dtype=dt)
        elif two_d:
            out[name] = split(cmd[src][:n], 4)
        else:
            vals = [int(x) for x in cmd[src][:n]]
            if src == "program_id_csr":
                vals = [v & 0xF for v in vals]
            out[name] = np.asarray(vals, dtype=np.uint64).astype(dt)
    return out


def run(mod, cmd, n, w=16):
    from examples.minitpu.units.ctrl_lanes import join

    a = args_of(cmd, n)
    mod(*[a[name] for name, *_ in BOUNDARY])
    got = {p: [int(x) for x in a[k]] for k, p in _OUT_OF.items()}
    got["dma_desc_cols"] = [0] * n
    got["sreg_o"] = join(a["SREG"])
    # D-23: the k-th real token of a slot belongs to the k-th cycle the issue unit reports it valid
    for vname, pname, toks, sh in (("v_valid_o", "v_pay_o", a["RV"], 42), ("x_valid_o", "x_pay_o", a["RX"], 19),
                                   ("m_valid_o", "m_pay_o", a["RM"], 15)):
        real = [int(x) for x in toks if int(x) >> sh]
        valid, pay, k = [0] * n, [0] * n, 0
        for t in range(n):
            if got[vname][t]:
                if k < len(real):
                    valid[t] = real[k] >> sh
                    pay[t] = real[k] & ((1 << sh) - 1)
                k += 1
        got[vname], got[pname] = valid, pay
    return got
