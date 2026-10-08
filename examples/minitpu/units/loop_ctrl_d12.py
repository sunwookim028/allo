# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track A, plan L1 with Q5: the loop buffer as a README D-12 memory.

``lb`` is a ``compose.Memory`` of ``LB_CAP`` x ``UInt(128)``, never reset (D-14:
``sequencer_loop_buffer.sv``'s LUTRAM), with two ports: ``cap`` (``w``, the
first pass's capture) and ``replay`` (``r``, latency 0: the asynchronous
``replay_data_o = mem[replay_idx_i]``). The RTL's two modules become two
units, one port each:

``capture``   ``sequencer_loop_buffer``'s write side: the write pointer and the
              capture-overflow flag (``wr_ptr == LB_CAP``), which it offers
              before it hears this cycle's controls (it is state);
``ctrl``      ``sequencer_loop_ctrl``, transcribed as ``loop_ctrl.l1``, plus
              the replay read at ``lb_replay_idx`` (an index at or above
              ``LB_CAP`` reads entry 0 here, a masked slot: the RTL reads X).

The capture overflow and the controls cross in one cycle in the RTL
(combinational); here each is one token per cycle, ``capture`` putting its
state-only flag before it gets the controls, so the cycle has no deadlock.
"""

from __future__ import annotations

import numpy as np

from allo.compose import Architecture, Channel, Memory, Port, unit
from allo.ir.types import UInt, int32, uint1  # noqa: F401  (names the bodies use)

from examples.minitpu.template.control_geometry import SHIPPED as G


@unit(memories=("RST", "BV", "BS", "LO", "HI", "STEP", "MS", "SKIP", "EV", "ISS", "CAPD"),
      writes=("c_rst", "c_bv", "c_bs", "c_lo", "c_hi", "c_step", "c_ms", "c_skip", "c_ev", "c_iss",
              "b_rst", "b_iss", "b_data"),
      parameters=("N", "AW"))
def src(xrst: uint1[N], xbv: uint1[N], xbs: UInt(AW)[N], xlo: UInt(8)[N], xhi: UInt(16)[N],
        xstep: UInt(8)[N], xms: uint1[N], xskip: UInt(AW)[N], xev: uint1[N], xiss: uint1[N],
        xcapd: UInt(32)[N, 4]):
    for t in range(N):
        d: UInt(128) = 0
        d[0:32] = xcapd[t, 0]
        d[32:64] = xcapd[t, 1]
        d[64:96] = xcapd[t, 2]
        d[96:128] = xcapd[t, 3]
        b_rst.put(xrst[t])
        b_iss.put(xiss[t])
        b_data.put(d)
        c_rst.put(xrst[t])
        c_bv.put(xbv[t])
        c_bs.put(xbs[t])
        c_lo.put(xlo[t])
        c_hi.put(xhi[t])
        c_step.put(xstep[t])
        c_ms.put(xms[t])
        c_skip.put(xskip[t])
        c_ev.put(xev[t])
        c_iss.put(xiss[t])


@unit(memories=("lb.cap",), reads=("b_rst", "b_iss", "b_data", "k_reset", "k_capen"),
      writes=("k_atcap",), parameters=("N", "CAP", "CW"))
def capture(mem):
    wr_ptr: UInt(CW) = 0
    for _ in range(N):
        live: uint1 = b_rst.get()
        if live == 0:
            wr_ptr = 0
        at_cap: uint1 = wr_ptr == CAP
        k_atcap.put(at_cap)
        lb_reset: uint1 = k_reset.get()
        cap_en: uint1 = k_capen.get()
        iss: uint1 = b_iss.get()
        d: UInt(128) = b_data.get()
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


@unit(memories=("lb.replay",),
      reads=("c_rst", "c_bv", "c_bs", "c_lo", "c_hi", "c_step", "c_ms", "c_skip", "c_ev", "c_iss", "k_atcap"),
      writes=("k_reset", "k_capen", "q_ivl", "q_rd", "q_taken", "q_target", "q_ivt", "q_lvt", "q_lbr", "q_capen",
              "q_repen", "q_ridx", "q_capovf", "q_depth", "q_ovf", "q_unf"),
      parameters=("N", "AW", "SD", "CAP", "CW", "IW", "SPW"))
def ctrl(mem):
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
    for _ in range(N):
        live: uint1 = c_rst.get()
        bv: uint1 = c_bv.get()
        bs: UInt(AW) = c_bs.get()
        lo: UInt(8) = c_lo.get()
        hi: UInt(16) = c_hi.get()
        step: UInt(8) = c_step.get()
        ms: uint1 = c_ms.get()
        skip_i: UInt(AW) = c_skip.get()
        endv: uint1 = c_ev.get()
        iss: uint1 = c_iss.get()
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
        at_cap: uint1 = k_atcap.get()
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
        rep_en: uint1 = 0
        if sp != 0 and warm == 1:
            rep_en = 1
        cap_ovf: uint1 = cap_en & iss & at_cap
        completes: uint1 = cap_en & endv & (1 - cap_ovf)
        cold_cont: uint1 = cont & (1 - warm)
        warm_exit: uint1 = exit_ & warm
        target: UInt(AW) = fr_bs[it] + body_len + 1
        if skip:
            target = bs + skip_i
        elif cold_cont:
            target = fr_bs[it]
        k_reset.put(lb_reset)
        k_capen.put(cap_en)
        ra: int32 = 0
        if ridx < CAP:
            ra = ridx
        q_rd.put(mem[ra])
        for k in range(SD):
            q_ivl.put(fr_iv[k])
        q_taken.put(cold_cont | warm_exit | skip)
        q_target.put(target)
        q_ivt.put(fr_iv[it])
        q_lvt.put(tos)
        q_lbr.put(lb_reset)
        q_capen.put(cap_en)
        q_repen.put(rep_en)
        q_ridx.put(ridx)
        q_capovf.put(cap_ovf)
        q_depth.put(sp)
        q_ovf.put(push & (sp == SD))
        q_unf.put(endv & (sp == 0))
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


@unit(memories=("IVL", "RD", "TAKEN", "TARGET", "IVT", "LVT", "LBR", "CAPEN", "REPEN", "RIDX", "CAPOVF", "DEPTH",
                "OVF", "UNF"),
      reads=("q_ivl", "q_rd", "q_taken", "q_target", "q_ivt", "q_lvt", "q_lbr", "q_capen", "q_repen", "q_ridx",
             "q_capovf", "q_depth", "q_ovf", "q_unf"),
      parameters=("N", "AW", "SD", "IW", "SPW"))
def sink(yivl: UInt(32)[N, SD], yrd: UInt(32)[N, 4], ytaken: uint1[N], ytarget: UInt(AW)[N], yivt: UInt(32)[N],
         ylvt: UInt(3)[N], ylbr: uint1[N], ycapen: uint1[N], yrepen: uint1[N], yridx: UInt(IW)[N],
         ycapovf: uint1[N], ydepth: UInt(SPW)[N], yovf: uint1[N], yunf: uint1[N]):
    for t in range(N):
        d: UInt(128) = q_rd.get()
        yrd[t, 0] = d[0:32]  # A5: the 2-D outputs stored first
        yrd[t, 1] = d[32:64]
        yrd[t, 2] = d[64:96]
        yrd[t, 3] = d[96:128]
        for k in range(SD):
            yivl[t, k] = q_ivl.get()
        ytaken[t] = q_taken.get()
        ytarget[t] = q_target.get()
        yivt[t] = q_ivt.get()
        ylvt[t] = q_lvt.get()
        ylbr[t] = q_lbr.get()
        ycapen[t] = q_capen.get()
        yrepen[t] = q_repen.get()
        yridx[t] = q_ridx.get()
        ycapovf[t] = q_capovf.get()
        ydepth[t] = q_depth.get()
        yovf[t] = q_ovf.get()
        yunf[t] = q_unf.get()


LB = Memory("lb", "UInt(128)", rows="CAP",
            ports=(Port("cap", "w", visible=1), Port("replay", "r", latency=0)),
            collision="refuse", reset=False)


def architecture(n):
    a = "UInt(AW)"
    streams = (("c_rst", "uint1"), ("c_bv", "uint1"), ("c_bs", a), ("c_lo", "UInt(8)"), ("c_hi", "UInt(16)"),
               ("c_step", "UInt(8)"), ("c_ms", "uint1"), ("c_skip", a), ("c_ev", "uint1"), ("c_iss", "uint1"),
               ("b_rst", "uint1"), ("b_iss", "uint1"), ("b_data", "UInt(128)"),
               ("k_atcap", "uint1"), ("k_reset", "uint1"), ("k_capen", "uint1"),
               ("q_rd", "UInt(128)"), ("q_taken", "uint1"), ("q_target", a), ("q_ivt", "UInt(32)"),
               ("q_lvt", "UInt(3)"), ("q_lbr", "uint1"), ("q_capen", "uint1"), ("q_repen", "uint1"),
               ("q_ridx", "UInt(IW)"), ("q_capovf", "uint1"), ("q_depth", "UInt(SPW)"), ("q_ovf", "uint1"),
               ("q_unf", "uint1"))
    ch = [Channel(c, d, "2") for c, d in streams] + [Channel("q_ivl", "UInt(32)", "SD")]
    mems = [Memory(name, dt) for name, dt in (
        ("RST", "uint1[N]"), ("BV", "uint1[N]"), ("BS", f"{a}[N]"), ("LO", "UInt(8)[N]"), ("HI", "UInt(16)[N]"),
        ("STEP", "UInt(8)[N]"), ("MS", "uint1[N]"), ("SKIP", f"{a}[N]"), ("EV", "uint1[N]"), ("ISS", "uint1[N]"),
        ("CAPD", "UInt(32)[N, 4]"), ("IVL", "UInt(32)[N, SD]"), ("RD", "UInt(32)[N, 4]"), ("TAKEN", "uint1[N]"),
        ("TARGET", f"{a}[N]"), ("IVT", "UInt(32)[N]"), ("LVT", "UInt(3)[N]"), ("LBR", "uint1[N]"),
        ("CAPEN", "uint1[N]"), ("REPEN", "uint1[N]"), ("RIDX", "UInt(IW)[N]"), ("CAPOVF", "uint1[N]"),
        ("DEPTH", "UInt(SPW)[N]"), ("OVF", "uint1[N]"), ("UNF", "uint1[N]"))]
    params = {"N": n, "AW": G.INSTR_ADDR_W, "SD": G.STACK_DEPTH, "CAP": G.LB_CAP, "CW": G.LB_COUNT_W,
              "IW": G.LB_IDX_W, "SPW": G.SP_W}
    return Architecture(name="loop_d12", parameters=params, memories=tuple(mems) + (LB,),
                        channels=tuple(ch), units=(src, capture, ctrl, sink))


def make(target):
    def f(n, w, inst):
        return architecture(n).region(target, {"lb": "registers"})

    f.__name__ = f"l1_d12_{target}"
    return f
