# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track B: W1, the VREG write port's one owner, and its calendar (plan W1, Q2).

The oracle is Phase 0's ``vpu_writeback``: ``vpu.sv`` unchanged behind
``vpu_ctrl_t`` (``rtl/u4_vpu_wb.sv``), observed at its six-source OR mux
(``wb_src_o``) and the two writeback registers after it; traces drive every
op class alone, the 586-case pair sweep, legal and unfiltered random streams
and early/late ``vmatpop``, with every payload field random outside its valid
cycle. Reference ``harness/ref_ctrl_wb.writeback_trace``.

The Allo side takes the command as the three D-23 slot commands (V, X, M:
``seq_issue.slots_of_ctrl``) and is two parts:

* **sources**: per writeback class a destination-tag pipeline as long as the
  bound unit's declared latency -- ``template/calendar.Calendar().L``, i.e.
  ``units/alu.py``'s 3, ``units/sfu.py``'s 5, the tree's 13/9 from
  ``TreeGeometry``, VMEM's port-c read 3 + 1, the transpose's 1 -- and the
  MXU pop engine's contract (a pop fires the cycle after it issues once its
  push's result is waiting, ``MxuGeometry().result_latency`` after the push).
  These stand for the VPU's units (U1-U3 built their data paths; the tags are
  what the calendar needs; data is U5's);
* **wb**: the one owner of ``vreg.w`` (D-12): the claims OR-ed (as ``vpu.sv``
  does, so a collision is visible, not arbitrated), then ``WB_STAGES`` = 2
  registers. Its "at most one claim per cycle" rule is an **obligation** on
  the composition (the schedule's), not hardware: no arbiter (D-16, P-6).

Variants:

* ``locked`` -- sources + wb in one cycle-locked kernel (one iteration a cycle).
* ``units``  -- ``compose.Architecture``: ``sources`` and ``wb`` as two units,
  ``wb`` binding port ``vreg.w`` of a declared ``Memory`` and writing the
  claimed register (the data is the claim's tag: what is checked is *when* and
  *where*), the seven sources on seven channels that carry one token a cycle
  (so the two units stay in lockstep by token count, not by a clock).
"""

from __future__ import annotations

import numpy as np

from examples.minitpu.harness import rtl
from examples.minitpu.template.calendar import Calendar
from examples.minitpu.units import seq_issue as SI
from examples.minitpu.units import vpu_writeback as P0

RTL = P0.RTL
INSTANCES = P0.INSTANCES
DEFAULT = P0.DEFAULT
WIDTH = {"shipped": 16}
LATENCY_SOURCE = P0.LATENCY_SOURCE
traces = P0.traces
REF = P0.REF

CAL = Calendar()
L = CAL.L
L_LOAD, L_ALU, L_SFU, L_RED, L_LANE, L_TX = (L[c] for c in ("load", "alu", "sfu", "reduce", "lane_reduce", "txout"))
WB_STAGES = CAL.WB_STAGES
L_POP = L["mpop"]
POP_FIRE = CAL.mpop_precondition + L_POP  # a pop fires >= push + result latency + its register
PQ = 8  # push/pop queue entries the pop engine stand-in holds (the traces keep <= 2 in flight)
assert WB_STAGES == 2, "the kernel below writes the two registers out; a different WB_STAGES needs a pipe"


def _slots(cmd, n):
    rst = [int(x) for x in np.asarray(cmd["rst_ni"]).reshape(-1)[:n]]
    raw = cmd["ctrl_i"]
    ctrl = rtl.unpack(raw) if isinstance(raw, np.ndarray) else [int(x) for x in raw]
    cols = {p: [] for p, _ in SI.SLOT_OUTS}
    for v in ctrl[:n]:
        s = SI.slots_of_ctrl(v)
        for p, _ in SI.SLOT_OUTS:
            cols[p].append(s[p])
    return rst, cols


def _args(cmd, n):
    rst, s = _slots(cmd, n)
    ins = [np.array(rst, dtype=np.uint8), np.array(s["v_valid_o"], dtype=np.uint8),
           np.array(s["v_pay_o"], dtype=np.uint64), np.array(s["x_valid_o"], dtype=np.uint8),
           np.array(s["x_pay_o"], dtype=np.uint32), np.array(s["m_valid_o"], dtype=np.uint8),
           np.array(s["m_pay_o"], dtype=np.uint16)]
    outs = [np.zeros(n, dtype=np.uint8) for _ in range(3)] + [np.zeros(n, dtype=np.uint8)]
    return ins, outs


def _collect(outs, n):
    src, stv, lv, la = outs
    got = {"ctrl_w_o": [85] * n, "wb_src_o": [int(x) for x in src], "wb_stage_valid_o": [int(x) for x in stv],
           "wb_local_valid_o": [int(x) for x in lv], "wb_local_addr_o": [int(x) for x in la]}
    for p, w, *_ in RTL.outputs:  # the ports Phase 0 does not model (data): masked by REF
        got.setdefault(p, [0] * n)
    return got


def run(mod, cmd, n, w):
    ins, outs = _args(cmd, n)
    mod(*ins, *outs)
    return _collect(outs, n)


# ---------------------------------------------------------------------------------
import allo.dataflow as df  # noqa: E402
from allo.ir.types import Stateful, UInt, int32, uint8, uint16, uint32, uint64  # noqa: E402,F401

# field positions in the slot payloads (seq_issue.V_PAY / X_PAY / M_PAY, LSB = 0)
P_TXOUT_VD = (23, 28)
P_ALU_VD = (14, 19)
P_SFU_VD = (7, 12)
P_RLANE = 5
P_RVD = (0, 5)
P_XOP = 18
P_XVI = (13, 18)
P_MPOP_VD = (0, 5)
VV_ALU, VV_TXOUT, VV_SFU, VV_RED = 4, 2, 1, 0
TX0, TX1 = P_TXOUT_VD
AD0, AD1 = P_ALU_VD
SD0, SD1 = P_SFU_VD
RD0, RD1 = P_RVD
XI0, XI1 = P_XVI
MD0, MD1 = P_MPOP_VD


def locked(n, w=16, inst="shipped"):
    @df.region()
    def top(RST: uint8[n], VV: uint8[n], VP: uint64[n], XV: uint8[n], XP: uint32[n], MV: uint8[n],
            MP: uint16[n], SRC: uint8[n], STV: uint8[n], LV: uint8[n], LA: uint8[n]):
        @df.kernel(mapping=[1], args=[RST, VV, VP, XV, XP, MV, MP, SRC, STV, LV, LA])
        def wbk(rst: uint8[n], vv: uint8[n], vp: uint64[n], xv: uint8[n], xp: uint32[n], mv: uint8[n],
                mp: uint16[n], src: uint8[n], stv: uint8[n], lv: uint8[n], la: uint8[n]):
            # sources: tag pipes (bit 5 = valid, bits 4:0 = vd), pipe[k] = issued k+1 cycles ago
            p_ld: uint8[L_LOAD] = 0
            p_alu: uint8[L_ALU] = 0
            p_sfu: uint8[L_SFU] = 0
            p_red: uint8[L_RED] = 0
            p_lane: uint8[L_LANE] = 0
            p_tx: uint8[L_TX] = 0
            # the pop engine's contract: pushes waiting for a pop, pops waiting to fire
            pq: uint32[PQ] = 0
            ph: uint8 = 0
            pt: uint8 = 0
            fq_t: uint32[PQ] = 0
            fq_v: uint8[PQ] = 0
            fh: uint8 = 0
            ft: uint8 = 0
            # wb: the two writeback registers
            s_v: uint8 = 0
            s_a: uint8 = 0
            l_v: uint8 = 0
            l_a: uint8 = 0
            for t in range(n):
                r: uint8 = rst[t]
                a_vv: uint8 = vv[t]
                a_vp: uint64 = vp[t]
                a_xv: uint8 = xv[t]
                a_xp: uint32 = xp[t]
                a_mv: uint8 = mv[t]
                a_mp: uint16 = mp[t]
                if r == 0:  # every valid pipe is reset: a claim in flight is cancelled
                    for k in range(L_LOAD):
                        p_ld[k] = 0
                    for k in range(L_ALU):
                        p_alu[k] = 0
                    for k in range(L_SFU):
                        p_sfu[k] = 0
                    for k in range(L_RED):
                        p_red[k] = 0
                    for k in range(L_LANE):
                        p_lane[k] = 0
                    for k in range(L_TX):
                        p_tx[k] = 0
                    ph = 0
                    pt = 0
                    fh = 0
                    ft = 0
                # ---- the mux: every source whose tag reaches it this cycle ----
                e_ld: uint8 = p_ld[L_LOAD - 1]
                e_alu: uint8 = p_alu[L_ALU - 1]
                e_sfu: uint8 = p_sfu[L_SFU - 1]
                e_red: uint8 = p_red[L_RED - 1]
                e_lane: uint8 = p_lane[L_LANE - 1]
                e_tx: uint8 = p_tx[L_TX - 1]
                e_pop: uint8 = 0
                fhi: int32 = fh
                tt: uint32 = t
                if fh != ft:
                    if fq_t[fhi] == tt:
                        e_pop = 32 | fq_v[fhi]
                sb: uint8 = 0
                adr: uint8 = 0
                if e_ld != 0:
                    sb = sb | 1
                    adr = adr | (e_ld & 31)
                if e_alu != 0:
                    sb = sb | 2
                    adr = adr | (e_alu & 31)
                if e_sfu != 0:
                    sb = sb | 4
                    adr = adr | (e_sfu & 31)
                if e_red != 0:
                    sb = sb | 8
                    adr = adr | (e_red & 31)
                if e_lane != 0:
                    sb = sb | 8
                    adr = adr | (e_lane & 31)
                if e_pop != 0:
                    sb = sb | 16
                    adr = adr | (e_pop & 31)
                if e_tx != 0:
                    sb = sb | 32
                    adr = adr | (e_tx & 31)
                src[t] = sb
                stv[t] = s_v
                lv[t] = 15 if l_v != 0 else 0
                la[t] = l_a
                # ---- the edge ----
                if r != 0:
                    l_v = s_v
                    l_a = s_a
                    s_v = 1 if sb != 0 else 0
                    s_a = adr
                    for k in range(L_LOAD - 1):
                        p_ld[L_LOAD - 1 - k] = p_ld[L_LOAD - 2 - k]
                    for k in range(L_ALU - 1):
                        p_alu[L_ALU - 1 - k] = p_alu[L_ALU - 2 - k]
                    for k in range(L_SFU - 1):
                        p_sfu[L_SFU - 1 - k] = p_sfu[L_SFU - 2 - k]
                    for k in range(L_RED - 1):
                        p_red[L_RED - 1 - k] = p_red[L_RED - 2 - k]
                    for k in range(L_LANE - 1):
                        p_lane[L_LANE - 1 - k] = p_lane[L_LANE - 2 - k]
                    for k in range(L_TX - 1):
                        p_tx[L_TX - 1 - k] = p_tx[L_TX - 2 - k]
                    t_ld: uint8 = 0
                    if a_xv != 0 and a_xp[P_XOP:P_XOP + 1] == 0:
                        t_ld = 32 | a_xp[XI0:XI1]
                    p_ld[0] = t_ld
                    t_alu: uint8 = 0
                    if a_vv[VV_ALU:VV_ALU + 1] != 0:
                        t_alu = 32 | a_vp[AD0:AD1]
                    p_alu[0] = t_alu
                    t_sfu: uint8 = 0
                    if a_vv[VV_SFU:VV_SFU + 1] != 0:
                        t_sfu = 32 | a_vp[SD0:SD1]
                    p_sfu[0] = t_sfu
                    t_red: uint8 = 0
                    t_lane: uint8 = 0
                    if a_vv[VV_RED:VV_RED + 1] != 0:
                        if a_vp[P_RLANE:P_RLANE + 1] != 0:
                            t_lane = 32 | a_vp[RD0:RD1]
                        else:
                            t_red = 32 | a_vp[RD0:RD1]
                    p_red[0] = t_red
                    p_lane[0] = t_lane
                    t_tx: uint8 = 0
                    if a_vv[VV_TXOUT:VV_TXOUT + 1] != 0:
                        t_tx = 32 | a_vp[TX0:TX1]
                    p_tx[0] = t_tx
                    # pop engine: retire a fired pop, queue a push, pair a pop
                    if e_pop != 0:
                        fh = (fh + 1) & (PQ - 1)
                    if a_mv[1:2] != 0:
                        pti: int32 = pt
                        pq[pti] = tt
                        pt = (pt + 1) & (PQ - 1)
                    if a_mv[0:1] != 0 and ph != pt:
                        phi: int32 = ph
                        ready: uint32 = pq[phi] + POP_FIRE
                        fire: uint32 = tt + L_POP
                        if ready > fire:
                            fire = ready
                        fti: int32 = ft
                        fq_t[fti] = fire
                        fq_v[fti] = a_mp[MD0:MD1]
                        ft = (ft + 1) & (PQ - 1)
                        ph = (ph + 1) & (PQ - 1)
                else:
                    l_v = 0
                    l_a = 0
                    s_v = 0
                    s_a = 0

    return top


# ---------------------------------------------------------------------------------
# ``units``: the D-12 form -- ``wb`` is the one owner of ``vreg.w`` (README D-12)
# ---------------------------------------------------------------------------------
from allo.compose import Architecture, Channel, Memory, Port, unit  # noqa: E402

CLASS_CH = ("c_ld", "c_alu", "c_sfu", "c_red", "c_lane", "c_tx", "c_pop")
PARAMS = ("N", "L_LOAD", "L_ALU", "L_SFU", "L_RED", "L_LANE", "L_TX", "L_POP", "POP_FIRE", "PQ")


@unit(memories=("RST", "VV", "VP", "XV", "XP", "MV", "MP"), writes=("c_rst",) + CLASS_CH, parameters=PARAMS)
def sources(rst: UInt(8)[N], vv: UInt(8)[N], vp: UInt(64)[N], xv: UInt(8)[N], xp: UInt(32)[N], mv: UInt(8)[N],
            mp: UInt(16)[N]):
    """The VPU's writeback sources as tag pipelines, sized by the bound units'
    declared latencies (the architecture's parameters, from ``Calendar``);
    one token a cycle on every channel: the tag reaching the mux, or 0."""
    p_ld: UInt(8)[L_LOAD] = 0
    p_alu: UInt(8)[L_ALU] = 0
    p_sfu: UInt(8)[L_SFU] = 0
    p_red: UInt(8)[L_RED] = 0
    p_lane: UInt(8)[L_LANE] = 0
    p_tx: UInt(8)[L_TX] = 0
    pq: UInt(32)[PQ] = 0
    ph: UInt(8) = 0
    pt: UInt(8) = 0
    fq_t: UInt(32)[PQ] = 0
    fq_v: UInt(8)[PQ] = 0
    fh: UInt(8) = 0
    ft: UInt(8) = 0
    for t in range(N):
        r: UInt(8) = rst[t]
        a_vv: UInt(8) = vv[t]
        a_vp: UInt(64) = vp[t]
        a_xv: UInt(8) = xv[t]
        a_xp: UInt(32) = xp[t]
        a_mv: UInt(8) = mv[t]
        a_mp: UInt(16) = mp[t]
        if r == 0:
            for k in range(L_LOAD):
                p_ld[k] = 0
            for k in range(L_ALU):
                p_alu[k] = 0
            for k in range(L_SFU):
                p_sfu[k] = 0
            for k in range(L_RED):
                p_red[k] = 0
            for k in range(L_LANE):
                p_lane[k] = 0
            for k in range(L_TX):
                p_tx[k] = 0
            ph = 0
            pt = 0
            fh = 0
            ft = 0
        e_pop: UInt(8) = 0
        fhi: int32 = fh
        tt: UInt(32) = t
        if fh != ft:
            if fq_t[fhi] == tt:
                e_pop = 32 | fq_v[fhi]
        c_rst.put(r)
        c_ld.put(p_ld[L_LOAD - 1])
        c_alu.put(p_alu[L_ALU - 1])
        c_sfu.put(p_sfu[L_SFU - 1])
        c_red.put(p_red[L_RED - 1])
        c_lane.put(p_lane[L_LANE - 1])
        c_tx.put(p_tx[L_TX - 1])
        c_pop.put(e_pop)
        if r != 0:
            for k in range(L_LOAD - 1):
                p_ld[L_LOAD - 1 - k] = p_ld[L_LOAD - 2 - k]
            for k in range(L_ALU - 1):
                p_alu[L_ALU - 1 - k] = p_alu[L_ALU - 2 - k]
            for k in range(L_SFU - 1):
                p_sfu[L_SFU - 1 - k] = p_sfu[L_SFU - 2 - k]
            for k in range(L_RED - 1):
                p_red[L_RED - 1 - k] = p_red[L_RED - 2 - k]
            for k in range(L_LANE - 1):
                p_lane[L_LANE - 1 - k] = p_lane[L_LANE - 2 - k]
            for k in range(L_TX - 1):
                p_tx[L_TX - 1 - k] = p_tx[L_TX - 2 - k]
            t_ld: UInt(8) = 0
            if a_xv != 0 and a_xp[18:19] == 0:
                t_ld = 32 | a_xp[13:18]
            p_ld[0] = t_ld
            t_alu: UInt(8) = 0
            if a_vv[4:5] != 0:
                t_alu = 32 | a_vp[14:19]
            p_alu[0] = t_alu
            t_sfu: UInt(8) = 0
            if a_vv[1:2] != 0:
                t_sfu = 32 | a_vp[7:12]
            p_sfu[0] = t_sfu
            t_red: UInt(8) = 0
            t_lane: UInt(8) = 0
            if a_vv[0:1] != 0:
                if a_vp[5:6] != 0:
                    t_lane = 32 | a_vp[0:5]
                else:
                    t_red = 32 | a_vp[0:5]
            p_red[0] = t_red
            p_lane[0] = t_lane
            t_tx: UInt(8) = 0
            if a_vv[2:3] != 0:
                t_tx = 32 | a_vp[23:28]
            p_tx[0] = t_tx
            if e_pop != 0:
                fh = (fh + 1) & (PQ - 1)
            if a_mv[1:2] != 0:
                pti: int32 = pt
                pq[pti] = tt
                pt = (pt + 1) & (PQ - 1)
            if a_mv[0:1] != 0 and ph != pt:
                phi: int32 = ph
                ready: UInt(32) = pq[phi] + POP_FIRE
                fire: UInt(32) = tt + L_POP
                if ready > fire:
                    fire = ready
                fti: int32 = ft
                fq_t[fti] = fire
                fq_v[fti] = a_mp[0:5]
                ft = (ft + 1) & (PQ - 1)
                ph = (ph + 1) & (PQ - 1)


@unit(memories=("vreg.w", "SRC", "STV", "LV", "LA"), reads=("c_rst",) + CLASS_CH, parameters=("N",))
def wb(mem, src: UInt(8)[N], stv: UInt(8)[N], lv: UInt(8)[N], la: UInt(8)[N]):
    """W1: the one owner of ``vreg.w``. The claims are OR-ed, as ``vpu.sv``
    does (a collision shows, nothing arbitrates: the one-claim-per-cycle rule
    is the composition's obligation), then two registers; the second writes
    the register file (its data here is the address: the calendar is what is
    held to the RTL, the data path is U5's)."""
    s_v: UInt(8) = 0
    s_a: UInt(8) = 0
    l_v: UInt(8) = 0
    l_a: UInt(8) = 0
    for t in range(N):
        r: UInt(8) = c_rst.get()
        e_ld: UInt(8) = c_ld.get()
        e_alu: UInt(8) = c_alu.get()
        e_sfu: UInt(8) = c_sfu.get()
        e_red: UInt(8) = c_red.get()
        e_lane: UInt(8) = c_lane.get()
        e_tx: UInt(8) = c_tx.get()
        e_pop: UInt(8) = c_pop.get()
        sb: UInt(8) = 0
        adr: UInt(8) = 0
        if e_ld != 0:
            sb = sb | 1
            adr = adr | (e_ld & 31)
        if e_alu != 0:
            sb = sb | 2
            adr = adr | (e_alu & 31)
        if e_sfu != 0:
            sb = sb | 4
            adr = adr | (e_sfu & 31)
        if e_red != 0:
            sb = sb | 8
            adr = adr | (e_red & 31)
        if e_lane != 0:
            sb = sb | 8
            adr = adr | (e_lane & 31)
        if e_pop != 0:
            sb = sb | 16
            adr = adr | (e_pop & 31)
        if e_tx != 0:
            sb = sb | 32
            adr = adr | (e_tx & 31)
        src[t] = sb
        stv[t] = s_v
        lv[t] = 15 if l_v != 0 else 0
        la[t] = l_a
        we: uint1 = 0
        if r != 0 and l_v != 0:
            we = 1
        wa: int32 = l_a
        if we:
            mem[wa] = l_a
        if r != 0:
            l_v = s_v
            l_a = s_a
            s_v = 1 if sb != 0 else 0
            s_a = adr
        else:
            l_v = 0
            l_a = 0
            s_v = 0
            s_a = 0


VREG_W = Memory("vreg", "UInt(8)", rows="32", ports=(Port("w", "w", visible=1),), collision="refuse")


def architecture(n, units=None, cal=CAL):
    L_ = cal.L
    params = {"N": n, "L_LOAD": L_["load"], "L_ALU": L_["alu"], "L_SFU": L_["sfu"], "L_RED": L_["reduce"],
              "L_LANE": L_["lane_reduce"], "L_TX": L_["txout"], "L_POP": L_["mpop"],
              "POP_FIRE": cal.mpop_precondition + L_["mpop"], "PQ": PQ}
    return Architecture(
        name="vpu_wb_units",
        parameters=params,
        memories=(Memory("RST", "UInt(8)[N]"), Memory("VV", "UInt(8)[N]"), Memory("VP", "UInt(64)[N]"),
                  Memory("XV", "UInt(8)[N]"), Memory("XP", "UInt(32)[N]"), Memory("MV", "UInt(8)[N]"),
                  Memory("MP", "UInt(16)[N]"), Memory("SRC", "UInt(8)[N]"), Memory("STV", "UInt(8)[N]"),
                  Memory("LV", "UInt(8)[N]"), Memory("LA", "UInt(8)[N]"), VREG_W),
        channels=tuple(Channel(c, "UInt(8)", "2") for c in ("c_rst",) + CLASS_CH),
        units=units or (sources, wb),
    )


def units_variant(n, w=16, inst="shipped"):
    return architecture(n).region("simulator")


@unit(memories=("vreg.w", "XV"), parameters=("N",))
def alu_direct(mem, xv: UInt(8)[N]):
    """H7 probe: an ALU that writes the register file itself (``vpu.sv``'s
    OR-ed second driver of the write port)."""
    for t in range(N):
        if xv[t] != 0:
            mem[0] = 1


def h7_probe(n=8):
    """The single-owner writeback refuses a second producer on ``vreg.w`` at
    composition (plan H7). Returns the refusal, or ``ACCEPTED`` (a bug)."""
    try:
        architecture(n, units=(sources, wb, alu_direct))
        return "ACCEPTED (bug)"
    except AssertionError as e:
        return "refused: " + str(e).splitlines()[0]


VARIANTS = {"locked": (locked, run), "units": (units_variant, run)}
