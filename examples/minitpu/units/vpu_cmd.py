# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track B: the sequencer -> VPU command boundary as README D-23 declares it.

``vpu_ctrl_t`` (85 bits, combinational, valid in the issue cycle only) becomes

* **three slot commands** -- V (vector: alu/txin/txout/sfu/reduce + operands
  and destination), X (memory: vld/vst, register, resolved word), M (matrix:
  vmatload/vmatpush/vmatpop + register) -- each a ``Stream`` carrying one token
  per valid command, sent in the command's issue cycle; and
* **four shared resources** declared as D-12 ports (``resources()``): VREG
  read ports A/B (owned by the V unit), port C (owned by an arbiter fed by the
  store and the matrix stream over channels: the RTL's "the matrix read wins
  silently", ``vpu.sv:430``, becomes the owner's declared priority), the
  write port (W1, ``units/vpu_wb.py``), the VMEM compute port (the X unit).

Oracle: ``sequencer.sv`` in the slot wrapper (``seq_issue``), Phase 0's
program traces. The payload outside a slot's valid cycle is **masked**: a
token carries a payload only when its command is valid, which is D-23's
allowed gate (a recorded deviation; Phase 0 measured that the VPU ignores it:
``vpu_ctrl_gating.log``, 32/32 VREGs and 260/260 VMEM beats under random
payloads). The census line shows how often the RTL's ungated payload was
non-zero there (``allo==rtl on`` counts the masked slots that still agree).

Variants (``depth`` = every command Stream's depth, the obligation's "sized"):

* ``streams`` (depth 2), ``streams_d1``, ``streams_d4`` -- the issue kernel
  (``seq_issue.locked``'s body, derived mechanically: only the three payload
  outputs became puts) and three receivers recording tokens in order. The
  runner puts the k-th token of a slot at the k-th cycle the issue kernel
  reports that slot valid, so a missing, extra, reordered or wrong token is a
  difference. Function only on the simulator and csim (``cycle=``); the
  timing half of D-23's check is ``d23_rate.py``.
"""

from __future__ import annotations

import numpy as np

from examples.minitpu.units import seq_issue as SI
from examples.minitpu.units.seq_issue import (  # noqa: F401  (constants the kernel text names)
    C_OP0, C_SO0, C_SO1, D_CH, D_DP0, D_DP1, D_HD, D_R0, D_R1, D_ST, D_V, D_VA0, D_VA1, M_RI0, M_RI1,
    M_SO0, M_SO1, S_D_WAIT, S_FLUSH_WAIT, S_HALT_DRAIN, S_IDLE, S_RUN, V_AD0, V_AD1, V_AO0, V_AO1,
    V_AV, V_RA0, V_RA1, V_RB0, V_RB1, V_RD0, V_RD1, V_RL, V_RO, V_RV, V_SD0, V_SD1, V_SO0, V_SO1,
    V_SV, V_TD0, V_TD1, V_TI0, V_TI1, V_TIV, V_TO0, V_TO1, V_TOV, X_AV, X_L0, X_L1, X_LV0, X_LV1,
    X_OP, X_SH0, X_SH1, X_V, X_VI0, X_VI1, resolve)

RTL = SI.RTL
INSTANCES = SI.INSTANCES
DEFAULT = SI.DEFAULT
WIDTH = SI.WIDTH
LATENCY_SOURCE = SI.LATENCY_SOURCE
traces = SI.traces
seeds = SI.seeds


def REF(inst, cmd):
    return SI.ref_slots(cmd, gated=True)


import allo.dataflow as df  # noqa: E402
from allo.ir.types import Stream, UInt, uint8, uint16, uint32, uint64  # noqa: E402,F401


def _streams(n, depth):
    """``seq_issue.locked``'s issue kernel (text derived from it by
    ``_derive.py``-style replacement, see the module docstring) with its V/X/M
    commands sent as tokens on three ``Stream``s, one token per valid command,
    and three VPU-side receivers. The valids stay per-cycle outputs of the
    issue kernel (they say *when* it issued); the payload is what arrives."""

    @df.region()
    def top(RST: uint8[n], START: uint8[n], CHD: uint8[n], DIDLE: uint8[n], ACC: uint8[n], HV: uint8[n],
            VS: uint64[n], MS: uint8[n], XS: uint32[n], DS: uint64[n], CS: uint8[n], DLY: uint8[n],
            IV0: uint32[n], IV1: uint32[n], IV2: uint32[n], IV3: uint32[n], IV4: uint32[n], IV5: uint32[n],
            IV6: uint32[n], IV7: uint32[n], SB: uint32[n], SS: uint32[n],
            ISS: uint8[n], DONE: uint8[n], CLR: uint8[n], SIDLE: uint8[n], SRUN: uint8[n], XADDR: uint16[n],
            DV: uint8[n], DST: uint8[n], DCH: uint8[n], DROW: uint16[n], DROWS: uint16[n], DBASE: uint32[n],
            DSTR: uint32[n], VV: uint8[n], XV: uint8[n], MV: uint8[n], RV: uint64[n], RX: uint32[n],
            RM: uint32[n]):
        sv: Stream[UInt(64), depth]
        sx: Stream[UInt(32), depth]
        sm: Stream[UInt(32), depth]

        @df.kernel(mapping=[1], args=[RST, START, CHD, DIDLE, ACC, HV, VS, MS, XS, DS, CS, DLY, IV0, IV1, IV2,
                                      IV3, IV4, IV5, IV6, IV7, SB, SS, ISS, DONE, CLR, SIDLE, SRUN, XADDR, DV,
                                      DST, DCH, DROW, DROWS, DBASE, DSTR, VV, XV, MV])
        def issue(rst: uint8[n], start: uint8[n], chd: uint8[n], didle: uint8[n], acc: uint8[n],
                  hv: uint8[n], vs: uint64[n], ms: uint8[n], xs: uint32[n], ds: uint64[n], cs: uint8[n],
                  dly: uint8[n], iv0: uint32[n], iv1: uint32[n], iv2: uint32[n], iv3: uint32[n],
                  iv4: uint32[n], iv5: uint32[n], iv6: uint32[n], iv7: uint32[n], sb: uint32[n],
                  ss: uint32[n], iss: uint8[n], done: uint8[n], clr: uint8[n], sidle: uint8[n],
                  srun: uint8[n], xaddr: uint16[n], dv: uint8[n], dst: uint8[n], dch: uint8[n],
                  drow: uint16[n], drows: uint16[n], dbase: uint32[n], dstr: uint32[n], vv: uint8[n],
                  xv: uint8[n], mv: uint8[n]):
            st: uint8 = 0
            dq: uint8 = 0          # the delay counter
            done_q: uint8 = 0
            clr_q: uint8 = 0
            fmask: uint8 = 0
            # A1: dma_desc_adapter's registers
            di: uint8 = 0
            d_st: uint8 = 0
            d_ch: uint8 = 0
            d_va: uint16 = 0
            d_rows: uint16 = 0
            d_base: uint32 = 0
            d_str: uint32 = 0
            kv: uint32 = 0
            kx: uint32 = 0
            km: uint32 = 0
            for t in range(n):
                r: uint8 = rst[t]
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
                # every port read unconditionally (S6)
                h: uint8 = hv[t]
                v: uint64 = vs[t]
                m: uint8 = ms[t]
                x: uint32 = xs[t]
                d: uint64 = ds[t]
                c: uint8 = cs[t]
                dl: uint8 = dly[t]
                go: uint8 = start[t]
                cd: uint8 = chd[t]
                idl: uint8 = didle[t]
                ac: uint8 = acc[t]
                b: uint32 = sb[t]
                sr: uint32 = ss[t]
                i0: uint32 = iv0[t]
                i1: uint32 = iv1[t]
                i2: uint32 = iv2[t]
                i3: uint32 = iv3[t]
                i4: uint32 = iv4[t]
                i5: uint32 = iv5[t]
                i6: uint32 = iv6[t]
                i7: uint32 = iv7[t]
                ra: uint8 = 0
                if st == S_RUN and h != 0 and dq == 0:
                    ra = 1
                d_valid: uint8 = d[D_V:D_V + 1]
                isdma: uint8 = ra & d_valid
                csub: uint8 = c[C_SO0:C_SO1]
                # ---- outputs (before the edge) ----
                iss[t] = ra
                done[t] = done_q
                clr[t] = clr_q
                sidle[t] = 1 if st == S_IDLE else 0
                srun[t] = 1 if st == S_RUN else 0
                # resolve: the X address (payload, follows the head)
                lev: uint8 = x[X_LV0:X_LV1]
                ivs: uint32 = i0
                if lev == 1:
                    ivs = i1
                elif lev == 2:
                    ivs = i2
                elif lev == 3:
                    ivs = i3
                elif lev == 4:
                    ivs = i4
                elif lev == 5:
                    ivs = i5
                elif lev == 6:
                    ivs = i6
                elif lev == 7:
                    ivs = i7
                lit: uint16 = x[X_L0:X_L1]
                agv: uint8 = x[X_AV:X_AV + 1]
                shf: uint8 = x[X_SH0:X_SH1]
                xa: uint16 = resolve(ivs, lit, agv, shf)
                xaddr[t] = xa
                # adapt: V/X/M commands -- the valids gated by run_accept, the payload not
                vval: uint8 = 0
                vval[4:5] = ra & v[V_AV:V_AV + 1]
                vval[3:4] = ra & v[V_TIV:V_TIV + 1]
                vval[2:3] = ra & v[V_TOV:V_TOV + 1]
                vval[1:2] = ra & v[V_SV:V_SV + 1]
                vval[0:1] = ra & v[V_RV:V_RV + 1]
                vv[t] = vval
                vpay: uint64 = 0
                vpay[37:42] = v[V_RA0:V_RA1]
                vpay[32:37] = v[V_RB0:V_RB1]
                vpay[30:32] = v[V_TI0:V_TI1]
                vpay[28:30] = v[V_TO0:V_TO1]
                vpay[23:28] = v[V_TD0:V_TD1]
                vpay[19:22] = v[V_AO0:V_AO1]  # alu_op: 3 bits cast to 4 (bit 22 stays 0)
                vpay[14:19] = v[V_AD0:V_AD1]
                vpay[12:14] = v[V_SO0:V_SO1]
                vpay[7:12] = v[V_SD0:V_SD1]
                vpay[6:7] = v[V_RO:V_RO + 1]
                vpay[5:6] = v[V_RL:V_RL + 1]
                vpay[0:5] = v[V_RD0:V_RD1]
                if vval != 0:  # D-23: a V command is a token, sent in its issue cycle
                    vtok: uint64 = 0
                    vtok[42:47] = vval
                    vtok[0:42] = vpay
                    sv.put(vtok)
                    kv = kv + 1
                xvalid: uint8 = x[X_V:X_V + 1]
                xop: uint8 = x[X_OP:X_OP + 1]
                xv[t] = ra & xvalid
                xpay: uint32 = 0
                xpay[18:19] = xop
                xpay[13:18] = x[X_VI0:X_VI1]
                xpay[1:13] = xa
                xpay[0:1] = xvalid & xop
                if (ra & xvalid) != 0:
                    xtok: uint32 = 0
                    xtok[19:20] = 1
                    xtok[0:19] = xpay
                    sx.put(xtok)
                    kx = kx + 1
                msub: uint8 = m[M_SO0:M_SO1]
                mreg: uint16 = m[M_RI0:M_RI1]
                mval: uint8 = 0
                if ra != 0 and msub == 1:
                    mval = 4
                elif ra != 0 and msub == 2:
                    mval = 2
                elif ra != 0 and msub == 3:
                    mval = 1
                mv[t] = mval
                if mval != 0:
                    mtok: uint32 = 0
                    mtok[15:18] = mval
                    mtok[0:15] = (mreg << 10) | (mreg << 5) | mreg
                    sm.put(mtok)
                    km = km + 1
                # A1 outputs
                dv[t] = di
                dst[t] = d_st
                dch[t] = d_ch
                drow[t] = d_va << 2
                drows[t] = (d_rows << 2) | 3
                dbase[t] = d_base
                dstr[t] = d_str
                # ---- the edge ----
                if r != 0:
                    d_done: uint8 = di & ac
                    if di == 0:
                        if isdma != 0:
                            d_st = d[D_ST:D_ST + 1]
                            d_ch = d[D_CH:D_CH + 1]
                            d_va = d[D_VA0:D_VA1]
                            d_rows = d[D_R0:D_R1]
                            d_str = sr
                            disp: uint32 = d[D_DP0:D_DP1]
                            if d[D_HD:D_HD + 1] != 0:
                                if disp >= 0x800000:
                                    disp = disp | 0xFF000000
                                d_base = b + disp
                            else:
                                d_base = b
                            di = 1
                    elif ac != 0:
                        di = 0
                    if st == S_IDLE:
                        dq = 0
                    elif ra != 0:
                        dq = dl
                    elif st == S_RUN and dq != 0:
                        dq = dq - 1
                    done_q = 0
                    clr_q = 0
                    if st == S_IDLE:
                        if go != 0:
                            st = S_RUN
                    elif st == S_RUN:
                        if ra != 0:
                            if isdma != 0:
                                st = S_D_WAIT
                            elif csub == 2:
                                fmask = c[C_OP0:C_OP0 + 2]
                                st = S_FLUSH_WAIT
                            elif csub == 3:
                                st = S_HALT_DRAIN
                    elif st == S_D_WAIT:
                        if d_done != 0:
                            st = S_RUN
                    elif st == S_FLUSH_WAIT:
                        if (cd & fmask) == fmask:
                            clr_q = fmask
                            st = S_RUN
                    elif st == S_HALT_DRAIN:
                        if idl != 0:
                            done_q = 1
                            st = S_IDLE

            # end of trace (harness only): pad every stream to n tokens so the
            # receivers' trip count is static; a pad token has no valid bit
            for j in range(n):
                jj: uint32 = j
                if jj >= kv:
                    sv.put(0)
                if jj >= kx:
                    sx.put(0)
                if jj >= km:
                    sm.put(0)

        @df.kernel(mapping=[1], args=[RV])
        def vrx(rv: uint64[n]):
            for k in range(n):
                rv[k] = sv.get()

        @df.kernel(mapping=[1], args=[RX])
        def xrx(rx: uint32[n]):
            for k in range(n):
                rx[k] = sx.get()

        @df.kernel(mapping=[1], args=[RM])
        def mrx(rm: uint32[n]):
            for k in range(n):
                rm[k] = sm.get()

    return top

_STREAM_OUTS = ["ISS", "DONE", "CLR", "SIDLE", "SRUN", "XADDR", "DV", "DST", "DCH", "DROW", "DROWS", "DBASE",
                "DSTR", "VV", "XV", "MV"]


def run_streams(mod, cmd, n, w):
    a, o = SI.args_of(cmd, n)
    _, spec = SI._io(n)
    keep = {k: arr for (k, _), arr in zip(spec, o)}
    outs = [keep[k] for k in _STREAM_OUTS]
    rv, rx, rm = np.zeros(n, dtype=np.uint64), np.zeros(n, dtype=np.uint32), np.zeros(n, dtype=np.uint32)
    mod(*a, *outs, rv, rx, rm)
    got = {SI._OUT_OF[k]: [int(x) for x in keep[k]] for k in _STREAM_OUTS}
    got["dma_desc_cols"] = [0] * n
    # place the k-th real token of each slot at the k-th cycle the issue kernel reports it valid
    for vname, pname, toks, sh, pw in (("v_valid_o", "v_pay_o", rv, 42, 42), ("x_valid_o", "x_pay_o", rx, 19, 19),
                                       ("m_valid_o", "m_pay_o", rm, 15, 15)):
        real = [int(x) for x in toks if int(x) >> sh]
        valid = [0] * n
        pay = [0] * n
        k = 0
        for t in range(n):
            if got[vname][t]:
                if k < len(real):
                    valid[t] = real[k] >> sh
                    pay[t] = real[k] & ((1 << pw) - 1)
                k += 1
        got[vname], got[pname] = valid, pay
        got[f"_{vname}_tokens"] = (len(real), sum(1 for x in got[vname] if x))
    for key in [k for k in got if k.startswith("_")]:
        got.pop(key)
    return got


def _variant(depth):
    def make(n, w=16, inst="shipped"):
        return _streams(n, depth)

    make.__name__ = f"streams_d{depth}"
    return make


VARIANTS = {"streams": (_variant(2), run_streams), "streams_d1": (_variant(1), run_streams),
            "streams_d4": (_variant(4), run_streams)}


# ---------------------------------------------------------------------------------
# The four shared resources (README D-23) as D-12 ports, checked at composition
# ---------------------------------------------------------------------------------
# A structural declaration: every unit body moves one token a cycle on each of
# its channels and touches its ports the way the RTL's unit does in a command's
# issue cycle. What it buys is the ownership check: the collisions ``vpu.sv``
# resolves by convention (one raddr_a/raddr_b for whichever V op issues; port C
# shared by a store and a streaming vmatload/vmatpush, the matrix read winning
# silently at vpu.sv:430; six sources OR-ed onto the write port; the VMEM
# compute port) are refused unless one unit owns each port.
from allo.compose import Architecture, Channel, Memory, Port, unit  # noqa: E402


@unit(memories=("CV", "CX", "CM"), writes=("cv", "cx", "cxs", "cm"), parameters=("N",))
def r_issue(xv: UInt(64)[N], xx: UInt(32)[N], xm: UInt(32)[N]):
    """The issue unit's three command Streams (V, X, M); the store's port-C
    read request leaves with the X command (``cxs``), as ``vpu_vmem_simd``
    samples port C in the issue cycle."""
    for t in range(N):
        cv.put(xv[t])
        cx.put(xx[t])
        cxs.put(xx[t])
        cm.put(xm[t])


@unit(memories=("vreg.ra", "vreg.rb"), reads=("cv",), writes=("wv",), parameters=("N",))
def r_vop(ra, rb):
    """Whichever V op issues reads VREG ports A and B (the ALU, SFU, tree and
    transpose share them: one owner, the V unit)."""
    for t in range(N):
        c: UInt(64) = cv.get()
        a: int32 = c[37:42]
        b: int32 = c[32:37]
        x: UInt(8) = ra[a]
        y: UInt(8) = rb[b]
        wv.put(x ^ y)


@unit(memories=("vmem.c",), reads=("cx", "sd"), writes=("wx",), parameters=("N",))
def r_xop(vm):
    """X: the VMEM compute port's one owner (vld reads it, vst writes the data
    port C returned)."""
    for t in range(N):
        c: UInt(32) = cx.get()
        d: UInt(8) = sd.get()
        a: int32 = c[1:13]
        q: UInt(8) = vm[a]
        st: uint1 = 0
        if c[19:20] != 0 and c[18:19] != 0:
            st = 1
        if st:
            vm[a] = d
        wx.put(q)


@unit(reads=("cm",), writes=("mreq", "wm"), parameters=("N",))
def r_mop():
    """M: the stream engine asks port C for a row each cycle it streams; the
    pop engine sends its beat to the write port."""
    for t in range(N):
        c: UInt(32) = cm.get()
        mreq.put(c[0:5])
        wm.put(c[10:15])


@unit(memories=("vreg.rc",), reads=("cxs", "mreq"), writes=("sd", "mdata"), parameters=("N",))
def r_portc(rc):
    """Port C's one owner: the matrix stream and a store both ask for it in a
    cycle; the matrix read wins (vpu.sv:430), now a declared priority."""
    for t in range(N):
        s: UInt(32) = cxs.get()
        m: UInt(8) = mreq.get()
        a: int32 = s[13:18]
        if m != 0:
            a = m
        d: UInt(8) = rc[a]
        sd.put(d)
        mdata.put(d)


@unit(reads=("mdata",), memories=("MD",), parameters=("N",))
def r_mxu(md: UInt(8)[N]):
    for t in range(N):
        md[t] = mdata.get()


@unit(memories=("vreg.w",), reads=("wv", "wx", "wm"), parameters=("N",))
def r_wb(w):
    """The write port's one owner (W1, units/vpu_wb.py)."""
    for t in range(N):
        a: UInt(8) = wv.get()
        b: UInt(8) = wx.get()
        c: UInt(8) = wm.get()
        x: int32 = (a | b | c) & 31
        w[x] = a


VREG = Memory("vreg", "UInt(8)", rows="32",
              ports=(Port("ra", "r", latency=0), Port("rb", "r", latency=0), Port("rc", "r", latency=0),
                     Port("w", "w", visible=1)),
              collision="refuse", reset=False)
VMEM_C = Memory("vmem", "UInt(8)", rows="4096", ports=(Port("c", "rw", latency=3, visible=1),),
                collision="refuse")
R_UNITS = (r_issue, r_vop, r_xop, r_mop, r_portc, r_mxu, r_wb)
R_CHANNELS = ("cv", "cx", "cxs", "cm", "wv", "wx", "wm", "sd", "mreq", "mdata")


def resources(n=8, units=R_UNITS):
    """The D-23 boundary with its four shared resources as D-12 ports: VREG A/B
    (``r_vop``), VREG C (``r_portc``), the write port (``r_wb``), the VMEM
    compute port (``r_xop``). Construction runs every composition check."""
    return Architecture(
        name="vpu_cmd_resources",
        parameters={"N": n},
        memories=(Memory("CV", "UInt(64)[N]"), Memory("CX", "UInt(32)[N]"), Memory("CM", "UInt(32)[N]"),
                  Memory("MD", "UInt(8)[N]"), VREG, VMEM_C),
        channels=tuple(Channel(c, "UInt(64)" if c == "cv" else "UInt(32)" if c in ("cx", "cxs", "cm") else "UInt(8)",
                               "2") for c in R_CHANNELS),
        units=units,
    )


@unit(memories=("vmem.c", "vreg.ra"), reads=("cx", "sd"), writes=("wx",), parameters=("N",))
def p_xop_reads_a(vm, ra):
    """``r_xop`` that also reads VREG port A (a second reader of the V unit's port)."""
    for t in range(N):
        c: UInt(32) = cx.get()
        d: UInt(8) = sd.get()
        a: int32 = c[1:13]
        r: int32 = c[13:18]
        q: UInt(8) = vm[a]
        e: UInt(8) = ra[r]
        wx.put(q ^ d ^ e)


@unit(memories=("vmem.c", "vreg.rc"), reads=("cx", "sd"), writes=("wx",), parameters=("N",))
def p_store_reads_c(vm, rc):
    """``r_xop`` reading port C itself for its store, beside the arbiter."""
    for t in range(N):
        c: UInt(32) = cx.get()
        d: UInt(8) = sd.get()
        a: int32 = c[1:13]
        r: int32 = c[13:18]
        q: UInt(8) = vm[a]
        e: UInt(8) = rc[r]
        wx.put(q ^ d ^ e)


@unit(memories=("vmem.c", "vreg.w"), reads=("cx", "sd"), writes=("wx",), parameters=("N",))
def p_load_writes(vm, w):
    """``r_xop`` writing its load into the register file itself (vpu.sv's
    load source driving the OR-ed write port), beside W1."""
    for t in range(N):
        c: UInt(32) = cx.get()
        d: UInt(8) = sd.get()
        a: int32 = c[1:13]
        r: int32 = c[13:18]
        q: UInt(8) = vm[a]
        w[r] = q
        wx.put(d)


def resource_probes(n=8):
    """[(label, verdict)]: the declared composition, and one wrong ownership per
    shared resource, each of which must be refused at composition."""
    out = []

    def try_(label, units, expect_ok=False):
        try:
            resources(n, units)
            out.append((label, "composed" if expect_ok else "ACCEPTED (bug)"))
        except AssertionError as e:
            out.append((label, ("REFUSED (bug): " if expect_ok else "refused: ") + str(e).splitlines()[0]))

    try_("D-23 boundary: four resources, one owner each", R_UNITS, expect_ok=True)
    try_("port A: the X unit also reads vreg.ra beside the V unit",
         (r_issue, r_vop, p_xop_reads_a, r_mop, r_portc, r_mxu, r_wb))
    try_("port C: the store reads vreg.rc itself beside the arbiter",
         (r_issue, r_vop, p_store_reads_c, r_mop, r_portc, r_mxu, r_wb))
    try_("write port: the load unit writes vreg.w beside W1",
         (r_issue, r_vop, p_load_writes, r_mop, r_portc, r_mxu, r_wb))
    try_("port C left without its owner (the arbiter dropped; the netlist rule on its channel fires first)",
         (r_issue, r_vop, r_xop, r_mop, r_mxu, r_wb))
    return out
