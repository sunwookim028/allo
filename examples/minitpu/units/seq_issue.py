# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track B: the sequencer's issue unit (plan I1 + A1), cycle-locked (D-24).

The oracle is MiniTPU's ``sequencer.sv`` unchanged, run on Phase 0's program
traces (``units/sequencer.py``: programs from MiniTPU's ``asm.py`` via
``harness/minitpu_asm``, the shipped images, an unscheduled program) inside a
flattening wrapper (``rtl/u4_seq_cmd.sv``) that presents ``vpu_ctrl_t`` as the
three D-23 slot commands V, X, M -- each a valid vector and a payload vector.
The reference is Phase 0's ``ref_ctrl_issue`` (REF-MATCH on every defined slot).

What is Allo here (track B's units):

* the **issue FSM** of ``sequencer.sv:254-550`` (IDLE, RUN, D_WAIT,
  FLUSH_WAIT, HALT_DRAIN; the ``delay`` hold; ``run_accept``);
* **A1**, the descriptor adapter (``dma_desc_adapter.sv``): snapshot at start,
  words to beats, held until accepted.

Track A's combinational C1 functions are called in the kernel (the U4
wave-1 integration): ``seq_decoder.decode_v/m/x/d/c`` on the head word (and its
``delay`` field), ``agu_resolve.resolve`` and ``vpu_adapter.adapt_v/x/m``.

What is still **stubbed**: the per-cycle inputs a composed sequencer gets from
track A's *stateful* units -- fetch (F1), loop control (L1) and the scalar
AGU (S1), which are kernel bodies, not functions. They are side columns of
the trace, produced by Phase 0's bit-exact references of those parts while
the reference runs the same program (``_Tap``): the head bundle's ``valid``
and its 128-bit word (fetch queue head or loop-buffer replay), the loop
frames' ``iv_by_level`` (L1's registered output), and S1's two descriptor
read ports. The side columns are open-loop replays of a closed loop: they are
what the front end did *given the RTL's issue decisions*, so an Allo issue
unit that decided differently would diverge at its first wrong cycle and
every later output would differ -- the check stays sound. Closing the loop
(F1, L1, S1 as kernels exchanging, per cycle, the head, the decoded L/S
slots, ``run_accept``, the loop bound read and the branch/flush) is a
composition of four stateful units, not a function swap. Names: ``HV``,
``HW0..3`` (the head word, four 32-bit lanes, P-8), ``IV0..7`` (L1
``iv_by_level``), ``SB``/``SS`` (S1 ``rd_data_d_base_o`` /
``rd_data_d_stride_o``).
"""

from __future__ import annotations

import os

import numpy as np

from examples.minitpu.harness import ref_ctrl_decode as D
from examples.minitpu.harness import ref_ctrl_issue as R
from examples.minitpu.harness import rtl
from examples.minitpu.units import sequencer as S

WRAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtl", "u4_seq_cmd.sv")

# ---- the three slot commands (D-23), MSB first, as the wrapper flattens them --------
V_VALID = ["alu_valid", "txin_valid", "txout_valid", "sfu_valid", "reduce_valid"]
V_PAY = [("raddr_a", 5), ("raddr_b", 5), ("txin_index", 2), ("txout_index", 2), ("txout_vd", 5),
         ("alu_op", 4), ("alu_vd", 5), ("sfu_op", 2), ("sfu_vd", 5), ("reduce_op", 1),
         ("reduce_lane", 1), ("reduce_vd", 5)]
X_PAY = [("vmem.op", 1), ("vmem.vreg_idx", 5), ("vmem.vmem_address", 12), ("vmem_store_read_hint", 1)]
M_VALID = ["vmatload_valid", "vmatpush_valid", "vmatpop_valid"]
M_PAY = [("vmatload_base", 5), ("vmatpush_vs", 5), ("vmatpop_vd", 5)]
SLOT_OUTS = [("v_valid_o", 5), ("v_pay_o", 42), ("x_valid_o", 1), ("x_pay_o", 19),
             ("m_valid_o", 3), ("m_pay_o", 15)]
SLOT_OF = {"v_pay_o": "v_valid_o", "x_pay_o": "x_valid_o", "m_pay_o": "m_valid_o"}

OUTS = [(p, w) for p, w in R.OUTS if p != "vpu_ctrl_o"] + SLOT_OUTS
RTL = rtl.RtlUnit(top="u4_seq_cmd", sources=S.SOURCES + [WRAP], inputs=S.INPUTS, outputs=OUTS,
                  shape="trace", clk="clk", rst_n="rst_n", assertions=True)
INSTANCES = {"shipped": RTL}
DEFAULT = "shipped"
WIDTH = {"shipped": 16}
LATENCY_SOURCE = S.LATENCY_SOURCE


def slots_of_ctrl(v):
    """85-bit ``vpu_ctrl_t`` -> the six slot ports' ints."""
    f = D.unpack(D.VPU_CTRL, int(v))
    vv = 0
    for n in V_VALID:
        vv = (vv << 1) | f[n]
    mv = 0
    for n in M_VALID:
        mv = (mv << 1) | f[n]
    return {"v_valid_o": vv, "v_pay_o": D.pack(V_PAY, f), "x_valid_o": f["vmem.valid"],
            "x_pay_o": D.pack(X_PAY, f), "m_valid_o": mv, "m_pay_o": D.pack(M_PAY, f)}


def ctrl_of_slots(s):
    """The inverse: six slot ints -> the 85-bit ``vpu_ctrl_t``."""
    f = {}
    for k, n in enumerate(V_VALID):
        f[n] = (s["v_valid_o"] >> (len(V_VALID) - 1 - k)) & 1
    for k, n in enumerate(M_VALID):
        f[n] = (s["m_valid_o"] >> (len(M_VALID) - 1 - k)) & 1
    f["vmem.valid"] = s["x_valid_o"]
    f.update(D.unpack(V_PAY, s["v_pay_o"]))
    f.update(D.unpack(X_PAY, s["x_pay_o"]))
    f.update(D.unpack(M_PAY, s["m_pay_o"]))
    return D.pack(D.VPU_CTRL, f)


GATE = "payload, slot not valid (D-23 gate)"


def ref_slots(cmd, gated=False):
    """Phase 0's sequencer reference, vpu_ctrl_o split into the slot ports.
    ``gated``: a slot's payload is masked in the cycles its valid is low
    (README D-23: an instance may gate it; Phase 0 showed the VPU ignores it)."""
    want, reason, ev = R.issue_trace(cmd)
    ctrl = rtl.unpack(want.pop("vpu_ctrl_o"))
    why = reason.pop("vpu_ctrl_o")
    n = len(ctrl)
    cols = {p: [] for p, _ in SLOT_OUTS}
    for v in ctrl:
        s = slots_of_ctrl(v)
        for p, _ in SLOT_OUTS:
            cols[p].append(s[p])
    for p, w in SLOT_OUTS:
        want[p] = rtl.pack(cols[p], w)
        reason[p] = why.copy()
    if gated:
        for p, pv in SLOT_OF.items():
            for t in range(n):
                if not reason[p][t] and cols[pv][t] == 0:
                    reason[p][t] = GATE
    return want, reason, ev


def REF(inst, cmd):
    return ref_slots(cmd)


# ---- the stubbed front end: side columns from Phase 0's part references ------------
SIDE = ["HV", "HW0", "HW1", "HW2", "HW3"] + [f"IV{k}" for k in range(8)] + ["SB", "SS"]


class _Tap(R.Sequencer):
    """``ref_ctrl_issue.Sequencer`` that records, per cycle, what the composed
    front end hands the issue unit (before the edge): the head bundle (fetch
    queue or loop-buffer replay), L1's ``iv_by_level`` and S1's two
    descriptor read ports."""

    def __init__(self):
        super().__init__()
        self.rec = []
        self._lo = self._so = None
        loop_step, sagu_out = self.loop.step, self.sagu.outputs

        def lstep(row):
            o = loop_step(row)
            self._lo = o[0]
            return o

        def sout(row):
            o = sagu_out(row)
            self._so = o
            return o

        self.loop.step, self.sagu.outputs = lstep, sout

    def step(self, r):
        lp = self.loop
        replay_en = lp.sp != 0 and bool(lp.warm)
        fq_valid, fq_data, _ = self.fetch.head()
        if replay_en:
            word = lp.lbmem[lp.ridx] if lp.ridx < len(lp.lbmem) else None
            valid = True
        else:
            word, valid = fq_data, fq_valid
        o, why = super().step(r)
        self.rec.append((int(bool(valid)), word, self._lo["iv_by_level"], self._so["rd_data_d_base_o"],
                         self._so["rd_data_d_stride_o"]))
        return o, why


def side_columns(cmd, tap=None):
    """Side columns for one trace. ``tap``: the model carried over from the
    previous trace -- the IRAM (and the loop buffer) are unreset, so a trace
    sees the words an earlier one wrote, exactly as the RTL run over the
    joined traces does (``check._trace_all`` joins them in this order)."""
    cols = {p: [int(x) for x in v] for p, v in cmd.items()}
    n = len(cols["rst_n"])
    m = tap or _Tap()
    k0 = len(m.rec)
    for t in range(n):
        m.step({p: cols[p][t] for p in cols})
    out = {p: [] for p in SIDE}
    for hv, word, ivs, sb, ss in m.rec[k0:]:
        out["HV"].append(hv)
        for k in range(4):
            out[f"HW{k}"].append(((word or 0) >> (32 * k)) & 0xFFFFFFFF)
        for k, v in enumerate(D.ivs_of(ivs)):
            out[f"IV{k}"].append(v)
        out["SB"].append(sb)
        out["SS"].append(ss)
    return out


_TAP = {}


def _with_side(cmd, key):
    """One model per joined trace set (``key``), stepped trace after trace."""
    tap = _TAP.setdefault(key, _Tap())
    return {**cmd, **side_columns(cmd, tap)}


def traces(inst):
    _TAP.pop("main", None)
    return [(lab, _with_side(c, "main"), legal) for lab, c, legal in S.traces("shipped")]


def seeds():
    """MiniTPU's sequencer-level tbs replayed at the ports (as ``sequencer``),
    with side columns, continuing the model of ``traces`` (check.py appends the
    seeds after the traces). Off unless ``U4B_SEEDS=1``: ``tb_loop_begin_r``
    alone is 271,834 cycles."""
    if os.environ.get("U4B_SEEDS") != "1":
        return []
    return [(lab, inst, _with_side(c, "main"), seen, legal) for lab, inst, c, seen, legal in S.seeds()]


# ---------------------------------------------------------------------------------
# Allo: I1 (cycle-locked issue) + A1 (descriptor adapter), one kernel
# ---------------------------------------------------------------------------------
import allo.dataflow as df  # noqa: E402
from allo.ir.types import UInt, int32, uint1, uint8, uint16, uint32, uint64  # noqa: E402,F401

# track A's C1 functions (plain Allo functions the kernel calls)
from examples.minitpu.units.agu_resolve import resolve  # noqa: E402
from examples.minitpu.units.seq_decoder import decode_c, decode_d, decode_m, decode_v, decode_x  # noqa: E402
from examples.minitpu.units.vpu_adapter import adapt_m, adapt_v, adapt_x  # noqa: E402


def _offsets(layout):
    """{field: (lo, hi)} bit positions of an MSB-first packed layout."""
    out, o = {}, sum(w for _, w in layout)
    for n, w in layout:
        o -= w
        out[n] = (o, o + w)
    return out


_V, _X, _D, _C = _offsets(D.V_SLOT), _offsets(D.X_SLOT), _offsets(D.D_SLOT), _offsets(D.C_SLOT)
_M = _offsets(D.M_SLOT)
# names used as constants in the kernel bodies (Allo resolves module globals)
V_RA0, V_RA1 = _V["raddr_a"]
V_RB0, V_RB1 = _V["raddr_b"]
V_AV = _V["alu_valid"][0]
V_TIV = _V["txin_valid"][0]
V_TI0, V_TI1 = _V["txin_index"]
V_TOV = _V["txout_valid"][0]
V_TO0, V_TO1 = _V["txout_index"]
V_TD0, V_TD1 = _V["txout_vd"]
V_AO0, V_AO1 = _V["alu_op"]
V_AD0, V_AD1 = _V["alu_vd"]
V_SV = _V["sfu_valid"][0]
V_SO0, V_SO1 = _V["sfu_op"]
V_SD0, V_SD1 = _V["sfu_vd"]
V_RV = _V["reduce_valid"][0]
V_RO = _V["reduce_op"][0]
V_RL = _V["reduce_lane"][0]
V_RD0, V_RD1 = _V["reduce_vd"]
X_V = _X["valid"][0]
X_OP = _X["op"][0]
X_VI0, X_VI1 = _X["vreg_idx"]
X_L0, X_L1 = _X["literal"]
X_AV = _X["agu_valid"][0]
X_LV0, X_LV1 = _X["agu_level"]
X_SH0, X_SH1 = _X["agu_shift"]
D_V = _D["valid"][0]
D_ST = _D["is_store"][0]
D_CH = _D["channel_sel"][0]  # DMA_CHANNEL_SEL_W = 1: bit 0 of channel_sel
D_VA0, D_VA1 = _D["vmem_address"]
D_R0, D_R1 = _D["rows"]
D_HD = _D["has_disp"][0]
D_DP0, D_DP1 = _D["disp"]
C_SO0, C_SO1 = _C["subop"]
C_OP0 = _C["operand"][0]
M_SO0, M_SO1 = _M["subop"]
M_RI0, M_RI1 = _M["reg_idx"]
S_IDLE, S_RUN, S_D_WAIT, S_FLUSH_WAIT, S_HALT_DRAIN = range(5)


def _io(n):
    """(inputs, outputs) of the issue kernel: name -> numpy dtype."""
    ins = [("RST", np.uint8), ("START", np.uint8), ("CHD", np.uint8), ("DIDLE", np.uint8), ("ACC", np.uint8),
           ("HV", np.uint8), ("HW0", np.uint32), ("HW1", np.uint32), ("HW2", np.uint32),
           ("HW3", np.uint32)] + [(f"IV{k}", np.uint32) for k in range(8)] + [
           ("SB", np.uint32), ("SS", np.uint32)]
    outs = [("ISS", np.uint8), ("DONE", np.uint8), ("CLR", np.uint8), ("SIDLE", np.uint8), ("SRUN", np.uint8),
            ("XADDR", np.uint16), ("DV", np.uint8), ("DST", np.uint8), ("DCH", np.uint8), ("DROW", np.uint16),
            ("DROWS", np.uint16), ("DBASE", np.uint32), ("DSTR", np.uint32),
            ("VV", np.uint8), ("VP", np.uint64), ("XV", np.uint8), ("XP", np.uint32), ("MV", np.uint8),
            ("MP", np.uint16)]
    return ins, outs


_CMD_OF = {"RST": "rst_n", "START": "start", "CHD": "dma_channel_done", "DIDLE": "dma_idle",
           "ACC": "dma_desc_accept"}
_OUT_OF = {"ISS": "bundle_issued_o", "DONE": "done", "CLR": "clear_channel_done", "SIDLE": "state_is_idle_o",
           "SRUN": "state_is_run_o", "XADDR": "x_issue_address", "DV": "dma_desc_valid",
           "DST": "dma_desc_is_store", "DCH": "dma_desc_channel", "DROW": "dma_desc_vmem_row",
           "DROWS": "dma_desc_rows", "DBASE": "dma_desc_base", "DSTR": "dma_desc_stride",
           "VV": "v_valid_o", "VP": "v_pay_o", "XV": "x_valid_o", "XP": "x_pay_o", "MV": "m_valid_o",
           "MP": "m_pay_o"}


def args_of(cmd, n):
    ins, outs = _io(n)
    a = [np.asarray([int(x) for x in cmd[_CMD_OF.get(k, k)][:n]], dtype=np.uint64).astype(dt) for k, dt in ins]
    o = [np.zeros(n, dtype=dt) for _, dt in outs]
    return a, o


def collect(outs, n):
    _, spec = _io(n)
    got = {_OUT_OF[k]: [int(x) for x in o] for (k, _), o in zip(spec, outs)}
    got["dma_desc_cols"] = [0] * n
    return got


def run_locked(mod, cmd, n, w):
    a, o = args_of(cmd, n)
    mod(*a, *o)
    return collect(o, n)


def locked(n, w=16, inst="shipped"):
    """I1 + A1 as one cycle-locked kernel: one iteration is one clock cycle;
    outputs are the state before the edge (``rtl.py``'s convention), then the
    edge. The V/X/M commands are the kernel's per-cycle outputs, payload
    ungated (the RTL's form)."""

    @df.region()
    def top(RST: uint8[n], START: uint8[n], CHD: uint8[n], DIDLE: uint8[n], ACC: uint8[n], HV: uint8[n],
            HW0: uint32[n], HW1: uint32[n], HW2: uint32[n], HW3: uint32[n],
            IV0: uint32[n], IV1: uint32[n], IV2: uint32[n], IV3: uint32[n], IV4: uint32[n], IV5: uint32[n],
            IV6: uint32[n], IV7: uint32[n], SB: uint32[n], SS: uint32[n],
            ISS: uint8[n], DONE: uint8[n], CLR: uint8[n], SIDLE: uint8[n], SRUN: uint8[n], XADDR: uint16[n],
            DV: uint8[n], DST: uint8[n], DCH: uint8[n], DROW: uint16[n], DROWS: uint16[n], DBASE: uint32[n],
            DSTR: uint32[n], VV: uint8[n], VP: uint64[n], XV: uint8[n], XP: uint32[n], MV: uint8[n],
            MP: uint16[n]):
        @df.kernel(mapping=[1], args=[RST, START, CHD, DIDLE, ACC, HV, HW0, HW1, HW2, HW3, IV0, IV1, IV2,
                                      IV3, IV4, IV5, IV6, IV7, SB, SS, ISS, DONE, CLR, SIDLE, SRUN, XADDR, DV,
                                      DST, DCH, DROW, DROWS, DBASE, DSTR, VV, VP, XV, XP, MV, MP])
        def issue(rst: uint8[n], start: uint8[n], chd: uint8[n], didle: uint8[n], acc: uint8[n],
                  hv: uint8[n], hw0: uint32[n], hw1: uint32[n], hw2: uint32[n], hw3: uint32[n],
                  iv0: uint32[n], iv1: uint32[n], iv2: uint32[n], iv3: uint32[n],
                  iv4: uint32[n], iv5: uint32[n], iv6: uint32[n], iv7: uint32[n], sb: uint32[n],
                  ss: uint32[n], iss: uint8[n], done: uint8[n], clr: uint8[n], sidle: uint8[n],
                  srun: uint8[n], xaddr: uint16[n], dv: uint8[n], dst: uint8[n], dch: uint8[n],
                  drow: uint16[n], drows: uint16[n], dbase: uint32[n], dstr: uint32[n], vv: uint8[n],
                  vp: uint64[n], xv: uint8[n], xp: uint32[n], mv: uint8[n], mp: uint16[n]):
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
                w: UInt(128) = 0  # the head bundle (F1 / loop-buffer replay: still a side column)
                w[0:32] = hw0[t]
                w[32:64] = hw1[t]
                w[64:96] = hw2[t]
                w[96:128] = hw3[t]
                # C1 decode (track A, units/seq_decoder.py): per-slot records, sequencer_pkg's layout
                v: UInt(46) = decode_v(w)
                m: UInt(8) = decode_m(w)
                x: UInt(27) = decode_x(w)
                d: UInt(57) = decode_d(w)
                c: UInt(7) = decode_c(w)
                dl: uint8 = w[15:22]
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
                # C1 resolve (track A, units/agu_resolve.py): the X address (payload, follows the head)
                ivs: UInt(32)[8]
                ivs[0] = i0
                ivs[1] = i1
                ivs[2] = i2
                ivs[3] = i3
                ivs[4] = i4
                ivs[5] = i5
                ivs[6] = i6
                ivs[7] = i7
                lev: UInt(3) = x[X_LV0:X_LV1]
                lit: UInt(12) = x[X_L0:X_L1]
                agv: uint1 = x[X_AV:X_AV + 1]
                shf: UInt(4) = x[X_SH0:X_SH1]
                xa: UInt(12) = resolve(ivs, lit, agv, lev, shf)
                xaddr[t] = xa
                # C1 adapt (track A, units/vpu_adapter.py): the three D-23 slot commands
                iss1: uint1 = ra
                vc: UInt(47) = adapt_v(v, iss1)
                xc: UInt(20) = adapt_x(x, iss1, xa)
                mc: UInt(18) = adapt_m(m, iss1)
                # the commands split into the wrapper's valid and payload vectors
                vval: uint8 = 0
                vval[4:5] = vc[36:37]  # alu_valid
                vval[3:4] = vc[35:36]  # txin_valid
                vval[2:3] = vc[32:33]  # txout_valid
                vval[1:2] = vc[15:16]  # sfu_valid
                vval[0:1] = vc[7:8]  # reduce_valid
                vv[t] = vval
                vpay: uint64 = 0
                vpay[0:7] = vc[0:7]
                vpay[7:14] = vc[8:15]
                vpay[14:23] = vc[16:25]
                vpay[23:30] = vc[25:32]
                vpay[30:32] = vc[33:35]
                vpay[32:42] = vc[37:47]
                vp[t] = vpay
                xval: uint8 = xc[19:20]
                xv[t] = xval
                xpay: uint32 = xc[0:19]
                xp[t] = xpay
                mval: uint8 = 0
                mval[2:3] = mc[17:18]  # vmatload_valid
                mval[1:2] = mc[11:12]  # vmatpush_valid
                mval[0:1] = mc[5:6]  # vmatpop_valid
                mv[t] = mval
                mpay: uint16 = 0
                mpay[10:15] = mc[12:17]
                mpay[5:10] = mc[6:11]
                mpay[0:5] = mc[0:5]
                mp[t] = mpay
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

    return top


VARIANTS = {"locked": (locked, run_locked)}
