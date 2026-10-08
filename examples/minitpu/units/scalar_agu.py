# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``sequencer_scalar_agu``, the four scalar address registers and one MAC.

S-slot ops ``SMOVI``, ``SMOV_ARG``, ``SADDI`` (sign-extended 24-bit IMM),
``SMAC`` (``sreg[rs] + (use_iv ? iv[level] : sreg[rs]) * imm``, the multiply
before the pipe register and the add after it) and ``SSHL``, written back
``S_LAT`` cycles after issue with **no bypass and no interlock**: the read
ports (a descriptor's base and stride, ``loop.begin.r``'s bound saturated to
16 bits) see committed ``sreg_q`` only. ``units/rtl/u4_scalar_agu.sv`` exposes
the four SREGs (``sreg_o``) so every cycle's state is checked, not only what
the read selectors happen to name.

Reference: ``ref_ctrl_decode.ScalarAgu`` (cycle model). Instances: ``lat2``
(shipped, ``sequencer_pkg::S_LAT``), ``lat1`` (the ``g_lat1`` branch, built
only when a bench overrides ``S_LAT``) and ``lat3``. Seeds: ``tb_bundle_scalar_agu``
part A (the unit alone, ``tb.dutA``) and part B (inside the sequencer,
``tb.dutB.u_scalar_agu``).
"""

import os
import sys

import numpy as np

import allo.dataflow as df
from allo.ir.types import UInt, int32, uint1

from examples.minitpu.harness import rtl
from examples.minitpu.harness import ref_ctrl_decode as R
from examples.minitpu.harness.traces import Trace, rng_for

HERE = os.path.dirname(os.path.abspath(__file__))
PKGS = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/sequencer/sequencer_pkg.sv"]
SOURCES = PKGS + ["src/core/sequencer/sequencer_scalar_agu.sv", os.path.join(HERE, "rtl", "u4_scalar_agu.sv")]
INPUTS = [("rst_n", 1), ("s_valid_i", 1), ("s_op_i", 3), ("s_rd_i", 2), ("s_rs_i", 2), ("s_use_iv_i", 1),
          ("s_level_i", 3), ("s_imm_i", 24), ("kernel_arg_csr_i", 128), ("iv_flat_i", 256),
          ("rd_sel_d_base_i", 2), ("rd_sel_d_stride_i", 2), ("rd_loop_bound_from_arg_i", 1),
          ("rd_sel_loop_bound_i", 2)]
OUTPUTS = [("rd_data_d_base_o", 32, "pre"), ("rd_data_d_stride_o", 32, "pre"),
           ("rd_data_loop_bound_o", 16, "pre"), ("sreg_written_o", 4, "pre"), ("sreg_o", 128, "pre")]


def _unit(s_lat):
    return rtl.RtlUnit(top="u4_scalar_agu", sources=SOURCES, inputs=INPUTS, outputs=OUTPUTS, shape="trace",
                       clk="clk", rst_n="rst_n", params={"S_LAT": s_lat})


LATS = {"lat2": 2, "lat1": 1, "lat3": 3}
INSTANCES = {k: _unit(v) for k, v in LATS.items()}
DEFAULT = "lat2"
RTL = INSTANCES[DEFAULT]
LATENCY_SOURCE = "S_LAT = 2 (sequencer_pkg.sv:233; isa_latency.json scalar.latency 2; asm.py S_LAT)"
SMOVI, SMOV_ARG, SADDI, SMAC, SSHL = 0, 1, 2, 3, 4
KARGS = (0x44444444 << 96) | (0x0001_2345 << 64) | (0xFFFF_FFFF << 32) | 0x0000_0007


def REF(inst, cmd):
    want, reason, ev = R.scalar_agu_trace(cmd, LATS[inst])
    # row 0: the async reset has had no edge yet (the trace starts low, no negedge)
    for p in reason:
        reason[p][0] = "before reset"
    return want, reason, ev


def _flat(ivs):
    return sum((v & 0xFFFFFFFF) << (32 * i) for i, v in enumerate(ivs))


def _trace():
    t = Trace({p: 0 for p, _ in INPUTS} | {"kernel_arg_csr_i": KARGS,
                                           "iv_flat_i": _flat([50, 3, 0x10000, 7, 1, 2, 0xFFFFFFFF, 9])})
    t.idle(2, rst_n=0)
    t.defaults["rst_n"] = 1
    return t


def op(t, o, rd, rs=0, imm=0, use_iv=0, level=0, **reads):
    t.cycle(s_valid_i=1, s_op_i=o, s_rd_i=rd, s_rs_i=rs, s_imm_i=imm & 0xFFFFFF, s_use_iv_i=use_iv,
            s_level_i=level, **reads)


def directed():
    out = []
    # no bypass: a write read at +0, +1, +2, +3 on every read port
    t = _trace()
    op(t, SMOVI, 0, imm=500, rd_sel_d_base_i=0)
    for k in range(4):
        t.cycle(rd_sel_d_base_i=0, rd_sel_d_stride_i=0, rd_sel_loop_bound_i=0)
    op(t, SADDI, 2, rs=0, imm=-5)                      # reads sreg0 exactly S_LAT after its write
    op(t, SADDI, 3, rs=2, imm=1)                       # reads sreg2 at +1: the OLD value
    t.idle(1)
    op(t, SADDI, 1, rs=2, imm=1)                       # +2 at S_LAT=2: the new one
    t.idle(4)
    out.append(("no-bypass", t.cmd(), True))
    # every op; SMAC with an iv and with an sreg; SSHL with high imm bits; SADDI negative
    t = _trace()
    op(t, SMOVI, 0, imm=0xFFFFFF)
    op(t, SMOV_ARG, 1, rs=3)
    op(t, SMOV_ARG, 2, rs=1)
    t.idle(3)
    op(t, SMAC, 3, rs=0, imm=3, use_iv=1, level=0)     # 0xFFFFFF + 50*3
    op(t, SMAC, 1, rs=2, imm=0x800001, use_iv=0)      # sreg2 + sreg2*imm, wraps 32 bits
    op(t, SMAC, 2, rs=0, imm=2, use_iv=1, level=6)     # iv 0xFFFFFFFF * 2 wraps
    op(t, SSHL, 0, rs=0, imm=0xFFFFE5)                 # only imm[4:0] = 5
    op(t, SADDI, 3, rs=3, imm=-(1 << 23))
    t.idle(4)
    out.append(("every-op", t.cmd(), True))
    # loop bound: saturation from an sreg and from a kernel argument
    t = _trace()
    op(t, SMOVI, 0, imm=0xFFFF)
    op(t, SMOVI, 1, imm=0x10000)
    t.idle(3)
    for src in range(4):
        for arg in (0, 1):
            t.cycle(rd_loop_bound_from_arg_i=arg, rd_sel_loop_bound_i=src)
    out.append(("loop-bound-saturation", t.cmd(), True))
    # two writes to one register in flight; undefined ops 5..7 (write zero)
    t = _trace()
    op(t, SMOVI, 2, imm=11)
    op(t, SMOVI, 2, imm=22)
    op(t, SADDI, 2, rs=2, imm=1)
    t.idle(3)
    for bad in (5, 6, 7):
        op(t, SMOVI, bad - 5, imm=99)
        op(t, bad, bad - 5, rs=2, imm=0x123)
        t.idle(3)
    out.append(("inflight-and-undefined", t.cmd(), False))
    # reset with ops in flight
    t = _trace()
    op(t, SMOVI, 0, imm=7)
    op(t, SMOVI, 1, imm=8)
    t.cycle(rst_n=0)
    op(t, SMOVI, 3, imm=9)
    t.idle(4)
    out.append(("reset-in-flight", t.cmd(), True))
    return out


def random_trace(rng, n):
    t = _trace()
    for _ in range(n):
        if rng.random() < 0.02:
            t.defaults["kernel_arg_csr_i"] = rng.getrandbits(128)
        if rng.random() < 0.05:
            t.defaults["iv_flat_i"] = rng.getrandbits(256)
        reads = dict(rd_sel_d_base_i=rng.getrandbits(2), rd_sel_d_stride_i=rng.getrandbits(2),
                     rd_loop_bound_from_arg_i=rng.getrandbits(1), rd_sel_loop_bound_i=rng.getrandbits(2))
        if rng.random() < 0.6:
            imm = rng.choice([rng.getrandbits(24), rng.getrandbits(4), (1 << 24) - rng.getrandbits(4) - 1])
            op(t, rng.choice([0, 1, 2, 3, 3, 4]), rng.getrandbits(2), rs=rng.getrandbits(2), imm=imm,
               use_iv=rng.getrandbits(1), level=rng.getrandbits(3), **reads)
        else:
            t.cycle(**reads)
        if rng.random() < 0.002:
            t.cycle(rst_n=0)
    return t.cmd()


def traces(inst):
    rng = rng_for("scalar_agu", inst)
    return directed() + [(f"random-{s}", random_trace(rng, 20000), True) for s in range(2)]


def _seed(tb_scope, label):
    names = ["rst_n", "s_valid_i", "s_op_i", "s_rd_i", "s_rs_i", "s_use_iv_i", "s_level_i", "s_imm_i",
             "kernel_arg_csr_i", "rd_sel_d_base_i", "rd_sel_d_stride_i", "rd_loop_bound_from_arg_i",
             "rd_sel_loop_bound_i"] + [f"iv_by_level_i[{i}]" for i in range(8)] + \
            ["rd_data_d_base_o", "rd_data_d_stride_o", "rd_data_loop_bound_o", "sreg_written_o"] + \
            [f"sreg_q[{i}]" for i in range(4)]
    srcs = PKGS + [f"src/core/sequencer/{f}" for f in (
        "sequencer_decoder.sv", "sequencer_iram.sv", "sequencer_fetch_queue.sv", "sequencer_loop_buffer.sv",
        "sequencer_loop_ctrl.sv", "sequencer_agu_resolve.sv", "sequencer_scalar_agu.sv", "dma_desc_adapter.sv",
        "sequencer_vpu_adapter.sv", "sequencer.sv")]
    rows = R.seed_rows("tb_bundle_scalar_agu", srcs, tb_scope, names, clk="clk")
    n = len(rows["rst_n"])
    cmd = {p: rows[p] for p, _ in INPUTS if p != "iv_flat_i"}
    cmd["iv_flat_i"] = [_flat([rows[f"iv_by_level_i[{i}]"][t] for i in range(8)]) for t in range(n)]
    seen = {p: rows[p] for p, _, _ in OUTPUTS if p != "sreg_o"}
    seen["sreg_o"] = [_flat([rows[f"sreg_q[{i}]"][t] for i in range(4)]) for t in range(n)]
    return (label, "lat2", cmd, seen, True)


def seeds():
    return [_seed("dutA", "tb_bundle_scalar_agu:A"), _seed("dutB.u_scalar_agu", "tb_bundle_scalar_agu:B")]


def probes(inst):
    u = INSTANCES[inst]
    res = []
    for port, sel in (("sreg_o", {}), ("rd_data_d_base_o", {"rd_sel_d_base_i": 1}),
                      ("rd_data_loop_bound_o", {"rd_sel_loop_bound_i": 1})):
        t = _trace()
        t.defaults.update(sel)
        t.idle(4)
        ev = len(t)
        op(t, SMOVI, 1, imm=1234)
        t.idle(6)
        cmd = {p: rtl.pack(v, w) for (p, w), v in zip(INPUTS, (t.cmd()[p] for p, _ in INPUTS))}
        res.append((f"S-op issue -> {port}", LATS[inst], rtl.probe_trace(u, cmd, port, ev)))
    return res


# ---------------------------------------------------------------------------
# Allo (U4 track A, plan S1). One kernel, one iteration per cycle (the trace
# convention): the read ports from committed ``sreg`` first ("pre"), then the
# edge. The ``S_LAT``-deep write pipeline is DATA (U3 checkpoint 6): ``S_LAT``
# entries of ``{valid, rd, data, product, is_mac}``; each edge shifts them
# and puts the issuing op in entry 0, then commits entry ``S_LAT - 1`` --
# ``S_LAT = 1`` commits the op of this very edge (RTL ``g_lat1``), ``S_LAT = 2``
# commits the one registered an edge earlier (``g_latN``). No bypass: the
# read ports never look into the pipe. SMAC's add happens at commit, after
# the pipe register, as the RTL's ``final_data``. ``S_LAT`` comes from the
# instance (``LATS``); it is the number ``asm.py`` consumes (a D-20 booking,
# ``template/control_geometry.py``).
# ---------------------------------------------------------------------------

WIDTH = {k: 32 for k in LATS}
U32 = UInt(32)


def scalar_result(op: UInt(3), base: UInt(32), imm: UInt(24), arg: UInt(32)) -> UInt(32):
    """``result_prepipe_comb``: SMAC's pre-pipe value is the base alone."""
    r: UInt(32) = 0
    if op == 0:  # SMOVI: zero-extend
        r = imm
    elif op == 1:  # SMOV_ARG
        r = arg
    elif op == 2:  # SADDI: sign-extend IMM
        e: UInt(32) = imm
        if imm[23]:
            e[24:32] = 0xFF
        r = base + e
    elif op == 3:  # SMAC: + product after the pipe register
        r = base
    elif op == 4:  # SSHL
        sh: UInt(5) = imm[0:5]
        r = base << sh
    return r


def s1(n, w, inst):
    SL = LATS[inst]

    @df.region()
    def top(RST: uint1[n], SV: uint1[n], SOP: UInt(3)[n], SRD: UInt(2)[n], SRS: UInt(2)[n], SIV: uint1[n],
            SLV: UInt(3)[n], SIMM: U32[n], KA: U32[n, 4], IV: U32[n, 8], SB: UInt(2)[n], SS: UInt(2)[n],
            LA: uint1[n], SLB: UInt(2)[n],
            SREG: U32[n, 4], OB: U32[n], OS: U32[n], OL: U32[n], OW: U32[n]):
        @df.kernel(mapping=[1], args=[RST, SV, SOP, SRD, SRS, SIV, SLV, SIMM, KA, IV, SB, SS, LA, SLB,
                                      SREG, OB, OS, OL, OW])
        def agu(rst: uint1[n], sv: uint1[n], sop: UInt(3)[n], srd: UInt(2)[n], srs: UInt(2)[n],
                siv: uint1[n], slv: UInt(3)[n], simm: U32[n], ka: U32[n, 4], iv: U32[n, 8],
                sb: UInt(2)[n], ss: UInt(2)[n], la: uint1[n], slb: UInt(2)[n],
                sreg_o: U32[n, 4], ob: U32[n], os: U32[n], ol: U32[n], ow: U32[n]):
            sreg: U32[4] = 0
            written: UInt(4) = 0
            pv: uint1[SL] = 0
            prd: UInt(2)[SL] = 0
            pdata: U32[SL] = 0
            pprod: U32[SL] = 0
            pmac: uint1[SL] = 0
            for t in range(n):
                live: uint1 = rst[t]
                if live == 0:  # async reset: the row shows the reset state
                    written = 0
                    for k in range(4):
                        sreg[k] = 0
                    for k in range(SL):
                        pv[k] = 0
                        prd[k] = 0
                        pdata[k] = 0
                        pprod[k] = 0
                        pmac[k] = 0
                # ---- "pre" outputs: committed sreg only (no bypass) ----
                for k in range(4):
                    sreg_o[t, k] = sreg[k]  # A5: the 2-D output stored first
                ib: int32 = sb[t]
                ob[t] = sreg[ib]
                i_s: int32 = ss[t]
                os[t] = sreg[i_s]
                il: int32 = slb[t]
                raw: U32 = sreg[il]
                if la[t]:
                    raw = ka[t, il]
                bound: U32 = raw[0:16]
                if raw[16:32] != 0:
                    bound = 0xFFFF
                ol[t] = bound
                ow[t] = written
                # ---- rising edge ----
                if live:
                    irs: int32 = srs[t]
                    ilv: int32 = slv[t]
                    imm: UInt(24) = simm[t]
                    operand: U32 = sreg[irs]
                    if siv[t]:
                        operand = iv[t, ilv]
                    prod: U32 = operand * imm
                    pre: U32 = scalar_result(sop[t], sreg[irs], imm, ka[t, irs])
                    for j in range(SL - 1):
                        k: int32 = SL - 1 - j
                        pv[k] = pv[k - 1]
                        prd[k] = prd[k - 1]
                        pdata[k] = pdata[k - 1]
                        pprod[k] = pprod[k - 1]
                        pmac[k] = pmac[k - 1]
                    pv[0] = sv[t]
                    prd[0] = srd[t]
                    pdata[0] = pre
                    pprod[0] = prod
                    pmac[0] = sop[t] == 3
                    written = 0
                    if pv[SL - 1]:
                        fin: U32 = pdata[SL - 1]
                        if pmac[SL - 1]:
                            fin = fin + pprod[SL - 1]
                        iw: int32 = prd[SL - 1]
                        sreg[iw] = fin
                        written[iw] = 1

    return top


S_IN = [("rst_n", np.uint8), ("s_valid_i", np.uint8), ("s_op_i", np.uint8), ("s_rd_i", np.uint8),
        ("s_rs_i", np.uint8), ("s_use_iv_i", np.uint8), ("s_level_i", np.uint8), ("s_imm_i", np.uint32)]
S_RD = [("rd_sel_d_base_i", np.uint8), ("rd_sel_d_stride_i", np.uint8), ("rd_loop_bound_from_arg_i", np.uint8),
        ("rd_sel_loop_bound_i", np.uint8)]


def run_s1(mod, cmd, n, w):
    from examples.minitpu.units.ctrl_lanes import col, join, split

    sreg = np.zeros((n, 4), dtype=np.uint32)
    outs = [np.zeros(n, dtype=np.uint32) for _ in range(4)]
    mod(*[col(cmd, p, n, d) for p, d in S_IN], split(cmd["kernel_arg_csr_i"][:n], 4),
        split(cmd["iv_flat_i"][:n], 8), *[col(cmd, p, n, d) for p, d in S_RD], sreg, *outs)
    return {"sreg_o": join(sreg), "rd_data_d_base_o": outs[0], "rd_data_d_stride_o": outs[1],
            "rd_data_loop_bound_o": outs[2], "sreg_written_o": outs[3]}


VARIANTS = {"s1": (s1, run_s1)}

if __name__ == "__main__":
    sys.exit(0)
