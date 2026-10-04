# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: ``xlu_transpose``, the 16 x 16 tile transpose (vtxin / vtxout).

A ``NUM_LANES x NUM_LANES`` tile register ``tile_q``, never reset. A write
(``vtxin`` beat ``q``) lands four whole tile rows ``4q .. 4q+3`` from the four
sublanes of ``write_data_i``; a read (``vtxout`` beat ``q``) is a registered
crossing mux, ``read_data_o[s][l] = tile[l][4q + s]``, one edge later. Only
``read_valid_q`` is reset; ``read_data_q`` loads every cycle. Driven as a
``trace`` unit; reference ``ref.xlu_transpose_trace``.

Geometry is fixed at the shipped 16 lanes x 4 sublanes: the module has no
parameter, and its row select ``{index[1:0], sublane[1:0]}`` hard-codes four
beats of four sublanes, so a 16-row tile; ``MINITPU_NUM_LANES=4`` would
elaborate a 4 x 4 tile indexed out of range. One instance.

Declared latency 1: ``vpu_pkg.sv:67`` ``VPU_TXOUT_LATENCY = 1``;
``isa_latency.json`` ``WB_W_TXOUT = 3`` (= 1 + ``VPU_WB_STAGES`` 2), measured
at VPU level by ``tb_vpu_latency_probe``. No MiniTPU tb drives this module
alone (only ``tb_vpu_latency_probe``, ``tb_port_exclusivity`` and
``tb_bundle_encoding``, through ``vpu``/the bundle path), so there is no
unit-level seed.
"""

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness.traces import Trace, rng_for, word

LANES, SUB = 16, 4
W = LANES * SUB * 16  # 1024

RTL = rtl.RtlUnit(
    top="xlu_transpose",
    sources=["src/core/vpu/vpu_pkg.sv", "src/core/xlu/xlu_transpose.sv"],
    inputs=[("rst_ni", 1), ("write_valid_i", 1), ("write_index_i", 2), ("write_data_i", W),
            ("read_valid_i", 1), ("read_index_i", 2)],
    outputs=[("read_valid_o", 1, "post"), ("read_data_o", W, "post")],
    shape="trace",
    assertions=True,
)
INSTANCES = {"t16x4": RTL}
DEFAULT = "t16x4"
LATENCY_SOURCE = "vpu_pkg.sv:67 VPU_TXOUT_LATENCY = 1 (isa_latency.json WB_W_TXOUT 3 = 1 + VPU_WB_STAGES 2)"


def REF(inst, cmd):
    return ref.xlu_transpose_trace(cmd, LANES, SUB)


def _defaults():
    return {"rst_ni": 1, "write_valid_i": 0, "write_index_i": 0, "write_data_i": 0,
            "read_valid_i": 0, "read_index_i": 0}


def _start():
    t = Trace(_defaults())
    t.idle(4, rst_ni=0)
    t.idle(1)
    return t


def _pack(vals):
    v = 0
    for i, x in enumerate(vals):
        v |= (int(x) & 0xFFFF) << (16 * i)
    return v


def _ident(q):
    """Beat ``q`` of a tile whose element (row r, col c) is ``r * 256 + c``."""
    return _pack([(4 * q + s) * 256 + l for s in range(SUB) for l in range(LANES)])


def directed(inst):
    out = []
    # tile check: the vtxin x4, vtxout x4 sequence of a kernel
    t = _start()
    for q in range(4):
        t.cycle(write_valid_i=1, write_index_i=q, write_data_i=_ident(q))
    for q in range(4):
        t.cycle(read_valid_i=1, read_index_i=q)
    t.idle(3)
    out.append(("tile", t.cmd()))
    # read before all four writes: the unwritten rows are stale (uninit)
    t = _start()
    t.cycle(write_valid_i=1, write_index_i=1, write_data_i=_ident(1))
    for q in range(4):
        t.cycle(read_valid_i=1, read_index_i=q)
    t.idle(2)
    out.append(("partial-tile (illegal)", t.cmd()))
    # write and read in one cycle: the read sees the tile before the write
    t = _start()
    for q in range(4):
        t.cycle(write_valid_i=1, write_index_i=q, write_data_i=_ident(q))
    for q in range(4):
        t.cycle(write_valid_i=1, write_index_i=q, write_data_i=_ident(q) ^ ((1 << W) - 1),
                read_valid_i=1, read_index_i=q)
    for q in range(4):
        t.cycle(read_valid_i=1, read_index_i=q)
    t.idle(2)
    out.append(("write+read same cycle", t.cmd()))
    # reset keeps the tile (only read_valid_q is reset)
    t = _start()
    for q in range(4):
        t.cycle(write_valid_i=1, write_index_i=q, write_data_i=_ident(q))
    t.idle(3, rst_ni=0)
    for q in range(4):
        t.cycle(read_valid_i=1, read_index_i=q)
    t.idle(2)
    out.append(("reset-keeps-tile", t.cmd()))
    return out


def random_trace(n, seed):
    rng = rng_for("xtp", seed)
    t = _start()
    for q in range(4):  # fill once so most reads are defined
        t.cycle(write_valid_i=1, write_index_i=q, write_data_i=word(rng, W))
    for _ in range(n):
        t.cycle(rst_ni=int(rng.random() > 0.005),
                write_valid_i=int(rng.random() < 0.4), write_index_i=rng.randrange(4),
                write_data_i=word(rng, W),
                read_valid_i=int(rng.random() < 0.6), read_index_i=rng.randrange(4))
    return t.cmd()


def traces(inst):
    tr = [(lab, c, "illegal" not in lab) for lab, c in directed(inst)]
    tr += [(f"random-{s}", random_trace(20000, s), True) for s in range(3)]
    return tr


def probes(inst):
    res = []
    t = _start()
    for q in range(4):
        t.cycle(write_valid_i=1, write_index_i=q, write_data_i=_ident(q))
    t.idle(6, read_valid_i=1, read_index_i=0)
    ev = len(t)
    t.idle(6, read_valid_i=1, read_index_i=1)
    res.append(("read index -> read_data_o", 1, rtl.probe_trace(RTL, t.cmd(), "read_data_o", ev)))
    t = _start()
    t.idle(6)
    ev = len(t)
    t.idle(6, read_valid_i=1)
    res.append(("read_valid_i -> read_valid_o", 1, rtl.probe_trace(RTL, t.cmd(), "read_valid_o", ev)))
    t = _start()
    for q in range(4):
        t.cycle(write_valid_i=1, write_index_i=q, write_data_i=_ident(q))
    t.idle(6, read_valid_i=1, read_index_i=2)
    ev = len(t)
    t.cycle(write_valid_i=1, write_index_i=0, write_data_i=0, read_valid_i=1, read_index_i=2)
    t.idle(6, read_valid_i=1, read_index_i=2)
    # the write lands at edge ev+1; the read of ev+1 shows it after edge ev+2
    res.append(("write -> read_data_o (visibility + read)", 2,
                rtl.probe_trace(RTL, t.cmd(), "read_data_o", ev)))
    return res


# ---------------------------------------------------------------------------
# Allo expressions (U3 track A; ``dev/records/minitpu/u3_track_a_2026-10-04.rst``)
#
# ``trace`` (plan X1)  one kernel; ``tile`` is ``uint16[16, 16] @
#                      Stateful(reset=False)`` (D-14, P-9: the RTL never
#                      resets it); a write lands four whole rows; the read is
#                      the crossing permutation ``out[s][l] = tile[l][4q + s]``
#                      of the tile before this cycle's write, delivered in the
#                      same row (the RTL's one register edge, ``"post"``).
# Ports are lane arrays (P-8): ``write_data_i``/``read_data_o`` are
# ``uint16[n, 64]`` (field ``s * 16 + l``); ``_run`` packs.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

import allo.dataflow as df  # noqa: E402
from allo.ir.types import Stateful, int32, uint8, uint16  # noqa: E402

WIDTH = {"t16x4": 16}
NF = LANES * SUB  # 64 fields of 16 bits
CMD = ("rst_ni", "write_valid_i", "write_index_i", "write_data_i", "read_valid_i", "read_index_i")


def _args(cmd, n):
    ins = []
    for p in CMD:
        if p == "write_data_i":
            ins.append(ref._split16(rtl.pack(cmd[p][:n], W), NF).astype(np.uint16))
        else:
            ins.append(np.asarray(cmd[p][:n], dtype=np.uint8))
    outs = [np.zeros(n, dtype=np.uint8), np.zeros((n, NF), dtype=np.uint16)]
    return ins, outs


def _run(mod, cmd, n, w):
    ins, outs = _args(cmd, n)
    mod(*ins, *outs)
    rvo, rdo = outs
    return {"read_valid_o": rvo,
            "read_data_o": rtl.unpack(ref._join16(rdo.astype(np.int64), W // 64))}


def trace(n, w=16, inst="t16x4"):
    """X1 with the tile RESET (``@ Stateful``): the form the SystemC emitter
    accepts on a kernel with array ports (D-14's unreset lowering is
    Wire-only, finding A6); a recorded deviation from the RTL's unreset tile.
    """
    @df.region()
    def top(RST: uint8[n], WV: uint8[n], WI: uint8[n], WD: uint16[n, NF],
            RV: uint8[n], RI: uint8[n], RVO: uint8[n], RDO: uint16[n, NF]):
        @df.kernel(mapping=[1], args=[RST, WV, WI, WD, RV, RI, RVO, RDO])
        def tx(rst: uint8[n], wv: uint8[n], wi: uint8[n], wd: uint16[n, NF],
               rv: uint8[n], ri: uint8[n], rvo: uint8[n], rdo: uint16[n, NF]):
            tile: uint16[LANES, LANES] @ Stateful
            for t in range(n):
                r: uint8 = rst[t]
                v: uint8 = rv[t]
                q: int32 = ri[t]  # B4: widen before indexing
                e: uint8 = wv[t]  # S6: every port read unconditionally
                wq: int32 = wi[t]
                # the registered crossing read, of the tile before this cycle's write
                for s in range(SUB):
                    for l in range(LANES):
                        rdo[t, s * LANES + l] = tile[l, 4 * q + s]
                if r == 0:
                    rvo[t] = 0  # read_valid_q is the one reset register
                else:
                    rvo[t] = v
                # a write lands four whole rows
                if e == 1:
                    for s in range(SUB):
                        for l in range(LANES):
                            tile[4 * wq + s, l] = wd[t, s * LANES + l]

    return top


def trace_unreset(n, w=16, inst="t16x4"):
    """X1 as planned (P-9): ``trace`` with ``tile @ Stateful(reset=False)``.
    The simulator treats it as ordinary storage; the SystemC emitter refuses
    it on this kernel shape, naming the storage and the cause (D-14, honest).
    The body is ``trace``'s (a closure flag would become an ``scf.if``)."""
    @df.region()
    def top(RST: uint8[n], WV: uint8[n], WI: uint8[n], WD: uint16[n, NF],
            RV: uint8[n], RI: uint8[n], RVO: uint8[n], RDO: uint16[n, NF]):
        @df.kernel(mapping=[1], args=[RST, WV, WI, WD, RV, RI, RVO, RDO])
        def tx(rst: uint8[n], wv: uint8[n], wi: uint8[n], wd: uint16[n, NF],
               rv: uint8[n], ri: uint8[n], rvo: uint8[n], rdo: uint16[n, NF]):
            tile: uint16[LANES, LANES] @ Stateful(reset=False)  # D-14, P-9
            for t in range(n):
                r: uint8 = rst[t]
                v: uint8 = rv[t]
                q: int32 = ri[t]  # B4: widen before indexing
                e: uint8 = wv[t]  # S6: every port read unconditionally
                wq: int32 = wi[t]
                for s in range(SUB):
                    for l in range(LANES):
                        rdo[t, s * LANES + l] = tile[l, 4 * q + s]
                if r == 0:
                    rvo[t] = 0
                else:
                    rvo[t] = v
                if e == 1:
                    for s in range(SUB):
                        for l in range(LANES):
                            tile[4 * wq + s, l] = wd[t, s * LANES + l]

    return top


VARIANTS = {"trace": (trace, _run), "trace_unreset": (trace_unreset, _run)}
