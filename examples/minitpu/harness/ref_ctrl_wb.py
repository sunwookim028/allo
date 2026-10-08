# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: the VREG write-port calendar and the ``vpu_ctrl_t`` interface.

``VPU_CTRL_FIELDS`` is ``vpu_pkg.sv:148-181``'s packed struct, MSB first
(``vpu_req_t vmem`` spelled out as its four fields): the layout a trace packs
``ctrl_i`` with. ``u4_vpu_wb.sv`` reports ``$bits(vpu_ctrl_t)`` on ``ctrl_w_o``
so a drifted table fails the first row of every trace.

The calendar reference (:func:`writeback_trace`) predicts, per cycle, which of
``vpu.sv``'s six writeback sources drive the OR-muxed write port (``wb_src``),
the two registers after it (``wb_stage_valid``, ``wb_local_valid``/``addr``)
from the ``vpu_ctrl_t`` command trace alone: an op issued in cycle ``t`` drives
the mux in ``t + L`` (its unit's latency, ``vpu_pkg``) and the VREG write
enable in ``t + W``, ``W = L + VPU_WB_STAGES``. The matrix pop is the one source
that is not a fixed offset (it waits for the MXU's result), so it is predicted
only for pops issued after their result is ready.
"""

import numpy as np

# (name, width), MSB first: vpu_pkg.sv:148-181
VPU_CTRL_FIELDS = [
    ("raddr_a", 5), ("raddr_b", 5),
    ("alu_valid", 1), ("txin_valid", 1), ("txin_index", 2), ("txout_valid", 1),
    ("txout_index", 2), ("txout_vd", 5), ("alu_op", 4), ("alu_vd", 5),
    ("sfu_valid", 1), ("sfu_op", 2), ("sfu_vd", 5),
    ("reduce_valid", 1), ("reduce_op", 1), ("reduce_lane", 1), ("reduce_vd", 5),
    ("vmem_valid", 1), ("vmem_op", 1), ("vmem_vreg_idx", 5), ("vmem_address", 12),
    ("vmem_store_read_hint", 1),
    ("vmatload_valid", 1), ("vmatload_base", 5), ("vmatpush_valid", 1), ("vmatpush_vs", 5),
    ("vmatpop_valid", 1), ("vmatpop_vd", 5),
]
CTRL_W = sum(w for _, w in VPU_CTRL_FIELDS)  # 85
_OFF = {}
_o = CTRL_W
for _n, _w in VPU_CTRL_FIELDS:
    _o -= _w
    _OFF[_n] = (_o, _w)
VALIDS = ("alu_valid", "txin_valid", "txout_valid", "sfu_valid", "reduce_valid", "vmem_valid",
          "vmatload_valid", "vmatpush_valid", "vmatpop_valid")


def pack_ctrl(**f):
    v = 0
    for n, x in f.items():
        o, w = _OFF[n]
        assert 0 <= x < (1 << w), (n, x)
        v |= x << o
    return v


def unpack_ctrl(v):
    return {n: (int(v) >> o) & ((1 << w) - 1) for n, (o, w) in _OFF.items()}


# vpu_pkg.sv:57-67 and sequencer_pkg.sv's WB_W_* (W = L + VPU_WB_STAGES)
VPU_WB_STAGES = 2
L = {"load": 1 + 3, "alu": 3, "sfu": 5, "reduce": 13, "lane_reduce": 9, "txout": 1, "mpop": 1}
W = {k: v + VPU_WB_STAGES for k, v in L.items()}
# wb_source_valid bit of each class (vpu.sv:290-292)
SRC_BIT = {"load": 0, "alu": 1, "sfu": 2, "reduce": 3, "lane_reduce": 3, "mpop": 4, "txout": 5}


def op_class(f):
    """The VREG-writing classes this cycle's command starts: [(class, vd)]."""
    out = []
    if f["alu_valid"]:
        out.append(("alu", f["alu_vd"]))
    if f["sfu_valid"]:
        out.append(("sfu", f["sfu_vd"]))
    if f["reduce_valid"]:
        out.append(("lane_reduce" if f["reduce_lane"] else "reduce", f["reduce_vd"]))
    if f["txout_valid"]:
        out.append(("txout", f["txout_vd"]))
    if f["vmem_valid"] and not f["vmem_op"]:
        out.append(("load", f["vmem_vreg_idx"]))
    return out


RESULT_LATENCY = 85  # isa_latency.json matrix.result_latency.vmatpush


def pop_mux_cycles(ctrl, rst):
    """{vmatpop issue cycle: the cycle its beat drives the mux}, by the MXU
    contract: pops pair with pushes in order, and a pop fires the cycle after
    it issues once its push's result is waiting (``RESULT_LATENCY`` after the
    push), else the cycle the result arrives (``mxu_pop_engine.sv:34``)."""
    pushes, out = [], {}
    for t, f in enumerate(ctrl):
        if not rst[t]:
            pushes.clear()
            continue
        if f["vmatpush_valid"]:
            pushes.append(t)
        if f["vmatpop_valid"] and pushes:
            p = pushes.pop(0)
            out[t] = max(t + L["mpop"], p + RESULT_LATENCY + L["mpop"])
    return out


def writeback_trace(cmd, pops=None):
    """Per-cycle prediction of the writeback port from a ``vpu_ctrl_t`` trace.

    ``cmd``: {"rst_ni": [...], "ctrl_i": [...]} (ints). ``pops`` overrides
    :func:`pop_mux_cycles`. Returns ``(want, claims, events)``.
    """
    rst = [int(x) for x in cmd["rst_ni"]]
    ctrl = [unpack_ctrl(int(x)) for x in cmd["ctrl_i"]]
    n = len(rst)
    src = np.zeros(n, dtype=np.int64)
    addr_mux = np.zeros(n, dtype=np.int64)
    claims = []  # (issue cycle, class, vd, mux cycle)
    pops = pop_mux_cycles(ctrl, rst) if pops is None else pops
    for t in range(n):
        if not rst[t]:
            continue
        for cls, vd in op_class(ctrl[t]):
            claims.append((t, cls, vd, t + L[cls]))
        if ctrl[t]["vmatpop_valid"] and t in pops:
            claims.append((t, "mpop", ctrl[t]["vmatpop_vd"], pops[t]))
    # a reset between issue and the mux cancels the claim (every valid pipe is reset)
    events = {}
    per = np.zeros(n, dtype=np.int64)
    for t, cls, vd, m in claims:
        if m >= n or any(not rst[c] for c in range(t + 1, m + 1)):
            continue
        src[m] |= 1 << SRC_BIT[cls]
        addr_mux[m] |= vd
        per[m] += 1
    # Two claims in one cycle merge on the OR mux. vpu.sv's $onehot0 sees it
    # only when they come from two sources; vredsum and vlanered share source
    # bit 3 (the tree's two outputs), so their meeting is silent (finding F-W1).
    loud = int(sum(1 for x in src if bin(int(x)).count("1") > 1))
    silent = int(sum(1 for m in range(n) if per[m] > 1 and bin(int(src[m])).count("1") == 1))
    if loud:
        events["write-port collisions (asserted)"] = loud
    if silent:
        events["write-port collisions (silent, one source bit)"] = silent
    stage = np.zeros(n, dtype=np.int64)
    local = np.zeros(n, dtype=np.int64)
    laddr = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        stage[t] = 0 if not rst[t - 1] else int(src[t - 1] != 0)
        if t >= 2:
            on = rst[t - 1] and rst[t - 2] and src[t - 2] != 0
            local[t] = 0xF if on else 0
            laddr[t] = addr_mux[t - 2] if (rst[t - 1] and rst[t - 2]) else 0
    want = {"ctrl_w_o": np.full(n, CTRL_W), "wb_src_o": src, "wb_stage_valid_o": stage,
            "wb_local_valid_o": local, "wb_local_addr_o": laddr}
    return want, claims, events
