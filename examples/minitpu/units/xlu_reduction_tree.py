# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: ``xlu_reduction_tree``, the cross-lane BF16 sum/max tree.

``N`` leaves, ``LEVELS = log2(N)`` levels, two pipeline stages per level (a
``vpu_bf16_add_pipe`` and a two-register max select per element, the op tag
choosing between them at the level's output), one leaf register: the root is
``1 + 2*LEVELS`` edges deep and accepts a reduction every cycle. A
per-sublane tap at level ``LANE_LEVELS = log2(NUM_LANES)`` gives
``NUM_SUBLANES`` results ``2*(LEVELS - LANE_LEVELS)`` edges earlier. Leaves
and the whole payload are unreset; ``valid_q``/``op_q`` are reset. Driven as
a ``trace`` unit (``data_i`` is ``N * 16`` bits); reference
``ref.xlu_reduction_tree_trace``.

Geometry comes from ``vpu_pkg``'s defines, not a parameter chain
(``docs/UNITS.md`` §2.1): ``N`` must equal ``NUM_LANES * NUM_SUBLANES``
(an elaboration ``$error``), so a small tree needs ``+define+
MINITPU_NUM_LANES=4``.

Instances:

``n64``
    The shipped tree: 16 lanes x 4 sublanes, ``N = 64``, ``LEVELS = 6``,
    tap at level 4. Declared: root 13 = ``vpu_pkg.sv:65``
    ``VPU_REDUCE_LATENCY``; tap 9 = ``vpu_pkg.sv:66``
    ``VPU_LANE_REDUCE_LATENCY`` (isa_latency.json ``WB_W_REDUCE`` 15 /
    ``WB_W_LANE_REDUCE`` 11 add ``VPU_WB_STAGES`` 2).
``n16``
    ``MINITPU_NUM_LANES=4``: ``N = 16``, ``LEVELS = 4``, tap at level 2.
    Declared by the same formulas (``UNITS.md`` §6: 13 = 1 + 2x6): root 9,
    tap 5. ``UNITS.md`` §1 says no geometry but 16x4 was ever built.
"""

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness.traces import Trace, rng_for, word

SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_bf16_add_pipe.sv",
           "src/core/xlu/xlu_reduction_tree.sv"]
GEOM = {"n64": (16, 4), "n16": (4, 4)}  # (NUM_LANES, NUM_SUBLANES)
LATENCY_SOURCE = "vpu_pkg.sv:65-66 VPU_REDUCE_LATENCY 13, VPU_LANE_REDUCE_LATENCY 9 (= 1 + 2*levels)"


def _n(inst):
    lanes, sub = GEOM[inst]
    return lanes * sub


def _unit(inst):
    lanes, sub = GEOM[inst]
    n = lanes * sub
    return rtl.RtlUnit(
        top="xlu_reduction_tree",
        sources=SOURCES,
        inputs=[("rst_ni", 1), ("valid_i", 1), ("op_i", 1), ("data_i", 16 * n)],
        outputs=[("valid_o", 1, "post"), ("result_o", 16, "post"),
                 ("lane_valid_o", 1, "post"), ("lane_result_o", 16 * sub, "post")],
        shape="trace",
        params={} if n == 64 else {"N": n},
        defines=[] if lanes == 16 else [f"MINITPU_NUM_LANES={lanes}"],
        assertions=True,
    )


INSTANCES = {k: _unit(k) for k in GEOM}
DEFAULT = "n64"
RTL = INSTANCES[DEFAULT]


def declared(inst):
    lanes, _ = GEOM[inst]
    levels = (_n(inst)).bit_length() - 1
    tap = lanes.bit_length() - 1
    return 1 + 2 * levels, 1 + 2 * tap


def REF(inst, cmd):
    lanes, _ = GEOM[inst]
    return ref.xlu_reduction_tree_trace(cmd, _n(inst), lanes)


def pack_leaves(vals):
    v = 0
    for i, x in enumerate(vals):
        v |= (int(x) & 0xFFFF) << (16 * i)
    return v


CORNERS = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x0001, 0x8001,
           0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80, 0x3F81, 0xBF81, 0x4000, 0x3B80]


def _defaults(inst):
    return {"rst_ni": 1, "valid_i": 0, "op_i": 0, "data_i": 0}


def _start(inst):
    t = Trace(_defaults(inst))
    t.idle(4, rst_ni=0)
    t.idle(2)
    return t


def _bf16(rng):
    r = rng.random()
    if r < 0.15:
        return rng.choice(CORNERS)
    if r < 0.6:  # a narrow exponent band: real cancellation and ties
        return (rng.getrandbits(1) << 15) | ((124 + rng.randrange(8)) << 7) | rng.getrandbits(7)
    return word(rng, 16)


def directed(inst):
    n = _n(inst)
    lanes, sub = GEOM[inst]
    out = []
    # tb_xlu_lane_tap's leakage pattern: every lane of sublane s holds 2**s
    t = _start(inst)
    pat = [(0x3F80 + (s << 7)) for s in range(sub) for _ in range(lanes)]
    t.cycle(valid_i=1, op_i=0, data_i=pack_leaves(pat))
    t.cycle(valid_i=1, op_i=1, data_i=pack_leaves(pat))
    t.idle(20)
    out.append(("lane-leak", t.cmd()))
    # corners crossed: each corner against every other in every position
    rng = rng_for("xlu", inst, "corners")
    t = _start(inst)
    for a in CORNERS:
        for op in (0, 1):
            vals = [a if i % 2 == 0 else rng.choice(CORNERS) for i in range(n)]
            t.cycle(valid_i=1, op_i=op, data_i=pack_leaves(vals))
            t.cycle(valid_i=1, op_i=op, data_i=pack_leaves([rng.choice(CORNERS) for _ in range(n)]))
    t.idle(20)
    out.append(("corners", t.cmd()))
    # mixed SUM/MAX on every cycle, back to back (UNITS.md 8.4)
    rng = rng_for("xlu", inst, "mixed")
    t = _start(inst)
    for i in range(400):
        t.cycle(valid_i=1, op_i=i % 2, data_i=pack_leaves([_bf16(rng) for _ in range(n)]))
    t.idle(20)
    out.append(("mixed-sum-max", t.cmd()))
    # a reset in the middle of a stream of valid reductions
    t = _start(inst)
    for i in range(60):
        t.cycle(rst_ni=int(not (25 <= i < 27)), valid_i=1, op_i=(i // 3) % 2,
                data_i=pack_leaves([_bf16(rng) for _ in range(n)]))
    t.idle(20)
    out.append(("reset-mid-stream", t.cmd()))
    return out


def random_trace(inst, ncyc, seed):
    n = _n(inst)
    rng = rng_for("xlu", inst, seed)
    t = _start(inst)
    for _ in range(ncyc):
        rst = rng.random() < 0.004
        t.cycle(rst_ni=int(not rst), valid_i=int(rng.random() < 0.7), op_i=rng.getrandbits(1),
                data_i=pack_leaves([_bf16(rng) for _ in range(n)]))
    return t.cmd()


def traces(inst):
    tr = [(lab, c, True) for lab, c in directed(inst)]
    m = 6000 if inst == "n64" else 20000
    tr += [(f"random-{s}", random_trace(inst, m, s), True) for s in range(3)]
    return tr


def probes(inst):
    u = INSTANCES[inst]
    n = _n(inst)
    root, tap = declared(inst)
    res = []
    for port, decl in (("result_o", root), ("lane_result_o", tap)):
        t = _start(inst)
        t.idle(24, valid_i=1, data_i=pack_leaves([0x3F80] * n))
        ev = len(t)
        t.idle(24, valid_i=1, data_i=pack_leaves([0x4000] * n))
        res.append((f"{port} (data step)", decl, rtl.probe_trace(u, t.cmd(), port, ev)))
    for port, decl in (("valid_o", root), ("lane_valid_o", tap)):
        t = _start(inst)
        t.idle(24)
        ev = len(t)
        t.idle(24, valid_i=1)
        res.append((f"{port} (valid step)", decl, rtl.probe_trace(u, t.cmd(), port, ev)))
    # op tag: SUM -> MAX on equal data (sum 2x, max x)
    t = _start(inst)
    t.idle(24, valid_i=1, op_i=0, data_i=pack_leaves([0x3F80] * n))
    ev = len(t)
    t.idle(24, valid_i=1, op_i=1, data_i=pack_leaves([0x3F80] * n))
    res.append(("result_o (op step)", root, rtl.probe_trace(u, t.cmd(), "result_o", ev)))
    return res


def seeds():
    """tb_reduce_pipe (the tree itself) and tb_xlu_lane_tap (the ``xlu``
    wrapper, same port names), both at the shipped 16x4."""
    from examples.minitpu.harness import vcd_seed

    out = []
    cmd, seen = vcd_seed.extract(
        "tb_reduce_pipe",
        ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_bf16_add.sv",
         "src/core/vpu/vpu_bf16_add_pipe.sv", "src/core/xlu/xlu_reduction_tree.sv"],
        dut="dut", clk="clk_i", unit=INSTANCES["n64"])
    out.append(("tb_reduce_pipe", "n64", cmd, seen, True))
    cmd, seen = vcd_seed.extract(
        "tb_xlu_lane_tap",
        ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv",
         "src/core/sequencer/sequencer_pkg.sv", "src/core/vpu/vpu_bf16_add.sv",
         "src/core/vpu/vpu_bf16_add_pipe.sv", "src/core/xlu/xlu_reduction_tree.sv",
         "src/core/xlu/xlu.sv"],
        dut="dut", clk="clk_i", unit=INSTANCES["n64"])
    out.append(("tb_xlu_lane_tap", "n64", cmd, seen, True))
    return out


# ---------------------------------------------------------------------------
# Allo expressions (U3 track A; ``dev/records/minitpu/u3_track_a_2026-10-04.rst``)
#
# ``bits`` (plan T1)   one kernel: ``N`` leaves, ``log2 N`` levels of
#                      ``bf16_add.add_bits`` or the ``bf16_gt`` select on the
#                      wavefront's op, the tap at level ``log2 NUM_LANES``;
#                      the ``1 + 2*LEVELS`` register edges of ``.sv`` are the
#                      output's delay line, written as data (checkpoint 6):
#                      the valid pipes are cleared by ``rst_ni``, the payload
#                      pipes never are (P-9). Untimed on the simulator; the
#                      depth that Catapult picks is track C's (H3).
# ``units`` (T2)       ``xlu_reduction_tree_units.py``: the same tree as
#                      ``N - 1`` composed adder units with derived legality.
#
# Ports are lane arrays (P-8): ``data_i`` arrives as ``uint16[n, N]``,
# ``lane_result_o`` leaves as ``uint16[n, NUM_SUBLANES]``; ``_run`` packs.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

import allo.dataflow as df  # noqa: E402
from allo.ir.types import int32, uint8, uint16  # noqa: E402
from examples.minitpu.units.alu import bf16_gt  # noqa: E402
from examples.minitpu.units.bf16_add import add_bits  # noqa: E402

WIDTH = {k: 16 for k in GEOM}
RESP = ("valid_o", "result_o", "lane_valid_o", "lane_result_o")
# ``check.py`` hands a runner ``(mod, cmd, n, w)`` and not the instance, so a
# geometry-bearing unit remembers the one it last built (harness note, record).
_BUILT = {}


def _args(cmd, n, inst):
    nl = _n(inst)
    _, sub = GEOM[inst]
    ins = [np.asarray(cmd[p][:n], dtype=np.uint8) for p in ("rst_ni", "valid_i", "op_i")]
    ins.append(ref._split16(rtl.pack(cmd["data_i"][:n], 16 * nl), nl).astype(np.uint16))
    outs = [np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint16),
            np.zeros(n, dtype=np.uint8), np.zeros((n, sub), dtype=np.uint16)]
    return ins, outs


def _run(mod, cmd, n, w):
    inst = _BUILT["inst"]
    ins, outs = _args(cmd, n, inst)
    mod(*ins, *outs)
    vo, ro, lvo, lro = outs
    _, sub = GEOM[inst]
    lane = rtl.unpack(ref._join16(lro.astype(np.int64), (16 * sub + 63) // 64))
    return {"valid_o": vo, "result_o": ro, "lane_valid_o": lvo, "lane_result_o": lane}


def bits(n, w=16, inst="n64"):
    nl = _n(inst)
    lanes, sub = GEOM[inst]
    root_d, tap_d = declared(inst)  # 1 + 2*LEVELS, 1 + 2*LANE_LEVELS: the pipes' depth
    tap_base = 2 * nl - 2 * sub  # the first node of the level that covers one sublane
    _BUILT["inst"] = inst

    @df.region()
    def top(RST: uint8[n], VLD: uint8[n], OP: uint8[n], D: uint16[n, nl],
            VO: uint8[n], RO: uint16[n], LVO: uint8[n], LRO: uint16[n, sub]):
        @df.kernel(mapping=[1], args=[RST, VLD, OP, D, VO, RO, LVO, LRO])
        def tree(rst: uint8[n], vld: uint8[n], op: uint8[n], d: uint16[n, nl],
                 vo: uint8[n], ro: uint16[n], lvo: uint8[n], lro: uint16[n, sub]):
            # node[0:N] are the leaves; node[N + m] has children 2m, 2m + 1,
            # so one ascending loop is the balanced tree (ip/units/reduction_tree.py)
            node: uint16[2 * nl - 1]
            vq: uint8[root_d]  # the valid pipe to the root: reset
            rq: uint16[root_d]  # its payload: never reset
            lvq: uint8[tap_d]
            lrq: uint16[tap_d, sub]
            for k in range(root_d):
                vq[k] = 0
            for k in range(tap_d):
                lvq[k] = 0
            for t in range(n):
                r: uint8 = rst[t]
                v: uint8 = vld[t]
                o: uint8 = op[t]
                for l in range(nl):
                    node[l] = d[t, l]
                for m in range(nl - 1):
                    a: uint16 = node[2 * m]
                    b: uint16 = node[2 * m + 1]
                    if o == 0:
                        node[nl + m] = add_bits(a, b)
                    else:
                        if bf16_gt(a, b):
                            node[nl + m] = a
                        else:
                            node[nl + m] = b
                # the register edges: stage k takes stage k - 1, stage 0 the tree
                for k in range(1, root_d):
                    vq[root_d - k] = vq[root_d - k - 1]
                    rq[root_d - k] = rq[root_d - k - 1]
                vq[0] = v
                rq[0] = node[2 * nl - 2]
                for j in range(1, tap_d):
                    lvq[tap_d - j] = lvq[tap_d - j - 1]
                    for s in range(sub):
                        lrq[tap_d - j, s] = lrq[tap_d - j - 1, s]
                lvq[0] = v
                for s in range(sub):
                    lrq[0, s] = node[tap_base + s]
                # synchronous reset of every valid stage; the payload keeps flowing
                if r == 0:
                    for k in range(root_d):
                        vq[k] = 0
                    for k in range(tap_d):
                        lvq[k] = 0
                # the lane array first: SystemC csim keeps only the first of the
                # last iteration's stores to a 2-D output when they come last (A5)
                for s in range(sub):
                    lro[t, s] = lrq[tap_d - 1, s]
                vo[t] = vq[root_d - 1]
                ro[t] = rq[root_d - 1]
                lvo[t] = lvq[tap_d - 1]

    return top


def units(n, w=16, inst="n64"):
    """T2: the tree as ``N - 1`` composed adder units (``xlu_tree_compose.py``)."""
    from examples.minitpu.units import xlu_tree_compose

    lanes, sub = GEOM[inst]
    _BUILT["inst"] = inst
    return xlu_tree_compose.architecture(lanes, sub, n).region()


VARIANTS = {"bits": (bits, _run), "units": (units, _run)}
