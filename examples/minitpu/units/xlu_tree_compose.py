# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3 T2: ``xlu_reduction_tree`` as ``N - 1`` composed adder *units*.

The same function as ``xlu_reduction_tree.bits`` (T1), built the way the RTL
is built: one adder per node, wired by level over Streams, composed with
``allo.compose`` (P-6: a parameterised unit is a ``compose.unit``). Every
node is an instance of one body, ``tree_node`` -- the wavefront's op (bit 16
of the 17-bit word) selects ``bf16_add.add_bits`` or the ``bf16_gt`` max
and travels on with the result -- in three instance groups: the levels below
the tap (``add_low``), the tap level, which also puts to ``tap`` (``add_tap``),
and the levels above it (``add_high``). ``tree_feed`` puts a leaf word per
lane per cycle, valid or not (the RTL's payload is ungated); ``tree_sink``
owns the valid and payload pipes whose depths are the declared latencies.

**Derived, never declared beside each other** (plan Q4, hypothesis H5):
``LEVELS``, ``TAP_LEVEL``, ``TAP_BASE``, ``ROOT_LATENCY``, ``TAP_LATENCY``
and ``MAX_LATENCY`` all follow from ``LANES``, ``SUB`` and ``ADD_LATENCY``
(the adder unit's declared latency, 2 for ``vpu_bf16_add_pipe``), and
``tree_legality`` refuses a set that disagrees -- MiniTPU checks the max
path's pacing and the tap/booking relation nowhere but in a testbench
(``UNITS.md``). A composed Stream tree is self-timed, so the declared
latencies are consumed only by the sink's pipes here; on cycle-locked RTL
(track C) they are what ``latency=`` would pin.

``refusals()`` builds each wrong set and returns what the composition said.
"""

from __future__ import annotations  # the bodies annotate with architecture parameters

from allo.compose import Architecture, Channel, Memory, unit
from allo.ir.types import UInt, uint1, uint16

from examples.minitpu.units.alu import bf16_gt
from examples.minitpu.units.bf16_add import add_bits

ADD_LATENCY = 2  # vpu_bf16_add_pipe (units/bf16_add_pipe.py): "Two-stage"


def tree_node(wa: UInt(17), wb: UInt(17)) -> UInt(17):
    """One node of the tree on two ``{op, value}`` words: ``vpu_bf16_add`` or
    the ``bf16_gt`` select (ties to ``b``), the op carried on."""
    o: uint1 = wa[16]
    a: uint16 = wa[0:16]
    b: uint16 = wb[0:16]
    r: uint16 = 0
    if o == 0:
        r = add_bits(a, b)
    else:
        if bf16_gt(a, b):
            r = a
        else:
            r = b
    w: UInt(17) = 0
    w[0:16] = r
    w[16] = o
    return w


def parameters(lanes, sub, ncyc, add_latency=ADD_LATENCY, qd=4):
    """The parameter set for a ``lanes x sub`` tree, the derived ones derived."""
    n = lanes * sub
    levels = n.bit_length() - 1
    tap_level = lanes.bit_length() - 1
    return {
        "LANES": lanes, "SUB": sub, "N": n, "NCYC": ncyc, "QD": qd,
        "LEVELS": levels, "TAP_LEVEL": tap_level, "TAP_BASE": 2 * n - 2 * sub,
        "ADD_LATENCY": add_latency, "MAX_LATENCY": add_latency,
        "ROOT_LATENCY": 1 + levels * add_latency,
        "TAP_LATENCY": 1 + tap_level * add_latency,
        "tree_node": tree_node,
    }


def tree_legality(p):
    lanes, sub, n = p["LANES"], p["SUB"], p["N"]
    assert n == lanes * sub, (
        f"N={n} must be LANES*SUB = {lanes * sub}: the leaves are the lanes of "
        f"every sublane (vpu_pkg's elaboration $error, as a legality)")
    assert n >= 2 and n & (n - 1) == 0 and sub & (sub - 1) == 0, (
        f"N={n}, SUB={sub}: a balanced binary tree with a per-sublane tap needs powers of two")
    assert p["LEVELS"] == n.bit_length() - 1, (
        f"LEVELS={p['LEVELS']} must be log2(N) = {n.bit_length() - 1}")
    assert p["TAP_LEVEL"] == lanes.bit_length() - 1, (
        f"TAP_LEVEL={p['TAP_LEVEL']} must be log2(LANES) = {lanes.bit_length() - 1}: the tap "
        f"is the level whose nodes each cover one sublane; a level declared on its own "
        f"reduces a rotated window (MiniTPU's tb-only tap/booking check)")
    assert p["TAP_BASE"] == 2 * n - 2 * sub, (
        f"TAP_BASE={p['TAP_BASE']} must be 2*N - 2*SUB = {2 * n - 2 * sub}: where the tap "
        f"level's nodes sit in the heap numbering; derived from the leaf mapping")
    assert p["MAX_LATENCY"] == p["ADD_LATENCY"], (
        f"MAX_LATENCY={p['MAX_LATENCY']} must equal the adder unit's ADD_LATENCY="
        f"{p['ADD_LATENCY']}: the max select is paced to the adder so an op-mixed "
        f"wavefront stays aligned (xlu_reduction_tree.sv assumes 2 twice, checks it nowhere)")
    assert p["ROOT_LATENCY"] == 1 + p["LEVELS"] * p["ADD_LATENCY"], (
        f"ROOT_LATENCY={p['ROOT_LATENCY']} must be 1 + LEVELS*ADD_LATENCY = "
        f"{1 + p['LEVELS'] * p['ADD_LATENCY']} (VPU_REDUCE_LATENCY is this number, not a constant)")
    assert p["TAP_LATENCY"] == 1 + p["TAP_LEVEL"] * p["ADD_LATENCY"], (
        f"TAP_LATENCY={p['TAP_LATENCY']} must be 1 + TAP_LEVEL*ADD_LATENCY = "
        f"{1 + p['TAP_LEVEL'] * p['ADD_LATENCY']} (VPU_LANE_REDUCE_LATENCY)")
    assert p["QD"] >= 2, f"QD={p['QD']}: a node reads two children per token; depth 1 can stall the feed"


@unit(memories=("OP", "D"), writes=("node",), parameters=("N", "NCYC"))
def tree_feed(op: UInt(8)[NCYC], d: UInt(16)[NCYC, N]):
    for t in range(NCYC):
        o: UInt(8) = op[t]
        with allo.meta_for(N) as l:
            w: UInt(17) = 0
            w[0:16] = d[t, l]
            w[16] = o[0]
            node[l].put(w)


@unit(instances=("TAP_BASE - N",), reads=("node",), writes=("node",),
      parameters=("N", "NCYC", "tree_node"), legality=tree_legality)
def add_low():
    m = df.get_pid()  # node N + m, children 2m and 2m + 1: the levels below the tap
    for t in range(NCYC):
        wa: UInt(17) = node[2 * m].get()
        wb: UInt(17) = node[2 * m + 1].get()
        node[N + m].put(tree_node(wa, wb))


@unit(instances=("SUB",), reads=("node",), writes=("node", "tap"),
      parameters=("N", "TAP_BASE", "NCYC", "tree_node"))
def add_tap():
    k = df.get_pid()  # node TAP_BASE + k covers sublane k: the tap level
    for t in range(NCYC):
        wa: UInt(17) = node[2 * (TAP_BASE - N + k)].get()
        wb: UInt(17) = node[2 * (TAP_BASE - N + k) + 1].get()
        w: UInt(17) = tree_node(wa, wb)
        node[TAP_BASE + k].put(w)
        tap[k].put(w)


@unit(instances=("SUB - 1",), reads=("node",), writes=("node",),
      parameters=("N", "SUB", "TAP_BASE", "NCYC", "tree_node"))
def add_high():
    h = df.get_pid()  # node TAP_BASE + SUB + h: the levels above the tap, the root last
    for t in range(NCYC):
        wa: UInt(17) = node[2 * (TAP_BASE - N + SUB + h)].get()
        wb: UInt(17) = node[2 * (TAP_BASE - N + SUB + h) + 1].get()
        node[TAP_BASE + SUB + h].put(tree_node(wa, wb))


@unit(memories=("RST", "VLD", "VO", "RO", "LVO", "LRO"), reads=("node", "tap"),
      parameters=("N", "SUB", "NCYC", "ROOT_LATENCY", "TAP_LATENCY"))
def tree_sink(rst: UInt(8)[NCYC], vld: UInt(8)[NCYC], vo: UInt(8)[NCYC],
              ro: UInt(16)[NCYC], lvo: UInt(8)[NCYC], lro: UInt(16)[NCYC, SUB]):
    # the register edges of the RTL as data: the valid pipes are reset, the
    # payload pipes are not (P-9); their depths are the DERIVED latencies
    vq: UInt(8)[ROOT_LATENCY]
    rq: UInt(16)[ROOT_LATENCY]
    lvq: UInt(8)[TAP_LATENCY]
    lrq: UInt(16)[TAP_LATENCY, SUB]
    for k in range(ROOT_LATENCY):
        vq[k] = 0
    for k in range(TAP_LATENCY):
        lvq[k] = 0
    for t in range(NCYC):
        r: UInt(8) = rst[t]
        v: UInt(8) = vld[t]
        root: UInt(17) = node[2 * N - 2].get()
        for k in range(1, ROOT_LATENCY):
            vq[ROOT_LATENCY - k] = vq[ROOT_LATENCY - k - 1]
            rq[ROOT_LATENCY - k] = rq[ROOT_LATENCY - k - 1]
        vq[0] = v
        rq[0] = root[0:16]
        for j in range(1, TAP_LATENCY):
            lvq[TAP_LATENCY - j] = lvq[TAP_LATENCY - j - 1]
            for s in range(SUB):
                lrq[TAP_LATENCY - j, s] = lrq[TAP_LATENCY - j - 1, s]
        lvq[0] = v
        with allo.meta_for(SUB) as s:
            tw: UInt(17) = tap[s].get()
            lrq[0, s] = tw[0:16]
        if r == 0:
            for k in range(ROOT_LATENCY):
                vq[k] = 0
            for k in range(TAP_LATENCY):
                lvq[k] = 0
        for s in range(SUB):  # the lane array first (finding A5, SystemC csim)
            lro[t, s] = lrq[TAP_LATENCY - 1, s]
        vo[t] = vq[ROOT_LATENCY - 1]
        ro[t] = rq[ROOT_LATENCY - 1]
        lvo[t] = lvq[TAP_LATENCY - 1]


MEMORIES = (Memory("RST", "UInt(8)[NCYC]"), Memory("VLD", "UInt(8)[NCYC]"),
            Memory("OP", "UInt(8)[NCYC]"), Memory("D", "UInt(16)[NCYC, N]"),
            Memory("VO", "UInt(8)[NCYC]"), Memory("RO", "UInt(16)[NCYC]"),
            Memory("LVO", "UInt(8)[NCYC]"), Memory("LRO", "UInt(16)[NCYC, SUB]"))
CHANNELS = (Channel("node", dtype="UInt(17)", depth="QD", shape=("2 * N - 1",),
                    carries="{op, value} per tree node; [0:N] the leaves, 2N-2 the root"),
            Channel("tap", dtype="UInt(17)", depth="QD", shape=("SUB",),
                    carries="the tap level's words, one per sublane"))
UNITS = (tree_feed, add_low, add_tap, add_high, tree_sink)


def architecture(lanes, sub, ncyc, **over):
    p = parameters(lanes, sub, ncyc)
    p.update(over)
    return Architecture(name=f"xlu_tree_{lanes}x{sub}", parameters=p,
                        memories=MEMORIES, channels=CHANNELS, units=UNITS)


WRONG = {
    "ADD_LATENCY 3 (a slower adder, latencies not re-derived)": {"ADD_LATENCY": 3},
    "MAX_LATENCY 1 (max path faster than the adder)": {"MAX_LATENCY": 1},
    "ROOT_LATENCY 13 at 16 leaves (the shipped constant, wrong geometry)": {"ROOT_LATENCY": 13},
    "TAP_LEVEL 3 at 4 lanes (tap one level up)": {"TAP_LEVEL": 3},
    "TAP_BASE 20 (tap base declared on its own)": {"TAP_BASE": 20},
    "LEVELS 5 at 16 leaves": {"LEVELS": 5},
    "N 12 (not LANES*SUB)": {"N": 12},
}


def refusals(lanes=4, sub=4, ncyc=8):
    """Each wrong set against ``tree_legality``: ``{case: message or 'ACCEPTED'}``."""
    out = {}
    for case, over in WRONG.items():
        try:
            architecture(lanes, sub, ncyc, **over)
            out[case] = "ACCEPTED"
        except AssertionError as e:
            out[case] = str(e).splitlines()[0]
    return out
