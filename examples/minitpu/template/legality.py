# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived parameters and their legality conditions (Q4).

MiniTPU declares a geometry (``DIM``, ``NUM_LANES``, ``NUM_SUBLANES``, ``N``,
FIFO depths) and a set of numbers that FOLLOW from it: the tree's ``LEVELS``
and tap level, the max path's extra delay, the array's weight-switch span,
the MXU's push->valid. In the RTL those relations are a ``$clog2`` in one
file, a hand-copied constant in another, a ``$error`` in a generate block
and an assertion in a testbench (``UNITS.md`` §5; the tap/booking relation
lived only in a tb). Phase 0 measured every one of them
(``u3_phase0_2026-10-04.rst``).

Here each geometry is a frozen record whose derived numbers are PROPERTIES,
never fields -- the ``ReduceParams`` pattern of ``ip/reduce.py`` -- and each
relation that a unit's correctness rests on is a ``legality`` callable in the
``reduction_tree_legality`` style: run at composition, naming the parameter
and why. The timing numbers are derived from the ENGINE's declared latencies
(``Engine.add_latency``), which is the D-10 hook: a backend that builds a
3-cycle adder as 5 cycles changes ``push_to_valid`` through this record, not
through a constant in the assembler.

``check_against_phase0`` holds every derived number here to the measured one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from examples.minitpu.harness import ref_mxu
from examples.minitpu.template.engines import BF16_ACC24, Engine


def _log2(n, what):
    assert n >= 1 and n & (n - 1) == 0, f"{what}={n}: must be a power of two"
    return n.bit_length() - 1


@dataclass(frozen=True)
class MxuGeometry:
    """``mxu.sv``'s parameters and what the MXU contract derives from them."""

    DIM: int = 16
    NUM_SUBLANES: int = 4
    INPUT_DEPTH: int = 4
    OUTPUT_ROWS: int = 64
    mac: Engine = BF16_ACC24

    @property
    def PE_LATENCY(self) -> int:
        """``MXU_PE_LATENCY = 1 + MXU_ACC_ADD_LATENCY``: the product register
        plus the adder the engine declares."""
        return 1 + self.mac.add_latency

    @property
    def push_to_valid(self) -> int:
        """Edges from a group's last push to ``output_valid_o`` (Phase 0)."""
        return 2 + self.DIM * (self.PE_LATENCY + 1)

    @property
    def switch_span(self) -> int:
        """``WEIGHT_SWITCH_SPAN``: cycles a started tile holds its bank."""
        return (self.DIM - 1) * self.PE_LATENCY + (self.DIM - 1)

    @property
    def result_latency(self) -> int:
        """``isa_latency.json`` ``vmatpush`` result latency: the stream
        engine's 4 rows from ``t + 1``, then push->valid, less the pop edge."""
        return 4 + self.push_to_valid - 1

    def namespace(self) -> dict:
        return {"DIM": self.DIM, "NUM_SUBLANES": self.NUM_SUBLANES,
                "INPUT_DEPTH": self.INPUT_DEPTH, "OUTPUT_ROWS": self.OUTPUT_ROWS,
                "PE_LATENCY": self.PE_LATENCY, "PUSH_TO_VALID": self.push_to_valid,
                "SWITCH_SPAN": self.switch_span}


def mxu_legality(p):
    """What ``mxu.sv`` asserts, elaborates or assumes, as one condition set."""
    dim, sub = p["DIM"], p["NUM_SUBLANES"]
    assert dim >= 2, f"DIM={dim}: a one-row array has no chain"
    assert p["OUTPUT_ROWS"] % sub == 0, (
        f"OUTPUT_ROWS={p['OUTPUT_ROWS']} must hold whole groups of "
        f"NUM_SUBLANES={sub} results: a lane FIFO that is full DROPS a group "
        f"(mxu.sv:44), and a partial group would be dropped by halves")
    assert p["PE_LATENCY"] >= 1, "a PE with no register has no skew to lock"
    assert p["PUSH_TO_VALID"] == 2 + dim * (p["PE_LATENCY"] + 1), (
        f"PUSH_TO_VALID={p['PUSH_TO_VALID']} must be 2 + DIM*(PE_LATENCY+1) = "
        f"{2 + dim * (p['PE_LATENCY'] + 1)}: it is the one timing constant the "
        f"contract exposes and the assembler consumes (85 at DIM=16); a "
        f"declared copy drifts when the adder's depth changes")
    assert p["SWITCH_SPAN"] == (dim - 1) * p["PE_LATENCY"] + (dim - 1), (
        f"SWITCH_SPAN={p['SWITCH_SPAN']} must be (DIM-1)*PE_LATENCY + (DIM-1) = "
        f"{(dim - 1) * p['PE_LATENCY'] + (dim - 1)}: a refill earlier than the "
        f"span corrupts the switching tile, and nothing but this relation and "
        f"a sim-only assertion (mxu.sv:259) says so")
    assert p["INPUT_DEPTH"] >= 1


@dataclass(frozen=True)
class TreeGeometry:
    """``xlu_reduction_tree.sv``'s parameters and the numbers the sequencer
    books from them (``WB_W_REDUCE``, ``WB_W_LANE_REDUCE``)."""

    N: int = 64
    NUM_LANES: int = 16
    ADDER_LATENCY: int = 2       # vpu_bf16_add_pipe; assumed in two places in the RTL
    LEAF_STAGES: int = 1

    @property
    def NUM_SUBLANES(self) -> int:
        return self.N // self.NUM_LANES

    @property
    def LEVELS(self) -> int:
        return _log2(self.N, "N")

    @property
    def LANE_LEVELS(self) -> int:
        """The tap level: the level whose nodes each cover one sublane group."""
        return _log2(self.NUM_LANES, "NUM_LANES")

    @property
    def latency(self) -> int:
        return self.LEAF_STAGES + self.ADDER_LATENCY * self.LEVELS

    @property
    def tap_latency(self) -> int:
        return self.LEAF_STAGES + self.ADDER_LATENCY * self.LANE_LEVELS

    @property
    def max_path_delay(self) -> int:
        """The compare path per level is registered to match the adder: its
        delay is DERIVED from the adder's declared latency, where the RTL
        writes two registers by hand."""
        return self.ADDER_LATENCY

    def namespace(self) -> dict:
        return {"N": self.N, "NUM_LANES": self.NUM_LANES,
                "NUM_SUBLANES": self.NUM_SUBLANES, "LEVELS": self.LEVELS,
                "LANE_LEVELS": self.LANE_LEVELS, "ADDER_LATENCY": self.ADDER_LATENCY,
                "REDUCE_LATENCY": self.latency, "LANE_REDUCE_LATENCY": self.tap_latency,
                "MAX_PATH_DELAY": self.max_path_delay}


def tree_legality(p):
    n, lanes = p["N"], p["NUM_LANES"]
    assert n >= 2 and n & (n - 1) == 0, f"N={n}: a balanced tree needs a power of two"
    assert lanes >= 1 and lanes & (lanes - 1) == 0 and lanes <= n, (
        f"NUM_LANES={lanes}: the tap is an adder LEVEL, so the lane count must "
        f"be a power of two dividing N={n}")
    assert p["NUM_SUBLANES"] == n // lanes, (
        f"NUM_SUBLANES={p['NUM_SUBLANES']} must be N/NUM_LANES = {n // lanes} "
        f"(xlu_reduction_tree.sv:54-58 elaborates `N >> LANE_LEVELS == NUM_SUBLANES`)")
    assert p["LEVELS"] == n.bit_length() - 1, (
        f"LEVELS={p['LEVELS']} must be log2(N) = {n.bit_length() - 1}; xlu.sv "
        f"passes it explicitly and the RTL $errors when they disagree")
    assert p["LANE_LEVELS"] == lanes.bit_length() - 1, (
        f"LANE_LEVELS={p['LANE_LEVELS']} must be log2(NUM_LANES) = "
        f"{lanes.bit_length() - 1}: a tap one level off broadcasts a stale "
        f"mid-tree value and nothing faults (UNITS.md §8.3)")
    assert p["REDUCE_LATENCY"] == 1 + p["ADDER_LATENCY"] * p["LEVELS"], (
        f"REDUCE_LATENCY={p['REDUCE_LATENCY']} must be 1 + ADDER_LATENCY*LEVELS = "
        f"{1 + p['ADDER_LATENCY'] * p['LEVELS']}: the sequencer books it "
        f"(WB_W_REDUCE); a declared copy outlives an adder change")
    assert p["LANE_REDUCE_LATENCY"] == 1 + p["ADDER_LATENCY"] * p["LANE_LEVELS"], (
        f"LANE_REDUCE_LATENCY={p['LANE_REDUCE_LATENCY']} must be "
        f"1 + ADDER_LATENCY*LANE_LEVELS = {1 + p['ADDER_LATENCY'] * p['LANE_LEVELS']}")
    assert p["MAX_PATH_DELAY"] == p["ADDER_LATENCY"], (
        f"MAX_PATH_DELAY={p['MAX_PATH_DELAY']} must equal ADDER_LATENCY="
        f"{p['ADDER_LATENCY']}: the max path's registers exist only to match "
        f"the adder, and MiniTPU writes the 2 by hand in two places")


def check_against_phase0():
    """Every derived number against the Phase 0 measurement. Returns the
    list of ``(label, derived, measured)``; raises on any difference."""
    rows = []
    for dim in (2, 4, 16):
        g = MxuGeometry(DIM=dim)
        mxu_legality(g.namespace())
        rows += [(f"mxu d{dim} push->valid", g.push_to_valid, ref_mxu.push_to_valid(dim)),
                 (f"mxu d{dim} switch span", g.switch_span, ref_mxu.switch_span(dim))]
    rows += [("mxu d16 PE_LATENCY", MxuGeometry().PE_LATENCY, ref_mxu.PE_LATENCY),
             ("mxu d16 vmatpush result latency", MxuGeometry().result_latency, 85)]
    # Phase 0 measured: n64 root 13 / tap 9; n16 (NUM_LANES=4) root 9 / tap 5
    for n, lanes, root, tap in ((64, 16, 13, 9), (16, 4, 9, 5)):
        g = TreeGeometry(N=n, NUM_LANES=lanes)
        tree_legality(g.namespace())
        rows += [(f"tree n{n} root latency", g.latency, root),
                 (f"tree n{n} tap latency", g.tap_latency, tap)]
    bad = [r for r in rows if r[1] != r[2]]
    assert not bad, f"derived != measured: {bad}"
    return rows


def refusals():
    """One wrong declaration per relation, each refused by its legality."""
    out = []
    for label, ns, fn in (
        ("mxu: stale PUSH_TO_VALID after an adder change",
         MxuGeometry().namespace() | {"PUSH_TO_VALID": 82, "PE_LATENCY": 6}, mxu_legality),
        ("mxu: OUTPUT_ROWS not a whole number of groups",
         MxuGeometry(OUTPUT_ROWS=62).namespace(), mxu_legality),
        ("tree: tap level declared one off",
         TreeGeometry().namespace() | {"LANE_LEVELS": 3}, tree_legality),
        ("tree: LEVELS declared beside N",
         TreeGeometry().namespace() | {"LEVELS": 5}, tree_legality),
        ("tree: max path delay hand-written for a 2-cycle adder, adder now 3",
         TreeGeometry(ADDER_LATENCY=3).namespace() | {"MAX_PATH_DELAY": 2}, tree_legality),
    ):
        try:
            fn(ns)
            out.append((label, "ACCEPTED (bug)"))
        except AssertionError as e:
            out.append((label, "refused: " + str(e).splitlines()[0][:90]))
    return out
