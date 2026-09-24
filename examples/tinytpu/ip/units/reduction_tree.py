# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A balanced adder tree: reduce ``RED_LANES`` lanes of one packed word.

The alternative to the systolic chain. ``pe`` reduces a column by handing its
partial sum south through ``p_fwd``, so the reduction is ``T`` deep in units
and its topology is the unit graph; this unit reduces the same lanes in
``log2(RED_LANES)`` levels inside ONE unit, so the topology is the unit -- and
an architecture chooses between them by choosing which unit it instantiates.

**Rounding is a property of the two ends, not of the levels.** The interior is
integer adders at one declared width, so the tree reassociates but does not
round: quantise into ``RED_IN`` once, sum exactly, narrow at the consumer once.
Instantiate it at a rounding type and that stops being true at every level.

**Two outputs at two depths of one structure**: ``red_group`` is taken from the
subtree roots partway up, ``red_full`` from the root -- one pass, two answers.
Allo expresses the dataflow of two depths (two ``put`` calls in one body); it
does not express the two *latencies*, which is a gap-table row.

The parameters, because a tree that does not declare them disagrees with
another correct tree in the last bits and nobody can see why:

``RED_LANES``      the width. A power of two.
``RED_IN``         the lane type; ``RED_IN_BITS`` is its width.
``RED_ACC``        the type EVERY node of the tree carries, at the SAME width
                   at every level: there is no widening down the tree.
``RED_GROUPS``     the leaf mapping. Input lane ``g + s * RED_GROUPS`` lands at
                   leaf ``g * RED_GROUP_SIZE + s``, so each group owns one
                   contiguous subtree.
``RED_TAP_LEVEL``  which adder level ``red_group`` comes off, with
                   ``RED_TAP_BASE`` where that level's nodes sit.

``RED_DEPTH``, ``RED_GROUP_SIZE``, ``RED_TAP_LEVEL`` and ``RED_TAP_BASE`` are
DERIVED from the ones above them, never declared beside them, and
``reduction_tree_legality`` refuses a set that disagrees. That check lives with
the parameters rather than in a testbench on purpose: on MiniTPU exactly one
assertion ties the tap to the booked writeback and it is in a testbench, so
changing the geometry moves the tap and a sequencer broadcasts a stale mid-tree
value with nothing faulting. In a *parametrized* unit that is the likeliest way
to ship a wrong design.

Two things of MiniTPU's this unit deliberately does **not** inherit: an unreset
datapath (a 1024-sink reset net avoided, safe only because the valid and op
pipelines are reset -- an invariant written down nowhere), and the toggling
every cycle regardless of ``valid`` that follows from it. Here the tree is
inside the work loop, so it runs ``n_word`` times and not once more; on the
ASIC flow that difference is power.

Why this form rather than MiniTPU's per-level rounding, with their own costing
of both, is entry 7 of ``dev/records/tinytpu/measured_negatives.rst``; the full
comparison is ``docs/source/designs/ip_gaps.rst`` ("The tree we did not
build").
"""

from __future__ import annotations

from allo.customize import Partition

from allo.compose import unit


def reduction_tree_legality(p):
    lanes, groups = p["RED_LANES"], p["RED_GROUPS"]
    depth = lanes.bit_length() - 1
    assert lanes >= 2 and lanes & (lanes - 1) == 0, (
        f"RED_LANES={lanes}: the recurrence in the body is a BALANCED binary "
        f"tree, which needs a power of two")
    assert 2 <= groups <= lanes and groups & (groups - 1) == 0, (
        f"RED_GROUPS={groups}: the leaf mapping is a power-of-two "
        f"de-interleave of RED_LANES={lanes}, and at least 2 -- at 1 the tap "
        f"IS the root, so `red_group` duplicates `red_full`, and the "
        f"full-width bit slice that would extract it "
        f"(`w[0:RED_ACC_BITS]` on a `UInt(RED_ACC_BITS)`) lowers to an "
        f"`arith.trunci` from i32 to i32 and fails to build")
    assert p["RED_TAP_LEVEL"] == p["RED_GROUP_SIZE"].bit_length() - 1, (
        f"RED_TAP_LEVEL={p['RED_TAP_LEVEL']} must be "
        f"log2(RED_GROUP_SIZE) = {p['RED_GROUP_SIZE'].bit_length() - 1}: the "
        f"tap is an adder level, and a level that is declared separately from "
        f"the leaf mapping reduces a rotated window -- a plausible wrong "
        f"number, which is MiniTPU's own worst failure mode here")
    assert p["RED_DEPTH"] == depth, (
        f"RED_DEPTH={p['RED_DEPTH']} must be log2(RED_LANES) = {depth}; a "
        f"width and a depth declared separately can disagree")
    assert p["RED_GROUP_SIZE"] == lanes // groups, (
        f"RED_GROUP_SIZE={p['RED_GROUP_SIZE']} must be RED_LANES/RED_GROUPS "
        f"= {lanes // groups}; it is derived, not chosen")
    assert p["RED_TAP_BASE"] == 2 * lanes - 2 * groups, (
        f"RED_TAP_BASE={p['RED_TAP_BASE']} must be 2*RED_LANES-2*RED_GROUPS "
        f"= {2 * lanes - 2 * groups}: the level whose nodes each cover one "
        f"group. It is derived from the leaf mapping, not chosen alongside it")
    for width, dtype in (("RED_IN_BITS", "RED_IN"),
                         ("RED_ACC_BITS", "RED_ACC")):
        assert p[width] == p[dtype].bits, (
            f"{width}={p[width]} but {dtype} is {p[dtype]}: the body needs a "
            f"type to annotate with and an integer to slice with, and a unit "
            f"whose slices are a different width from its arithmetic reads "
            f"the wrong bits without failing")
    assert p["RED_ACC_BITS"] >= p["RED_IN_BITS"] + depth, (
        f"RED_ACC_BITS={p['RED_ACC_BITS']} carries a {depth}-level tree of "
        f"RED_IN_BITS={p['RED_IN_BITS']} lanes, which needs "
        f"{p['RED_IN_BITS'] + depth} bits. Narrower is not a smaller tree, it "
        f"is a DIFFERENT function: every level rounds, so the leaf mapping "
        f"becomes observable and two correct trees disagree")


def reduction_tree_directives(s, ctx):
    # `node` is the tree itself, and it must be WIRES. Every index is constant
    # after the meta_for unrolls, but left as an array Vitis is free to give
    # it a RAM -- and a RAM with RED_LANES-1 write sites is the shape standard
    # cells refuse (ELAB-366, asic_synthesis/README.md). Partitioning it
    # completely is what makes this unit synthesisable through the ASIC flow
    # rather than only through an FPGA's block RAM.
    s.partition(f"{ctx.instance('reduce_tree')}:node", Partition.Complete, dim=1)


@unit(
    reads=("c_red", "red_in"),
    writes=("red_full", "red_group"),
    parameters=("RED_LANES", "RED_IN", "RED_IN_BITS", "RED_ACC",
                "RED_ACC_BITS", "RED_GROUPS", "RED_GROUP_SIZE",
                "RED_TAP_BASE"),
    directives=reduction_tree_directives,
    legality=reduction_tree_legality,
)
def reduce_tree():
    # node[0 : RED_LANES] are the leaves; node[RED_LANES + n] is the n-th
    # internal node and its two children are already written when n is
    # reached, so ONE ascending loop is the whole balanced tree.
    node: RED_ACC[2 * RED_LANES - 1]
    count_word: UInt(64) = c_red.get()
    n_word: int32 = count_word[0:16]
    for work in range(n_word):
        packed: UInt(RED_LANES * RED_IN_BITS) = red_in.get()
        with allo.meta_for(RED_GROUPS) as g:
            with allo.meta_for(RED_GROUP_SIZE) as s:
                lane: RED_IN = packed[RED_IN_BITS * (g + s * RED_GROUPS):
                                      RED_IN_BITS * (g + s * RED_GROUPS + 1)]
                node[g * RED_GROUP_SIZE + s] = lane
        with allo.meta_for(RED_LANES - 1) as n:
            node[RED_LANES + n] = node[2 * n] + node[2 * n + 1]
        group_word: UInt(RED_GROUPS * RED_ACC_BITS) = 0
        with allo.meta_for(RED_GROUPS) as tap:
            group_word[RED_ACC_BITS * tap : RED_ACC_BITS * (tap + 1)] = \
                node[RED_TAP_BASE + tap]
        red_group.put(group_word)
        red_full.put(node[2 * RED_LANES - 2])
