# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DotTree: the second architecture the library composes, and the first one
that is not TinyTPU.

It exists to make ``reduce_tree`` run. A capability nothing instantiates
cannot be seen to be broken, and this project has been bitten by that twice,
so the adder tree ships inside a three-unit region that computes a real
``M x RED_LANES x N`` GEMM and is compared against numpy.

It is the **output-stationary** counterpart of TinyTPU's systolic chain, on
purpose: one output element is finished by one pass of one tree, so ``M`` is
just a trip count and ``M = 1`` costs exactly ``1/M`` of ``M``'s work. TinyTPU
cannot do that -- its accumulator's dependence claim needs ``AR_RAW_DIST``
accu steps between a write and its read, and at ``M = 1`` every dependent pair
has to be padded apart (``docs/source/designs/ip_gaps.rst``, row 2).

``M`` and ``N`` arrive in ``ctl`` at run time, as they do on TinyTPU: one
build runs every shape, and the tree takes its trip count over ``c_red`` like
every other unit in this library takes its work count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from allo.ir.types import int16, int32

from allo.actions import (
    Action, Contract, EXACT, Instruction, Port, structure)
from allo.compose import (
    Architecture, Channel, Memory, unit)
from examples.tinytpu.ip.units.reduction_tree import reduce_tree


@unit(
    memories=("A", "B", "ctl"),
    writes=("c_red", "c_snk", "red_in"),
    parameters=("DOT_MAX", "RED_LANES", "RED_IN", "RED_IN_BITS"),
)
def dot_feed(dram_a: int8[DOT_MAX * DOT_MAX], dram_b: int8[DOT_MAX * DOT_MAX],
             shape: int32[2]):
    rows: int32 = shape[0]
    cols: int32 = shape[1]
    count_word: UInt(64) = 0
    count_word[0:16] = rows * cols
    c_red.put(count_word)
    c_snk.put(count_word)
    for m in range(rows):
        for n in range(cols):
            # The lane multiply, packed as the tree's input word. The product
            # is formed in RED_IN, so the feeder states the arithmetic it
            # hands the tree instead of leaving it to be read off a literal:
            # an int8 x int8 product is exact in the shipped int16.
            packed: UInt(RED_LANES * RED_IN_BITS) = 0
            with allo.meta_for(RED_LANES) as k:
                a: RED_IN = dram_a[m * DOT_MAX + k]
                b: RED_IN = dram_b[k * DOT_MAX + n]
                packed[RED_IN_BITS * k : RED_IN_BITS * (k + 1)] = a * b
            red_in.put(packed)


@unit(
    memories=("C", "G"),
    reads=("c_snk", "red_full", "red_group"),
    parameters=("DOT_MAX", "RED_GROUPS", "RED_ACC", "RED_ACC_BITS"),
)
def dot_sink(out_c: int32[DOT_MAX * DOT_MAX],
             out_g: int32[DOT_MAX * DOT_MAX * RED_GROUPS]):
    count_word: UInt(64) = c_snk.get()
    n_word: int32 = count_word[0:16]
    for work in range(n_word):
        full: RED_ACC = red_full.get()
        out_c[work] = full
        group_word: UInt(RED_GROUPS * RED_ACC_BITS) = red_group.get()
        with allo.meta_for(RED_GROUPS) as g:
            part: RED_ACC = group_word[RED_ACC_BITS * g : RED_ACC_BITS * (g + 1)]
            out_g[work * RED_GROUPS + g] = part


@dataclass(frozen=True)
class ReduceParams:
    """What the tree is built at. ``RED_DEPTH``, ``RED_GROUP_SIZE``,
    ``RED_TAP_LEVEL`` and ``RED_TAP_BASE`` are DERIVED: a width and a depth
    that are declared separately can disagree, and a tap that is declared
    separately from the leaf mapping reads a level that does not cover a
    group.

    Each width is declared beside its type rather than read off it, because
    the unit needs both -- a type to annotate with and an integer to slice
    with -- and ``reduction_tree_legality`` holds each pair to the other."""

    RED_LANES: int = 8          # tree width; a power of two
    RED_GROUPS: int = 2         # leaf mapping: this many contiguous subtrees
    RED_IN: type = int16        # the lane type: an int8 x int8 product
    RED_IN_BITS: int = 16
    RED_ACC: type = int32       # the type EVERY node of the tree carries
    RED_ACC_BITS: int = 32
    DOT_MAX: int = 8            # the rig's DRAM stride
    QD: int = 8

    @property
    def RED_DEPTH(self) -> int:
        return self.RED_LANES.bit_length() - 1

    @property
    def RED_GROUP_SIZE(self) -> int:
        return self.RED_LANES // self.RED_GROUPS

    @property
    def RED_TAP_LEVEL(self) -> int:
        """Which adder level the tap comes off. ``RED_DEPTH - RED_TAP_LEVEL``
        levels of adder separate the two outputs, and that difference is the
        whole of what a scheduler would have to book differently for them."""
        return self.RED_GROUP_SIZE.bit_length() - 1

    @property
    def RED_TAP_BASE(self) -> int:
        """Where the level whose nodes each cover one group begins, in the
        body's ascending node numbering."""
        return 2 * self.RED_LANES - 2 * self.RED_GROUPS

    def namespace(self) -> dict:
        return {name: getattr(self, name) for name in (
            "RED_LANES", "RED_GROUPS", "RED_IN", "RED_IN_BITS", "RED_ACC",
            "RED_ACC_BITS", "RED_DEPTH", "RED_GROUP_SIZE", "RED_TAP_LEVEL",
            "RED_TAP_BASE", "DOT_MAX", "QD")}


def channels():
    return (
        Channel("c_red", "UInt(64)", "QD", carries="dot_feed -> reduce_tree"),
        Channel("c_snk", "UInt(64)", "QD", carries="dot_feed -> dot_sink"),
        # The width is DERIVED from the lane count and the lane width: the
        # tree's leaf order is checked against the lane count, and the one
        # place that count may be written is here.
        Channel("red_in", depth="QD", lanes="RED_LANES",
                lane_bits="RED_IN_BITS",
                carries="one packed word of RED_LANES lanes, per output"),
        Channel("red_full", "RED_ACC", "QD", carries="the root of the tree"),
        Channel("red_group", depth="QD", lanes="RED_GROUPS",
                lane_bits="RED_ACC_BITS",
                carries="the tap: RED_GROUPS subtree roots, one level down"),
    )


def memories():
    return (Memory("A", "int8[DOT_MAX * DOT_MAX]"),
            Memory("B", "int8[DOT_MAX * DOT_MAX]"),
            Memory("ctl", "int32[2]"),
            Memory("C", "int32[DOT_MAX * DOT_MAX]"),
            Memory("G", "int32[DOT_MAX * DOT_MAX * RED_GROUPS]"))


def architecture(params=None, name="dot_tree"):
    params = params or ReduceParams()
    return Architecture(name=name, parameters=params.namespace(),
                        memories=memories(), channels=channels(),
                        units=(dot_feed, reduce_tree, dot_sink))


class DotTree:
    """One built DotTree: the region and its Vitis directives."""

    def __init__(self, params=None, name="dot_tree"):
        self.params = params or ReduceParams()
        self.architecture = architecture(self.params, name)

    @property
    def region(self):
        return self.architecture.region()

    def schedule(self, s):
        return self.architecture.directives(s)


#: What ``reduce_latency_probe.py`` MEASURED on the emitted Verilog at 8:2,
#: two-sided and value-checked: the tree is three adder levels deep and Vitis
#: retires the whole of it in two cycles, and the tap -- one level above the
#: root -- arrives in the SAME cycle as the root. A number derived from the
#: geometry would be wrong in the direction a ``>=`` check accepts, which is
#: why a declared latency is a measurement and not an estimate.
MEASURED = {"red_full": 2, "red_group": 2}


def machine(params=None, arithmetic=EXACT, leaves=None, name=None):
    """DotTree as an ``allo.actions.Machine`` -- the same region, asked what
    it MEANS rather than what it is wired to.

    The units, their channel and memory ports, the states they own and the
    lane count of the packed word are not declared here: they are
    ``structure(architecture(...))``, read off the composition above. What
    this function adds is the part no structural declaration carries --

    * two COMPUTE ports and their widths (``RED_LANES`` multipliers, and a
      tree of ``RED_LANES - 1`` adders), which is arithmetic;
    * two MEASURED latencies, from the probe;
    * the leaf order, which is a permutation of the operands and the one
      thing MiniTPU's owner said an interface must be able to say;
    * the contract the accumulator width rests on.

    Every one of those is a fact about behaviour, and every fact about
    structure comes from the architecture -- which is the positive half of the
    claim, on the architecture where the two models' grain agrees.
    """
    p = params or ReduceParams()
    arch = architecture(p, name or "dot_tree")
    leaves = leaves or (f"(i % {p.RED_GROUPS}) * {p.RED_GROUP_SIZE} "
                        f"+ (i // {p.RED_GROUPS})")
    skeleton = structure(arch)
    # The arithmetic, added to the units the composition already named.
    adds = {"dot_feed": (Port("mul", physical=p.RED_LANES),),
            "reduce_tree": (Port("adder", physical=p.RED_LANES - 1),)}
    units = tuple(
        u if u.name not in adds else replace(u, ports=u.ports + adds[u.name])
        for u in skeleton.units)
    return replace(
        skeleton, units=units, arithmetic=arithmetic,
        contracts=(Contract(
            "accumulator_is_wide_enough",
            "RED_ACC_BITS >= RED_IN_BITS + RED_DEPTH, so no level rounds and "
            "the leaf order is unobservable",
            discharged_by="reduction_tree_legality, at composition time"),),
        instructions=(Instruction("dot", rows="m * n", actions=(
            Action("dot_feed", "read", "A.read", state="A", base="a_s",
                   into="a", role="one row of A"),
            Action("dot_feed", "read", "B.read", state="B", base="b_s",
                   into="b", role="one column of B"),
            Action("dot_feed", "compute", "mul", compute="mul",
                   args=("a", "b"), into="prod",
                   role="RED_LANES lane products, packed"),
            Action("dot_feed", "emit", "red_in", args=("prod",), into="word"),
            Action("reduce_tree", "receive", "red_in", into="word"),
            # ONE fold, TWO taps off it. The tree is not run twice: the group
            # sums are the level the root is built from. The lane count the
            # leaf order is checked against comes from the CHANNEL the word
            # arrived on -- `red_in` declares `lanes="RED_LANES"` -- and not
            # from a one-row memory invented to carry it.
            Action("reduce_tree", "compute", "adder", compute="reduce_add",
                   args=("word",), into="full", lanes=leaves,
                   at=MEASURED["red_full"], role="the root: every lane"),
            Action("reduce_tree", "compute", "adder",
                   compute="reduce_add_group", args=("word",), into="groups",
                   lanes=leaves, at=MEASURED["red_group"],
                   role="the tap: RED_GROUPS subtree roots"),
            Action("reduce_tree", "emit", "red_full", args=("full",),
                   into="full_w", at=MEASURED["red_full"]),
            Action("reduce_tree", "emit", "red_group", args=("groups",),
                   into="group_w", at=MEASURED["red_group"]),
            Action("dot_sink", "receive", "red_full", into="full_w"),
            Action("dot_sink", "receive", "red_group", into="group_w"),
            Action("dot_sink", "write", "C.write", state="C", base="c_d",
                   args=("full_w",)),
            Action("dot_sink", "write", "G.write", state="G", base="g_d",
                   args=("group_w",)),
        )),))
