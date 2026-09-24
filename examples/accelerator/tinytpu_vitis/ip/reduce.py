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

from dataclasses import dataclass

from allo.ir.types import int16, int32

from examples.accelerator.tinytpu_vitis.ip.compose import (
    Architecture, Channel, Memory, unit)
from examples.accelerator.tinytpu_vitis.ip.units.reduction_tree import reduce_tree


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
        Channel("red_in", "UInt(RED_LANES * RED_IN_BITS)", "QD",
                carries="one packed word of RED_LANES lanes, per output"),
        Channel("red_full", "RED_ACC", "QD", carries="the root of the tree"),
        Channel("red_group", "UInt(RED_GROUPS * RED_ACC_BITS)", "QD",
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
