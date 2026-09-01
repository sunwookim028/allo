# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data movement as a *priced* choice (Stage 3 routing).

The planner routes a value over ``(buffer, residence)`` states, and until now it did
so by hop count: every move was one step and every step was worth the same. That made
a relayout free to *find* but impossible to *weigh* — a gather that repacks on the fly
and a plain copy priced identically, which is the "cost model with no memory model"
``solve_layouts`` names.

An edge now carries its mover's own ``InstructionSpec.cost_of``, evaluated at the
value's size and the edge's ordering, and routing minimizes the sum. That is the whole
of pricing a relayout: an ISA states what a move costs exactly the way it states what
any other instruction costs, and no new notion of cost enters the frontend.

Fewest-hops remains the behaviour of a machine that prices nothing — every edge is
then 1.0 and the search is the breadth-first one it replaces, tie-breaks included.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import layout, view
from allo.exp.dsa.core import ISA
from allo.lang.core import f32

N = 8
T = f"tensor<{N}xf32>"
SRC = f"""
func.func @main(%x: {T}) -> {T} {{
  %y = tosa.abs %x : ({T}) -> {T}
  return %y : {T}
}}
"""
X = np.arange(N, dtype=np.float32) - 3


def _two_routes(direct_cost):
    """One direct ``mem -> vmem`` move against a two-hop path through ``spm``."""
    isa = ISA("priced")
    mem = isa.global_("mem", shape=(256,), dtype=f32)
    spm = isa.scalar("spm", 64, f32)
    vmem = isa.scalar("vmem", 64, f32)

    def mover(name, src, dst, cost):
        @isa.instruction(src=[src], dst=dst, name=name, cost=cost)
        def _(I):
            @I.access
            def _(s, d, n):
                return (view(src, s, n), view(dst, d, n))

            @I.compute
            def _(s, d):
                return primitive.identity(s)

    mover("slow_load", mem, vmem, direct_cost)  # one hop, priced by the caller
    mover("hop_in", mem, spm, 1.0)  # \ two hops,
    mover("hop_on", spm, vmem, 1.0)  # /  1.0 each
    mover("store", vmem, mem, 1.0)

    @isa.instruction(src=[vmem], dst=vmem)
    def vabs(I):
        @I.access
        def _(s, d, n):
            return (view(vmem, s, n), view(vmem, d, n))

        @I.compute
        def _(s, d):
            return primitive.abs(s)

    return isa


def test_routing_follows_price_not_hop_count():
    """The direct move costs 10 and the detour costs 2, so the planner detours."""
    prog = _two_routes(10.0).compile_program(SRC)
    assert [e.name for e in prog.emits] == ["hop_in", "hop_on", "vabs", "store"]
    np.testing.assert_allclose(prog(X), np.abs(X))


def test_an_unpriced_machine_still_takes_the_shortest_path():
    """With every edge at 1.0 the search is the hop-counting one it replaces."""
    prog = _two_routes(1.0).compile_program(SRC)
    assert [e.name for e in prog.emits] == ["slow_load", "vabs", "store"]
    np.testing.assert_allclose(prog(X), np.abs(X))


A, B, C = 2, 3, 4
CL = (2, 0, 1)
TT = f"tensor<{A}x{B}x{C}xf32>"
# `produce` writes channel-last; `consume` reads row-major out of `vmem`. Something
# has to repack, and the machine offers two ways to do it.
RELAYOUT_SRC = f"""
func.func @main(%x: {TT}) -> {TT} {{
  %t = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : ({TT}) -> {TT}
  %y = tosa.abs %t : ({TT}) -> {TT}
  return %y : {TT}
}}
"""


def _repack_or_gather(gather_cost):
    """A one-hop gather that repacks while it moves, against repack-then-copy."""
    isa = ISA("repack")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)
    vmem = isa.scalar("vmem", 256, f32)

    @isa.instruction(src=[mem], dst=mem)
    def produce(I):
        @I.access
        def _(s, d):
            return (
                layout(mem, s, (A, B, C)),
                layout(mem, d, (A, B, C), order=CL),
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[mem], dst=vmem, cost=gather_cost)
    def gather(I):
        """Reads channel-last, writes dense — one move, one relayout."""

        @I.access
        def _(s, d):
            return (layout(mem, s, (A, B, C), order=CL), view(vmem, d, (A, B, C)))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[mem], dst=mem, cost=1.0)
    def repack(I):
        @I.access
        def _(s, d):
            return (
                layout(mem, s, (A, B, C), order=CL),
                layout(mem, d, (A, B, C)),
            )

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[mem], dst=vmem, cost=1.0)
    def load(I):
        @I.access
        def _(s, d, n):
            return (view(mem, s, n), view(vmem, d, n))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[vmem], dst=mem, cost=1.0)
    def store(I):
        @I.access
        def _(s, d, n):
            return (view(vmem, s, n), view(mem, d, n))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[vmem], dst=vmem)
    def consume(I):
        @I.access
        def _(s, d):
            return (view(vmem, s, (A, B, C)), view(vmem, d, (A, B, C)))

        @I.compute
        def _(s, d):
            return primitive.abs(s)

    return isa


@pytest.mark.parametrize(
    "gather_cost, moves",
    [
        (1.0, ["gather"]),  # cheaper than repack+load
        (10.0, ["repack", "load"]),  # dearer, so the machine repacks in place first
    ],
)
def test_a_relayout_is_priced_like_any_other_move(gather_cost, moves):
    """The value is produced channel-last and consumed dense, so a repack is
    unavoidable; which one the planner buys is now a question with an answer."""
    prog = _repack_or_gather(gather_cost).compile_program(RELAYOUT_SRC)
    names = [e.name for e in prog.emits]
    assert names[: 1 + len(moves)] == ["produce"] + moves
    data = (np.arange(A * B * C, dtype=np.float32) + 1).reshape(A, B, C)
    np.testing.assert_allclose(prog(data), np.abs(np.maximum(data, 0)))
