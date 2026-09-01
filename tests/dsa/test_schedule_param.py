# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``@I.schedule`` — the instruction's third kind of parameter.

An ``@I.access`` param is *solved* (from the source shapes, or from a neighbour's
residence); an ``@I.compute`` param (α) is *bound* from a constant in the source and
the computed value depends on it. A schedule param is neither: it is freely chosen
by the compiler, the value does not depend on it, and it reaches the compiler through
exactly two channels — legality (the ``@I.schedule`` predicate, ACT's ``e_θ``) and
cost. It is what a schedule-ISA machine's configuration carries.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, layout, strided, view
from allo.exp.dsa.core import ISA
from allo.exp.dsa.errors import (
    AcceleratorDescriptionError,
    CompileError,
    NoMatchError,
)
from allo.lang.core import f32


def _add(n: int) -> str:
    return f"""
func.func @main(%a: tensor<{n}xf32>, %b: tensor<{n}xf32>) -> tensor<{n}xf32> {{
  %r = tosa.add %a, %b : (tensor<{n}xf32>, tensor<{n}xf32>) -> tensor<{n}xf32>
  return %r : tensor<{n}xf32>
}}"""


def _isa(name, *, domains=None, predicate=None, cost=None):
    """An ISA with one parametric vector add, optionally carrying a schedule param
    ``lanes_used`` that says how many of the machine's lanes the pass engages."""
    isa = ISA(name)
    mem = isa.global_("mem", shape=(4096,), dtype=f32)

    @isa.instruction(src=[mem, mem], dst=mem, cost=cost)
    def vadd(I):
        @I.access
        def _(a, b, d, n):
            return (
                contiguous(mem, a, n),
                contiguous(mem, b, n),
                contiguous(mem, d, n),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

        if predicate is not None:
            I.schedule(**(domains or {}))(predicate)

    return isa


def test_a_schedule_param_is_chosen_and_emitted():
    """The compiler picks a legal value, and it lands in the instruction word as a
    field of its own — distinct from the addresses and from α."""
    isa = _isa(
        "ChosenTPU",
        domains={"lanes_used": [1, 2, 4]},
        predicate=lambda n, lanes_used: n % lanes_used == 0,
        # Fewer lanes is more trips, so the cheapest legal choice is the widest.
        cost=lambda n, lanes_used: float(n / lanes_used),
    )
    prog = isa.compile_program(_add(8))
    (rec,) = [e for e in prog.emits if e.name == "vadd"]
    assert rec.schedule == [4]
    assert "@4" in str(prog)  # printed as a chosen field, not an address


def test_the_cheapest_legal_configuration_wins():
    """``configure`` minimizes cost over the domain, so a predicate that rules the
    cheap choices out is visible in the emitted configuration."""
    isa = _isa(
        "PickyTPU",
        domains={"lanes_used": [1, 2, 4]},
        predicate=lambda lanes_used: lanes_used <= 2,  # the widest is illegal here
        cost=lambda n, lanes_used: float(n / lanes_used),
    )
    (rec,) = [e for e in isa.compile_program(_add(8)).emits if e.name == "vadd"]
    assert rec.schedule == [2]


def test_an_unconfigurable_instruction_is_not_a_candidate():
    """A predicate no assignment satisfies means the hardware cannot be put into a
    configuration that runs this site — reported as such, not as a shape mismatch and
    not by emitting an impossible instruction."""
    isa = _isa(
        "NarrowTPU",
        domains={"lanes_used": [4]},
        predicate=lambda n, lanes_used: n % lanes_used == 0,
    )
    with pytest.raises(NoMatchError, match="no legal configuration"):
        isa.compile_program(_add(6))  # 6 % 4 != 0


def test_the_value_does_not_depend_on_the_schedule_param():
    """The defining property: a schedule param changes the configuration, never the
    result. The simulator ignores it and the program still matches NumPy."""
    isa = _isa(
        "RunTPU",
        domains={"lanes_used": [1, 2, 4]},
        predicate=lambda n, lanes_used: n % lanes_used == 0,
        cost=lambda n, lanes_used: float(n / lanes_used),
    )
    rng = np.random.default_rng(7)
    a = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    np.testing.assert_allclose(isa.compile_program(_add(8))(a, b), a + b, rtol=1e-6)


def test_a_predicate_with_no_domains_restricts_shapes():
    """Declaring no schedule params is allowed and useful: the predicate then simply
    says which solved shapes the instruction accepts — the ``e_θ`` half alone."""
    isa = _isa("EvenTPU", domains={}, predicate=lambda n: n % 4 == 0)
    assert len(isa.compile_program(_add(8)).emits) == 1
    with pytest.raises(NoMatchError, match="no legal configuration"):
        isa.compile_program(_add(6))


def test_a_shape_param_cannot_be_given_a_domain():
    """Naming an access param does not introduce one, it makes that one *chosen* — and
    a shape param is not the compiler's to choose: the source program pins it."""
    with pytest.raises(AcceleratorDescriptionError, match="'shape' access param"):
        _isa("ShadowTPU", domains={"n": [1, 2]}, predicate=lambda n: True)


def test_an_empty_domain_is_refused():
    with pytest.raises(AcceleratorDescriptionError, match="empty domain"):
        _isa("EmptyTPU", domains={"lanes_used": []}, predicate=lambda lanes_used: True)


def test_the_predicate_may_only_read_declared_params():
    with pytest.raises(AcceleratorDescriptionError, match="neither an @access param"):
        _isa("TypoTPU", domains={"lanes_used": [1]}, predicate=lambda lanes: True)


# ==========================================================================#
# @I.expand issues instructions, and those get configured too
# ==========================================================================#


def _layered(name, *, tile_predicate, layer_predicate, n_tiles=4):
    """An ISA with a fixed-size ``vadd4`` carrying a schedule param, plus a
    layer-level ``vadd_layer`` whose ``@I.expand`` issues ``n_tiles`` of them."""
    isa = ISA(name)
    mem = isa.global_("mem", shape=(4096,), dtype=f32)

    @isa.instruction(src=[mem, mem], dst=mem, cost=lambda lanes_used: 4.0 / lanes_used)
    def vadd4(I):
        @I.access
        def _(a, b, d):
            return (
                contiguous(mem, a, 4),
                contiguous(mem, b, 4),
                contiguous(mem, d, 4),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

        I.schedule(lanes_used=[1, 2, 4])(tile_predicate)

    @isa.instruction(src=[mem, mem], dst=mem, cost=lambda n: float(n))
    def vadd_layer(I):
        @I.access
        def _(a, b, d, n):
            return (
                contiguous(mem, a, n),
                contiguous(mem, b, n),
                contiguous(mem, d, n),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

        I.schedule()(layer_predicate)

        @I.expand
        def _(a, b, d, n):
            for k in range(n // 4):
                vadd4(a=a + 4 * k, b=b + 4 * k, d=d + 4 * k)

    return isa


def test_the_expansion_configures_the_instructions_it_issues():
    """An expansion is the compiler's own lowering, so the schedule params of what it
    issues are the compiler's to choose — not left empty, and not hard-coded by the
    ISA author in the expansion body."""
    isa = _layered(
        "ExpandTPU",
        tile_predicate=lambda lanes_used: lanes_used <= 2,
        layer_predicate=lambda n: n % 4 == 0,
    )
    prog = isa.compile_program(_add(16))
    tiles = [e for e in prog.emits if e.name == "vadd4"]
    assert len(tiles) == 4
    # 2 is the cheapest the tile's own predicate admits — chosen per issued
    # instruction, from the tile's domain, not the layer's.
    assert all(t.schedule == [2] for t in tiles), tiles


def test_an_expansion_that_issues_an_unconfigurable_instruction_is_caught():
    """A layer-level instruction states what it can lower; if its predicate admits
    more than its expansion can actually issue, the expansion says so rather than
    emitting an instruction with no configuration."""
    isa = _layered(
        "OverpromiseTPU",
        tile_predicate=lambda lanes_used: False,  # no tile configuration is legal
        layer_predicate=lambda n: n % 4 == 0,  # but the layer claims it can lower
    )
    with pytest.raises(CompileError, match="its @expand issues 'vadd4'"):
        isa.compile_program(_add(16))


# ==========================================================================#
# A mover configures too — the same decorator, chosen by the router
# ==========================================================================#
#
# `InstructionSpec.configurations` is the one place a finite parameter domain is
# enumerated. A matched instruction's chosen params are its fresh ones; a mover also
# chooses its *residence* params, because the planner is what inserts it and so it
# unifies with nothing. Naming an access param in `@I.schedule` is what turns that one
# from solved into chosen — permitted only there, and only for a residence role.


def _strided_isa(name, *, domain=(1, 2, 4), lane_stride=2):
    """A machine whose vector unit reads lanes ``lane_stride`` slots apart, reached by
    movers that carry the stride as a *chosen* parameter."""
    isa = ISA(name)
    mem = isa.global_("mem", shape=(256,), dtype=f32)
    vmem = isa.scalar("vmem", 64, f32)

    def mover(mnemonic, src, dst, pattern):
        @isa.instruction(src=[src], dst=dst, name=mnemonic)
        def _(I):
            @I.access
            def _(s, d, n, k):
                return pattern(s, d, n, k)

            @I.compute
            def _(s, d):
                return primitive.identity(s)

            @I.schedule(k=domain)
            def _(k):
                return True

    mover(
        "scatter",
        mem,
        vmem,
        lambda s, d, n, k: (view(mem, s, n), strided(vmem, d, n, k)),
    )
    mover(
        "gather",
        vmem,
        mem,
        lambda s, d, n, k: (strided(vmem, s, n, k), view(mem, d, n)),
    )

    @isa.instruction(src=[vmem], dst=vmem)
    def vabs(I):
        @I.access
        def _(s, d, n):
            return (strided(vmem, s, n, lane_stride), strided(vmem, d, n, lane_stride))

        @I.compute
        def _(s, d):
            return primitive.abs(s)

    return isa


_ABS = """
func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
  %y = tosa.abs %x : (tensor<8xf32>) -> tensor<8xf32>
  return %y : tensor<8xf32>
}"""


def test_a_movers_stride_is_chosen_by_the_router():
    """A free stride, which had no home at all before: nothing unifies with a move's
    residence params, so the ISA declares the domain and the router picks from it —
    here the stride the consumer's own access describes."""
    prog = _strided_isa("Strided").compile_program(_ABS)
    assert [(e.name, e.addr) for e in prog.emits] == [
        ("scatter", [0, 0, 8, 2]),
        ("vabs", [0, 0, 8]),
        ("gather", [0, 0, 8, 2]),
    ]
    x = np.arange(8, dtype=np.float32) - 3
    np.testing.assert_allclose(prog(x), np.abs(x))


def test_a_movers_domain_bounds_the_residences_it_can_produce():
    """The domain is the whole of what the router may choose from, so a stride the
    hardware cannot encode is a layout the value simply cannot be brought into."""
    isa = _strided_isa("TooCoarse", domain=(1, 4))
    with pytest.raises(CompileError, match=r"strides \[2\].*no data movement"):
        isa.compile_program(_ABS)


def test_a_mover_that_admits_no_configuration_is_not_an_edge():
    """The predicate is consulted where the move is *inserted*, so an unsatisfiable one
    removes the mover from the movement graph rather than being silently dropped."""
    isa = _strided_isa("Impossible", domain=(1, 2, 4))
    isa._ops["scatter"].spec.schedule_fn = lambda k: False
    with pytest.raises(CompileError, match=r"\['scatter'\] leave\(s\) those buffers"):
        isa.compile_program(_ABS)


def test_a_matched_instructions_residence_param_cannot_be_given_a_domain():
    """It is *solved* by unifying the value's accesses, so a domain would go unused —
    only a mover, which unifies with nothing, chooses its own residence."""
    isa = ISA("Matched")
    mem = isa.global_("mem", shape=(256,), dtype=f32)

    with pytest.raises(AcceleratorDescriptionError, match="is \\*matched\\*"):

        @isa.instruction(src=[mem], dst=mem)
        def vabs(I):
            @I.access
            def _(s, d, n, k):
                return (strided(mem, s, n, k), view(mem, d, n))

            @I.compute
            def _(s, d):
                return primitive.abs(s)

            @I.schedule(k=[1, 2])
            def _(k):
                return True


def _network_isa(reachable):
    """A gather whose permutation network reaches only some packings in one pass —
    the cheap half of the interconnect reachability relation ``R``, stated as a
    predicate over an ordering the router would otherwise be free to choose."""
    A, B, C = 2, 3, 4
    isa = ISA("Network")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)
    vmem = isa.scalar("vmem", 256, f32)

    @isa.instruction(src=[mem], dst=mem)
    def produce(I):
        @I.access
        def _(s, d):
            return (
                layout(mem, s, (A, B, C)),
                layout(mem, d, (A, B, C), order=(2, 0, 1)),
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[mem], dst=vmem)
    def birrd(I):
        @I.access
        def _(s, d, q):
            return (layout(mem, s, (A, B, C), order=q), view(vmem, d, (A, B, C)))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

        @I.schedule
        def _(q):
            return q in reachable

    @isa.instruction(src=[vmem], dst=mem)
    def store(I):
        @I.access
        def _(s, d, q):
            return (view(vmem, s, (A, B, C)), layout(mem, d, (A, B, C), order=q))

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


_RELU_ABS = """
func.func @main(%x: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %t = tosa.clamp %x {min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %y = tosa.abs %t : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %y : tensor<2x3x4xf32>
}"""


def test_a_movers_predicate_narrows_the_packings_it_can_gather():
    """An ordering param needs no declaration — its domain is intrinsic — but a
    predicate over it is how a machine says its network cannot reach every packing."""
    prog = _network_isa({(2, 0, 1), (0, 1, 2)}).compile_program(_RELU_ABS)
    assert [e.name for e in prog.emits] == ["produce", "birrd", "consume", "store"]
    data = (np.arange(24, dtype=np.float32) + 1).reshape(2, 3, 4)
    np.testing.assert_allclose(prog(data), np.abs(np.maximum(data, 0)))

    with pytest.raises(CompileError, match="no data movement"):
        _network_isa({(0, 1, 2)}).compile_program(_RELU_ABS)
