# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout as residence, and the relayout the planner inserts to repair a mismatch.

Stage 2b *solves* a value's layout params; it does not decide whether two accesses
that end up disagreeing are compilable, because that depends on the machine having a
mover that repacks between them — and only the planner knows the move graph. So a
location carries its residence (``_Loc.map``), an access states the residence it
needs, and getting a value from one to the other is **routing**: a value in the right
buffer but the wrong layout is one repacking edge away, exactly as a value in the
wrong buffer is one copy away.

Two rules make the graph work, and both are properties of what a mover physically
does (``search._Edge``):

- a mover whose two accesses describe the **same** map copies its region verbatim, so
  it carries *whatever* residence the value had — which is why a plain dma can spill a
  channel-last value and reload it unharmed;
- a mover whose accesses **differ** permutes addresses, so it applies only to a value
  laid out exactly as it reads, and then lays it out exactly as it writes.

This closes ``todos/act-backend.md`` §5's open item 2, and the gap phase 7 left: a
producer packing channel-last on chip while a dense mover copies to the I/O pool used
to compile and hand the host scrambled data.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import layout, view
from allo.exp.dsa.core import ISA
from allo.exp.dsa.errors import AllocationError, LayoutError
from allo.exp.dsa.search import _movement_catalog
from allo.lang.core import f32

A, B, C = 2, 3, 4
N = A * B * C
CL = (2, 0, 1)  # dim 2 outermost -- "channel last"
ODD = (1, 2, 0)  # a third packing, so "some ordering" is not "any ordering"
_T = f"tensor<{A}x{B}x{C}xf32>"

# relu then abs: two matchable ops with one value between them. On strictly positive
# input the answer is the input, so a mis-routed layout shows up as a permutation.
_SRC = f"""
func.func @main(%x: {_T}) -> {_T} {{
  %t = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : ({_T}) -> {_T}
  %y = tosa.abs %t : ({_T}) -> {_T}
  return %y : {_T}
}}
"""

_X = (np.arange(N, dtype=np.float32) + 1).reshape(A, B, C)


def _machine(*, produced=None, consumed=CL, written=None, repacks=()):
    """A flat I/O pool plus an on-chip scratchpad, joined by dense movers.

    ``vrelu`` writes its result in ``produced``; ``vabs`` reads its operand in
    ``consumed`` and writes its own in ``written``. Each entry of ``repacks`` adds a
    ``vmem -> vmem`` relayout between the given orderings. Nothing else in the machine
    touches a layout, so any repacking in a compiled program was put there by the
    planner."""
    isa = ISA("repairing")
    mem = isa.global_("mem", shape=(512,), dtype=f32)
    vmem = isa.scalar("vmem", 256, f32)

    @isa.instruction(src=[mem], dst=vmem)
    def load(I):
        @I.access
        def _(s, d, n):
            return (view(mem, s, n), view(vmem, d, n))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[vmem], dst=mem)
    def store(I):
        @I.access
        def _(s, d, n):
            return (view(vmem, s, n), view(mem, d, n))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    for i, (src, dst) in enumerate(repacks):
        isa.instruction(src=[vmem], dst=vmem, name=f"repack{i}")(
            _repack_body(vmem, src, dst)
        )

    @isa.instruction(src=[vmem], dst=vmem)
    def vrelu(I):
        @I.access
        def _(s, d):
            return (
                layout(vmem, s, (A, B, C)),
                layout(vmem, d, (A, B, C), order=produced),
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[vmem], dst=vmem)
    def vabs(I):
        @I.access
        def _(s, d):
            return (
                layout(vmem, s, (A, B, C), order=consumed),
                layout(vmem, d, (A, B, C), order=written),
            )

        @I.compute
        def _(s, d):
            return primitive.abs(s)

    return isa


def _repack_body(vmem, src, dst):
    """An identity move that reads one packing and writes another."""

    def body(I):
        @I.access
        def _(s, d):
            return (
                layout(vmem, s, (A, B, C), order=src),
                layout(vmem, d, (A, B, C), order=dst),
            )

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    return body


def _names(prog) -> list:
    return [e.name for e in prog.emits]


# --- the repair ------------------------------------------------------------------


def test_the_planner_inserts_the_relayout_a_mismatch_needs():
    """The acceptance. ``vrelu`` writes row-major, ``vabs`` reads channel-last, and
    nothing in the source program says so — the planner sees a value that is not
    resident in the layout the next access wants, routes to it over the move graph,
    and the repack lands between the two."""
    prog = _machine(repacks=[(None, CL)]).compile_program(_SRC)
    assert _names(prog) == ["load", "vrelu", "repack0", "vabs", "store"]
    np.testing.assert_allclose(prog(_X), _X)


def test_no_relayout_is_inserted_when_the_two_ends_agree():
    """The control: the same machine, with the consumer reading what the producer
    wrote. A repacking mover exists and must stay unused — routing is driven by the
    mismatch, not by the mover being available."""
    prog = _machine(consumed=None, repacks=[(None, CL)]).compile_program(_SRC)
    assert _names(prog) == ["load", "vrelu", "vabs", "store"]
    np.testing.assert_allclose(prog(_X), _X)


def test_a_machine_that_cannot_repack_says_so():
    """Same program, same disagreement, no mover that bridges it. The message names
    the access that cannot be satisfied and both residences — the two ends the plan
    asks for."""
    with pytest.raises(LayoutError) as exc:
        _machine().compile_program(_SRC)
    text = str(exc.value)
    assert "vabs operand 0" in text
    assert "as sizes [2, 3, 4] strides [3, 1, 6]" in text  # needs channel-last
    assert "as sizes [2, 3, 4] strides [12, 4, 1]" in text  # has row-major


def test_the_host_abi_is_repaired_on_the_way_out():
    """The gap phase 7 left: the producer packs channel-last *on chip* and the mover
    to the I/O pool copies densely, so the host would read a permutation of its
    result. Nothing checked that before — the accesses were in different buffers.
    Now the output's residence is the ABI's, and the planner repacks to reach it."""
    packed = dict(produced=CL, consumed=CL, written=CL)
    prog = _machine(**packed, repacks=[(CL, None)]).compile_program(_SRC)
    assert _names(prog) == ["load", "vrelu", "vabs", "repack0", "store"]
    np.testing.assert_allclose(prog(_X), _X)

    with pytest.raises(LayoutError, match="result #0"):
        _machine(**packed).compile_program(_SRC)


# --- what a mover does to a layout -------------------------------------------------


def test_a_mover_that_repacks_applies_only_to_what_it_reads():
    """A repacking mover permutes addresses, so it is meaningful only on a value laid
    out exactly as it reads. Offered a third packing it is not a usable edge at all,
    rather than a way to scramble the data further."""
    with pytest.raises(LayoutError):
        _machine(produced=ODD, repacks=[(None, CL)]).compile_program(_SRC)
    # ... and with the edge that *does* read that packing, the same program compiles.
    prog = _machine(produced=ODD, repacks=[(ODD, CL)]).compile_program(_SRC)
    assert _names(prog) == ["load", "vrelu", "repack0", "vabs", "store"]
    np.testing.assert_allclose(prog(_X), _X)


def test_a_mover_that_agrees_with_itself_carries_any_layout():
    """The other rule, and the one every existing machine relies on. A mover whose two
    accesses agree copies its region verbatim, so it moves a *channel-last* value to
    another buffer and leaves it channel-last — no repack at either end, and no repack
    exists in this machine to hide the answer. It is also why an ordinary dma can spill
    a repacked value and reload it unharmed."""
    isa = ISA("carrier")
    mem = isa.global_("mem", shape=(512,), dtype=f32)
    vmem = isa.scalar("vmem", 256, f32)
    bram = isa.scalar("bram", 256, f32)

    def copier(name, src, dst):
        def body(I):
            @I.access
            def _(s, d, n):
                return (view(src, s, n), view(dst, d, n))

            @I.compute
            def _(s, d):
                return primitive.identity(s)

        isa.instruction(src=[src], dst=[dst], name=name)(body)

    copier("load", mem, vmem)
    copier("hop", vmem, bram)
    copier("save", bram, mem)

    @isa.instruction(src=[vmem], dst=vmem)
    def vrelu(I):
        @I.access
        def _(s, d):
            return (layout(vmem, s, (A, B, C)), layout(vmem, d, (A, B, C), order=CL))

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[bram], dst=bram)
    def babs(I):
        @I.access
        def _(s, d):
            return (layout(bram, s, (A, B, C), order=CL), layout(bram, d, (A, B, C)))

        @I.compute
        def _(s, d):
            return primitive.abs(s)

    prog = isa.compile_program(_SRC)
    assert _names(prog) == ["load", "vrelu", "hop", "babs", "save"]
    np.testing.assert_allclose(prog(_X), _X)
    assert _movement_catalog(isa) == ["load", "hop", "save"]


def test_several_movers_may_join_one_pair_of_buffers():
    """A machine can offer a plain copy and a repacking dma between the same two
    buffers — they differ only in what they do to the layout, which a buffer pair
    cannot express. So a mover is an edge in its own right, and routing picks by
    residence rather than by buffer pair."""
    isa = _machine(repacks=[(None, CL), (CL, None)])
    assert sorted(_movement_catalog(isa)) == ["load", "repack0", "repack1", "store"]
    prog = isa.compile_program(_SRC)
    assert _names(prog) == ["load", "vrelu", "repack0", "vabs", "store"]
    np.testing.assert_allclose(prog(_X), _X)


def test_a_relayout_that_does_not_fit_the_value_is_not_an_edge():
    """Movers are sized against the value before they are routed over, so a fixed-size
    repack simply is not an edge for a value it cannot carry. Routing used to pick a
    move by buffer pair alone and discover the misfit at emission."""
    isa = ISA("misfit")
    mem = isa.global_("mem", shape=(512,), dtype=f32)
    vmem = isa.scalar("vmem", 256, f32)

    @isa.instruction(src=[mem], dst=vmem)
    def load(I):
        @I.access
        def _(s, d, n):
            return (view(mem, s, n), view(vmem, d, n))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[vmem], dst=mem)
    def store(I):
        @I.access
        def _(s, d, n):
            return (view(vmem, s, n), view(mem, d, n))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    # a repack for 2x3x4 tensors only ...
    isa.instruction(src=[vmem], dst=vmem, name="repack")(_repack_body(vmem, None, CL))

    @isa.instruction(src=[vmem], dst=vmem)
    def vrelu(I):
        @I.access
        def _(s, d):
            return (view(vmem, s, (B, C)), view(vmem, d, (B, C)))

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[vmem], dst=vmem)
    def vabs(I):
        """... but this reads a 3x4 tensor channel-last-ish, which it cannot produce."""

        @I.access
        def _(s, d):
            return (
                layout(vmem, s, (B, C), order=(1, 0)),
                view(vmem, d, (B, C)),
            )

        @I.compute
        def _(s, d):
            return primitive.abs(s)

    small = f"tensor<{B}x{C}xf32>"
    src = f"""
func.func @main(%x: {small}) -> {small} {{
  %t = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : ({small}) -> {small}
  %y = tosa.abs %t : ({small}) -> {small}
  return %y : {small}
}}
"""
    with pytest.raises(LayoutError, match="vabs operand 0"):
        isa.compile_program(src)


def test_an_unreachable_buffer_is_still_reported_as_routing():
    """Residence did not replace the old diagnosis, it refined it: a value that cannot
    reach the buffer at all in *any* layout is a routing failure, not a layout one."""
    isa = ISA("stranded")
    mem = isa.global_("mem", shape=(512,), dtype=f32)
    vmem = isa.scalar("vmem", 256, f32)

    @isa.instruction(src=[vmem], dst=vmem)
    def vrelu(I):
        @I.access
        def _(s, d):
            return (view(vmem, s, (B, C)), view(vmem, d, (B, C)))

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    small = f"tensor<{B}x{C}xf32>"
    src = f"""
func.func @main(%x: {small}) -> {small} {{
  %y = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : ({small}) -> {small}
  return %y : {small}
}}
"""
    with pytest.raises(AllocationError, match="no data-movement route"):
        isa.compile_program(src)
    assert mem.name == "mem"  # the machine has an I/O pool, just no way off it
