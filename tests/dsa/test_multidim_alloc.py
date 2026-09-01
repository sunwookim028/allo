# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-dimensional allocation: placing values in a buffer addressed by a coordinate.

``plan`` used to hold one flat offset per location, which is all a 1-D buffer needs. A
buffer declared with several extents is addressed by a full coordinate, so an offset
param names one *component* — and until a location could carry one, a relayout could be
written by hand but never selected: routing had no address to give it.

With placements as coordinates the pieces join up. The ISA below puts program I/O in a
row-major ``dram`` array and computes on a flat on-chip ``sram``; the source program
mentions neither, and the compiler inserts the gathers and the scatter by itself.

Allocation still packs a **single** axis (the outermost), so a value occupies a whole
band rather than a sub-rectangle — packing rectangles is 2-D bin packing, and the price
of skipping it is unused columns, not wrong code. These tests pin that down too."""

import numpy as np
import pytest

from allo.exp.dsa.errors import AllocationError

from allo.exp.dsa import primitive
from allo.exp.dsa.access import strided, view
from allo.exp.dsa.core import ISA
from allo.lang.core import f32

T = 4  # the tile the compute unit works on
ROWS, COLS = 32, 8


def _isa(sram_slots=64):
    """Program I/O lives in a 2-D row-major ``dram``; ``add`` works on flat ``sram``
    tiles. The only way between them is a relayout."""
    isa = ISA("tiled")
    dram = isa.hbm("dram", shape=(ROWS, COLS), dtype=f32)
    dram.is_global = True  # program I/O: the host writes plain 2-D arrays
    sram = isa.scalar("sram", sram_slots, dtype=f32)

    @isa.instruction(src=[dram], dst=sram)
    def gather(I):
        @I.access
        def _(r, c, d):
            return (strided(dram, [r, c], [T, T], [1, 1]), view(sram, d, (T, T)))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[sram], dst=dram)
    def scatter(I):
        @I.access
        def _(s, r, c):
            return (view(sram, s, (T, T)), strided(dram, [r, c], [T, T], [1, 1]))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    def binary(I, prim):
        @I.access
        def _(a, b, d):
            return (
                view(sram, a, (T, T)),
                view(sram, b, (T, T)),
                view(sram, d, (T, T)),
            )

        @I.compute
        def _(a, b, d):
            return prim(a, b)

    @isa.instruction(src=[sram, sram], dst=sram)
    def add(I):
        binary(I, primitive.add)

    return isa


def _src(n_args: int) -> str:
    """A left-deep chain over ``T x T`` tensors, e.g. ``((a+b)+c)``."""
    args = ", ".join(f"%a{i}: tensor<{T}x{T}xf32>" for i in range(n_args))
    ty = f"(tensor<{T}x{T}xf32>, tensor<{T}x{T}xf32>) -> tensor<{T}x{T}xf32>"
    lines, cur = [], "%a0"
    for i in range(1, n_args):
        lines.append(f"  %t{i} = tosa.add {cur}, %a{i} : {ty}")
        cur = f"%t{i}"
    body = "\n".join(lines)
    return (
        f"func.func @main({args}) -> tensor<{T}x{T}xf32> {{\n{body}\n"
        f"  return {cur} : tensor<{T}x{T}xf32>\n}}\n"
    )


def _names(prog):
    return [e.name for e in prog.emits]


def test_the_planner_inserts_the_relayouts_itself():
    """The source program is pure ``tosa.add`` — it says nothing about memories. The
    operands are in ``dram`` because that is where program I/O lives, the adder reads
    ``sram``, and routing closes the gap with a relayout per operand plus one to write
    the result back."""
    prog = _isa().compile_program(_src(3))
    assert _names(prog) == ["gather", "gather", "add", "gather", "add", "scatter"]


def test_it_computes_the_right_answer():
    """End to end through the multi-dimensional I/O path: the host hands over plain 2-D
    arrays, and the result comes back in the same layout."""
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((T, T)).astype(np.float32) for _ in range(4)]
    prog = _isa().compile_program(_src(4))
    np.testing.assert_allclose(prog(*xs), sum(xs), rtol=1e-5, atol=1e-5)


def test_placements_are_coordinates_and_pack_the_outermost_axis():
    """A location in a multi-dimensional buffer holds a coordinate, not a number. The
    allocator packs one axis, so the four inputs take successive ``T``-row bands at
    column 0 — and the gathers' ``(r, c)`` params are read straight off those
    placements."""
    prog = _isa().compile_program(_src(4))
    assert [off for off, _shape in prog.inputs] == [(0, 0), (4, 0), (8, 0), (12, 0)]
    gathers = [e for e in prog.emits if e.name == "gather"]
    assert [(e.addr[0], e.addr[1]) for e in gathers] == [
        (0, 0),
        (4, 0),
        (8, 0),
        (12, 0),
    ]
    # the result is written back to a band the allocator freed and reused
    (out_off, _shape, _label) = prog.outputs[0]
    assert out_off[1] == 0 and out_off[0] % T == 0


def test_a_dead_bands_rows_are_reused():
    """Releasing a location returns its rows to the free list exactly as slots, so a
    long chain does not grow ``dram`` without bound: the result reuses a dead input's
    band rather than extending past all of them."""
    prog = _isa().compile_program(_src(6))
    (out_off, _shape, _label) = prog.outputs[0]
    assert out_off[0] < 6 * T  # not appended after every input


def test_a_move_that_does_not_fit_the_value_is_refused():
    """Routing picks a move by *buffer pair*, so a fixed-size relayout can be handed a
    value it cannot carry — nothing upstream checked, because a move never goes through
    Stage-2 solve. Each pattern's element count is now checked against the value's;
    without it the copy would silently truncate to the tile the instruction describes.
    """
    from allo.exp.dsa.search import _solve_move_params

    spec = _isa()._ops["gather"].spec
    assert _solve_move_params(spec, T * T) == {}  # the size it does carry
    with pytest.raises(AllocationError, match="element"):
        _solve_move_params(spec, 2 * T * T)


def test_a_value_of_the_wrong_rank_has_no_placement():
    """A multi-dimensional buffer holds values of its own rank (modulo the leading unit
    dims torch brackets matmuls in, which are a rank alias). A 1-D value has no
    coordinate there, and an over-wide one does not fit the array."""
    from allo.exp.dsa.search import _placement_dims

    dram = _isa().buffers["dram"]
    assert _placement_dims((T, T), dram) == (T, T)
    assert _placement_dims((1, T, T), dram) == (T, T)  # the batched rank alias
    with pytest.raises(AllocationError, match="addressed by 2 indices"):
        _placement_dims((16,), dram)
    with pytest.raises(AllocationError, match="does not fit"):
        _placement_dims((T, COLS * 2), dram)


def _wide_src() -> str:
    """``(a0+a1) + (a2+a3) + (a4+a5)`` with the three partials computed first, so
    several tiles are live at once."""
    args = ", ".join(f"%a{i}: tensor<{T}x{T}xf32>" for i in range(6))
    ty = f"(tensor<{T}x{T}xf32>, tensor<{T}x{T}xf32>) -> tensor<{T}x{T}xf32>"
    lines = [f"  %p{i} = tosa.add %a{2 * i}, %a{2 * i + 1} : {ty}" for i in range(3)]
    lines += [f"  %s0 = tosa.add %p0, %p1 : {ty}", f"  %s1 = tosa.add %s0, %p2 : {ty}"]
    body = "\n".join(lines)
    return (
        f"func.func @main({args}) -> tensor<{T}x{T}xf32> {{\n{body}\n"
        f"  return %s1 : tensor<{T}x{T}xf32>\n}}\n"
    )


def test_spilling_still_works_across_the_relayout():
    """Nothing above the placement change knows about coordinates, so liveness /
    best-fit / Belady are untouched: shrink ``sram`` until the partials cannot all stay
    resident, and the extra traffic is ordinary moves back out to the 2-D ``dram`` —
    spilling *through* a relayout, with the numbers still right."""
    rng = np.random.default_rng(1)
    xs = [rng.standard_normal((T, T)).astype(np.float32) for _ in range(6)]
    prog = _isa(sram_slots=3 * T * T).compile_program(_wide_src())
    assert _names(prog).count("scatter") > 1  # a spill, on top of the one output
    np.testing.assert_allclose(prog(*xs), sum(xs), rtol=1e-5, atol=1e-5)
