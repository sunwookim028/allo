# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A buffer is an address space times a slot.

The model used to have two unrelated shapes of memory. An on-chip buffer was a 1-D
array *of slots* (a slot could be a vector or a tile), and the IR refused it anything
but a rank-1 access. Off-chip memory was a separate buffer *kind* — ``!allo.hbm`` —
whose "element" was secretly the whole array, carried a forced ``size(1)``, and was
the only thing a rank-2 access was legal on.

Those are the same object seen twice: ``extents`` addressable positions, each holding
one slot. Saying it once removes the ``hbm`` kind, removes the 1-D restriction, and —
because the restriction was never the point — leaves a rank-2 *on-chip* buffer as an
ordinary declaration rather than a thing the IR happens to forbid.

Three defects came out of the seam, all reproduced before this file existed:

- the bounds check on an on-chip buffer read a **dangling pointer**, so it never
  fired: a 64-element access into an 8-slot buffer verified clean;
- a 1-slot buffer with a non-scalar slot had its unit dim dropped from the memref but
  not from the access, so it verified and then failed to lower;
- a multi-dimensional access with a unit count disagreed about its own rank between
  the Python frontend and the IR.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, strided, view
from allo.exp.dsa.core import ISA
from allo.exp.dsa.errors import AllocationError
from allo.lang.core import f32

R, C, T = 8, 8, 4


def _nd_isa(rows=R, cols=C):
    """A rank-2 **on-chip** scratchpad, which the IR used to refuse outright: ``pad``
    reads a ``T x T`` block of it and writes the block back one row down."""
    isa = ISA("nd")
    mem = isa.global_("mem", shape=(256,), dtype=f32)
    sram = isa.buffer("sram", (rows, cols), f32)

    @isa.instruction(src=mem, dst=sram)
    def load(I):
        @I.access
        def _(s, r, c):
            return view(mem, s, (T, T)), strided(sram, [r, c], [T, T], [1, 1])

        @I.compute
        def _(a, d):
            return primitive.identity(a)

    @isa.instruction(src=sram, dst=mem)
    def store(I):
        @I.access
        def _(r, c, d):
            return strided(sram, [r, c], [T, T], [1, 1]), view(mem, d, (T, T))

        @I.compute
        def _(a, d):
            return primitive.identity(a)

    @isa.instruction(src=sram, dst=sram)
    def block_relu(I):
        @I.access
        def _(r, c, dr, dc):
            return (
                strided(sram, [r, c], [T, T], [1, 1]),
                strided(sram, [dr, dc], [T, T], [1, 1]),
            )

        @I.compute
        def _(a, d):
            return primitive.relu(a)

    return isa


# ==========================================================================#
# The declaration
# ==========================================================================#


def test_a_rank_2_on_chip_buffer_is_an_ordinary_declaration():
    """Previously: "on-chip buffers must be accessed in 1D patterns"."""
    text = str(_nd_isa().catalog())
    assert f"allo.buffer @sram extents({R}, {C}) : !allo.scalar<f32>" in text
    assert "allo.patterns.strided basis(%arg1, %arg2) counts(4, 4)" in text


def test_off_chip_memory_is_no_longer_a_separate_kind():
    """``ISA.hbm`` is now sugar: extents = the array's shape, slot = a word. The two
    declarations below are the same buffer, so the ``!allo.hbm`` type is gone."""
    isa = ISA("same")
    a = isa.hbm("a", shape=(R, C), dtype=f32)
    b = isa.buffer("b", (R, C), f32)
    assert (a.extents, a.kind) == (b.extents, b.kind)
    assert "!allo.hbm" not in str(isa.catalog())


def test_the_memref_is_extents_times_slot():
    """Including the unit extents the old rule dropped — that drop was invisible to
    the access patterns, which is what broke a 1-slot buffer."""
    isa = ISA("shapes")
    cases = [
        (isa.scalar("s", 256, dtype=f32), [256]),
        (isa.vector("v", 32, (16,), dtype=f32), [32, 16]),
        (isa.vector("v1", 1, (16,), dtype=f32), [1, 16]),
        (isa.tile("t1", 1, (16, 16), dtype=f32), [1, 16, 16]),
        (isa.hbm("h", (R, C), dtype=f32), [R, C]),
        (isa.buffer("nd", (R, C), f32, slot=(16,)), [R, C, 16]),
    ]
    assert [buf.memref_shape for buf, _want in cases] == [want for _buf, want in cases]


def test_extents_and_slot_are_independent():
    """A 2-D array *of vector registers* — a shape the old model could not spell at
    all, since a multi-dimensional address forced a scalar element."""
    isa = ISA("banked")
    banks = isa.buffer("banks", (4, 8), f32, slot=(16,))
    assert banks.address_rank == 2 and banks.slot_size == 16
    assert "allo.buffer @banks extents(4, 8) : !allo.vector<16xf32>" in str(
        isa.catalog()
    )


# ==========================================================================#
# The rules the unified model enforces
# ==========================================================================#


def test_an_access_must_have_one_component_per_extent():
    """Both directions, since it is one rule and not two policies."""
    isa = ISA("ranks")
    flat = isa.global_("flat", shape=(64,), dtype=f32)
    nd = isa.buffer("nd", (R, C), f32)

    def define(name, access):
        @isa.instruction(src=flat, dst=nd, name=name)
        def _(I):
            I.access(access)
            I.compute(lambda a, d: primitive.identity(a))

    define(
        "too_many",
        lambda s, r, c: (
            strided(flat, [s, 0], [T, T], [1, 1]),
            strided(nd, [r, c], [T, T], [1, 1]),
        ),
    )
    with pytest.raises(Exception, match="addressed by 1"):
        isa.catalog()

    isa2 = ISA("ranks2")
    flat2 = isa2.global_("flat", shape=(64,), dtype=f32)
    nd2 = isa2.buffer("nd", (R, C), f32)

    @isa2.instruction(src=flat2, dst=nd2)
    def too_few(I):
        @I.access
        def _(s, d):
            return contiguous(flat2, s, T * T), contiguous(nd2, d, T * T)

        @I.compute
        def _(a, d):
            return primitive.identity(a)

    with pytest.raises(Exception, match="addressed by 2"):
        isa2.catalog()


def test_the_bounds_check_actually_fires_on_a_flat_buffer():
    """It did not before: the extent was passed as an ``unsigned`` and read back
    through an ``ArrayRef`` bound to the temporary, so the comparison looked at freed
    stack memory and a wildly out-of-range access verified clean."""
    isa = ISA("oob")
    mem = isa.global_("mem", shape=(8,), dtype=f32)

    @isa.instruction(src=mem, dst=mem)
    def op(I):
        @I.access
        def _(s, d):
            return contiguous(mem, s, 64), contiguous(mem, d, 64)

        @I.compute
        def _(a, o):
            return primitive.relu(a)

    with pytest.raises(Exception, match="out of bounds in dimension 0"):
        isa.catalog()


def test_the_bounds_check_fires_per_axis():
    """Each address axis is checked against its own extent — the whole point of
    carrying extents rather than one number. A ``T x T`` block fits eight columns and
    overruns two."""
    _nd_isa(cols=T).catalog()  # exactly fits
    with pytest.raises(Exception, match="dimension 1: max index is 3"):
        _nd_isa(cols=2).catalog()


def test_a_unit_count_selects_rather_than_spans():
    """A count of 1 picks one position along that axis and so carries no tensor
    dimension — the rule that lets ``vld`` hand the compute region lanes rather than
    a ``1 x lanes`` tensor. It now reads the same on a multi-dimensional buffer,
    where the frontend used to keep the unit dim while the IR dropped it."""
    isa = ISA("unit")
    nd = isa.buffer("nd", (R, C), f32)
    mem = isa.global_("mem", shape=(64,), dtype=f32)

    @isa.instruction(src=nd, dst=mem)
    def row(I):
        @I.access
        def _(r, c, d):
            return strided(nd, [r, c], [1, C], [1, 1]), contiguous(mem, d, C)

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    # both sides rank 1: the frontend's visible shape and the IR's slice agree
    assert f"compute(%arg0: tensor<{C}xf32>, %arg1: tensor<{C}xf32>)" in str(
        isa.catalog()
    )


# ==========================================================================#
# End to end
# ==========================================================================#


def test_a_one_slot_vector_buffer_runs():
    """It used to verify and then fail to lower ("expected type to be 'tensor<1xf32>'
    or a rank-reduced version"): the memref had dropped the unit slot count, so the
    access wrote a slice of the wrong rank."""
    isa = ISA("oneslot")
    mem = isa.global_("mem", shape=(64,), dtype=f32)
    v = isa.vector("v", slots=1, shape=(8,), dtype=f32)

    def move(name, src, dst, access):
        @isa.instruction(src=src, dst=dst, name=name)
        def _(I):
            I.access(access)
            I.compute(lambda a, d: primitive.identity(a))

    move("ld", mem, v, lambda s, d: (contiguous(mem, s, 8), contiguous(v, d, 1)))
    move("st", v, mem, lambda s, d: (contiguous(v, s, 1), contiguous(mem, d, 8)))

    x = np.arange(8, dtype=np.float32)
    init = np.zeros(64, np.float32)
    init[:8] = x

    @isa.oracle(init={mem: init})
    def prog():
        isa._ops["ld"](s=0, d=0)
        isa._ops["st"](s=0, d=16)
        isa.inspect(mem[16:24], label="y")

    np.testing.assert_allclose(prog()["y"], x)


def test_a_rank_2_on_chip_buffer_computes():
    """The whole point: an on-chip scratchpad addressed by a coordinate, used as one.
    ``load`` places a tile at ``(2, 1)``, ``block_relu`` writes the rectified block to
    ``(4, 4)``, and ``store`` reads it back — none of which a 1-D access can say."""
    isa = _nd_isa()
    mem = isa.buffers["mem"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, T)).astype(np.float32)
    init = np.zeros(256, np.float32)
    init[: T * T] = x.reshape(-1)

    @isa.oracle(init={mem: init})
    def prog():
        isa._ops["load"](s=0, r=2, c=1)
        isa._ops["block_relu"](r=2, c=1, dr=4, dc=4)
        isa._ops["store"](r=4, c=4, d=64)
        isa.inspect(mem[64 : 64 + T * T], label="y")

    np.testing.assert_allclose(prog()["y"].reshape(T, T), np.maximum(x, 0.0))


def test_selecting_one_element_of_a_2d_buffer_reduces_to_a_scalar():
    """Two unit counts drop two dims, so an instruction addressing a single position
    of a 2-D buffer sees a rank-0 tensor. The IR used to reduce by exactly one dim
    whatever the counts said, which agreed with the frontend only for a 1-D access —
    the disagreement is invisible in the catalog and surfaces at lowering."""
    isa = ISA("elem")
    mem = isa.global_("mem", shape=(64,), dtype=f32)
    nd = isa.buffer("nd", (R, C), f32)

    def at(buf, r, c):
        return strided(buf, [r, c], [1, 1], [1, 1])

    def op(name, src, dst, access, compute):
        @isa.instruction(src=src, dst=dst, name=name)
        def _(I):
            I.access(access)
            I.compute(compute)

    ident = lambda a, d: primitive.identity(a)
    op("ld", mem, nd, lambda s, r, c: (contiguous(mem, s, 1), at(nd, r, c)), ident)
    op("st", nd, mem, lambda r, c, d: (at(nd, r, c), contiguous(mem, d, 1)), ident)
    op(
        "neg",
        nd,
        nd,
        lambda r, c, dr, dc: (at(nd, r, c), at(nd, dr, dc)),
        lambda a, d: primitive.negate(a),
    )

    assert "compute(%arg0: tensor<f32>, %arg1: tensor<f32>)" in str(isa.catalog())

    init = np.zeros(64, np.float32)
    init[0] = 3.5

    @isa.oracle(init={mem: init})
    def prog():
        isa._ops["ld"](s=0, r=2, c=3)
        isa._ops["neg"](r=2, c=3, dr=5, dc=1)
        isa._ops["st"](r=5, c=1, d=32)
        isa.inspect(mem[32:33], label="y")

    np.testing.assert_allclose(prog()["y"], [-3.5])


def test_a_unit_slot_dim_is_never_dropped():
    """Only *counts* select; a slot's own dims are always real tensor dims. The IR
    used to reduce by one dim regardless of the counts, so on a 1-lane vector
    register file it dropped the lane instead — and the compute region, which the
    frontend had typed rank-2, was handed a rank-1 operand."""
    isa = ISA("onelane")
    v = isa.vector("v", slots=4, shape=(1,), dtype=f32)
    mem = isa.global_("mem", shape=(64,), dtype=f32)

    @isa.instruction(src=v, dst=v)
    def rowsum(I):
        @I.access
        def _(s, d):
            return contiguous(v, s, 4), contiguous(v, d, 4)

        @I.compute
        def _(a, o):
            return primitive.reduce_sum(a, axis=1)  # needs the lane dim to exist

    assert "compute(%arg0: tensor<4x1xf32>, %arg1: tensor<4x1xf32>)" in str(
        isa.catalog()
    )

    def move(name, src, dst, access):
        @isa.instruction(src=src, dst=dst, name=name)
        def _(I):
            I.access(access)
            I.compute(lambda a, d: primitive.identity(a))

    move("ld", mem, v, lambda s, d: (view(mem, s, (4, 1)), contiguous(v, d, 4)))
    move("st", v, mem, lambda s, d: (contiguous(v, s, 4), view(mem, d, (4, 1))))

    x = np.arange(4, dtype=np.float32)
    init = np.zeros(64, np.float32)
    init[:4] = x

    @isa.oracle(init={mem: init})
    def prog():
        isa._ops["ld"](s=0, d=0)
        isa._ops["rowsum"](s=0, d=0)
        isa._ops["st"](s=0, d=16)
        isa.inspect(mem[16:20], label="y")

    np.testing.assert_allclose(prog()["y"], x)  # a 1-lane row sums to itself


def test_a_rank_2_on_chip_buffer_is_allocated_and_compiled_onto():
    """Placement is a coordinate on-chip too: the source is plain ``tosa.add`` and the
    planner routes through the 2-D scratchpad by itself."""
    isa = ISA("nd-compile")
    dram = isa.hbm("dram", shape=(32, T), dtype=f32)
    dram.is_global = True
    sram = isa.buffer("sram", (16, T), f32)

    def relay(name, src, dst):
        @isa.instruction(src=src, dst=dst, name=name)
        def _(I):
            I.access(
                lambda r, c, dr, dc: (
                    strided(src, [r, c], [T, T], [1, 1]),
                    strided(dst, [dr, dc], [T, T], [1, 1]),
                )
            )
            I.compute(lambda a, d: primitive.identity(a))

    relay("gather", dram, sram)
    relay("scatter", sram, dram)

    @isa.instruction(src=[sram, sram], dst=sram)
    def add(I):
        @I.access
        def _(ar, ac, br, bc, dr, dc):
            return (
                strided(sram, [ar, ac], [T, T], [1, 1]),
                strided(sram, [br, bc], [T, T], [1, 1]),
                strided(sram, [dr, dc], [T, T], [1, 1]),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    src = f"""
func.func @main(%a: tensor<{T}x{T}xf32>, %b: tensor<{T}x{T}xf32>) -> tensor<{T}x{T}xf32> {{
  %r = tosa.add %a, %b : (tensor<{T}x{T}xf32>, tensor<{T}x{T}xf32>) -> tensor<{T}x{T}xf32>
  return %r : tensor<{T}x{T}xf32>
}}
"""
    prog = isa.compile_program(src)
    assert [e.name for e in prog.emits] == ["gather", "gather", "add", "scatter"]
    # every location in a rank-2 buffer is a coordinate, on chip as well as off
    assert all(len(off) == 2 for off, _shape in prog.inputs)

    rng = np.random.default_rng(1)
    xs = [rng.standard_normal((T, T)).astype(np.float32) for _ in range(2)]
    np.testing.assert_allclose(prog(*xs), xs[0] + xs[1], rtol=1e-6, atol=1e-6)


def test_a_value_that_overflows_a_row_is_still_refused():
    """The fit check reads the buffer's extents; it used to read the *slot* shape,
    which for the unified model is empty — so it would have silently accepted
    anything."""
    from allo.exp.dsa.search import _placement_dims

    nd = ISA("fit").buffer("nd", (R, C), f32)
    assert _placement_dims((T, T), nd) == (T, T)
    with pytest.raises(AllocationError, match="does not fit"):
        _placement_dims((T, C * 2), nd)
