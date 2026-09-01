# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relayout DMA: a move whose two access patterns differ.

A ``TILE x TILE`` block of a row-major matrix is not contiguous — its rows are one
stride apart — so a tile-granular accelerator cannot consume a matrix directly. That
gap is what "converting a matrix into tiles is a data-movement problem" means, and the
mechanism that closes it is a **relayout**: identity compute (the same values, in the
same order) with *different* access patterns on the two sides, which is exactly how a
DMA engine rearranges.

The rank-2 side needs a buffer *declared* with two extents, since an access carries
one component per extent (``StridedOp::verifyCompatibility``). ``ISA.hbm`` is the
usual way to spell that — off-chip memory as one row-major array — but nothing about
it is off-chip-specific; see ``test_nd_buffer.py``.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, strided, view
from allo.exp.dsa.core import ISA
from allo.exp.dsa.search import _movement_catalog
from allo.lang.core import f32

R, C, T = 8, 8, 4


def _isa():
    """A row-major ``dram`` array plus flat on-chip ``sram``, joined by a relayout."""
    isa = ISA("relayout")
    dram = isa.hbm("dram", shape=(R, C), dtype=f32)
    sram = isa.global_("sram", shape=(256,), dtype=f32)

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

    return isa, dram, sram, gather, scatter


def test_an_access_carries_one_component_per_buffer_extent():
    """``dram`` is addressed by a coordinate because it was *declared* with two
    extents — nothing about it being off-chip. The rule the IR enforces is rank
    agreement between the access and the buffer's address space, so a rank-2 pattern
    on a flat buffer is refused for the same reason a rank-1 pattern on ``dram``
    would be."""
    isa, _dram, sram, _g, _s = _isa()
    text = str(isa.catalog())
    assert f"allo.buffer @dram extents({R}, {C}) : !allo.scalar<f32>" in text

    bad = ISA("bad")
    flat = bad.global_("mem", shape=(64,), dtype=f32)

    @bad.instruction(src=[flat], dst=flat)
    def gather2d(I):
        @I.access
        def _(r, c, d):
            return (strided(flat, [r, c], [T, T], [1, 1]), view(flat, d, (T, T)))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    with pytest.raises(Exception, match="addressed by 1"):
        bad.catalog()


def test_gather_rearranges_a_row_major_block_into_a_contiguous_tile():
    """The defining property: the block's rows are ``C`` apart in ``dram`` and adjacent
    in ``sram``. Deliberately gathers an *unaligned* block too — the relayout is an
    arbitrary window, not a fixed tiling."""
    isa, dram, sram, gather, _scatter = _isa()
    A = np.arange(R * C, dtype=np.float32).reshape(R, C)

    @isa.oracle(init={dram: A})
    def prog():
        gather(r=0, c=0, d=0)
        gather(r=4, c=4, d=16)
        gather(r=2, c=1, d=32)  # unaligned to the tile grid
        isa.inspect(sram[0:48], label="tiles")

    got = prog()["tiles"].reshape(3, T, T)
    want = np.stack([A[0:4, 0:4], A[4:8, 4:8], A[2:6, 1:5]])
    np.testing.assert_allclose(got, want)


def test_scatter_is_the_inverse_relayout():
    """The write direction: a contiguous tile lands back as a strided 2-D block, so a
    program can return its result in the layout the host handed it."""
    isa, dram, sram, gather, scatter = _isa()
    A = np.arange(R * C, dtype=np.float32).reshape(R, C)

    @isa.oracle(init={dram: A})
    def prog():
        gather(r=0, c=0, d=0)  # top-left block ...
        scatter(s=0, r=4, c=4)  # ... written over the bottom-right one
        isa.inspect(dram, label="dram")

    got = prog()["dram"]
    want = A.copy()
    want[4:8, 4:8] = A[0:4, 0:4]
    np.testing.assert_allclose(got, want)


def test_a_relayout_is_an_ordinary_edge_of_the_move_graph():
    """Once a location carries a *coordinate* rather than a flat offset, a relayout is
    routable like any other move — the multi-dimensional basis is filled from the
    operand's placement. Both directions register, so a value can be brought on chip
    and written back."""
    isa, _dram, sram, _g, _s = _isa()

    @isa.instruction(src=[sram], dst=sram)
    def copy(I):
        @I.access
        def _(s, d, n):
            return (contiguous(sram, s, n), contiguous(sram, d, n))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    assert _movement_catalog(isa) == ["gather", "scatter", "copy"]


# --- end to end: a tile-granular GEMM over row-major host memory -----------------

GM, GK, GN = 8, 12, 8  # 2x2 output blocks over 3 K-passes
A_ROW, B_ROW, C_ROW = 0, GM, GM + GK  # bands of the host array
A_TILE, B_TILE, C_TILE = 0, 16, 32  # staging slots on chip


def _gemm_isa():
    """The relayout ISA plus a tile-granular contraction: ``mm`` writes an output
    block, ``mac`` accumulates into it (that pair *is* the K-reduction)."""
    isa = ISA("relayout-gemm")
    dram = isa.hbm("dram", shape=(C_ROW + GM, GK), dtype=f32)
    sram = isa.global_("sram", shape=(256,), dtype=f32)

    def tile(addr):
        return view(sram, addr, (1, T, T))

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

    @isa.instruction(src=[sram, sram], dst=sram)
    def mm(I):
        @I.access
        def _(a, b, c):
            return (tile(a), tile(b), tile(c))

        @I.compute
        def _(a, b, c):
            return primitive.matmul(a, b)

    @isa.instruction(src=[sram, sram, sram], dst=sram)
    def mac(I):
        @I.access
        def _(a, b, c, d):
            return (tile(a), tile(b), tile(c), tile(d))

        @I.compute
        def _(a, b, c, d):
            return primitive.add(c, primitive.matmul(a, b))

    return isa, dram, sram, gather, scatter, mm, mac


def test_a_tile_granular_gemm_takes_row_major_operands():
    """End to end: the host writes plain 2-D arrays and the *program* converts. Without
    relayout the host had to marshal A and B into tile-major order first — which is the
    assumption "the backend only accepts already-tiled input" was hiding. Here the
    K-reduction is a run of gathers + ``mm``/``mac`` and nothing outside the program
    rearranges a single word."""
    isa, dram, sram, gather, scatter, mm, mac = _gemm_isa()
    rng = np.random.default_rng(1)
    A = rng.standard_normal((GM, GK)).astype(np.float32)
    B = rng.standard_normal((GK, GN)).astype(np.float32)

    host = np.zeros((C_ROW + GM, GK), np.float32)  # one plain row-major array
    host[A_ROW : A_ROW + GM, :GK] = A
    host[B_ROW : B_ROW + GK, :GN] = B

    tiles = 0

    @isa.oracle(init={dram: host})
    def prog():
        nonlocal tiles
        for mb in range(GM // T):
            for nb in range(GN // T):
                for kb in range(GK // T):
                    gather(r=A_ROW + mb * T, c=kb * T, d=A_TILE)
                    gather(r=B_ROW + kb * T, c=nb * T, d=B_TILE)
                    if kb == 0:
                        mm(a=A_TILE, b=B_TILE, c=C_TILE)
                    else:
                        mac(a=A_TILE, b=B_TILE, c=C_TILE, d=C_TILE)
                    tiles += 1
                scatter(s=C_TILE, r=C_ROW + mb * T, c=nb * T)
        isa.inspect(dram, label="dram")

    got = prog()["dram"][C_ROW : C_ROW + GM, :GN]
    assert tiles == 2 * 2 * 3  # 2x2 output blocks x 3 K-passes
    np.testing.assert_allclose(got, A @ B, rtol=1e-4, atol=1e-4)
