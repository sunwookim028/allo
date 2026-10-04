# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bit slice whose result keeps its word's width lowers (E5, E4).

``lower_bit_ops`` ended every ``allo.get_slice`` with ``arith.trunci`` to the
result type. The builder types a slice whose bounds only fold after
``meta_for`` expansion (and a full-width one) with the word's own width, so
``lane: int32 = w[32*c : 32*(c+1)]`` on a ``UInt(32)`` became ``trunci i32 ->
i32`` and the simulator refused the module. The MiniTPU int8 matrix engine at
DIM=4 (a ``UInt(32)`` packed row of four int8 lanes) is the same slice. See
``dev/records/limitations/u3_fixes_2026-10-04.rst``.
"""
import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import int8, int32, UInt, Stream

N = 4
IN_BITS = 8
WORDS = (np.arange(N, dtype=np.uint64) * 0x91A2B3C5 + 0x80000001).astype(np.uint32)


def _full_width(form, LT):
    @df.region()
    def top(A: UInt(32)[N], B: LT[N]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: UInt(32)[N], b: LT[N]):
            for i in range(N):
                w: UInt(32) = a[i]
                with allo.meta_if(form == 0):
                    lane: LT = w[0:32]
                    b[i] = lane
                with allo.meta_else():
                    with allo.meta_for(1) as c:
                        lane2: LT = w[32 * c : 32 * (c + 1)]
                        b[i] = lane2

    return top


@pytest.mark.parametrize("form", [0, 1], ids=["constant", "meta_for"])
@pytest.mark.parametrize(
    "LT,view", [(int32, np.int32), (UInt(32), np.uint32)], ids=["int32", "uint32"]
)
def test_full_width_slice_simulator(form, LT, view):
    mod = df.build(_full_width(form, LT), target="simulator")
    out = np.zeros(N, dtype=view)
    mod(WORDS, out)
    np.testing.assert_array_equal(out, WORDS.view(view))


def test_packed_int8_lanes_simulator():
    """Four int8 lanes of a UInt(32) row through a stream: the E4 shape. The
    bounds name a global, so the slice is typed i32 and truncated after."""
    D = 4

    @df.region()
    def top(A: UInt(32)[N], W: int8[D], O: int32[N]):
        row: Stream[UInt(32), 2]

        @df.kernel(mapping=[1], args=[A])
        def feed(a: UInt(32)[N]):
            for t in range(N):
                row.put(a[t])

        @df.kernel(mapping=[1], args=[W, O])
        def dot(w: int8[D], o: int32[N]):
            for t in range(N):
                packed: UInt(32) = row.get()
                acc: int32 = 0
                with allo.meta_for(D) as r:
                    lane: int8 = packed[IN_BITS * r : IN_BITS * (r + 1)]
                    acc += lane * w[r]
                o[t] = acc

    w = np.array([1, -2, 3, -4], dtype=np.int8)
    out = np.zeros(N, dtype=np.int32)
    df.build(top, target="simulator")(WORDS, w, out)
    lanes = WORDS.view(np.int8).reshape(N, D).astype(np.int32)
    np.testing.assert_array_equal(out, lanes @ w.astype(np.int32))


def test_full_width_slice_llvm():
    def k(A: UInt(32)[N], B: int32[N]):
        for i in range(N):
            w: UInt(32) = A[i]
            with allo.meta_for(1) as c:
                lane: int32 = w[32 * c : 32 * (c + 1)]
                B[i] = lane

    out = np.zeros(N, dtype=np.int32)
    allo.customize(k).build()(WORDS, out)
    np.testing.assert_array_equal(out, WORDS.view(np.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
