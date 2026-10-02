# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``x: T = s.get()`` converts the element to ``T`` (B5).

``build_assign_stmt`` skipped the cast for every get op, so a ``UInt(5)``
stream read into an ``int32`` stored an ``i5`` into an ``i32`` memref and
the verifier refused the module. See
``dev/records/limitations/uint_index_2026-10-02.rst``.
"""
import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import UInt, Int, int32, uint8, Stream, Channel, valid_ready

N = 8


def _widen_region(ET, TT, n=N):
    @df.region()
    def top(A: ET[n], B: TT[n]):
        s: Stream[ET, 2]

        @df.kernel(mapping=[1], args=[A])
        def p(a: ET[n]):
            for t in range(n):
                s.put(a[t])

        @df.kernel(mapping=[1], args=[B])
        def c(b: TT[n]):
            for t in range(n):
                x: TT = s.get()
                b[t] = x

    return top


def test_uint5_stream_get_into_int32():
    top = _widen_region(UInt(5), int32)
    a = np.array([0, 1, 15, 16, 20, 30, 31, 7], np.uint8)
    b = np.zeros(N, np.int32)
    df.build(top, target="simulator")(a, b)
    assert b.tolist() == a.tolist()  # zero-extended, 31 stays 31


def test_int5_stream_get_into_int32_sign_extends():
    top = _widen_region(Int(5), int32)
    a = np.array([0, 1, -16, 15, -1, 7, -8, 3], np.int8)
    b = np.zeros(N, np.int32)
    df.build(top, target="simulator")(a, b)
    assert b.tolist() == a.tolist()


def test_int32_stream_get_into_uint8_truncates():
    top = _widen_region(int32, uint8)
    a = np.array([0, 255, 256, 300, -1, 7, 1000, 3], np.int32)
    b = np.zeros(N, np.uint8)
    df.build(top, target="simulator")(a, b)
    assert b.tolist() == (a.astype(np.int64) % 256).tolist()


def test_get_into_same_type_is_unchanged():
    top = _widen_region(int32, int32)
    a = np.arange(N, dtype=np.int32)
    b = np.zeros(N, np.int32)
    df.build(top, target="simulator")(a, b)
    assert b.tolist() == a.tolist()


def test_channel_get_casts_too():
    @df.region()
    def top(A: UInt(5)[N], B: int32[N]):
        ch: Channel[UInt(5), valid_ready]

        @df.kernel(mapping=[1], args=[A])
        def p(a: UInt(5)[N]):
            for t in range(N):
                ch.put(a[t])

        @df.kernel(mapping=[1], args=[B])
        def c(b: int32[N]):
            for t in range(N):
                x: int32 = ch.get()
                b[t] = x

    ir = str(df.customize(top).module)
    assert "affine.store" in ir
    assert "arith.extui" in ir, ir


if __name__ == "__main__":
    pytest.main([__file__])
