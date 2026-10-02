# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arrays of integers wider than 64 bits cross the LLVM/simulator boundary
whole, or are refused (B6).

A ``uint64`` array handed to a ``UInt(256)`` argument used to be passed as
is, with a warning: the kernel read 32 bytes per element, 4x past the end of
the buffer (heap corruption, a hang at the MiniTPU regfile). Now a > 64-bit
argument is packed element by element from Python ints (any integer dtype,
or ``dtype=object``), range-checked, and the results come back as Python
ints -- into an object array whole, into a narrower array only if they fit.
See ``dev/records/limitations/uint_index_2026-10-02.rst``.
"""
import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import UInt, Int, uint64, int32
from allo.utils import pack_wide_int_array, unpack_wide_int_array

N = 4


def _copy(T):
    """(``T`` appears only in annotations, so each caller holds a local ``T``:
    allo resolves annotation names up the call stack.)"""

    def k(a: T[N], b: T[N]):
        for t in range(N):
            b[t] = a[t]

    return k


def _shift_region(T):
    @df.region()
    def top(A: T[N], B: T[N]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: T[N], b: T[N]):
            for t in range(N):
                b[t] = a[t] + 1

    return top


def _obj(vals):
    out = np.empty(len(vals), dtype=object)
    out[:] = vals
    return out


WIDE_U = [0, 1, (1 << 64) + 5, (1 << 256) - 1]
WIDE_S = [0, -1, -(1 << 127), (1 << 127) - 1]


def test_pack_unpack_roundtrip():
    for bw, signed, vals in [(256, False, WIDE_U), (128, True, WIDE_S), (200, False, [1 << 199])]:
        packed = pack_wide_int_array(_obj(vals), bw, signed)
        assert packed.dtype.itemsize == max(bw, 8) // 8 if bw & (bw - 1) == 0 else True
        assert unpack_wide_int_array(packed, bw, signed).tolist() == vals


def test_pack_refuses_out_of_range_and_non_integers():
    with pytest.raises(ValueError):
        pack_wide_int_array(_obj([1 << 256]), 256, False)
    with pytest.raises(ValueError):
        pack_wide_int_array(_obj([-1]), 256, False)
    with pytest.raises(ValueError):
        pack_wide_int_array(_obj([1 << 127]), 128, True)
    with pytest.raises(TypeError):
        pack_wide_int_array(np.zeros(2, np.float64), 256, False)


def test_uint256_object_array_llvm():
    T = UInt(256)
    a = _obj(WIDE_U)
    b = _obj([0] * N)
    allo.customize(_copy(T)).build()(a, b)
    assert b.tolist() == WIDE_U


def test_int128_negative_llvm():
    T = Int(128)
    a = _obj(WIDE_S)
    b = _obj([0] * N)
    allo.customize(_copy(T)).build()(a, b)
    assert b.tolist() == WIDE_S


def test_uint256_simulator_with_values_past_64_bits():
    T = UInt(256)
    a = _obj(WIDE_U[:3] + [(1 << 255)])
    b = _obj([0] * N)
    df.build(_shift_region(T), target="simulator")(a, b)
    assert b.tolist() == [v + 1 for v in a.tolist()]


def test_uint64_input_is_widened_not_read_past_its_end():
    """The MiniTPU case: a uint64 array for a UInt(256) argument. The input
    is widened by value; the output comes back into the uint64 array because
    every result fits."""
    T = UInt(256)
    a = np.array([0, 1, 2, (1 << 64) - 2], np.uint64)
    b = np.zeros(N, np.uint64)
    df.build(_shift_region(T), target="simulator")(a, b)
    assert b.tolist() == [1, 2, 3, (1 << 64) - 1]


def test_narrow_output_array_refuses_a_value_it_cannot_hold():
    """A result past 2^64 into a uint64 array is an error, never a silent
    truncation (and never a write past the buffer)."""
    T = UInt(256)
    a = np.array([0, 1, 2, (1 << 64) - 1], np.uint64)
    b = np.zeros(N, np.uint64)
    with pytest.raises(OverflowError):
        df.build(_shift_region(T), target="simulator")(a, b)


def test_float_array_for_wide_int_is_refused():
    T = UInt(256)
    with pytest.raises(TypeError):
        allo.customize(_copy(T)).build()(
            np.zeros(N, np.float64), _obj([0] * N)
        )


def test_narrow_types_unchanged():
    """The <= 64-bit path (anywidth packing) is untouched."""
    T = UInt(9)
    a = np.array([0, 1, 100, 300], np.int64)
    b = np.zeros(N, np.int64)
    allo.customize(_copy(T)).build()(a, b)
    assert b.tolist() == [0, 1, 100, 300]


if __name__ == "__main__":
    pytest.main([__file__])
