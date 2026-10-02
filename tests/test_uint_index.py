# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An unsigned array index is zero-extended (B4).

``build_cast_op`` mapped ``(UInt, Index)`` to ``arith.index_cast``, which
sign-extends: a ``uint8`` index of 200 read ``mem[-56]`` on the LLVM backend
and in the dataflow simulator, silently. The HLS emitters were right by
accident (they index with the C type, ``uint8_t``). See
``dev/records/limitations/uint_index_2026-10-02.rst``.
"""
import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import UInt, Int, int8, int32, uint8, uint16, uint32, uint64, index


def _lut_kernel(IT, N):
    """q[t] = mem[idx[t]] with mem[i] = i, for a table of N entries.
    (``IT`` appears only in the signature, so the caller holds it as a local
    of the same name: allo resolves annotation names up the call stack.)"""

    def k(idx: IT[4], q: uint32[4]):
        mem: uint32[N] = 0
        for i in range(N):
            mem[i] = i
        for t in range(4):
            q[t] = mem[idx[t]]

    return k


@pytest.mark.parametrize(
    "IT, N, idx",
    [
        (uint8, 256, [1, 200, 255, 128]),  # every index >= 128 was negative
        (UInt(5), 32, [1, 20, 31, 16]),  # the MiniTPU regfile's address width
        (UInt(3), 8, [0, 4, 7, 5]),
        (uint16, 65536, [0, 40000, 65535, 32768]),
    ],
)
def test_uint_index_reads_the_right_element(IT, N, idx):
    s = allo.customize(_lut_kernel(IT, N))
    ir = str(s.module)
    assert "index_castui" in ir, ir
    q = np.zeros(4, np.uint32)
    s.build()(np.array(idx, np.uint64), q)
    assert q.tolist() == idx


def test_signed_index_stays_signed():
    """An ``int8`` index keeps ``arith.index_cast``: -1 must stay -1 in
    index arithmetic (``mem[i + off]`` with a negative offset)."""

    def k(off: int8[1], q: uint32[1]):
        mem: uint32[8] = 0
        for i in range(8):
            mem[i] = i + 10
        for t in range(1):
            q[t] = mem[5 + off[t]]

    s = allo.customize(k)
    ir = str(s.module)
    assert "arith.index_cast " in ir and "index_castui" not in ir, ir
    q = np.zeros(1, np.uint32)
    s.build()(np.array([-3], np.int8), q)
    assert q.tolist() == [12]


def test_uint_loop_bound():
    """A loop bound from a ``uint8`` at or above 128 runs that many times."""

    def k(n: uint8[1], q: uint32[1]):
        acc: uint32 = 0
        for i in range(n[0]):
            acc += 1
        q[0] = acc

    q = np.zeros(1, np.uint32)
    allo.customize(k).build()(np.array([200], np.uint8), q)
    assert q.tolist() == [200]


def test_uint_index_in_simulator():
    """The dataflow simulator lowers the same IR through LLVM."""
    N = 256

    @df.region()
    def top(A: uint8[4], Q: uint32[4]):
        @df.kernel(mapping=[1], args=[A, Q])
        def k(a: uint8[4], q: uint32[4]):
            mem: uint32[N] = 0
            for i in range(N):
                mem[i] = i
            for t in range(4):
                q[t] = mem[a[t]]

    q = np.zeros(4, np.uint32)
    df.build(top, target="simulator")(np.array([1, 200, 255, 128], np.uint8), q)
    assert q.tolist() == [1, 200, 255, 128]


def test_hls_emitters_index_with_unsigned_c_type():
    """The HLS C++ is unchanged by the op switch: ``index_castui`` emits like
    ``index_cast`` (``int v = x;`` from a ``uint8_t`` operand)."""
    IT, N = uint8, 256  # the factory's annotations resolve in this frame
    s = allo.customize(_lut_kernel(IT, N))
    code = str(s.build(target="vhls"))
    assert "uint8_t" in code
    # the cast target is `int`, read from an unsigned C variable
    assert "index_cast" not in code


def test_float_to_index_is_unsigned():
    """float -> index goes through fptoui; the index_cast after it must not
    re-sign the ui32 (a float index >= 2^31 was negative)."""

    def k(f: allo.ir.types.float32[1], q: uint32[1]):
        mem: uint32[4] = 0
        for i in range(4):
            mem[i] = i + 10
        for t in range(1):
            q[t] = mem[f[t]]

    s = allo.customize(k)
    ir = str(s.module)
    assert "fptoui" in ir and "index_castui" in ir, ir
    q = np.zeros(1, np.uint32)
    s.build()(np.array([2.0], np.float32), q)
    assert q.tolist() == [12]


if __name__ == "__main__":
    pytest.main([__file__])
