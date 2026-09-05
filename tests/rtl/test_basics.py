# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanity checks for the RTL backend's end-to-end path"""

import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import i32

sys.path.insert(0, os.path.dirname(__file__))
from _common import _to_rtl  # noqa: E402

needs_verilator = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF
B16 = (np.arange(16, dtype=np.int32) * 5 + 3) & 0xFF


# Elementwise kernels over the basic address shapes: direct index, neighbor
# offset, constant stride, a scalar operand, and a func-scope literal.
@needs_verilator
def test_elementwise_and_addressing():
    @kernel
    def vand(A: i32[16], B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] & B[i]

    out = np.zeros(16, np.int32)
    _to_rtl(vand).cosim(A16, B16, out)
    assert np.array_equal(out, A16 & B16)

    @kernel
    def shift(A: i32[16], out: i32[16]):
        for i in range(15):
            out[i] = A[i] & A[i + 1]

    out = np.zeros(16, np.int32)
    _to_rtl(shift).cosim(A16, out)
    assert np.array_equal(out[:15], A16[:15] & A16[1:16])

    # A[2*i]: the address linearizes to iv*2 -- a multiply by the constant stride.
    @kernel
    def stride2(A: i32[16], out: i32[8]):
        for i in range(8):
            out[i] = A[2 * i] & A[2 * i]

    out = np.zeros(8, np.int32)
    _to_rtl(stride2).cosim(A16, out)
    assert np.array_equal(out, A16[0:16:2])

    @kernel
    def scaled(A: i32[16], out: i32[16], s: i32):
        for i in range(16):
            out[i] = A[i] & s

    out = np.zeros(16, np.int32)
    _to_rtl(scaled).cosim(A16, out, np.int32(0x0F))
    assert np.array_equal(out, A16 & 0x0F)

    # A func-scope literal tied into the compute.
    @kernel
    def constd(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] & 5

    out = np.zeros(16, np.int32)
    _to_rtl(constd).cosim(A16, out)
    assert np.array_equal(out, A16 & 5)


# csim delegates to the CPU/LLVM-JIT path; it is the golden cosim compares to.
def test_csim_golden_matches_reference():
    @kernel
    def vand(A: i32[16], B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] & B[i]

    golden = np.zeros(16, np.int32)
    _to_rtl(vand).csim(A16, B16, golden)
    assert np.array_equal(golden, A16 & B16)
