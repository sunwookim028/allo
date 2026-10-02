# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Comparison semantics. MLIR integers are signless, so the predicate's
# signedness must come from the Allo type: an unsigned compare lowered as a
# signed `arith.cmpi` gives 200 > 100 == False for uint8.

import itertools
import pytest
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import (
    UInt,
    Fixed,
    UFixed,
    uint1,
    uint8,
    int8,
    int32,
    uint32,
    float32,
)

# Port width (numpy-native) used to carry a UInt(w) value in and out.
_PORT = {1: 8, 8: 8, 16: 16, 32: 32, 64: 64}
_NP = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
_OPS = ["==", "!=", "<", "<=", ">", ">="]


def _boundary_pairs(w):
    vals = sorted({0, 1, (1 << (w - 1)) - 1, 1 << (w - 1), (1 << w) - 1})
    pairs = list(itertools.product(vals, vals))
    a = np.array([p[0] for p in pairs], dtype=_NP[_PORT[w]])
    b = np.array([p[1] for p in pairs], dtype=_NP[_PORT[w]])
    return a, b


def _golden(a, b):
    a = [int(x) for x in a]
    b = [int(x) for x in b]
    return np.array(
        [[x == y, x != y, x < y, x <= y, x > y, x >= y] for x, y in zip(a, b)],
        dtype=np.uint8,
    )


@pytest.mark.parametrize("w", [1, 8, 16, 32, 64])
def test_uint_compare_llvm(w):
    a, b = _boundary_pairs(w)
    n = len(a)
    Tp = UInt(_PORT[w])
    Tv = UInt(w)

    def kernel(A: Tp[n], B: Tp[n]) -> uint8[n, 6]:
        C: uint8[n, 6] = 0
        for i in range(n):
            x: Tv = A[i]
            y: Tv = B[i]
            C[i, 0] = 1 if x == y else 0
            C[i, 1] = 1 if x != y else 0
            C[i, 2] = 1 if x < y else 0
            C[i, 3] = 1 if x <= y else 0
            C[i, 4] = 1 if x > y else 0
            C[i, 5] = 1 if x >= y else 0
        return C

    s = allo.customize(kernel)
    if w > 1:
        mlir = str(s.module)
        for pred in ("sgt", "sge", "slt", "sle"):
            assert f"arith.cmpi {pred}" not in mlir, mlir
    mod = s.build()
    np.testing.assert_array_equal(mod(a, b), _golden(a, b))


@pytest.mark.parametrize("w", [1, 8, 16, 32])
def test_uint_compare_simulator(w):
    a, b = _boundary_pairs(w)
    a = a.astype(np.uint32)
    b = b.astype(np.uint32)
    n = len(a)
    Tv = UInt(w)

    @df.region()
    def top(A: uint32[n], B: uint32[n], C: int32[n, 6]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def cmp(lA: uint32[n], lB: uint32[n], lC: int32[n, 6]):
            for i in range(n):
                x: Tv = lA[i]
                y: Tv = lB[i]
                lC[i, 0] = 1 if x == y else 0
                lC[i, 1] = 1 if x != y else 0
                lC[i, 2] = 1 if x < y else 0
                lC[i, 3] = 1 if x <= y else 0
                lC[i, 4] = 1 if x > y else 0
                lC[i, 5] = 1 if x >= y else 0

    C = np.zeros((n, 6), dtype=np.int32)
    sim_mod = df.build(top, target="simulator")
    sim_mod(a, b, C)
    np.testing.assert_array_equal(C, _golden(a, b).astype(np.int32))


def test_int_compare_signed():
    # Signed compares must stay signed.
    def kernel(A: int8[3], B: int8[3]) -> uint8[3]:
        C: uint8[3] = 0
        for i in range(3):
            C[i] = 1 if A[i] < B[i] else 0
        return C

    s = allo.customize(kernel)
    assert "arith.cmpi slt" in str(s.module)
    mod = s.build()
    A = np.array([-128, 127, -1], dtype=np.int8)
    B = np.array([127, -128, 0], dtype=np.int8)
    np.testing.assert_array_equal(mod(A, B), [1, 0, 1])


def test_mixed_sign_compare():
    # Int(m) vs UInt(n) is compared in Int(max(m, n + 1)) (typing_rule.cmp_rule):
    # by value, as in mathematics. NOTE: this differs from C for equal widths
    # (C converts int32 to unsigned, so -1 < 1u is false); Allo says true.
    def kernel(A: int32[4], B: uint32[4]) -> uint8[4]:
        C: uint8[4] = 0
        for i in range(4):
            C[i] = 1 if A[i] < B[i] else 0
        return C

    s = allo.customize(kernel)
    assert "i33" in str(s.module)
    mod = s.build()
    A = np.array([-1, 5, -2147483648, 2147483647], dtype=np.int32)
    B = np.array([1, 4294967295, 0, 2147483648], dtype=np.uint32)
    np.testing.assert_array_equal(mod(A, B), [1, 1, 1, 1])


def test_fixed_compare_signed():
    # Signed Fixed compares used to get unsigned predicates (`ult`).
    def kernel(A: int8[3], B: int8[3]) -> uint8[3]:
        C: uint8[3] = 0
        for i in range(3):
            a: Fixed(8, 2) = A[i]
            b: Fixed(8, 2) = B[i]
            C[i] = 1 if a < b else 0
        return C

    s = allo.customize(kernel)
    assert "cmp_fixed slt" in str(s.module)
    mod = s.build()
    A = np.array([-2, 3, -30], dtype=np.int8)
    B = np.array([3, -2, -29], dtype=np.int8)
    np.testing.assert_array_equal(mod(A, B), [1, 0, 1])


def test_ufixed_compare_unsigned():
    def kernel(A: uint8[2], B: uint8[2]) -> uint8[2]:
        C: uint8[2] = 0
        for i in range(2):
            a: UFixed(8, 2) = A[i]
            b: UFixed(8, 2) = B[i]
            C[i] = 1 if a > b else 0
        return C

    s = allo.customize(kernel)
    assert "cmp_fixed ugt" in str(s.module)
    mod = s.build()
    # 40.0 is 0xA0 in UFixed(8, 2): negative if read as signed
    A = np.array([40, 10], dtype=np.uint8)
    B = np.array([10, 40], dtype=np.uint8)
    np.testing.assert_array_equal(mod(A, B), [1, 0])


def test_float_compare():
    def kernel(A: float32[2], B: float32[2]) -> uint8[2]:
        C: uint8[2] = 0
        for i in range(2):
            C[i] = 1 if A[i] >= B[i] else 0
        return C

    mod = allo.customize(kernel).build()
    A = np.array([1.5, -2.0], dtype=np.float32)
    B = np.array([1.5, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(mod(A, B), [1, 0])


def test_compare_result_is_uint1():
    # B2: a comparison's type is uint1, not its operand type.
    def kernel(a: uint8, b: uint8) -> uint1:
        r: uint1 = a == b
        return r

    mod = allo.customize(kernel).build()
    assert mod(3, 3) == 1
    assert mod(3, 4) == 0

    def kernel2(a: uint8, b: uint8) -> uint1:
        return a >= b

    mod2 = allo.customize(kernel2).build()
    assert mod2(200, 100) == 1
    assert mod2(100, 200) == 0


def test_compare_result_widens_to_one():
    # True widens to 1 (zero-extended), never -1.
    def kernel(A: int32[2], B: int32[2]) -> int32[2]:
        C: int32[2] = 0
        for i in range(2):
            r: int32 = A[i] < B[i]
            C[i] = r
        return C

    mod = allo.customize(kernel).build()
    A = np.array([1, 5], dtype=np.int32)
    B = np.array([2, 3], dtype=np.int32)
    np.testing.assert_array_equal(mod(A, B), [1, 0])


def test_uint_compare_hls_unchanged():
    # The HLS C++ emitter already compared with C types; check it still does.
    def kernel(A: uint8[2], B: uint8[2]) -> uint8[2]:
        C: uint8[2] = 0
        for i in range(2):
            C[i] = 1 if A[i] > B[i] else 0
        return C

    code = str(allo.customize(kernel).build(target="vhls"))
    assert "uint8_t" in code and ">" in code


if __name__ == "__main__":
    pytest.main([__file__])
