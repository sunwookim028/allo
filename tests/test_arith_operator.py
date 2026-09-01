# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from allo.compiler.errors import CompilationError
from allo.compiler.mlir_codegen import compile as compile_kernel
from allo.lang.core import f32, i32, i64, u1, u32
from allo.lang.kernel import KernelOptions, kernel
from allo.operators import arith as allo_arith


def _compile_ir(fn) -> str:
    return str(compile_kernel(fn))


def _assert_contains(ir: str, *patterns: str):
    for pattern in patterns:
        assert pattern in ir


def _assert_compile_error(fn, *patterns: str):
    with pytest.raises(CompilationError) as exc_info:
        _compile_ir(fn)
    message = exc_info.value.error_msg
    for pattern in patterns:
        assert pattern in message


def test_tensor_add_linalg():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: f32[4], y: f32[4]) -> f32[4]:
        return x + y

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.add")


def test_tensor_rank0_add_linalg():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: f32[()], y: f32[()]) -> f32[()]:
        return x + y

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.add", "tensor<f32>")


def test_tensor_add_scalar_broadcast():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: f32[4]) -> f32[4]:
        return x + 1.0

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.add", "arith.constant 1.000000e+00")


def test_memref_add_requires_acc():
    @kernel
    def top(x: f32[4], y: f32[4], out: f32[1]):
        z: f32[4] = x + y
        out[0] = z[0]

    _assert_compile_error(top, "requires acc for memref output")


def test_memref_add_acc():
    @kernel
    def top(x: f32[4], y: f32[4], out: f32[4]):
        allo_arith.add(x, y, acc=out)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.add")


def test_memref_div_positional_acc():
    @kernel
    def top(x: u32[4], y: u32[4], out: u32[4]):
        allo_arith.div(x, y, out)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.div_unsigned")


def test_int_div_mod_signedness_follows_dtype():
    @kernel
    def signed(A: i32[8], B: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] // B[i]
            out[i] = A[i] % B[i]

    @kernel
    def unsigned(A: u32[8], B: u32[8], out: u32[8]):
        for i in range(8):
            out[i] = A[i] // B[i]
            out[i] = A[i] % B[i]

    # Integer // truncates toward zero (divsi), matching / and %, for HLS-native
    # single-op codegen; unsigned // stays divui.
    _assert_contains(_compile_ir(signed), "arith.divsi", "arith.remsi")
    _assert_contains(_compile_ir(unsigned), "arith.divui", "arith.remui")


def test_int_rshift_signedness_follows_dtype():
    @kernel
    def signed(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] >> 2

    @kernel
    def unsigned(A: u32[8], out: u32[8]):
        for i in range(8):
            out[i] = A[i] >> 2

    _assert_contains(_compile_ir(signed), "arith.shrsi")
    _assert_contains(_compile_ir(unsigned), "arith.shrui")


def test_int_div_mod_constant_fold_matches_codegen():
    # Folding must match integer codegen, which truncates toward zero: /, // and %
    # all follow C semantics (divsi/remsi), unlike Python's floored //, %.
    @kernel
    def top(out: i32[4]):
        out[0] = -7 / 2  # divsi(-7, 2) == -3 (trunc), not floor -4
        out[1] = -7 // 2  # // also truncates: -3, not Python's floor -4
        out[2] = -7 % 2  # remsi(-7, 2) == -1, not Python's 1
        out[3] = 7 % -2  # remsi(7, -2) == 1, not Python's -1

    ir = _compile_ir(top)
    _assert_contains(ir, "%c-3_i32", "%c-1_i32", "%c1_i32")
    assert "%c-4_i32" not in ir  # -7 // 2 must not floor to -4


def test_int_div_constant_fold_is_exact():
    # Folding through a Python float would lose precision on large integers.
    @kernel
    def top(out: i64[1]):
        out[0] = 987654321987654321 / 7

    _assert_contains(_compile_ir(top), str(987654321987654321 // 7))


def test_tensor_lt_generic():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: f32[4], y: f32[4]) -> u1[4]:
        return x < y

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.generic", "arith.cmpf")


def test_tensor_lt_positional_acc():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: f32[4], y: f32[4], out: u1[4]) -> u1[4]:
        return allo_arith.lt(x, y, out, ordered=True)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.generic", "arith.cmpf")


def test_tensor_max_positional_acc():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: f32[4], y: f32[4], out: f32[4]) -> f32[4]:
        return allo_arith.max(x, y, out, propagate_nan=True)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.generic")


def test_scalar_add_acc_error():
    @kernel
    def top(x: f32, y: f32, out: f32[4]):
        allo_arith.add(x, y, acc=out)

    _assert_compile_error(top, "acc requires at least one shaped operand")
