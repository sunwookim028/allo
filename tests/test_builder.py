# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

import numpy as np
import pytest

import allo
from allo.compiler.errors import CompilationError
from allo.compiler.mlir_codegen import compile as compile_kernel
from allo.lang.core import (
    Template,
    bool as allo_bool,
    constexpr,
    f32,
    i32,
    index,
    u1,
    u8,
    u32,
    Stream,
)
from allo.lang.kernel import KernelOptions, consteval, kernel
from allo.operators.arith import bitcast as allo_bitcast, max as allo_max

_GLOBAL_SHAPE_M = 2
_GLOBAL_SHAPE_N = 3
_GLOBAL_INT_CONST = 3
_GLOBAL_FLOAT_CONST = 1.5

_GLOBAL_NP_INT = np.array([[1, 2], [3, 4]], dtype=np.int32)
_GLOBAL_NP_FLOAT = np.array([1.5, 2.5], dtype=np.float32)
_GLOBAL_NP_BOOL = np.array([True, False])


def _compile_ir(fn, *, options=None) -> str:
    return str(
        compile_kernel(fn, options=options).operation.get_asm(
            use_name_loc_as_prefix=True
        )
    )


def _assert_contains(ir: str, *patterns: str):
    for pattern in patterns:
        assert pattern in ir


def _assert_compile_error(fn, *patterns: str):
    with pytest.raises(CompilationError) as exc_info:
        _compile_ir(fn)
    message = exc_info.value.error_msg
    for pattern in patterns:
        assert pattern in message


def test_error_diagnostic_source():
    src = "def broken(x):\n    return x + y\n"
    module = ast.parse(src)
    fn = module.body[0]
    assert isinstance(fn, ast.FunctionDef)
    ret = fn.body[0]
    assert isinstance(ret, ast.Return)
    expr = ret.value
    assert isinstance(expr, ast.BinOp)

    err = CompilationError(
        src,
        "Name 'y' is not defined",
        expr.right,
        file_name="broken.py",
        begin_line=10,
    )
    message = err.render(color=False)

    assert "broken.py:11:16: error: Name 'y' is not defined" in message
    assert "11 |     return x + y" in message
    assert "^" in message
    assert "\x1b[" not in message
    assert str(err).startswith("\n")


def test_scalar_int_add():
    @kernel
    def top(x: i32, y: i32, out: i32[1]):
        out[0] = x + y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.extsi",
        "to i33",
        "arith.addi",
        "i33 to i32",
    )


def test_hls_nary_add_sub():
    @kernel
    def top(x: i32, y: i32, z: i32, out: i32[1]):
        out[0] = x + y - z

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant 0 : i34",
        "to i34",
        "arith.subi",
        "arith.addi",
        "i34 to i32",
    )


def test_hls_nary_mul():
    @kernel
    def top(x: i32, y: i32, z: i32, out: i32[1]):
        out[0] = x * y * z

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "to i96",
        "arith.muli",
        "i96 to i32",
    )


def test_mixed_int_float_add():
    @kernel
    def top(x: i32, y: f32, out: f32[1]):
        out[0] = x + y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.sitofp",
        "i32 to f32",
        "arith.addf",
    )


def test_float_add():
    @kernel
    def top(x: f32, y: f32, out: f32[1]):
        out[0] = x + y

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.addf")


def test_out_of_int64_const_reports_error():
    # int(-1e30) exceeds C int64_t, which IntegerAttr.get's nanobind binding
    # rejects with an opaque TypeError; a readable compile error is raised first.
    @kernel
    def top(out: i32[1]):
        out[0] = -1e30

    _assert_compile_error(top, "is out of range")


def test_large_unsigned_const_wraps():
    @kernel
    def top(out: u8[1]):
        out[0] = 200

    ir = _compile_ir(top)
    # 200 fits int64, so it materializes as its two's-complement i8 value -56.
    _assert_contains(ir, "arith.constant", "-56 : i8")


def test_scalar_bitcast_float_to_int():
    @kernel
    def top(x: f32, out: i32[1]):
        out[0] = allo_bitcast(x, i32)

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.bitcast", "f32 to i32")


def test_scalar_bitcast_int_to_float():
    @kernel
    def top(x: u32, out: f32[1]):
        out[0] = allo_bitcast(x, f32)

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.bitcast", "i32 to f32")


def test_bitcast_roundtrip_folds_to_identity():
    @kernel
    def top(x: f32, out: f32[1]):
        bits: i32 = allo_bitcast(x, i32)
        out[0] = allo_bitcast(bits, f32)

    # bitcast(bitcast(x)) is the identity, so canonicalization removes both.
    ir = _compile_ir(top)
    assert "arith.bitcast" not in ir


def test_bitcast_width_mismatch_error():
    @kernel
    def top(x: f32, out: u8[1]):
        out[0] = allo_bitcast(x, u8)

    _assert_compile_error(top, "Cannot bitcast", "bit widths 32 and 8 differ")


def test_unary_neg():
    @kernel
    def top(x: i32, out: i32[1]):
        out[0] = -x

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant 0 : i33",
        "arith.extsi",
        "arith.subi",
        "i33 to i32",
    )


def test_bitwise_xor():
    @kernel
    def top(x: u32, y: u32, out: u32[1]):
        out[0] = x ^ y

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.xori")


def test_shift_by_range_index():
    @kernel
    def top(x: i32, out: i32[4]):
        for i in range(4):
            out[i] = x >> (i * 2)

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.index_cast",
        "arith.shrsi",
    )


def test_bit_get_slice():
    @kernel
    def top(x: u32, out: u32[1]):
        out[0] = x[4:8]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.bit.get_slice",
        "[%c4 : %c8]",
        "i4 from i32",
    )


def test_bit_get_single_bit():
    @kernel
    def top(x: u32, out: u32[1]):
        out[0] = x[3]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.bit.get_slice",
        "i1 from i32",
    )


def test_bit_get_slice_dynamic_offset_static_width():
    # A dynamic offset with a statically-constant width: the `i` terms cancel in
    # `(i + 2) - i`, so the result is exactly 2 bits (`i2`), not the full source.
    @kernel
    def top(x: u32, out: u32[2]):
        for i in range(2):
            out[i] = x[i : i + 2]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.addi",
        "allo.bit.get_slice",
        "i2 from i32",
    )


def test_bit_get_slice_constexpr_width():
    @kernel
    def top(x: u32, out: u32[2]):
        W: constexpr = 3
        for i in range(2):
            out[i] = x[i : i + W]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.addi",
        "allo.bit.get_slice",
        "i3 from i32",
    )


def test_bit_get_slice_dynamic_width_error():
    @kernel
    def top(lo: i32, hi: i32, x: u32, out: u32[1]):
        out[0] = x[lo:hi]

    _assert_compile_error(
        top,
        "Bit slice width 'hi - lo' must be a compile-time constant",
    )


def test_bit_set_slice():
    @kernel
    def top(x: u32, out: u32[1]):
        y: u32 = x
        y[0:4] = 5
        out[0] = y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant 5 : i4",
        "allo.bit.set_slice",
        "i4 into i32",
    )


def test_bit_set_slice_memref_writeback():
    @kernel
    def top(a: u8[4], b: u32[4]):
        for i in range(4):
            b[i][0:2] = a[i]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "affine.load",
        "allo.bit.set_slice",
        "i2 into i32",
        "affine.store",
    )


def test_bit_slice_requires_integer():
    @kernel
    def top(x: f32, out: f32[1]):
        out[0] = x[0:4]

    _assert_compile_error(
        top,
        "Bit slicing is only supported on signless integer scalars.",
    )


def test_comparison_lt():
    @kernel
    def top(x: i32, y: i32, out: u1[1]):
        out[0] = x < y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.cmpi slt",
        "memref<1xi1>",
    )


def test_bool_and_not():
    @kernel
    def top(x: allo_bool, y: allo_bool, out: u1[1]):
        out[0] = x and not y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant true",
        "arith.xori",
        "arith.andi",
        "memref<1xi1>",
    )


def test_if_statement_phi():
    @kernel
    def top(cond: allo_bool, x: i32, y: i32, out: i32[1]):
        v = x
        if cond:
            v = y
        else:
            v = x + y
        out[0] = v

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "scf.if",
        "-> (i32)",
        "scf.yield",
    )


def test_int_condition_is_nonzero_test_not_low_bit():
    # A non-bool integer used in a boolean context (the test of an
    # `if` / `while` / ternary) must lower to a `!= 0` truthiness test, NOT a
    # `trunci` that keeps only the low bit. Otherwise `if x & (1 << k):` is
    # silently wrong for every k > 0 (e.g. the butterfly routing in
    # `reverse_bits`, which only ever reversed bit 0).
    @kernel
    def top(data: i32, out: i32[1]):
        out[0] = 0
        if data & 2:  # truth lives in bit 1, not bit 0
            out[0] = 7

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.cmpi ne")
    assert "arith.trunci" not in ir  # a low-bit truncation would be the bug

    # Runtime truthiness must hold across if / while / ternary.
    @kernel
    def cond_if(data: i32, out: i32[1]):
        out[0] = 0
        if data & 2:
            out[0] = 7

    @kernel
    def cond_while(n: i32, out: i32[1]):
        acc: i32 = 0
        x: i32 = n
        while x & 4:  # bit 2 set -> enter; the bug would skip every iteration
            acc = acc + 1
            x = x - 4
        out[0] = acc

    @kernel
    def cond_tern(data: i32, a: i32, b: i32) -> i32:
        return a if (data & 2) else b

    out = np.zeros(1, dtype=np.int32)
    cond_if(np.int32(2), out)  # bit 1 set -> taken
    assert out[0] == 7
    cond_if(np.int32(4), out)  # bit 1 clear -> not taken
    assert out[0] == 0

    cond_while(np.int32(12), out)  # 0b1100: one pass (12&4 set, then 8&4 clear)
    assert out[0] == 1

    assert int(cond_tern(np.int32(2), np.int32(5), np.int32(9))) == 5
    assert int(cond_tern(np.int32(1), np.int32(5), np.int32(9))) == 9


def test_if_branch_local_buffers():
    @kernel
    def top(out: i32[8]):
        for r in range(2):
            r_i32: i32 = r
            if r_i32 == 0:
                then_buf: i32[4]
                for j in range(4):
                    then_buf[j] = j
                    out[j] = then_buf[j]
            else:
                else_buf: i32[4]
                for j in range(4):
                    else_buf[j] = j + 1
                    out[j + 4] = else_buf[j]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "scf.if",
        "affine.for",
        "memref.alloc",
    )


def test_if_branch_local_loop_carried_value():
    @kernel
    def top(cond: allo_bool, x: i32, out: i32[1]):
        if cond:
            out[0] = x
        else:
            c: i32 = 0
            for _ in range(2):
                c += x
            out[0] = c

    ir = _compile_ir(top)
    _assert_contains(ir, "scf.if", "affine.for")


def test_if_constexpr_branch():
    dtype = f32

    @kernel
    def top(x: i32[1]):
        if False:
            x[0] = 1
        elif dtype == i32:
            x[0] = 2
        else:
            x[0] = 3

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.constant 3")
    assert "scf.if" not in ir


def test_if_affine_iv_condition():
    @kernel
    def top(out: i32[16]):
        for i in range(16):
            if i < 8:
                out[i] = 1
            else:
                out[i] = 2

    ir = _compile_ir(top)
    _assert_contains(ir, "affine.if", "affine_set", "affine.for")
    assert "scf.if" not in ir


def test_if_affine_mod_condition():
    @kernel
    def top(out: i32[16]):
        for i in range(16):
            if i % 2 == 0:
                out[i] = 1

    ir = _compile_ir(top)
    _assert_contains(ir, "affine.if", "mod 2")
    assert "scf.if" not in ir


def test_if_affine_conjunction_with_symbol():
    @kernel
    def top(n: i32, out: i32[16]):
        for i in range(16):
            if i >= 2 and i < n:
                out[i] = 1

    ir = _compile_ir(top)
    # two constraints from the `and`, plus a symbol operand from `n`.
    _assert_contains(ir, "affine.if", "affine_set")
    assert "scf.if" not in ir


def test_if_affine_phi_result():
    @kernel
    def top(out: i32[16]):
        for i in range(16):
            v: i32 = 0
            if i < 4:
                v = 5
            else:
                v = 7
            out[i] = v

    ir = _compile_ir(top)
    _assert_contains(ir, "affine.if", "-> i32", "affine.yield")
    assert "scf.if" not in ir


def test_if_non_affine_falls_back_to_scf():
    # `!=` is a disjunction with no integer-set form, and a data-dependent
    # comparison is not affine: both must keep using scf.if.
    @kernel
    def neq(out: i32[16]):
        for i in range(16):
            if i != 3:
                out[i] = 1

    @kernel
    def data_dependent(inp: i32[16], out: i32[16]):
        for i in range(16):
            if inp[i] > 0:
                out[i] = 1

    for fn in (neq, data_dependent):
        ir = _compile_ir(fn)
        _assert_contains(ir, "scf.if")
        assert "affine.if" not in ir


def test_match_case_index_switch():
    @kernel
    def top(sel: i32, out: i32[1]):
        match sel:
            case 0:
                out[0] = 10
            case 1:
                out[0] = 20
            case _:
                out[0] = 99

    ir = _compile_ir(top)
    # The subject is index-cast and carried by scf.index_switch; the wildcard
    # becomes the default region.
    _assert_contains(
        ir,
        "arith.index_cast",
        "to index",
        "scf.index_switch",
        "case 0 {",
        "case 1 {",
        "default {",
    )


def test_match_case_negative_and_buffer_subject():
    @kernel
    def top(a: i32[1], out: i32[1]):
        match a[0]:
            case -1:
                out[0] = 5
            case _:
                out[0] = 6

    ir = _compile_ir(top)
    _assert_contains(ir, "scf.index_switch", "case -1 {", "default {")


def test_match_case_without_wildcard_has_empty_default():
    @kernel
    def top(sel: i32, out: i32[1]):
        match sel:
            case 0:
                out[0] = 1
            case 2:
                out[0] = 2

    ir = _compile_ir(top)
    # scf.index_switch always carries a default region even with no `case _`.
    _assert_contains(ir, "scf.index_switch", "case 0 {", "case 2 {", "default {")


def test_match_case_guard_error():
    @kernel
    def top(sel: i32, out: i32[1]):
        match sel:
            case x if x > 0:
                out[0] = 1
            case _:
                out[0] = 2

    _assert_compile_error(top, "guards (`case ... if ...:`) are not supported")


def test_match_case_capture_pattern_error():
    @kernel
    def top(sel: i32, out: i32[1]):
        match sel:
            case 0:
                out[0] = 1
            case y:
                out[0] = y

    _assert_compile_error(
        top, "Only integer-literal patterns (`case <int>:`) and the wildcard"
    )


def test_match_case_float_subject_error():
    @kernel
    def top(sel: f32, out: i32[1]):
        match sel:
            case 0:
                out[0] = 1
            case _:
                out[0] = 2

    _assert_compile_error(top, "match is only supported on integer subjects.")


def test_match_case_phi_scalar():
    @kernel
    def top(sel: i32, out: i32[1]):
        acc: i32 = 0
        match sel:
            case 0:
                acc = 5
            case 1:
                acc = 7
            case _:
                acc = acc + 100
        out[0] = acc

    ir = _compile_ir(top)
    # A scalar reassigned across cases is threaded out as an index_switch result
    # (phi), and each region yields its value.
    _assert_contains(ir, "scf.index_switch", "-> i32", "scf.yield")


def test_match_case_phi_partial_redefine():
    # `acc` is redefined only in `case 0`; the other regions must yield the
    # dominating live-in value rather than dropping it.
    @kernel
    def top(sel: i32, out: i32[1]):
        acc: i32 = 3
        match sel:
            case 0:
                acc = 10
            case _:
                out[0] = 0
        out[0] = acc

    ir = _compile_ir(top)
    _assert_contains(ir, "scf.index_switch", "-> i32")


def test_match_case_duplicate_value_error():
    @kernel
    def top(sel: i32, out: i32[1]):
        match sel:
            case 0:
                out[0] = 1
            case 0:
                out[0] = 2
            case _:
                out[0] = 3

    _assert_compile_error(top, "duplicate match case value 0.")


def test_ternary_expression():
    @kernel
    def top(cond: allo_bool, x: i32, y: i32, out: i32[1]):
        out[0] = x if cond else y

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.select")


def test_memref_load_store():
    @kernel
    def top(inp: i32[4], out: i32[1]):
        out[0] = inp[0]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "affine.load",
        "memref<4xi32>",
        "affine.store",
        "memref<1xi32>",
    )


def test_range_loop_store():
    @kernel
    def top(out: i32[4]):
        for i in allo.range(4):
            out[i] = i

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "affine.for",
        "= 0 to 4",
        "arith.index_cast",
        "index to i32",
    )


def test_index_runtime_arithmetic():
    @kernel
    def top(stride: i32, out: i32[8]):
        offset: i32 = 2
        for i in range(4):
            out[offset + stride * i] = i

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.index_cast",
        "arith.addi",
        "arith.muli",
    )


def test_builtin_range_loop_store():
    @kernel
    def top(out: i32[4]):
        for i in range(4):
            out[i] = i

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "affine.for",
        "= 0 to 4",
    )


def test_grid_loop_store():
    @kernel
    def top(out: i32[2, 2]):
        for i, j in allo.grid(2, 2):
            out[i, j] = i + j

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "affine.parallel",
        "= (0, 0) to (2, 2)",
        "arith.addi",
        "arith.index_cast",
        "memref<2x2xi32>",
    )


def test_affine_index_floordiv_mod_mul():
    @kernel
    def top(a: f32[16], b: f32[8]):
        for i in range(8):
            b[i] = a[i * 2] + a[i // 2] + a[i % 4]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "affine.load %a[%i * 2]",
        "affine.load %a[%i floordiv 2]",
        "affine.load %a[%i mod 4]",
        "affine.store",
    )


def test_affine_per_access_fallback():
    # Decoupled per-access affine: b[i] is affine, but the indirect a[k] access
    # (k is not an affine induction variable) falls back to memref.load.
    @kernel
    def top(a: f32[16], idx: i32[16], b: f32[16]):
        for i in range(16):
            k: i32 = idx[i]
            b[i] = a[k]

    ir = _compile_ir(top)
    _assert_contains(ir, "affine.for", "memref.load %a", "affine.store %1, %b")


def test_affine_symbol_bound():
    # A runtime upper bound that is a kernel parameter is a valid affine symbol:
    # the loop stays affine.for, with an index_cast hoisted to the entry block.
    @kernel
    def top(n: i32, a: f32[64], b: f32[64]):
        for i in range(n):
            b[i] = a[i] + 1.0

    ir = _compile_ir(top)
    _assert_contains(
        ir, "arith.index_cast %n", "affine.for %i = 0 to %0", "affine.load"
    )


def test_affine_symbol_in_index():
    # An index-typed parameter used inside an index expression becomes a symbol.
    @kernel
    def top(n: index, a: f32[64], b: f32[64]):
        for i in range(n):
            b[i] = a[i + n]

    ir = _compile_ir(top)
    _assert_contains(ir, "affine.load %a[%i + symbol(%n)]")


def test_affine_tiled_dim_bound():
    # Bounds that are affine over an enclosing affine IV stay affine (no symbol,
    # no scf): the inner loop ranges over `i` .. `i + 8`.
    @kernel
    def top(a: f32[64], b: f32[64]):
        for i in range(0, 64, 8):
            for j in range(i, i + 8):
                b[j] = a[j] * 2.0

    ir = _compile_ir(top)
    assert ir.count("affine.for") == 2
    _assert_contains(ir, "affine.for", "step 8")
    assert "scf.for" not in ir


def test_affine_dynamic_grid():
    # grid() with runtime (symbol) bounds lowers to affine.parallel.
    @kernel
    def top(n: index, m: index, a: f32[16, 16], b: f32[16, 16]):
        for i, j in allo.grid(n, m):
            b[i, j] = a[i, j]

    ir = _compile_ir(top)
    _assert_contains(
        ir, "affine.parallel", "to (symbol(%n), symbol(%m))", "affine.load"
    )


def test_non_affine_bound_falls_back_to_scf():
    # A runtime bound that is a loaded value (not a top-level symbol) is not
    # affine, so the loop stays scf.for and its accesses use memref.
    @kernel
    def top(bounds: i32[4], a: f32[64]):
        k: i32 = bounds[0]
        for i in range(k):
            a[i] = 0.0

    ir = _compile_ir(top)
    _assert_contains(ir, "scf.for", "memref.store")


def test_direct_operator_invoke():
    @kernel
    def top(x: i32, y: i32, out: i32[1]):
        out[0] = allo_max(x, y)

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.maxsi")


def test_builtin_max_min():
    @kernel
    def top(x: i32, y: i32, out: i32[2]):
        out[0] = max(x, y)
        out[1] = min(x, y)

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.maxsi",
        "arith.minsi",
    )


def test_global_scalar_constexpr():
    @kernel
    def top(x: i32, y: f32, out: f32[2]):
        out[0] = x + _GLOBAL_INT_CONST
        out[1] = y + _GLOBAL_FLOAT_CONST

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant 3",
        "arith.constant 1.500000e+00",
        "arith.addi",
        "arith.addf",
    )


def test_global_shape_annotation():
    @kernel
    def top(
        inp: i32[_GLOBAL_SHAPE_M * _GLOBAL_SHAPE_N],
        out: i32[_GLOBAL_SHAPE_M, _GLOBAL_SHAPE_N],
    ):
        for i in range(_GLOBAL_SHAPE_M):
            for j in range(_GLOBAL_SHAPE_N):
                out[i, j] = inp[i * _GLOBAL_SHAPE_N + j]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "memref<6xi32>",
        "memref<2x3xi32>",
        "affine.for",
    )


def test_scope_shape_annotation():
    rows = 2
    cols = 2

    @kernel
    def top(out: i32[rows, cols]):
        for i, j in allo.grid(2, 2):
            out[i, j] = i + j

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "memref<2x2xi32>",
        "affine.parallel",
    )


def test_template_signature_shape():
    T = Template("T")
    N = Template("N")

    @kernel(T, N)
    def top(x: T, out: T[N]):
        tmp: T = x
        for i in range(N):
            out[i] = tmp

    ir = _compile_ir(top[f32, 2])
    _assert_contains(
        ir,
        "f32",
        "memref<2xf32>",
        "affine.for",
    )


def test_template_helper_specialization():
    T = Template("T")

    @kernel(T)
    def worker(x: T) -> T:
        return x

    @kernel(T)
    def top(x: T, out: T[1]):
        out[0] = worker[T](x)

    ir = _compile_ir(top[i32])
    _assert_contains(
        ir,
        "allo.kernel private @top.worker",
        "invoke @top.worker",
        "i32",
    )


def test_template_specialization_object():
    T = Template("T")

    @kernel(T)
    def top(x: T, out: T[1]):
        out[0] = x

    specialized = top[f32]
    ir = _compile_ir(specialized)
    _assert_contains(ir, "f32", "memref<1xf32>")


def test_local_memref_declaration():
    @kernel
    def top(out: i32[4]):
        N: constexpr = 4
        buf: i32[N]
        for i in range(N):
            buf[i] = i
            out[i] = buf[i]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "memref.alloc",
        "memref<4xi32>",
        "affine.load",
    )


def test_local_tensor_declaration():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top() -> f32[4]:
        N: constexpr = 4
        buf: f32[N]
        return buf

    ir = _compile_ir(top)
    _assert_contains(ir, "tensor.empty", "tensor<4xf32>")


def test_memref_list_initializer():
    @kernel
    def top(out: i32[2, 2]):
        scale: constexpr = _GLOBAL_INT_CONST
        buf: i32[2, 2] = [[1, scale], [scale + 1, scale + 2]]
        for i, j in allo.grid(2, 2):
            out[i, j] = buf[i, j]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        'memref.global "private" @_allo_const_top_buf_l3c4',
        "memref.get_global @_allo_const_top_buf_l3c4",
        "dense<[[1, 3], [4, 5]]>",
        "affine.load",
    )


def test_tensor_list_initializer():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top() -> i32[2, 2]:
        buf: i32[2, 2] = [[1, 2], [3, 4]]
        return buf

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>")


def test_memref_numpy_initializer():
    # A captured NumPy array becomes a module-global constant buffer, just like a
    # nested-list literal initializer.
    @kernel
    def top(out: i32[2, 2]):
        buf: i32[2, 2] = _GLOBAL_NP_INT
        for i, j in allo.grid(2, 2):
            out[i, j] = buf[i, j]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        'memref.global "private" @_allo_const_top_buf',
        "memref.get_global @_allo_const_top_buf",
        "dense<[[1, 2], [3, 4]]>",
        "affine.load",
    )


def test_tensor_numpy_initializer():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top() -> i32[2, 2]:
        buf: i32[2, 2] = _GLOBAL_NP_INT
        return buf

    ir = _compile_ir(top, options=KernelOptions(enable_tensor=True))
    _assert_contains(ir, "arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>")


def test_numpy_initializer_float():
    @kernel
    def top(out: f32[2]):
        buf: f32[2] = _GLOBAL_NP_FLOAT
        out[0] = buf[0]
        out[1] = buf[1]

    ir = _compile_ir(top)
    _assert_contains(ir, "dense<[1.500000e+00, 2.500000e+00]>")


def test_numpy_initializer_shape_mismatch():
    @kernel
    def top(out: i32[2, 3]):
        buf: i32[2, 3] = _GLOBAL_NP_INT
        out[0, 0] = buf[0, 0]

    _assert_compile_error(top, "shape mismatch", "expected (2, 3), got (2, 2)")


def test_numpy_initializer_unsupported_dtype():
    @kernel
    def top(out: i32[2]):
        buf: i32[2] = _GLOBAL_NP_BOOL
        out[0] = buf[0]

    _assert_compile_error(top, "integer or floating-point dtype")


def test_numpy_initializer_requires_shaped_type():
    @kernel
    def top() -> i32:
        x: i32 = _GLOBAL_NP_INT
        return x

    _assert_compile_error(top, "can only initialize a shaped variable")


def test_bufferize_bound_static_slice():
    # Bounded call `src.bufferize(...)`: a strided slice lowers to a module-level
    # private copy kernel whose affine.for reads `src[offset + i*stride]`.
    @kernel
    def top(A: i32[8], out: i32[4]):
        new = A.bufferize([1], [4], [2])
        for i in range(4):
            out[i] = new[i]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "invoke @allo_bufferize_top_A",
        "allo.kernel private @allo_bufferize_top_A",
        "(%dst: memref<4xi32>, %src: memref<8xi32>)",
        "affine.load %src[%i0 * 2 + 1]",
        "affine.store %new, %dst[%i0]",
    )


def test_bufferize_free_numpy_dynamic_offset():
    # Free-function call `allo.bufferize(np_array, ...)`: the NumPy constant becomes
    # a module global and a dynamic offset is threaded through as an affine symbol
    # (extra `%off0` kernel parameter).
    @kernel
    def top(r: index, out: i32[2, 2]):
        new = allo.bufferize(_GLOBAL_NP_INT, [r, 0], [2, 2], [1, 1])
        for i in range(2):
            for j in range(2):
                out[i, j] = new[i, j]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        'memref.global "private" @_allo_const_top__GLOBAL_NP_INT',
        "allo.kernel private @allo_bufferize_top__GLOBAL_NP_INT",
        "(%dst: memref<2x2xi32>, %src: memref<2x2xi32>, %off0: index)",
        "affine.load %src[%i0 + symbol(%off0), %i1]",
    )


def test_stream_scalar_ir():
    @kernel
    def top(x: i32, out: i32[1]):
        fifo: Stream[i32][2, 2]
        fifo[0, 1].put(x)
        out[0] = fifo[0, 1].get()

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.stream.create",
        "!allo.stream<i32,2,[2,2]>",
        "allo.stream.put",
        "allo.stream.get",
    )


def test_stream_nested_parameter_ir():
    @kernel
    def top(x: i32, out: i32[1]):
        fifo: Stream[i32][2, 2]

        @kernel
        def worker(s: Stream[i32][2, 2], v: i32):
            s[0, 1].put(v)

        worker(fifo, x)
        out[0] = fifo[0, 1].get()

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.stream.create",
        "!allo.stream<i32,2,[2,2]>",
        "allo.kernel private @top.worker",
        "(%s: !allo.stream<i32,2,[2,2]>",
        "invoke @top.worker",
        "allo.stream.put",
        "allo.stream.get",
    )


def test_nested_kernel_mapping_ir():
    @kernel
    def top(out: i32[1]):
        workers: constexpr = 2

        @kernel(mapping=[workers])
        def worker(buf: i32[1]):
            buf[0] = 1

        worker(out)

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.kernel private @top.worker(%buf: memref<1xi32>) mapping=[2]",
        "invoke @top.worker",
    )


def test_bound_method_compile_errors():
    @kernel
    def top(x: i32):
        x.put(1)

    @kernel
    def worker():
        x: constexpr = 1
        x.put(1)

    _assert_compile_error(
        top,
        "Stream get/put expects a stream value, got 'int32'.",
    )
    _assert_compile_error(
        worker,
        "constexpr value '1' has no attribute 'put'.",
    )


def test_for_loop_carried_values():
    @kernel
    def top(out: i32[1]):
        acc: i32 = 0
        for i in range(4):
            i_i32: i32 = i
            acc += i_i32
        out[0] = acc

    ir = _compile_ir(top)
    _assert_contains(ir, "affine.for", "iter_args", "affine.yield")


def test_while_loop_carried_values():
    @kernel
    def top(out: i32[1]):
        i: i32 = 0
        acc: i32 = 0
        while i < 4:
            acc += i
            i += 1
        out[0] = acc

    ir = _compile_ir(top)
    _assert_contains(ir, "scf.while", "scf.condition", "scf.yield")


def test_consteval_expression():
    @consteval
    def factor():
        return 3

    @kernel
    def top(x: i32, out: i32[1]):
        out[0] = x + factor()

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant 3 : i33",
        "arith.extsi",
        "arith.addi",
        "i33 to i32",
    )


def test_lazy_consteval_kernel_folds_to_constant():
    # `@consteval(lazy=True)` does NOT run at trace time: it enters the IR as an
    # `allo.kernel` tagged `allo.lazy`, called via `allo.invoke`. The
    # `fold-constant-calls` pass then evaluates the invoke at compile time and
    # deletes the lazy kernel, so it never reaches codegen.
    from allo.backend.base import run_pipeline

    @consteval(lazy=True)
    def reverse_low_bits(data: i32, bit_range: i32) -> i32:
        mask = (1 << bit_range) - 1
        rev: i32 = 0
        for i in range(0, bit_range):
            i_32: i32 = i
            if data & (1 << i_32):
                rev |= 1 << (bit_range - 1 - i_32)
        return (data & ~mask) | rev

    @kernel
    def top(out: i32[1]):
        out[0] = reverse_low_bits(1, 3)  # reverse 0b001 over 3 bits -> 0b100 = 4

    # Frontend keeps it lazy: an `allo.lazy` kernel reached through an invoke.
    ir = _compile_ir(top)
    _assert_contains(ir, "allo.lazy", "invoke @")

    # The pass evaluates the invoke and removes the lazy kernel entirely.
    module = compile_kernel(top)
    run_pipeline(module, "builtin.module(fold-constant-calls)")
    folded = str(module)
    assert "allo.lazy" not in folded
    assert "invoke @" not in folded
    _assert_contains(folded, "arith.constant 4 : i32")

    # Left unfolded (e.g. the CPU/JIT path), it still runs as a normal kernel.
    out = np.zeros(1, dtype=np.int32)
    top(out)
    assert out[0] == 4


def test_lazy_consteval_folds_local_scratch_array():
    # The evaluator is not limited to scalar arithmetic: a lazy consteval that
    # builds a local scratch array (memref) over a loop still folds to a constant
    # (loop unroll + affine store-to-load forwarding).
    from allo.backend.base import run_pipeline

    @consteval(lazy=True)
    def square_table(sel: i32) -> i32:
        tbl: i32[4]
        for t in range(0, 4):
            tbl[t] = t * t
        return tbl[sel]

    @kernel
    def top(out: i32[1]):
        out[0] = square_table(3)  # [0, 1, 4, 9][3] = 9

    module = compile_kernel(top)
    run_pipeline(module, "builtin.module(fold-constant-calls)")
    folded = str(module)
    assert "allo.lazy" not in folded
    assert "invoke @" not in folded
    _assert_contains(folded, "arith.constant 9 : i32")

    out = np.zeros(1, dtype=np.int32)
    top(out)
    assert out[0] == 9


def test_lazy_consteval_requires_constant_args():
    # `@consteval(lazy=True)` is an explicit request to fold the call away, so an
    # invoke whose arguments are not compile-time constants is a hard error.
    from allo.backend.base import run_pipeline

    @consteval(lazy=True)
    def add(a: i32, b: i32) -> i32:
        return a + b

    @kernel
    def top(r: i32, out: i32[1]):
        out[0] = add(r, 3)  # r is a runtime value -> cannot be folded

    module = compile_kernel(top)
    with pytest.raises(Exception, match="lazy consteval"):
        run_pipeline(module, "builtin.module(fold-constant-calls)")


def test_nested_invoke_store():
    @kernel
    def top(x: i32, out: i32[1]):
        @kernel
        def worker(v: i32) -> i32:
            return v + 1

        out[0] = worker(x)

    ir = _compile_ir(top)
    _assert_contains(ir, "allo.kernel private @top.worker", "invoke @top.worker")


def test_nested_multiple_returns():
    @kernel
    def top(x: i32, y: i32, out: i32[1]):
        @kernel
        def worker(a: i32, b: i32) -> (i32, i32):
            return a, b

        lhs, rhs = worker(x, y)
        out[0] = lhs + rhs

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.kernel private @top.worker",
        "-> (i32, i32)",
        "invoke @top.worker",
        "arith.addi",
    )


def test_nested_capture_constexpr():
    @kernel
    def top(x: i32, out: i32[1]):
        offset: constexpr = 3

        @kernel
        def worker(v: i32) -> i32:
            return v + offset

        out[0] = worker(x)

    ir = _compile_ir(top)
    _assert_contains(ir, "allo.kernel private @top.worker", "arith.constant 3")


def test_nested_capture_type_alias():
    @kernel
    def top(out: i32[1]):
        T: constexpr = i32

        @kernel
        def worker() -> T:
            return 7

        out[0] = worker()

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.kernel private @top.worker",
        "-> i32",
        "arith.constant 7 : i32",
    )


def test_nested_capture_consteval():
    @consteval
    def amount():
        return 5

    @kernel
    def top(x: i32, out: i32[1]):
        @kernel
        def worker(v: i32) -> i32:
            return v + amount()

        out[0] = worker(x)

    ir = _compile_ir(top)
    _assert_contains(ir, "allo.kernel private @top.worker", "arith.constant 5")


def test_nested_capture_kernel_alias():
    @kernel
    def callee(v: i32) -> i32:
        return v + 2

    @kernel
    def top(x: i32, out: i32[1]):
        invokeee: constexpr = callee

        @kernel
        def worker(v: i32) -> i32:
            return invokeee(v)

        out[0] = worker(x)

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.kernel private @top.worker.callee",
        "invoke @top.worker.callee",
        "invoke @top.worker",
    )


def test_nested_capture_module_alias():
    @kernel
    def top(out: i32[2]):
        M: constexpr = allo.lang.core

        @kernel
        def worker(buf: i32[2]):
            for i in M.range(2):
                buf[i] = i

        worker(out)

    ir = _compile_ir(top)
    _assert_contains(ir, "allo.kernel private @top.worker", "affine.for")


def test_cpp_typing_compile():
    @kernel(options=KernelOptions(typing_style="cpp"))
    def top(x: u32, y: i32, out: u32[1]):
        out[0] = x + y

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.addi", ": i32")


def test_return_scalar_value():
    @kernel
    def top(x: i32, y: i32) -> i32:
        return x + y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "-> i32",
        "arith.addi",
        "i33 to i32",
    )


def test_return_constexpr_literal():
    @kernel
    def top() -> i32:
        return 3

    ir = _compile_ir(top)
    _assert_contains(ir, "-> i32", "arith.constant 3 : i32")


def test_return_multiple_values():
    @kernel
    def top(x: i32, y: f32) -> (i32, f32):
        return x, y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "-> (i32, f32)",
        "return",
        ": i32, f32",
    )


def test_return_if_else():
    @kernel
    def top(cond: allo_bool, x: i32, y: i32) -> i32:
        if cond:
            return x
        else:
            return y

    ir = _compile_ir(top)
    _assert_contains(ir, "cf.cond_br", "return", ": i32")


def test_return_if_fallthrough():
    @kernel
    def top(cond: allo_bool, x: i32, y: i32) -> i32:
        if cond:
            return x
        return y

    ir = _compile_ir(top)
    _assert_contains(ir, "cf.cond_br", "return", ": i32")


def test_return_requires_annotation():
    @kernel
    def top(x: i32):
        return x

    _assert_compile_error(
        top,
        "Return values require an explicit return annotation.",
    )


def test_return_missing_non_void():
    @kernel
    def top(x: i32) -> i32:
        y = x + x

    _assert_compile_error(
        top,
        "Missing return statement for non-void function",
    )


def test_return_count_mismatch():
    @kernel
    def top(x: i32, y: i32) -> (i32, i32):
        return x

    _assert_compile_error(
        top,
        "Return value count mismatch: expected 2, got 1.",
    )


def test_return_type_mismatch():
    @kernel
    def top(x: i32[2]) -> i32[1]:
        return x

    _assert_compile_error(
        top,
        "Cannot cast from memref<2xint32> to memref<1xint32>",
    )


def test_return_inside_loop_error():
    @kernel
    def top(x: i32) -> i32:
        for i in allo.range(4):
            return x
        return x

    _assert_compile_error(
        top,
        "'return' is not supported inside loops",
    )


def test_return_nested_if_error():
    @kernel
    def top(cond: allo_bool, inner: allo_bool, x: i32) -> i32:
        if cond:
            if inner:
                return x
        return x

    _assert_compile_error(
        top,
        "'return' is not supported inside nested 'if' statements.",
    )


def test_kernel_defined_via_python_dash_c():
    """A kernel defined through ``python -c`` must be compilable."""
    import subprocess
    import sys

    code = (
        "from __future__ import annotations\n"
        "from allo.lang import kernel\n"
        "from allo.lang.core import i32\n"
        "@kernel\n"
        "def top(x: i32, y: i32, out: i32[1]):\n"
        "    out[0] = x + y\n"
        "top.schedule()\n"
        "print(str(top.module))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "extra_arg"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    _assert_contains(result.stdout, "allo.kernel public @top", "arith.addi")
