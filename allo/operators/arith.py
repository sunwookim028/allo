# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin
# pylint: disable=unused-argument

from __future__ import annotations

import builtins

from ..compiler.builder import AlloOpBuilder, CmpPred
from ..lang.core import (
    AlloValue,
    ConstexprValue,
    DType,
    ShapedType,
    TypeBase,
    bool as AlloBool,
)
from ..lang.operator import NO_FOLD, operator
from .utils import (
    emit_linalg_binary,
    emit_linalg_unary,
    is_default_acc,
    operator_body_unreachable,
)
from .._mlir.dialects import linalg as linalg_d


def _fold_binary(lhs, rhs, fn):
    if not (isinstance(lhs, ConstexprValue) and isinstance(rhs, ConstexprValue)):
        return NO_FOLD
    try:
        return ConstexprValue(fn(lhs.value, rhs.value))
    except Exception:
        return NO_FOLD


def _fold_unary(operand, fn):
    if not isinstance(operand, ConstexprValue):
        return NO_FOLD
    try:
        return ConstexprValue(fn(operand.value))
    except Exception:
        return NO_FOLD


def _fold_const_bool(value):
    if isinstance(value, ConstexprValue) and isinstance(value.value, bool):
        return value.value
    return None


def _const_bool(value, name: str) -> bool:
    assert isinstance(value, ConstexprValue) and isinstance(
        value.value, bool
    ), f"'{name}' must be a boolean constexpr"
    return value.value


def _fold_div(lhs, rhs):
    """Constant division matching codegen: integers truncate toward zero (like
    ``arith.divsi``/``divui``), floats use true division (``arith.divf``)."""
    if isinstance(lhs, float) or isinstance(rhs, float):
        return lhs / rhs
    quotient = abs(lhs) // abs(rhs)
    return -quotient if (lhs < 0) != (rhs < 0) else quotient


def _fold_mod(lhs, rhs):
    """Constant remainder matching codegen: the remainder takes the dividend's
    sign (like ``arith.remsi``/``remui``/``remf``), unlike Python's floored ``%``."""
    remainder = abs(lhs) % abs(rhs)
    return -remainder if lhs < 0 else remainder


def _fold_floordiv(lhs, rhs):
    """Constant ``//`` matching codegen: integers truncate toward zero like ``/``
    (``divsi``/``divui``); floats keep Python's floor (``divf`` + ``math.floor``)."""
    if isinstance(lhs, float) or isinstance(rhs, float):
        return lhs // rhs
    return _fold_div(lhs, rhs)


def _fold_enabled(acc) -> bool:
    return is_default_acc(acc)


def _has_shaped(*values) -> bool:
    return any(
        isinstance(value, AlloValue) and isinstance(value.type, ShapedType)
        for value in values
    )


def _materialize_binary_operands(
    builder: AlloOpBuilder, lhs, rhs, acc=ConstexprValue(None), op_name="operator"
):
    if isinstance(lhs, ConstexprValue) and isinstance(rhs, ConstexprValue):
        if not is_default_acc(acc):
            if not isinstance(acc, AlloValue):
                return builder.compile_error(
                    f"Operator '{op_name}' acc must be a shaped value"
                )
            return builder.cast(lhs, acc.dtype), builder.cast(rhs, acc.dtype)
        return lhs, rhs
    if isinstance(lhs, ConstexprValue):
        assert isinstance(rhs, AlloValue)
        lhs = builder.cast(lhs, rhs.dtype)
    if isinstance(rhs, ConstexprValue):
        assert isinstance(lhs, AlloValue)
        rhs = builder.cast(rhs, lhs.dtype)
    return lhs, rhs


def _materialize_unary_operand(
    builder: AlloOpBuilder, operand, acc=ConstexprValue(None), op_name="operator"
):
    if isinstance(operand, ConstexprValue):
        if not is_default_acc(acc):
            if not isinstance(acc, AlloValue):
                return builder.compile_error(
                    f"Operator '{op_name}' acc must be a shaped value"
                )
            return builder.cast(operand, acc.dtype)
        return operand
    assert isinstance(operand, AlloValue)
    return operand


def _promote_binary_operands(
    builder: AlloOpBuilder,
    lhs: AlloValue,
    rhs: AlloValue,
    op_name: str,
    *,
    term_signs=None,
):
    result_dtype = builder.get_promoted_dtype_nary(
        op_name, [lhs.dtype, rhs.dtype], term_signs=term_signs
    )
    lhs = builder.cast_to_dtype(lhs, result_dtype)
    rhs = builder.cast_to_dtype(rhs, result_dtype)
    return lhs, rhs, result_dtype


def _binary_result_dtype(
    builder: AlloOpBuilder,
    lhs: AlloValue,
    rhs: AlloValue,
    op_name: str,
    *,
    term_signs=None,
) -> DType:
    return builder.get_promoted_dtype_nary(
        op_name, [lhs.dtype, rhs.dtype], term_signs=term_signs
    )


def _named_binary_op(named_op_cls, lhs: AlloValue, rhs: AlloValue, result_dtype: DType):
    if named_op_cls is None:
        return None
    if lhs.dtype == result_dtype and rhs.dtype == result_dtype:
        return named_op_cls
    return None


def _lower_binary_arith(
    builder: AlloOpBuilder,
    lhs,
    rhs,
    acc,
    op_name: str,
    result_dtype: DType,
    build_fn,
    *,
    named_op_cls=None,
):
    assert isinstance(lhs, AlloValue) and isinstance(rhs, AlloValue)
    if not _has_shaped(lhs, rhs) and is_default_acc(acc):
        return build_fn(lhs, rhs)
    return emit_linalg_binary(
        builder,
        lhs,
        rhs,
        result_dtype,
        build_fn,
        named_op_cls=_named_binary_op(named_op_cls, lhs, rhs, result_dtype),
        acc=acc,
        op_name=op_name,
    )


def _lower_unary_arith(
    builder: AlloOpBuilder,
    operand,
    acc,
    op_name: str,
    result_dtype: DType,
    build_fn,
):
    assert isinstance(operand, AlloValue)
    if not isinstance(operand.type, ShapedType) and is_default_acc(acc):
        return build_fn(operand)
    return emit_linalg_unary(
        builder, operand, result_dtype, build_fn, acc=acc, op_name=op_name
    )


def _signed_and_floating(lhs: AlloValue, rhs: AlloValue):
    floating = lhs.dtype.is_float() and rhs.dtype.is_float()
    signed = not (lhs.dtype.is_uint() and rhs.dtype.is_uint())
    if floating:
        signed = False
    return signed, floating


def _build_add(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    lhs, rhs, _ = _promote_binary_operands(builder, lhs, rhs, "add")
    return builder.create_add(lhs, rhs, floating=lhs.dtype.is_float())


def _build_sub(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    lhs, rhs, _ = _promote_binary_operands(builder, lhs, rhs, "sub", term_signs=[1, -1])
    return builder.create_sub(lhs, rhs, floating=lhs.dtype.is_float())


def _build_mul(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    lhs, rhs, _ = _promote_binary_operands(builder, lhs, rhs, "mul")
    return builder.create_mul(lhs, rhs, floating=lhs.dtype.is_float())


def _build_div(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    lhs, rhs, result_dtype = _promote_binary_operands(builder, lhs, rhs, "div")
    floating = result_dtype.is_float()
    return builder.create_div(lhs, rhs, signed=result_dtype.is_int(), floating=floating)


def _build_floordiv(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    lhs, rhs, result_dtype = _promote_binary_operands(builder, lhs, rhs, "floordiv")
    floating = result_dtype.is_float()
    return builder.create_floordiv(
        lhs, rhs, signed=result_dtype.is_int(), floating=floating
    )


def _build_mod(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    lhs, rhs, result_dtype = _promote_binary_operands(builder, lhs, rhs, "mod")
    floating = result_dtype.is_float()
    return builder.create_mod(lhs, rhs, signed=result_dtype.is_int(), floating=floating)


def _build_pow(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    lhs, rhs, _ = _promote_binary_operands(builder, lhs, rhs, "pow")
    return builder.create_pow(
        lhs, rhs, base_floating=lhs.dtype.is_float(), exp_floating=rhs.dtype.is_float()
    )


def _build_lshift(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    lhs, rhs, _ = _promote_binary_operands(builder, lhs, rhs, "lshift")
    return builder.create_lshift(lhs, rhs)


def _build_rshift(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    lhs, rhs, result_dtype = _promote_binary_operands(builder, lhs, rhs, "rshift")
    return builder.create_rshift(lhs, rhs, signed=result_dtype.is_int())


def _build_bitwise(
    builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue, op_name: str
):
    lhs, rhs, _ = _promote_binary_operands(builder, lhs, rhs, op_name)
    if op_name == "bitwise_and":
        return builder.create_bitwise_and(lhs, rhs)
    if op_name == "bitwise_or":
        return builder.create_bitwise_or(lhs, rhs)
    assert op_name == "bitwise_xor"
    return builder.create_bitwise_xor(lhs, rhs)


def _build_cmp(
    builder: AlloOpBuilder,
    lhs: AlloValue,
    rhs: AlloValue,
    pred: CmpPred,
    op_name: str,
    ordered: bool,
):
    lhs, rhs, _ = _promote_binary_operands(builder, lhs, rhs, op_name)
    if lhs.dtype.is_float():
        return builder.create_cmpf(lhs, rhs, pred, ordered=ordered)
    return builder.create_cmpi(lhs, rhs, pred, signed=lhs.dtype.is_int())


def _build_logical_binary(
    builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue, op_name: str
):
    lhs, rhs, _ = _promote_binary_operands(builder, lhs, rhs, op_name)
    if op_name == "logical_and":
        return builder.create_logical_and(lhs, rhs)
    assert op_name == "logical_or"
    return builder.create_logical_or(lhs, rhs)


def _build_max_min(
    builder: AlloOpBuilder,
    lhs: AlloValue,
    rhs: AlloValue,
    op_name: str,
    propagate_nan: bool,
):
    lhs, rhs, _ = _promote_binary_operands(builder, lhs, rhs, op_name)
    signed, floating = _signed_and_floating(lhs, rhs)
    if op_name == "max":
        return builder.create_max(
            lhs,
            rhs,
            signed=signed,
            floating=floating,
            propagate_nan=propagate_nan,
        )
    assert op_name == "min"
    return builder.create_min(
        lhs, rhs, signed=signed, floating=floating, propagate_nan=propagate_nan
    )


def _build_neg(builder: AlloOpBuilder, operand: AlloValue):
    result_dtype = builder.get_promoted_dtype_nary("neg", [operand.dtype])
    operand = builder.cast_to_dtype(operand, result_dtype)
    return builder.create_neg(operand, floating=operand.dtype.is_float())


def _build_invert(builder: AlloOpBuilder, operand: AlloValue):
    result_dtype = builder.get_promoted_dtype_nary("invert", [operand.dtype])
    operand = builder.cast_to_dtype(operand, result_dtype)
    return builder.create_invert(operand)


def _build_logical_not(builder: AlloOpBuilder, operand: AlloValue):
    result_dtype = builder.get_promoted_dtype_nary("logical_not", [operand.dtype])
    operand = builder.cast_to_dtype(operand, result_dtype)
    return builder.create_logical_not(operand)


@operator
def add(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@add.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs + rhs)


@add.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "add")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "add")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "add",
        result_dtype,
        lambda lhs, rhs: _build_add(builder, lhs, rhs),
        named_op_cls=linalg_d.AddOp,
    )


@operator
def sub(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@sub.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs - rhs)


@sub.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "sub")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "sub", term_signs=[1, -1])
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "sub",
        result_dtype,
        lambda lhs, rhs: _build_sub(builder, lhs, rhs),
        named_op_cls=linalg_d.SubOp,
    )


@operator
def mul(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@mul.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs * rhs)


@mul.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "mul")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "mul")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "mul",
        result_dtype,
        lambda lhs, rhs: _build_mul(builder, lhs, rhs),
        named_op_cls=linalg_d.MulOp,
    )


@operator
def div(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@div.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, _fold_div)


@div.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "div")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "div")
    named_op_cls = linalg_d.DivUnsignedOp if result_dtype.is_uint() else linalg_d.DivOp
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "div",
        result_dtype,
        lambda lhs, rhs: _build_div(builder, lhs, rhs),
        named_op_cls=named_op_cls,
    )


@operator
def floordiv(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@floordiv.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, _fold_floordiv)


@floordiv.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "floordiv")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "floordiv")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "floordiv",
        result_dtype,
        lambda lhs, rhs: _build_floordiv(builder, lhs, rhs),
    )


@operator
def mod(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@mod.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, _fold_mod)


@mod.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "mod")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "mod")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "mod",
        result_dtype,
        lambda lhs, rhs: _build_mod(builder, lhs, rhs),
    )


@operator
def pow(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@pow.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs**rhs)


@pow.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "pow")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "pow")
    named_op_cls = (
        linalg_d.PowFOp if x.dtype.is_float() and y.dtype.is_float() else None
    )
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "pow",
        result_dtype,
        lambda lhs, rhs: _build_pow(builder, lhs, rhs),
        named_op_cls=named_op_cls,
    )


@operator
def lshift(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@lshift.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs << rhs)


@lshift.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "lshift")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "lshift")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "lshift",
        result_dtype,
        lambda lhs, rhs: _build_lshift(builder, lhs, rhs),
    )


@operator
def rshift(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@rshift.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    # A constant is only negative when it is signed, so Python's arithmetic
    # ``>>`` matches both ``arith.shrsi`` and (for non-negative) ``shrui``.
    return _fold_binary(x, y, lambda lhs, rhs: lhs >> rhs)


@rshift.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "rshift")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "rshift")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "rshift",
        result_dtype,
        lambda lhs, rhs: _build_rshift(builder, lhs, rhs),
    )


@operator
def bitwise_and(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@bitwise_and.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs & rhs)


@bitwise_and.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "bitwise_and")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "bitwise_and")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "bitwise_and",
        result_dtype,
        lambda lhs, rhs: _build_bitwise(builder, lhs, rhs, "bitwise_and"),
    )


@operator
def bitwise_or(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@bitwise_or.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs | rhs)


@bitwise_or.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "bitwise_or")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "bitwise_or")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "bitwise_or",
        result_dtype,
        lambda lhs, rhs: _build_bitwise(builder, lhs, rhs, "bitwise_or"),
    )


@operator
def bitwise_xor(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@bitwise_xor.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs ^ rhs)


@bitwise_xor.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "bitwise_xor")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "bitwise_xor")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "bitwise_xor",
        result_dtype,
        lambda lhs, rhs: _build_bitwise(builder, lhs, rhs, "bitwise_xor"),
    )


def _lower_cmp(builder, x, y, acc, op_name: str, pred: CmpPred, ordered):
    ordered_value = _const_bool(ordered, "ordered")
    x, y = _materialize_binary_operands(builder, x, y, acc, op_name)
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        op_name,
        AlloBool,
        lambda lhs, rhs: _build_cmp(builder, lhs, rhs, pred, op_name, ordered_value),
    )


@operator
def eq(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    operator_body_unreachable()


@eq.fold
def _(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    if not _fold_enabled(acc) or _fold_const_bool(ordered) is None:
        return NO_FOLD
    if isinstance(x, TypeBase) and isinstance(y, TypeBase):
        return ConstexprValue(x == y)
    return _fold_binary(x, y, lambda lhs, rhs: lhs == rhs)


@eq.build
def _(
    builder: AlloOpBuilder,
    x,
    y,
    acc=ConstexprValue(None),
    ordered=ConstexprValue(False),
):
    return _lower_cmp(builder, x, y, acc, "eq", CmpPred.EQ, ordered)


@operator
def ne(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    operator_body_unreachable()


@ne.fold
def _(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    if not _fold_enabled(acc) or _fold_const_bool(ordered) is None:
        return NO_FOLD
    if isinstance(x, TypeBase) and isinstance(y, TypeBase):
        return ConstexprValue(x != y)
    return _fold_binary(x, y, lambda lhs, rhs: lhs != rhs)


@ne.build
def _(
    builder: AlloOpBuilder,
    x,
    y,
    acc=ConstexprValue(None),
    ordered=ConstexprValue(False),
):
    return _lower_cmp(builder, x, y, acc, "ne", CmpPred.NE, ordered)


@operator
def lt(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    operator_body_unreachable()


@lt.fold
def _(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    if not _fold_enabled(acc) or _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs < rhs)


@lt.build
def _(
    builder: AlloOpBuilder,
    x,
    y,
    acc=ConstexprValue(None),
    ordered=ConstexprValue(False),
):
    return _lower_cmp(builder, x, y, acc, "lt", CmpPred.LT, ordered)


@operator
def le(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    operator_body_unreachable()


@le.fold
def _(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    if not _fold_enabled(acc) or _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs <= rhs)


@le.build
def _(
    builder: AlloOpBuilder,
    x,
    y,
    acc=ConstexprValue(None),
    ordered=ConstexprValue(False),
):
    return _lower_cmp(builder, x, y, acc, "le", CmpPred.LE, ordered)


@operator
def gt(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    operator_body_unreachable()


@gt.fold
def _(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    if not _fold_enabled(acc) or _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs > rhs)


@gt.build
def _(
    builder: AlloOpBuilder,
    x,
    y,
    acc=ConstexprValue(None),
    ordered=ConstexprValue(False),
):
    return _lower_cmp(builder, x, y, acc, "gt", CmpPred.GT, ordered)


@operator
def ge(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    operator_body_unreachable()


@ge.fold
def _(x, y, acc=ConstexprValue(None), ordered=ConstexprValue(False)):
    if not _fold_enabled(acc) or _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs >= rhs)


@ge.build
def _(
    builder: AlloOpBuilder,
    x,
    y,
    acc=ConstexprValue(None),
    ordered=ConstexprValue(False),
):
    return _lower_cmp(builder, x, y, acc, "ge", CmpPred.GE, ordered)


@operator
def pos(x, acc=ConstexprValue(None)):
    operator_body_unreachable()


@pos.fold
def _(x, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if isinstance(x, ConstexprValue):
        return x
    return NO_FOLD


@pos.build
def _(builder: AlloOpBuilder, x: AlloValue, acc=ConstexprValue(None)):  # noqa: ARG001
    x = _materialize_unary_operand(builder, x, acc, "pos")
    assert isinstance(x, AlloValue)
    if is_default_acc(acc):
        return x
    return _lower_unary_arith(builder, x, acc, "pos", x.dtype, lambda operand: operand)


@operator
def neg(x, acc=ConstexprValue(None)):
    operator_body_unreachable()


@neg.fold
def _(x, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_unary(x, lambda operand: -operand)


@neg.build
def _(builder: AlloOpBuilder, x, acc=ConstexprValue(None)):
    x = _materialize_unary_operand(builder, x, acc, "neg")
    assert isinstance(x, AlloValue)
    result_dtype = builder.get_promoted_dtype_nary("neg", [x.dtype])
    return _lower_unary_arith(
        builder,
        x,
        acc,
        "neg",
        result_dtype,
        lambda operand: _build_neg(builder, operand),
    )


@operator
def invert(x, acc=ConstexprValue(None)):
    operator_body_unreachable()


@invert.fold
def _(x, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_unary(x, lambda operand: ~operand)


@invert.build
def _(builder: AlloOpBuilder, x, acc=ConstexprValue(None)):
    x = _materialize_unary_operand(builder, x, acc, "invert")
    assert isinstance(x, AlloValue)
    result_dtype = builder.get_promoted_dtype_nary("invert", [x.dtype])
    return _lower_unary_arith(
        builder,
        x,
        acc,
        "invert",
        result_dtype,
        lambda operand: _build_invert(builder, operand),
    )


@operator
def logical_and(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@logical_and.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: bool(lhs) and bool(rhs))


@logical_and.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "logical_and")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "logical_and",
        AlloBool,
        lambda lhs, rhs: _build_logical_binary(builder, lhs, rhs, "logical_and"),
    )


@operator
def logical_or(x, y, acc=ConstexprValue(None)):
    operator_body_unreachable()


@logical_or.fold
def _(x, y, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: bool(lhs) or bool(rhs))


@logical_or.build
def _(builder: AlloOpBuilder, x, y, acc=ConstexprValue(None)):
    x, y = _materialize_binary_operands(builder, x, y, acc, "logical_or")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "logical_or",
        AlloBool,
        lambda lhs, rhs: _build_logical_binary(builder, lhs, rhs, "logical_or"),
    )


@operator
def logical_not(x, acc=ConstexprValue(None)):
    operator_body_unreachable()


@logical_not.fold
def _(x, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_unary(x, lambda operand: not bool(operand))


@logical_not.build
def _(builder: AlloOpBuilder, x, acc=ConstexprValue(None)):
    x = _materialize_unary_operand(builder, x, acc, "logical_not")
    assert isinstance(x, AlloValue)
    result_dtype = builder.get_promoted_dtype_nary("logical_not", [x.dtype])
    return _lower_unary_arith(
        builder,
        x,
        acc,
        "logical_not",
        result_dtype,
        lambda operand: _build_logical_not(builder, operand),
    )


@operator
def max(x, y, acc=ConstexprValue(None), propagate_nan=ConstexprValue(False)):
    operator_body_unreachable()


@max.fold
def _(x, y, acc=ConstexprValue(None), propagate_nan=ConstexprValue(False)):
    if not _fold_enabled(acc) or _fold_const_bool(propagate_nan) is None:
        return NO_FOLD
    return _fold_binary(x, y, builtins.max)


@max.build
def _(
    builder: AlloOpBuilder,
    x,
    y,
    acc=ConstexprValue(None),
    propagate_nan=ConstexprValue(False),
):
    propagate_nan_value = _const_bool(propagate_nan, "propagate_nan")
    x, y = _materialize_binary_operands(builder, x, y, acc, "max")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "max")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "max",
        result_dtype,
        lambda lhs, rhs: _build_max_min(builder, lhs, rhs, "max", propagate_nan_value),
    )


@operator
def min(x, y, acc=ConstexprValue(None), propagate_nan=ConstexprValue(False)):
    operator_body_unreachable()


@min.fold
def _(x, y, acc=ConstexprValue(None), propagate_nan=ConstexprValue(False)):
    if not _fold_enabled(acc) or _fold_const_bool(propagate_nan) is None:
        return NO_FOLD
    return _fold_binary(x, y, builtins.min)


@min.build
def _(
    builder: AlloOpBuilder,
    x,
    y,
    acc=ConstexprValue(None),
    propagate_nan=ConstexprValue(False),
):
    propagate_nan_value = _const_bool(propagate_nan, "propagate_nan")
    x, y = _materialize_binary_operands(builder, x, y, acc, "min")
    assert isinstance(x, AlloValue) and isinstance(y, AlloValue)
    result_dtype = _binary_result_dtype(builder, x, y, "min")
    return _lower_binary_arith(
        builder,
        x,
        y,
        acc,
        "min",
        result_dtype,
        lambda lhs, rhs: _build_max_min(builder, lhs, rhs, "min", propagate_nan_value),
    )


@operator
def cast(x, dst_type):
    operator_body_unreachable()


@cast.build
def _(builder: AlloOpBuilder, x: AlloValue | ConstexprValue, dst_type: TypeBase):
    assert isinstance(dst_type, TypeBase)
    return builder.cast(x, dst_type)


@operator
def bitcast(x, dst_type):
    operator_body_unreachable()


@bitcast.build
def _(builder: AlloOpBuilder, x: AlloValue, dst_type: TypeBase):
    assert isinstance(dst_type, DType), "bitcast destination must be a dtype"
    return builder.bitcast(x, dst_type)
