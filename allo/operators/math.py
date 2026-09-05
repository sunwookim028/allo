# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin
# pylint: disable=unused-argument

from __future__ import annotations

import builtins
import math as py_math

from ..compiler.builder import AlloOpBuilder
from ..lang.core import (
    AlloValue,
    ConstexprValue,
    ShapedType,
    f32,
    i32,
)
from ..lang.operator import NO_FOLD, operator
from .utils import (
    emit_linalg_binary,
    emit_linalg_unary,
    is_default_acc,
    operator_body_unreachable,
)
from .._mlir.dialects import linalg, math


def _is_const(value, expected):
    return isinstance(value, ConstexprValue) and value.value == expected


def _fold_unary(value, fn):
    if not isinstance(value, ConstexprValue):
        return NO_FOLD
    try:
        return ConstexprValue(fn(value.value))
    except Exception:
        return NO_FOLD


def _fold_binary(lhs, rhs, fn):
    if not (isinstance(lhs, ConstexprValue) and isinstance(rhs, ConstexprValue)):
        return NO_FOLD
    try:
        return ConstexprValue(fn(lhs.value, rhs.value))
    except Exception:
        return NO_FOLD


def _fold_enabled(acc) -> bool:
    return is_default_acc(acc)


def _materialize_unary_operand(builder: AlloOpBuilder, value, acc):
    if isinstance(value, ConstexprValue):
        dtype = acc.dtype if isinstance(acc, AlloValue) else f32
        return builder.cast(value, dtype)
    assert isinstance(value, AlloValue)
    return value


def _emit_unary_math(
    builder: AlloOpBuilder, value, acc, op_name: str, op_cls, linalg_op_cls=None
):
    operand = _materialize_unary_operand(builder, value, acc)
    result_dtype = builder.get_promoted_dtype_nary(op_name, [operand.dtype])
    operand = builder.cast_to_dtype(operand, result_dtype)

    def build_fn(inner):
        return op_cls(
            inner.handle, ip=builder.save_insertion_point(), loc=builder.get_loc()
        ).result

    if isinstance(operand.type, ShapedType) or not is_default_acc(acc):
        return emit_linalg_unary(
            builder,
            operand,
            result_dtype,
            build_fn,
            named_op_cls=linalg_op_cls,
            acc=acc,
            op_name=op_name,
        )
    return AlloValue(build_fn(operand), result_dtype)


def _emit_binary_math(
    builder: AlloOpBuilder,
    lhs: AlloValue,
    rhs: AlloValue,
    acc,
    result_dtype,
    op_cls,
    op_name: str,
    linalg_op_cls=None,
):
    def build_fn(lhs_arg, rhs_arg):
        return op_cls(
            lhs_arg.handle,
            rhs_arg.handle,
            ip=builder.save_insertion_point(),
            loc=builder.get_loc(),
        ).result

    if (
        isinstance(lhs.type, ShapedType)
        or isinstance(rhs.type, ShapedType)
        or not is_default_acc(acc)
    ):
        return emit_linalg_binary(
            builder,
            lhs,
            rhs,
            result_dtype,
            build_fn,
            named_op_cls=linalg_op_cls,
            acc=acc,
            op_name=op_name,
        )
    return AlloValue(build_fn(lhs, rhs), result_dtype)


def _materialize_pow_operands(builder: AlloOpBuilder, base, exponent, acc):
    if isinstance(base, ConstexprValue) and isinstance(exponent, ConstexprValue):
        if not is_default_acc(acc):
            assert isinstance(acc, AlloValue)
            base = builder.cast(base, acc.dtype)
            exp_dtype = f32 if isinstance(exponent.value, float) else i32
            exponent = builder.cast(exponent, exp_dtype)
        return base, exponent

    if isinstance(base, ConstexprValue):
        assert isinstance(exponent, AlloValue)
        base_dtype = (
            f32
            if isinstance(base.value, float) and not exponent.dtype.is_float()
            else exponent.dtype
        )
        base = builder.cast(base, base_dtype)

    if isinstance(exponent, ConstexprValue):
        assert isinstance(base, AlloValue)
        if isinstance(exponent.value, float):
            exp_dtype = base.dtype if base.dtype.is_float() else f32
        elif isinstance(exponent.value, int) and not isinstance(exponent.value, bool):
            exp_dtype = base.dtype if not base.dtype.is_float() else i32
        else:
            exp_dtype = base.dtype
        exponent = builder.cast(exponent, exp_dtype)

    return base, exponent


def _lower_pow(builder: AlloOpBuilder, base, exponent, acc):
    base, exponent = _materialize_pow_operands(builder, base, exponent, acc)
    assert isinstance(base, AlloValue) and isinstance(exponent, AlloValue)

    if exponent.dtype.is_float():
        result_dtype = builder.get_promoted_dtype_nary(
            "pow", [base.dtype, exponent.dtype]
        )
        base = builder.cast_to_dtype(base, result_dtype)
        exponent = builder.cast_to_dtype(exponent, result_dtype)
        op_cls = math.PowFOp
        linalg_op_cls = linalg.PowFOp
    elif base.dtype.is_float():
        result_dtype = base.dtype
        op_cls = math.FPowIOp
        linalg_op_cls = None
    else:
        result_dtype = builder.get_promoted_dtype_nary(
            "pow", [base.dtype, exponent.dtype]
        )
        base = builder.cast_to_dtype(base, result_dtype)
        exponent = builder.cast_to_dtype(exponent, result_dtype)
        op_cls = math.IPowIOp
        linalg_op_cls = None

    return _emit_binary_math(
        builder, base, exponent, acc, result_dtype, op_cls, "pow", linalg_op_cls
    )


@operator
def exp(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@exp.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 0):
        return ConstexprValue(1)
    return _fold_unary(value, py_math.exp)


@exp.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "exp", math.ExpOp, linalg.ExpOp)


@operator
def exp2(exponent, acc=ConstexprValue(None)):
    operator_body_unreachable()


@exp2.fold
def _(exponent, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(exponent, 0):
        return ConstexprValue(1)
    return _fold_unary(exponent, py_math.exp2)


@exp2.build
def _(builder: AlloOpBuilder, exponent, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, exponent, acc, "exp2", math.Exp2Op)


@operator
def log(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@log.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 1):
        return ConstexprValue(0)
    return _fold_unary(value, py_math.log)


@log.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "log", math.LogOp, linalg.LogOp)


@operator
def log2(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@log2.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 1):
        return ConstexprValue(0)
    return _fold_unary(value, py_math.log2)


@log2.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "log2", math.Log2Op)


@operator
def abs(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@abs.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_unary(value, builtins.abs)


@abs.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    operand = _materialize_unary_operand(builder, value, acc)
    result_dtype = builder.get_promoted_dtype_nary("abs", [operand.dtype])
    operand = builder.cast_to_dtype(operand, result_dtype)

    def build_fn(inner):
        op_cls = math.AbsFOp if inner.dtype.is_float() else math.AbsIOp
        return op_cls(
            inner.handle, ip=builder.save_insertion_point(), loc=builder.get_loc()
        ).result

    if isinstance(operand.type, ShapedType) or not is_default_acc(acc):
        return emit_linalg_unary(
            builder,
            operand,
            result_dtype,
            build_fn,
            acc=acc,
            op_name="abs",
        )
    return AlloValue(build_fn(operand), result_dtype)


@operator
def pow(base, exponent, acc=ConstexprValue(None)):
    operator_body_unreachable()


@pow.fold
def _(base, exponent, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    folded = _fold_binary(base, exponent, lambda lhs, rhs: lhs**rhs)
    if folded is not NO_FOLD:
        return folded
    if _is_const(exponent, 0):
        return ConstexprValue(1)
    if _is_const(exponent, 1):
        return base
    if _is_const(base, 1):
        return ConstexprValue(1)
    return NO_FOLD


@pow.build
def _(builder: AlloOpBuilder, base, exponent, acc=ConstexprValue(None)):
    return _lower_pow(builder, base, exponent, acc)


@operator
def sqrt(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@sqrt.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 0) or _is_const(value, 1):
        return value
    return _fold_unary(value, py_math.sqrt)


@sqrt.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "sqrt", math.SqrtOp, linalg.SqrtOp)


@operator
def rsqrt(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@rsqrt.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 1):
        return ConstexprValue(1)
    return _fold_unary(value, lambda operand: 1 / py_math.sqrt(operand))


@rsqrt.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "rsqrt", math.RsqrtOp, linalg.RsqrtOp)


@operator
def sin(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@sin.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 0):
        return ConstexprValue(0)
    return _fold_unary(value, py_math.sin)


@sin.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "sin", math.SinOp)


@operator
def cos(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@cos.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 0):
        return ConstexprValue(1)
    return _fold_unary(value, py_math.cos)


@cos.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "cos", math.CosOp)


@operator
def tan(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@tan.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 0):
        return ConstexprValue(0)
    return _fold_unary(value, py_math.tan)


@tan.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "tan", math.TanOp)


@operator
def floor(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@floor.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_unary(value, py_math.floor)


@floor.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "floor", math.FloorOp, linalg.FloorOp)


@operator
def ceil(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@ceil.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    return _fold_unary(value, py_math.ceil)


@ceil.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "ceil", math.CeilOp)


@operator
def erf(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@erf.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 0):
        return ConstexprValue(0)
    return _fold_unary(value, py_math.erf)


@erf.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "erf", math.ErfOp)


@operator
def tanh(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@tanh.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 0):
        return ConstexprValue(0)
    return _fold_unary(value, py_math.tanh)


@tanh.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "tanh", math.TanhOp)


@operator
def sinh(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@sinh.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 0):
        return ConstexprValue(0)
    return _fold_unary(value, py_math.sinh)


@sinh.build
def _(builder: AlloOpBuilder, value, acc=ConstexprValue(None)):
    return _emit_unary_math(builder, value, acc, "sinh", math.SinhOp)


@operator
def cosh(value, acc=ConstexprValue(None)):
    operator_body_unreachable()


@cosh.fold
def _(value, acc=ConstexprValue(None)):
    if not _fold_enabled(acc):
        return NO_FOLD
    if _is_const(value, 0):
        return ConstexprValue(1)
    return _fold_unary(value, py_math.cosh)


@operator
def fma(a, b, c):
    operator_body_unreachable()


@fma.fold
def _(a, b, c):
    if not _fold_enabled(c):
        return NO_FOLD
    if _is_const(a, 0) or _is_const(b, 0):
        return c
    if _is_const(c, 0):
        return ConstexprValue(a.value * b.value)
    if _is_const(a, 1):
        return ConstexprValue(b.value + c.value)
    if _is_const(b, 1):
        return ConstexprValue(a.value + c.value)
    if (
        isinstance(a, ConstexprValue)
        and isinstance(b, ConstexprValue)
        and isinstance(c, ConstexprValue)
    ):
        return ConstexprValue(a.value * b.value + c.value)
    return NO_FOLD


@fma.build
def _(builder: AlloOpBuilder, a, b, c):
    def is_const_or_float(value):
        return isinstance(value, ConstexprValue) or (
            isinstance(value, AlloValue) and value.dtype.is_float()
        )

    if any(not is_const_or_float(v) for v in (a, b, c)):
        return builder.compile_error("fma operands must be float or constexpr")

    # all AlloValue must be the same float type
    float_types = {v.dtype for v in (a, b, c) if isinstance(v, AlloValue)}
    if len(float_types) > 1:
        return builder.compile_error("fma operands must have the same float type")
    # if len(float_types) == 0:
    # all operands are constexpr, should be folded
    result_dtype = float_types.pop()
    handle = math.FmaOp(
        builder.cast_to_dtype(a, result_dtype).handle,
        builder.cast_to_dtype(b, result_dtype).handle,
        builder.cast_to_dtype(c, result_dtype).handle,
        ip=builder.save_insertion_point(),
        loc=builder.get_loc(),
    ).result
    return AlloValue(handle, result_dtype)
