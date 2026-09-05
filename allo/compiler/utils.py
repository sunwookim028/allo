# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from collections.abc import Sequence

from .._mlir.ir import Context, FunctionType
from ..lang.core import TypeBase, ConstexprType, StreamType, ShapedType, DType


def generate_function_type(
    context: Context, arg_types: Sequence[TypeBase], res_types: Sequence[TypeBase]
) -> FunctionType:
    mlir_arg_types = []
    for ty in arg_types:
        if isinstance(ty, ConstexprType):
            continue
        mlir_arg_types.append(ty.materialize(context))
    mlir_res_types = []
    for ty in res_types:
        if isinstance(ty, StreamType):
            raise TypeError("Stream is not allowed as a kernel return type.")
        if isinstance(ty, ConstexprType):
            continue
        mlir_res_types.append(ty.materialize(context))
    return FunctionType.get(mlir_arg_types, mlir_res_types, context)


def generate_signedness_marker(
    arg_types: Sequence[TypeBase], res_types: Sequence[TypeBase]
) -> str:
    """Build the ``allo.signed`` marker: one char per MLIR func operand then
    result, in order. 's' = signed integer, 'u' = unsigned integer, 'x' =
    non-integer. The filtering mirrors ``generate_function_type`` so the marker
    length equals the function's operand + result count."""

    def sign_char(ty: TypeBase) -> str:
        if isinstance(ty, ShapedType):
            ty = ty.dtype
        if isinstance(ty, DType):
            if ty.is_int():
                return "s"
            if ty.is_uint():
                return "u"
        if isinstance(ty, StreamType):
            return sign_char(ty.base_type)
        return "x"

    chars = [sign_char(ty) for ty in arg_types if not isinstance(ty, ConstexprType)]
    chars += [
        sign_char(ty)
        for ty in res_types
        if not isinstance(ty, (ConstexprType, StreamType))
    ]
    return "".join(chars)


def global_symbol(func_name: str, var_id: str, kind: str, node: ast.AST) -> str:
    """Canonical name for a compiler-emitted module global or helper kernel,
    shared by stateful variables (``kind="stateful"``), list/NumPy-initialized
    constants (``kind="const"``) and bufferize copy kernels (``kind="bufferize"``).
    Keyed on the source declaration -- enclosing kernel, variable, line and column
    -- so the name is stable and unique: repeated kernel instantiations resolve to
    one symbol, while distinct declarations never collide. The C++ emitter
    sanitizes it into a valid identifier."""
    return f"_allo_{kind}_{func_name}_{var_id}_l{node.lineno}c{node.col_offset}"
