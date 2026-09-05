# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

import ast

import numpy as np

from ..lang.operator import operator
from ..lang.core import (
    APInt,
    DType,
    ShapedType,
    StreamType,
    BufferType,
    index,
    u1,
    ConstexprValue,
    AlloValue,
    torch_dtype_map,
)
from ..compiler.builder import AlloOpBuilder
from .utils import operator_body_unreachable, BitSlice

from .._mlir.ir import (
    Location,
    TypeAttr,
    DenseI32ArrayAttr,
    StringAttr,
    AffineMap,
    AffineDimExpr,
    AffineSymbolExpr,
    AffineConstantExpr,
    IntegerAttr,
    IntegerType,
)
from .._mlir.dialects.allo import (
    KernelOp,
    InvokeOp,
    ReturnOp,
    SIGNED_ATTR_NAME,
)
from .._mlir.dialects.affine import AffineYieldOp


def _normalize_stream_indices(
    builder: AlloOpBuilder, stream_type: StreamType, slices, context: str
):
    assert isinstance(stream_type, StreamType)
    if not isinstance(slices, tuple):
        return builder.compile_error(
            f"{context} indices must be a tuple of scalar index expressions."
        )
    if len(slices) != stream_type.rank:
        return builder.compile_error(
            f"{context} expects {stream_type.rank} indices, got {len(slices)}."
        )
    for idx, dim in zip(slices, stream_type.shape):
        if not isinstance(idx, ConstexprValue):
            continue
        if type(idx.value) is not int:
            return builder.compile_error(
                f"{context} constexpr indices must be integers."
            )
        if idx.value < 0 or idx.value >= dim:
            return builder.compile_error(
                f"{context} index {idx.value} is out of bounds for dimension size {dim}."
            )
    return builder.normalize_indices(
        slices,
        expected_len=stream_type.rank,
        context=context,
    )


def _load_stream_value(builder: AlloOpBuilder, stream: AlloValue, slices):
    assert isinstance(stream.type, StreamType)
    if stream.is_indexed:
        return builder.compile_error(
            "Cannot index a specific stream, Use get() or put(value) on the specific stream."
        )
    indices = _normalize_stream_indices(builder, stream.type, slices, "Stream")
    ref = AlloValue(stream.handle, stream.type)
    ref.indices = tuple(indices)
    return ref


def _bit_slice(builder: AlloOpBuilder, value: AlloValue, slc: BitSlice):
    # The result width ``hi - lo`` must be statically known (the offset may be
    # dynamic); the codegen infers it affinely and leaves ``width`` as ``None``
    # when it is not a compile-time constant.
    if not isinstance(value, AlloValue) or not isinstance(value.dtype, APInt):
        return builder.compile_error(
            "Bit slicing is only supported on signless integer scalars."
        )
    if slc.lo is None or slc.hi is None:
        return builder.compile_error(
            "Bit slice requires explicit lower and upper bounds, e.g. 'x[lo:hi]'."
        )
    if slc.width is None:
        return builder.compile_error(
            "Bit slice width 'hi - lo' must be a compile-time constant; "
            "only the offset may be dynamic."
        )
    if slc.width <= 0:
        return builder.compile_error(
            "Bit slice upper bound must be greater than the lower bound."
        )
    return slc.lo, slc.hi, APInt(slc.width, signed=False)


@operator
def load(lhs, slices):
    operator_body_unreachable()


@load.build
def _(builder: AlloOpBuilder, lhs, slices: BitSlice | tuple):
    if isinstance(lhs, AlloValue) and isinstance(lhs.type, StreamType):
        return _load_stream_value(builder, lhs, slices)

    if isinstance(slices, tuple):
        indices = builder.normalize_indices(slices)
        if isinstance(lhs.type, ShapedType):
            if len(indices) != lhs.type.rank:
                return builder.compile_error(
                    f"Load with tuple indices must have the same number of indices as the rank of the array, got {len(indices)} indices for an array of rank {lhs.type.rank}."
                )
            return builder.create_load(lhs, indices)
        if isinstance(lhs.type, DType):
            if len(indices) != 1:
                return builder.compile_error(
                    f"Bit extraction with tuple indices must have exactly one index for scalar types, got {len(indices)}."
                )
            return builder.create_bit_extract(lhs, indices[0])

    elif isinstance(slices, BitSlice):
        lo, hi, result_dtype = _bit_slice(builder, lhs, slices)
        return builder.create_bit_get_slice(lhs, lo, hi, result_dtype)

    return builder.compile_error(
        f"Unsupported load operation: lhs of type {lhs.type} with indices of type {type(slices)}"
    )


@operator
def store(dst, slices, value):
    operator_body_unreachable()


@store.build
def _(builder: AlloOpBuilder, dst, slices: BitSlice | tuple, value):
    if isinstance(dst, AlloValue) and isinstance(dst.type, StreamType):
        return builder.compile_error(
            "Cannot assign to a stream. Use put(value) on the stream reference."
        )

    if isinstance(slices, tuple):
        indices = builder.normalize_indices(slices)
        if isinstance(dst.type, ShapedType):
            if len(indices) != dst.type.rank:
                builder.compile_error(
                    f"Store with tuple indices must have the same number of indices as the rank of the array, got {len(indices)} indices for an array of rank {dst.type.rank}."
                )
            val = builder.cast(value, dst.dtype)
            return builder.create_store(val, dst, indices)
        if isinstance(dst.type, DType):
            if len(indices) != 1:
                return builder.compile_error(
                    f"Bit insertion with tuple indices must have exactly one index for scalar types, got {len(indices)}."
                )
            val = builder.cast(value, u1)
            return builder.create_bit_insert(val, dst, indices[0])

    elif isinstance(slices, BitSlice):
        lo, hi, slice_dtype = _bit_slice(builder, dst, slices)
        val = builder.cast(value, slice_dtype)
        return builder.create_bit_set_slice(dst, lo, hi, val)

    raise builder.compile_error(
        f"Unsupported store operation: dst of type {dst.type} with indices of type {type(slices)}"
    )


# ---------------------------------------------------------------------------
# bufferize: copy a (strided) slice of a buffer / captured NumPy constant into a
# freshly allocated buffer, via a module-level private copy kernel.
# ---------------------------------------------------------------------------


def _bufferize_var_id(node) -> str:
    """Best-effort readable name for the ``src`` operand of a bufferize call,
    recovered from the call AST. Uniqueness is guaranteed by ``_global_symbol``'s
    line/column suffix, so this is purely cosmetic."""
    if isinstance(node, ast.Call):
        exprs = []
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "bufferize":
            exprs.append(func.value)  # bound form: `src.bufferize(...)`
        exprs.extend(node.args)  # free form: `bufferize(src, ...)`
        for expr in exprs:
            if isinstance(expr, ast.Name) and expr.id != "allo":
                return expr.id
    return "src"


def _enclosing_kernel_name(builder: AlloOpBuilder) -> str:
    """Symbol name of the ``allo.kernel`` enclosing the current insertion point."""
    op = builder.save_insertion_point().block.owner.operation
    while op is not None and op.name != "allo.kernel":
        op = op.parent
    assert op is not None, "bufferize must be built inside a kernel body"
    return StringAttr(op.attributes["sym_name"]).value


def _require_const_int(builder: AlloOpBuilder, value, what: str, dim: int) -> int:
    v = value.value if isinstance(value, ConstexprValue) else value
    if type(v) is not int:
        return builder.compile_error(
            f"bufferize {what} must be a constant integer sequence; dimension "
            f"{dim} is not a compile-time integer constant."
        )
    return v


def _normalize_spec(builder: AlloOpBuilder, spec, rank: int, what: str) -> list:
    if spec is None:
        return [None] * rank
    spec = [spec] if isinstance(spec, (ConstexprValue, AlloValue)) else list(spec)
    if len(spec) != rank:
        return builder.compile_error(
            f"bufferize {what} must match the source rank {rank}, got {len(spec)}."
        )
    return spec


def _resolve_src(builder: AlloOpBuilder, src, node):
    """Return ``(src_value, shape, dtype)`` for a buffer value or a captured NumPy
    constant array (materialized as a module-level constant buffer)."""
    if isinstance(src, AlloValue) and isinstance(src.type, BufferType):
        return src, tuple(src.type.shape), src.type.dtype
    if isinstance(src, np.ndarray):
        dtype_name = str(src.dtype)
        if dtype_name not in torch_dtype_map:
            return builder.compile_error(
                f"bufferize source NumPy array has unsupported dtype '{src.dtype}'."
            )
        dtype = torch_dtype_map[dtype_name]
        shape = tuple(int(d) for d in src.shape)
        from ..compiler.utils import global_symbol

        func_name = _enclosing_kernel_name(builder)
        global_name = global_symbol(func_name, _bufferize_var_id(node), "const", node)
        src_value = builder.make_shaped_constant(
            src.reshape(-1).tolist(), BufferType(shape, dtype), global_name
        )
        return src_value, shape, dtype
    return builder.compile_error(
        "bufferize source must be a buffer value or a captured NumPy constant "
        f"array, got '{type(src).__name__}'."
    )


@operator(cls=AlloValue)
def bufferize(src, offsets, sizes, strides):
    operator_body_unreachable()


@bufferize.build
def _(builder: AlloOpBuilder, src, offsets=None, sizes=None, strides=None):
    node = builder.curr_node
    ctx = builder.context
    src_value, src_shape, dtype = _resolve_src(builder, src, node)
    rank = len(src_shape)

    offs = _normalize_spec(builder, offsets, rank, "offsets")
    szs = _normalize_spec(builder, sizes, rank, "sizes")
    strs = _normalize_spec(builder, strides, rank, "strides")

    # Per-dim slice resolution. `sizes`/`strides` must be static so the result
    # buffer has a static shape and the affine.for uses a constant step; `offsets`
    # may be dynamic and are threaded into the copy kernel as affine symbols.
    result_shape = []
    load_exprs = []
    dyn_offsets = []  # AlloValue(index), one per dynamic-offset dim, in order
    for d in range(rank):
        stride = (
            1 if strs[d] is None else _require_const_int(builder, strs[d], "strides", d)
        )
        if stride < 1:
            return builder.compile_error(
                f"bufferize strides must be >= 1; dimension {d} has stride {stride}."
            )
        size = (
            src_shape[d]
            if szs[d] is None
            else _require_const_int(builder, szs[d], "sizes", d)
        )
        if size < 1:
            return builder.compile_error(
                f"bufferize sizes must be >= 1; dimension {d} has size {size}."
            )
        span = (size - 1) * stride
        off_item = offs[d]
        if off_item is None or isinstance(off_item, ConstexprValue):
            off = (
                0
                if off_item is None
                else _require_const_int(builder, off_item, "offsets", d)
            )
            if off < 0:
                return builder.compile_error(
                    f"bufferize offsets must be >= 0; dimension {d} has offset {off}."
                )
            if off + span >= src_shape[d]:
                return builder.compile_error(
                    f"bufferize slice out of bounds on dimension {d}: offset {off} + "
                    f"(size {size} - 1) * stride {stride} = {off + span} >= source "
                    f"size {src_shape[d]}."
                )
            base_expr = AffineConstantExpr.get(off)
        elif isinstance(off_item, AlloValue):
            if span >= src_shape[d]:
                return builder.compile_error(
                    f"bufferize slice window exceeds source dimension {d}: (size "
                    f"{size} - 1) * stride {stride} = {span} >= source size "
                    f"{src_shape[d]}."
                )
            base_expr = AffineSymbolExpr.get(len(dyn_offsets))
            dyn_offsets.append(builder.cast(off_item, index))
        else:
            return builder.compile_error(
                f"bufferize offsets dimension {d} must be a constant or a runtime "
                "index value."
            )
        result_shape.append(size)
        load_exprs.append(
            base_expr + AffineDimExpr.get(d) * AffineConstantExpr.get(stride)
        )

    result_type = BufferType(tuple(result_shape), dtype)
    load_map = AffineMap.get(rank, len(dyn_offsets), load_exprs, ctx)
    store_map = AffineMap.get_identity(rank, ctx)

    from ..compiler.utils import (
        global_symbol,
        generate_function_type,
        generate_signedness_marker,
    )

    kernel_name = global_symbol(
        _enclosing_kernel_name(builder), _bufferize_var_id(node), "bufferize", node
    )[
        1:
    ]  # remove the leading '_' for the kernel name

    call_ip, call_loc = builder.get_insertion_point_and_loc()

    # Destination buffer, allocated at the call site.
    dst = builder.make_buffer(result_type)

    # Build the private copy kernel at module level: (%dst, %src, %off0, ...).
    arg_types = [result_type, src_value.type] + [index] * len(dyn_offsets)
    fn_ty = generate_function_type(ctx, arg_types, [])
    builder.set_insertion_point_to_end(builder.module.body)
    kernel_op = KernelOp(
        kernel_name,
        TypeAttr.get(fn_ty),
        DenseI32ArrayAttr.get([]),
        sym_visibility="private",
        ip=builder.save_insertion_point(),
        loc=call_loc,
    )
    kernel_op.operation.attributes[SIGNED_ATTR_NAME] = builder.get_string_attr(
        generate_signedness_marker(arg_types, [])
    )
    arg_names = ["dst", "src"] + [f"off{j}" for j in range(len(dyn_offsets))]
    arg_locs = [Location.name(nm, Location.unknown(ctx)) for nm in arg_names]
    entry = kernel_op.regions[0].blocks.append(*fn_ty.inputs, arg_locs=arg_locs)
    dst_arg = AlloValue(entry.arguments[0], result_type)
    src_arg = AlloValue(entry.arguments[1], src_value.type)
    off_args = [
        AlloValue(entry.arguments[2 + j], index) for j in range(len(dyn_offsets))
    ]

    # Nested affine.for over the destination: dst[i...] = src[off + i*stride...].
    builder.set_insertion_point_to_start(entry)
    zero_map = AffineMap.get(0, 0, [AffineConstantExpr.get(0)], ctx)
    for_ops, ivs = [], []
    for d in range(rank):
        ub_map = AffineMap.get(0, 0, [AffineConstantExpr.get(result_shape[d])], ctx)
        for_op = builder.create_affine_for(
            zero_map,
            [],
            ub_map,
            [],
            1,
            [],
            arg_locs=[Location.name(f"i{d}", Location.unknown(ctx))],
        )
        # auto pipeline the innermost loop
        if d == rank - 1:
            i64_ty = IntegerType.get(64, context=ctx)
            for_op.operation.attributes["allo.pipeline.ii"] = IntegerAttr.get(i64_ty, 1)
        for_ops.append(for_op)
        ivs.append(AlloValue(for_op.induction_variable, index))
        builder.set_insertion_point_to_start(for_op.body)

    value = builder.create_affine_load(src_arg, load_map, ivs + off_args)
    builder.create_affine_store(value, dst_arg, store_map, ivs)

    for for_op in for_ops:
        builder.set_insertion_point_to_end(for_op.body)
        AffineYieldOp([], ip=builder.save_insertion_point(), loc=call_loc)
    builder.set_insertion_point_to_end(entry)
    ReturnOp([], ip=builder.save_insertion_point(), loc=call_loc)

    # Invoke the copy kernel at the original call site and hand back the buffer.
    builder.set_insertion_point_and_loc(call_ip, call_loc)
    InvokeOp(
        [],
        kernel_name,
        [dst.handle, src_value.handle] + [o.handle for o in dyn_offsets],
        ip=builder.save_insertion_point(),
        loc=call_loc,
    )
    return dst
