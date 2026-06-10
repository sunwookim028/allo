# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the ``allo.buffer`` + ``allo.define`` catalog from an ``ISA``.

This is the single place that knows the ``allo`` dialect. It traces each
instruction's access + compute regions (pure Python) and emits:

- one ``allo.buffer`` per declared buffer,
- one ``allo.define`` per instruction, whose ``addr`` region materializes the
  access-pattern DAG into ``allo.patterns.*`` and whose ``semantics`` region
  materializes the compute DAG into value-semantics TOSA ops.

Compute block-arg tensor types are inferred from the access patterns' visible
shapes, so the IR satisfies the ``define`` verifier without annotations.
"""

from __future__ import annotations

from ..._mlir import ir
from ..._mlir.ir import InsertionPoint, Location, Module
from ..._mlir.dialects import allo as allo_d, arith, func as func_d, tosa

from . import primitive
from .core import (
    ISA,
    IndexExpr,
    InstructionSpec,
    PatternExpr,
    TensorProxy,
    arity,
    trace_instruction,
)

K_DYNAMIC = ir.ShapedType.get_dynamic_size()


def encode_index_list(items):
    """Split ``[int | ir.Value]`` into ``(dynamic_values, static_attr)`` using
    the kDynamic sentinel — the ``custom<DynamicIndexList>`` encoding."""
    static, dynamic = [], []
    for it in items:
        if isinstance(it, int):
            static.append(it)
        else:
            static.append(K_DYNAMIC)
            dynamic.append(it)
    return dynamic, static


def _val(x) -> ir.Value:
    """Normalize a dialect-builder return into a single ``ir.Value``."""
    if isinstance(x, ir.Value):
        return x
    if isinstance(x, (ir.Operation, ir.OpView)):
        return x.results[0]
    return x[0]  # OpResultList


# ==========================================================================#
# Catalog construction
# ==========================================================================#


def build_catalog(isa: ISA) -> Module:
    context = ir.Context()
    allo_d.register_dialect(context)
    with context, Location.unknown(context):
        module = Module.create()
        with InsertionPoint(module.body):
            emit_catalog(context, isa)
        assert module.operation.verify(), "generated catalog failed verification"
    return module


def emit_catalog(context: ir.Context, isa: ISA):
    """Emit ``allo.buffer`` + ``allo.define`` ops at the current insertion point
    (assumed to be a module body)."""
    for buf in isa.buffers.values():
        allo_d.DeclareBufferOp(
            buf.name, buf.size, ir.TypeAttr.get(buf.kind.materialize(context))
        )
    for spec in isa.instructions:
        _build_define(context, spec)


def _build_define(context: ir.Context, spec: InstructionSpec):
    n_buffers = len(spec.buffers)
    n_addr = arity(spec.access_fn)
    patterns, arg_shapes, results = trace_instruction(spec)

    # --- infer compute arg tensor types from visible shapes ---
    # A symbolic (parametric) dim becomes a dynamic ``?``; the emit supplies the
    # concrete size, and the lowering re-infers the real shape from the slice.
    sem_arg_types = []
    for buf, shape in zip(spec.buffers, arg_shapes):
        dims = [d if isinstance(d, int) else K_DYNAMIC for d in shape]
        sem_arg_types.append(
            ir.RankedTensorType.get(dims, buf.kind.dtype.materialize(context))
        )
    assert len(results) == len(spec.destinations), (
        f"{spec.name}: compute must yield {len(spec.destinations)} value(s), "
        f"got {len(results)}"
    )

    # --- build the op ---
    define = allo_d.DefineOp(
        spec.name,
        [b.name for b in spec.sources],
        [b.name for b in spec.destinations],
    )
    pattern_ty = ir.Type.parse("!allo.pattern", context)
    index_ty = ir.IndexType.get(context)

    access_block = define.access.blocks.append(*([index_ty] * n_addr))
    with InsertionPoint(access_block):
        env = {i: access_block.arguments[i] for i in range(n_addr)}
        tokens = [_emit_pattern(p, env, pattern_ty) for p in patterns]
        allo_d.YieldOp(tokens)

    sem_block = define.semantics.blocks.append(*sem_arg_types)
    with InsertionPoint(sem_block):
        venv = {i: sem_block.arguments[i] for i in range(n_buffers)}
        # Compute is value-semantics TOSA; the yielded value is written into the
        # destination buffer by the lowering's writeback (no DPS init needed).
        outs = [_emit_value(r, venv, context) for r in results]
        allo_d.YieldOp(outs)


# ==========================================================================#
# Access region emission
# ==========================================================================#


def _emit_index(e: IndexExpr, env) -> ir.Value:
    if e.kind == "param":
        return env[e.param_index]
    if e.kind == "const":
        return arith.constant(ir.IndexType.get(), e.value)
    if e.kind == "add":
        return _val(arith.AddIOp(_emit_index(e.lhs, env), _emit_index(e.rhs, env)))
    if e.kind == "mul":
        return _val(arith.MulIOp(_emit_index(e.lhs, env), _emit_index(e.rhs, env)))
    raise NotImplementedError(f"index expr '{e.kind}'")


def _resolve_item(x, env):
    """An int | IndexExpr -> int (static) or ir.Value (dynamic)."""
    if isinstance(x, int):
        return x
    s = x.static_int()
    return s if s is not None else _emit_index(x, env)


def _encode(items, env):
    return encode_index_list([_resolve_item(x, env) for x in items])


def _reassoc_attr(reassociation) -> ir.ArrayAttr:
    i64 = ir.IntegerType.get_signless(64)
    return ir.ArrayAttr.get(
        [
            ir.ArrayAttr.get([ir.IntegerAttr.get(i64, i) for i in group])
            for group in reassociation
        ]
    )


def _emit_pattern(p: PatternExpr, env, pattern_ty) -> ir.Value:
    if p.kind == "strided":
        b_dyn, b_st = _encode(p.basis, env)
        c_dyn, c_st = _encode(p.counts, env)
        s_dyn, s_st = _encode(p.strides, env)
        op = allo_d.StridedOp(
            b_dyn, c_dyn, s_dyn, b_st, c_st, s_st, results=[pattern_ty]
        )
        return op.result
    if p.kind == "expand":
        src = _emit_pattern(p.source, env, pattern_ty)
        os_dyn, os_st = _encode(p.output_shape, env)
        op = allo_d.ExpandShapeOp(
            src, _reassoc_attr(p.reassociation), os_dyn, os_st, results=[pattern_ty]
        )
        return op.result
    if p.kind == "collapse":
        src = _emit_pattern(p.source, env, pattern_ty)
        op = allo_d.CollapseShapeOp(
            src, _reassoc_attr(p.reassociation), results=[pattern_ty]
        )
        return op.result
    raise NotImplementedError(f"access pattern '{p.kind}'")


# ==========================================================================#
# Compute region emission
# ==========================================================================#


def _emit_value(r: TensorProxy, venv, context) -> ir.Value:
    """Recursively materialize a compute value as a value-semantics TOSA DAG.

    Leaves (``arg``) are buffer block args; interior nodes are prims. The few ops
    with irregular shape/codegen are handled explicitly; everything else dispatches
    by ``primitive`` category (``_CATEGORY_EMIT``), so a new same-category prim needs no
    change here. There is no DPS init / ``tensor.empty`` — that is the whole point
    of TOSA semantics, and it lets multi-node instructions chain naturally."""
    if r.kind == "arg":
        return venv[r.buffer_index]
    if r.kind == "identity":
        return _emit_value(r.args[0], venv, context)
    if r.kind == "relu":
        return _emit_relu(_emit_value(r.args[0], venv, context))
    if r.kind == "transpose":
        return _emit_transpose(_emit_value(r.args[0], venv, context), r.permutation)
    if r.kind == "matmul":
        a = _emit_value(r.args[0], venv, context)
        b = _emit_value(r.args[1], venv, context)
        return _emit_matmul(a, b)
    if r.kind == "reverse":
        x = _emit_value(r.args[0], venv, context)
        return tosa.ReverseOp(x.type, x, r.axis).result
    if r.kind in ("conv2d", "depthwise_conv2d"):
        ops = [_emit_value(a, venv, context) for a in r.args]
        return _emit_conv(r, ops, context)
    if r.kind in ("max_pool2d", "avg_pool2d"):
        x = _emit_value(r.args[0], venv, context)
        return _emit_pool(r, x, context)
    spec = primitive.REGISTRY[r.kind]
    operands = [_emit_value(a, venv, context) for a in r.args]
    return _CATEGORY_EMIT[spec.category](r, spec, operands, context)


# --- category emitters: one per TOSA calling convention (see primitive categories) - #


def _tosa_op(spec):
    return getattr(tosa, spec.tosa_class())


def _with_elt(ty: ir.Type, elt: ir.Type) -> ir.Type:
    """``ty`` with its element type replaced (same shape) — compare/cast results."""
    return ir.RankedTensorType.get(ir.RankedTensorType(ty).shape, elt)


def _bcast_result_type(operands) -> ir.Type:
    """The TOSA broadcast result type of elementwise ``operands`` (equal rank, each
    dim equal or one side 1): per dim a static size > 1 wins, else dynamic if any
    operand is dynamic there, else 1. Element type follows the first operand. (For a
    single operand or equal shapes this is just that operand's type.)"""
    tys = [ir.RankedTensorType(o.type) for o in operands]
    shape = []
    for dims in zip(*[t.shape for t in tys]):
        statics = [d for d in dims if d >= 0 and d != 1]
        if statics:
            shape.append(statics[0])
        elif any(d < 0 for d in dims):
            shape.append(K_DYNAMIC)
        else:
            shape.append(1)
    return ir.RankedTensorType.get(shape, tys[0].element_type)


def _reduce_type(ty: ir.Type, axis: int) -> ir.Type:
    t = ir.RankedTensorType(ty)
    shape = list(t.shape)
    shape[axis] = 1  # TOSA reduce keeps the reduced dim
    return ir.RankedTensorType.get(shape, t.element_type)


def _zero_point(value: ir.Value) -> ir.Value:
    """A ``tensor<1xelt>`` zero (the per-operand zero-point tosa.negate requires)."""
    elt = ir.RankedTensorType(value.type).element_type
    zp_ty = ir.RankedTensorType.get([1], elt)
    return tosa.ConstOp(
        ir.DenseElementsAttr.get_splat(zp_ty, ir.FloatAttr.get(elt, 0.0))
    ).result


def _emit_elementwise(r, spec, operands, context) -> ir.Value:
    """UNARY / BINARY: ``Op(out, *operands)`` with out == the broadcast of inputs."""
    return _tosa_op(spec)(_bcast_result_type(operands), *operands).result


def _emit_unary_zp(r, spec, operands, context) -> ir.Value:
    x = operands[0]
    zp = _zero_point(x)
    return _tosa_op(spec)(x.type, x, zp, zp).result


def _emit_binary_shift(r, spec, operands, context) -> ir.Value:
    a, b = operands
    return _tosa_op(spec)(_bcast_result_type([a, b]), a, b, _mul_shift()).result


def _emit_compare(r, spec, operands, context) -> ir.Value:
    a, b = operands
    out_ty = _with_elt(_bcast_result_type([a, b]), r.dtype.materialize(context))  # bool
    if spec.tag == "equal":
        return _tosa_op(spec)(a, b, results=[out_ty]).result  # no positional output
    return _tosa_op(spec)(out_ty, a, b).result


def _emit_select(r, spec, operands, context) -> ir.Value:
    cond, a, b = operands
    return _tosa_op(spec)(_bcast_result_type([a, b]), cond, a, b).result  # out == a∼b


def _emit_reduce(r, spec, operands, context) -> ir.Value:
    x = operands[0]
    out_ty = _reduce_type(x.type, r.axis)
    return _tosa_op(spec)(x, r.axis, results=[out_ty]).result


def _emit_cast(r, spec, operands, context) -> ir.Value:
    x = operands[0]
    out_ty = _with_elt(x.type, r.dtype.materialize(context))  # target dtype
    return _tosa_op(spec)(out_ty, x).result


# --- contraction / conv family (bespoke; out type from the inferred result shape) ---


def _result_type(r, context) -> ir.Type:
    return ir.RankedTensorType.get(list(r.shape), r.dtype.materialize(context))


def _i64(xs) -> ir.Attribute:
    return ir.DenseI64ArrayAttr.get(list(xs))


def _acc_type(value) -> ir.Attribute:
    return ir.TypeAttr.get(ir.RankedTensorType(value.type).element_type)


def _emit_conv(r, operands, context) -> ir.Value:
    """conv2d / depthwise_conv2d: ``(input, weight, bias)`` + zero zero-points."""
    inp, weight, bias = operands
    cls = tosa.Conv2DOp if r.kind == "conv2d" else tosa.DepthwiseConv2DOp
    a = r.attrs
    return cls(
        _result_type(r, context),
        inp,
        weight,
        bias,
        _zero_point(inp),
        _zero_point(weight),
        pad=_i64(a["pad"]),
        stride=_i64(a["stride"]),
        dilation=_i64(a["dilation"]),
        acc_type=_acc_type(inp),
    ).result


def _emit_pool(r, x, context) -> ir.Value:
    a = r.attrs
    out_ty = _result_type(r, context)
    if r.kind == "max_pool2d":
        return tosa.MaxPool2dOp(
            out_ty,
            x,
            kernel=_i64(a["kernel"]),
            stride=_i64(a["stride"]),
            pad=_i64(a["pad"]),
        ).result
    return tosa.AvgPool2dOp(  # avg_pool2d carries input/output zero-points + acc_type
        out_ty,
        x,
        _zero_point(x),
        _zero_point(x),
        kernel=_i64(a["kernel"]),
        stride=_i64(a["stride"]),
        pad=_i64(a["pad"]),
        acc_type=_acc_type(x),
    ).result


_CATEGORY_EMIT = {
    primitive.UNARY: _emit_elementwise,
    primitive.BINARY: _emit_elementwise,
    primitive.UNARY_ZP: _emit_unary_zp,
    primitive.BINARY_SHIFT: _emit_binary_shift,
    primitive.COMPARE: _emit_compare,
    primitive.SELECT: _emit_select,
    primitive.REDUCE: _emit_reduce,
    primitive.CAST: _emit_cast,
}


def _emit_transpose(a: ir.Value, permutation) -> ir.Value:
    """``primitive.transpose`` -> ``tosa.transpose`` (perms is an ``array<i32>`` attr)."""
    at = ir.RankedTensorType(a.type)
    out_ty = ir.RankedTensorType.get(
        [at.shape[p] for p in permutation], at.element_type
    )
    return tosa.TransposeOp(out_ty, a, list(permutation)).result


def _mul_shift() -> ir.Value:
    """The ``tensor<1xi8>`` zero-shift operand tosa.mul requires (no shift)."""
    i8 = ir.IntegerType.get_signless(8)
    shift_ty = ir.RankedTensorType.get([1], i8)
    splat = ir.DenseElementsAttr.get_splat(shift_ty, ir.IntegerAttr.get(i8, 0))
    return tosa.ConstOp(splat).result


def _emit_relu(a: ir.Value) -> ir.Value:
    elt = ir.RankedTensorType(a.type).element_type
    lo = ir.FloatAttr.get(elt, 0.0)
    hi = ir.FloatAttr.get(elt, float("inf"))
    return tosa.ClampOp(a.type, a, lo, hi).result


def _emit_matmul(a: ir.Value, b: ir.Value) -> ir.Value:
    """``primitive.matmul`` -> ``tosa.matmul`` (batched). Operands are already 3-D
    (``BxMxK`` / ``BxKxN``) from the access region, so this is a direct lowering
    with zero zero-points; result is ``BxMxN``."""
    at, bt = ir.RankedTensorType(a.type), ir.RankedTensorType(b.type)
    bdim, m, _ = at.shape
    n = bt.shape[2]
    elt = at.element_type
    zp_ty = ir.RankedTensorType.get([1], elt)
    zp = tosa.ConstOp(ir.DenseElementsAttr.get_splat(zp_ty, ir.FloatAttr.get(elt, 0.0)))
    out_ty = ir.RankedTensorType.get([bdim, m, n], elt)
    return tosa.MatMulOp(out_ty, a, b, zp.result, zp.result).result


# ==========================================================================#
# Program (func @main) construction for the oracle simulator
# ==========================================================================#

INSPECT_PREFIX = "__inspect_"


def buffer_memref_shape(buf) -> list[int]:
    """The shape of a buffer's lowered ``memref.global``. Mirrors
    ``ConvertDeclareBufferOpPattern``: the slot dim is dropped when size == 1."""
    leading = [] if buf.size == 1 else [buf.size]
    return leading + list(buf.kind.element_shape)


def build_main(context: ir.Context, isa: ISA, program):
    """Build ``func @main(out0, out1, ...)`` of ``allo.emit`` ops, with a
    placeholder ``call @__inspect_k`` anchor at each inspect point. Each output
    arg captures the *whole* inspected buffer (the slice is applied host-side),
    filled post-lowering by replacing the anchor with a get_global + copy."""
    inspects = program.inspects
    out_types = [
        ir.MemRefType.get(
            buffer_memref_shape(ins.buffer), ins.buffer.kind.dtype.materialize(context)
        )
        for ins in inspects
    ]
    main = func_d.FuncOp("main", ir.FunctionType.get(out_types, []))
    main.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(context)
    block = main.add_entry_block()
    with InsertionPoint(block):
        k = 0
        for kind, rec in program.steps:
            if kind == "emit":
                a_dyn, a_st = encode_index_list(rec.addr)
                c_dyn, c_st = encode_index_list(rec.compute)
                allo_d.EmitOp(rec.name, a_dyn, c_dyn, a_st, c_st)
            else:
                func_d.CallOp([], f"{INSPECT_PREFIX}{k}", [])
                k += 1
        func_d.ReturnOp([])
    for k in range(len(inspects)):
        func_d.FuncOp(
            f"{INSPECT_PREFIX}{k}", ir.FunctionType.get([], []), visibility="private"
        )
    return out_types
