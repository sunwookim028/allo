# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import contextmanager
from enum import Enum
from typing import cast, Literal, NoReturn
from collections.abc import Sequence

from .._mlir import ir
from .._mlir.dialects import arith, tensor, linalg, math, memref
from .._mlir.dialects import affine as affine_d
from .._mlir.dialects import allo as allo_d
from .._mlir.dialects._affine_ops_gen import AffineForOp
from .._mlir.dialects._scf_ops_gen import ForOp

from .errors import CompilationError
from ..lang.core import (
    APInt,
    APFloat,
    DType,
    AlloValue,
    IndexType,
    index,
    u1,
    ShapedType,
    bool as AlloBool,
    TensorType,
    BufferType,
    StreamType,
    ConstexprValue,
    StatefulValue,
    TypeBase,
)
from ..lang.rule import get_type_rules


def _fill_named_linalg_region(op) -> None:
    """Populate a named-linalg op's body region from its registered region
    builder. ``fill_builtin_region`` is emitted by the upstream bindings but is
    absent from their type stubs, so the attribute access is silenced here in
    one place rather than at every call site."""
    linalg.fill_builtin_region(op.operation)  # type: ignore[attr-defined]


class CmpPred(Enum):
    EQ = 0
    NE = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5


# one public method per MLIR op this frontend can emit
# pylint: disable-next=too-many-instance-attributes,too-many-public-methods
class AlloOpBuilder:
    """IR construction helper backed by upstream `allo._mlir`.

    Mirrors `allo.exp.compiler.builder.AlloOpBuilder`'s public API, but is a
    plain Python object holding the insertion point and location.
    """

    def __init__(
        self, context: ir.Context, *, typing_style: Literal["hls", "cpp"] = "hls"
    ):
        self.context = context
        self._ip: ir.InsertionPoint | None = None
        self._loc: ir.Location = ir.Location.unknown(context)
        self.typing_style = typing_style
        self.type_rules = get_type_rules(typing_style)
        self.src: str = ""
        self.file_name: str | None = None
        self.begin_line: int = 1
        self.curr_node: ast.AST | None = None
        self.module: ir.Module | None = None
        # symbol -> memref type of every compiler-emitted module global (stateful
        # variables and list-initialized constants), so multiple instantiations
        # of the same kernel share one global instead of redefining it.
        self._module_globals: dict[str, ir.Type] = {}
        # Cached attr for linalg parallel iterator type.
        self._par_iter = ir.Attribute.parse("#linalg.iterator_type<parallel>", context)

    # =====================================================================
    # Base builder API (insertion point / location / blocks / attrs)
    # =====================================================================

    def set_insertion_point_to_start(self, block: ir.Block):
        self._ip = ir.InsertionPoint.at_block_begin(block)

    def set_insertion_point_to_end(self, block: ir.Block):
        self._ip = ir.InsertionPoint(block)

    def set_insertion_point_after(self, op):
        self._ip = ir.InsertionPoint.after(op)

    def save_insertion_point(self) -> ir.InsertionPoint:
        assert self._ip is not None
        return self._ip

    def restore_insertion_point(self, ip: ir.InsertionPoint):
        self._ip = ip

    @contextmanager
    def at_block_end(self, block: ir.Block):
        """Scope op construction at the end of ``block``, restoring the prior
        insertion point on exit (including when the body raises)."""
        saved = self._ip
        self._ip = ir.InsertionPoint(block)
        try:
            yield
        finally:
            self._ip = saved

    def get_insertion_point_and_loc(self) -> tuple[ir.InsertionPoint, ir.Location]:
        assert self._ip is not None
        return self._ip, self._loc

    def set_insertion_point_and_loc(self, ip, loc):
        self._ip = ip
        self._loc = loc

    def get_loc(self) -> ir.Location:
        return self._loc

    def set_loc(self, loc: ir.Location):
        self._loc = loc

    def set_unknown_loc(self):
        self._loc = ir.Location.unknown(self.context)

    def create_block(self, region: ir.Region, arg_types: Sequence[ir.Type] = ()):
        locs = [self._loc] * len(arg_types)
        return (
            region.blocks.append(*arg_types, arg_locs=locs)
            if arg_types
            else region.blocks.append()
        )

    def get_string_attr(self, value: str) -> ir.Attribute:
        return ir.StringAttr.get(value, self.context)

    def compile_error(self, message: str) -> NoReturn:
        raise CompilationError(
            self.src,
            message,
            self.curr_node,
            file_name=self.file_name,
            begin_line=self.begin_line,
        )

    def _materialize(self, ty: TypeBase) -> ir.Type:
        """Materialize a frontend type into an upstream `allo._mlir` MLIR type."""
        return ty.materialize(self.context)

    #####################
    # Constant Creation
    #####################

    def create_const_float(self, value: float, dtype: DType) -> AlloValue:
        ir_ty = dtype.materialize(self.context)
        return AlloValue(
            # the MLIR op builders take an extension __init__ that pylint cannot see
            # pylint: disable-next=too-many-function-args
            arith.ConstantOp(ir_ty, float(value), ip=self._ip, loc=self._loc).result,
            dtype,
        )

    def _checked_int_const(self, value, dtype: DType) -> int:
        """Coerce a numeric literal to the ``int64_t`` that ``IntegerAttr``'s
        nanobind binding accepts, raising a readable error when it overflows.

        The binding wraps modulo the type width internally, but only takes a C
        ``int64_t``, so a value outside ``[-2**63, 2**63)`` -- e.g. ``int(-1e30)``
        from casting a large float constant to ``i32`` -- would otherwise surface
        as an opaque nanobind ``TypeError``."""
        ival = int(value)
        if not -(1 << 63) <= ival < (1 << 63):
            self.compile_error(f"Integer constant {ival} is out of range for '{dtype}'")
        return ival

    def create_const_int(self, value: int, dtype: DType) -> AlloValue:
        assert dtype.is_int_signless()
        ir_ty = dtype.materialize(self.context)
        return AlloValue(
            # pylint: disable-next=too-many-function-args
            arith.ConstantOp(
                ir_ty, self._checked_int_const(value, dtype), ip=self._ip, loc=self._loc
            ).result,
            dtype,
        )

    def create_const_index(self, value: int) -> AlloValue:
        ir_ty = ir.IndexType.get(self.context)
        return AlloValue(
            # pylint: disable-next=too-many-function-args
            arith.ConstantOp(ir_ty, int(value), ip=self._ip, loc=self._loc).result,
            index,
        )

    def make_scalar(self, value, dtype: DType) -> AlloValue:
        if dtype.is_float():
            return self.create_const_float(value, dtype)
        if dtype.is_int_signless():
            return self.create_const_int(value, dtype)
        if dtype.is_index():
            return self.create_const_index(value)
        return self.compile_error(f"Unsupported scalar type: {dtype}")

    def _fill_shaped_value(self, scalar: AlloValue, shaped: AlloValue) -> AlloValue:
        assert isinstance(shaped.type, ShapedType)
        if isinstance(shaped.type, TensorType):
            res_ty = self._materialize(shaped.type)
            op = linalg.FillOp(
                [res_ty], [scalar.handle], [shaped.handle], ip=self._ip, loc=self._loc
            )
            _fill_named_linalg_region(op)
            return AlloValue(op.results[0], shaped.type)
        assert isinstance(shaped.type, BufferType)
        op = linalg.FillOp(
            [], [scalar.handle], [shaped.handle], ip=self._ip, loc=self._loc
        )
        _fill_named_linalg_region(op)
        return shaped

    def _splat_to_shaped(self, scalar: AlloValue, dst_type: ShapedType) -> AlloValue:
        return self._fill_shaped_value(scalar, self.make_buffer(dst_type))

    def materialize_literal_like(self, value, proxy: AlloValue) -> AlloValue:
        if isinstance(proxy.type, DType):
            return self.make_scalar(value, proxy.type)
        assert isinstance(proxy.type, ShapedType)
        return self._fill_shaped_value(self.make_scalar(value, proxy.type.dtype), proxy)

    def _dense_element_attr(self, value, dtype: DType):
        assert type(value) in (int, float)
        ir_ty = self._materialize(dtype)
        if dtype.is_float():
            return ir.FloatAttr.get(ir_ty, float(value))
        if dtype.is_int_signless() or dtype.is_index():
            return ir.IntegerAttr.get(ir_ty, self._checked_int_const(value, dtype))
        assert False, f"Unsupported dense element type: {dtype}"

    def _dense_initializer(self, values: Sequence[int | float], dtype: DType, shape):
        attr_type = self._materialize(TensorType(shape, dtype))
        elements = [self._dense_element_attr(v, dtype) for v in values]
        return ir.DenseElementsAttr.get(elements, type=attr_type)

    def _define_module_global(self, global_name: str, memref_type, dense_attr):
        """Emit (once) a mutable module-level `memref.global` initialized with
        `dense_attr`, returning a `get_global` result. A name already defined
        reuses its single definition, so kernels instantiated multiple times
        share one global. See ``_global_symbol`` for the naming convention."""
        assert self.module is not None
        if global_name not in self._module_globals:
            ip, loc = self.get_insertion_point_and_loc()
            self.set_insertion_point_to_end(self.module.body)
            memref.GlobalOp(
                global_name,
                ir.TypeAttr.get(memref_type),
                sym_visibility=ir.StringAttr.get("private", self.context),
                initial_value=dense_attr,
                constant=False,
                ip=self._ip,
                loc=self._loc,
            )
            self.set_insertion_point_and_loc(ip, loc)
            self._module_globals[global_name] = memref_type
        return memref.GetGlobalOp(
            memref_type, global_name, ip=self._ip, loc=self._loc
        ).result

    def make_shaped_constant(
        self, values: Sequence[int | float], dst_type: ShapedType, global_name: str
    ) -> AlloValue:
        num_elements = 1
        for dim in dst_type.shape:
            num_elements *= dim
        assert len(values) == num_elements

        dense_attr = self._dense_initializer(values, dst_type.dtype, dst_type.shape)
        if isinstance(dst_type, TensorType):
            return AlloValue(
                # arith.ConstantOp accepts an Attribute value at runtime, but the
                # upstream stub types `value` as int|float|array only.
                # pylint: disable-next=too-many-function-args
                arith.ConstantOp(
                    self._materialize(dst_type), dense_attr, ip=self._ip, loc=self._loc  # type: ignore[arg-type]
                ).result,
                dst_type,
            )

        assert isinstance(dst_type, BufferType)
        result = self._define_module_global(
            global_name, self._materialize(dst_type), dense_attr
        )
        return AlloValue(result, dst_type)

    def make_stateful(
        self, global_name: str, inner: DType | BufferType, values: Sequence[int | float]
    ) -> StatefulValue:
        """Create a persistent variable backed by a mutable module-level global.

        Scalars use a rank-0 `memref<dtype>`; arrays use the declared buffer type.
        `global_name` is keyed on the source declaration, so repeated kernel
        instantiations resolve to the same global (emitted once) and share state.
        `values` is the flat compile-time initializer (one element for scalars).
        """
        dtype = inner if isinstance(inner, DType) else inner.dtype
        shape = () if isinstance(inner, DType) else tuple(inner.shape)
        storage_type = BufferType(shape, dtype)
        dense_attr = self._dense_initializer(values, dtype, shape)
        result = self._define_module_global(
            global_name, self._materialize(storage_type), dense_attr
        )
        return StatefulValue(AlloValue(result, storage_type), inner)

    def store_into_buffer(self, dst: AlloValue, value) -> None:
        """Whole-buffer write: copy a source buffer, or splat a scalar/constexpr."""
        assert isinstance(dst.type, BufferType)
        if isinstance(value, AlloValue) and isinstance(value.type, BufferType):
            memref.CopyOp(value.handle, dst.handle, ip=self._ip, loc=self._loc)
            return
        self._fill_shaped_value(self.cast(value, dst.dtype), dst)

    #####################
    # Stream Creation
    #####################

    def create_stream(
        self, stream_type: StreamType, init: Sequence[int | float] | None = None
    ) -> AlloValue:
        assert isinstance(stream_type, StreamType)
        op = allo_d.StreamCreateOp(
            self._materialize(stream_type), ip=self._ip, loc=self._loc
        )
        # tag the signess for backend codegen
        base = stream_type.base_type
        elem = base.dtype if isinstance(base, ShapedType) else base
        signed = isinstance(elem, DType) and elem.is_int()
        op.operation.attributes[allo_d.SIGNED_ATTR_NAME] = self.get_string_attr(
            "s" if signed else "u"
        )
        # Initial tokens (feedback cycles): an ArrayAttr of typed scalar attrs of
        # the base type, the earliest tokens in the channel history.
        if init:
            if not isinstance(base, DType):
                self.compile_error(
                    "stream initial tokens are only supported for a scalar base type"
                )
            elements = [self._dense_element_attr(v, base) for v in init]
            op.operation.attributes["init"] = ir.ArrayAttr.get(elements)
        return AlloValue(op.result, stream_type)

    #####################
    # Type Casting
    #####################

    def create_index_cast(self, value: ir.Value, dst_type: ir.Type) -> ir.Value:
        return arith.IndexCastOp(dst_type, value, ip=self._ip, loc=self._loc).result

    def create_ext(
        self, value, dst_type, *, signed: bool = True, floating: bool = False
    ):
        assert not (signed and floating), "Cannot be both signed and floating"
        if floating:
            return arith.ExtFOp(dst_type, value, ip=self._ip, loc=self._loc).result
        if signed:
            return arith.ExtSIOp(dst_type, value, ip=self._ip, loc=self._loc).result
        return arith.ExtUIOp(dst_type, value, ip=self._ip, loc=self._loc).result

    def create_trunc(self, value, dst_type, *, floating: bool = False):
        if floating:
            return arith.TruncFOp(dst_type, value, ip=self._ip, loc=self._loc).result
        return arith.TruncIOp(dst_type, value, ip=self._ip, loc=self._loc).result

    def create_itofp(self, value, dst_type, signed: bool = True):
        if signed:
            return arith.SIToFPOp(dst_type, value, ip=self._ip, loc=self._loc).result
        return arith.UIToFPOp(dst_type, value, ip=self._ip, loc=self._loc).result

    def create_fptoi(self, value, dst_type, signed: bool = True):
        if signed:
            return arith.FPToSIOp(dst_type, value, ip=self._ip, loc=self._loc).result
        return arith.FPToUIOp(dst_type, value, ip=self._ip, loc=self._loc).result

    def create_bitcast(self, operand: AlloValue, dst_dtype: DType) -> AlloValue:
        dst_ir_type = self._materialize(dst_dtype)
        return self._emit_elementwise_unary(
            operand,
            dst_dtype,
            lambda value: arith.BitcastOp(
                dst_ir_type, value.handle, ip=self._ip, loc=self._loc
            ).result,
        )

    def scalar_cast(self, src: AlloValue, dst_type: DType) -> AlloValue:
        assert isinstance(src.type, DType)
        src_type = src.type
        value = src.handle
        if src_type == dst_type:
            return src

        dst_ir_type = self._materialize(dst_type)

        if src_type.is_int_signless() and dst_type.is_int_signless():
            if src_type.primitive_width < dst_type.primitive_width:
                return AlloValue(
                    self.create_ext(value, dst_ir_type, signed=src_type.is_int()),
                    dst_type,
                )
            if src_type.primitive_width > dst_type.primitive_width:
                return AlloValue(self.create_trunc(value, dst_ir_type), dst_type)
            return AlloValue(value, dst_type)

        if src_type.is_int_signless() and dst_type.is_index():
            return AlloValue(self.create_index_cast(value, dst_ir_type), dst_type)
        if src_type.is_index() and dst_type.is_int_signless():
            return AlloValue(self.create_index_cast(value, dst_ir_type), dst_type)
        if src_type.is_int_signless() and dst_type.is_float():
            return AlloValue(
                self.create_itofp(value, dst_ir_type, signed=src_type.is_int()),
                dst_type,
            )
        if src_type.is_float() and dst_type.is_int_signless():
            return AlloValue(
                self.create_fptoi(value, dst_ir_type, signed=dst_type.is_int()),
                dst_type,
            )
        if src_type.is_float() and dst_type.is_float():
            if src_type.primitive_width < dst_type.primitive_width:
                return AlloValue(
                    self.create_ext(value, dst_ir_type, signed=False, floating=True),
                    dst_type,
                )
            return AlloValue(
                self.create_trunc(value, dst_ir_type, floating=True), dst_type
            )

        return self.compile_error(
            f"Unsupported scalar cast from {src_type} to {dst_type}"
        )

    @staticmethod
    def _shaped_type_with_dtype(src_type: ShapedType, dtype: DType) -> ShapedType:
        if isinstance(src_type, TensorType):
            return TensorType(src_type.shape, dtype)
        assert isinstance(src_type, BufferType)
        return BufferType(src_type.shape, dtype)

    def shaped_cast(self, src: AlloValue, dst_type: DType) -> AlloValue:
        assert isinstance(src.type, ShapedType) and isinstance(src.dtype, DType)
        if src.dtype == dst_type:
            return src
        new_type = self._shaped_type_with_dtype(src.type, dst_type)
        handle = tensor.CastOp(
            self._materialize(new_type), src.handle, ip=self._ip, loc=self._loc
        ).result
        return AlloValue(handle, new_type)

    def cast_to_dtype(self, src: AlloValue, dtype: DType) -> AlloValue:
        if isinstance(src.type, DType):
            return self.scalar_cast(src, dtype)
        if isinstance(src.type, ShapedType):
            return self.shaped_cast(src, dtype)
        assert False, f"Unsupported value type: {src.type}"

    def _broadcast_shaped_to_type(self, src: AlloValue, dst_type: TensorType):
        assert isinstance(src.type, TensorType)
        shape, indices_src, _ = self.infer_broadcast_shape(
            src.type.shape, dst_type.shape
        )
        if tuple(shape) != tuple(dst_type.shape) or not indices_src:
            return None
        elem = self._materialize(dst_type.dtype)
        init = tensor.EmptyOp(list(shape), elem, ip=self._ip, loc=self._loc).result
        res_ty = self._materialize(dst_type)
        op = linalg.BroadcastOp(
            [res_ty], src.handle, init, indices_src, ip=self._ip, loc=self._loc
        )
        _fill_named_linalg_region(op)
        return AlloValue(op.results[0], dst_type)

    def cast(self, src: AlloValue | ConstexprValue, dst_type: TypeBase) -> AlloValue:
        assert isinstance(dst_type, TypeBase)
        if isinstance(src, ConstexprValue):
            if isinstance(dst_type, DType):
                return self.make_scalar(src.value, dst_type)
            if isinstance(dst_type, ShapedType):
                return self._splat_to_shaped(
                    self.make_scalar(src.value, dst_type.dtype), dst_type
                )
            assert False, f"Unsupported destination type: {dst_type}"

        assert isinstance(src, AlloValue)
        if isinstance(dst_type, StreamType):
            if src.type == dst_type:
                return src
            return self.compile_error(f"Cannot cast from {src.type} to {dst_type}")
        if isinstance(dst_type, DType):
            return self.cast_to_dtype(src, dst_type)
        if isinstance(src.type, DType) and isinstance(dst_type, ShapedType):
            return self._splat_to_shaped(
                self.scalar_cast(src, dst_type.dtype), dst_type
            )
        if isinstance(src.type, TensorType) and isinstance(dst_type, TensorType):
            if tuple(src.type.shape) == tuple(dst_type.shape):
                return self.shaped_cast(src, dst_type.dtype)
            if src.dtype == dst_type.dtype:
                broadcast = self._broadcast_shaped_to_type(src, dst_type)
                if broadcast is not None:
                    return broadcast
        if isinstance(src.type, BufferType) and isinstance(dst_type, BufferType):
            if src.type == dst_type:
                return src
        return self.compile_error(
            f"Cannot cast from {src.type} to {dst_type}, unsupported type "
            "combination or value is not broadcastable"
        )

    def bitcast(self, src: AlloValue, dst_dtype: DType) -> AlloValue:
        # Reinterpret the bits of `src` as `dst_dtype` without conversion; the
        # two dtypes must share the same bit width.
        assert isinstance(src, AlloValue), "bitcast requires a materialized value"
        assert isinstance(dst_dtype, DType), "bitcast destination must be a dtype"
        if src.dtype.primitive_width != dst_dtype.primitive_width:
            return self.compile_error(
                f"Cannot bitcast from {src.dtype} to {dst_dtype}: bit widths "
                f"{src.dtype.primitive_width} and {dst_dtype.primitive_width} differ"
            )
        if src.dtype == dst_dtype:
            return src
        return self.create_bitcast(src, dst_dtype)

    def normalize_indices(self, indices, *, expected_len=None, context=None):
        out = []
        for val in indices:
            if isinstance(val, tuple):
                return self.compile_error("Nested tuples are not supported in indices.")
            out.append(self.cast(val, index))
        if expected_len is not None and len(out) != expected_len:
            prefix = f"{context} " if context else ""
            return self.compile_error(
                f"{prefix}expects {expected_len} indices, got {len(out)}."
            )
        return out

    def get_promoted_dtype_nary(self, op_name, dtypes, term_signs=None) -> DType:
        if len(dtypes) == 0:
            return self.compile_error("Type promotion requires at least one operand")
        ret = self.type_rules.promote(op_name, dtypes, term_signs=term_signs)
        if ret is not None:
            return ret
        if len(dtypes) == 1:
            operand_desc = f"operand type {dtypes[0]}"
        else:
            operand_desc = "operand types " + ", ".join(str(d) for d in dtypes)
        return self.compile_error(
            f"No {self.typing_style} type promotion rule for operator "
            f"'{op_name}' with {operand_desc}"
        )

    def get_promoted_dtype(self, lhs, rhs, op_name):
        if rhs is None:
            return self.get_promoted_dtype_nary(op_name, [lhs])
        signs = [1, -1] if op_name == "sub" else None
        return self.get_promoted_dtype_nary(op_name, [lhs, rhs], term_signs=signs)

    def _reduce_balanced(self, operands, combine):
        if len(operands) == 0:
            return self.compile_error("Reduction requires at least one operand")
        curr = list(operands)
        while len(curr) > 1:
            nxt = []
            i = 0
            while i < len(curr):
                if i + 1 < len(curr):
                    nxt.append(combine(curr[i], curr[i + 1]))
                    i += 2
                else:
                    nxt.append(curr[i])
                    i += 1
            curr = nxt
        return curr[0]

    def reduce_balanced(self, operands, combine):
        return self._reduce_balanced(operands, combine)

    def create_add_nary(self, operands, *, floating=False):
        return self._reduce_balanced(
            operands, lambda l, r: self.create_add(l, r, floating=floating)
        )

    def create_sub_nary(self, operands, term_signs, *, floating=False):
        if len(operands) != len(term_signs):
            return self.compile_error(
                f"Sub reduction expects {len(operands)} signs, got {len(term_signs)}"
            )
        normalized = []
        for operand, sign in zip(operands, term_signs):
            normalized.append(
                self.create_neg(operand, floating=floating) if sign < 0 else operand
            )
        return self.create_add_nary(normalized, floating=floating)

    def create_mul_nary(self, operands, *, floating=False):
        return self._reduce_balanced(
            operands, lambda l, r: self.create_mul(l, r, floating=floating)
        )

    ######################
    # Basic arithmetic ops
    ######################
    def _emit_linalg_elementwise_binary(self, lhs, rhs, result_type, build_fn):
        assert isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType)
        assert tuple(lhs.type.shape) == tuple(rhs.type.shape)
        res_ir_type = self._materialize(result_type.dtype)
        init = tensor.EmptyOp(
            list(result_type.shape), res_ir_type, ip=self._ip, loc=self._loc
        ).result
        ident = ir.AffineMap.get_identity(result_type.rank, self.context)
        maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(ident)] * 3)
        iters = ir.ArrayAttr.get([self._par_iter] * result_type.rank)
        res_ty = self._materialize(result_type)

        op = linalg.GenericOp(
            [res_ty],
            [lhs.handle, rhs.handle],
            [init],
            maps,
            iters,
            ip=self._ip,
            loc=self._loc,
        )
        body = op.regions[0].blocks.append(
            self._materialize(lhs.dtype),
            self._materialize(rhs.dtype),
            res_ir_type,
        )
        lhs_arg = AlloValue(body.arguments[0], lhs.dtype)
        rhs_arg = AlloValue(body.arguments[1], rhs.dtype)
        with self.at_block_end(body):
            res = build_fn(lhs_arg, rhs_arg)
            linalg.YieldOp([res], ip=self._ip, loc=self._loc)
        return AlloValue(op.results[0], result_type)

    def _emit_linalg_elementwise_unary(self, operand, result_type, build_fn):
        assert isinstance(operand.type, ShapedType)
        res_ir_type = self._materialize(result_type.dtype)
        init = tensor.EmptyOp(
            list(result_type.shape), res_ir_type, ip=self._ip, loc=self._loc
        ).result
        ident = ir.AffineMap.get_identity(result_type.rank, self.context)
        maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(ident)] * 2)
        iters = ir.ArrayAttr.get([self._par_iter] * result_type.rank)
        res_ty = self._materialize(result_type)

        op = linalg.GenericOp(
            [res_ty], [operand.handle], [init], maps, iters, ip=self._ip, loc=self._loc
        )
        body = op.regions[0].blocks.append(
            self._materialize(operand.dtype), res_ir_type
        )
        region_arg = AlloValue(body.arguments[0], operand.dtype)
        with self.at_block_end(body):
            res = build_fn(region_arg)
            linalg.YieldOp([res], ip=self._ip, loc=self._loc)
        return AlloValue(op.results[0], result_type)

    def _emit_elementwise_binary(self, lhs, rhs, result_dtype, build_fn):
        lhs_is_shaped = isinstance(lhs.type, ShapedType)
        rhs_is_shaped = isinstance(rhs.type, ShapedType)
        assert lhs_is_shaped == rhs_is_shaped
        if lhs_is_shaped:
            result_type = self._shaped_type_with_dtype(
                cast(ShapedType, lhs.type), result_dtype
            )
            return self._emit_linalg_elementwise_binary(lhs, rhs, result_type, build_fn)
        return AlloValue(build_fn(lhs, rhs), result_dtype)

    def _emit_elementwise_unary(self, operand, result_dtype, build_fn):
        if isinstance(operand.type, ShapedType):
            result_type = self._shaped_type_with_dtype(operand.type, result_dtype)
            return self._emit_linalg_elementwise_unary(operand, result_type, build_fn)
        return AlloValue(build_fn(operand), result_dtype)

    def _emit_binary_op(self, lhs, rhs, result_dtype, op_cls) -> AlloValue:
        return self._emit_elementwise_binary(
            lhs,
            rhs,
            result_dtype,
            lambda l, r: op_cls(l.handle, r.handle, ip=self._ip, loc=self._loc).result,
        )

    def create_add(self, lhs, rhs, *, floating=False):
        return self._emit_binary_op(
            lhs, rhs, lhs.dtype, arith.AddFOp if floating else arith.AddIOp
        )

    def create_sub(self, lhs, rhs, *, floating=False):
        return self._emit_binary_op(
            lhs, rhs, lhs.dtype, arith.SubFOp if floating else arith.SubIOp
        )

    def create_mul(self, lhs, rhs, *, floating=False):
        return self._emit_binary_op(
            lhs, rhs, lhs.dtype, arith.MulFOp if floating else arith.MulIOp
        )

    def create_div(self, lhs, rhs, *, signed=True, floating=False):
        assert not (signed and floating)
        if floating:
            op_cls = arith.DivFOp
        elif signed:
            op_cls = arith.DivSIOp
        else:
            op_cls = arith.DivUIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_floordiv(self, lhs, rhs, *, signed=True, floating=False):
        assert not (signed and floating)

        def build_fn(l, r):
            if floating:
                # Floats keep Python floor semantics (there is no HLS QoR concern).
                divf = arith.DivFOp(
                    l.handle, r.handle, ip=self._ip, loc=self._loc
                ).result
                return math.FloorOp(divf, ip=self._ip, loc=self._loc).result
            # Integer ``//`` truncates toward zero, like ``/`` and ``%``: a single
            # HLS-native divide that Vitis recognizes for addressing, and it keeps
            # the div/mod identity intact against ``remsi``/``remui``.
            if signed:
                return arith.DivSIOp(
                    l.handle, r.handle, ip=self._ip, loc=self._loc
                ).result
            return arith.DivUIOp(l.handle, r.handle, ip=self._ip, loc=self._loc).result

        return self._emit_elementwise_binary(lhs, rhs, lhs.dtype, build_fn)

    def create_mod(self, lhs, rhs, *, signed=True, floating=False):
        assert not (signed and floating)
        if floating:
            op_cls = arith.RemFOp
        elif signed:
            op_cls = arith.RemSIOp
        else:
            op_cls = arith.RemUIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_pow(self, base, exp, *, base_floating=False, exp_floating=False):
        assert not (
            not base_floating and exp_floating
        ), "If rhs is floating, base must be floating too"

        def build_fn(l, r):
            if base_floating and exp_floating:
                return math.PowFOp(
                    l.handle, r.handle, ip=self._ip, loc=self._loc
                ).result
            if base_floating:
                return math.FPowIOp(
                    l.handle, r.handle, ip=self._ip, loc=self._loc
                ).result
            return math.IPowIOp(l.handle, r.handle, ip=self._ip, loc=self._loc).result

        return self._emit_elementwise_binary(base, exp, base.dtype, build_fn)

    def create_lshift(self, lhs, rhs):
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.ShLIOp)

    def create_rshift(self, lhs, rhs, signed=True):
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        return self._emit_binary_op(
            lhs, rhs, lhs.dtype, arith.ShRSIOp if signed else arith.ShRUIOp
        )

    def create_neg(self, operand, *, floating=False):
        def build_fn(value):
            if floating:
                return arith.NegFOp(value.handle, ip=self._ip, loc=self._loc).result
            zero = self.make_scalar(0, value.dtype).handle
            return arith.SubIOp(zero, value.handle, ip=self._ip, loc=self._loc).result

        return self._emit_elementwise_unary(operand, operand.dtype, build_fn)

    def create_invert(self, operand):
        assert isinstance(operand.dtype, APInt)

        def build_fn(value):
            ones = 2**value.dtype.primitive_width - 1
            ones_val = self.make_scalar(ones, value.dtype).handle
            return arith.XOrIOp(
                ones_val, value.handle, ip=self._ip, loc=self._loc
            ).result

        return self._emit_elementwise_unary(operand, operand.dtype, build_fn)

    #########################
    # Comparison ops
    #########################

    def create_cmpi(self, lhs, rhs, predicate: CmpPred, *, signed=False):
        assert isinstance(lhs.dtype, (IndexType, APInt)) and isinstance(
            rhs.dtype, (IndexType, APInt)
        )
        pred_val = predicate.value
        if not signed and predicate in {CmpPred.LT, CmpPred.LE, CmpPred.GT, CmpPred.GE}:
            pred_val += 4

        def build_fn(l, r):
            return arith.CmpIOp(
                pred_val, l.handle, r.handle, ip=self._ip, loc=self._loc
            ).result

        return self._emit_elementwise_binary(lhs, rhs, AlloBool, build_fn)

    _cmpf_pred_map = {
        CmpPred.EQ: 1,
        CmpPred.NE: 6,
        CmpPred.LT: 4,
        CmpPred.LE: 5,
        CmpPred.GT: 2,
        CmpPred.GE: 3,
    }

    def create_cmpf(self, lhs, rhs, predicate: CmpPred, *, ordered=False):
        assert isinstance(lhs.dtype, APFloat) and isinstance(rhs.dtype, APFloat)
        pred_val = self._cmpf_pred_map[predicate]
        if ordered and predicate in {CmpPred.LT, CmpPred.LE, CmpPred.GT, CmpPred.GE}:
            pred_val += 7

        def build_fn(l, r):
            return arith.CmpFOp(
                pred_val, l.handle, r.handle, ip=self._ip, loc=self._loc
            ).result

        return self._emit_elementwise_binary(lhs, rhs, AlloBool, build_fn)

    def to_condition(self, cond: AlloValue) -> AlloValue:
        """Convert a scalar used in a boolean context (the test of an ``if`` /
        ``while`` / ternary) to ``i1`` with *truthiness* semantics: ``cond != 0``."""
        assert isinstance(cond.type, DType), "a boolean condition must be a scalar"
        if cond.type == AlloBool:
            return cond
        zero = self.make_scalar(0, cond.type)
        if cond.type.is_float():
            return self.create_cmpf(cond, zero, CmpPred.NE)
        return self.create_cmpi(cond, zero, CmpPred.NE)

    def create_max(self, lhs, rhs, *, signed=True, floating=False, propagate_nan=True):
        assert not (signed and floating)
        if floating:
            op_cls = arith.MaximumFOp if propagate_nan else arith.MaxNumFOp
        elif signed:
            op_cls = arith.MaxSIOp
        else:
            op_cls = arith.MaxUIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_min(self, lhs, rhs, *, signed=True, floating=False, propagate_nan=True):
        assert not (signed and floating)
        if floating:
            op_cls = arith.MinimumFOp if propagate_nan else arith.MinNumFOp
        elif signed:
            op_cls = arith.MinSIOp
        else:
            op_cls = arith.MinUIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    ##########################
    # Bitwise / logical ops
    ##########################

    def create_bitwise_and(self, lhs, rhs):
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.AndIOp)

    def create_bitwise_or(self, lhs, rhs):
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.OrIOp)

    def create_bitwise_xor(self, lhs, rhs):
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.XOrIOp)

    def create_logical_and(self, lhs, rhs):
        assert lhs.dtype == AlloBool and rhs.dtype == AlloBool
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.AndIOp)

    def create_logical_or(self, lhs, rhs):
        assert lhs.dtype == AlloBool and rhs.dtype == AlloBool
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.OrIOp)

    def create_logical_not(self, operand):
        assert operand.dtype == AlloBool

        def build_fn(value):
            ones = self.make_scalar(1, value.dtype).handle
            return arith.XOrIOp(ones, value.handle, ip=self._ip, loc=self._loc).result

        return self._emit_elementwise_unary(operand, operand.dtype, build_fn)

    ##########################
    # Bit slice operations
    ##########################
    # `x[lo:hi]` extracts the bits in the half-open range ``[lo, hi)`` (Python
    # slice convention), so the slice holds ``hi - lo`` bits. A single index
    # ``x[i]`` is the width-one slice ``x[i:i+1]``.

    def _index_succ(self, idx: AlloValue) -> AlloValue:
        one = self.create_const_index(1)
        handle = arith.AddIOp(idx.handle, one.handle, ip=self._ip, loc=self._loc).result
        return AlloValue(handle, index)

    def create_bit_get_slice(self, src, lo, hi, result_dtype: DType) -> AlloValue:
        assert isinstance(src.dtype, APInt)
        lo = self.cast(lo, index)
        hi = self.cast(hi, index)
        op = allo_d.BitGetSliceOp(
            self._materialize(result_dtype),
            src.handle,
            lo.handle,
            hi.handle,
            ip=self._ip,
            loc=self._loc,
        )
        return AlloValue(op.result, result_dtype)

    def create_bit_set_slice(self, src, lo, hi, value: AlloValue) -> AlloValue:
        assert isinstance(src.dtype, APInt)
        lo = self.cast(lo, index)
        hi = self.cast(hi, index)
        op = allo_d.BitSetSliceOp(
            src.handle,
            lo.handle,
            hi.handle,
            value.handle,
            ip=self._ip,
            loc=self._loc,
        )
        return AlloValue(op.result, src.dtype)

    def create_bit_extract(self, src, idx) -> AlloValue:
        idx = self.cast(idx, index)
        return self.create_bit_get_slice(src, idx, self._index_succ(idx), u1)

    def create_bit_insert(self, value: AlloValue, src, idx) -> AlloValue:
        idx = self.cast(idx, index)
        return self.create_bit_set_slice(src, idx, self._index_succ(idx), value)

    ###########################
    # Broadcasting
    ###########################
    @staticmethod
    def infer_broadcast_shape(shape1, shape2):
        res_shape, a_indices, b_indices = [], [], []
        len_a, len_b = len(shape1), len(shape2)
        max_rank = max(len_a, len_b)
        for i in range(1, max_rank + 1):
            idx = max_rank - i
            dim_a = shape1[-i] if i <= len_a else 1
            dim_b = shape2[-i] if i <= len_b else 1
            if dim_a == dim_b:
                res_shape.append(dim_a)
            elif dim_a == 1:
                res_shape.append(dim_b)
                if dim_b > 1:
                    a_indices.append(idx)
            elif dim_b == 1:
                res_shape.append(dim_a)
                if dim_a > 1:
                    b_indices.append(idx)
            else:
                return [], [], []
        res_shape.reverse()
        a_indices.reverse()
        b_indices.reverse()
        return res_shape, a_indices, b_indices

    def broadcast_pair(self, lhs, rhs):
        assert (
            lhs.dtype == rhs.dtype
        ), "Broadcasting requires operands to have the same dtype"
        lhs_is_shaped = isinstance(lhs.type, ShapedType)
        rhs_is_shaped = isinstance(rhs.type, ShapedType)

        if lhs_is_shaped and rhs_is_shaped:
            lhs_shape = cast(ShapedType, lhs.type).shape
            rhs_shape = cast(ShapedType, rhs.type).shape
            shape, indices_lhs, indices_rhs = self.infer_broadcast_shape(
                lhs_shape, rhs_shape
            )
            if not shape:
                return self.compile_error(
                    f"Shapes {lhs_shape} and {rhs_shape} are not broadcastable"
                )
            elem = self._materialize(lhs.dtype)
            init = tensor.EmptyOp(list(shape), elem, ip=self._ip, loc=self._loc).result
            if not indices_lhs and not indices_rhs:
                return lhs, rhs
            if indices_lhs:
                lhs_type = (
                    TensorType(shape, lhs.dtype)
                    if isinstance(lhs.type, TensorType)
                    else BufferType(shape, lhs.dtype)
                )
                res_ty = self._materialize(lhs_type)
                bcast = linalg.BroadcastOp(
                    [res_ty], lhs.handle, init, indices_lhs, ip=self._ip, loc=self._loc
                )
                _fill_named_linalg_region(bcast)
                return AlloValue(bcast.results[0], lhs_type), rhs
            rhs_type = (
                TensorType(shape, rhs.dtype)
                if isinstance(rhs.type, TensorType)
                else BufferType(shape, rhs.dtype)
            )
            res_ty = self._materialize(rhs_type)
            bcast = linalg.BroadcastOp(
                [res_ty], rhs.handle, init, indices_rhs, ip=self._ip, loc=self._loc
            )
            _fill_named_linalg_region(bcast)
            return lhs, AlloValue(bcast.results[0], rhs_type)

        if not lhs_is_shaped and not rhs_is_shaped:
            return lhs, rhs

        if isinstance(lhs.type, BufferType) or isinstance(rhs.type, BufferType):
            return self.compile_error("Scalars cannot broadcast to buffer types")
        if lhs_is_shaped and not rhs_is_shaped:
            shaped = cast(ShapedType, lhs.type)
            res_ty = self._materialize(shaped)
            rhs_handle = tensor.SplatOp(
                res_ty, rhs.handle, [], ip=self._ip, loc=self._loc
            ).result
            return lhs, AlloValue(rhs_handle, lhs.type)
        shaped = cast(ShapedType, rhs.type)
        res_ty = self._materialize(shaped)
        lhs_handle = tensor.SplatOp(
            res_ty, lhs.handle, [], ip=self._ip, loc=self._loc
        ).result
        return AlloValue(lhs_handle, rhs.type), rhs

    ###########################
    # Memory operations
    ###########################

    def make_buffer(self, buffer_type: ShapedType) -> AlloValue:
        if isinstance(buffer_type, BufferType):
            op = memref.AllocOp(
                self._materialize(buffer_type), [], [], ip=self._ip, loc=self._loc
            )
            # Tag element signedness for backend codegen (mirrors create_stream);
            # MLIR integers are signless, so this marker is the only record of
            # whether a body-local buffer holds signed data.
            op.operation.attributes[allo_d.SIGNED_ATTR_NAME] = self.get_string_attr(
                "s" if buffer_type.dtype.is_int() else "u"
            )
            return AlloValue(op.result, buffer_type)
        if isinstance(buffer_type, TensorType):
            op = tensor.EmptyOp(
                list(buffer_type.shape),
                self._materialize(buffer_type.dtype),
                ip=self._ip,
                loc=self._loc,
            )
            return AlloValue(op.result, buffer_type)
        assert False, f"Unsupported shaped type: {buffer_type}"

    def fill_buffer(self, buffer: AlloValue, value: AlloValue):
        assert isinstance(buffer.type, ShapedType)
        if isinstance(buffer.type, TensorType):
            return self._fill_shaped_value(value, buffer)
        assert isinstance(buffer.type, BufferType)
        self._fill_shaped_value(value, buffer)
        return None

    def create_load(self, lhs: AlloValue, indices: Sequence[AlloValue]) -> AlloValue:
        assert isinstance(lhs.type, ShapedType)
        index_values = [idx.handle for idx in indices]
        if isinstance(lhs.type, BufferType):
            op = memref.LoadOp(lhs.handle, index_values, ip=self._ip, loc=self._loc)
            return AlloValue(op.result, lhs.dtype)
        if isinstance(lhs.type, TensorType):
            # the two branches are mutually exclusive
            # pylint: disable-next=redefined-variable-type
            op = tensor.ExtractOp(lhs.handle, index_values, ip=self._ip, loc=self._loc)
            return AlloValue(op.result, lhs.dtype)
        assert False, f"Unsupported shaped type: {lhs.type}"

    def create_store(
        self, value: AlloValue, buffer: AlloValue, indices: Sequence[AlloValue]
    ):
        assert isinstance(buffer.type, ShapedType)
        index_values = [idx.handle for idx in indices]
        if isinstance(buffer.type, BufferType):
            memref.StoreOp(
                value.handle, buffer.handle, index_values, ip=self._ip, loc=self._loc
            )
            return None
        if isinstance(buffer.type, TensorType):
            op = tensor.InsertOp(
                value.handle, buffer.handle, index_values, ip=self._ip, loc=self._loc
            )
            return AlloValue(op.result, buffer.type)
        assert False, f"Unsupported shaped type: {buffer.type}"

    def create_affine_load(
        self, buffer: AlloValue, affine_map: ir.AffineMap, operands: Sequence[AlloValue]
    ) -> AlloValue:
        assert isinstance(buffer.type, BufferType)
        op = affine_d.AffineLoadOp(
            self._materialize(buffer.dtype),
            buffer.handle,
            [v.handle for v in operands],
            ir.AffineMapAttr.get(affine_map),
            ip=self._ip,
            loc=self._loc,
        )
        return AlloValue(op.result, buffer.dtype)

    def create_affine_store(
        self,
        value: AlloValue,
        buffer: AlloValue,
        affine_map: ir.AffineMap,
        operands: Sequence[AlloValue],
    ) -> None:
        assert isinstance(buffer.type, BufferType)
        affine_d.AffineStoreOp(
            value.handle,
            buffer.handle,
            [v.handle for v in operands],
            ir.AffineMapAttr.get(affine_map),
            ip=self._ip,
            loc=self._loc,
        )

    def create_affine_for(
        self,
        lb_map: ir.AffineMap,
        lb_operands: Sequence,
        ub_map: ir.AffineMap,
        ub_operands: Sequence,
        step: int,
        iter_args: Sequence,
        arg_locs: Sequence[ir.Location] | None = None,
    ):
        """Build an ``affine.for`` whose body block carries ``arg_locs`` on its
        induction variable and loop-carried arguments. ``arg_locs`` is ordered
        ``[iv, *iter_args]``. Returns the specialized op view."""
        results = [v.type for v in iter_args]
        op = AffineForOp(
            results,
            list(lb_operands),
            list(ub_operands),
            list(iter_args),
            ir.AffineMapAttr.get(lb_map),
            ir.AffineMapAttr.get(ub_map),
            step,
            ip=self._ip,
            loc=self._loc,
        )
        op.regions[0].blocks.append(ir.IndexType.get(), *results, arg_locs=arg_locs)
        return op.operation.opview

    def create_scf_for(
        self,
        lb,
        ub,
        step,
        iter_args: Sequence,
        arg_locs: Sequence[ir.Location] | None = None,
    ):
        """Build an ``scf.for`` whose body block carries ``arg_locs`` on its
        induction variable and loop-carried arguments. ``arg_locs`` is ordered
        ``[iv, *iter_args]``. Returns the specialized op view."""
        results = [v.type for v in iter_args]
        op = ForOp(results, lb, ub, step, list(iter_args), ip=self._ip, loc=self._loc)
        op.regions[0].blocks.append(op.operands[0].type, *results, arg_locs=arg_locs)
        return op.operation.opview

    def _stream_handle_and_indices(self, stream):
        assert isinstance(stream, AlloValue)
        assert isinstance(stream.type, StreamType)
        assert stream.indices is not None
        return stream.handle, stream.type, stream.indices

    def create_stream_get(self, stream) -> AlloValue:
        handle, stream_type, indices = self._stream_handle_and_indices(stream)
        index_values = [idx.handle for idx in indices]
        value_ty = self._materialize(stream_type.base_type)
        op = allo_d.StreamGetOp(
            value_ty, handle, index_values, ip=self._ip, loc=self._loc
        )
        return AlloValue(op.result, stream_type.base_type)

    def create_stream_put(self, stream, value) -> None:
        handle, stream_type, indices = self._stream_handle_and_indices(stream)
        value = self.cast(value, stream_type.base_type)
        index_values = [idx.handle for idx in indices]
        allo_d.StreamPutOp(
            handle, index_values, value.handle, ip=self._ip, loc=self._loc
        )
