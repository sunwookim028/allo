# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
from collections.abc import Sequence


from dataclasses import dataclass

from .._mlir import ir
from .._mlir.ir import Context, Type, Value
from .._mlir.dialects.allo import StreamType as MlirStreamType

# ==========================================================================#
# Frontend type system
# ==========================================================================#


class TypeBase:
    """
    Represents a frontend type in the Allo compiler.

    The frontend type should be able to compare itself with other frontend types,
    and generate a corresponding underlying MLIR type.

    Every concrete frontend type should have a unique name, which is used for type comparison and debugging purposes.
    """

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, value: object, /):
        return isinstance(value, TypeBase) and self.name == value.name

    def __ne__(self, value: object, /):
        return not self.__eq__(value)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def materialize(self, context: Context, /) -> Type:
        raise NotImplementedError()


class Template:
    def __init__(self, name: str):
        assert (
            isinstance(name, str) and name.isidentifier()
        ), f"invalid template name: {name}"
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __getitem__(self, shape) -> "ShapeExpr":
        return ShapeExpr(self, shape)


class ConstexprType(TypeBase):
    """
    Represents a constexpr type in the Allo compiler.

    A constexpr type is a frontend-only type that will not be materialized into MLIR.
    Assert out if some logic tries to materialize a constexpr type, as it should not happen.
    """

    def __init__(self) -> None:
        super().__init__("constexpr")

    def materialize(self, context: Context, /) -> Type:
        assert False, "constexpr type should not be materialized"


constexpr = ConstexprType()  # singleton instance for constexpr type


class DType(TypeBase):
    """
    Represents a primitive data type in the Allo compiler, such as int32, float64, etc.

    Every concrete DType should have unique name.
    """

    def __init__(self, name: str, primitive_width: int):
        super().__init__(name)
        self.primitive_width = primitive_width

    def __hash__(self) -> int:
        return hash((self.name, self.primitive_width))

    def materialize(self, context: Context, /) -> Type:
        raise NotImplementedError(
            "Every concrete DType should implement its own materialization logic"
        )

    def is_int(self):
        return self.name.startswith("int")

    def is_intn(self, n: int):
        return self.is_int() and self.primitive_width == n

    def is_uint(self):
        return self.name.startswith("uint")

    def is_uintn(self, n: int):
        return self.is_uint() and self.primitive_width == n

    def is_int_signless(self):
        return self.is_int() or self.is_uint()

    def is_fp16(self):
        return self.name == "float16"

    def is_fp32(self):
        return self.name == "float32"

    def is_fp64(self):
        return self.name == "float64"

    def is_bf16(self):
        return self.name == "bfloat16"

    def is_float(self):
        return self.is_fp16() or self.is_fp32() or self.is_fp64() or self.is_bf16()

    def is_index(self):
        return self.name == "index"

    def __getitem__(self, shape) -> "ShapeExpr":
        return ShapeExpr(self, shape)


class APInt(DType):
    """
    Represents an arbitrary precision integer type in the Allo compiler.

    The APInt type is parameterized by its bit width, which can be any positive integer.
    """

    def __init__(self, bit_width: int, signed=False):
        if bit_width <= 0:
            raise ValueError("bit_width must be a positive integer")
        name = f"int{bit_width}" if signed else f"uint{bit_width}"
        super().__init__(name, bit_width)
        self.signed = signed

    def materialize(self, context: Context, /) -> Type:
        return ir.IntegerType.get_signless(self.primitive_width, context)


apint = APInt  # name alias for easier usage


def widen_apint_to_std(dtype: "DType") -> "DType":
    """Round a non-standard-width ``APInt`` up to the next standard width
    (8/16/32/64), preserving signedness; other dtypes are returned unchanged.

    Host marshalling (numpy/ctypes) cannot represent arbitrary integer widths, so
    the backends widen the boundary to a standard width. This mirrors the
    ``generate-apint-wrapper`` MLIR pass exactly so the host view matches the
    wrapper's ABI. Widths > 64 are unsupported at the boundary.
    """
    if not isinstance(dtype, APInt):
        return dtype
    w = dtype.primitive_width
    if w in {1, 8, 16, 32, 64}:
        return dtype
    if w > 64:
        raise TypeError(
            f"APInt width {w} > 64 bits is not supported at the host boundary"
        )
    std = next(s for s in (8, 16, 32, 64) if w <= s)
    return APInt(std, signed=dtype.signed)


### make some commonly used DType for easier usage
# i1 = APInt(1, signed=True) # use u1 instead
i2 = APInt(2, signed=True)
i3 = APInt(3, signed=True)
i4 = APInt(4, signed=True)
i5 = APInt(5, signed=True)
i6 = APInt(6, signed=True)
i7 = APInt(7, signed=True)
i8 = APInt(8, signed=True)
i9 = APInt(9, signed=True)
i10 = APInt(10, signed=True)
i11 = APInt(11, signed=True)
i12 = APInt(12, signed=True)
i13 = APInt(13, signed=True)
i14 = APInt(14, signed=True)
i15 = APInt(15, signed=True)
i16 = APInt(16, signed=True)
i32 = APInt(32, signed=True)
i64 = APInt(64, signed=True)
i128 = APInt(128, signed=True)
i256 = APInt(256, signed=True)

u1 = APInt(1, signed=False)  # also used as boolean type
u2 = APInt(2, signed=False)
u3 = APInt(3, signed=False)
u4 = APInt(4, signed=False)
u5 = APInt(5, signed=False)
u6 = APInt(6, signed=False)
u7 = APInt(7, signed=False)
u8 = APInt(8, signed=False)
u9 = APInt(9, signed=False)
u10 = APInt(10, signed=False)
u11 = APInt(11, signed=False)
u12 = APInt(12, signed=False)
u13 = APInt(13, signed=False)
u14 = APInt(14, signed=False)
u15 = APInt(15, signed=False)
u16 = APInt(16, signed=False)
u32 = APInt(32, signed=False)
u64 = APInt(64, signed=False)
u128 = APInt(128, signed=False)
u256 = APInt(256, signed=False)

# the DSL's boolean type; inside a kernel `bool` is this, not the builtin.
# pylint: disable-next=redefined-builtin
bool = u1


class APFloat(DType):
    """
    Represents an arbitrary precision floating-point type in the Allo compiler.

    The APFloat type is parameterized by its bit width, which can be any positive integer.

    TODO: maybe support real arbitrary precision floating-point types in the future
    """

    def __init__(self, exp_width: int, sig_width: int):
        if exp_width <= 0 or sig_width <= 0:
            raise ValueError("exp_width and sig_width must be positive integers")
        width = 1 + exp_width + sig_width  # 1 bit for sign
        if (exp_width, sig_width) == (5, 10):
            name = "float16"
        elif (exp_width, sig_width) == (8, 23):
            name = "float32"
        elif (exp_width, sig_width) == (11, 52):
            name = "float64"
        elif (exp_width, sig_width) == (8, 7):
            name = "bfloat16"
        else:
            raise NotImplementedError(
                "only fp16, fp32, fp64 and bf16 are supported for now"
            )
        super().__init__(name, width)

    def materialize(self, context: Context, /) -> Type:
        if self.name == "float16":
            return ir.F16Type.get(context)
        if self.name == "float32":
            return ir.F32Type.get(context)
        if self.name == "float64":
            return ir.F64Type.get(context)
        if self.name == "bfloat16":
            return ir.BF16Type.get(context)
        assert False, f"unsupported floating-point type: {self.name}"


apfloat = APFloat  # name alias for easier usage

### make some commonly used APFloat for easier usage
f16 = APFloat(5, 10)
f32 = APFloat(8, 23)
f64 = APFloat(11, 52)
bf16 = APFloat(8, 7)


class IndexType(DType):
    """
    Represents an index type in the Allo compiler.

    The index type is a special type used for indexing and loop bounds,
    its an opaque type that does not have a fixed bit width
    """

    def __init__(self):
        super().__init__("index", 2**32 - 1)

    def materialize(self, context: Context, /) -> Type:
        return ir.IndexType.get(context)


index = IndexType()  # singleton instance for index type


# `materialize` stays abstract here: ShapedType is itself an abstract base.
# pylint: disable-next=abstract-method
class ShapedType(TypeBase):
    """
    Represents a shaped type in the Allo compiler, such as tensor, memref, etc.
    It's an abstract base class for all shaped types.

    The ShapedType is parameterized by its shape and element type.
    """

    def __init__(self, name: str, shape: Sequence[int], dtype: DType):
        super().__init__(name)
        self.shape = shape
        self.dtype = dtype
        self.rank = len(shape)


class TensorType(ShapedType):
    """
    Represents a tensor type in the Allo compiler.

    The TensorType is a concrete shaped type that represents a multi-dimensional array of elements.
    """

    def __init__(self, shape: Sequence[int], dtype: DType):
        prefix = "x".join(str(dim) for dim in shape)
        name = f"tensor<{prefix + 'x' if prefix else ''}{dtype.name}>"
        super().__init__(name, shape, dtype)

    def materialize(self, context: Context, /) -> Type:
        mlir_dtype = self.dtype.materialize(context)
        return ir.RankedTensorType.get(list(self.shape), mlir_dtype)


class BufferType(ShapedType):
    """
    Represents a memref type in the Allo compiler.

    The MemRefType is a concrete shaped type that represents a multi-dimensional array of elements with a specific memory layout.
    """

    def __init__(self, shape: Sequence[int], dtype: DType):
        prefix = "x".join(str(dim) for dim in shape)
        name = f"memref<{prefix + 'x' if prefix else ''}{dtype.name}>"
        super().__init__(name, shape, dtype)

    def materialize(self, context: Context, /) -> Type:
        mlir_dtype = self.dtype.materialize(context)
        # `loc` is only used for diagnostics during layout verification.
        return ir.MemRefType.get(
            list(self.shape), mlir_dtype, loc=ir.Location.unknown(context)
        )


DEFAULT_STREAM_DEPTH = 2


class StreamType(TypeBase):
    """
    Represents an Allo stream type.

    `base_type` is the transmission unit. It can be either a scalar dtype or a
    shaped buffer type. `shape` describes an array of streams; the empty shape is
    a single rank-0 stream.
    """

    def __init__(
        self,
        base_type: DType | ShapedType,
        depth: int = DEFAULT_STREAM_DEPTH,
        shape: Sequence[int] = (),
    ):
        assert isinstance(base_type, (DType, ShapedType))
        assert isinstance(depth, int) and depth > 0
        shape = tuple(shape)
        assert all(isinstance(dim, int) and dim >= 0 for dim in shape)
        shape_suffix = "[" + ",".join(str(dim) for dim in shape) + "]" if shape else ""
        super().__init__(f"Stream[{base_type},{depth}]{shape_suffix}")
        self.base_type = base_type
        self.depth = depth
        self.shape = shape
        self.rank = len(shape)

    def materialize(self, context: Context, /) -> Type:
        base = self.base_type.materialize(context)
        return MlirStreamType.get(base, self.depth, list(self.shape))


class StatefulType(TypeBase):
    """`Stateful[T]`: marks a local declaration as persistent across kernel
    invocations (C ``static`` semantics). ``inner`` is the concrete scalar
    (``DType``) or buffer (``BufferType``) type the variable presents; the
    stateful wrapper itself is a declaration-only marker and is never
    materialized into MLIR (the backing storage is a module-level global)."""

    def __init__(self, inner: DType | BufferType):
        super().__init__(f"Stateful[{inner}]")
        self.inner = inner

    def materialize(self, context: Context, /) -> Type:
        assert False, "StatefulType is declaration-only and is never materialized"


# ==========================================================================#
# Deferred type annotations
#
# `dtype[shape]`, `Template[shape]` and `Stream[base][shape]` evaluate to these
# lightweight descriptors so that complex annotations can be written without
# quotes (and without `from __future__ import annotations`). They are resolved
# into concrete frontend types by `Kernel.parse_type_annotation`, which is the
# only place that knows the kernel's options and template bindings.
# ==========================================================================#


def _as_shape(key) -> tuple:
    return key if isinstance(key, tuple) else (key,)


class ShapeExpr:
    """A `dtype[shape]` annotation, unresolved. `dtype` is a `DType` or
    `Template`; `shape` entries may be ints, `Template`s or `ConstexprValue`s."""

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = _as_shape(shape)

    def __repr__(self) -> str:
        return f"{self.dtype}[{', '.join(str(dim) for dim in self.shape)}]"


class StreamExpr:
    """A `Stream[base, depth?][shape]` annotation, unresolved. `base` may be a
    `DType`, `ShapedType`, `ShapeExpr` or `Template`; `depth` is the optional FIFO
    depth; `shape` is the stream-array shape."""

    def __init__(self, base, depth=DEFAULT_STREAM_DEPTH, shape: Sequence = ()):
        self.base = base
        self.depth = depth
        self.shape = tuple(shape)

    def __getitem__(self, key) -> "StreamExpr":
        if self.shape:
            raise TypeError(f"Stream type '{self!r}' already has a shape")
        return StreamExpr(self.base, self.depth, _as_shape(key))

    def __repr__(self) -> str:
        suffix = f"[{','.join(str(dim) for dim in self.shape)}]" if self.shape else ""
        return f"Stream[{self.base!r}, {self.depth}]{suffix}"


class _StreamFactory:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def __getitem__(self, base_type) -> StreamExpr:
        # `Stream[base, depth]` arrives as a tuple; `Stream[base]` uses the default.
        if isinstance(base_type, tuple):
            base, depth = base_type
            return StreamExpr(base, depth)
        return StreamExpr(base_type)

    def __repr__(self) -> str:
        return self.prefix

    __str__ = __repr__


Stream = _StreamFactory("Stream")


class StatefulExpr:
    """A `Stateful[T]` annotation, unresolved (mirrors `StreamExpr`). `base` may
    be a `DType`, `ShapedType`, `ShapeExpr` or `Template`."""

    def __init__(self, base):
        self.base = base

    def __repr__(self) -> str:
        return f"Stateful[{self.base!r}]"


class _StatefulFactory:
    def __getitem__(self, base) -> "StatefulExpr":
        return StatefulExpr(base)

    def __repr__(self) -> str:
        return "Stateful"

    __str__ = __repr__


Stateful = _StatefulFactory()


# =========================================================================#
# Frontend value system
# =========================================================================#


@dataclass
class ValueBase:
    """
    Represents a frontend value in the Allo compiler.

    The frontend value should should hold its frontend type,
    and its underlying MLIR value handle if any.
    """

    type: TypeBase

    @property
    def handle(self):
        raise NotImplementedError()


class ConstexprValue(ValueBase):
    """
    Represents a constexpr value in the Allo compiler.

    A constexpr value is a frontend-only value that does not have a corresponding MLIR value handle.
    """

    # pylint: disable-next=super-init-not-called
    def __init__(self, value):
        # peel out nested constexpr value
        while isinstance(value, ConstexprValue):
            value = value.value
        self.value = value
        self.type = ConstexprType()

    def __str__(self) -> str:
        return f"constexpr({self.value})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def handle(self):
        return None


class AlloValue(ValueBase):
    """
    Proxy value class for all non-constexpr values in the Allo compiler.

    An Allo value should always have a corresponding MLIR value handle, as it represents a value that will be materialized into MLIR.
    """

    # pylint: disable-next=super-init-not-called
    def __init__(self, handle: Value, value_type: TypeBase):
        assert handle is not None, "handle cannot be None for AlloValue"
        self._handle = handle
        self.type = value_type
        self.dtype = (
            value_type.dtype if isinstance(value_type, ShapedType) else value_type
        )
        # wrap the shape to frontend values
        self.shape = (
            [ConstexprValue(s) for s in value_type.shape]
            if isinstance(value_type, (ShapedType, StreamType))
            else ()
        )
        self.rank = len(self.shape)
        self.indices: tuple[AlloValue, ...] | None = None

    def __str__(self) -> str:
        return f"AlloValue<{self.type}>"

    def __repr__(self) -> str:
        return f"AlloValue<{self.type}>({self.handle})"

    @property
    def handle(self) -> Value:
        return self._handle

    @property
    def is_indexed(self) -> builtins.bool:
        return self.indices is not None


class StatefulValue(ValueBase):
    """A persistent, memory-backed variable (C ``static`` semantics).

    It exists only as a binding in the local scope: reading the name loads from
    the backing global and writing the name stores into it, so a StatefulValue
    never flows into expression evaluation as itself. This keeps it out of the
    SSA phi / loop-iter-arg machinery, which only tracks `AlloValue`s.

    `storage` is the rank-0 (scalar) or shaped (array) `AlloValue<BufferType>`
    produced by `memref.get_global`; `type` is the logical type the user sees
    (a `DType` for scalars, a `BufferType` for arrays).
    """

    # pylint: disable-next=super-init-not-called
    def __init__(self, storage: AlloValue, value_type: TypeBase):
        self.storage = storage
        self.type = value_type

    @property
    def handle(self) -> Value:
        return self.storage.handle

    @property
    def is_scalar(self) -> builtins.bool:
        return isinstance(self.type, DType)

    def __repr__(self) -> str:
        return f"StatefulValue<{self.type}>"


# map from PyTorch dtype string to Allo DType, for easier interop with PyTorch/NumPy
torch_dtype_map: dict[str, DType] = {
    "bool": u1,
    "int8": i8,
    "int16": i16,
    "short": i16,
    "int32": i32,
    "int": i32,
    "int64": i64,
    "intp": i64,
    "uint8": u8,
    "uint16": u16,
    "uint32": u32,
    "uint64": u64,
    "uintp": u64,
    "float16": f16,
    "half": f16,
    "float32": f32,
    "float": f32,
    "float64": f64,
    "double": f64,
    "bfloat16": bf16,
}


def unwrap_if_constexpr(o):
    """
    Helper function to unwrap the value from a Constexpr wrapper if needed.
    If the input value is not a Constexpr, return it as is.
    """
    if isinstance(o, list):
        return [unwrap_if_constexpr(v) for v in o]
    if isinstance(o, tuple):
        return tuple(unwrap_if_constexpr(v) for v in o)
    return o.value if isinstance(o, ConstexprValue) else o


class Range:
    def __init__(
        self,
        start,
        stop=None,
        step=None,
        *,
        name: ConstexprValue = ConstexprValue(""),
    ):
        self.name = unwrap_if_constexpr(name)
        self.step = step if step is not None else ConstexprValue(1)
        if stop is None:
            self.start = ConstexprValue(0)
            self.stop = start
        else:
            self.start = start
            self.stop = stop

    def __iter__(self) -> Range:
        raise RuntimeError("allo.range can only be used within allo kernels")

    def __next__(self) -> AlloValue:
        raise RuntimeError("allo.range can only be used within allo kernels")


# the DSL's loop constructor; inside a kernel `range` is this, not the builtin.
# pylint: disable-next=redefined-builtin
range = Range


class Grid:
    def __init__(self, *ranges: tuple, name: ConstexprValue = ConstexprValue("")):
        self.name = name.value
        self.starts = []
        self.stops = []
        self.steps = []

        # canonicalize expressions
        for r in ranges:
            if isinstance(r, (ConstexprValue, AlloValue)):
                self.starts.append(ConstexprValue(0))
                self.stops.append(r)
                self.steps.append(ConstexprValue(1))
            elif len(r) == 1:
                self.starts.append(ConstexprValue(0))
                self.stops.append(r[0])
                self.steps.append(ConstexprValue(1))
            elif len(r) == 2:
                self.starts.append(r[0])
                self.stops.append(r[1])
                self.steps.append(ConstexprValue(1))
            elif len(r) == 3:
                self.starts.append(r[0])
                self.stops.append(r[1])
                self.steps.append(r[2])
            else:
                raise ValueError(
                    f"invalid range specification {r} in grid, expected 1, 2 or 3 elements"
                )

    def __iter__(self) -> Grid:
        raise RuntimeError("allo.grid can only be used within allo kernels")

    def __next__(self) -> tuple[AlloValue, ...]:
        raise RuntimeError("allo.grid can only be used within allo kernels")


grid = Grid  # name alias for easier usage
