# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed compute vocabulary (traced).

``primitive.*`` is the compute vocabulary the matcher recognizes and the oracle
executes. Each returns a ``TensorProxy``; result shapes/dtypes are inferred from
the inputs. ``codegen`` maps these onto value-semantics TOSA ops.

The bulk of the vocabulary is **table-driven**: every entry in ``REGISTRY`` names
a prim, its *category* (which fixes shape/dtype inference here and the TOSA calling
convention in ``codegen``), and whether it commutes (for the matcher). Adding a
prim that fits an existing category is one row — no new branch anywhere. The few
ops with irregular shape/codegen (``identity``/``relu``/``transpose``/``matmul``)
are written out explicitly and are not in ``REGISTRY``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...lang.core import DType, f32, u1
from .core import IndexExpr, TensorProxy
from .errors import AcceleratorDescriptionError

# ==========================================================================#
# Categories + registry
# ==========================================================================#

# Each category fixes (a) shape/dtype inference below and (b) the TOSA calling
# convention in ``codegen._CATEGORY_EMIT``.
UNARY = "unary"  # Op(out, x);            shape = in,        dtype = in
UNARY_ZP = "unary_zp"  # Op(out, x, zp, zp);    shape = in,        dtype = in
BINARY = "binary"  # Op(out, a, b);         shape = a (== b),  dtype = a
BINARY_SHIFT = "binary_shift"  # Op(out, a, b, shift);  shape = a,         dtype = a
COMPARE = "compare"  # bool result;           shape = a,         dtype = u1
SELECT = "select"  # Op(out, cond, a, b);   shape = a,         dtype = a
REDUCE = "reduce"  # Op(in, axis, results); shape = in[axis→1],dtype = in
CAST = "cast"  # Op(out, x);            shape = in,        dtype = target


def _camel(tag: str) -> str:
    return "".join(part.capitalize() for part in tag.split("_")) + "Op"


@dataclass(frozen=True)
class PrimSpec:
    tag: str
    category: str
    commutative: bool = False
    cls: str = ""  # TOSA op class name override (default: Camel(tag) + "Op")

    def tosa_class(self) -> str:
        return self.cls or _camel(self.tag)


_REGISTRY = [
    # binary arithmetic
    PrimSpec("add", BINARY, commutative=True),
    PrimSpec("sub", BINARY),
    PrimSpec("mul", BINARY_SHIFT, commutative=True),
    PrimSpec("maximum", BINARY, commutative=True),
    PrimSpec("minimum", BINARY, commutative=True),
    PrimSpec("pow", BINARY),
    PrimSpec("intdiv", BINARY, cls="IntDivOp"),  # op name tosa.intdiv, class IntDivOp
    # unary math / activations
    PrimSpec("abs", UNARY),
    PrimSpec("exp", UNARY),
    PrimSpec("log", UNARY),
    PrimSpec("rsqrt", UNARY),
    PrimSpec("reciprocal", UNARY),
    PrimSpec("floor", UNARY),
    PrimSpec("ceil", UNARY),
    PrimSpec("sin", UNARY),
    PrimSpec("cos", UNARY),
    PrimSpec("tanh", UNARY),
    PrimSpec("sigmoid", UNARY),
    PrimSpec("erf", UNARY),
    PrimSpec("negate", UNARY_ZP),
    # comparison + select
    PrimSpec("equal", COMPARE, commutative=True),
    PrimSpec("greater", COMPARE),
    PrimSpec("greater_equal", COMPARE),
    PrimSpec("select", SELECT),
    # reductions
    PrimSpec("reduce_sum", REDUCE),
    PrimSpec("reduce_max", REDUCE),
    PrimSpec("reduce_min", REDUCE),
    PrimSpec("reduce_product", REDUCE),
    # type conversion
    PrimSpec("cast", CAST),
]

REGISTRY: dict[str, PrimSpec] = {p.tag: p for p in _REGISTRY}


# ==========================================================================#
# Shape helpers
# ==========================================================================#


def _same_dtype(tag: str, a: TensorProxy, b: TensorProxy) -> None:
    if a.dtype != b.dtype:
        raise AcceleratorDescriptionError(
            f"{tag}: dtype mismatch {a.dtype} vs {b.dtype}"
        )


def _check_axis(tag: str, a: TensorProxy, axis: int) -> None:
    if not 0 <= axis < len(a.shape):
        raise AcceleratorDescriptionError(
            f"{tag}: axis {axis} out of range for {a.shape}"
        )


def _conflict(x, y) -> bool:
    """Two shape dims definitely disagree only when both are statically known and
    differ. A symbolic (parametric) dim is left for ``solve`` to constrain, so it
    never conflicts here — equal-looking ``IndexExpr`` objects are distinct values."""
    xs = x.static_int() if isinstance(x, IndexExpr) else x
    ys = y.static_int() if isinstance(y, IndexExpr) else y
    return xs is not None and ys is not None and xs != ys


def _bcast_dim(x, y):
    """Broadcast one dim pair (TOSA/NumPy rule): a static ``1`` yields the other
    side; otherwise the two must not statically conflict (a symbolic dim is left for
    ``solve``)."""
    xs = x.static_int() if isinstance(x, IndexExpr) else x
    ys = y.static_int() if isinstance(y, IndexExpr) else y
    if xs == 1:
        return y
    if ys == 1:
        return x
    if _conflict(x, y):
        raise AcceleratorDescriptionError(f"shapes not broadcastable: {x} vs {y}")
    return x


def _broadcast(a, b) -> tuple:
    """TOSA-style elementwise broadcast shape: equal rank, each dim equal or one
    side ``1`` (a size-1 dim stretches). This is how ``codegen`` lowers (TOSA
    elementwise ops broadcast) and how a lowered source program spells it — no
    explicit broadcast node, so the matcher stays isomorphic."""
    if len(a) != len(b):
        raise AcceleratorDescriptionError(f"rank mismatch {a} vs {b}")
    return tuple(_bcast_dim(x, y) for x, y in zip(a, b))


# ==========================================================================#
# Category constructors (one per inference rule)
# ==========================================================================#


def _unary(tag: str, a: TensorProxy) -> TensorProxy:
    return TensorProxy(tag, a.dtype, a.shape, args=(a,))


def _binary(tag: str, a: TensorProxy, b: TensorProxy) -> TensorProxy:
    _same_dtype(tag, a, b)
    return TensorProxy(tag, a.dtype, _broadcast(a.shape, b.shape), args=(a, b))


def _compare(tag: str, a: TensorProxy, b: TensorProxy) -> TensorProxy:
    _same_dtype(tag, a, b)
    return TensorProxy(tag, u1, _broadcast(a.shape, b.shape), args=(a, b))  # bool out


def _reduce(tag: str, a: TensorProxy, axis: int) -> TensorProxy:
    _check_axis(tag, a, axis)
    shape = list(a.shape)
    shape[axis] = 1  # TOSA reduce keeps the reduced dim (size 1)
    return TensorProxy(tag, a.dtype, tuple(shape), args=(a,), axis=axis)


# ==========================================================================#
# Public vocabulary
# ==========================================================================#

# --- bespoke prims (irregular shape/codegen; not in REGISTRY) ---


def const(value, dtype: DType = f32, shape=(1,)) -> TensorProxy:
    """A literal baked into the instruction — the compute DAG's second kind of leaf.

    Not every operand of a real instruction comes from a buffer: MiniNPU's ``vexp``
    computes ``2**x`` from one register, so the ``2`` is part of the *instruction*,
    not of the program. Without this, such an instruction can only be described by
    lying about its arity (an extra buffer operand nobody supplies) or about its
    semantics (base-e for base-2) — so it was left undescribed instead.

    ``value`` is either a number — a **fixed** literal, which the matcher compares by
    value (rounded through ``dtype``), making it load-bearing for selection exactly
    like a transpose's permutation — or a ``ScalarProxy``, i.e. one of ``@I.compute``'s
    extra params: a **parametric** literal (ACT's α), which the matcher *binds* from
    whatever constant the source supplies and emits in the instruction word. The two
    are one construct because they are one thing to the hardware, an immediate; the
    ISA says whether that immediate is wired or encoded.

    ``shape`` defaults to rank 1, which is how TOSA spells a broadcast scalar
    (``tensor<1xf32>``) and what torch's backend emits; give it explicitly to
    broadcast against a higher-rank operand."""
    return TensorProxy("const", dtype, tuple(shape), value=value)


def identity(a: TensorProxy) -> TensorProxy:
    return TensorProxy("identity", a.dtype, a.shape, args=(a,))


def relu(a: TensorProxy) -> TensorProxy:
    return TensorProxy("relu", a.dtype, a.shape, args=(a,))  # codegen: clamp(min=0)


def transpose(a: TensorProxy, permutation) -> TensorProxy:
    """Permute a tensor's dims. A value-reordering relayout, so it lives in the
    compute vocabulary (semantics), not in the access patterns."""
    perm = list(permutation)
    if sorted(perm) != list(range(len(a.shape))):
        raise AcceleratorDescriptionError(
            f"transpose: {perm} is not a permutation of {len(a.shape)} dims"
        )
    shape = tuple(a.shape[p] for p in perm)
    return TensorProxy("transpose", a.dtype, shape, args=(a,), permutation=perm)


def matmul(a: TensorProxy, b: TensorProxy) -> TensorProxy:
    # Batched, matching TOSA's matmul: (B, M, K) x (B, K, N) -> (B, M, N).
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise AcceleratorDescriptionError("matmul expects batched 3-D operands")
    if _conflict(a.shape[0], b.shape[0]):
        raise AcceleratorDescriptionError(
            f"matmul: batch mismatch {a.shape} x {b.shape}"
        )
    if _conflict(a.shape[2], b.shape[1]):
        raise AcceleratorDescriptionError(
            f"matmul: inner dims mismatch {a.shape} x {b.shape}"
        )
    return TensorProxy(
        "matmul", a.dtype, (a.shape[0], a.shape[1], b.shape[2]), args=(a, b)
    )


def reverse(a: TensorProxy, axis: int) -> TensorProxy:
    """Reverse ``a`` along ``axis``. Like transpose, a value-reordering relayout, so
    it is a compute prim (not an access pattern); shape is preserved."""
    _check_axis("reverse", a, axis)
    return TensorProxy("reverse", a.dtype, a.shape, args=(a,), axis=axis)


# --- contraction / conv family (bespoke: bias + zero-points + spatial inference) ---
# TOSA layout: conv2d input NHWC, weight OHWI; depthwise weight HWCM; pool NHWC.
# pad is [top, bottom, left, right]; stride / dilation / kernel are [h, w].


def _spatial(i: int, k: int, pad_lo: int, pad_hi: int, stride: int, dil: int) -> int:
    return (i + pad_lo + pad_hi - dil * (k - 1) - 1) // stride + 1


def _conv_attrs(pad, stride, dilation) -> dict:
    return {"pad": list(pad), "stride": list(stride), "dilation": list(dilation)}


def conv2d(inp, weight, bias, *, stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1)):
    """2-D convolution. ``inp`` is NHWC, ``weight`` OHWI, ``bias`` per-output-channel;
    ``stride`` / ``dilation`` are ``[h, w]`` and ``pad`` is ``[top, bottom, left,
    right]``. Result is NHWC ``(N, OH, OW, OC)``."""
    n, ih, iw, _ic = inp.shape
    oc, kh, kw, _ = weight.shape
    oh = _spatial(ih, kh, pad[0], pad[1], stride[0], dilation[0])
    ow = _spatial(iw, kw, pad[2], pad[3], stride[1], dilation[1])
    return TensorProxy(
        "conv2d",
        inp.dtype,
        (n, oh, ow, oc),
        args=(inp, weight, bias),
        attrs=_conv_attrs(pad, stride, dilation),
    )


def depthwise_conv2d(
    inp, weight, bias, *, stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1)
):
    """Depthwise 2-D convolution: each input channel convolved by its own ``m``
    filters. ``inp`` is NHWC, ``weight`` HWCM (``m`` filters per channel); same
    ``stride`` / ``pad`` / ``dilation`` as ``conv2d``. Result is ``(N, OH, OW, C*m)``.
    """
    n, ih, iw, c = inp.shape
    kh, kw, _c, m = weight.shape  # HWCM
    oh = _spatial(ih, kh, pad[0], pad[1], stride[0], dilation[0])
    ow = _spatial(iw, kw, pad[2], pad[3], stride[1], dilation[1])
    return TensorProxy(
        "depthwise_conv2d",
        inp.dtype,
        (n, oh, ow, c * m),
        args=(inp, weight, bias),
        attrs=_conv_attrs(pad, stride, dilation),
    )


def _pool(tag, inp, kernel, stride, pad):
    n, ih, iw, c = inp.shape
    oh = _spatial(ih, kernel[0], pad[0], pad[1], stride[0], 1)
    ow = _spatial(iw, kernel[1], pad[2], pad[3], stride[1], 1)
    return TensorProxy(
        tag,
        inp.dtype,
        (n, oh, ow, c),
        args=(inp,),
        attrs={"kernel": list(kernel), "stride": list(stride), "pad": list(pad)},
    )


def max_pool2d(inp, *, kernel, stride=(1, 1), pad=(0, 0, 0, 0)):
    """Max pooling over NHWC ``inp``. ``kernel`` / ``stride`` are ``[h, w]``, ``pad``
    is ``[top, bottom, left, right]``. Result is NHWC, same channel count."""
    return _pool("max_pool2d", inp, kernel, stride, pad)


def avg_pool2d(inp, *, kernel, stride=(1, 1), pad=(0, 0, 0, 0)):
    """Average pooling over NHWC ``inp`` (same arguments as ``max_pool2d``)."""
    return _pool("avg_pool2d", inp, kernel, stride, pad)


# --- binary arithmetic ---


def add(a, b):
    return _binary("add", a, b)


def sub(a, b):
    return _binary("sub", a, b)


def mul(a, b):
    return _binary("mul", a, b)


def maximum(a, b):
    return _binary("maximum", a, b)


def minimum(a, b):
    return _binary("minimum", a, b)


def pow(a, b):
    return _binary("pow", a, b)


def intdiv(a, b):
    return _binary("intdiv", a, b)


# --- unary math / activations ---


def abs(a):
    return _unary("abs", a)


def exp(a):
    return _unary("exp", a)


def log(a):
    return _unary("log", a)


def rsqrt(a):
    return _unary("rsqrt", a)


def reciprocal(a):
    return _unary("reciprocal", a)


def floor(a):
    return _unary("floor", a)


def ceil(a):
    return _unary("ceil", a)


def sin(a):
    return _unary("sin", a)


def cos(a):
    return _unary("cos", a)


def tanh(a):
    return _unary("tanh", a)


def sigmoid(a):
    return _unary("sigmoid", a)


def erf(a):
    return _unary("erf", a)


def negate(a):
    return _unary("negate", a)


# --- comparison + select ---


def equal(a, b):
    return _compare("equal", a, b)


def greater(a, b):
    return _compare("greater", a, b)


def greater_equal(a, b):
    return _compare("greater_equal", a, b)


def select(cond: TensorProxy, a: TensorProxy, b: TensorProxy) -> TensorProxy:
    _same_dtype("select", a, b)
    return TensorProxy(
        "select", a.dtype, _broadcast(a.shape, b.shape), args=(cond, a, b)
    )


# --- reductions (axis is part of the semantics, like transpose's permutation) ---


def reduce_sum(a, axis):
    return _reduce("reduce_sum", a, axis)


def reduce_max(a, axis):
    return _reduce("reduce_max", a, axis)


def reduce_min(a, axis):
    return _reduce("reduce_min", a, axis)


def reduce_product(a, axis):
    return _reduce("reduce_product", a, axis)


# --- type conversion ---


def cast(a: TensorProxy, dtype: DType) -> TensorProxy:
    return TensorProxy("cast", dtype, a.shape, args=(a,))
