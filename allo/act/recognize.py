# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Provenance: cherry-picked from Kai Shao's ACT work --
# https://github.com/kkkaishao/allo, branch ``act``, commit ``3c1ad38``,
# file ``allo/exp/dsa/search.py``. Ported pieces and their original lines:
# the bespoke-op table and ``_LAYOUT_AND_CONST`` / ``_canon`` (95-121), the
# source recognizer ``_DATA_OPERANDS`` / ``_source_ins`` / ``const_elements`` /
# ``source_tag`` (157-227), ``_FLOAT_MAX`` and ``_is_relu_clamp`` (169-175,
# 229-246), and ``normalize_source`` (338-389). ``instruction_pattern``
# (131-155) was deliberately left behind: it reaches into ACT's ``Instruction``
# and is not portable. See ``ATTRIBUTION.md``.

"""Recognize a TOSA op as a prim of the compute vocabulary.

``source_tag`` decides what a TOSA op *means*. Every part of an op's definition it
ignores is a way to compile a program that runs and returns the wrong numbers --
which is what happened in Kai's tree: ``tosa.clamp`` was read as relu on
``min_val == 0`` alone, so relu6 compiled as relu; ``tosa.mul``'s ``shift`` and the
matmul / conv / ``negate`` / ``avg_pool`` zero-points were *dropped* rather than
checked, so a fixed-point multiply selected a float one. Recognition is therefore
**fail-safe**: an op earns a tag only when every part of its definition is
accounted for, and anything unmodeled costs a clean "no instruction matches"
rather than a program that compiles and computes a different function.

Two things differ from the original, both because this fork is int8/int32 where
every ACT example ISA was float with zero integer:

**Quantization is carried, not merely rejected.** :func:`recognize` returns a
:class:`Match` holding the op's shift and zero-point *values*, read through
:func:`const_elements`. ``source_tag`` still refuses anything non-neutral -- the
check is not deleted -- but the numbers survive the recognizer so a later stage can
name what it refused and, once the hardware exists, honour it.

**The refusal of non-zero zero-points is kept on purpose.** It is not an oversight
to be relaxed later. ``(A - za)(B - zb)`` expands to ``AB - za*B - zb*A + za*zb``,
and the three correction terms are row/column sums our MXU cannot produce: it
contracts two int8 operands into an int32 accumulator and has no path to a
per-row or per-column reduction of an operand on its own. Requiring ``za == zb ==
0`` means requiring **per-tensor symmetric** int8 quantization, which is what
PyTorch's symmetric observers emit. That is a constraint on the corpus, not a
change to the RTL, and it is stated here and in
``docs/source/extensions/act.rst`` rather than silently assumed.

``tosa.rescale`` is out of scope here and refused by name. A hardware opcode for
it has been approved, to live in ``dma_st`` as an ``mvout`` mode; it is scoped
separately, and this module must not paper over its absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ml_dtypes
import numpy as np

from .._mlir import ir
from .._mlir.dialects import tosa
from . import primitive

# ==========================================================================#
# The vocabulary: registry-derived names plus the bespoke ones
# ==========================================================================#

# Source op-name -> prim tag. The source is value-semantics TOSA throughout, and a
# prim's source op name is always `tosa.<tag>`, so the map is derived from the prim
# registry. relu is recognized separately (it is a tosa.clamp with min == 0 and an
# open upper bound, not a tosa.relu). tosa.matmul is batched 3-D (its 2-D<->3-D
# reshapes are handled by `_canon`).
_BESPOKE = (
    "matmul",
    "transpose",
    "reverse",
    "conv2d",
    "depthwise_conv2d",
    "max_pool2d",
    "avg_pool2d",
)

_NAMED_TAG = {f"tosa.{tag}": tag for tag in primitive.REGISTRY}
for _b in _BESPOKE:
    _NAMED_TAG[f"tosa.{_b}"] = _b

# Pure layout / constant ops: transparent to matching (reshape is an alias; consts
# carry no compute). `_canon` peels reshapes; use-counting skips all of these.
_LAYOUT_AND_CONST = {"tosa.reshape", "tosa.const", "tosa.const_shape"}


def _canon(value):
    """Peel ``tosa.reshape`` chains to the underlying value. Reshape is a layout
    alias (e.g. TOSA's batched-matmul 2-D<->3-D wrapping at I/O), transparent for
    both matching and allocation -- two reshapes of one value share its slot."""
    while True:
        owner = value.owner
        if isinstance(owner, ir.Block) or owner.operation.name != "tosa.reshape":
            return value
        value = owner.operands[0]  # reshape input1 (the data operand)


# ==========================================================================#
# Quantization: trailing non-data operands
# ==========================================================================#

# Recognized source ops carrying trailing *non-data* operands: the count of leading
# data operands. Everything after them -- tosa.mul's shift, the matmul / conv /
# negate / avg_pool zero-points -- is quantization. They are not dropped: they are
# read and carried on the Match, and required to be neutral before an op earns a
# tag, else a `>>3` fixed-point multiply would select a plain `mul` and run.
_DATA_OPERANDS = {
    "tosa.mul": 2,  # + shift
    "tosa.matmul": 2,  # + a_zp, b_zp
    "tosa.negate": 1,  # + input_zp, output_zp
    "tosa.conv2d": 3,  # input, weight, bias (+ input_zp, weight_zp)
    "tosa.depthwise_conv2d": 3,
    "tosa.avg_pool2d": 1,  # + input_zp, output_zp
}

# What each trailing operand means, in order. Only tosa.mul's is a shift; every
# other one is a zero-point, and it is the zero-points the symmetric-quantization
# constraint is about.
_QUANT_ROLES = {
    "tosa.mul": ("shift",),
    "tosa.matmul": ("a_zp", "b_zp"),
    "tosa.negate": ("input_zp", "output_zp"),
    "tosa.conv2d": ("input_zp", "weight_zp"),
    "tosa.depthwise_conv2d": ("input_zp", "weight_zp"),
    "tosa.avg_pool2d": ("input_zp", "output_zp"),
}


def _source_ins(op) -> list:
    """The data-input operands of a recognized (value-semantics TOSA) source op."""
    n = _DATA_OPERANDS.get(op.operation.name)
    return list(op.operands)[:n] if n is not None else list(op.operands)


def const_elements(value) -> list | None:
    """The elements of a source ``tosa.const``, or ``None`` if ``value`` is not a
    constant we can read.

    ``None`` covers both "not a constant" and "a constant whose data lives in a
    ``dialect_resource`` blob" -- torch's TOSA backend stores model weights that way
    and the Python bindings expose no reader. Callers must treat unknown as
    *unusable*, never as a default value."""
    owner = value.owner
    if isinstance(owner, ir.Block) or owner.operation.name != "tosa.const":
        return None
    attr = owner.operation.attributes["values"]
    if not isinstance(attr, (ir.DenseFPElementsAttr, ir.DenseIntElementsAttr)):
        return None
    return list(attr)


def _scalar_const(value):
    """The single value of a splat ``tosa.const``, or ``None`` if unreadable.

    A shift or zero-point is per-tensor in the forms we accept, so a constant with
    more than one distinct element is *unknown* rather than an error: unknown reads
    as non-neutral everywhere below."""
    elems = const_elements(value)
    if not elems or len(set(elems)) != 1:
        return None
    return elems[0]


@dataclass(frozen=True)
class Quantization:
    """An op's trailing shift / zero-point operands, by role.

    ``unknown`` names the operands that are not readable constants -- a zero-point
    passed as a function argument, or a weight living in a ``dialect_resource``
    blob. Unknown is never treated as zero."""

    values: dict = field(default_factory=dict)
    unknown: tuple = ()

    @property
    def shift(self) -> int:
        return self.values.get("shift", 0)

    @property
    def zero_points(self) -> dict:
        return {k: v for k, v in self.values.items() if k.endswith("zp")}

    @property
    def is_symmetric(self) -> bool:
        """Every zero-point is a readable zero: the corpus constraint this fork
        requires (see the module docstring)."""
        if any(r.endswith("zp") for r in self.unknown):
            return False
        return all(v == 0 for v in self.zero_points.values())

    @property
    def is_neutral(self) -> bool:
        """Symmetric *and* unshifted: the op is its unquantized namesake."""
        return not self.unknown and all(v == 0 for v in self.values.values())

    def why_not_neutral(self) -> str:
        if self.unknown:
            return (
                f"non-zero shift / zero-point: {', '.join(self.unknown)} is not a "
                f"readable constant, so it cannot be proven zero"
            )
        bad = sorted(k for k, v in self.values.items() if v != 0)
        return f"non-zero shift / zero-point: {', '.join(f'{k}={self.values[k]}' for k in bad)}"


def quantization_of(op) -> Quantization:
    """Read an op's trailing shift / zero-point operands into a :class:`Quantization`."""
    name = op.operation.name
    n = _DATA_OPERANDS.get(name)
    if n is None:
        return Quantization()
    roles = _QUANT_ROLES[name]
    values, unknown = {}, []
    for role, operand in zip(roles, list(op.operands)[n:]):
        v = _scalar_const(operand)
        if v is None:
            unknown.append(role)
        else:
            values[role] = v
    return Quantization(values=values, unknown=tuple(unknown))


def _is_zero_const(value) -> bool:
    """True only if ``value`` is *provably* an all-zero constant."""
    elems = const_elements(value)
    return elems is not None and all(v == 0 for v in elems)


def _quantization_is_neutral(op) -> bool:
    """Whether ``op``'s trailing shift / zero-point operands are all zero."""
    n = _DATA_OPERANDS.get(op.operation.name)
    return n is None or all(_is_zero_const(v) for v in list(op.operands)[n:])


# ==========================================================================#
# relu: a bounded tosa.clamp
# ==========================================================================#

# The largest finite value of each float type. torch_mlir's TOSA backend spells
# relu's *open* upper bound as this value rather than +inf, so both count as "does
# not clip from above".
_FLOAT_MAX = {
    "f16": float(np.finfo(np.float16).max),
    "bf16": float(ml_dtypes.finfo(ml_dtypes.bfloat16).max),
    "f32": float(np.finfo(np.float32).max),
    "f64": float(np.finfo(np.float64).max),
}


def _int_max(elt) -> int | None:
    """The largest value of a signless integer type, read as signed -- the integer
    analogue of ``_FLOAT_MAX``. TOSA integer types are signless and interpreted
    signed by every op in the vocabulary."""
    if not isinstance(elt, ir.IntegerType) or elt.width < 2:
        return None
    return (1 << (elt.width - 1)) - 1


def _nan_mode_propagates(op) -> bool:
    """A ``tosa.clamp`` with ``nan_mode = IGNORE`` is not relu: it returns the
    bound rather than the NaN. Kai's version did not look at this attribute; it is
    free to check and it is semantics."""
    attrs = op.operation.attributes
    if "nan_mode" not in attrs:
        return True
    return "PROPAGATE" in str(attrs["nan_mode"])


def _is_relu_clamp(op) -> bool:
    """``tosa.clamp`` is relu only when it clamps to ``[0, max)`` -- the form
    torch_mlir's TOSA backend emits for ``aten.relu``.

    *Both* bounds are semantics. Checking only the lower one recognized relu6
    (``[0, 6]``) as relu, which compiled and ran and returned wrong numbers.

    The integer branch is this fork's. Kai's version hard-returned ``False`` for
    ``IntegerAttr`` bounds, because reading them as ``FloatAttr`` raises and his
    corpus was entirely float -- so on an int8/int32 corpus *relu itself* was
    unrecognizable, not merely quantized relu. An integer clamp is relu on the same
    reading: lower bound 0, upper bound at or above the element type's maximum. A
    clamp to ``[0, 127]`` on an i32 value is therefore **not** relu but relu
    composed with a saturating narrow -- that is ``tosa.rescale``'s job, out of
    scope here, and refusing it is the point."""
    elt = ir.ShapedType(op.operands[0].type).element_type
    attrs = op.operation.attributes
    if not _nan_mode_propagates(op):
        return False
    if str(elt) in _FLOAT_MAX:
        return (
            ir.FloatAttr(attrs["min_val"]).value == 0.0
            and ir.FloatAttr(attrs["max_val"]).value >= _FLOAT_MAX[str(elt)]
        )
    hi = _int_max(elt)
    if hi is None:
        return False
    return (
        ir.IntegerAttr(attrs["min_val"]).value == 0
        and ir.IntegerAttr(attrs["max_val"]).value >= hi
    )


# ==========================================================================#
# The recognizer
# ==========================================================================#


@dataclass(frozen=True)
class Match:
    """What the recognizer made of one source op.

    ``tag`` is ``None`` when the op is not in the vocabulary at all, and ``refusal``
    then says why. A recognized op keeps its ``tag`` even when its quantization is
    not neutral: the numbers are on ``quant``, and it is the *caller* that decides
    whether it can honour them. ``source_tag`` cannot, and returns ``None``."""

    tag: str | None
    ins: tuple = ()
    quant: Quantization = field(default_factory=Quantization)
    refusal: str = ""

    def __bool__(self) -> bool:
        return self.tag is not None


def recognize(op) -> Match:
    """Recognize a source op, carrying its quantization. Never raises.

    The wider entry point: :func:`source_tag` is this with the fail-safe policy
    applied on top."""
    name = op.operation.name
    if name == "tosa.rescale":
        return Match(
            None,
            refusal=(
                "tosa.rescale: requantization is not in the compute vocabulary. A "
                "hardware opcode for it has been approved -- an `mvout` mode of "
                "`dma_st` -- and is scoped as separate work; until it exists, "
                "refusing is the only honest answer."
            ),
        )
    tag = _NAMED_TAG.get(name)
    if tag is None:
        if name != "tosa.clamp":
            return Match(None, refusal=f"{name} is not in the compute vocabulary")
        if not _is_relu_clamp(op):
            return Match(
                None,
                refusal=(
                    "tosa.clamp is relu only when it clamps to [0, max); a bounded "
                    "clamp (relu6, or a saturating narrow) is a different function"
                ),
            )
        tag = "relu"
    quant = quantization_of(op)
    return Match(
        tag,
        tuple(_source_ins(op)),
        quant,
        "" if quant.is_neutral else quant.why_not_neutral(),
    )


def source_tag(op) -> str | None:
    """Recognize a source op into a prim tag, or ``None`` if unsupported.

    Fail-safe: an op earns a tag only when every part of its definition is
    accounted for by a prim. Anything unmodeled -- a non-zero shift or zero-point,
    a clamp that is not relu, a rescale -- yields ``None``, which costs a clean
    "no instruction matches" rather than a program that computes a different
    function."""
    m = recognize(op)
    return m.tag if m and m.quant.is_neutral else None


# ==========================================================================#
# Source normalization (run on the parsed TOSA before matching)
# ==========================================================================#


def entry_block(module):
    """The single ``func.func`` of a source module, and its entry block."""
    for op in module.body.operations:
        if op.operation.name == "func.func":
            return op, op.regions[0].blocks[0]
    raise ValueError("source module has no func.func")


def _const_shape(shape):
    n = len(shape)
    vals = ir.Attribute.parse(
        f"dense<[{', '.join(str(d) for d in shape)}]> : tensor<{n}xindex>"
    )
    return tosa.ConstShapeOp(ir.Type.parse(f"!tosa.shape<{n}>"), vals).result


def normalize_source(module):
    """Canonicalize torch_mlir's TOSA so instruction patterns can match it.

    Sinks ``reshape(transpose(X, p))`` -> ``transpose(reshape(X), p')`` when the
    reshape only prepends unit (batch) dims. torch lowers ``a @ b.T`` as a 2-D
    ``tosa.transpose`` (perms ``[1, 0]``) *then* a batch reshape to 3-D, whereas an
    instruction's semantics carry the weight transpose in batched 3-D form (perms
    ``[0, 2, 1]``). Without this rewrite the two never line up and the systolic
    matmul cannot absorb the transpose."""
    _, block = entry_block(module)
    for op in list(block.operations):
        if op.operation.name != "tosa.reshape":
            continue
        t = op.operands[0].owner
        if isinstance(t, ir.Block) or t.operation.name != "tosa.transpose":
            continue
        in_ty = ir.RankedTensorType(t.operands[0].type)
        t_out = ir.RankedTensorType(t.results[0].type)
        r_out = ir.RankedTensorType(op.results[0].type)
        k = r_out.rank - t_out.rank  # number of prepended dims
        if (
            k <= 0
            or list(r_out.shape[:k]) != [1] * k
            or list(r_out.shape[k:]) != list(t_out.shape)
        ):
            continue  # reshape does more than prepend unit dims -> leave it alone
        perms = list(ir.DenseI32ArrayAttr(t.operation.attributes["perms"]))
        new_perms = list(range(k)) + [p + k for p in perms]
        new_shape = [1] * k + list(in_ty.shape)
        with ir.InsertionPoint(op), ir.Location.unknown():
            reshaped = tosa.ReshapeOp(
                t.operands[0],
                _const_shape(new_shape),
                results=[ir.RankedTensorType.get(new_shape, in_ty.element_type)],
            )
            new_t = tosa.TransposeOp(r_out, reshaped.result, new_perms)
        op.results[0].replace_all_uses_with(new_t.result)
        op.operation.erase()
        if not list(
            t.results[0].uses
        ):  # the old transpose precedes op, already visited
            t.operation.erase()
    return module


def perms_of(op) -> list:
    """The permutation of a ``tosa.transpose`` source op (an ``array<i32>`` attr)."""
    return list(ir.DenseI32ArrayAttr(op.operation.attributes["perms"]))
