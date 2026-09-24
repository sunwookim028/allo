# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Provenance: ported from Kai Shao's ``tests/dsa/test_recognizer.py`` --
# https://github.com/kkkaishao/allo, branch ``act``, commit ``3c1ad38``.
# The cases that went through his ``ISA.compile_program`` are restated against
# ``source_tag`` directly, because this fork took the recognizer and not the
# matcher it fed. See ``ATTRIBUTION.md``.

"""The recognizer is fail-safe: perturb one attribute, and the op stops matching.

``source_tag`` decides what a TOSA op *means*. Every part of an op's definition it
ignores is a way to compile a program that runs and returns the wrong numbers --
which is what happened upstream: ``tosa.clamp`` was read as relu on ``min_val ==
0`` alone, so relu6 compiled as relu; ``tosa.mul``'s ``shift`` and the matmul /
conv / ``negate`` / ``avg_pool`` zero-points were *dropped* rather than checked, so
a fixed-point multiply selected a float one.

The fork-local cases are the integer ones. Kai's ``_is_relu_clamp`` hard-returned
``False`` on ``IntegerAttr`` bounds, so on an int8/int32 corpus relu itself was
unrecognizable; and quantization is now *carried* on the match rather than only
rejected, so a refusal can name its reason.
"""

import numpy as np
import pytest

from allo._mlir import ir
from allo._mlir.dialects import allo as allo_d
from allo.act import primitive
from allo.act.recognize import (
    _canon,
    normalize_source,
    quantization_of,
    recognize,
    source_tag,
)

FLT_MAX = repr(float(np.finfo(np.float32).max))  # full precision: a rounded
# literal parses to a *smaller* value, which is a bounded clamp and not relu
I32_MAX = 2**31 - 1


_KEEP = []  # parsed modules: an op is a view into its module, so pin them


def _ops(src):
    """Parse a TOSA ``func @main`` and return (module, compute ops)."""
    ctx = ir.Context()
    allo_d.register_dialect(ctx)
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(src)
        _KEEP.append(module)  # its ops are views into it
        block = module.body.operations[0].regions[0].blocks[0]
        ops = [
            op
            for op in block.operations
            if op.operation.name
            not in ("tosa.const", "tosa.const_shape", "func.return")
        ]
        return module, ops


def _tag_of(src):
    """The recognizer's verdict on the last compute op of a parsed module."""
    _module, ops = _ops(src)
    return source_tag(ops[-1])


def _match_of(src):
    _module, ops = _ops(src)
    return recognize(ops[-1])


# ==========================================================================#
# Item 1 -- the registry is the vocabulary
# ==========================================================================#


def test_the_registry_is_29_prims_over_8_categories():
    """The number is not sacred, but a silent change to it is: the registry is what
    ``_NAMED_TAG`` is derived from, so a dropped row silently narrows the
    vocabulary."""
    assert len(primitive.REGISTRY) == 29
    assert len(primitive.CATEGORIES) == 8
    assert {p.category for p in primitive.REGISTRY.values()} <= set(
        primitive.CATEGORIES
    )


def test_every_registry_tag_is_recognized_by_its_tosa_name():
    from allo.act.recognize import _NAMED_TAG

    for tag in primitive.REGISTRY:
        assert _NAMED_TAG[f"tosa.{tag}"] == tag


def test_the_bespoke_ops_are_not_in_the_registry():
    """matmul / transpose / the conv family have irregular shape rules, so they are
    listed separately -- putting them in the registry would claim a calling
    convention they do not have."""
    from allo.act.recognize import _BESPOKE

    assert not set(_BESPOKE) & set(primitive.REGISTRY)


def test_intdiv_keeps_its_op_class_override():
    assert primitive.REGISTRY["intdiv"].tosa_class() == "IntDivOp"
    assert primitive.REGISTRY["add"].tosa_class() == "AddOp"


# ==========================================================================#
# Item 4 -- relu is a bounded tosa.clamp, in float and in integer
# ==========================================================================#


def _clamp(lo, hi, elt="f32", attr=None):
    a = attr or (f"{lo} : {elt}", f"{hi} : {elt}")
    return f"""
func.func @main(%a: tensor<8x{elt}>) -> tensor<8x{elt}> {{
  %r = tosa.clamp %a {{min_val = {a[0]}, max_val = {a[1]}}}
       : (tensor<8x{elt}>) -> tensor<8x{elt}>
  return %r : tensor<8x{elt}>
}}"""


def test_an_open_float_clamp_is_relu():
    assert _tag_of(_clamp(None, None, attr=("0.0 : f32", f"{FLT_MAX} : f32"))) == "relu"


@pytest.mark.parametrize(
    "lo,hi",
    [("0.0 : f32", "6.0 : f32"), ("-1.0 : f32", f"{FLT_MAX} : f32")],
    ids=["relu6-is-not-relu", "min_val"],
)
def test_perturbing_either_float_bound_is_refused(lo, hi):
    assert _tag_of(_clamp(None, None, attr=(lo, hi))) is None


def test_an_integer_clamp_to_the_type_max_is_relu():
    """Kai's version hard-returned False here, so int8/int32 relu -- the only relu
    this fork's corpus has -- was unrecognizable."""
    assert _tag_of(_clamp(None, None, "i32", ("0 : i32", f"{I32_MAX} : i32"))) == "relu"
    assert _tag_of(_clamp(None, None, "i8", ("0 : i8", "127 : i8"))) == "relu"


@pytest.mark.parametrize(
    "lo,hi",
    [("0 : i32", "127 : i32"), ("-1 : i32", f"{I32_MAX} : i32")],
    ids=["saturating-narrow-is-not-relu", "min_val"],
)
def test_perturbing_either_integer_bound_is_refused(lo, hi):
    """``[0, 127]`` on an i32 value is relu *composed with* a saturating narrow --
    which is ``tosa.rescale``'s job, and refusing it is the point."""
    assert _tag_of(_clamp(None, None, "i32", (lo, hi))) is None


def test_a_clamp_that_does_not_propagate_nan_is_not_relu():
    src = f"""
func.func @main(%a: tensor<8xf32>) -> tensor<8xf32> {{
  %r = tosa.clamp %a {{min_val = 0.0 : f32, max_val = {FLT_MAX} : f32,
       nan_mode = #tosa.nan_mode<IGNORE>}}
       : (tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}}"""
    assert _tag_of(src) is None


# ==========================================================================#
# Items 2, 3 -- shift / zero-point operands are checked, not dropped
# ==========================================================================#


def _quantized(q) -> dict:
    """The five ops that carry quantization operands, with ``q`` substituted as the
    shift / zero-point. ``q = 0`` is the neutral (unquantized) form."""
    return {
        "mul": f"""
func.func @main(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<8xi32> {{
  %s = "tosa.const"() {{values = dense<{q}> : tensor<1xi8>}} : () -> tensor<1xi8>
  %r = tosa.mul %a, %b, %s : (tensor<8xi32>, tensor<8xi32>, tensor<1xi8>) -> tensor<8xi32>
  return %r : tensor<8xi32>
}}""",
        "matmul": f"""
func.func @main(%a: tensor<1x4x4xi8>, %b: tensor<1x4x4xi8>) -> tensor<1x4x4xi32> {{
  %z = "tosa.const"() {{values = dense<{q}> : tensor<1xi8>}} : () -> tensor<1xi8>
  %r = tosa.matmul %a, %b, %z, %z : (tensor<1x4x4xi8>, tensor<1x4x4xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4xi32>
  return %r : tensor<1x4x4xi32>
}}""",
        "negate": f"""
func.func @main(%a: tensor<8xi8>) -> tensor<8xi8> {{
  %z = "tosa.const"() {{values = dense<{q}> : tensor<1xi8>}} : () -> tensor<1xi8>
  %r = tosa.negate %a, %z, %z : (tensor<8xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<8xi8>
  return %r : tensor<8xi8>
}}""",
        "conv2d": f"""
func.func @main(%i: tensor<1x4x4x2xi8>, %w: tensor<3x2x2x2xi8>, %b: tensor<3xi32>)
    -> tensor<1x3x3x3xi32> {{
  %z = "tosa.const"() {{values = dense<{q}> : tensor<1xi8>}} : () -> tensor<1xi8>
  %r = "tosa.conv2d"(%i, %w, %b, %z, %z) <{{acc_type = i32,
       dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>,
       stride = array<i64: 1, 1>}}>
       : (tensor<1x4x4x2xi8>, tensor<3x2x2x2xi8>, tensor<3xi32>, tensor<1xi8>,
          tensor<1xi8>) -> tensor<1x3x3x3xi32>
  return %r : tensor<1x3x3x3xi32>
}}""",
        "avg_pool2d": f"""
func.func @main(%x: tensor<1x4x4x2xi8>) -> tensor<1x2x2x2xi8> {{
  %z = "tosa.const"() {{values = dense<{q}> : tensor<1xi8>}} : () -> tensor<1xi8>
  %r = "tosa.avg_pool2d"(%x, %z, %z) <{{acc_type = i32, kernel = array<i64: 2, 2>,
       pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}}>
       : (tensor<1x4x4x2xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x2x2xi8>
  return %r : tensor<1x2x2x2xi8>
}}""",
    }


QUANTIZED_OPS = sorted(_quantized(0))


@pytest.mark.parametrize("tag", QUANTIZED_OPS)
def test_the_unquantized_form_is_recognized(tag):
    """The control: a zero shift / zero-point is the plain op."""
    assert _tag_of(_quantized(0)[tag]) == tag


@pytest.mark.parametrize("tag", QUANTIZED_OPS)
def test_a_nonzero_shift_or_zero_point_is_not_recognized(tag):
    """A quantized op is not its unquantized namesake. Dropping the operand let a
    ``>>3`` fixed-point multiply select a float ``mul`` and run."""
    assert _tag_of(_quantized(3)[tag]) is None


def test_a_nonconstant_zero_point_is_not_recognized():
    """Unknown must read as non-neutral: a zero-point that is a function argument
    cannot be proven zero, so the op is not recognized."""
    src = """
func.func @main(%a: tensor<8xi8>, %z: tensor<1xi8>) -> tensor<8xi8> {
  %r = tosa.negate %a, %z, %z : (tensor<8xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<8xi8>
  return %r : tensor<8xi8>
}"""
    assert _tag_of(src) is None
    m = recognize(_ops(src)[1][-1])
    assert m.tag == "negate" and m.quant.unknown == ("input_zp", "output_zp")
    assert not m.quant.is_symmetric


# ==========================================================================#
# Fork-local: the quantization is carried, and the refusal names its reason
# ==========================================================================#


def test_a_nonzero_zero_point_is_carried_on_the_match():
    """The relaxation this fork needs is not "stop checking"; it is "know what you
    refused". ``recognize`` keeps the tag and the numbers; ``source_tag`` still
    says no."""
    m = _match_of(_quantized(3)["matmul"])
    assert m.tag == "matmul"
    assert m.quant.zero_points == {"a_zp": 3, "b_zp": 3}
    assert not m.quant.is_symmetric
    assert "a_zp=3" in m.quant.why_not_neutral()


def test_a_symmetric_matmul_with_a_nonzero_shift_is_not_possible_but_mul_is():
    """``tosa.mul``'s trailing operand is a *shift*, not a zero-point: symmetric
    (there are no zero-points to be asymmetric) yet not neutral. The two properties
    are separate on purpose -- one is a corpus constraint, the other is the
    rescale opcode."""
    m = _match_of(_quantized(3)["mul"])
    assert m.tag == "mul" and m.quant.shift == 3
    assert m.quant.is_symmetric and not m.quant.is_neutral


def test_the_neutral_matmul_reports_symmetric_zero_points():
    q = quantization_of(_ops(_quantized(0)["matmul"])[1][-1])
    assert q.zero_points == {"a_zp": 0, "b_zp": 0}
    assert q.is_symmetric and q.is_neutral


def test_rescale_is_refused_by_name():
    """Out of scope here on purpose: the opcode is approved as an ``mvout`` mode of
    ``dma_st`` and scoped separately. The refusal has to say so, or the next reader
    reads it as an oversight."""
    src = """
func.func @main(%a: tensor<8xi32>) -> tensor<8xi8> {
  %m = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
  %s = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
  %z = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %y = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %r = tosa.rescale %a, %m, %s, %z, %y {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>,
       scale32 = true, per_channel = false, input_unsigned = false, output_unsigned = false}
       : (tensor<8xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<8xi8>
  return %r : tensor<8xi8>
}"""
    m = _match_of(src)
    assert m.tag is None
    assert "mvout" in m.refusal and "dma_st" in m.refusal


def test_an_unknown_op_is_refused_rather_than_guessed():
    src = """
func.func @main(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<16xi32> {
  %r = tosa.concat %a, %b {axis = 0 : i32} : (tensor<8xi32>, tensor<8xi32>) -> tensor<16xi32>
  return %r : tensor<16xi32>
}"""
    m = _match_of(src)
    assert m.tag is None and "not in the compute vocabulary" in m.refusal


# ==========================================================================#
# Item 5 -- reshape / const transparency
# ==========================================================================#


def test_canon_peels_a_reshape_chain_to_the_underlying_value():
    src = """
func.func @main(%a: tensor<16x8xi8>) -> tensor<1x1x16x8xi8> {
  %s0 = tosa.const_shape {values = dense<[1, 16, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %r0 = tosa.reshape %a, %s0 : (tensor<16x8xi8>, !tosa.shape<3>) -> tensor<1x16x8xi8>
  %s1 = tosa.const_shape {values = dense<[1, 1, 16, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %r1 = tosa.reshape %r0, %s1 : (tensor<1x16x8xi8>, !tosa.shape<4>) -> tensor<1x1x16x8xi8>
  return %r1 : tensor<1x1x16x8xi8>
}"""
    module, _ = _ops(src)
    block = module.body.operations[0].regions[0].blocks[0]
    ret = [op for op in block.operations if op.operation.name == "func.return"][0]
    assert _canon(ret.operands[0]) == block.arguments[0]


def test_layout_and_const_ops_are_the_transparent_set():
    from allo.act.recognize import _LAYOUT_AND_CONST

    assert _LAYOUT_AND_CONST == {"tosa.reshape", "tosa.const", "tosa.const_shape"}


# ==========================================================================#
# Item 6 -- normalize_source brackets torch-mlir's 2-D transpose into 3-D
# ==========================================================================#


def test_normalize_sinks_a_batch_reshape_through_a_2d_transpose():
    """torch lowers ``a @ b.T`` as a 2-D ``tosa.transpose`` then a batch reshape;
    a batched-matmul pattern carries the transpose in 3-D. Without this rewrite the
    two never line up."""
    src = """
func.func @main(%b: tensor<8x32xi8>) -> tensor<1x32x8xi8> {
  %t = tosa.transpose %b {perms = array<i32: 1, 0>} : (tensor<8x32xi8>) -> tensor<32x8xi8>
  %s = tosa.const_shape {values = dense<[1, 32, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %r = tosa.reshape %t, %s : (tensor<32x8xi8>, !tosa.shape<3>) -> tensor<1x32x8xi8>
  return %r : tensor<1x32x8xi8>
}"""
    ctx = ir.Context()
    allo_d.register_dialect(ctx)
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(src)
        normalize_source(module)
        text = str(module)
    assert "array<i32: 0, 2, 1>" in text
    assert "tensor<1x8x32xi8>" in text  # the reshape now precedes the transpose
    assert "array<i32: 1, 0>" not in text


def test_normalize_leaves_a_reshape_that_does_more_than_prepend_units():
    src = """
func.func @main(%b: tensor<8x32xi8>) -> tensor<16x16xi8> {
  %t = tosa.transpose %b {perms = array<i32: 1, 0>} : (tensor<8x32xi8>) -> tensor<32x8xi8>
  %s = tosa.const_shape {values = dense<[16, 16]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %r = tosa.reshape %t, %s : (tensor<32x8xi8>, !tosa.shape<2>) -> tensor<16x16xi8>
  return %r : tensor<16x16xi8>
}"""
    ctx = ir.Context()
    allo_d.register_dialect(ctx)
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(src)
        normalize_source(module)
        assert "array<i32: 1, 0>" in str(module)
