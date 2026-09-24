# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A TOSA program in, one ``Workload`` out -- in the same words TOSA uses.

The point of the port is the vocabulary. ``allo/act/workloads.py`` is four
hand-written entries and ``examples/tinytpu/act/spec.py`` spells its epilogue set
``("relu", "saturate")``; neither is standard. These tests pin the other
direction: a TOSA module (the form torch-mlir's TOSA backend emits) is recognized
into exactly the ``Workload`` the registry already holds, and everything the model
cannot express is a *named* refusal rather than an approximation.

Sources are built as MLIR text rather than through torch-mlir, which is not
installed in this environment. The shapes torch-mlir's bracketing produces -- a
2-D argument reshaped to 3-D, a weight reached through ``a @ b.T`` -- are covered
explicitly below.
"""

import numpy as np
import pytest

from allo.act import workloads
from allo.act.errors import NoMatchError, QuantizationError, ShapeError
from allo.act.frontend import workload_from_tosa

I32_MAX = 2**31 - 1
ZP = '  %z = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>'


def _mm(out, a, b, ashape, bshape, oshape, zp="%z"):
    return (
        f"  {out} = tosa.matmul {a}, {b}, {zp}, {zp} : "
        f"(tensor<{ashape}xi8>, tensor<{bshape}xi8>, tensor<1xi8>, tensor<1xi8>) "
        f"-> tensor<{oshape}xi32>"
    )


GEMM = f"""
func.func @main(%a: tensor<1x16x8xi8>, %b: tensor<1x8x32xi8>) -> tensor<1x16x32xi32> {{
{ZP}
{_mm("%r", "%a", "%b", "1x16x8", "1x8x32", "1x16x32")}
  return %r : tensor<1x16x32xi32>
}}"""

GEMM_RELU = f"""
func.func @main(%a: tensor<1x16x8xi8>, %b: tensor<1x8x32xi8>) -> tensor<1x16x32xi32> {{
{ZP}
{_mm("%m", "%a", "%b", "1x16x8", "1x8x32", "1x16x32")}
  %r = tosa.clamp %m {{min_val = 0 : i32, max_val = {I32_MAX} : i32}}
       : (tensor<1x16x32xi32>) -> tensor<1x16x32xi32>
  return %r : tensor<1x16x32xi32>
}}"""

GEMM_SUM = f"""
func.func @main(%a: tensor<1x16x8xi8>, %b: tensor<1x8x32xi8>, %b2: tensor<1x8x32xi8>)
    -> tensor<1x16x32xi32> {{
{ZP}
{_mm("%m0", "%a", "%b", "1x16x8", "1x8x32", "1x16x32")}
{_mm("%m1", "%a", "%b2", "1x16x8", "1x8x32", "1x16x32")}
  %r = tosa.add %m0, %m1 : (tensor<1x16x32xi32>, tensor<1x16x32xi32>) -> tensor<1x16x32xi32>
  return %r : tensor<1x16x32xi32>
}}"""

GEMM_SUM_RELU = GEMM_SUM.replace(
    "  return %r : tensor<1x16x32xi32>\n}",
    f"""  %y = tosa.clamp %r {{min_val = 0 : i32, max_val = {I32_MAX} : i32}}
       : (tensor<1x16x32xi32>) -> tensor<1x16x32xi32>
  return %y : tensor<1x16x32xi32>
}}""",
)


# ==========================================================================#
# The registry's four workloads, recognized from TOSA
# ==========================================================================#


@pytest.mark.parametrize(
    "name,src",
    [
        ("gemm", GEMM),
        ("gemm.relu", GEMM_RELU),
        ("gemm.sum", GEMM_SUM),
        ("gemm.sum.relu", GEMM_SUM_RELU),
    ],
)
def test_a_tosa_program_is_the_registered_workload(name, src):
    """Not "equivalent to": *equal to*. If the recognizer produced a differently
    shaped Workload the mapper would silently see a different problem."""
    assert workload_from_tosa(src, name=name).workload == workloads.get(name)


def test_the_relu_epilogue_is_spelled_in_tosa_s_words():
    """``tosa.clamp`` to ``[0, INT32_MAX]`` is relu. The fork's own epilogue
    vocabulary was ``{"relu"}`` / ``("relu", "saturate")``; this one comes from the
    registry that TOSA's op names are derived from."""
    w = workload_from_tosa(GEMM_RELU).workload
    assert w.epilogue == ("relu",)


def test_the_extents_and_dtypes_the_workload_cannot_hold_are_kept():
    r = workload_from_tosa(GEMM_RELU)
    assert r.extents == {"M": 16, "K": 8, "N": 32}
    assert r.dtypes == {"C": "i32", "A": "i8", "B": "i8"}
    assert r.arg_shapes == (("A", (1, 16, 8)), ("B", (1, 8, 32)))
    assert r.symmetric


def test_a_recognized_workload_evaluates_itself():
    """A spec is its own gold: the Workload the recognizer built runs under
    ``np.einsum`` and agrees with the program it came from."""
    r = workload_from_tosa(GEMM_SUM_RELU, name="gemm.sum.relu")
    rng = np.random.default_rng(0)
    vals = {
        "A": rng.integers(-4, 5, (16, 8), np.int8),
        "B": rng.integers(-4, 5, (8, 32), np.int8),
        "B2": rng.integers(-4, 5, (8, 32), np.int8),
    }
    ref = np.maximum(
        vals["A"].astype(np.int64) @ vals["B"].astype(np.int64)
        + vals["A"].astype(np.int64) @ vals["B2"].astype(np.int64),
        0,
    )
    np.testing.assert_array_equal(r.workload.evaluate(vals), ref)


# ==========================================================================#
# torch-mlir's bracketing: 2-D arguments, and a transposed weight
# ==========================================================================#


def test_a_2d_argument_reshaped_to_3d_is_the_same_workload():
    """torch-mlir wraps a 2-D matmul in reshapes to reach TOSA's batched 3-D
    ``tosa.matmul``. ``_canon`` peels them, so the operand is still the argument."""
    src = f"""
func.func @main(%a: tensor<16x8xi8>, %b: tensor<8x32xi8>) -> tensor<16x32xi32> {{
{ZP}
  %sa = tosa.const_shape {{values = dense<[1, 16, 8]> : tensor<3xindex>}} : () -> !tosa.shape<3>
  %a3 = tosa.reshape %a, %sa : (tensor<16x8xi8>, !tosa.shape<3>) -> tensor<1x16x8xi8>
  %sb = tosa.const_shape {{values = dense<[1, 8, 32]> : tensor<3xindex>}} : () -> !tosa.shape<3>
  %b3 = tosa.reshape %b, %sb : (tensor<8x32xi8>, !tosa.shape<3>) -> tensor<1x8x32xi8>
{_mm("%m", "%a3", "%b3", "1x16x8", "1x8x32", "1x16x32")}
  %so = tosa.const_shape {{values = dense<[16, 32]> : tensor<2xindex>}} : () -> !tosa.shape<2>
  %r = tosa.reshape %m, %so : (tensor<1x16x32xi32>, !tosa.shape<2>) -> tensor<16x32xi32>
  return %r : tensor<16x32xi32>
}}"""
    r = workload_from_tosa(src, name="gemm")
    assert r.workload == workloads.get("gemm")
    assert r.arg_shapes == (("A", (16, 8)), ("B", (8, 32)))


def test_a_transposed_weight_becomes_an_n_k_rank_order():
    """``nn.Linear`` is ``a @ W.T``. The transpose is absorbed into the operand's
    rank order rather than refused, so the weight is one tensor indexed ``(N, K)``
    and ``np.einsum`` does the rest."""
    src = f"""
func.func @main(%a: tensor<1x16x8xi8>, %w: tensor<1x32x8xi8>) -> tensor<1x16x32xi32> {{
{ZP}
  %b = tosa.transpose %w {{perms = array<i32: 0, 2, 1>}} : (tensor<1x32x8xi8>) -> tensor<1x8x32xi8>
{_mm("%r", "%a", "%b", "1x16x8", "1x8x32", "1x16x32")}
  return %r : tensor<1x16x32xi32>
}}"""
    w = workload_from_tosa(src, name="linear").workload
    assert w.operand("B").ranks == ("N", "K")
    rng = np.random.default_rng(1)
    a = rng.integers(-4, 5, (16, 8), np.int8)
    wt = rng.integers(-4, 5, (32, 8), np.int8)
    np.testing.assert_array_equal(
        w.evaluate({"A": a, "B": wt}),
        a.astype(np.int64) @ wt.astype(np.int64).T,
    )


def test_a_2d_transpose_under_a_batch_reshape_is_normalized_then_absorbed():
    """The shape torch-mlir actually emits for ``a @ b.T``: a 2-D transpose and then
    a batch reshape. ``normalize_source`` brackets it into 3-D first."""
    src = f"""
func.func @main(%a: tensor<1x16x8xi8>, %w: tensor<32x8xi8>) -> tensor<1x16x32xi32> {{
{ZP}
  %t = tosa.transpose %w {{perms = array<i32: 1, 0>}} : (tensor<32x8xi8>) -> tensor<8x32xi8>
  %s = tosa.const_shape {{values = dense<[1, 8, 32]> : tensor<3xindex>}} : () -> !tosa.shape<3>
  %b = tosa.reshape %t, %s : (tensor<8x32xi8>, !tosa.shape<3>) -> tensor<1x8x32xi8>
{_mm("%r", "%a", "%b", "1x16x8", "1x8x32", "1x16x32")}
  return %r : tensor<1x16x32xi32>
}}"""
    r = workload_from_tosa(src, name="linear")
    assert r.workload.operand("B").ranks == ("N", "K")
    assert r.arg_shapes == (("A", (1, 16, 8)), ("B", (32, 8)))


# ==========================================================================#
# Refusals -- each one named, none of them an approximation
# ==========================================================================#


def test_a_nonzero_zero_point_is_refused_with_the_reason_stated():
    """Kept on purpose. ``(A-za)(B-zb)`` needs correction terms the MXU cannot
    produce, so the corpus must be per-tensor symmetric int8."""
    src = GEMM.replace("dense<0>", "dense<7>")
    with pytest.raises(QuantizationError, match="symmetric"):
        workload_from_tosa(src)


def test_a_nonconstant_zero_point_is_refused():
    src = """
func.func @main(%a: tensor<1x16x8xi8>, %b: tensor<1x8x32xi8>, %z: tensor<1xi8>)
    -> tensor<1x16x32xi32> {
  %r = tosa.matmul %a, %b, %z, %z : (tensor<1x16x8xi8>, tensor<1x8x32xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x16x32xi32>
  return %r : tensor<1x16x32xi32>
}"""
    with pytest.raises(QuantizationError, match="readable constant"):
        workload_from_tosa(src)


def test_rescale_is_refused_and_names_the_opcode_that_would_accept_it():
    src = f"""
func.func @main(%a: tensor<1x16x8xi8>, %b: tensor<1x8x32xi8>) -> tensor<1x16x32xi8> {{
{ZP}
{_mm("%m", "%a", "%b", "1x16x8", "1x8x32", "1x16x32")}
  %mu = "tosa.const"() {{values = dense<1073741824> : tensor<1xi32>}} : () -> tensor<1xi32>
  %sh = "tosa.const"() {{values = dense<30> : tensor<1xi8>}} : () -> tensor<1xi8>
  %zi = "tosa.const"() {{values = dense<0> : tensor<1xi32>}} : () -> tensor<1xi32>
  %r = tosa.rescale %m, %mu, %sh, %zi, %z {{rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>,
       scale32 = true, per_channel = false, input_unsigned = false, output_unsigned = false}}
       : (tensor<1x16x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>)
       -> tensor<1x16x32xi8>
  return %r : tensor<1x16x32xi8>
}}"""
    with pytest.raises(NoMatchError, match="mvout"):
        workload_from_tosa(src)


def test_a_real_batch_dim_is_refused_rather_than_flattened():
    src = f"""
func.func @main(%a: tensor<4x16x8xi8>, %b: tensor<4x8x32xi8>) -> tensor<4x16x32xi32> {{
{ZP}
{_mm("%r", "%a", "%b", "4x16x8", "4x8x32", "4x16x32")}
  return %r : tensor<4x16x32xi32>
}}"""
    with pytest.raises(ShapeError, match="unit batch dim"):
        workload_from_tosa(src)


def test_two_contractions_that_disagree_on_a_rank_are_refused():
    src = f"""
func.func @main(%a: tensor<1x16x8xi8>, %b: tensor<1x8x32xi8>,
                %a2: tensor<1x16x4xi8>, %b2: tensor<1x4x32xi8>) -> tensor<1x16x32xi32> {{
{ZP}
{_mm("%m0", "%a", "%b", "1x16x8", "1x8x32", "1x16x32")}
{_mm("%m1", "%a2", "%b2", "1x16x4", "1x4x32", "1x16x32")}
  %r = tosa.add %m0, %m1 : (tensor<1x16x32xi32>, tensor<1x16x32xi32>) -> tensor<1x16x32xi32>
  return %r : tensor<1x16x32xi32>
}}"""
    with pytest.raises(ShapeError, match="not one iteration space"):
        workload_from_tosa(src)


def test_an_op_outside_the_workload_grammar_is_refused_by_name():
    """``tosa.sub`` is a perfectly good prim -- it is just not something a
    contraction-plus-epilogue Workload has a node for."""
    src = f"""
func.func @main(%a: tensor<1x16x8xi8>, %b: tensor<1x8x32xi8>, %c: tensor<1x16x32xi32>)
    -> tensor<1x16x32xi32> {{
{ZP}
{_mm("%m", "%a", "%b", "1x16x8", "1x8x32", "1x16x32")}
  %r = tosa.sub %m, %c : (tensor<1x16x32xi32>, tensor<1x16x32xi32>) -> tensor<1x16x32xi32>
  return %r : tensor<1x16x32xi32>
}}"""
    with pytest.raises(NoMatchError, match="sum of contractions"):
        workload_from_tosa(src)


def test_compute_that_does_not_reach_the_result_is_refused():
    """A Workload describes the *whole* program. An op the result does not depend
    on would be silently dropped, and the mapping would be for a smaller program."""
    src = f"""
func.func @main(%a: tensor<1x16x8xi8>, %b: tensor<1x8x32xi8>) -> tensor<1x16x32xi32> {{
{ZP}
{_mm("%m", "%a", "%b", "1x16x8", "1x8x32", "1x16x32")}
  %dead = tosa.abs %m : (tensor<1x16x32xi32>) -> tensor<1x16x32xi32>
  return %m : tensor<1x16x32xi32>
}}"""
    with pytest.raises(NoMatchError, match="not reachable from the result"):
        workload_from_tosa(src)


def test_a_bounded_clamp_is_not_taken_for_a_relu_epilogue():
    """relu6, or a saturating narrow to int8: neither is relu, and neither may
    quietly become the ``relu`` epilogue."""
    src = GEMM_RELU.replace(f"max_val = {I32_MAX} : i32", "max_val = 127 : i32")
    with pytest.raises(NoMatchError, match="a bounded clamp"):
        workload_from_tosa(src)
