# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recognizer is fail-safe: perturb one attribute, and the op stops matching.

``source_tag`` decides what a TOSA op *means*. Every part of an op's definition it
ignores is a way to compile a program that runs and returns the wrong numbers —
which is what happened: ``tosa.clamp`` was read as relu on ``min_val == 0`` alone,
so relu6 compiled as relu; ``tosa.mul``'s ``shift`` and the matmul / conv /
``negate`` / ``avg_pool`` zero-points were *dropped* rather than checked, so a
fixed-point multiply selected a float one.

Each case builds the same op twice, differing in exactly one attribute: the
unperturbed form must compile (otherwise the test proves nothing), and the
perturbed form must be refused.

Two groups, because the two defects live at different element types. The semantic
attributes below are float-typed and go end to end through the compiler; the
quantization operands are checked at the recognizer, since TOSA's own verifier
already refuses a non-zero shift or zero-point on a float type — the fail-open was
reachable only through *integer* programs, which is exactly where quantized models
live.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from allo._mlir import ir
from allo._mlir.dialects import allo as allo_d, func as func_d, tosa
from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, view
from allo.exp.dsa.core import ISA
from allo.exp.dsa.errors import CompileError, DTypeError, NoMatchError
from allo.exp.dsa.search import source_tag
from allo.lang.core import bf16, f32, i32

FLT_MAX = float(np.finfo(np.float32).max)


# ==========================================================================#
# Source builder: one op, its operands the function arguments
# ==========================================================================#


def _module(arg_shapes, result_shape, emit, elt_name="f32") -> str:
    """A ``func @main`` returning the single value ``emit`` builds from its args."""
    ctx = ir.Context()
    allo_d.register_dialect(ctx)
    with ctx, ir.Location.unknown(ctx):
        elt = (
            ir.F32Type.get()
            if elt_name == "f32"
            else ir.IntegerType.get_signless(int(elt_name[1:]))
        )
        tys = [ir.RankedTensorType.get(s, elt) for s in arg_shapes]
        out_ty = ir.RankedTensorType.get(result_shape, elt)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = func_d.FuncOp("main", ir.FunctionType.get(tys, [out_ty]))
            block = fn.add_entry_block()
        with ir.InsertionPoint(block):
            func_d.ReturnOp([emit(out_ty, *block.arguments)])
        return str(module)


def _arr(*v):
    return ir.DenseI64ArrayAttr.get(list(v))


def _zero(shape):
    """A zero ``tosa.const`` of ``shape`` (a neutral f32 zero-point)."""
    f32t = ir.F32Type.get()
    ty = ir.RankedTensorType.get(shape, f32t)
    return tosa.ConstOp(
        ir.DenseElementsAttr.get_splat(ty, ir.FloatAttr.get(f32t, 0.0))
    ).result


def _clamp(t, x, *, lo=0.0, hi=FLT_MAX):
    f32t = ir.F32Type.get()
    return tosa.ClampOp(
        t, x, ir.FloatAttr.get(f32t, lo), ir.FloatAttr.get(f32t, hi)
    ).result


def _conv(t, i, w, b, *, pad=(1, 0, 0, 1)):
    return tosa.Conv2DOp(
        t,
        i,
        w,
        b,
        _zero([1]),
        _zero([1]),
        pad=_arr(*pad),
        stride=_arr(1, 1),
        dilation=_arr(1, 1),
        acc_type=ir.TypeAttr.get(ir.F32Type.get()),
    ).result


def _maxpool(t, x, *, pad=(1, 0, 0, 1)):
    return tosa.MaxPool2dOp(
        t, x, kernel=_arr(2, 2), stride=_arr(1, 1), pad=_arr(*pad)
    ).result


@dataclass
class Case:
    """One op, one attribute, two values of it."""

    id: str
    args: list
    result: tuple
    emit: object  # (out_ty, *args, **knob) -> ir.Value
    bad: dict
    bad_result: tuple | None = None  # when the perturbation also changes the shape

    def good_source(self) -> str:
        return _module(self.args, self.result, self.emit)

    def bad_source(self) -> str:
        return _module(
            self.args,
            self.bad_result or self.result,
            lambda t, *a: self.emit(t, *a, **self.bad),
        )


CASES = [
    # --- S1: both clamp bounds are semantics ---
    Case("clamp-max_val (relu6 is not relu)", [(8,)], (8,), _clamp, {"hi": 6.0}),
    Case("clamp-min_val", [(8,)], (8,), _clamp, {"lo": -1.0}),
    # --- shape-preserving perturbations: only the semantic check can catch these ---
    Case(
        "transpose-perms",
        [(4, 4, 4)],
        (4, 4, 4),
        lambda t, x, *, perms=(0, 2, 1): tosa.TransposeOp(t, x, list(perms)).result,
        {"perms": (2, 1, 0)},
    ),
    Case(
        "reverse-axis",
        [(4, 4)],
        (4, 4),
        lambda t, x, *, axis=0: tosa.ReverseOp(t, x, axis).result,
        {"axis": 1},
    ),
    Case(
        "conv2d-pad",
        [(1, 4, 4, 2), (3, 2, 2, 2), (3,)],
        (1, 4, 4, 3),
        _conv,
        {"pad": (0, 1, 1, 0)},
    ),
    Case(
        "max_pool2d-pad",
        [(1, 4, 4, 2)],
        (1, 4, 4, 2),
        _maxpool,
        {"pad": (0, 1, 1, 0)},
    ),
    # A reduce's axis is pinned by its result shape — TOSA will not even build the
    # perturbed form at the original shape — so Stage 2 is a second net here. The
    # case still fixes the behaviour: perturbing the axis must not compile.
    Case(
        "reduce_sum-axis",
        [(4, 4)],
        (4, 1),
        lambda t, x, *, axis=1: tosa.ReduceSumOp(x, axis, results=[t]).result,
        {"axis": 0},
        bad_result=(1, 4),
    ),
]


# ==========================================================================#
# An ISA implementing exactly the unperturbed form of every case above
# ==========================================================================#


def _isa() -> ISA:
    isa = ISA("recognizer")
    mem = isa.global_("mem", shape=(4096,), dtype=f32)

    def unary(name, shape, out_shape, fn):
        @isa.instruction(src=mem, dst=mem, name=name)
        def _(I):
            @I.access
            def _(s, d):
                return (view(mem, s, shape), view(mem, d, out_shape))

            @I.compute
            def _(a, o):
                return fn(a)

    unary("vrelu", (8,), (8,), primitive.relu)
    unary("vtrans", (4, 4, 4), (4, 4, 4), lambda a: primitive.transpose(a, [0, 2, 1]))
    unary("vrev", (4, 4), (4, 4), lambda a: primitive.reverse(a, axis=0))
    unary("vrowsum", (4, 4), (4, 1), lambda a: primitive.reduce_sum(a, axis=1))
    unary(
        "vmaxpool",
        (1, 4, 4, 2),
        (1, 4, 4, 2),
        lambda a: primitive.max_pool2d(
            a, kernel=(2, 2), stride=(1, 1), pad=(1, 0, 0, 1)
        ),
    )

    @isa.instruction(src=[mem, mem, mem], dst=mem)
    def vconv(I):
        @I.access
        def _(i, w, b, o):
            return (
                view(mem, i, (1, 4, 4, 2)),
                view(mem, w, (3, 2, 2, 2)),
                view(mem, b, (3,)),
                view(mem, o, (1, 4, 4, 3)),
            )

        @I.compute
        def _(i, w, b, o):
            return primitive.conv2d(
                i, w, b, stride=(1, 1), pad=(1, 0, 0, 1), dilation=(1, 1)
            )

    return isa


ISA_UNDER_TEST = _isa()


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_the_unperturbed_form_matches(case):
    """The control. Without this the rejection test below proves nothing — an ISA
    that matches nothing would pass it trivially."""
    ISA_UNDER_TEST.compile_program(case.good_source())


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_perturbing_one_attribute_is_rejected(case):
    with pytest.raises(CompileError):
        ISA_UNDER_TEST.compile_program(case.bad_source())


def test_an_integer_clamp_is_not_relu():
    """A ``tosa.clamp`` over an integer type carries ``IntegerAttr`` bounds. Reading
    them as ``FloatAttr`` raises ``ValueError`` — an uncaught crash, not a rejection
    — so the recognizer checks the element type before the bounds."""

    def clamp(t, x):
        i32t = ir.IntegerType.get_signless(32)
        lo, hi = ir.IntegerAttr.get(i32t, 0), ir.IntegerAttr.get(i32t, 2**31 - 1)
        return tosa.ClampOp(t, x, lo, hi).result

    src = _module([(8,)], (8,), clamp, elt_name="i32")
    with pytest.raises(NoMatchError):
        ISA_UNDER_TEST.compile_program(src)


# ==========================================================================#
# S4 — shift / zero-point operands are checked, not dropped
# ==========================================================================#


def _tag_of(src: str) -> str | None:
    """The recognizer's verdict on the last compute op of a parsed module."""
    ctx = ir.Context()
    allo_d.register_dialect(ctx)
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(src)  # kept alive: its ops are views into it
        block = module.body.operations[0].regions[0].blocks[0]
        ops = [
            op
            for op in block.operations
            if op.operation.name not in ("tosa.const", "func.return")
        ]
        return source_tag(ops[-1])


def _quantized(q) -> dict:
    """The four ops that carry quantization operands, with ``q`` substituted as the
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


def test_a_quantized_op_is_refused_end_to_end():
    """The link from "not recognized" to "not compiled", with the reason named."""
    isa = _elementwise_isa("i32-mul", i32, n=8)
    isa.compile_program(_quantized(0)["mul"])
    with pytest.raises(NoMatchError, match="non-zero shift / zero-point"):
        isa.compile_program(_quantized(3)["mul"])


# ==========================================================================#
# S3 — the element type is part of the match
# ==========================================================================#


def _elementwise_isa(name, dtype, n=4) -> ISA:
    isa = ISA(name)
    mem = isa.global_("mem", shape=(256,), dtype=dtype)

    def binary(mnemonic, fn):
        @isa.instruction(src=[mem, mem], dst=mem, name=mnemonic)
        def _(I):
            @I.access
            def _(a, b, d):
                return (
                    contiguous(mem, a, n),
                    contiguous(mem, b, n),
                    contiguous(mem, d, n),
                )

            @I.compute
            def _(a, b, o):
                return fn(a, b)

    binary("vadd", primitive.add)
    binary("vmul", primitive.mul)
    return isa


def _add_src(elt="f32", n=4) -> str:
    return _module([(n,), (n,)], (n,), lambda t, a, b: tosa.AddOp(t, a, b).result, elt)


def test_an_integer_program_does_not_compile_onto_a_float_isa():
    """Integer ``add`` wraps and ``intdiv`` truncates: running them on a float
    datapath is a different function, not a rounding difference."""
    isa = _elementwise_isa("f32-add", f32)
    isa.compile_program(_add_src("f32"))
    with pytest.raises(DTypeError, match="is float32 but the source value is i32"):
        isa.compile_program(_add_src("i32"))


def test_a_float_program_does_not_compile_onto_an_integer_isa():
    isa = _elementwise_isa("i32-add", i32)
    isa.compile_program(_add_src("i32"))
    with pytest.raises(DTypeError, match="is int32 but the source value is f32"):
        isa.compile_program(_add_src("f32"))


def test_a_narrower_float_datapath_is_allowed():
    """The one deliberate relaxation: float precision is a property of the hardware,
    not of the program. QKV's whole point is a bf16 datapath running an f32-typed
    source graph — see ``examples/accelerator/qkv``."""
    _elementwise_isa("bf16-add", bf16).compile_program(_add_src("f32"))


def test_a_compiled_program_runs_at_the_io_buffer_dtype():
    """``CompiledProgram.__call__`` staged its inputs through ``np.float32``, which
    silently rounds any integer past 2**24."""
    isa = _elementwise_isa("i32-run", i32)
    prog = isa.compile_program(_add_src("i32"))
    a = np.array([2**24 + 1, 2**24 + 3, -5, 7], np.int32)
    out = prog(a, a)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, a + a)
