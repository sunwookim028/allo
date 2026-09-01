# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-1 prim vocabulary: one oracle round-trip per category (proving the
registry-driven codegen lowers and runs), plus a few instruction-selection matches
(proving the registry-derived recognizer in the search backend)."""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, view
from allo.exp.dsa.core import ISA
from allo.lang.core import f32, i32

# ==========================================================================#
# An ISA exercising every Tier-1 category, over a flat fp32 scratchpad (+ an
# i32 buffer for the cast destination).
# ==========================================================================#

isa = ISA("PrimTest")
mem = isa.scalar("mem", slots=256, dtype=f32)
qmem = isa.scalar("qmem", slots=256, dtype=i32)


def _unary(mnemonic, fn):
    @isa.instruction(src=mem, dst=mem, name=mnemonic)
    def _(I):
        @I.access
        def _(s, d):
            return (contiguous(mem, s, 8), contiguous(mem, d, 8))

        @I.compute
        def _(a, o):
            return fn(a)

    return isa._ops[mnemonic]


def _binary(mnemonic, fn):
    @isa.instruction(src=[mem, mem], dst=mem, name=mnemonic)
    def _(I):
        @I.access
        def _(a, b, d):
            return (contiguous(mem, a, 8), contiguous(mem, b, 8), contiguous(mem, d, 8))

        @I.compute
        def _(a, b, o):
            return fn(a, b)

    return isa._ops[mnemonic]


vexp = _unary("vexp", primitive.exp)
vneg = _unary("vneg", primitive.negate)
vmax = _binary("vmax", primitive.maximum)
vselmax = _binary(
    "vselmax", lambda a, b: primitive.select(primitive.greater(a, b), a, b)
)


@isa.instruction(src=mem, dst=mem)
def vrowsum(I):
    """Row-sum a 2x4 tile -> 2x1 (reduce over axis 1)."""

    @I.access
    def _(s, d):
        return (view(mem, s, (2, 4)), view(mem, d, (2, 1)))

    @I.compute
    def _(a, o):
        return primitive.reduce_sum(a, axis=1)


@isa.instruction(src=mem, dst=qmem)
def vcast(I):
    @I.access
    def _(s, d):
        return (contiguous(mem, s, 8), contiguous(qmem, d, 8))

    @I.compute
    def _(a, o):
        return primitive.cast(a, i32)


# --- Tier 3/4: contraction / conv family + reverse (NHWC tiles staged in mem) ---


@isa.instruction(src=mem, dst=mem)
def vrev(I):
    @I.access
    def _(s, d):
        return (contiguous(mem, s, 8), contiguous(mem, d, 8))

    @I.compute
    def _(a, o):
        return primitive.reverse(a, axis=0)


@isa.instruction(src=[mem, mem, mem], dst=mem)
def vconv(I):
    """conv2d: input 1x4x4x2 (NHWC), weight 3x2x2x2 (OHWI), bias 3 -> 1x3x3x3."""

    @I.access
    def _(i, w, b, o):
        return (
            view(mem, i, (1, 4, 4, 2)),
            view(mem, w, (3, 2, 2, 2)),
            contiguous(mem, b, 3),
            view(mem, o, (1, 3, 3, 3)),
        )

    @I.compute
    def _(i, w, b, o):
        return primitive.conv2d(
            i, w, b, stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1)
        )


@isa.instruction(src=[mem, mem, mem], dst=mem)
def vdwconv(I):
    """depthwise_conv2d: input 1x4x4x2, weight 2x2x2x1 (HWCM, M=1), bias 2 -> 1x3x3x2."""

    @I.access
    def _(i, w, b, o):
        return (
            view(mem, i, (1, 4, 4, 2)),
            view(mem, w, (2, 2, 2, 1)),
            contiguous(mem, b, 2),
            view(mem, o, (1, 3, 3, 2)),
        )

    @I.compute
    def _(i, w, b, o):
        return primitive.depthwise_conv2d(
            i, w, b, stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1)
        )


@isa.instruction(src=mem, dst=mem)
def vmaxpool(I):
    @I.access
    def _(s, d):
        return (view(mem, s, (1, 4, 4, 2)), view(mem, d, (1, 2, 2, 2)))

    @I.compute
    def _(a, o):
        return primitive.max_pool2d(a, kernel=(2, 2), stride=(2, 2), pad=(0, 0, 0, 0))


@isa.instruction(src=mem, dst=mem)
def vavgpool(I):
    @I.access
    def _(s, d):
        return (view(mem, s, (1, 4, 4, 2)), view(mem, d, (1, 2, 2, 2)))

    @I.compute
    def _(a, o):
        return primitive.avg_pool2d(a, kernel=(2, 2), stride=(2, 2), pad=(0, 0, 0, 0))


# ==========================================================================#
# Catalog + per-category oracle round-trips
# ==========================================================================#


def test_catalog_verifies():
    assert isa.catalog().operation.verify()


def test_unary_exp():
    x = np.linspace(-1.0, 1.0, 8).astype(np.float32)

    @isa.oracle(init={mem: x})
    def prog():
        vexp(s=0, d=8)
        isa.inspect(mem[8:16], label="r")

    np.testing.assert_allclose(prog()["r"], np.exp(x), rtol=1e-5, atol=1e-5)


def test_unary_zp_negate():
    x = np.linspace(-1.0, 1.0, 8).astype(np.float32)

    @isa.oracle(init={mem: x})
    def prog():
        vneg(s=0, d=8)
        isa.inspect(mem[8:16], label="r")

    np.testing.assert_allclose(prog()["r"], -x)


def test_binary_maximum():
    a = np.array([1, -2, 3, -4, 5, -6, 7, -8], np.float32)
    b = np.zeros(8, np.float32)

    @isa.oracle(init={mem: np.concatenate([a, b])})
    def prog():
        vmax(a=0, b=8, d=16)
        isa.inspect(mem[16:24], label="r")

    np.testing.assert_allclose(prog()["r"], np.maximum(a, b))


def test_compare_select():
    """select(greater(a, b), a, b) == elementwise max."""
    a = np.array([1, -2, 3, -4, 5, -6, 7, -8], np.float32)
    b = np.zeros(8, np.float32)

    @isa.oracle(init={mem: np.concatenate([a, b])})
    def prog():
        vselmax(a=0, b=8, d=16)
        isa.inspect(mem[16:24], label="r")

    np.testing.assert_allclose(prog()["r"], np.maximum(a, b))


def test_reduce_sum():
    x = np.arange(8, dtype=np.float32)

    @isa.oracle(init={mem: x})
    def prog():
        vrowsum(s=0, d=16)
        isa.inspect(mem[16:18], label="r")

    np.testing.assert_allclose(prog()["r"], x.reshape(2, 4).sum(axis=1))


def test_cast_f32_to_i32():
    x = (np.arange(8) - 4).astype(np.float32)  # integer-valued: no rounding ambiguity

    @isa.oracle(init={mem: x})
    def prog():
        vcast(s=0, d=0)
        isa.inspect(qmem[0:8], label="r")

    np.testing.assert_array_equal(prog()["r"], x.astype(np.int32))


# ==========================================================================#
# Instruction selection: the registry-derived recognizer matches TOSA source
# programs (from torch_mlir) onto the new prims.
# ==========================================================================#


def _torch_tosa(model, *inputs) -> str:
    import torch

    fx = pytest.importorskip("torch_mlir.fx")
    tensors = [torch.from_numpy(np.asarray(x, np.float32)) for x in inputs]
    module = fx.export_and_import(
        model.eval(), *tensors, output_type=fx.OutputType.TOSA
    )
    return str(module)


def _match_isa() -> ISA:
    """An ISA whose single global buffer is also the compute buffer, so a matched
    program needs no data movement."""
    m = ISA("PrimMatch")
    g = m.global_("mem", shape=(4096,), dtype=f32)

    @m.instruction(src=g, dst=g)
    def vexp(I):
        @I.access
        def _(s, d):
            return (contiguous(g, s, 8), contiguous(g, d, 8))

        @I.compute
        def _(a, o):
            return primitive.exp(a)

    @m.instruction(src=[g, g], dst=g)
    def vmax(I):
        @I.access
        def _(a, b, d):
            return (contiguous(g, a, 8), contiguous(g, b, 8), contiguous(g, d, 8))

        @I.compute
        def _(a, b, o):
            return primitive.maximum(a, b)

    @m.instruction(src=g, dst=g)
    def rowsum(I):
        @I.access
        def _(s, d):
            return (
                view(g, s, (2, 4)),
                view(g, d, (2, 1)),
            )

        @I.compute
        def _(a, o):
            return primitive.reduce_sum(a, axis=1)

    return m


def test_match_exp():
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def forward(self, a):
            return torch.exp(a)

    x = np.linspace(-1.0, 1.0, 8).astype(np.float32)
    prog = _match_isa().compile_program(_torch_tosa(M(), x))
    assert [e.name for e in prog.emits] == ["vexp"]
    np.testing.assert_allclose(prog(x), np.exp(x), rtol=1e-5, atol=1e-5)


def test_match_maximum():
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def forward(self, a, b):
            return torch.maximum(a, b)

    a = np.linspace(-1.0, 1.0, 8).astype(np.float32)
    b = np.linspace(1.0, -1.0, 8).astype(np.float32)
    prog = _match_isa().compile_program(_torch_tosa(M(), a, b))
    assert [e.name for e in prog.emits] == ["vmax"]
    np.testing.assert_allclose(prog(a, b), np.maximum(a, b))


def test_match_reduce_sum_axis():
    """The reduced axis is part of the semantics: a sum over dim=1 matches the
    axis-1 reduce instruction."""
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def forward(self, a):
            return torch.sum(a, dim=1, keepdim=True)

    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    prog = _match_isa().compile_program(_torch_tosa(M(), x))
    assert [e.name for e in prog.emits] == ["rowsum"]
    np.testing.assert_allclose(prog(x).reshape(2, 1), x.sum(axis=1, keepdims=True))


# ==========================================================================#
# Tier 3/4 oracle round-trips (references from torch, NHWC layout)
# ==========================================================================#


def test_reverse():
    x = np.arange(8, dtype=np.float32)

    @isa.oracle(init={mem: x})
    def prog():
        vrev(s=0, d=8)
        isa.inspect(mem[8:16], label="r")

    np.testing.assert_allclose(prog()["r"], x[::-1])


def test_conv2d():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 4, 4, 2)).astype(np.float32)  # NHWC
    w = rng.standard_normal((3, 2, 2, 2)).astype(np.float32)  # OHWI
    b = rng.standard_normal((3,)).astype(np.float32)
    init = np.zeros(256, np.float32)
    init[0:32], init[32:56], init[56:59] = inp.reshape(-1), w.reshape(-1), b

    @isa.oracle(init={mem: init})
    def prog():
        vconv(i=0, w=32, b=56, o=64)
        isa.inspect(mem[64:91], label="z")

    ref = (
        torch.conv2d(
            torch.tensor(inp).permute(0, 3, 1, 2),
            torch.tensor(w).permute(0, 3, 1, 2),
            torch.tensor(b),
        )
        .permute(0, 2, 3, 1)
        .numpy()
    )
    np.testing.assert_allclose(
        prog()["z"].reshape(1, 3, 3, 3), ref, rtol=1e-4, atol=1e-4
    )


def test_depthwise_conv2d():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    inp = rng.standard_normal((1, 4, 4, 2)).astype(np.float32)  # NHWC
    w = rng.standard_normal((2, 2, 2, 1)).astype(np.float32)  # HWCM, M=1
    b = rng.standard_normal((2,)).astype(np.float32)
    init = np.zeros(256, np.float32)
    init[0:32], init[32:40], init[40:42] = inp.reshape(-1), w.reshape(-1), b

    @isa.oracle(init={mem: init})
    def prog():
        vdwconv(i=0, w=32, b=40, o=48)
        isa.inspect(mem[48:66], label="z")

    w_torch = np.transpose(w[:, :, :, 0], (2, 0, 1))[:, None]  # HWC -> [C,1,KH,KW]
    ref = (
        torch.conv2d(
            torch.tensor(inp).permute(0, 3, 1, 2),
            torch.tensor(w_torch),
            torch.tensor(b),
            groups=2,
        )
        .permute(0, 2, 3, 1)
        .numpy()
    )
    np.testing.assert_allclose(
        prog()["z"].reshape(1, 3, 3, 2), ref, rtol=1e-4, atol=1e-4
    )


def test_max_pool2d():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    inp = rng.standard_normal((1, 4, 4, 2)).astype(np.float32)

    @isa.oracle(init={mem: inp.reshape(-1)})
    def prog():
        vmaxpool(s=0, d=64)
        isa.inspect(mem[64:72], label="p")

    ref = (
        torch.max_pool2d(torch.tensor(inp).permute(0, 3, 1, 2), 2)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    np.testing.assert_allclose(
        prog()["p"].reshape(1, 2, 2, 2), ref, rtol=1e-5, atol=1e-5
    )


def test_avg_pool2d():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    inp = rng.standard_normal((1, 4, 4, 2)).astype(np.float32)

    @isa.oracle(init={mem: inp.reshape(-1)})
    def prog():
        vavgpool(s=0, d=64)
        isa.inspect(mem[64:72], label="p")

    ref = (
        torch.nn.functional.avg_pool2d(torch.tensor(inp).permute(0, 3, 1, 2), 2)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    np.testing.assert_allclose(
        prog()["p"].reshape(1, 2, 2, 2), ref, rtol=1e-5, atol=1e-5
    )


# ==========================================================================#
# Tier 3/4 matching: reverse from torch.flip; conv2d from a hand-built NHWC
# source (torch wraps conv in NCHW<->NHWC transposes + constant weights, which
# the planner does not yet route, so the conv source is built directly).
# ==========================================================================#


def test_match_reverse():
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def forward(self, a):
            return torch.flip(a, [0])

    isa2 = ISA("RevMatch")
    g = isa2.global_("mem", shape=(64,), dtype=f32)

    @isa2.instruction(src=g, dst=g)
    def vrev(I):
        @I.access
        def _(s, d):
            return (contiguous(g, s, 8), contiguous(g, d, 8))

        @I.compute
        def _(a, o):
            return primitive.reverse(a, axis=0)

    x = np.arange(8, dtype=np.float32)
    prog = isa2.compile_program(_torch_tosa(M(), x))
    assert [e.name for e in prog.emits] == ["vrev"]
    np.testing.assert_allclose(prog(x), x[::-1])


def _nhwc_conv_source() -> str:
    """A TOSA ``func @main`` with a single NHWC ``tosa.conv2d`` whose input,
    weight, and bias are function arguments (no constants, no layout wrapping)."""
    from allo._mlir import ir
    from allo._mlir.ir import InsertionPoint, Location, Module
    from allo._mlir.dialects import tosa, func as func_d, allo as allo_d

    ctx = ir.Context()
    allo_d.register_dialect(ctx)
    with ctx, Location.unknown(ctx):
        m = Module.create()
        f32t = ir.F32Type.get()

        def T(s):
            return ir.RankedTensorType.get(s, f32t)

        ft = ir.FunctionType.get(
            [T([1, 4, 4, 2]), T([3, 2, 2, 2]), T([3])], [T([1, 3, 3, 3])]
        )
        with InsertionPoint(m.body):
            fn = func_d.FuncOp("main", ft)
            blk = fn.add_entry_block()
        with InsertionPoint(blk):
            inp, w, bias = blk.arguments

            def zp():
                return tosa.ConstOp(
                    ir.DenseElementsAttr.get_splat(T([1]), ir.FloatAttr.get(f32t, 0.0))
                ).result

            arr = ir.DenseI64ArrayAttr.get
            out = tosa.Conv2DOp(
                T([1, 3, 3, 3]),
                inp,
                w,
                bias,
                zp(),
                zp(),
                pad=arr([0, 0, 0, 0]),
                stride=arr([1, 1]),
                dilation=arr([1, 1]),
                acc_type=ir.TypeAttr.get(f32t),
            ).result
            func_d.ReturnOp([out])
        return str(m)


def test_match_conv2d():
    """A bare NHWC conv2d (attrs pad/stride/dilation as semantics) selects the
    matching conv instruction; the zero-point constants are trimmed as non-data."""
    torch = pytest.importorskip("torch")

    isa2 = ISA("ConvMatch")
    g = isa2.global_("mem", shape=(4096,), dtype=f32)

    @isa2.instruction(src=[g, g, g], dst=g)
    def conv(I):
        @I.access
        def _(i, w, b, o):
            return (
                view(g, i, (1, 4, 4, 2)),
                view(g, w, (3, 2, 2, 2)),
                contiguous(g, b, 3),
                view(g, o, (1, 3, 3, 3)),
            )

        @I.compute
        def _(i, w, b, o):
            return primitive.conv2d(
                i, w, b, stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1)
            )

    prog = isa2.compile_program(_nhwc_conv_source())
    assert [e.name for e in prog.emits] == ["conv"]

    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 4, 4, 2)).astype(np.float32)
    w = rng.standard_normal((3, 2, 2, 2)).astype(np.float32)
    b = rng.standard_normal((3,)).astype(np.float32)
    ref = (
        torch.conv2d(
            torch.tensor(inp).permute(0, 3, 1, 2),
            torch.tensor(w).permute(0, 3, 1, 2),
            torch.tensor(b),
        )
        .permute(0, 2, 3, 1)
        .numpy()
    )
    np.testing.assert_allclose(
        prog(inp, w, b).reshape(1, 3, 3, 3), ref, rtol=1e-4, atol=1e-4
    )
