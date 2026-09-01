# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants, at the two places they occur.

**In an instruction** (``primitive.const``). Not every operand of a real instruction
comes from a buffer: MiniNPU's ``vexp`` computes ``2**x`` from one register, so the
``2`` belongs to the instruction. Before this the compute DAG's only leaf was a
buffer arg, so such an instruction could be described only by inventing an operand
nobody supplies or by calling ``2**x`` an ``exp`` — and it was left undescribed
instead. The literal is *load-bearing for selection*: ``pow(2, x)`` and ``pow(3, x)``
are different functions, so the matcher compares it like a transpose's permutation.

**In a program** (the constant pool). A ``tosa.const`` used as a data operand — a
bias — is program data with nowhere to live: allocation only ever placed inputs and
instruction results, so such a value had no address and compilation stopped. It is
now placed in the I/O buffer alongside the inputs, which is ACT Def 3.8's
``concat(bflat(X), bflat(const))``, and ``CompiledProgram.__call__`` writes it in
before the run.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, view
from allo.exp.dsa.core import ISA
from allo.exp.dsa.errors import CompileError, NoMatchError
from allo.lang.core import f32

N = 4
_V = f"tensor<{N}xf32>"


# ==========================================================================#
# A constant inside an instruction
# ==========================================================================#


def _pow_isa(base=2.0):
    """``vexp``: a one-operand ``base ** x``, the base baked into the instruction."""
    isa = ISA(f"pow{base}")
    mem = isa.global_("mem", shape=(64,), dtype=f32)

    @isa.instruction(src=mem, dst=mem)
    def vexp(I):
        @I.access
        def _(s, d):
            return (contiguous(mem, s, N), contiguous(mem, d, N))

        @I.compute
        def _(a, o):
            return primitive.pow(primitive.const(base), a)

    return isa


def _pow_src(base="2.000000e+00") -> str:
    return f"""
func.func @main(%x: {_V}) -> {_V} {{
  %b = "tosa.const"() {{values = dense<{base}> : tensor<1xf32>}} : () -> tensor<1xf32>
  %r = tosa.pow %b, %x : (tensor<1xf32>, {_V}) -> {_V}
  return %r : {_V}
}}
"""


def test_an_instruction_may_carry_a_constant_leaf():
    isa = _pow_isa()
    assert isa.catalog().operation.verify()
    prog = isa.compile_program(_pow_src())
    assert [e.name for e in prog.emits] == ["vexp"]

    x = np.linspace(-2.0, 2.0, N).astype(np.float32)
    np.testing.assert_allclose(prog(x), 2.0**x, rtol=1e-6)


def test_the_constant_selects():
    """The literal is part of the semantics: a base-3 source must not pick up the
    base-2 instruction. This is the whole reason the matcher compares values rather
    than treating a constant as a wildcard."""
    with pytest.raises(NoMatchError):
        _pow_isa().compile_program(_pow_src("3.000000e+00"))


def test_a_non_constant_operand_does_not_match_a_constant_leaf():
    """The base arriving as a program input is a different instruction's job."""
    src = f"""
func.func @main(%b: tensor<1xf32>, %x: {_V}) -> {_V} {{
  %r = tosa.pow %b, %x : (tensor<1xf32>, {_V}) -> {_V}
  return %r : {_V}
}}
"""
    with pytest.raises(NoMatchError):
        _pow_isa().compile_program(src)


def test_a_constant_leaf_round_trips_through_the_oracle():
    """The catalog's ``tosa.const`` has to survive lowering and JIT, not just match."""
    isa = _pow_isa()
    mem = isa.buffers["mem"]
    x = np.linspace(-2.0, 2.0, N).astype(np.float32)

    @isa.oracle(init={mem: x})
    def prog():
        isa._ops["vexp"](s=0, d=N)
        isa.inspect(mem[N : 2 * N], label="r")

    np.testing.assert_allclose(prog()["r"], 2.0**x, rtol=1e-6)


def test_a_constant_is_compared_at_the_pattern_dtype():
    """The ISA is written with a Python double and the source holds an f32, so the
    comparison rounds through the dtype — otherwise no float constant would ever
    match."""
    third = 1.0 / 3.0
    isa = _pow_isa(third)
    isa.compile_program(_pow_src(repr(float(np.float32(third)))))


def test_mininpu_models_its_base_2_exponential():
    """The instruction this feature exists for. ``vexp`` was in the "not modeled"
    list precisely because the ``2`` had nowhere to go."""
    from examples.accelerator.mininpu.isa import npu, VEC_LANES

    src = f"""
func.func @main(%x: tensor<{VEC_LANES}xf32>) -> tensor<{VEC_LANES}xf32> {{
  %b = "tosa.const"() {{values = dense<2.000000e+00> : tensor<1xf32>}} : () -> tensor<1xf32>
  %r = tosa.pow %b, %x : (tensor<1xf32>, tensor<{VEC_LANES}xf32>)
       -> tensor<{VEC_LANES}xf32>
  return %r : tensor<{VEC_LANES}xf32>
}}
"""
    prog = npu.compile_program(src)
    assert "vexp" in [e.name for e in prog.emits]
    x = np.linspace(-3.0, 3.0, VEC_LANES).astype(np.float32)
    np.testing.assert_allclose(prog(x), 2.0**x, rtol=1e-6)


# ==========================================================================#
# A constant in the program: the constant pool
# ==========================================================================#


def _bias_isa():
    isa = ISA("bias")
    mem = isa.global_("mem", shape=(256,), dtype=f32)

    @isa.instruction(src=[mem, mem], dst=mem)
    def mm(I):
        @I.access
        def _(a, b, d):
            return (
                view(mem, a, (1, N, N)),
                view(mem, b, (1, N, N)),
                view(mem, d, (1, N, N)),
            )

        @I.compute
        def _(a, b, o):
            return primitive.matmul(a, b)

    @isa.instruction(src=[mem, mem], dst=mem)
    def vadd(I):
        @I.access
        def _(a, b, d):
            return (
                view(mem, a, (1, N, N)),
                view(mem, b, (1, N, 1)),
                view(mem, d, (1, N, N)),
            )

        @I.compute
        def _(a, b, o):
            return primitive.add(a, b)

    return isa


_T = f"tensor<1x{N}x{N}xf32>"
_BIAS = "[[[1.0],[2.0],[3.0],[4.0]]]"


def _bias_src(bias=_BIAS) -> str:
    return f"""
func.func @main(%a: {_T}, %b: {_T}) -> {_T} {{
  %zp = "tosa.const"() {{values = dense<0.0> : tensor<1xf32>}} : () -> tensor<1xf32>
  %c = "tosa.const"() {{values = dense<{bias}> : tensor<1x{N}x1xf32>}}
       : () -> tensor<1x{N}x1xf32>
  %m = tosa.matmul %a, %b, %zp, %zp : ({_T}, {_T}, tensor<1xf32>, tensor<1xf32>) -> {_T}
  %r = tosa.add %m, %c : ({_T}, tensor<1x{N}x1xf32>) -> {_T}
  return %r : {_T}
}}
"""


def test_a_bias_constant_is_placed_and_loaded():
    """The plan's acceptance case: ``matmul`` then ``add`` of a literal bias."""
    prog = _bias_isa().compile_program(_bias_src())
    assert len(prog.constants) == 1
    (offset, data) = prog.constants[0]
    np.testing.assert_array_equal(data.reshape(-1), [1.0, 2.0, 3.0, 4.0])

    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, N, N)).astype(np.float32)
    b = rng.standard_normal((1, N, N)).astype(np.float32)
    got = np.asarray(prog(a, b)).reshape(1, N, N)
    np.testing.assert_allclose(got, a @ b + data, rtol=1e-5, atol=1e-6)


def test_a_constant_does_not_overlap_the_inputs():
    """It is resident from the start, like an input — so the allocator has to reserve
    its space rather than hand it out to a result."""
    prog = _bias_isa().compile_program(_bias_src())
    (const_off, data) = prog.constants[0]
    taken = set()
    for off, shape in prog.inputs:
        taken |= set(range(off[0], off[0] + int(np.prod(shape))))
    assert not taken & set(range(const_off[0], const_off[0] + data.size))


def test_a_splat_constant_expands_to_its_extent():
    """A splat prints as a single element whatever its declared shape."""
    prog = _bias_isa().compile_program(_bias_src("5.0"))
    (_offset, data) = prog.constants[0]
    np.testing.assert_array_equal(data.reshape(-1), [5.0] * N)


def test_the_dump_shows_the_constant_pool():
    text = str(_bias_isa().compile_program(_bias_src()))
    assert "constants:" in text
    assert f"shape=(1, {N}, 1)" in text


def test_a_resource_backed_constant_is_refused_with_its_reason():
    """torch stores model weights as ``dense_resource`` blobs, which the MLIR Python
    bindings expose no reader for. Unknown data must not become a default value, so
    the constant is refused — and the message says what to do instead."""
    src = f"""
module {{
func.func @main(%a: {_T}, %b: {_T}) -> {_T} {{
  %zp = "tosa.const"() {{values = dense<0.0> : tensor<1xf32>}} : () -> tensor<1xf32>
  %c = "tosa.const"() {{values = dense_resource<bias> : tensor<1x{N}x1xf32>}}
       : () -> tensor<1x{N}x1xf32>
  %m = tosa.matmul %a, %b, %zp, %zp : ({_T}, {_T}, tensor<1xf32>, tensor<1xf32>) -> {_T}
  %r = tosa.add %m, %c : ({_T}, tensor<1x{N}x1xf32>) -> {_T}
  return %r : {_T}
}}
}}
{{-#
  dialect_resources: {{ builtin: {{ bias: "0x04000000000000000000803F0000004000004040" }} }}
#-}}
"""
    with pytest.raises(CompileError, match="dialect_resource"):
        _bias_isa().compile_program(src)
