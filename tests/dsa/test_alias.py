# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An address param shared by two buffers is a constraint, not a coincidence.

When an instruction's access region uses one param as the basis of both an operand
and the destination, the ISA is saying they are *at the same address* — the
instruction overwrites what it reads. QKV's ``softmax`` is exactly that:

    @I.access
    def _(addr, n):
        return (contiguous(d2, addr, n), contiguous(d2, addr, n))

The planner recorded only the *last* buffer each param was seen under, so the
emitted address came from the destination. When in-place coalescing happened to put
the destination on the operand's slot, that was the same number and everything
worked; when the operand outlived the instruction, no coalescing happened and the
instruction read and wrote a slot that held neither — silently.

So the constraint is now carried through allocation: the write is *forced* onto the
operand's slot, and a program that cannot satisfy that is refused with a message
saying why. QKV keeps compiling — and now because it is guaranteed to, not because
the allocator happened to agree.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, strided, view
from allo.exp.dsa.core import ISA, param_roles
from allo.exp.dsa.errors import AcceleratorDescriptionError, AllocationError
from allo.exp.dsa.search import _alias_groups
from allo.lang.core import f32

N = 4


def _isa(name="alias"):
    """``sq`` overwrites its operand (one ``x`` for both buffers); ``vmul`` and
    ``vadd`` are ordinary out-of-place instructions."""
    isa = ISA(name)
    mem = isa.global_("mem", shape=(64,), dtype=f32)

    @isa.instruction(src=mem, dst=mem)
    def sq(I):
        @I.access
        def _(x):
            return (contiguous(mem, x, N), contiguous(mem, x, N))

        @I.compute
        def _(a, o):
            return primitive.mul(a, a)

    @isa.instruction(src=[mem, mem], dst=mem)
    def vadd(I):
        @I.access
        def _(a, b, d):
            return (
                contiguous(mem, a, N),
                contiguous(mem, b, N),
                contiguous(mem, d, N),
            )

        @I.compute
        def _(a, b, o):
            return primitive.add(a, b)

    return isa


_T = f"tensor<{N}xf32>"
_SHIFT = '%z = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>'


def _square_only() -> str:
    """``x*x``: the square is the result, so nothing reads ``x`` afterwards."""
    return f"""
func.func @main(%x: {_T}) -> {_T} {{
  {_SHIFT}
  %s = tosa.mul %x, %x, %z : ({_T}, {_T}, tensor<1xi8>) -> {_T}
  return %s : {_T}
}}
"""


def _square_and_reuse() -> str:
    """``x*x + x``: ``x`` is read again *after* the square, so the in-place write
    would destroy it."""
    return f"""
func.func @main(%x: {_T}) -> {_T} {{
  {_SHIFT}
  %s = tosa.mul %x, %x, %z : ({_T}, {_T}, tensor<1xi8>) -> {_T}
  %r = tosa.add %s, %x : ({_T}, {_T}) -> {_T}
  return %r : {_T}
}}
"""


# ==========================================================================#
# The constraint is recorded at all
# ==========================================================================#


def test_a_shared_address_param_records_every_buffer_it_names():
    """``param_roles`` used to keep one ``(buffer, axis)`` per param, so the second
    reference erased the first."""
    _, offset_of = param_roles(_isa()._ops["sq"].spec)
    assert offset_of[0] == [(0, 0), (1, 0)]
    assert _alias_groups(offset_of) == [(0, [0, 1])]


def test_an_ordinary_instruction_has_no_alias_constraint():
    _, offset_of = param_roles(_isa()._ops["vadd"].spec)
    assert _alias_groups(offset_of) == []


# ==========================================================================#
# What the planner does with it
# ==========================================================================#


def test_an_in_place_instruction_is_emitted_at_its_operand():
    """The address the instruction is given must be where its operand actually is."""
    isa = _isa()
    prog = isa.compile_program(_square_only())
    (sq,) = prog.emits
    (operand_offset, _shape) = prog.inputs[0]
    assert sq.addr == [operand_offset[0]]

    x = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    np.testing.assert_allclose(prog(x), x * x)


def test_an_operand_that_outlives_an_in_place_write_is_refused():
    """This program used to compile and return ``x`` instead of ``x*x + x``: the
    square ran on an unrelated slot, so the add read a zeroed one. There is no
    correct placement, so the only sound answer is to refuse."""
    isa = _isa()
    with pytest.raises(AllocationError, match="sq: writes its result over the operand"):
        isa.compile_program(_square_and_reuse())


def test_the_refusal_names_the_step_that_still_needs_the_operand():
    isa = _isa()
    with pytest.raises(AllocationError, match=r"read again at step 1"):
        isa.compile_program(_square_and_reuse())


def test_two_aliased_operands_bound_to_different_values_are_refused():
    """One param as the basis of two *source* buffers says they are the same
    address, so binding them to different values is unsatisfiable."""
    isa = ISA("two-src")
    mem = isa.global_("mem", shape=(64,), dtype=f32)

    @isa.instruction(src=[mem, mem], dst=mem)
    def same_src_add(I):
        @I.access
        def _(x, d):
            return (
                contiguous(mem, x, N),
                contiguous(mem, x, N),
                contiguous(mem, d, N),
            )

        @I.compute
        def _(a, b, o):
            return primitive.add(a, b)

    src = f"""
func.func @main(%a: {_T}, %b: {_T}) -> {_T} {{
  %r = tosa.add %a, %b : ({_T}, {_T}) -> {_T}
  return %r : {_T}
}}
"""
    with pytest.raises(AllocationError, match="at one address, but they are bound"):
        isa.compile_program(src)


def test_a_data_movement_instruction_may_not_share_an_address_param():
    """Moves are inserted between two independently placed locations, so there is
    nothing that could honour "same address" — say so instead of emitting the read's
    placement for both."""
    isa = _isa("bad-move")
    mem = isa.buffers["mem"]
    reg = isa.scalar("reg", slots=16, dtype=f32)

    @isa.instruction(src=mem, dst=reg)
    def ld(I):
        @I.access
        def _(x):
            return (contiguous(mem, x, N), contiguous(reg, x, N))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    src = f"""
func.func @main(%a: {_T}) -> {_T} {{
  %r = tosa.add %a, %a : ({_T}, {_T}) -> {_T}
  return %r : {_T}
}}
"""
    with pytest.raises(AcceleratorDescriptionError, match="cannot share an address"):
        isa.compile_program(src)


def test_an_address_param_may_not_stand_for_two_different_axes():
    """A param names one coordinate component. Spanning axes would make the emitted
    address depend on which reference the planner happened to pick."""
    isa = ISA("diagonal")
    grid = isa.hbm("grid", shape=(16, 16), dtype=f32, is_global=True)

    @isa.instruction(src=grid, dst=grid)
    def diag(I):
        @I.access
        def _(x, d0, d1):
            return (
                strided(grid, [x, x], [1, N], [1, 1]),
                strided(grid, [d0, d1], [1, N], [1, 1]),
            )

        @I.compute
        def _(a, o):
            return primitive.relu(a)

    with pytest.raises(AcceleratorDescriptionError, match="cannot stand for"):
        param_roles(diag.spec)


# ==========================================================================#
# The instruction this was found in
# ==========================================================================#


def test_qkv_softmax_is_an_in_place_instruction():
    """Not a synthetic case: the real ISA that was relying on luck. Its whole
    ``@I.access`` is ``(contiguous(d2, addr, n), contiguous(d2, addr, n))``."""
    pytest.importorskip("ml_dtypes")
    from examples.accelerator.qkv.isa import qkv

    _, offset_of = param_roles(qkv._ops["softmax"].spec)
    assert _alias_groups(offset_of) == [(0, [0, 1])]


def test_an_in_place_softmax_still_compiles_and_runs():
    """A local stand-in for the QKV path, small enough to run: the operand dies at
    the in-place step, so the forced placement is satisfiable."""
    isa = ISA("softmaxish")
    mem = isa.global_("mem", shape=(64,), dtype=f32)

    @isa.instruction(src=mem, dst=mem)
    def normalize(I):
        @I.access
        def _(addr):
            return (view(mem, addr, (2, 4)), view(mem, addr, (2, 4)))

        @I.compute
        def _(x, o):
            s = primitive.reduce_sum(x, axis=1)
            return primitive.mul(x, primitive.reciprocal(s))

    t = "tensor<2x4xf32>"
    src = f"""
func.func @main(%x: {t}) -> {t} {{
  %z = "tosa.const"() {{values = dense<0> : tensor<1xi8>}} : () -> tensor<1xi8>
  %s = tosa.reduce_sum %x {{axis = 1 : i32}} : ({t}) -> tensor<2x1xf32>
  %i = tosa.reciprocal %s : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %r = tosa.mul %x, %i, %z : ({t}, tensor<2x1xf32>, tensor<1xi8>) -> {t}
  return %r : {t}
}}
"""
    prog = isa.compile_program(src)
    (op,) = prog.emits
    assert op.addr == [prog.inputs[0][0][0]]

    x = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
    np.testing.assert_allclose(prog(x), x / x.sum(axis=1, keepdims=True), rtol=1e-6)
