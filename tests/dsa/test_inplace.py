# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which operands a result may be written over.

The allocator coalesces an instruction's result onto a dying operand's slot when
that saves a slot. It is only sound when output position *i* depends on that
operand's position *i* — otherwise the instruction overwrites input it has not read
yet. The predicate excluded ``matmul`` and nothing else, so ``transpose``,
``reverse``, ``reduce`` and the conv/pool family all looked position-preserving.

**The functional oracle cannot catch this.** It evaluates an instruction's semantics
as a value expression — every operand is read before any destination is written — so
the overwrite the hardware would perform simply never happens in simulation. What is
observable is the *allocation*: the tests below compare the offsets the planner hands
out for two programs that differ only in whether the consuming op permutes.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import view
from allo.exp.dsa.core import ISA
from allo.exp.dsa.search import _reusable_operands
from allo.lang.core import f32, i32


# ==========================================================================#
# The predicate itself, one instruction shape per prim family
# ==========================================================================#


def _isa() -> ISA:
    isa = ISA("inplace")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)
    qmem = isa.scalar("qmem", slots=64, dtype=i32)

    def instr(name, src_shapes, dst_shape, fn, dst=None):
        dst_buf = dst or mem

        def patterns(*p):
            srcs = [view(mem, p[i], s) for i, s in enumerate(src_shapes)]
            return (*srcs, view(dst_buf, p[-1], dst_shape))

        # An access fn's arity is read off its signature, so it needs named params.
        access = {
            1: lambda a, d: patterns(a, d),
            2: lambda a, b, d: patterns(a, b, d),
            3: lambda a, b, c, d: patterns(a, b, c, d),
        }[len(src_shapes)]

        @isa.instruction(src=[mem] * len(src_shapes), dst=dst_buf, name=name)
        def _(I):
            I.access(access)

            @I.compute
            def _(*a):
                return fn(*a[: len(src_shapes)])

    instr("vadd", [(8,), (8,)], (8,), primitive.add)
    instr(
        "vrelu_add",
        [(8,), (8,)],
        (8,),
        lambda a, b: primitive.relu(primitive.add(a, b)),
    )
    instr("vcast", [(8,)], (8,), lambda a: primitive.cast(a, i32), dst=qmem)
    instr("mm", [(1, 4, 4), (1, 4, 4)], (1, 4, 4), primitive.matmul)
    instr(
        "mm_acc",
        [(1, 4, 4), (1, 4, 4), (1, 4, 4)],
        (1, 4, 4),
        lambda a, b, c: primitive.add(c, primitive.matmul(a, b)),
    )
    instr(
        "add_t",
        [(4, 4), (4, 4)],
        (4, 4),
        lambda y, x: primitive.add(y, primitive.transpose(x, [1, 0])),
    )
    instr("vrev", [(4, 4)], (4, 4), lambda a: primitive.reverse(a, axis=0))
    instr("vrowsum", [(4, 4)], (4, 1), lambda a: primitive.reduce_sum(a, axis=1))
    instr(
        "vmaxpool",
        [(1, 4, 4, 2)],
        (1, 2, 2, 2),
        lambda a: primitive.max_pool2d(
            a, kernel=(2, 2), stride=(2, 2), pad=(0, 0, 0, 0)
        ),
    )
    return isa


ISA_UNDER_TEST = _isa()

REUSE = [
    # elementwise: every operand's position i feeds output position i
    ("vadd", {0, 1}),
    ("vrelu_add", {0, 1}),
    ("vcast", {0}),
    # a contraction mixes positions...
    ("mm", set()),
    # ...but an accumulator read only by the elementwise add still lines up. This
    # is what lets a K-reduction chain collapse onto one slot.
    ("mm_acc", {2}),
    # everything below was wrongly reusable: same element count, different order
    ("add_t", {0}),  # y lines up; x reaches the result through a transpose
    ("vrev", set()),
    ("vrowsum", set()),
    ("vmaxpool", set()),
]


@pytest.mark.parametrize("name,expected", REUSE, ids=[n for n, _ in REUSE])
def test_reusable_operands(name, expected):
    assert _reusable_operands(ISA_UNDER_TEST._ops[name]) == expected


# ==========================================================================#
# What that means for allocation
# ==========================================================================#


def _consumer_isa() -> ISA:
    """``vadd`` plus two 1:1 consumers of its result — one position-preserving,
    one not — so a program's only difference is which one consumes."""
    isa = ISA("inplace-alloc")
    mem = isa.global_("mem", shape=(256,), dtype=f32)

    @isa.instruction(src=[mem, mem], dst=mem)
    def vadd(I):
        @I.access
        def _(a, b, d):
            return (view(mem, a, (2, 2)), view(mem, b, (2, 2)), view(mem, d, (2, 2)))

        @I.compute
        def _(a, b, o):
            return primitive.add(a, b)

    def unary(name, fn):
        @isa.instruction(src=mem, dst=mem, name=name)
        def _(I):
            @I.access
            def _(s, d):
                return (view(mem, s, (2, 2)), view(mem, d, (2, 2)))

            @I.compute
            def _(a, o):
                return fn(a)

    unary("vrelu", primitive.relu)
    unary("vrev", lambda a: primitive.reverse(a, axis=0))
    return isa


def _source(consumer: str) -> str:
    """``consumer(a + b)`` — the sum is single-use, so it dies exactly where the
    consumer reads it: the one step where coalescing is offered."""
    t = "tensor<2x2xf32>"
    op = (
        "tosa.clamp %s {min_val = 0.000000e+00 : f32, max_val = 3.40282347E+38 : f32}"
        if consumer == "relu"
        else "tosa.reverse %s {axis = 0 : i32}"
    )
    return f"""
func.func @main(%a: {t}, %b: {t}) -> {t} {{
  %s = tosa.add %a, %b : ({t}, {t}) -> {t}
  %r = {op} : ({t}) -> {t}
  return %r : {t}
}}
"""


def _offsets(consumer: str) -> tuple:
    prog = _consumer_isa().compile_program(_source(consumer))
    add, use = prog.emits[0], prog.emits[1]
    return add.addr[-1], use.addr[-1]  # the sum's slot, then the result's slot


def test_a_position_preserving_consumer_reuses_the_dying_operand():
    """The control: relu's output position i reads only position i, so writing over
    the sum is sound — and saves a slot."""
    produced, written = _offsets("relu")
    assert produced == written


def test_a_permuting_consumer_does_not_reuse_the_dying_operand():
    """``reverse`` reads position n-1-i for output position i. Writing over its own
    input would clobber values it has not read yet, so it gets its own slot."""
    produced, written = _offsets("reverse")
    assert produced != written


def test_the_permuting_consumer_still_computes_the_right_values():
    """Not a proof of the fix (the simulator reads all operands before writing, so
    it would pass either way) — it just pins that refusing to coalesce did not break
    the program."""
    prog = _consumer_isa().compile_program(_source("reverse"))
    a = np.arange(4, dtype=np.float32).reshape(2, 2)
    b = np.ones((2, 2), np.float32)
    np.testing.assert_allclose(prog(a, b), (a + b)[::-1])
