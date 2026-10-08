# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track E finding E-F1: an untyped integer literal was always ``int32``.

``x | (1 << 35)`` on a ``UInt(64)`` left bit 35 clear, with no warning, on
``build("llvm")`` and on the dataflow simulator
(``tests/limits/new_wide_literal_shift.py``). The front end's rules now
(``allo/ir/infer.py``, the ``E-F1`` block):

1. a literal expression whose value is not a 32-bit pattern is folded to one
   constant of its own minimal width;
2. next to a typed operand (or as the value of a target) wider than 32 bits,
   a literal in ``[2**31, 2**32)`` and the literal left of a shift by a
   variable amount take that type;
3. a folded literal stored into an integer it does not fit is refused, naming
   the expression.

A literal whose value is a 32-bit pattern next to an operand of at most 32
bits is typed as before; ``test_narrow_unchanged`` checks values, and
``test_narrow_ir_unchanged`` checks that the IR text is identical to the IR
of the same kernel with the literal written as a typed ``int32`` local.
"""

import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import Int, UInt, int8, int32, uint8, uint32

U64 = UInt(64)
BIG = 1 << 40  # a global int that is not a 32-bit pattern
A = np.array([0, 5], dtype=np.uint64)


def _run(fn, a=A):
    o = np.zeros(2, dtype=np.uint64)
    allo.customize(fn).build(target="llvm")(a.copy(), o)
    return o.tolist()


# ---- wide operands: shift, or, and, add, compare; UInt(64) and Int(64) ----


def or_shift(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x | (1 << 35)


def and_shift(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i] + (3 << 34)
        o[i] = x & (1 << 35)


def add_shift(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x + (1 << 40)


def add_lone(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x + 1099511627776


def add_mask(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x + 0x80000000  # a 32-bit pattern, was sign-extended


def and_mask(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i] + (7 << 32)
        o[i] = x & 0xFFFFFFFF  # was -1, keeping every bit


def cmp_shift(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i] + (1 << 35)
        c: UInt(64) = 0
        if x >= (1 << 35):
            c = 1
        o[i] = c


def cmp_eq_var_shift(a: U64[2], o: U64[2]):
    for i in range(2):
        k: int32 = 35
        x: UInt(64) = a[i] | (1 << 35)
        c: UInt(64) = 0
        if (x & (1 << k)) == (1 << k):
            c = 1
        o[i] = c


def i64_or(a: U64[2], o: U64[2]):
    for i in range(2):
        x: Int(64) = a[i]
        y: Int(64) = x | (1 << 33)
        o[i] = y


def i64_neg(a: U64[2], o: U64[2]):
    for i in range(2):
        x: Int(64) = a[i]
        y: Int(64) = x + (-(1 << 40))
        z: Int(64) = y + (1 << 40)
        o[i] = z


def ann_target(a: U64[2], o: U64[2]):
    for i in range(2):
        y: UInt(64) = 1 << 35
        o[i] = y + a[i]


def var_shift(a: U64[2], o: U64[2]):
    for i in range(2):
        k: int32 = 35
        x: UInt(64) = a[i]
        o[i] = x | (1 << k)


def var_shift_target(a: U64[2], o: U64[2]):
    for i in range(2):
        k: int32 = 35
        y: UInt(64) = 1 << k
        o[i] = y | a[i]


def var_mask(a: U64[2], o: U64[2]):
    for i in range(2):
        k: int32 = 40
        x: UInt(64) = a[i] | (1 << 39) | (1 << 45)
        o[i] = x & ((1 << k) - 1)


def global_big(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x | BIG


def aug_or(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        x |= 1 << 35
        o[i] = x


def store_literal(a: U64[2], o: U64[2]):
    for i in range(2):
        o[i] = (1 << 35) + i


def invert_mask(a: U64[2], o: U64[2]):
    for i in range(2):
        x: UInt(64) = a[i] | (1 << 35)
        o[i] = x & ~(1 << 35)


WIDE = [
    (or_shift, [1 << 35, (1 << 35) | 5]),
    (and_shift, [1 << 35, 1 << 35]),
    (add_shift, [1 << 40, (1 << 40) + 5]),
    (add_lone, [1 << 40, (1 << 40) + 5]),
    (add_mask, [1 << 31, (1 << 31) + 5]),
    (and_mask, [0, 5]),
    (cmp_shift, [1, 1]),
    (cmp_eq_var_shift, [1, 1]),
    (i64_or, [1 << 33, (1 << 33) | 5]),
    (i64_neg, [0, 5]),
    (ann_target, [1 << 35, (1 << 35) + 5]),
    (var_shift, [1 << 35, (1 << 35) | 5]),
    (var_shift_target, [1 << 35, (1 << 35) | 5]),
    (var_mask, [1 << 39, (1 << 39) | 5]),
    (global_big, [1 << 40, (1 << 40) | 5]),
    (aug_or, [1 << 35, (1 << 35) | 5]),
    (store_literal, [1 << 35, (1 << 35) + 1]),
    (invert_mask, [0, 5]),
]


@pytest.mark.parametrize("fn,want", WIDE, ids=[f.__name__ for f, _ in WIDE])
def test_wide_literal_llvm(fn, want):
    assert _run(fn) == want


# ---- narrow operands: values as before ----


def n_u8_or(a: U64[2], o: U64[2]):
    for i in range(2):
        x: uint8 = a[i]
        y: uint8 = x | (1 << 7)
        o[i] = y


def n_u8_wrap(a: U64[2], o: U64[2]):
    for i in range(2):
        x: uint8 = a[i]
        y: uint8 = x + 300
        o[i] = y


def n_u32_or31(a: U64[2], o: U64[2]):
    for i in range(2):
        x: uint32 = a[i]
        y: uint32 = x | (1 << 31)
        o[i] = y


def n_i32_mask(a: U64[2], o: U64[2]):
    for i in range(2):
        x: int32 = a[i]
        c: uint32 = 0
        if (x | 0xFFFFFFFF) == 0xFFFFFFFF:  # -1 == the pattern: as before
            c = 1
        o[i] = c


def n_i8_neg(a: U64[2], o: U64[2]):
    for i in range(2):
        x: int8 = a[i]
        y: int8 = x & -2
        o[i] = y


def n_i32_var_shift(a: U64[2], o: U64[2]):
    for i in range(2):
        k: int32 = 3
        x: int32 = a[i]
        o[i] = x | (1 << k)


NARROW = [
    (n_u8_or, [128, 133]),
    (n_u8_wrap, [44, 49]),
    (n_u32_or31, [1 << 31, (1 << 31) | 5]),
    (n_i32_mask, [1, 1]),
    (n_i8_neg, [0, 4]),
    (n_i32_var_shift, [8, 13]),
]


@pytest.mark.parametrize("fn,want", NARROW, ids=[f.__name__ for f, _ in NARROW])
def test_narrow_unchanged(fn, want):
    assert _run(fn) == want


def lit_narrow(a: int32[4], o: int32[4]):
    for i in range(4):
        x: int32 = a[i]
        o[i] = ((x | (1 << 4)) & 0xFFFFFFFF) + 300 - (x >> 31)


# the arith lines of `lit_narrow` built by the front end BEFORE the E-F1 rules
# (captured from origin/u1-pilot 0e6f950a; SSA names elided)
LIT_NARROW_GOLDEN = """\
arith.constant 1 : i32
arith.constant 1 : i32
arith.constant 4 : i32
arith.constant 4 : i32
arith.shli %v, %v : i32
arith.ori %v, %v : i32
arith.constant -1 : i32
arith.constant -1 : i32
arith.andi %v, %v : i32
arith.extsi %v : i32 to i33
arith.constant 300 : i32
arith.constant 300 : i32
arith.extsi %v : i32 to i33
arith.addi %v, %v : i33
arith.constant 31 : i32
arith.constant 31 : i32
arith.shrsi %v, %v : i32
arith.extsi %v : i33 to i34
arith.extsi %v : i32 to i34
arith.subi %v, %v : i34
arith.trunci %v : i34 to i32"""


def test_narrow_ir_unchanged():
    """Literals that are 32-bit patterns next to operands of at most 32 bits
    build exactly the IR they did before the E-F1 rules."""
    import re

    lines = []
    for ln in str(allo.customize(lit_narrow).module).splitlines():
        if "arith." in ln:
            ln = re.sub(r"%[\w-]+", "%v", ln.strip())
            lines.append(ln.split("= ", 1)[1] if "= " in ln else ln)
    assert "\n".join(lines) == LIT_NARROW_GOLDEN


def sum_mask(a: U64[2], o: U64[2]):
    for i in range(2):
        x: int32 = a[i]
        y: int32 = 7
        w: Int(64) = (x - y) & 0xFFFFFFFF  # the Int(33) difference, masked
        n: int32 = (x - y) & 0xFFFFFFFF
        o[i] = w + n


def test_pattern_literal_next_to_int33():
    """The one narrow-source change: Allo types ``int32 - int32`` as Int(33),
    and next to it ``0xFFFFFFFF`` is now 4294967295, not -1 sign-extended.
    The int32 result is the same; a wider result is the masked value."""
    got = _run(sum_mask, np.array([0, 5], dtype=np.uint64))
    # x - 7 = -7 and -2: masked to 32 bits (w), and as int32 (n, unchanged)
    want = [((-7) & 0xFFFFFFFF) + (-7), ((-2) & 0xFFFFFFFF) + (-2)]
    assert got == [v & ((1 << 64) - 1) for v in want]


# ---- refusals ----


def r_target(a: U64[2], o: U64[2]):
    for i in range(2):
        y: UInt(64) = 1 << 70
        o[i] = y


def r_narrow_target(a: U64[2], o: U64[2]):
    for i in range(2):
        y: uint32 = 1 << 35
        o[i] = y


def r_store(a: U64[2], o: U64[2]):
    for i in range(2):
        o[i] = 1 << 64


@pytest.mark.parametrize("fn,frag", [
    (r_target, "`1 << 70`"),
    (r_narrow_target, "`1 << 35`"),
    (r_store, "`1 << 64`"),
], ids=["wide_target", "narrow_target", "array_store"])
def test_unfit_literal_refused(fn, frag, capsys):
    # customize() reports a front-end error and exits (limitations item C)
    with pytest.raises(SystemExit):
        allo.customize(fn)
    out = " ".join(capsys.readouterr().out.split())
    assert frag in out and "does not fit" in out, out


# ---- the limits repro (tests/limits/new_wide_literal_shift.py), as a test ----


def test_repro_llvm_and_simulator():
    def lit(A: uint32[2], O: uint32[2]):
        for i in range(2):
            x: UInt(64) = A[i]
            y: UInt(64) = x | (1 << 35)
            O[i] = y[32:64]

    @df.region()
    def top(A: uint32[2], O: uint32[2]):
        @df.kernel(mapping=[1], args=[A, O])
        def k(a: uint32[2], o: uint32[2]):
            for i in range(2):
                x: UInt(64) = a[i]
                y: UInt(64) = x | (1 << 35)
                o[i] = y[32:64]

    a = np.array([0, 5], dtype=np.uint32)
    o = np.zeros(2, dtype=np.uint32)
    allo.customize(lit).build(target="llvm")(a, o)
    assert o.tolist() == [8, 8]
    o = np.zeros(2, dtype=np.uint32)
    df.build(top, target="simulator")(a, o)
    assert o.tolist() == [8, 8]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
