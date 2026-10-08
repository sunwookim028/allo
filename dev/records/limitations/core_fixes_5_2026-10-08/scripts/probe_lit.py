"""E-F1 before/after table: literal typing against wide and narrow operands (llvm)."""
import sys, traceback
import numpy as np
import allo
from allo.ir.types import UInt, Int, uint32, int32, uint8, int8, uint64, int64

BIG = 1 << 40

def case(name, fn, a, want):
    if len(sys.argv) > 1 and sys.argv[1] != fn.__name__:
        return
    try:
        o = np.zeros(2, dtype=np.uint64)
        allo.customize(fn).build(target="llvm")(a.copy(), o)
        got = o.tolist()
        print(f"{name:28s} {'OK ' if got == want else 'BAD'} got={got} want={want}")
    except Exception as e:
        msg = str(e).strip().splitlines()[-1][:160] if str(e).strip() else type(e).__name__
        print(f"{name:28s} ERR {type(e).__name__}: {msg}")

A = np.array([0, 5], dtype=np.uint64)

def or_shift(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x | (1 << 35)
def and_shift(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: UInt(64) = a[i] + (3 << 34)
        o[i] = x & (1 << 35)
def add_big(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x + (1 << 40)
def add_lone_big(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x + 1099511627776
def cmp_shift(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: UInt(64) = a[i] + (1 << 35)
        c: UInt(64) = 0
        if x >= (1 << 35):
            c = 1
        o[i] = c
def int64_or(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: Int(64) = a[i]
        y: Int(64) = x | (1 << 33)
        o[i] = y
def ann_shift(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        y: UInt(64) = 1 << 35
        o[i] = y + a[i]
def var_shift(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        k: int32 = 35
        x: UInt(64) = a[i]
        o[i] = x | (1 << k)
def mul_big(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x + 65536 * 65536
def global_big(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        o[i] = x | BIG
def aug_or(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: UInt(64) = a[i]
        x |= 1 << 35
        o[i] = x
# narrow: must not change
def n_u8(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: uint8 = a[i]
        y: uint8 = x | (1 << 7)
        o[i] = y
def n_u8_wrap(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: uint8 = a[i]
        y: uint8 = x + 300
        o[i] = y
def n_i32_shift31(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: uint32 = a[i]
        y: uint32 = x | (1 << 31)
        o[i] = y
def n_i8_neg(a: UInt(64)[2], o: UInt(64)[2]):
    for i in range(2):
        x: int8 = a[i]
        y: int8 = x & -2
        o[i] = y

M = (1 << 64) - 1
case("or_shift UInt64", or_shift, A, [1 << 35, (1 << 35) | 5])
case("and_shift UInt64", and_shift, A, [1 << 35, 1 << 35])
case("add (1<<40) UInt64", add_big, A, [1 << 40, (1 << 40) + 5])
case("add lone 2^40 UInt64", add_lone_big, A, [1 << 40, (1 << 40) + 5])
case("cmp >= 1<<35 UInt64", cmp_shift, A, [1, 1])
case("or_shift Int64", int64_or, A, [1 << 33, (1 << 33) | 5])
case("ann y: UInt64 = 1<<35", ann_shift, A, [1 << 35, (1 << 35) + 5])
case("var shift 1<<k UInt64", var_shift, A, [1 << 35, (1 << 35) | 5])
case("65536*65536 UInt64", mul_big, A, [1 << 32, (1 << 32) + 5])
case("global BIG UInt64", global_big, A, [1 << 40, (1 << 40) | 5])
case("x |= 1<<35 UInt64", aug_or, A, [1 << 35, (1 << 35) | 5])
case("narrow u8 | 1<<7", n_u8, A, [128, 133])
case("narrow u8 + 300", n_u8_wrap, A, [44, 49])
case("narrow u32 | 1<<31", n_i32_shift31, A, [1 << 31, (1 << 31) | 5])
case("narrow i8 & -2", n_i8_neg, A, [0, 4])
