"""Dump the IR of narrow-literal kernels (run under the pre-fix and the fixed tree; diff)."""
import sys, os
import allo
from allo.ir.types import int8, int16, int32, uint8, uint16, uint32, UInt, Int, index

MASK = 0xFF
SH = 3
N = 8


def k_shift_or(a: int32[N], o: int32[N]):
    for i in range(N):
        x: int32 = a[i]
        o[i] = (x | (1 << 4)) ^ (x >> 2)


def k_masks(a: uint32[N], o: uint32[N]):
    for i in range(N):
        x: uint32 = a[i]
        o[i] = (x & 0xFFFFFFFF) | (x & 0x80000000) | (1 << 31)


def k_cmp32(a: int32[N], o: int32[N]):
    for i in range(N):
        x: int32 = a[i]
        c: int32 = 0
        if x == 0xFFFFFFFF:
            c = 1
        if x < -5:
            c = c + 2
        o[i] = c


def k_u8(a: uint8[N], o: uint8[N]):
    for i in range(N):
        x: uint8 = a[i]
        y: uint8 = (x + 300) & MASK
        o[i] = y | (1 << 7)


def k_i8_neg(a: int8[N], o: int8[N]):
    for i in range(N):
        x: int8 = a[i]
        o[i] = (x & -2) - 128


def k_globals(a: int16[N], o: int16[N]):
    for i in range(N):
        x: int16 = a[i]
        o[i] = (x << SH) + N * 4 - 1


def k_index(a: int32[N], o: int32[N]):
    for i in range(1, N - 1):
        o[i] = a[i + 1] + a[i - 1] + (i << 1)


def k_var_shift(a: int32[N], o: int32[N]):
    for i in range(N):
        k: int32 = a[i] & 7
        o[i] = (1 << k) | (3 << (k + 1))


def k_aug(a: uint16[N], o: uint16[N]):
    for i in range(N):
        x: uint16 = a[i]
        x |= 1 << 15
        x += 70000
        o[i] = x


def k_wide_small(a: UInt(64)[N], o: UInt(64)[N]):
    for i in range(N):
        x: UInt(64) = a[i]
        o[i] = (x | 5) + (x & 255) - 1  # small literals next to a wide operand


KS = [k_shift_or, k_masks, k_cmp32, k_u8, k_i8_neg, k_globals, k_index, k_var_shift, k_aug, k_wide_small]
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for fn in KS:
    try:
        t = str(allo.customize(fn).module)
    except BaseException as e:  # noqa: BLE001
        t = f"ERROR {type(e).__name__}: {e}"
    open(os.path.join(out, fn.__name__ + ".mlir"), "w").write(t)
    print(fn.__name__, len(t))
