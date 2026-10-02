# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-level numpy references for MiniTPU's arithmetic units.

Each reference states the RTL's semantics, not IEEE's: where the two differ,
the difference is named here and was measured against the RTL with
``harness/rtl.py``. The Allo side is compared with the RTL directly; these
references exist to *explain* a mismatch, not to replace the RTL as oracle.
"""

import numpy as np


def _bf16_to_f32(x):
    return (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(f):
    """float32 bits -> bf16 bits, round to nearest even, NaN kept with sign."""
    f = np.asarray(f, dtype=np.float32).view(np.uint32)
    out = ((f + 0x7FFF + ((f >> 16) & 1)) >> 16).astype(np.uint16)
    nan = (((f >> 23) & 0xFF) == 0xFF) & ((f & 0x7FFFFF) != 0)
    out[nan] = (((f[nan] >> 31) << 15) | 0x7FC0).astype(np.uint16)
    return out


def ieee_bf16_add(a, b):
    """IEEE semantics: exact sum, one RNE rounding to bf16.

    fp32 has 24 significand bits >= 2*8+2, so rounding the exact sum to fp32
    first and then to bf16 is innocuous (no double-rounding error).
    """
    with np.errstate(all="ignore"):
        return f32_to_bf16_rne(_bf16_to_f32(a) + _bf16_to_f32(b))


def vpu_bf16_add(a, b):
    """``vpu_bf16_add.sv`` at b3ba0a4d: IEEE RNE add with subnormals, except

    * every NaN result is the positive canonical NaN ``0x7FC0`` (IEEE keeps a
      sign; MiniTPU's ``tb/tb_bf16_add.cpp`` skips specials, so its own test
      does not see this);
    * ``(+0) + (-0)`` is ``-0`` (IEEE RNE gives ``+0``). Measured on the RTL
      2026-10-02 over 251,936 vectors (corners crossed, ties, random): these
      two are the only differences.
    """
    a = np.asarray(a, dtype=np.uint16)
    b = np.asarray(b, dtype=np.uint16)
    out = ieee_bf16_add(a, b)
    nan = ((out & 0x7F80) == 0x7F80) & ((out & 0x7F) != 0)
    out[nan] = 0x7FC0
    zz = ((a & 0x7FFF) == 0) & ((b & 0x7FFF) == 0) & (a != b)
    out[zz] = b[zz]  # (+0)+(-0) -> -0 measured; (-0)+(+0) -> +0 measured
    return out


# ---------------------------------------------------------------------------
# Exact rounding, shared by the units below. A value is ``sign``, an integer
# magnitude ``mag`` and a power-of-two ``scale`` (value = mag * 2**scale), so
# products and sums are exact before the one rounding the hardware does.
# Formats have an 8-bit exponent, bias 127, and ``frac`` fraction bits: bf16
# is frac=7, MiniTPU's acc24 is frac=15.

ACC24_NAN = 0x7FC000  # sign 0, exponent 0xff, fraction MSB: every unit's NaN


def _decode(x, frac):
    """Fields of a 1+8+``frac`` float: sign, significand, effective exponent."""
    x = np.asarray(x, dtype=np.int64)
    sign = (x >> (8 + frac)) & 1
    exp = (x >> frac) & 0xFF
    f = x & ((1 << frac) - 1)
    sig = np.where(exp != 0, f | (1 << frac), f)
    eeff = np.maximum(exp, 1)  # subnormals share exponent 1's scale
    return sign, exp, f, sig, eeff


def _bitlen(m):
    n = np.zeros(m.shape, dtype=np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        big = (m >> s) > 0
        n += s * big
        m = np.where(big, m >> s, m)
    return n + (m > 0)


def _pack(sign, mag, scale, frac):
    """Round ``mag * 2**scale`` once to nearest-even in the 1+8+``frac`` format.

    Gradual underflow, overflow to signed infinity, a zero keeps ``sign``.
    ``mag`` must be below 2**62.
    """
    sign = np.asarray(sign, dtype=np.int64)
    mag = np.asarray(mag, dtype=np.int64)
    scale = np.asarray(scale, dtype=np.int64)
    n = _bitlen(mag)
    e = np.maximum(scale + n - 1, -126)  # exponent of the result's hidden bit
    shift = (e - frac) - scale  # bits below the result's LSB
    sh = np.clip(shift, 0, 62)
    q = mag >> sh
    rem = mag & ((np.int64(1) << sh) - 1)
    half = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((q & 1) == 1)))
    up &= shift <= 62  # beyond, the value is under half an LSB
    q = np.where(shift > 0, q + up, mag << np.clip(-shift, 0, 62))
    carry = q >> (frac + 1) != 0
    q = np.where(carry, q >> 1, q)
    e = e + carry
    normal = q >> frac != 0
    biased = np.where(normal, e + 127, 0)
    out = (sign << (8 + frac)) | (biased << frac) | (q & ((1 << frac) - 1))
    inf = (sign << (8 + frac)) | (0xFF << frac)
    return np.where(biased >= 255, inf, out)


def _mul_exact(a, b):
    """bf16 x bf16, exact: sign, 16-bit magnitude, scale."""
    sa, _, _, ma, ea = _decode(a, 7)
    sb, _, _, mb, eb = _decode(b, 7)
    return sa ^ sb, ma * mb, ea + eb - 2 * (127 + 7)


def _bf16_class(x):
    x = np.asarray(x, dtype=np.int64)
    nan = ((x & 0x7F80) == 0x7F80) & ((x & 0x7F) != 0)
    return nan, (x & 0x7FFF) == 0x7F80, (x & 0x7FFF) == 0


def _mul_specials(a, b):
    """NaN and Inf of a bf16 product: ``(nan, inf)`` masks.

    IEEE and the RTL agree on which operand pairs these are, because the RTL's
    "Inf x 0" test reads the literal pattern: ``Inf x subnormal`` is ``Inf``
    there too, even though the RTL flushes a subnormal operand otherwise.
    """
    na, ia, za = _bf16_class(a)
    nb, ib, zb = _bf16_class(b)
    nan = na | nb | (ia & zb) | (ib & za)
    return nan, ~nan & (ia | ib)


def ieee_bf16_mul(a, b):
    """IEEE: exact product, one RNE rounding to bf16, gradual underflow.

    A NaN result is ``0x7FC0`` with the product's sign (IEEE leaves the sign
    of a NaN unspecified; only "is NaN" is compared).
    """
    s, m, sc = _mul_exact(a, b)
    nan, inf = _mul_specials(a, b)
    out = np.where(inf, (s << 15) | 0x7F80, _pack(s, m, sc, 7))
    return np.where(nan, (s << 15) | 0x7FC0, out).astype(np.uint16)


def _ftz_mul(a, b, frac, nan_value):
    """The two multipliers' shared structure, as the RTL orders its tests."""
    s, m, sc = _mul_exact(a, b)
    nan, inf = _mul_specials(a, b)
    ea = (np.asarray(a, dtype=np.int64) >> 7) & 0xFF
    eb = (np.asarray(b, dtype=np.int64) >> 7) & 0xFF
    flush = (ea == 0) | (eb == 0)  # a zero or subnormal operand
    flush |= sc + _bitlen(m) - 1 < -126  # |exact product| < 2**-126
    top = 8 + frac
    out = np.where(flush, s << top, _pack(s, m, sc, frac))
    out = np.where(inf, (s << top) | (0xFF << frac), out)
    return np.where(nan, nan_value, out)


def vpu_bf16_mul(a, b):
    """``vpu_bf16_mul`` and ``vpu_bf16_mul_pipe`` at b3ba0a4d.

    The exact product rounded once to nearest-even, overflowing to signed
    infinity, except:

    * a subnormal operand is zero: the result is the signed zero, not the
      (possibly normal) product (flush-to-zero on input);
    * a product below ``2**-126`` before rounding is the signed zero, even one
      that would round up to the smallest normal (flush-to-zero on output);
    * ``Inf x subnormal`` is ``Inf``, not NaN: the Inf x 0 test reads the bit
      pattern before the input flush (ARITHMETIC.md section 6);
    * every NaN result is ``+0x7FC0``.
    """
    return _ftz_mul(a, b, 7, 0x7FC0).astype(np.uint16)


def ieee_bf16_mul_acc24(a, b):
    """IEEE: bf16 x bf16 rounded once into acc24 (1+8+15, gradual underflow).

    Exact unless the product is below acc24's smallest normal.
    """
    s, m, sc = _mul_exact(a, b)
    nan, inf = _mul_specials(a, b)
    out = np.where(inf, (s << 23) | 0x7F8000, _pack(s, m, sc, 15))
    return np.where(nan, (s << 23) | ACC24_NAN, out).astype(np.uint32)


def mxu_bf16_mul_acc24(a, b):
    """``mxu_bf16_mul_acc24`` at b3ba0a4d: the exact product in acc24, except

    * a subnormal operand is zero (flush on input), as in ``vpu_bf16_mul``;
    * a product below ``2**-126`` is the signed zero although acc24 has
      subnormals that could hold it (flush on output);
    * a product of ``2**128`` or more is signed infinity (acc24's own range);
    * ``Inf x subnormal`` is ``Inf``; every NaN is ``+0x7FC000``.
    """
    return _ftz_mul(a, b, 15, ACC24_NAN).astype(np.uint32)


_GAP = 40  # far operands become a sticky unit; RNE cannot tell the difference


def _add_exact(a, b, frac):
    """``a + b`` (finite) as sign, magnitude, scale; |a|>=|b| ordering inside."""
    sa, _, _, ma, ea = _decode(a, frac)
    sb, _, _, mb, eb = _decode(b, frac)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    low = (1 << (8 + frac)) - 1
    a_large = (a & low) >= (b & low)
    sl, ml, el = np.where(a_large, sa, sb), np.where(a_large, ma, mb), np.where(a_large, ea, eb)
    ss, ms, es = np.where(a_large, sb, sa), np.where(a_large, mb, ma), np.where(a_large, eb, ea)
    gap = el - es
    small = np.where(gap <= _GAP, ms << np.clip(_GAP - gap, 0, _GAP), (ms != 0).astype(np.int64))
    mag = np.where(sl == ss, (ml << _GAP) + small, (ml << _GAP) - small)
    return sl, mag, el - 127 - frac - _GAP


def ieee_acc24_add(a, b):
    """IEEE binary-style add in acc24: RNE once, gradual underflow, overflow
    to Inf, ``(+0)+(-0) = +0``; NaN keeps no particular sign."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    s, m, sc = _add_exact(a, b, 15)
    s = np.where(m == 0, ((a & b) >> 23) & 1, s)  # exact zero: +0 unless both -0
    out = _pack(s, m, sc, 15)
    isnan = ((a >> 15) & 0xFF == 0xFF) & ((a & 0x7FFF) != 0), ((b >> 15) & 0xFF == 0xFF) & ((b & 0x7FFF) != 0)
    ia, ib = (a & 0x7FFFFF) == 0x7F8000, (b & 0x7FFFFF) == 0x7F8000
    nan = isnan[0] | isnan[1] | (ia & ib & ((a ^ b) >> 23 == 1))
    out = np.where(ia, a, np.where(ib, b, out))
    out = np.where(nan, ACC24_NAN | (((a | b) >> 23 & 1) << 23), out)
    return out.astype(np.uint32)


def mxu_acc24_add(a, b):
    """``mxu_acc24_add_pipe`` at b3ba0a4d: IEEE RNE add in acc24 with
    subnormals and overflow to Inf (``ieee_acc24_add``), except

    * every NaN result is ``+0x7FC000``;
    * an operand whose 23 low bits are zero is a bypass: the result is the
      other operand bit for bit, so ``(+0) + (-0) = -0`` and
      ``(-0) + (+0) = +0`` (IEEE: ``+0`` for both), as in ``vpu_bf16_add``.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    out = ieee_acc24_add(a, b).astype(np.int64)
    nan = ((out >> 15) & 0xFF == 0xFF) & ((out & 0x7FFF) != 0)
    out = np.where(nan, ACC24_NAN, out)
    za, zb = (a & 0x7FFFFF) == 0, (b & 0x7FFFFF) == 0
    special = nan | ((a >> 15) & 0xFF == 0xFF) | ((b >> 15) & 0xFF == 0xFF)
    out = np.where(~special & za, b, np.where(~special & zb, a, out))
    return out.astype(np.uint32)


# vpu_pkg::vpu_alu_op_e, in declaration order (logic [3:0], from 0).
ALU_OPS = ("ADD", "SUB", "MUL", "MOV", "MAX", "MIN", "AND", "OR", "XOR")
ALU_OP = {name: i for i, name in enumerate(ALU_OPS)}


def bf16_gt(a, b):
    """``vpu_pkg::bf16_gt``: unsigned compare of a sign-flipped key.

    A total order on bit patterns: ``-NaN < -Inf < ... < -0 < +0 < ... <
    +Inf < +NaN``, and NaNs ordered among themselves by payload.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    key = lambda x: np.where(x >> 15 == 1, ~x & 0xFFFF, x ^ 0x8000)
    return key(a) > key(b)


def vpu_alu(op, a, b):
    """``vpu_alu`` at b3ba0a4d, one lane, by ``vpu_alu_op_e`` value.

    * ADD is ``vpu_bf16_add``; SUB is ``vpu_bf16_add(a, b ^ 0x8000)``, so
      ``(+0) - (+0) = -0`` (IEEE: ``+0``) and ``(-0) - (-0) = +0``;
    * MUL is ``vpu_bf16_mul`` (flush-to-zero both ways);
    * MAX / MIN select by ``bf16_gt``: ``-0 < +0``, a positive NaN wins MAX and
      a negative NaN wins MIN, payload and sign kept; a NaN on the losing side
      is dropped (neither IEEE 754-2019 ``maximum``, which returns NaN, nor
      ``maximumNumber``, which drops every NaN);
    * MOV, AND, OR, XOR and the unused codes 9..15 all return ``a`` bit for
      bit: the result mux's ``default`` arm. AND/OR/XOR are declared in the
      enum and listed in ``docs/UNITS.md`` but not implemented; the decoder
      never issues them.
    """
    op = np.asarray(op, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    gt = bf16_gt(a, b)
    out = np.select(
        [op == 0, op == 1, op == 2, op == 4, op == 5],
        [vpu_bf16_add(a, b), vpu_bf16_add(a, b ^ 0x8000), vpu_bf16_mul(a, b),
         np.where(gt, a, b), np.where(gt, b, a)],
        a,
    )
    return out.astype(np.uint16)


def ieee_vpu_alu(op, a, b):
    """What the op names promise: IEEE add/sub/mul, IEEE 754-2019
    ``maximum``/``minimum`` (NaN wins, ``-0 < +0``), bitwise AND/OR/XOR."""
    op = np.asarray(op, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    nan = lambda x: ((x & 0x7F80) == 0x7F80) & ((x & 0x7F) != 0)
    gt = bf16_gt(a, b)
    anynan = nan(a) | nan(b)
    mx = np.where(anynan, 0x7FC0, np.where(gt, a, b))
    mn = np.where(anynan, 0x7FC0, np.where(gt, b, a))
    out = np.select(
        [op == 0, op == 1, op == 2, op == 4, op == 5, op == 6, op == 7, op == 8],
        [ieee_bf16_add(a, b), ieee_bf16_add(a, b ^ 0x8000), ieee_bf16_mul(a, b),
         mx, mn, a & b, a | b, a ^ b],
        a,
    )
    return out.astype(np.uint16)


# Scalar predicates for classification rules (units' DEVIATIONS / EXPLAIN).
def is_nan(x, frac=7):
    x = int(x)
    return (x >> frac) & 0xFF == 0xFF and x & ((1 << frac) - 1) != 0


def exp_field(x, frac=7):
    return (int(x) >> frac) & 0xFF


def is_zero(x, frac=7):
    return int(x) & ((1 << (8 + frac)) - 1) == 0
