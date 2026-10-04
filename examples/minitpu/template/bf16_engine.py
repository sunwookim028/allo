# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniTPU's MXU arithmetic as module-level functions another unit can call.

Lifted from U1's kernels (``units/mul_acc24.py`` ``bits``,
``units/acc24_add_pipe.py`` ``bits``), which hold them bit-exact against the
RTL; here the same text is a *function* so that a composed unit can bind it
as its MAC engine. acc24 (1+8+15) travels as ``UInt(24)``. SV ``x[hi:lo]`` is
Allo ``x[lo:hi+1]``; unsigned compares get a spare zero bit on top (B1).
"""

from allo.ir.types import UInt, uint1


def leading_zeros19(value: UInt(19)) -> UInt(5):
    lz: UInt(5) = 19
    found: uint1 = 0
    for offset in range(19):
        if not found and value[18 - offset]:
            lz = offset
            found = 1
    return lz


def mul_acc24_bits(a_i: UInt(16), b_i: UInt(16)) -> UInt(24):
    """``mxu_bf16_mul_acc24.sv``: exact bf16 x bf16 -> acc24, flush-to-zero."""
    sign: uint1 = a_i[15] ^ b_i[15]
    exp_a: UInt(8) = a_i[7:15]
    exp_b: UInt(8) = b_i[7:15]
    frac_a: UInt(7) = a_i[0:7]
    frac_b: UInt(7) = b_i[0:7]
    mant_a: UInt(8) = 0
    mant_a[7] = 1
    mant_a[0:7] = frac_a
    mant_b: UInt(8) = 0
    mant_b[7] = 1
    mant_b[0:7] = frac_b
    product: UInt(16) = mant_a * mant_b

    exp_sum: UInt(9) = exp_a + exp_b
    exp_low: UInt(9) = exp_sum - 127
    exp_high: UInt(9) = exp_sum - 126

    finite_low: UInt(24) = 0
    finite_low[23] = sign
    finite_low[15:23] = exp_low[0:8]
    finite_low[1:15] = product[0:14]
    if exp_sum <= 127:
        finite_low = 0
        finite_low[23] = sign
    elif exp_sum >= 382:
        finite_low = 0
        finite_low[23] = sign
        finite_low[15:23] = 0xFF

    finite_high: UInt(24) = 0
    finite_high[23] = sign
    finite_high[15:23] = exp_high[0:8]
    finite_high[0:15] = product[0:15]
    if exp_sum <= 126:
        finite_high = 0
        finite_high[23] = sign
    elif exp_sum >= 381:
        finite_high = 0
        finite_high[23] = sign
        finite_high[15:23] = 0xFF

    finite_result: UInt(24) = finite_high if product[15] else finite_low

    result_o: UInt(24) = 0
    if (
        (exp_a == 0xFF and frac_a != 0)
        or (exp_b == 0xFF and frac_b != 0)
        or (exp_a == 0xFF and b_i[0:15] == 0)
        or (exp_b == 0xFF and a_i[0:15] == 0)
    ):
        result_o = 0x7FC000
    elif exp_a == 0xFF or exp_b == 0xFF:
        result_o[23] = sign
        result_o[15:23] = 0xFF
    elif exp_a == 0 or exp_b == 0:
        result_o[23] = sign
    else:
        result_o = finite_result
    return result_o


def acc24_add_bits(a_i: UInt(24), b_i: UInt(24)) -> UInt(24):
    """``mxu_acc24_add_pipe.sv``'s three stages in sequence: RNE add in acc24
    with subnormals, overflow to Inf, zero bypass, every NaN ``+0x7FC000``."""
    # ---- stage 1: classify, align (jam), add ----
    sign_a: uint1 = a_i[23]
    sign_b: uint1 = b_i[23]
    exp_a: UInt(8) = a_i[15:23]
    exp_b: UInt(8) = b_i[15:23]
    frac_a: UInt(15) = a_i[0:15]
    frac_b: UInt(15) = b_i[0:15]
    sig_a: UInt(16) = 0
    sig_a[15] = exp_a != 0
    sig_a[0:15] = frac_a
    sig_b: UInt(16) = 0
    sig_b[15] = exp_b != 0
    sig_b[0:15] = frac_b
    same_sign: uint1 = sign_a == sign_b
    key_a: UInt(24) = 0
    key_a[0:23] = a_i[0:23]
    key_b: UInt(24) = 0
    key_b[0:23] = b_i[0:23]
    a_is_large: uint1 = key_a >= key_b
    sign_large: uint1 = sign_b
    exp_large: UInt(9) = 0
    exp_small: UInt(9) = 0
    sig_large: UInt(16) = 0
    sig_small: UInt(16) = 0
    if a_is_large:
        sign_large = sign_a
        exp_large = 1 if exp_a == 0 else exp_a
        exp_small = 1 if exp_b == 0 else exp_b
        sig_large = sig_a
        sig_small = sig_b
    else:
        exp_large = 1 if exp_b == 0 else exp_b
        exp_small = 1 if exp_a == 0 else exp_a
        sig_large = sig_b
        sig_small = sig_a
    exp_diff: UInt(9) = exp_large - exp_small
    align_shift: UInt(5) = 19 if exp_diff >= 19 else exp_diff[0:5]
    wide: UInt(40) = 0
    wide[3:19] = sig_small
    one: UInt(40) = 1
    mask: UInt(40) = (one << align_shift) - one
    jam: uint1 = (wide & mask) != 0
    shifted: UInt(40) = wide >> align_shift
    small_aligned: UInt(19) = shifted[0:19]
    small_aligned[0] = small_aligned[0] | jam
    mant_large: UInt(19) = 0
    mant_large[3:19] = sig_large
    magnitude_s1: UInt(20) = (
        (mant_large + small_aligned)
        if same_sign
        else (mant_large - small_aligned)
    )
    special_s1: UInt(2) = 0  # NORMAL 0, BYPASS 1, INF 2, NAN 3
    if (
        (exp_a == 0xFF and frac_a != 0)
        or (exp_b == 0xFF and frac_b != 0)
        or (exp_a == 0xFF and exp_b == 0xFF and sign_a != sign_b)
    ):
        special_s1 = 3
    elif exp_a == 0xFF or exp_b == 0xFF:
        special_s1 = 2
    elif a_i[0:23] == 0:
        special_s1 = 1
        sign_large = sign_b
    elif b_i[0:23] == 0:
        special_s1 = 1
        sign_large = sign_a
    s1_special: UInt(2) = special_s1
    s1_sign: uint1 = sign_large
    s1_mag: UInt(20) = magnitude_s1
    s1_exp: UInt(9) = exp_large

    # ---- stage 2: normalize ----
    norm_overflow: uint1 = s1_mag[19]
    norm_needed: uint1 = 0
    if s1_mag != 0 and not s1_mag[18] and not norm_overflow:
        norm_needed = 1
    lzc: UInt(5) = leading_zeros19(s1_mag[0:19])
    max_ns: UInt(5) = 18 if s1_exp > 19 else s1_exp - 1
    lz6: UInt(6) = lzc
    max6: UInt(6) = max_ns
    normalize_shift: UInt(5) = 0
    if norm_needed:
        normalize_shift = lzc if lz6 < max6 else max_ns
    mag19: UInt(19) = s1_mag[0:19]
    mag_normalized: UInt(19) = mag19 << normalize_shift
    mag_s2: UInt(20) = s1_mag
    exp_s2: UInt(10) = s1_exp
    if s1_special == 1:
        exp_s2 = s1_exp
    elif norm_overflow:
        mag_s2[1] = mag_s2[1] | mag_s2[0]
        mag_s2 >>= 1
        exp_s2 = s1_exp + 1
    elif norm_needed:
        mag_s2 = 0
        mag_s2[0:19] = mag_normalized
        exp_s2 = s1_exp - normalize_shift
    s2_special: UInt(2) = s1_special
    s2_sign: uint1 = s1_sign
    s2_mag: UInt(20) = mag_s2
    s2_exp: UInt(10) = exp_s2[0:9]

    # ---- stage 3: round and pack ----
    guard_bit: uint1 = s2_mag[2]
    round_bit: uint1 = s2_mag[1]
    sticky_bit: uint1 = s2_mag[0]
    round_up: uint1 = guard_bit & (round_bit | sticky_bit | s2_mag[3])
    frac16: UInt(16) = 0
    frac16[0:15] = s2_mag[3:18]
    rounded: UInt(16) = frac16 + round_up
    inc_exp: UInt(10) = s2_exp + 1
    packed: UInt(24) = 0
    if s2_special == 3:
        packed = 0x7FC000
    elif s2_special == 2:
        packed[23] = s2_sign
        packed[15:23] = 0xFF
    elif s2_special == 1:
        packed[23] = s2_sign
        if s2_mag[18]:
            packed[15:23] = s2_exp[0:8]
        packed[0:15] = s2_mag[3:18]
    elif s2_mag == 0:
        packed = 0
    elif rounded[15]:
        packed[23] = s2_sign
        if inc_exp >= 255:
            packed[15:23] = 0xFF
        else:
            packed[15:23] = inc_exp[0:8]
    elif s2_exp >= 255:
        packed[23] = s2_sign
        packed[15:23] = 0xFF
    elif s2_exp <= 1 and not s2_mag[18]:
        packed[23] = s2_sign
        packed[0:15] = rounded[0:15]
    else:
        packed[23] = s2_sign
        packed[15:23] = s2_exp[0:8]
        packed[0:15] = rounded[0:15]
    return packed


def pack_bf16_bits(v: UInt(24)) -> UInt(16):
    """``mxu.sv`` ``pack_bf16``: acc24 -> bf16, RNE by ``+ 0x7F + bit 8``,
    keep ``[23:8]`` (no NaN guard: the accumulator canonicalises first)."""
    half: UInt(25) = v + 0x7F
    lsb: UInt(25) = v[8]
    total: UInt(25) = half + lsb
    out: UInt(16) = total[8:24]
    return out
