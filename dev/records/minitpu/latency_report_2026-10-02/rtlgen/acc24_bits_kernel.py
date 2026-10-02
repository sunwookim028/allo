# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniTPU mxu_acc24_add_pipe `bits` (examples/minitpu/units/acc24_add_pipe.py::bits)
transcribed to RTLGen's frontend (kkkaishao/allo allo-rtlgen @ 13b55a63). Loop body copied
verbatim (dedented); UInt(n) -> APInt(n);
the leading_zeros19 loop helper -> an unrolled 19-way priority chain (last write wins)."""
import os
from allo import kernel
from allo.lang import u32
from allo.lang.core import APInt

N = int(os.environ.get("ACC_N", "16"))
UInt = lambda n: APInt(n)
uint1 = APInt(1)


@kernel
def acc24_add_bits(av: u32[N], bv: u32[N], cv: u32[N]):
    for i in range(N, name="i"):
        # ---- stage 1: classify, align (jam), add ----
        a_i: UInt(24) = av[i]
        b_i: UInt(24) = bv[i]
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
        key_a: UInt(24) = 0  # {exp, frac} is 23 bits; spare top bit (B1)
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
        # align_significand_jam: shift {sig, 3'b0} right, OR every
        # discarded bit into the LSB. 40 bits: no shift reaches the width.
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
        # RTLGen frontend: `+` widens to uint20, `-` to int21; typed temps unify them.
        mag_add: UInt(20) = mant_large + small_aligned
        mag_sub: UInt(20) = mant_large - small_aligned
        magnitude_s1: UInt(20) = mag_add if same_sign else mag_sub
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
        # stage-1 registers
        s1_special: UInt(2) = special_s1
        s1_sign: uint1 = sign_large
        s1_mag: UInt(20) = magnitude_s1
        s1_exp: UInt(9) = exp_large

        # ---- stage 2: normalize ----
        norm_overflow: uint1 = s1_mag[19]
        norm_needed: uint1 = 0
        if s1_mag != 0 and not s1_mag[18] and not norm_overflow:
            norm_needed = 1
        lzc: UInt(5) = 19
        lzv: UInt(19) = s1_mag[0:19]
        if lzv[0]:
            lzc = 18
        if lzv[1]:
            lzc = 17
        if lzv[2]:
            lzc = 16
        if lzv[3]:
            lzc = 15
        if lzv[4]:
            lzc = 14
        if lzv[5]:
            lzc = 13
        if lzv[6]:
            lzc = 12
        if lzv[7]:
            lzc = 11
        if lzv[8]:
            lzc = 10
        if lzv[9]:
            lzc = 9
        if lzv[10]:
            lzc = 8
        if lzv[11]:
            lzc = 7
        if lzv[12]:
            lzc = 6
        if lzv[13]:
            lzc = 5
        if lzv[14]:
            lzc = 4
        if lzv[15]:
            lzc = 3
        if lzv[16]:
            lzc = 2
        if lzv[17]:
            lzc = 1
        if lzv[18]:
            lzc = 0
        max_ns: UInt(5) = 18 if s1_exp > 19 else s1_exp - 1
        lz6: UInt(6) = lzc  # spare top bit (B1): 19 is negative as i5
        max6: UInt(6) = max_ns
        normalize_shift: UInt(5) = 0
        if norm_needed:
            normalize_shift = lzc if lz6 < max6 else max_ns
        mag19: UInt(19) = s1_mag[0:19]
        mag_normalized: UInt(19) = mag19 << normalize_shift
        mag_s2: UInt(20) = s1_mag
        exp_s2: UInt(10) = s1_exp  # 10 bits: compares stay unsigned (B1)
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
        cv[i] = packed

