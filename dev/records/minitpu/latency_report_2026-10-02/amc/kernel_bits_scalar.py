# Generated from U1 kernel_bits_amc.py by removing the element loop (scalar in,
# scalar out); see scalar_kernel.py. Body otherwise unchanged.
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``vpu_bf16_add`` ``bits`` as a plain (non-dataflow) function.

Derived from ``kernel_bits.py`` (the ``@df.kernel`` body of
``examples/minitpu/units/bf16_add.py::bits``, dedented) by the edits marked with
the finding ids ``An``, each forced by an AMC failure recorded in
``dev/records/minitpu/u1_bf16_add_amc_2026-10-02.rst``.
"""
from allo.ir.types import UInt, uint1, uint16

N = 16


def bf16_add_bits_scalar(a: uint16, b: uint16) -> uint16:
    # A4: leading_zeros17 inlined at its call site (scalar-returning
    # calls fail in FixedPointToInteger::updateCallOp).
    a_i: uint16 = a
    b_i: uint16 = b
    sign_a: uint1 = a_i[15:16]  # A2: x[k] -> x[k:k+1]
    sign_b: uint1 = b_i[15:16]  # A2: x[k] -> x[k:k+1]
    exp_a: UInt(8) = a_i[7:15]
    exp_b: UInt(8) = b_i[7:15]
    frac_a: UInt(7) = a_i[0:7]
    frac_b: UInt(7) = b_i[0:7]
    # A8: slice assignment lowers to a bit-serial scf.for that
    # LoopScheduleToFSM crashes on; build fields with shifts and ors.
    # Not `= exp_a != 0`: our frontend casts that i1 with trunci (B2).
    hid_a: UInt(17) = 1 if exp_a != 0 else 0
    hid_b: UInt(17) = 1 if exp_b != 0 else 0
    fa17: UInt(17) = frac_a
    fb17: UInt(17) = frac_b
    mant_a: UInt(17) = (hid_a << 16) | (fa17 << 9)
    mant_b: UInt(17) = (hid_b << 16) | (fb17 << 9)
    same_sign: uint1 = sign_a == sign_b
    sign_large: uint1 = 0
    exp_large: UInt(9) = 0
    exp_small: UInt(9) = 0
    exp_result: UInt(9) = 0
    exp_diff: UInt(9) = 0
    mant_large: UInt(17) = 0
    mant_small: UInt(17) = 0
    small_aligned: UInt(17) = 0
    magnitude: UInt(18) = 0
    align_shift: UInt(4) = 0
    leading_zeros: UInt(5) = 0
    normalize_shift: UInt(5) = 0
    max_normalize_shift: UInt(5) = 0
    guard_bit: uint1 = 0
    round_bit: uint1 = 0
    sticky_bit: uint1 = 0
    round_up: uint1 = 0
    rounded: UInt(8) = 0
    result_o: uint16 = 0

    # {exp_a, mant_a} is 25 bits; one spare zero bit on top because
    # Allo lowers UInt >= UInt to a *signed* cmpi (findings B1).
    ea26: UInt(26) = exp_a  # A8
    eb26: UInt(26) = exp_b
    ma26: UInt(26) = mant_a
    mb26: UInt(26) = mant_b
    key_a: UInt(26) = (ea26 << 17) | ma26
    key_b: UInt(26) = (eb26 << 17) | mb26
    a_is_large: uint1 = key_a >= key_b
    if a_is_large:
        sign_large = sign_a
        exp_large = 1 if exp_a == 0 else exp_a
        exp_small = 1 if exp_b == 0 else exp_b
        mant_large = mant_a
        mant_small = mant_b
    else:
        sign_large = sign_b
        exp_large = 1 if exp_b == 0 else exp_b
        exp_small = 1 if exp_a == 0 else exp_a
        mant_large = mant_b
        mant_small = mant_a

    exp_diff = exp_large - exp_small
    align_shift = 10 if exp_diff >= 10 else exp_diff[0:4]
    small_aligned = mant_small >> align_shift
    magnitude = (
        (mant_large + small_aligned)
        if same_sign
        else (mant_large - small_aligned)
    )
    exp_result = exp_large

    # A5: `x or y or z` keeps only x and y (build_BoolOp); nest them.
    sa16: uint16 = sign_a  # A8
    sb16: uint16 = sign_b
    sl16: uint16 = sign_large
    if (
        ((exp_a == 0xFF and frac_a != 0)
         or (exp_b == 0xFF and frac_b != 0))
        or (exp_a == 0xFF and (exp_b == 0xFF and sign_a != sign_b))
    ):
        result_o = 0x7FC0
    elif exp_a == 0xFF:
        result_o = (sa16 << 15) | 0x7F80  # A8
    elif exp_b == 0xFF:
        result_o = (sb16 << 15) | 0x7F80  # A8
    elif a_i[0:15] == 0:
        result_o = b_i
    elif b_i[0:15] == 0:
        result_o = a_i
    elif exp_diff >= 10:
        result_o = a_i if a_is_large else b_i
    elif magnitude == 0:
        result_o = 0
    else:
        if magnitude[17:18]:  # A2: x[k] -> x[k:k+1]
            magnitude = magnitude | ((magnitude & 1) << 1)  # A8
            magnitude >>= 1
            exp_result += 1
        elif magnitude[16:17] == 0:  # A1
            lzv: UInt(17) = magnitude[0:17]  # A4: inlined leading_zeros17
            lz: UInt(5) = 17
            # A7: no `found` flag; a second loop-carried scalar is
            # mis-wired at the loop's exit (lz == 17 means not found yet).
            # loop over offset unrolled by hand (a loop inside `if` is not in
            # s.get_loops() bands, so s.unroll cannot name it)
            if lz == 17 and ((lzv >> 16) & 1) == 1:
                lz = 0
            if lz == 17 and ((lzv >> 15) & 1) == 1:
                lz = 1
            if lz == 17 and ((lzv >> 14) & 1) == 1:
                lz = 2
            if lz == 17 and ((lzv >> 13) & 1) == 1:
                lz = 3
            if lz == 17 and ((lzv >> 12) & 1) == 1:
                lz = 4
            if lz == 17 and ((lzv >> 11) & 1) == 1:
                lz = 5
            if lz == 17 and ((lzv >> 10) & 1) == 1:
                lz = 6
            if lz == 17 and ((lzv >> 9) & 1) == 1:
                lz = 7
            if lz == 17 and ((lzv >> 8) & 1) == 1:
                lz = 8
            if lz == 17 and ((lzv >> 7) & 1) == 1:
                lz = 9
            if lz == 17 and ((lzv >> 6) & 1) == 1:
                lz = 10
            if lz == 17 and ((lzv >> 5) & 1) == 1:
                lz = 11
            if lz == 17 and ((lzv >> 4) & 1) == 1:
                lz = 12
            if lz == 17 and ((lzv >> 3) & 1) == 1:
                lz = 13
            if lz == 17 and ((lzv >> 2) & 1) == 1:
                lz = 14
            if lz == 17 and ((lzv >> 1) & 1) == 1:
                lz = 15
            if lz == 17 and ((lzv >> 0) & 1) == 1:
                lz = 16
            leading_zeros = lz
            max_normalize_shift = 16 if exp_result > 17 else exp_result - 1
            # Spare top bit again (B1): 17 is negative as a signed i5.
            lz6: UInt(6) = leading_zeros
            max6: UInt(6) = max_normalize_shift
            normalize_shift = (
                leading_zeros if lz6 < max6 else max_normalize_shift
            )
            magnitude <<= normalize_shift
            exp_result -= normalize_shift

        guard_bit = magnitude[8:9]  # A2: x[k] -> x[k:k+1]
        round_bit = magnitude[7:8]  # A2: x[k] -> x[k:k+1]
        sticky_bit = magnitude[0:7] != 0
        round_up = guard_bit & (round_bit | sticky_bit | magnitude[9:10])  # A2: x[k] -> x[k:k+1]
        rounded = magnitude[9:16] + round_up
        if rounded[7:8]:  # A2: x[k] -> x[k:k+1]
            rounded = 0
            exp_result += 1

        if exp_result >= 255:
            result_o = (sl16 << 15) | 0x7F80  # A8
        elif exp_result <= 1 and magnitude[16:17] == 0:  # A1
            r16: uint16 = rounded[0:7]  # A8
            result_o = (sl16 << 15) | r16
        else:
            e16: uint16 = exp_result[0:8]  # A8
            r16b: uint16 = rounded[0:7]
            result_o = (sl16 << 15) | (e16 << 7) | r16b
    return result_o
