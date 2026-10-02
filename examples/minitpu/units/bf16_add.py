# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1 pilot: ``vpu_bf16_add``, a combinational BF16 adder.

Two Allo expressions of the same unit, because they probe different things:

``native``
    ``a + b`` on Allo's ``bfloat16``. Probes the type system's numerics: what
    the simulator, the SystemC emitter and Catapult's ``ac_std_float`` each
    make of a bf16 add.
``bits``
    The RTL's own algorithm on ``uint16`` fields (align, add, normalize,
    round-to-nearest-even). Probes whether Allo can say what the hardware does
    when the type system's numerics are not the ones wanted.

Each build maps the unit over ``n`` vectors in one kernel; ``n`` is fixed per
build because Allo's array shapes are static.
"""

import ml_dtypes
import numpy as np

import allo.dataflow as df
from allo.ir.types import UInt, bfloat16, uint1, uint16

from examples.minitpu.harness import rtl

RTL = rtl.RtlUnit(
    top="vpu_bf16_add",
    sources=["src/core/vpu/vpu_bf16_add.sv"],
    inputs=[("a_i", 16), ("b_i", 16)],
    outputs=[("result_o", 16)],
    shape="comb",
    latency=0,
)


def native(n):
    @df.region()
    def top(A: bfloat16[n], B: bfloat16[n], C: bfloat16[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def add(a: bfloat16[n], b: bfloat16[n], c: bfloat16[n]):
            for i in range(n):
                c[i] = a[i] + b[i]

    return top


def run_native(mod, stim):
    """``stim``: uint16[n, 2] bit patterns. Returns uint16[n] bit patterns."""
    a = np.ascontiguousarray(stim[:, 0]).view(ml_dtypes.bfloat16)
    b = np.ascontiguousarray(stim[:, 1]).view(ml_dtypes.bfloat16)
    c = np.zeros(len(stim), dtype=ml_dtypes.bfloat16)
    mod(a, b, c)
    return c.view(np.uint16)


def bits(n):
    # Line-for-line vpu_bf16_add.sv. SV ``x[hi:lo]`` is Allo ``x[lo:hi+1]``.
    @df.region()
    def top(A: uint16[n], B: uint16[n], C: uint16[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def add(av: uint16[n], bv: uint16[n], cv: uint16[n]):
            def leading_zeros17(value: UInt(17)) -> UInt(5):
                lz: UInt(5) = 17
                found: uint1 = 0
                for offset in range(17):
                    if not found and value[16 - offset]:
                        lz = offset
                        found = 1
                return lz

            for i in range(n):
                a_i: uint16 = av[i]
                b_i: uint16 = bv[i]
                sign_a: uint1 = a_i[15]
                sign_b: uint1 = b_i[15]
                exp_a: UInt(8) = a_i[7:15]
                exp_b: UInt(8) = b_i[7:15]
                frac_a: UInt(7) = a_i[0:7]
                frac_b: UInt(7) = b_i[0:7]
                mant_a: UInt(17) = 0
                mant_a[16] = exp_a != 0
                mant_a[9:16] = frac_a
                mant_b: UInt(17) = 0
                mant_b[16] = exp_b != 0
                mant_b[9:16] = frac_b
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
                key_a: UInt(26) = 0
                key_a[0:17] = mant_a
                key_a[17:25] = exp_a
                key_b: UInt(26) = 0
                key_b[0:17] = mant_b
                key_b[17:25] = exp_b
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

                if (
                    (exp_a == 0xFF and frac_a != 0)
                    or (exp_b == 0xFF and frac_b != 0)
                    or (exp_a == 0xFF and exp_b == 0xFF and sign_a != sign_b)
                ):
                    result_o = 0x7FC0
                elif exp_a == 0xFF:
                    result_o[15] = sign_a
                    result_o[7:15] = 0xFF
                elif exp_b == 0xFF:
                    result_o[15] = sign_b
                    result_o[7:15] = 0xFF
                elif a_i[0:15] == 0:
                    result_o = b_i
                elif b_i[0:15] == 0:
                    result_o = a_i
                elif exp_diff >= 10:
                    result_o = a_i if a_is_large else b_i
                elif magnitude == 0:
                    result_o = 0
                else:
                    if magnitude[17]:
                        magnitude[1] = magnitude[1] | magnitude[0]
                        magnitude >>= 1
                        exp_result += 1
                    elif not magnitude[16]:
                        leading_zeros = leading_zeros17(magnitude[0:17])
                        max_normalize_shift = 16 if exp_result > 17 else exp_result - 1
                        # Spare top bit again (B1): 17 is negative as a signed i5.
                        lz6: UInt(6) = leading_zeros
                        max6: UInt(6) = max_normalize_shift
                        normalize_shift = (
                            leading_zeros if lz6 < max6 else max_normalize_shift
                        )
                        magnitude <<= normalize_shift
                        exp_result -= normalize_shift

                    guard_bit = magnitude[8]
                    round_bit = magnitude[7]
                    sticky_bit = magnitude[0:7] != 0
                    round_up = guard_bit & (round_bit | sticky_bit | magnitude[9])
                    rounded = magnitude[9:16] + round_up
                    if rounded[7]:
                        rounded = 0
                        exp_result += 1

                    if exp_result >= 255:
                        result_o[15] = sign_large
                        result_o[7:15] = 0xFF
                    elif exp_result <= 1 and not magnitude[16]:
                        result_o[15] = sign_large
                        result_o[0:7] = rounded[0:7]
                    else:
                        result_o[15] = sign_large
                        result_o[7:15] = exp_result[0:8]
                        result_o[0:7] = rounded[0:7]
                cv[i] = result_o

    return top


def run_bits(mod, stim):
    """As ``run_native``, on the raw bit patterns."""
    a = np.ascontiguousarray(stim[:, 0]).astype(np.uint16)
    b = np.ascontiguousarray(stim[:, 1]).astype(np.uint16)
    c = np.zeros(len(stim), dtype=np.uint16)
    mod(a, b, c)
    return c


VARIANTS = {"native": (native, run_native), "bits": (bits, run_bits)}


def _is_nan(x):
    return (int(x) & 0x7F80) == 0x7F80 and (int(x) & 0x7F) != 0


# Rules that name a difference between an Allo result and the RTL's, first
# match wins. Each is a measured RTL deviation from IEEE (harness/ref.py).
EXPLAIN = [
    ("NaN sign: allo keeps it, rtl always +0x7fc0",
     lambda s, g, w: _is_nan(g) and int(w) == 0x7FC0),
    ("NaN payload/canonical form differs",
     lambda s, g, w: _is_nan(g) and _is_nan(w)),
    ("(+0)+(-0): allo +0 (IEEE), rtl -0",
     lambda s, g, w: (int(s[0]) & 0x7FFF) == 0 and (int(s[1]) & 0x7FFF) == 0),
    ("subnormal result flushed/changed",
     lambda s, g, w: (int(w) & 0x7F80) == 0 or (int(g) & 0x7F80) == 0),
    ("subnormal operand",
     lambda s, g, w: (int(s[0]) & 0x7F80) == 0 or (int(s[1]) & 0x7F80) == 0),
    ("off by one ulp (rounding)",
     lambda s, g, w: abs(int(g) - int(w)) == 1),
]
