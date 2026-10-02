# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``vpu_bf16_mul``, the combinational BF16 multiplier (vmul's twin).

``src/core/vpu/vpu_bf16_mul.sv`` holds two modules: this combinational one and
``vpu_bf16_mul_pipe`` (``bf16_mul_pipe.py``), the two-stage form ``vpu_alu``
instantiates. Both flush subnormals, on input and on output.
"""

import ml_dtypes
import numpy as np

import allo.dataflow as df
from allo.ir.types import Int, UInt, bfloat16, uint1, uint16

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness import stimulus as stim

RTL = rtl.RtlUnit(
    top="vpu_bf16_mul",
    sources=["src/core/vpu/vpu_bf16_mul.sv"],
    inputs=[("a_i", 16), ("b_i", 16)],
    outputs=[("result_o", 16)],
    shape="comb",
    latency=0,
)
LATENCY_SOURCE = "combinational (always_comb, no clock port)"

REF = ref.vpu_bf16_mul
IEEE = ref.ieee_bf16_mul
PROBE = None


def stimulus():
    """Corners crossed, ties, random, then every ``a`` against every corner."""
    return np.concatenate([stim.binary_bf16(), stim.bf16_sweep_a()])


def native(n):
    """``a * b`` on Allo's ``bfloat16``: IEEE RNE with gradual underflow."""

    @df.region()
    def top(A: bfloat16[n], B: bfloat16[n], C: bfloat16[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def mul(a: bfloat16[n], b: bfloat16[n], c: bfloat16[n]):
            for i in range(n):
                c[i] = a[i] * b[i]

    return top


def run_native(mod, stim_):
    """``stim_``: uint16[n, 2] bit patterns. Returns uint16[n] bit patterns."""
    a = np.ascontiguousarray(stim_[:, 0]).view(ml_dtypes.bfloat16)
    b = np.ascontiguousarray(stim_[:, 1]).view(ml_dtypes.bfloat16)
    c = np.zeros(len(stim_), dtype=ml_dtypes.bfloat16)
    mod(a, b, c)
    return c.view(np.uint16)


def bits(n):
    # Line-for-line ``vpu_bf16_mul`` (the always_comb module). SV ``x[hi:lo]``
    # is Allo ``x[lo:hi+1]``; ``{...}`` concatenations are slice stores (M1).
    @df.region()
    def top(A: uint16[n], B: uint16[n], C: uint16[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def mul(av: uint16[n], bv: uint16[n], cv: uint16[n]):
            for i in range(n):
                a_i: uint16 = av[i]
                b_i: uint16 = bv[i]
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

                exp_r: Int(11) = 0
                frac: UInt(7) = 0
                guard: uint1 = 0
                sticky: uint1 = 0
                if product[15]:
                    exp_r = exp_a + exp_b - 126
                    frac = product[8:15]
                    guard = product[7]
                    sticky = product[0:7] != 0
                else:
                    exp_r = exp_a + exp_b - 127
                    frac = product[7:14]
                    guard = product[6]
                    sticky = product[0:6] != 0

                round_up: uint1 = guard & (sticky | frac[0])
                rounded: UInt(8) = frac + round_up
                frac_final: UInt(7) = 0 if rounded[7] else rounded[0:7]
                exp_final: Int(11) = (exp_r + 1) if rounded[7] else exp_r

                result_o: uint16 = 0
                if (
                    (exp_a == 0xFF and frac_a != 0)
                    or (exp_b == 0xFF and frac_b != 0)
                    or (exp_a == 0xFF and b_i[0:15] == 0)
                    or (exp_b == 0xFF and a_i[0:15] == 0)
                ):
                    result_o = 0x7FC0
                elif exp_a == 0xFF or exp_b == 0xFF:
                    result_o[15] = sign
                    result_o[7:15] = 0xFF
                elif exp_a == 0 or exp_b == 0:
                    result_o[15] = sign
                elif exp_r >= 255:
                    result_o[15] = sign
                    result_o[7:15] = 0xFF
                elif exp_r <= 0:
                    result_o[15] = sign
                elif exp_final >= 255:
                    result_o[15] = sign
                    result_o[7:15] = 0xFF
                else:
                    result_o[15] = sign
                    result_o[7:15] = exp_final[0:8]
                    result_o[0:7] = frac_final
                cv[i] = result_o

    return top


def run_bits(mod, stim_):
    """As ``run_native``, on the raw bit patterns."""
    a = np.ascontiguousarray(stim_[:, 0]).astype(np.uint16)
    b = np.ascontiguousarray(stim_[:, 1]).astype(np.uint16)
    c = np.zeros(len(stim_), dtype=np.uint16)
    mod(a, b, c)
    return c


VARIANTS = {"native": (native, run_native), "bits": (bits, run_bits)}


def _sub_in(s):
    return ref.exp_field(s[0]) == 0 or ref.exp_field(s[1]) == 0


DEVIATIONS = [
    ("NaN result is always +0x7fc0 (IEEE keeps a sign)",
     lambda s, g, w: ref.is_nan(g) and int(w) == 0x7FC0),
    ("subnormal operand flushed: rtl signed 0, IEEE a normal product",
     lambda s, g, w: _sub_in(s) and ref.is_zero(w) and ref.exp_field(g) != 0),
    ("subnormal operand flushed: rtl signed 0, IEEE a subnormal product",
     lambda s, g, w: _sub_in(s) and ref.is_zero(w)),
    ("product < 2^-126 flushed: rtl signed 0, IEEE rounds up to 2^-126",
     lambda s, g, w: ref.is_zero(w) and int(g) & 0x7FFF == 0x0080),
    ("product < 2^-126 flushed: rtl signed 0, IEEE a subnormal",
     lambda s, g, w: ref.is_zero(w) and ref.exp_field(g) == 0),
]
# Allo-side NaN encodings first (as in bf16_add), then the RTL's deviations.
def _any_nan_in(s):
    return ref.is_nan(s[0]) or ref.is_nan(s[1])


EXPLAIN = [
    ("NaN payload: ac_std_float all-ones 0x7fff/0xffff, rtl +0x7fc0",
     lambda s, g, w: (int(g) & 0x7FFF) == 0x7FFF and int(w) == 0x7FC0
     and not _any_nan_in(s)),
    # The simulator gives Inf x 0 the x86 default NaN (-0x7fc0); SystemC's
    # ac::bfloat16 gives every NaN the sign a^b.
    ("Inf x 0: allo -0x7fc0, rtl +0x7fc0",
     lambda s, g, w: ref.is_nan(g) and int(w) == 0x7FC0 and not _any_nan_in(s)),
    # Simulator: the NaN operand's sign; ac::bfloat16: a^b. Payload is 0x40.
    ("NaN operand: allo -0x7fc0, rtl +0x7fc0",
     lambda s, g, w: ref.is_nan(g) and int(w) == 0x7FC0),
] + DEVIATIONS + [
    ("off by one ulp (rounding)", lambda s, g, w: abs(int(g) - int(w)) == 1),
]
