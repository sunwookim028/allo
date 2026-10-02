# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``vpu_bf16_mul_pipe``, the two-stage BF16 multiplier ``vpu_alu`` uses.

It has a clock and nothing else: no reset, no valid (the ALU's valid rides
``vpu_bf16_add_pipe``), so the harness drives it ``bare``. Two register banks
(``*_s1_q`` and ``result_o``) give latency 2; MiniTPU states no
``localparam`` for it alone. Same numerics as ``vpu_bf16_mul``.

Allo variants. The function is ``vpu_bf16_mul``'s, so ``native`` and ``bits``
are that unit's; what is new here is the pipe's own text and its latency:

``bits_pipe``
    The pipe's ``always_comb`` stage 2, line for line. It is not the comb
    module's text: 9-bit *unsigned* exponent arithmetic with a selected bias
    (``exp_sum_s2 <= exp_bias_s2``), which is where B1 bites.
``stages``
    The same text split at the RTL's register boundary: kernel ``s1`` (the
    DSP product and the ``*_s1_q`` fields) feeds kernel ``s2`` through one
    ``Stream`` per register (depth 1 by default). The closest Allo comes to stating "two
    stages". The simulator is untimed and csim's timing is the threads' and
    buffers' (a depth-1 Stream halves throughput there), so the split is
    checked for function only; latency 2 is not a property of the Allo
    program (``u1_mul_2026-10-02.rst``, L1-L3).
"""

import allo.dataflow as df
from allo.ir.types import Stream, UInt, uint1, uint16

from examples.minitpu.harness import rtl
from examples.minitpu.units import bf16_mul

RTL = rtl.RtlUnit(
    top="vpu_bf16_mul_pipe",
    sources=["src/core/vpu/vpu_bf16_mul.sv"],
    inputs=[("a_i", 16), ("b_i", 16)],
    outputs=[("result_o", 16)],
    shape="bare",
    latency=2,
)
LATENCY_SOURCE = "vpu_bf16_mul.sv comment (\"Two-stage form\"); no localparam"

REF = bf16_mul.REF
IEEE = bf16_mul.IEEE
PROBE = ((0x3F80, 0x3F80), (0x4000, 0x3F80))
stimulus = bf16_mul.stimulus


def mul_bits(a_i: uint16, b_i: uint16) -> uint16:
    """``vpu_bf16_mul_pipe`` (the form ``vpu_alu`` instantiates), one pair.

    Its 9-bit unsigned compares are held in 10 bits: Allo lowers unsigned
    ``>=``/``<=`` to a signed ``cmpi`` (B1), and 9-bit sums reach 510.
    """
    sign: uint1 = a_i[15] ^ b_i[15]
    exp_a: UInt(8) = a_i[7:15]
    exp_b: UInt(8) = b_i[7:15]
    frac_a: UInt(7) = a_i[0:7]
    frac_b: UInt(7) = b_i[0:7]
    # {1'b1, frac} * {1'b1, frac} at 16 bits: operands widened first, since
    # an expression's width is sized from its operands (M4).
    mant_a: UInt(16) = 0
    mant_a[0:7] = frac_a
    mant_a[7] = 1
    mant_b: UInt(16) = 0
    mant_b[0:7] = frac_b
    mant_b[7] = 1
    product: UInt(16) = mant_a * mant_b

    ea: UInt(10) = exp_a
    eb: UInt(10) = exp_b
    exp_sum: UInt(10) = ea + eb
    exp_bias: UInt(10) = 0
    frac: UInt(7) = 0
    guard: uint1 = 0
    sticky: uint1 = 0
    if product[15]:
        exp_bias = 126
        frac = product[8:15]
        guard = product[7]
        sticky = product[0:7] != 0
    else:
        exp_bias = 127
        frac = product[7:14]
        guard = product[6]
        sticky = product[0:6] != 0
    exp_s: UInt(9) = exp_sum[0:9] - exp_bias[0:9]
    round_up: uint1 = guard & (sticky | frac[0])
    frac8: UInt(8) = frac
    rounded: UInt(8) = frac8 + round_up
    frac_final: UInt(7) = 0 if rounded[7] else rounded[0:7]
    exp_final: UInt(9) = exp_s + rounded[7]
    exp_final10: UInt(10) = exp_final
    bias_inf: UInt(10) = exp_bias + 255

    result: uint16 = 0
    if (
        (exp_a == 0xFF and frac_a != 0)
        or (exp_b == 0xFF and frac_b != 0)
        or (exp_a == 0xFF and exp_b == 0 and frac_b == 0)
        or (exp_b == 0xFF and exp_a == 0 and frac_a == 0)
    ):
        result = 0x7FC0
    elif exp_a == 0xFF or exp_b == 0xFF:
        result[15] = sign
        result[7:15] = 0xFF
    elif exp_a == 0 or exp_b == 0:
        result[15] = sign
    elif exp_sum >= bias_inf:
        result[15] = sign
        result[7:15] = 0xFF
    elif exp_sum <= exp_bias:
        result[15] = sign
    elif exp_final10 >= 255:
        result[15] = sign
        result[7:15] = 0xFF
    else:
        result[15] = sign
        result[7:15] = exp_final[0:8]
        result[0:7] = frac_final
    return result

def fn(n):
    """The unit as one reusable function, ``mul_bits``, called per pair.

    ``vpu_alu`` imports the same function, so the pipe and the ALU check one
    implementation against two RTL units.
    """

    @df.region()
    def top(A: uint16[n], B: uint16[n], C: uint16[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def mul(a: uint16[n], b: uint16[n], c: uint16[n]):
            for i in range(n):
                c[i] = mul_bits(a[i], b[i])

    return top


def bits_pipe(n):
    # vpu_bf16_mul_pipe's two always blocks, flattened into one iteration.
    @df.region()
    def top(A: uint16[n], B: uint16[n], C: uint16[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def mul(av: uint16[n], bv: uint16[n], cv: uint16[n]):
            for i in range(n):
                a_i: uint16 = av[i]
                b_i: uint16 = bv[i]
                # stage 1 (registered in the RTL)
                mant_a: UInt(8) = 0
                mant_a[7] = 1
                mant_a[0:7] = a_i[0:7]
                mant_b: UInt(8) = 0
                mant_b[7] = 1
                mant_b[0:7] = b_i[0:7]
                product_s1_q: UInt(16) = mant_a * mant_b
                sign_s1_q: uint1 = a_i[15] ^ b_i[15]
                exp_a_s1_q: UInt(8) = a_i[7:15]
                exp_b_s1_q: UInt(8) = b_i[7:15]
                frac_a_s1_q: UInt(7) = a_i[0:7]
                frac_b_s1_q: UInt(7) = b_i[0:7]
                # stage 2
                exp_sum_s2: UInt(9) = exp_a_s1_q + exp_b_s1_q
                exp_bias_s2: UInt(9) = 0
                frac_s2: UInt(7) = 0
                guard_s2: uint1 = 0
                sticky_s2: uint1 = 0
                if product_s1_q[15]:
                    exp_bias_s2 = 126
                    frac_s2 = product_s1_q[8:15]
                    guard_s2 = product_s1_q[7]
                    sticky_s2 = product_s1_q[0:7] != 0
                else:
                    exp_bias_s2 = 127
                    frac_s2 = product_s1_q[7:14]
                    guard_s2 = product_s1_q[6]
                    sticky_s2 = product_s1_q[0:6] != 0
                exp_s2: UInt(9) = exp_sum_s2 - exp_bias_s2
                round_up_s2: uint1 = guard_s2 & (sticky_s2 | frac_s2[0])
                rounded_s2: UInt(8) = frac_s2 + round_up_s2
                frac_final_s2: UInt(7) = 0 if rounded_s2[7] else rounded_s2[0:7]
                exp_final_s2: UInt(9) = exp_s2 + rounded_s2[7]
                # SV sizes ``exp_bias_s2 + 9'd255`` to 9 bits; it cannot wrap
                # (<= 382), so Allo's wider sum is the same value (M4).
                # Spare top bit (B1): exp_sum_s2 and exp_bias_s2 as UInt(9)
                # compare signed, and exp_sum_s2 >= 256 reads negative.
                sum10: UInt(10) = exp_sum_s2
                bias10: UInt(10) = exp_bias_s2

                result_s2: uint16 = 0
                if (
                    (exp_a_s1_q == 0xFF and frac_a_s1_q != 0)
                    or (exp_b_s1_q == 0xFF and frac_b_s1_q != 0)
                    or (exp_a_s1_q == 0xFF and exp_b_s1_q == 0 and frac_b_s1_q == 0)
                    or (exp_b_s1_q == 0xFF and exp_a_s1_q == 0 and frac_a_s1_q == 0)
                ):
                    result_s2 = 0x7FC0
                elif exp_a_s1_q == 0xFF or exp_b_s1_q == 0xFF:
                    result_s2[15] = sign_s1_q
                    result_s2[7:15] = 0xFF
                elif exp_a_s1_q == 0 or exp_b_s1_q == 0:
                    result_s2[15] = sign_s1_q
                elif exp_sum_s2 >= exp_bias_s2 + 255:
                    result_s2[15] = sign_s1_q
                    result_s2[7:15] = 0xFF
                elif sum10 <= bias10:
                    result_s2[15] = sign_s1_q
                elif exp_final_s2 >= 255:
                    result_s2[15] = sign_s1_q
                    result_s2[7:15] = 0xFF
                else:
                    result_s2[15] = sign_s1_q
                    result_s2[7:15] = exp_final_s2[0:8]
                    result_s2[0:7] = frac_final_s2
                cv[i] = result_s2

    return top


def stages(n, depth=1):
    # The pipe split at its register boundary: one Stream per ``*_s1_q``
    # register (depth 1, like a register), kernel s1 = stage 1, s2 = stage 2.
    @df.region()
    def top(A: uint16[n], B: uint16[n], C: uint16[n]):
        sign_q: Stream[uint1, depth]
        exp_a_q: Stream[UInt(8), depth]
        exp_b_q: Stream[UInt(8), depth]
        frac_a_q: Stream[UInt(7), depth]
        frac_b_q: Stream[UInt(7), depth]
        product_q: Stream[UInt(16), depth]

        @df.kernel(mapping=[1], args=[A, B])
        def s1(av: uint16[n], bv: uint16[n]):
            for i in range(n):
                a_i: uint16 = av[i]
                b_i: uint16 = bv[i]
                mant_a: UInt(8) = 0
                mant_a[7] = 1
                mant_a[0:7] = a_i[0:7]
                mant_b: UInt(8) = 0
                mant_b[7] = 1
                mant_b[0:7] = b_i[0:7]
                product_s1: UInt(16) = mant_a * mant_b
                sign_q.put(a_i[15] ^ b_i[15])
                exp_a_q.put(a_i[7:15])
                exp_b_q.put(b_i[7:15])
                frac_a_q.put(a_i[0:7])
                frac_b_q.put(b_i[0:7])
                product_q.put(product_s1)

        @df.kernel(mapping=[1], args=[C])
        def s2(cv: uint16[n]):
            for i in range(n):
                sign_s1_q: uint1 = sign_q.get()
                exp_a_s1_q: UInt(8) = exp_a_q.get()
                exp_b_s1_q: UInt(8) = exp_b_q.get()
                frac_a_s1_q: UInt(7) = frac_a_q.get()
                frac_b_s1_q: UInt(7) = frac_b_q.get()
                product_s1_q: UInt(16) = product_q.get()
                exp_sum_s2: UInt(9) = exp_a_s1_q + exp_b_s1_q
                exp_bias_s2: UInt(9) = 0
                frac_s2: UInt(7) = 0
                guard_s2: uint1 = 0
                sticky_s2: uint1 = 0
                if product_s1_q[15]:
                    exp_bias_s2 = 126
                    frac_s2 = product_s1_q[8:15]
                    guard_s2 = product_s1_q[7]
                    sticky_s2 = product_s1_q[0:7] != 0
                else:
                    exp_bias_s2 = 127
                    frac_s2 = product_s1_q[7:14]
                    guard_s2 = product_s1_q[6]
                    sticky_s2 = product_s1_q[0:6] != 0
                exp_s2: UInt(9) = exp_sum_s2 - exp_bias_s2
                round_up_s2: uint1 = guard_s2 & (sticky_s2 | frac_s2[0])
                rounded_s2: UInt(8) = frac_s2 + round_up_s2
                frac_final_s2: UInt(7) = 0 if rounded_s2[7] else rounded_s2[0:7]
                exp_final_s2: UInt(9) = exp_s2 + rounded_s2[7]
                sum10: UInt(10) = exp_sum_s2  # spare top bit (B1)
                bias10: UInt(10) = exp_bias_s2

                result_s2: uint16 = 0
                if (
                    (exp_a_s1_q == 0xFF and frac_a_s1_q != 0)
                    or (exp_b_s1_q == 0xFF and frac_b_s1_q != 0)
                    or (exp_a_s1_q == 0xFF and exp_b_s1_q == 0 and frac_b_s1_q == 0)
                    or (exp_b_s1_q == 0xFF and exp_a_s1_q == 0 and frac_a_s1_q == 0)
                ):
                    result_s2 = 0x7FC0
                elif exp_a_s1_q == 0xFF or exp_b_s1_q == 0xFF:
                    result_s2[15] = sign_s1_q
                    result_s2[7:15] = 0xFF
                elif exp_a_s1_q == 0 or exp_b_s1_q == 0:
                    result_s2[15] = sign_s1_q
                elif exp_sum_s2 >= exp_bias_s2 + 255:
                    result_s2[15] = sign_s1_q
                    result_s2[7:15] = 0xFF
                elif sum10 <= bias10:
                    result_s2[15] = sign_s1_q
                elif exp_final_s2 >= 255:
                    result_s2[15] = sign_s1_q
                    result_s2[7:15] = 0xFF
                else:
                    result_s2[15] = sign_s1_q
                    result_s2[7:15] = exp_final_s2[0:8]
                    result_s2[0:7] = frac_final_s2
                cv[i] = result_s2

    return top


VARIANTS = {
    "native": bf16_mul.VARIANTS["native"],
    "bits": bf16_mul.VARIANTS["bits"],
    "bits_pipe": (bits_pipe, bf16_mul.run_bits),
    "stages": (stages, bf16_mul.run_bits),
    "fn": (fn, bf16_mul.run_bits),
}
DEVIATIONS = bf16_mul.DEVIATIONS
EXPLAIN = bf16_mul.EXPLAIN
