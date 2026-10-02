# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``mxu_acc24_add_pipe``, the array's three-stage acc24 adder.

acc24 is 1+8+15 with subnormals (``vpu_pkg::MXU_ACC_FRAC_W = 15``). Latency
is ``vpu_pkg::MXU_ACC_ADD_LATENCY = 3`` ("mxu_acc24_add_pipe's stages").
Built with the package default ``MXU_ACC_USE_DSP = 0`` (fabric shifters);
MiniTPU's ``tb_acc24_add_dsp_equiv`` holds the DSP profile bit-identical.

Three Allo expressions. Allo has no acc24 type, so every one carries acc24 as
its bit pattern in a ``uint32`` (top 8 bits zero):

``native``
    The workaround ``microarch.py``'s PE uses: widen to ``float32`` by a shift
    and a bitcast, add in ``float32``, round back to 15 fraction bits by the
    RTL's add-half-plus-lsb rule. It rounds **twice** (float32, then acc24),
    which is README divergence 5; the harness counts it.
``bits``
    ``mxu_acc24_add_pipe.sv`` line for line, its three stages in sequence in
    one kernel (the fabric profile, ``USE_DSP = 0``).
``staged``
    The same three stages as three kernels joined by ``Stream``\\ s, each
    stream carrying exactly the RTL's stage register bank (special class,
    sign, 20-bit magnitude, 9-bit exponent: 32 bits). The only way Allo can
    write "three register stages" down -- and what it does and does not
    mean for latency is ``dev/records/minitpu/u1_pipe_2026-10-02.rst``.

SV ``x[hi:lo]`` is Allo ``x[lo:hi+1]``. Every unsigned ``<``/``>=`` whose top
bit can be set gets a spare zero bit on top, because Allo lowers unsigned
compares as signed (B1; ``bf16_add.py``).
"""

import numpy as np

import allo.dataflow as df
from allo.ir.types import Stream, UInt, float32, uint1, uint32

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness import stimulus as stim

RTL = rtl.RtlUnit(
    top="mxu_acc24_add_pipe",
    sources=["src/core/vpu/vpu_pkg.sv", "src/core/mxu/mxu_acc24_add_pipe.sv"],
    inputs=[("a_i", 24), ("b_i", 24)],
    outputs=[("result_o", 24)],
    shape="valid",
    latency=3,
)
LATENCY_SOURCE = "vpu_pkg.sv localparam MXU_ACC_ADD_LATENCY = 3"

REF = ref.mxu_acc24_add
IEEE = ref.ieee_acc24_add
PROBE = ((0x3F8000, 0x3F8000), (0x400000, 0x3F8000))


def stimulus():
    return stim.binary_acc24()


def native(n):
    @df.region()
    def top(A: uint32[n], B: uint32[n], C: uint32[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def add(av: uint32[n], bv: uint32[n], cv: uint32[n]):
            for i in range(n):
                ua: uint32 = av[i] << 8
                ub: uint32 = bv[i] << 8
                fa: float32 = ua.bitcast()
                fb: float32 = ub.bitcast()
                s: float32 = fa + fb
                u: uint32 = s.bitcast()
                lsb: uint32 = (u >> 8) & 1
                r: uint32 = u + 127 + lsb
                cv[i] = r >> 8

    return top


def bits(n):
    @df.region()
    def top(A: uint32[n], B: uint32[n], C: uint32[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def add(av: uint32[n], bv: uint32[n], cv: uint32[n]):
            def leading_zeros19(value: UInt(19)) -> UInt(5):
                lz: UInt(5) = 19
                found: uint1 = 0
                for offset in range(19):
                    if not found and value[18 - offset]:
                        lz = offset
                        found = 1
                return lz

            for i in range(n):
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
                lzc: UInt(5) = leading_zeros19(s1_mag[0:19])
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

    return top


def staged(n):
    @df.region()
    def top(A: uint32[n], B: uint32[n], C: uint32[n]):
        s12: Stream[uint32, 2]
        s23: Stream[uint32, 2]

        @df.kernel(mapping=[1], args=[A, B])
        def stage1(av: uint32[n], bv: uint32[n]):
            for i in range(n):
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

                w1: uint32 = 0  # the stage-1 register bank, packed
                w1[0:20] = s1_mag
                w1[20:29] = s1_exp
                w1[29] = s1_sign
                w1[30:32] = s1_special
                s12.put(w1)

        @df.kernel(mapping=[1], args=[])
        def stage2():
            def leading_zeros19(value: UInt(19)) -> UInt(5):
                lz: UInt(5) = 19
                found: uint1 = 0
                for offset in range(19):
                    if not found and value[18 - offset]:
                        lz = offset
                        found = 1
                return lz

            for i in range(n):
                w1: uint32 = s12.get()
                s1_mag: UInt(20) = w1[0:20]
                s1_exp: UInt(9) = w1[20:29]
                s1_sign: uint1 = w1[29]
                s1_special: UInt(2) = w1[30:32]
                # ---- stage 2: normalize ----
                norm_overflow: uint1 = s1_mag[19]
                norm_needed: uint1 = 0
                if s1_mag != 0 and not s1_mag[18] and not norm_overflow:
                    norm_needed = 1
                lzc: UInt(5) = leading_zeros19(s1_mag[0:19])
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
                w2: uint32 = 0  # the stage-2 register bank, packed
                w2[0:20] = s2_mag
                w2[20:29] = s2_exp[0:9]
                w2[29] = s2_sign
                w2[30:32] = s2_special
                s23.put(w2)

        @df.kernel(mapping=[1], args=[C])
        def stage3(cv: uint32[n]):
            for i in range(n):
                w2: uint32 = s23.get()
                s2_mag: UInt(20) = w2[0:20]
                s2_exp: UInt(10) = w2[20:29]
                s2_sign: uint1 = w2[29]
                s2_special: UInt(2) = w2[30:32]
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

    return top


def run_bits(mod, stim):
    """``stim``: acc24 bit patterns, ``uint32[n, 2]``. Returns ``uint32[n]``."""
    a = np.ascontiguousarray(stim[:, 0]).astype(np.uint32)
    b = np.ascontiguousarray(stim[:, 1]).astype(np.uint32)
    c = np.zeros(len(stim), dtype=np.uint32)
    mod(a, b, c)
    return c


VARIANTS = {"native": (native, run_bits), "bits": (bits, run_bits), "staged": (staged, run_bits)}

DEVIATIONS = [
    ("NaN result is always +0x7fc000 (IEEE keeps a sign)",
     lambda s, g, w: ref.is_nan(g, 15) and int(w) == ref.ACC24_NAN),
    ("(+0)+(-0) = -0 (IEEE RNE: +0)",
     lambda s, g, w: ref.is_zero(s[0], 15) and ref.is_zero(s[1], 15) and int(w) == int(s[1])),
]


def _round_once_differs(s, g, w):
    """``native``'s float32 add then acc24 round vs the RTL's round-once."""
    return abs(int(g) - int(w)) == 1 and int(w) == int(REF(s[0], s[1]))


EXPLAIN = [
    ("NaN encoding: allo keeps payload/sign, rtl always +0x7fc000",
     lambda s, g, w: ref.is_nan(g, 15) and int(w) == ref.ACC24_NAN),
    ("NaN becomes a zero: ac_ieee_float's NaN is 0x7fffffff, and native's acc24 round carries its payload into the sign",
     lambda s, g, w: int(w) == ref.ACC24_NAN and ref.is_zero(g, 15)),
    DEVIATIONS[1],
    ("double rounding: float32 add then acc24 round, one acc24 ulp off (divergence 5)",
     _round_once_differs),
    ("subnormal result (float32 path)",
     lambda s, g, w: ref.exp_field(w, 15) == 0 or ref.exp_field(g, 15) == 0),
]
