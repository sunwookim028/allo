# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``mxu_bf16_mul_acc24``, the array's multiplier: bf16 x bf16 -> acc24.

Combinational. The product is exact in acc24 (1+8+15, the MXU accumulator
format, ``vpu_pkg::MXU_ACC_W = 24``) except where it underflows or overflows,
and, like ``vpu_bf16_mul``, it flushes subnormals on input and on output --
though acc24, unlike the multiplier, has subnormals.
"""

import ml_dtypes
import numpy as np

import allo.dataflow as df
from allo.ir.types import UInt, bfloat16, float32, uint1, uint16, uint32

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness import stimulus as stim

RTL = rtl.RtlUnit(
    top="mxu_bf16_mul_acc24",
    sources=["src/core/vpu/vpu_pkg.sv", "src/core/mxu/mxu_bf16_mul_acc24.sv"],
    inputs=[("a_i", 16), ("b_i", 16)],
    outputs=[("result_o", 24)],
    shape="comb",
    latency=0,
)
LATENCY_SOURCE = ("combinational; the PE registers the product "
                  "(vpu_pkg MXU_PE_LATENCY = 1 + MXU_ACC_ADD_LATENCY)")

REF = ref.mxu_bf16_mul_acc24
IEEE = ref.ieee_bf16_mul_acc24
PROBE = None


def stimulus():
    return np.concatenate([stim.binary_bf16(), stim.bf16_sweep_a()])


def native(n):
    """No acc24 type: a float32 product, rounded RNE to 15 fraction bits by
    integer arithmetic on its bit pattern (as ``microarch.py`` and
    ``tests/dataflow/test_bf16_dataflow.py::test_acc24_emulation_via_bitcast``
    do), returned as the 24-bit pattern in a ``uint32``. A **workaround** for a
    missing abstraction (no custom float format); the rounding is a no-op
    unless the float32 product is itself subnormal."""

    @df.region()
    def top(A: bfloat16[n], B: bfloat16[n], C: uint32[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def mul(a: bfloat16[n], b: bfloat16[n], c: uint32[n]):
            for i in range(n):
                av: float32 = a[i]
                bv: float32 = b[i]
                prod: float32 = av * bv
                u: uint32 = prod.bitcast()
                lsb: uint32 = (u >> 8) & 1
                r: uint32 = u + 127 + lsb
                c[i] = r >> 8

    return top


def native_bitext(n):
    """``native`` with the two bf16 -> float32 widenings done on bit
    patterns (``bits << 16``, bitcast) instead of by assignment: works around
    the SystemC emitter's copy-initialisation of ``ac_ieee_float<binary32>``
    from ``ac::bfloat16`` (S4, ``u1_mul_2026-10-02.rst``). Same function."""

    @df.region()
    def top(A: bfloat16[n], B: bfloat16[n], C: uint32[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def mul(a: bfloat16[n], b: bfloat16[n], c: uint32[n]):
            for i in range(n):
                ab: uint16 = a[i].bitcast()
                bb: uint16 = b[i].bitcast()
                aw: uint32 = ab
                bw: uint32 = bb
                au: uint32 = aw << 16
                bu: uint32 = bw << 16
                av: float32 = au.bitcast()
                bv: float32 = bu.bitcast()
                prod: float32 = av * bv
                u: uint32 = prod.bitcast()
                lsb: uint32 = (u >> 8) & 1
                r: uint32 = u + 127 + lsb
                c[i] = r >> 8

    return top


def run_native(mod, stim_):
    """``stim_``: uint16[n, 2] bit patterns. Returns uint32[n] acc24 patterns."""
    a = np.ascontiguousarray(stim_[:, 0]).view(ml_dtypes.bfloat16)
    b = np.ascontiguousarray(stim_[:, 1]).view(ml_dtypes.bfloat16)
    c = np.zeros(len(stim_), dtype=np.uint32)
    mod(a, b, c)
    return c


def bits(n):
    # Line-for-line ``mxu_bf16_mul_acc24.sv``; MXU_ACC_W = 24, FRAC_W = 15.
    # The 24-bit result travels in a uint32 port (bits 31:24 zero).
    @df.region()
    def top(A: uint16[n], B: uint16[n], C: uint32[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def mul(av: uint16[n], bv: uint16[n], cv: uint32[n]):
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
                cv[i] = result_o

    return top


def run_bits(mod, stim_):
    a = np.ascontiguousarray(stim_[:, 0]).astype(np.uint16)
    b = np.ascontiguousarray(stim_[:, 1]).astype(np.uint16)
    c = np.zeros(len(stim_), dtype=np.uint32)
    mod(a, b, c)
    return c


VARIANTS = {
    "native": (native, run_native),
    "native_bitext": (native_bitext, run_native),
    "bits": (bits, run_bits),
}


def _sub_in(s):
    return ref.exp_field(s[0]) == 0 or ref.exp_field(s[1]) == 0


DEVIATIONS = [
    ("NaN result is always +0x7fc000 (IEEE keeps a sign)",
     lambda s, g, w: ref.is_nan(g, 15) and int(w) == ref.ACC24_NAN),
    ("subnormal operand flushed: rtl signed 0, IEEE a normal product",
     lambda s, g, w: _sub_in(s) and ref.is_zero(w, 15) and ref.exp_field(g, 15) != 0),
    ("subnormal operand flushed: rtl signed 0, IEEE an acc24 subnormal",
     lambda s, g, w: _sub_in(s) and ref.is_zero(w, 15)),
    ("product < 2^-126 flushed: rtl signed 0, IEEE rounds up to 2^-126",
     lambda s, g, w: ref.is_zero(w, 15) and int(g) & 0x7FFFFF == 0x008000),
    ("product < 2^-126 flushed: rtl signed 0, IEEE an acc24 subnormal",
     lambda s, g, w: ref.is_zero(w, 15) and ref.exp_field(g, 15) == 0),
]
def _any_nan_in(s):
    return ref.is_nan(s[0]) or ref.is_nan(s[1])


EXPLAIN = [
    ("Inf x 0: allo NaN from the float32 multiply, rtl +0x7fc000",
     lambda s, g, w: ref.is_nan(g, 15) and int(w) == ref.ACC24_NAN
     and not _any_nan_in(s)),
    # float32 keeps the NaN operand's sign and (quieted) payload.
    ("NaN operand: allo keeps its sign/payload, rtl +0x7fc000",
     lambda s, g, w: ref.is_nan(g, 15) and int(w) == ref.ACC24_NAN),
] + DEVIATIONS + [
    ("off by one acc24 ulp (rounding)", lambda s, g, w: abs(int(g) - int(w)) == 1),
]
