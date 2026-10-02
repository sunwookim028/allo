# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``mxu_bf16_mul_acc24``, the array's multiplier: bf16 x bf16 -> acc24.

Combinational. The product is exact in acc24 (1+8+15, the MXU accumulator
format, ``vpu_pkg::MXU_ACC_W = 24``) except where it underflows or overflows,
and, like ``vpu_bf16_mul``, it flushes subnormals on input and on output --
though acc24, unlike the multiplier, has subnormals.
"""

import numpy as np

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


VARIANTS = {}


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
EXPLAIN = DEVIATIONS
