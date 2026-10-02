# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``vpu_bf16_mul``, the combinational BF16 multiplier (vmul's twin).

``src/core/vpu/vpu_bf16_mul.sv`` holds two modules: this combinational one and
``vpu_bf16_mul_pipe`` (``bf16_mul_pipe.py``), the two-stage form ``vpu_alu``
instantiates. Both flush subnormals, on input and on output.
"""

import numpy as np

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


VARIANTS = {}


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
EXPLAIN = DEVIATIONS
