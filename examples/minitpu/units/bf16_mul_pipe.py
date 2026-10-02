# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``vpu_bf16_mul_pipe``, the two-stage BF16 multiplier ``vpu_alu`` uses.

It has a clock and nothing else: no reset, no valid (the ALU's valid rides
``vpu_bf16_add_pipe``), so the harness drives it ``bare``. Two register banks
(``*_s1_q`` and ``result_o``) give latency 2; MiniTPU states no
``localparam`` for it alone. Same numerics as ``vpu_bf16_mul``.
"""

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
VARIANTS = {}
DEVIATIONS = bf16_mul.DEVIATIONS
EXPLAIN = DEVIATIONS
