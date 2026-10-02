# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``vpu_bf16_add_pipe``, the two-stage BF16 adder behind vadd/vsub.

Bit-identical to ``vpu_bf16_add`` by MiniTPU's own ``tb/tb_bf16_add_pipe_equiv.sv``
(all 2^32 pairs), so it shares that unit's reference. Latency 2: "Two-stage"
in the module's header comment and two register banks (``s1_*_q``, then
``result_o``/``valid_o``); MiniTPU states no ``localparam`` for it on its own.
"""

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness import stimulus as stim
from examples.minitpu.units import bf16_add

RTL = rtl.RtlUnit(
    top="vpu_bf16_add_pipe",
    sources=["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_bf16_add_pipe.sv"],
    inputs=[("a_i", 16), ("b_i", 16)],
    outputs=[("result_o", 16)],
    shape="valid",
    latency=2,
)
LATENCY_SOURCE = "vpu_bf16_add_pipe.sv header comment (\"Two-stage\"); no localparam"

REF = ref.vpu_bf16_add
IEEE = ref.ieee_bf16_add
PROBE = ((0x3F80, 0x3F80), (0x4000, 0x3F80))  # 1+1 -> 1+2: the output moves


def stimulus():
    return stim.binary_bf16()


# Same function as vpu_bf16_add, so the same two Allo expressions: ``bits`` is
# bf16_add's line-for-line transcription (with its B1 spare-bit workaround).
# Neither states the latency -- the simulator is untimed; what an Allo unit can
# say about "latency 2" is dev/records/minitpu/u1_pipe_2026-10-02.rst.
VARIANTS = bf16_add.VARIANTS

# Every difference between IEEE and the RTL, by cause (first match wins);
# also the EXPLAIN rules for an Allo variant that computes IEEE.
DEVIATIONS = [
    ("NaN result is always +0x7fc0 (IEEE keeps a sign)",
     lambda s, g, w: ref.is_nan(g) and int(w) == 0x7FC0),
    ("(+0)+(-0) = -0 (IEEE RNE: +0)",
     lambda s, g, w: ref.is_zero(s[0]) and ref.is_zero(s[1]) and int(w) == int(s[1])),
]
# Allo variants are explained by bf16_add's rules, which add Catapult's
# ac_std_float NaN encoding (met by the SystemC ``native`` build).
EXPLAIN = bf16_add.EXPLAIN
