# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``vpu_alu``, one lane of the VPU's elementwise ALU.

An input register, then ``vpu_bf16_add_pipe`` and ``vpu_bf16_mul_pipe`` side
by side and a two-deep delay for mov/max/min: latency
``vpu_pkg::VPU_ALU_LATENCY = 3`` for every op. (``docs/isa_latency.json``'s
``w: 5`` for vadd..vmov is the ISA-level writeback offset, which adds the
operand read and ``VPU_WB_STAGES``; it is not this unit's latency.)

``op_i`` is ``vpu_pkg::vpu_alu_op_e`` (``logic [3:0]``); ``harness/ref.py``
``ALU_OP`` holds its values. The stimulus drives all 16 codes, including
the declared AND/OR/XOR and the seven unused ones.
"""

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness import stimulus as stim

RTL = rtl.RtlUnit(
    top="vpu_alu",
    sources=[
        "src/core/vpu/vpu_pkg.sv",
        "src/core/vpu/vpu_bf16_add_pipe.sv",
        "src/core/vpu/vpu_bf16_mul.sv",
        "src/core/vpu/vpu_alu.sv",
    ],
    inputs=[("op_i", 4), ("op_a_i", 16), ("op_b_i", 16)],
    outputs=[("result_o", 16)],
    shape="valid",
    latency=3,
)
LATENCY_SOURCE = "vpu_pkg.sv localparam VPU_ALU_LATENCY = 3"


REF = ref.vpu_alu
IEEE = ref.ieee_vpu_alu
PROBE = ((ref.ALU_OP["ADD"], 0x3F80, 0x3F80), (ref.ALU_OP["ADD"], 0x4000, 0x3F80))


def stimulus():
    return stim.alu_ops(range(16), n_random=200_000)


VARIANTS = {}


def _op(s):
    return int(s[0])


def _nan_in(s):
    return ref.is_nan(s[1]) or ref.is_nan(s[2])


DEVIATIONS = [
    ("AND/OR/XOR not implemented: rtl returns a (mov)",
     lambda s, g, w: _op(s) in (6, 7, 8) and int(w) == int(s[1])),
    ("ADD/SUB/MUL: NaN result is always +0x7fc0",
     lambda s, g, w: _op(s) in (0, 1, 2) and ref.is_nan(g) and int(w) == 0x7FC0),
    ("ADD: (+0)+(-0) = -0 (IEEE: +0)",
     lambda s, g, w: _op(s) == 0 and ref.is_zero(s[1]) and ref.is_zero(s[2])),
    ("SUB: (+0)-(+0) = -0 (IEEE: +0)",
     lambda s, g, w: _op(s) == 1 and ref.is_zero(s[1]) and ref.is_zero(s[2])),
    ("MUL: subnormal operand flushed to signed 0",
     lambda s, g, w: _op(s) == 2 and ref.is_zero(w)
     and (ref.exp_field(s[1]) == 0 or ref.exp_field(s[2]) == 0)),
    ("MUL: product < 2^-126 flushed to signed 0",
     lambda s, g, w: _op(s) == 2 and ref.is_zero(w)),
    ("MAX/MIN: NaN ordered by bf16_gt key, payload kept (IEEE maximum: NaN)",
     lambda s, g, w: _op(s) in (4, 5) and _nan_in(s)),
]
EXPLAIN = DEVIATIONS
