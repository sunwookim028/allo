# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: ``sfu``, the special-function unit: vgelu, vexp, vrecip, vrsqrt.

One element per cycle, II=1, five registers deep (``input_*_q``, stages 1-3,
``result_o``); ``valid_i``/``valid_o`` and the op tag are reset, the payload
is not (``sfu.sv`` "No reset on payload"). vgelu/vexp are 2048-entry ROMs
(``$readmemh`` of ``gelu_bf16.mem``/``exp_bf16.mem``), vrecip/vrsqrt are
32-bin piecewise-linear tables written as ``case`` functions. ``sfu_group``
is ``LANES`` copies of this module sharing ``valid_i``/``op_i`` (4 groups of
16 lanes in ``vpu.sv``): replication, not characterized separately.

Declared latency 5: ``vpu_pkg.sv:64`` ``VPU_SFU_LATENCY = 5 // sfu_group``;
``docs/isa_latency.json`` ``WB_W_SFU = 7`` = 5 + ``VPU_WB_STAGES`` (2).

Reference ``ref.sfu``: the logic modelled bit for bit; the ROM contents and
the two PWL tables are read from the clone as data (``ref._sfu_tables``), so
``tb_sfu_math_sweep`` -- not this harness -- is what checks the tables.
"""

import os

import numpy as np

from examples.minitpu.harness import ref, rtl

_HOME = rtl.minitpu_home()

RTL = rtl.RtlUnit(
    top="sfu",
    sources=["src/core/vpu/vpu_pkg.sv", "src/core/sfu/sfu.sv"],
    inputs=[("op_i", 2), ("operand_i", 16)],
    outputs=[("result_o", 16)],
    shape="valid",
    latency=5,
    # the driver does not run in the clone: $readmemh needs absolute paths
    params={"GELU_MEM_FILE": '"%s"' % os.path.join(_HOME, "src/core/sfu/gelu_bf16.mem"),
            "EXP_MEM_FILE": '"%s"' % os.path.join(_HOME, "src/core/sfu/exp_bf16.mem")},
)
LATENCY_SOURCE = "vpu_pkg.sv:64 VPU_SFU_LATENCY = 5 (isa_latency.json WB_W_SFU 7 = 5 + VPU_WB_STAGES 2)"

REF = ref.sfu
IEEE = ref.ieee_sfu
# rsqrt(4) = 0.5 -> rsqrt(1) = 1: the output moves
PROBE = ((3, 0x4080), (3, 0x3F80))


def stimulus():
    """Every op over every bf16 operand (op-major), then 200,000 vectors with
    a random op on every cycle, so the op tag must travel with its operand."""
    x = np.arange(65536, dtype=np.uint64)
    blocks = [np.stack([np.full(65536, op, dtype=np.uint64), x], axis=1) for op in range(4)]
    rng = np.random.default_rng(0x5F0)
    mix = np.stack([rng.integers(0, 4, 200_000, dtype=np.uint64),
                    rng.integers(0, 65536, 200_000, dtype=np.uint64)], axis=1)
    return np.concatenate(blocks + [mix])


def _f(x):
    return float(ref._bf16_to_f32(np.uint16(x)))


def _ulps(g, w):
    """Distance in bf16 ulps between two finite same-sign patterns."""
    return abs(int(g) - int(w))


def _ulp(w):
    e = (int(w) >> 7) & 0xFF
    return 2.0 ** (max(e, 1) - 127 - 7)


def _rel(g, w):
    a, b = _f(g), _f(w)
    return abs(a - b) / max(abs(a), 1e-38)


_SUB = lambda x: (int(x) >> 7) & 0xFF == 0 and int(x) & 0x7F != 0

# Every difference between the math (float64, one RNE to bf16) and the RTL,
# by cause, first match wins. s = (op, x); g = IEEE; w = RTL.
DEVIATIONS = [
    ("NaN in -> +0x7fc0 (IEEE keeps the operand's NaN sign/payload)",
     lambda s, g, w: ref.is_nan(s[1]) and int(w) == 0x7FC0),
    ("vexp(x > 0) = 1.0 (unit covers x <= 0 only; clamps silently)",
     lambda s, g, w: s[0] == 1 and int(s[1]) >> 15 == 0 and not ref.is_zero(s[1])
     and int(w) == 0x3F80),
    ("vrsqrt(+-0) = NaN (IEEE: +-Inf)",
     lambda s, g, w: s[0] == 3 and ref.is_zero(s[1]) and int(w) == 0x7FC0),
    ("vrsqrt(x < 0) = +0x7fc0 (IEEE: NaN, sign differs)",
     lambda s, g, w: s[0] == 3 and ref.is_nan(g) and int(w) == 0x7FC0),
    ("vexp(x <= -16) = 0 (table domain [-16, 0); IEEE a tiny normal/subnormal)",
     lambda s, g, w: s[0] == 1 and _f(s[1]) <= -16 and int(w) == 0),
    ("vgelu(x < -8) = 0 (table domain; IEEE a tiny negative)",
     lambda s, g, w: s[0] == 0 and _f(s[1]) <= -8 and int(w) == 0),
    ("subnormal operand: exponent field 0 read as a normal exponent (recip/rsqrt)",
     lambda s, g, w: s[0] in (2, 3) and _SUB(s[1])),
    ("vrecip, |x| in [2^127, 2^128): exponent 253 - 254 wraps to 0xff -> Inf or a "
     "NON-canonical NaN pattern (e.g. 0x7fff)",
     lambda s, g, w: s[0] == 2 and ((int(s[1]) >> 7) & 0xFF) == 254 and (int(w) >> 7) & 0xFF == 0xFF),
    ("vrecip, |x| in [2^126, 2^127): exponent field 0 with the fraction bits -> a subnormal "
     "pattern worth half (hidden bit lost); IEEE the subnormal 1/x",
     lambda s, g, w: s[0] == 2 and ((int(s[1]) >> 7) & 0xFF) == 253),
    ("vgelu/vexp subnormal operand -> table centre bin",
     lambda s, g, w: s[0] in (0, 1) and _SUB(s[1])),
    ("vrecip/vrsqrt PWL truncation, <= 2 ulp",
     lambda s, g, w: s[0] in (2, 3) and int(g) >> 15 == int(w) >> 15 and _ulps(g, w) <= 2),
    ("vexp table step (input quantized to 1/128, bin centre): rel <= 1%",
     lambda s, g, w: s[0] == 1 and _rel(g, w) <= 0.01),
    ("vgelu table step: abs <= 4e-3 (half a 1/128 bin) + 1 ulp of the bf16 entry, or rel <= 1%",
     lambda s, g, w: s[0] == 0 and (abs(_f(g) - _f(w)) <= 4e-3 + _ulp(w) or _rel(g, w) <= 0.01)),
]
