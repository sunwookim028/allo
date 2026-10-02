# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``mxu_acc24_add_pipe``, the array's three-stage acc24 adder.

acc24 is 1+8+15 with subnormals (``vpu_pkg::MXU_ACC_FRAC_W = 15``). Latency
is ``vpu_pkg::MXU_ACC_ADD_LATENCY = 3`` ("mxu_acc24_add_pipe's stages").
Built with the package default ``MXU_ACC_USE_DSP = 0`` (fabric shifters);
MiniTPU's ``tb_acc24_add_dsp_equiv`` holds the DSP profile bit-identical.
"""

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


VARIANTS = {}

DEVIATIONS = [
    ("NaN result is always +0x7fc000 (IEEE keeps a sign)",
     lambda s, g, w: ref.is_nan(g, 15) and int(w) == ref.ACC24_NAN),
    ("(+0)+(-0) = -0 (IEEE RNE: +0)",
     lambda s, g, w: ref.is_zero(s[0], 15) and ref.is_zero(s[1], 15) and int(w) == int(s[1])),
]
EXPLAIN = DEVIATIONS
