# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1 pilot: ``vpu_bf16_add``, a combinational BF16 adder.

Two Allo expressions of the same unit, because they probe different things:

``native``
    ``a + b`` on Allo's ``bfloat16``. Probes the type system's numerics: what
    the simulator, the SystemC emitter and Catapult's ``ac_std_float`` each
    make of a bf16 add.
``bits``
    The RTL's own algorithm on ``uint16`` fields (align, add, normalize,
    round-to-nearest-even). Probes whether Allo can say what the hardware does
    when the type system's numerics are not the ones wanted.

Each build maps the unit over ``n`` vectors in one kernel; ``n`` is fixed per
build because Allo's array shapes are static.
"""

import ml_dtypes
import numpy as np

import allo.dataflow as df
from allo.ir.types import bfloat16, uint16

from examples.minitpu.harness import rtl

RTL = rtl.RtlUnit(
    top="vpu_bf16_add",
    sources=["src/core/vpu/vpu_bf16_add.sv"],
    inputs=[("a_i", 16), ("b_i", 16)],
    outputs=[("result_o", 16)],
    shape="comb",
    latency=0,
)


def native(n):
    @df.region()
    def top(A: bfloat16[n], B: bfloat16[n], C: bfloat16[n]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def add(a: bfloat16[n], b: bfloat16[n], c: bfloat16[n]):
            for i in range(n):
                c[i] = a[i] + b[i]

    return top


def run_native(mod, stim):
    """``stim``: uint16[n, 2] bit patterns. Returns uint16[n] bit patterns."""
    a = np.ascontiguousarray(stim[:, 0]).view(ml_dtypes.bfloat16)
    b = np.ascontiguousarray(stim[:, 1]).view(ml_dtypes.bfloat16)
    c = np.zeros(len(stim), dtype=ml_dtypes.bfloat16)
    mod(a, b, c)
    return c.view(np.uint16)


VARIANTS = {"native": (native, run_native)}


def _is_nan(x):
    return (int(x) & 0x7F80) == 0x7F80 and (int(x) & 0x7F) != 0


# Rules that name a difference between an Allo result and the RTL's, first
# match wins. Each is a measured RTL deviation from IEEE (harness/ref.py).
EXPLAIN = [
    ("NaN sign: allo keeps it, rtl always +0x7fc0",
     lambda s, g, w: _is_nan(g) and int(w) == 0x7FC0),
    ("NaN payload/canonical form differs",
     lambda s, g, w: _is_nan(g) and _is_nan(w)),
    ("(+0)+(-0): allo +0 (IEEE), rtl -0",
     lambda s, g, w: (int(s[0]) & 0x7FFF) == 0 and (int(s[1]) & 0x7FFF) == 0),
    ("subnormal result flushed/changed",
     lambda s, g, w: (int(w) & 0x7F80) == 0 or (int(g) & 0x7F80) == 0),
    ("subnormal operand",
     lambda s, g, w: (int(s[0]) & 0x7F80) == 0 or (int(s[1]) & 0x7F80) == 0),
    ("off by one ulp (rounding)",
     lambda s, g, w: abs(int(g) - int(w)) == 1),
]
