# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an array-ported C++ IP inside an Allo dataflow kernel."""

from pathlib import Path

import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import int32


def test_array_ip_dataflow():
    vadd = allo.IPModule(
        top="vadd",
        impl=Path(__file__).with_name("vadd.cpp"),
        link_hls=False,
    )

    @df.region()
    def top(A: int32[32], B: int32[32], C: int32[32]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def compute(a: int32[32], b: int32[32], c: int32[32]):
            vadd(a, b, c)

    mod = df.build(top, target="simulator")
    a = np.arange(-16, 16, dtype=np.int32)
    b = np.arange(32, dtype=np.int32)
    c = np.zeros(32, dtype=np.int32)
    mod(a, b, c)
    np.testing.assert_array_equal(c, a + b)
