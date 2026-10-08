# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""M-R0 probe (a): a region boundary int32 array read and written in place by one kernel, target="simulator".

This is the Allo side of the MemPort-style boundary the (a-K, RAM form) shim needs, without any RTL.
"""
import numpy as np
import allo.dataflow as df
from allo.ir.types import int32

N = 8


@df.region()
def top(ddr: int32[N]):
    @df.kernel(mapping=[1], args=[ddr])
    def host(mem: int32[N]):
        for i in range(N):
            mem[i] = mem[i] * 2 + 1  # read, then write the same boundary word


mod = df.build(top, target="simulator")
x = np.arange(N, dtype=np.int32)
mod(x)
np.testing.assert_array_equal(x, np.arange(N) * 2 + 1)
print("BOUNDARY_ARRAY_SIM PASS", x.tolist())
