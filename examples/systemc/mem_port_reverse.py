# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Random-access (memory-port) SystemC example.  A is read REVERSED (a[N-1-i]),
# which is NOT a sequential 1-D scan, so the backend routes A to an internal
# AlloMem (req/resp memory port) instead of a stream. B is a normal sequential
# output stream (drives sc_stop completion).  B[i] = A[N-1-i] + 1.
#
# Demonstrates #4: a boundary array that a kernel accesses at arbitrary indices
# becomes an internal memory (the only memory kind the SystemC flow supports),
# preloaded by the testbench, addressed over a Connections req/resp handshake.

import numpy as np
import allo
from allo.ir.types import int32
import allo.dataflow as df

N = 8


@df.region()
def top(A: int32[N], B: int32[N]):
    @df.kernel(mapping=[1], args=[A, B])
    def rev(a: int32[N], b: int32[N]):
        for i in range(N):
            b[i] = a[N - 1 - i] + 1


if __name__ == "__main__":
    import os

    code = df.build(top, target="systemc").hls_code
    open("mem_port_reverse.cpp", "w").write(code)
    print("wrote mem_port_reverse.cpp")
    # Memory boundary is RAM PINS now (AlloMemPins + _radr/_re/_q).
    assert "AlloMemPins<" in code, "expected an AlloMemPins instance for A"
    assert "_rd(" in code, "expected a modulario _rd() load in the body"
    print("emitted AlloMem + req/resp handshake")

    if os.environ.get("MGC_HOME"):
        mod = df.build(top, target="systemc", mode="csim", project="mem_port_reverse.prj")
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)
        exp = A[::-1] + 1
        print("A =", A)
        print("B =", B, "(expected A[::-1] + 1 =", exp, ")")
        print("PASS" if np.array_equal(B, exp) else "FAIL")
