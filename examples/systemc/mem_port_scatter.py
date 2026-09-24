# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Store-side (write) memory-port SystemC example. B is WRITTEN reversed
# (b[N-1-i] = ...), a scatter that is NOT a sequential Push-in-order stream, so
# the backend routes B to an internal write-only AlloMemW addressed by STORE
# reqs. A is a normal sequential input stream. B[N-1-i] = A[i] + 1  =>  B = (A+1)[::-1].
#
# There is NO stream output here, so completion is time-based (sc_start(T)); the
# testbench then reads AlloMemW.mem[] out to output0.data.

import numpy as np
import allo
from allo.ir.types import int32
import allo.dataflow as df

N = 8


@df.region()
def top(A: int32[N], B: int32[N]):
    @df.kernel(mapping=[1], args=[A, B])
    def scat(a: int32[N], b: int32[N]):
        for i in range(N):
            b[N - 1 - i] = a[i] + 1


if __name__ == "__main__":
    import os

    code = df.build(top, target="systemc").hls_code
    open("mem_port_scatter.cpp", "w").write(code)
    print("wrote mem_port_scatter.cpp")
    # Memory boundary is RAM PINS now (AlloMemPins + _wadr/_d/_we), not the old
    # packed-request AlloMemW over Connections.
    assert "AlloMemPins<" in code, "expected an AlloMemPins instance for B"
    assert "_wr(" in code, "expected a modulario _wr() store in the body"
    print("emitted AlloMemW + STORE handshake")

    if os.environ.get("MGC_HOME"):
        mod = df.build(top, target="systemc", mode="csim", project="mem_port_scatter.prj")
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)
        exp = (A + 1)[::-1]
        print("A =", A)
        print("B =", B, "(expected (A+1)[::-1] =", exp, ")")
        print("PASS" if np.array_equal(B, exp) else "FAIL")
