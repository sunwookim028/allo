# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Systolic-chain SystemC example: a mapping=[P] grid of PEs between an
# array->stream feeder and a stream->array drain. Demonstrates that the SystemC
# backend handles grids with NO special emitter logic: Allo unrolls mapping=[P]
# into P separate kernel funcs (pe_0..pe_{P-1}) with get_pid()/link[i] resolved
# to concrete neighbor streams, and emitTopModule wires them like any other
# kernels.  B = A + P  (each PE adds 1).
#
# Verified on zhang-21 (Catapult 2024.2): csim B = A + P, and DESIGN_HIERARCHY
# {top} synthesizes non-empty (P=4,N=8 -> Real ops 106, area ~1486) with A/B as
# Connections valid/ready stream ports.

import numpy as np
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

P, N = 4, 8  # P PEs each +1  =>  B = A + P


@df.region()
def top(A: int32[N], B: int32[N]):
    link: Stream[int32, 4][P + 1]

    @df.kernel(mapping=[1], args=[A])  # feeder: array A -> link[0]
    def feed(a: int32[N]):
        for k in range(N):
            link[0].put(a[k])

    @df.kernel(mapping=[P])  # grid: P PEs, each link[i] -> +1 -> link[i+1]
    def pe():
        i = df.get_pid()
        for k in range(N):
            v: int32 = link[i].get()
            w: int32 = v + 1  # explicit int32 (avoid i33 widening)
            link[i + 1].put(w)

    @df.kernel(mapping=[1], args=[B])  # drain: link[P] -> array B
    def drain(b: int32[N]):
        for k in range(N):
            b[k] = link[P].get()


if __name__ == "__main__":
    import os

    code = df.build(top, target="systemc").hls_code
    open("systolic_chain.cpp", "w").write(code)
    mods = [l for l in code.splitlines() if l.startswith("SC_MODULE")]
    print("emitted modules:", mods)  # feed_0, pe_0..pe_3, drain_0, top, tb

    if os.environ.get("MGC_HOME"):
        mod = df.build(top, target="systemc", mode="csim", project="systolic_chain.prj")
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)  # Option B fills B with the design's output
        print("A =", A)
        print("B =", B, "(expected A + P =", A + P, ")")
        print("PASS" if np.array_equal(B, A + P) else "FAIL")
