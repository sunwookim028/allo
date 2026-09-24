# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stream-boundary SystemC example for the Catapult backend.
#
# KEY FACT: an Allo @df.region ALWAYS has array/scalar boundaries (its function
# args) — there is no Stream-typed region argument. Streams exist only INSIDE the
# region, as Connections::In/Out PORTS on the individual @df.kernel modules.
#
# Consequence for synthesis:
#   - The whole {top} region has ARRAY I/O (A, B) at its edge. Those arrays are
#     top-level MEMBERS, not memory PORTS, so Catapult DCE's {top} to empty
#     (until boundary arrays are modeled as memory-interface ports — see README).
#   - A single KERNEL whose interface is STREAMS (Connections::In/Out) synthesizes
#     to real, non-empty RTL (observable valid/ready ports). That is `compute`
#     below — synthesize DESIGN_HIERARCHY {compute_0}.
#
# So this file shows the split cleanly: source/sink touch the array boundary,
# `compute` is pure stream->stream and is the synthesizable unit.

import numpy as np
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

N = 8


@df.region()
def top(A: int32[N], B: int32[N]):
    s_in: Stream[int32, 4][1]
    s_out: Stream[int32, 4][1]

    # array -> stream (touches the array boundary)
    @df.kernel(mapping=[1], args=[A])
    def source(a: int32[N]):
        for i in range(N):
            s_in[0].put(a[i])

    # stream -> stream : THE synthesizable DUT (Connections::In + Connections::Out,
    # no arrays). DESIGN_HIERARCHY {compute_0} -> real non-empty RTL.
    @df.kernel(mapping=[1])
    def compute():
        for i in range(N):
            s_out[0].put(s_in[0].get() + 1)

    # stream -> array (touches the array boundary)
    @df.kernel(mapping=[1], args=[B])
    def sink(b: int32[N]):
        for i in range(N):
            b[i] = s_out[0].get()


if __name__ == "__main__":
    # (a) emit + inspect the SystemC
    code = df.build(top, target="vitis_hls").hls_code
    open("stream_boundary.cpp", "w").write(code)
    print("wrote stream_boundary.cpp")
    assert "SC_MODULE(compute_0)" in code
    print("compute_0 is a stream->stream module (Connections::In + Connections::Out)")

    # (b) csim the whole region (simulation ignores the port/DCE issue)
    import os
    if os.environ.get("MGC_HOME"):
        mod = df.build(top, target="vitis_hls", mode="csim", project="vstream_boundary.prj")
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)  # Option B fills B with the design's output
        print("A =", A)
        print("B =", B, "(expected A + 1 =", A + 1, ")")
        print("PASS" if np.array_equal(B, A + 1) else "FAIL")
