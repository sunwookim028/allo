# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal non-blocking stream example, lowered to RTL via Vitis HLS.

Producer writes 4 values into a depth-4 FIFO with a non-blocking try_put
(spin until it succeeds); consumer drains it with a non-blocking try_get
(spin until it succeeds) and writes the result out.

Usage:
    # 1. quick numeric check in the in-process simulator
    python examples/nb_stream_rtl.py sim

    # 2. inspect the generated Vitis HLS C++ (no tools needed)
    python examples/nb_stream_rtl.py codegen

    # 3. run Vitis HLS C-synthesis -> RTL (Verilog) under nb_stream.prj
    python examples/nb_stream_rtl.py csyn
"""
import sys
import numpy as np
import allo
from allo.ir.types import int32, int1, Stream
import allo.dataflow as df


@df.region()
def top_nb(out: int32[4]):
    S: Stream[int32, 4][1]

    @df.kernel(mapping=[1])
    def producer():
        for i in range(4):
            while not S[0].try_put(i * 10):
                pass

    @df.kernel(mapping=[1], args=[out])
    def consumer(out_buf: int32[4]):
        for i in range(4):
            val: int32 = 0
            ok: int1 = 0
            while ok == 0:
                val, ok = S[0].try_get()
            out_buf[i] = val


def run_sim():
    sim = df.build(top_nb, target="simulator")
    np_out = np.zeros(4, dtype=np.int32)
    sim(np_out)
    np.testing.assert_array_equal(np_out, [0, 10, 20, 30])
    print("sim result:", np_out, "-> PASSED")


def run_codegen():
    mod = df.build(top_nb, target="vitis_hls", mode="csyn", project="nb_stream.prj")
    print(mod.hls_code)


def run_csyn():
    mod = df.build(top_nb, target="vitis_hls", mode="csyn", project="nb_stream.prj")
    mod()  # no args in csyn mode -> triggers `vitis_hls -f run.tcl`
    print("Synthesis done. RTL under nb_stream.prj/out.prj/solution1/syn/verilog/")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "sim"
    {"sim": run_sim, "codegen": run_codegen, "csyn": run_csyn}[what]()
