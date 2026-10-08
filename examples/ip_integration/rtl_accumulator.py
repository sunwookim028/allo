# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supplied RTL wrapper in an Allo dataflow graph; no HLS invocation by default."""

import argparse
from pathlib import Path

import numpy as np
import allo.dataflow as df
from allo import RTLModule, Port, HLSBlackBox
from allo.ir.types import int32, Stream

HERE = Path(__file__).resolve().parent
N = 8
ip = RTLModule(
    top="accumulator",
    rtl=HERE / "rtl_accumulator.v",
    ports=[
        Port("A", "a_dout", "a_empty_n", "a_read", size=1, protocol="ap_fifo"),
        Port(
            "C", "c_din", "c_write", "c_full_n", dir="out", size=1, protocol="ap_fifo"
        ),
    ],
    done="ap_done",
    persistent=True,
    hls=HLSBlackBox(c_model=str(HERE / "rtl_accumulator.cpp"), latency=3),
)


@df.region()
def rtl_accumulate(A: int32[N], C: int32[N]):
    a: Stream[int32, 2]
    c: Stream[int32, 2]

    @df.kernel(mapping=[1], args=[A])
    def feed(x: int32[N]):
        for i in range(N):
            a.put(x[i])

    @df.kernel(mapping=[1])
    def compute():
        for i in range(N):
            ip(a, c)

    @df.kernel(mapping=[1], args=[C])
    def drain(x: int32[N]):
        for i in range(N):
            x[i] = c.get()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--validate-rtl", action="store_true")
    parser.add_argument("--project", help="Emit a Vitis project without running Vitis")
    args = parser.parse_args()
    if args.validate_rtl:
        ip.validate_rtl()
    if args.simulate:
        mod = df.build(rtl_accumulate, target="simulator")
        x = np.arange(N, dtype=np.int32)
        y = np.zeros(N, dtype=np.int32)
        mod(x, y)
        np.testing.assert_array_equal(y, np.cumsum(x, dtype=np.int32))
        print("RTL simulation passed:", y)
    if args.project:
        project = Path(args.project)
        df.build(rtl_accumulate, target="vitis_hls", mode="csyn", project=str(project))
        # A deterministic testbench for manual vendor-tool validation.
        (project / "host.cpp").write_text(
            """// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "kernel.h"
int main() {
  int A[8], C[8] = {};
  for (int i = 0; i < 8; ++i) A[i] = i;
  rtl_accumulate(A, C);
  int sum = 0;
  for (int i = 0; i < 8; ++i) {
    sum += A[i];
    if (C[i] != sum) return 1;
  }
  return 0;
}
"""
        )
        tcl = (project / "run.tcl").read_text()
        (project / "validate.tcl").write_text(
            tcl.replace(
                "csynth_design",
                "csim_design\ncsynth_design\ncosim_design -rtl verilog\nexport_design -rtl verilog -format ip_catalog",
            )
        )
        print(f"Project emitted at {project}; Vitis has not been run.")
    if not (args.simulate or args.validate_rtl or args.project):
        parser.print_help()


if __name__ == "__main__":
    main()
