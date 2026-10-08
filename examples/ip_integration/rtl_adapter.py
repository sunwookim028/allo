# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a FIFO/chain wrapper and simulate it inside an Allo dataflow graph."""

import argparse
from pathlib import Path

import numpy as np
import allo.dataflow as df
from allo import RTLModule, Port, HLSBlackBox, ReadyValidAdapter
from allo.ir.types import int32, Stream

HERE = Path(__file__).resolve().parent
N = 8
ip = RTLModule(
    top="ready_valid_accumulator",
    rtl=HERE / "ready_valid_accumulator.v",
    ports=[
        Port("A", "a_data", "a_valid", "a_ready", size=4),
        Port("B", "b_data", "b_valid", "b_ready", size=4),
        Port("C", "c_data", "c_valid", "c_ready", dir="out", size=4),
    ],
    clock="clk",
    reset="rst_n",
    reset_active_high=False,
    start=None,
    persistent=True,
    hls=HLSBlackBox(
        c_model=str(HERE / "ready_valid_accumulator.cpp"),
        latency=8,
        adapter=ReadyValidAdapter(clock_enable="ce", start_mode="none"),
    ),
)


@df.region()
def adapted_accumulate(A: int32[N], B: int32[N], C: int32[N]):
    a: Stream[int32, 2]
    b: Stream[int32, 2]
    c: Stream[int32, 2]

    @df.kernel(mapping=[1], args=[A, B])
    def feed(x: int32[N], y: int32[N]):
        for i in range(N):
            a.put(x[i])
            b.put(y[i])

    @df.kernel(mapping=[1])
    def compute():
        for tile in range(N // 4):
            ip(a, b, c)

    @df.kernel(mapping=[1], args=[C])
    def drain(x: int32[N]):
        for i in range(N):
            x[i] = c.get()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", help="Write the generated Verilog for inspection")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--project", help="Emit a Vitis project, without running Vitis")
    args = parser.parse_args()
    if args.wrapper:
        Path(args.wrapper).write_text(ip.generate_wrapper(), encoding="utf-8")
    if args.simulate:
        mod = df.build(adapted_accumulate, target="simulator")
        x = np.arange(N, dtype=np.int32)
        y = np.ones(N, dtype=np.int32)
        out = np.zeros(N, dtype=np.int32)
        mod(x, y, out)
        np.testing.assert_array_equal(out, np.cumsum(x + y, dtype=np.int32))
        print("Generated wrapper simulation passed:", out)
    if args.project:
        project = Path(args.project)
        df.build(
            adapted_accumulate, target="vitis_hls", mode="csyn", project=str(project)
        )
        (project / "host.cpp").write_text(
            """// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "kernel.h"
int main() {
  int32_t A[8], B[8], C[8] = {};
  int32_t sum=0;
  for (int call=0; call<3; ++call) {
    for (int i=0; i<8; ++i) { A[i]=i-call; B[i]=2*call+1; C[i]=-1; }
    adapted_accumulate(A, B, C);
    for (int i=0; i<8; ++i) { sum+=A[i]+B[i]; if (C[i]!=sum) return 1; }
  }
  return 0;
}
""",
            encoding="utf-8",
        )
        script = (project / "run.tcl").read_text(encoding="utf-8")
        (project / "validate.tcl").write_text(
            script.replace(
                "csynth_design",
                "csim_design\ncsynth_design\ncosim_design -rtl verilog\nexport_design -rtl verilog -format ip_catalog",
            ),
            encoding="utf-8",
        )
        print(f"Vitis project emitted at {project}; Vitis has not been run.")
    if not (args.wrapper or args.simulate or args.project):
        parser.print_help()


if __name__ == "__main__":
    main()
