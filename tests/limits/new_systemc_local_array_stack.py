# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track C finding C7 (dev/records/minitpu/u4_track_c_2026-10-08.rst): a
large kernel-local array is emitted by the SystemC emitter as a local of the
process's ``run()``, which runs on the SC_THREAD's coroutine stack (~64 KB),
so csim SEGFAULTS ("Simulation failed") once the array outgrows it. The
emitter already moves ``@ Stateful`` arrays and constant arrays to module
members for exactly this reason (EmitSystemC.cpp, "Not locals in run()");
plain locals -- including the storage ``compose`` generates for a README D-12
memory on Stream links, e.g. MiniTPU's VMEM (4,096 x 1,024 b = 512 KB) --
are not. The simulator runs the same region. Fix: emit a kernel-local array
above a size bound as a module member with no reset action (a plain local's
initial value is undefined anyway)."""
import os
import tempfile

import _worktree
from _worktree import verdict

ITEM = "new-systemc-local-array-stack"
import numpy as np  # noqa: E402

import allo.dataflow as df  # noqa: E402
from allo.ir.types import int32  # noqa: E402


def make(rows):
    @df.region()
    def top(A: int32[8], B: int32[8]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: int32[8], b: int32[8]):
            mem: int32[rows]
            for i in range(8):
                mem[(i * 977) % rows] = a[i]
                b[i] = mem[(i * 977) % rows] + 1

    return top


def main():
    a = np.arange(8, dtype=np.int32)
    res = {}
    with tempfile.TemporaryDirectory() as d:
        for rows in (1024, 262144):  # 4 KB fits the stack; 1 MB does not
            b = np.zeros(8, dtype=np.int32)
            try:
                mod = df.build(make(rows), target="systemc", mode="csim",
                               project=os.path.join(d, str(rows)))
                mod(a, b)
                res[rows] = b.tolist()
            except Exception as e:  # noqa: BLE001
                res[rows] = f"{type(e).__name__}: {str(e)[:60]}"
    sim = np.zeros(8, dtype=np.int32)
    df.build(make(262144), target="simulator")(a, sim)
    res["simulator 262144"] = sim.tolist()
    print(res)
    want = (a + 1).tolist()
    verdict(ITEM, res[1024] == want and res["simulator 262144"] == want and res[262144] != want, str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
