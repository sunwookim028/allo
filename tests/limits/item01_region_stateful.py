# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 1: region-scope `@ Stateful` read-modify-written by a kernel inside a
loop AND a branch, shared with a second kernel.

Original claim: AttributeError `global_op_cache`, or MLIR `Assertion 'value'
failed`, when a kernel reads-and-writes a region-scope Stateful inside a loop
or branch.
"""
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 1
import allo.dataflow as df
from allo.ir.types import Stateful, int32


@df.region()
def top(sel: int32[1], out_a: int32[4], out_b: int32[4]):
    acc: int32[4] @ Stateful = 0
    pc: int32[1] @ Stateful = 0

    @df.kernel(mapping=[1], args=[sel, out_a])
    def decoder(s: int32[1], po: int32[4]):
        for i in range(4):
            if s[0] == 1:
                acc[i] = acc[i] + i
            else:
                acc[i] = acc[i] - 1
            po[i] = acc[i]
        pc[0] = pc[0] + 1

    @df.kernel(mapping=[1], args=[out_b])
    def driver(pb: int32[4]):
        for i in range(4):
            if pc[0] > 0:
                pb[i] = acc[i] + pc[0]


def main():
    try:
        mod = df.build(top, target="simulator")
        sel = np.array([1], dtype=np.int32)
        a = np.zeros(4, dtype=np.int32)
        b = np.zeros(4, dtype=np.int32)
        mod(sel, a, b)
        mod(sel, a, b)
        exp_a = np.array([0, 2, 4, 6], dtype=np.int32)
        ok = np.array_equal(a, exp_a)
        # driver's view of acc/pc is racy across kernels (no ordering), so
        # only decoder's persistent output is checked.
        verdict(1, not ok, f"out_a after 2 calls = {a.tolist()} (expect {exp_a.tolist()}), out_b = {b.tolist()}")
    except (Exception, SystemExit) as e:  # noqa: BLE001  (customize() sys.exit(1)s on frontend errors)
        traceback.print_exc()
        verdict(1, True, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
