# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 2: `@ Stateful` declared inside a `@df.kernel` body.

Original claim: AttributeError `'ASTContext' object has no attribute
'global_op_cache'`, forcing PC / counters to be hoisted to region scope.
"""
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 2
import allo.dataflow as df
from allo.ir.types import Stateful, int32


@df.region()
def top(out: int32[2]):
    @df.kernel(mapping=[1], args=[out])
    def k(o: int32[2]):
        pc: int32[1] @ Stateful = 0
        halted: int32[1] @ Stateful = 0
        for i in range(3):
            if halted[0] == 0:
                pc[0] = pc[0] + 1
        if pc[0] >= 6:
            halted[0] = 1
        o[0] = pc[0]
        o[1] = halted[0]


def main():
    try:
        mod = df.build(top, target="simulator")
        out = np.zeros(2, dtype=np.int32)
        mod(out)
        first = out.tolist()
        mod(out)
        second = out.tolist()
        mod(out)
        third = out.tolist()
        ok = first == [3, 0] and second == [6, 1] and third == [6, 1]
        verdict(2, not ok, f"calls -> {first}, {second}, {third} (expect [3,0],[6,1],[6,1])")
    except (Exception, SystemExit) as e:  # noqa: BLE001  (customize() sys.exit(1)s on frontend errors)
        traceback.print_exc()
        verdict(2, True, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
