# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 3: simulator stream lowering skips sub-region calls nested in
affine.for / affine.if (the callee's stream ops survive to LLVM lowering and
`convert-func-to-llvm` fails on a `func.func`)."""
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 3
import allo.dataflow as df
from allo.ir.types import Stream, int32


@df.region()
def inner(slot: int32[1]):
    s: Stream[int32, 4]

    @df.kernel(mapping=[1])
    def producer():
        s.put(7)

    @df.kernel(mapping=[1], args=[slot])
    def consumer(o: int32[1]):
        v: int32 = s.get()
        o[0] = o[0] + v


@df.region()
def top(sel: int32[1], out: int32[1]):
    @df.kernel(mapping=[1], args=[sel, out])
    def driver(s_: int32[1], o: int32[1]):
        for _ in range(3):
            if s_[0] == 1:
                inner(o)


def main():
    try:
        mod = df.build(top, target="simulator")
        out = np.zeros(1, dtype=np.int32)
        mod(np.array([1], dtype=np.int32), out)
        verdict(3, out[0] != 21, f"out = {out[0]} (expect 21)")
    except (Exception, SystemExit) as e:  # noqa: BLE001  (customize() sys.exit(1)s on frontend errors)
        traceback.print_exc()
        verdict(3, True, f"{type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
