# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 8: the same sub-region (`mxu(...)`) called from two mutually
exclusive `if`/`elif` arms of one kernel.

Original claim: each callsite becomes an independent instance and the design
"silently breaks", so every level keeps `mxu(...)` in exactly one combined arm.
Checked two ways: a stream-carrying sub-region (the mxu shape) and a plain
one, each called from two arms, compared against the one-callsite form."""
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 8
import allo.dataflow as df
from allo.ir.types import Stream, int32


@df.region()
def mxu(a: int32[4], c: int32[4]):
    s: Stream[int32, 4]

    @df.kernel(mapping=[1], args=[a])
    def feed(la: int32[4]):
        for i in range(4):
            s.put(la[i] * 2)

    @df.kernel(mapping=[1], args=[c])
    def drain(lc: int32[4]):
        for i in range(4):
            lc[i] = lc[i] + s.get()


@df.region()
def top_two(ops: int32[3], a: int32[4], c: int32[4]):
    @df.kernel(mapping=[1], args=[ops, a, c])
    def driver(o: int32[3], la: int32[4], lc: int32[4]):
        for t in range(3):
            if o[t] == 1:        # "preloaded"
                mxu(la, lc)
            elif o[t] == 2:      # "accumulated"
                mxu(la, lc)
                for i in range(4):
                    lc[i] = lc[i] + 1


@df.region()
def top_one(ops: int32[3], a: int32[4], c: int32[4]):
    @df.kernel(mapping=[1], args=[ops, a, c])
    def driver(o: int32[3], la: int32[4], lc: int32[4]):
        for t in range(3):
            if o[t] == 1 or o[t] == 2:
                mxu(la, lc)
            if o[t] == 2:
                for i in range(4):
                    lc[i] = lc[i] + 1


def run(top):
    mod = df.build(top, target="simulator")
    ops = np.array([1, 2, 1], dtype=np.int32)
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    c = np.zeros(4, dtype=np.int32)
    mod(ops, a, c)
    return c.tolist()


def main():
    exp = [3 * 2 * x + 1 for x in (1, 2, 3, 4)]
    res = {}
    for name, top in (("one callsite", top_one), ("two callsites", top_two)):
        try:
            got = run(top)
            res[name] = "ok" if got == exp else f"WRONG {got} != {exp}"
        except (Exception, SystemExit) as e:  # noqa: BLE001
            traceback.print_exc()
            res[name] = f"{type(e).__name__}: {str(e)[:150]}"
    print(res)
    verdict(ITEM, res["two callsites"] != "ok", str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
