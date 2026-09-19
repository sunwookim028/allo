# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 5: a kernel-local name that matches an enclosing `@df.region`
parameter.

Original claim: `d_addr: int32 = cmd[3]` inside a kernel raises
`AssertionError: Invalid assignment to d_addr, type mismatch` because the
region's `d_addr: int32[1]` parameter leaks into the kernel scope."""
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 5
import allo.dataflow as df
from allo.ir.types import int32


@df.region()
def top(cmd: int32[4], d_addr: int32[1], out: int32[1]):
    @df.kernel(mapping=[1], args=[cmd, out])
    def driver(c: int32[4], o: int32[1]):
        d_addr: int32 = c[3]  # same name as the region parameter
        o[0] = d_addr + 1

    @df.kernel(mapping=[1], args=[d_addr])
    def other(d: int32[1]):
        d[0] = 5


@df.region()
def top_same_type(buf: int32[4], out: int32[4]):
    @df.kernel(mapping=[1], args=[out])
    def k(o: int32[4]):
        buf: int32[4] = 0  # same name AND same type as the region parameter
        for i in range(4):
            buf[i] = i + 1
        for i in range(4):
            o[i] = buf[i]


def variant_b():
    try:
        mod = df.build(top_same_type, target="simulator")
        b = np.full(4, 9, dtype=np.int32)
        o = np.zeros(4, dtype=np.int32)
        mod(b, o)
        return "ok" if o.tolist() == [1, 2, 3, 4] and b.tolist() == [9] * 4 else f"WRONG o={o} b={b}"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[-1][:160] if str(e) else ''}"


def main():
    print("variant b (same type):", variant_b())
    try:
        mod = df.build(top, target="simulator")
        cmd = np.array([0, 0, 0, 41], dtype=np.int32)
        d = np.zeros(1, dtype=np.int32)
        out = np.zeros(1, dtype=np.int32)
        mod(cmd, d, out)
        verdict(5, out[0] != 42, f"variant a: out = {out[0]} (expect 42), d = {d[0]} (expect 5)")
    except (Exception, SystemExit) as e:  # noqa: BLE001  (customize() sys.exit(1)s on frontend errors)
        traceback.print_exc()
        verdict(5, True, f"variant a (int32 local vs int32[1] param): {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
