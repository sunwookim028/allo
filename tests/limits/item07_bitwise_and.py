# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 7: bitwise `&` (and friends) inside a `@df.kernel` body.

Original claim: Allo rejects `&`, so `(iflags & 2) >> 1` had to be written as
`(iflags // 2) - ((iflags // 4) * 2)`."""
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 7
import allo.dataflow as df
from allo.ir.types import int32, UInt


@df.region()
def top(flags: int32[4], out: int32[4]):
    @df.kernel(mapping=[1], args=[flags, out])
    def dec(f: int32[4], o: int32[4]):
        for i in range(4):
            iflags: int32 = f[i]
            a_on: int32 = (iflags & 2) >> 1
            b_on: int32 = (iflags >> 2) & 1
            mix: int32 = (iflags | 16) ^ (iflags << 1)
            o[i] = a_on + 10 * b_on + 100 * mix


@df.region()
def top_u(flags: UInt(64)[2], out: int32[2]):
    @df.kernel(mapping=[1], args=[flags, out])
    def dec(f: UInt(64)[2], o: int32[2]):
        for i in range(2):
            w: UInt(64) = f[i]
            o[i] = (w >> 54) & 0xFF


def ref(x):
    return ((x & 2) >> 1) + 10 * ((x >> 2) & 1) + 100 * ((x | 16) ^ (x << 1))


def main():
    res = {}
    try:
        mod = df.build(top, target="simulator")
        f = np.array([0, 2, 6, 13], dtype=np.int32)
        o = np.zeros(4, dtype=np.int32)
        mod(f, o)
        exp = [ref(int(x)) for x in f]
        res["int32 & | ^ << >>"] = "ok" if o.tolist() == exp else f"WRONG {o.tolist()} != {exp}"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        traceback.print_exc()
        res["int32 & | ^ << >>"] = f"{type(e).__name__}: {str(e)[:150]}"
    try:
        mod = df.build(top_u, target="simulator")
        f = np.array([(200 << 54) | 5, (64 << 54)], dtype=np.uint64)
        o = np.zeros(2, dtype=np.int32)
        mod(f, o)
        res["UInt(64) (w>>54)&0xFF"] = "ok" if o.tolist() == [200, 64] else f"WRONG {o.tolist()} != [200, 64]"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        traceback.print_exc()
        res["UInt(64) (w>>54)&0xFF"] = f"{type(e).__name__}: {str(e)[:150]}"
    # Informational: the HLS emitter, and the cpp-style typing rule set (AIE).
    import allo
    from allo.customize import customize as _c

    def plain(x: int32) -> int32:
        return (x & 2) >> 1

    try:
        code = str(allo.customize(plain).build(target="vhls"))
        res["vhls emits &"] = "ok" if "&" in code else "NO & in code"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        res["vhls emits &"] = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        _c(plain, typing_rule_set="cpp-style")
        info = "accepted"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        info = f"REJECTED ({type(e).__name__})"
    print(res)
    print(f"info: typing_rule_set='cpp-style' (AIE target): {info}")
    verdict(ITEM, any(v != "ok" for v in res.values()), str(res),
            absent="CANNOT-REPRODUCE (supported in the default rule set since 12f898d7, 2023)")


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
