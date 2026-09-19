# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 14: `wrap_io=False` rejects multi-dimensional arguments to nested
kernels ("Top-level multi-dimensional arrays are linearized to 1D pointers").

(a) the documented case: a 2-D region argument handed to a kernel.
(b) the check's own admitted false positive: the top has a 2-D argument, and a
    nested function takes a 2-D *local* array that never touches the top-level
    pointer. Nothing about (b) needs linearizing, yet it is rejected too.
(c) control: the same as (a) with flat arguments.
Codegen only (Vitis is not run)."""
import os
import shutil
import traceback
import _worktree
from _worktree import verdict

ITEM = 14
import allo
import allo.dataflow as df
from allo.ir.types import int32

PRJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_item14.prj")


@df.region()
def top2d(A: int32[4, 4], B: int32[4, 4]):
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: int32[4, 4], b: int32[4, 4]):
        for i, j in allo.grid(4, 4):
            b[i, j] = a[i, j] + 1


@df.region()
def top1d(A: int32[16], B: int32[16]):
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: int32[16], b: int32[16]):
        for i in range(16):
            b[i] = a[i] + 1


def helper(t: int32[2, 2]) -> int32:
    return t[0, 0] + t[1, 1]


def plain_local2d(A: int32[4, 4]) -> int32:
    loc: int32[2, 2] = 0
    loc[0, 0] = A[0, 0]
    loc[1, 1] = A[3, 3]
    return helper(loc)


def emit(build):
    try:
        mod = build()
        code = mod.hls_code if hasattr(mod, "hls_code") else str(mod)
        return "ok" if code else "empty"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        return f"{type(e).__name__}: {msg[:110]}"
    finally:
        shutil.rmtree(PRJ, ignore_errors=True)


def main():
    res = {
        "(a) 2-D region arg -> kernel": emit(lambda: df.build(top2d, target="vitis_hls", mode="csyn", project=PRJ, wrap_io=False)),
        "(b) 2-D local -> nested fn": emit(lambda: allo.customize(plain_local2d).build(target="vitis_hls", mode="csyn", project=PRJ, wrap_io=False)),
        "(c) flat args (control)": emit(lambda: df.build(top1d, target="vitis_hls", mode="csyn", project=PRJ, wrap_io=False)),
    }
    for k, v in res.items():
        print(f"  {k}: {v}")
    verdict(ITEM, res["(a) 2-D region arg -> kernel"] != "ok", "; ".join(f"{k}={v[:40]}" for k, v in res.items()))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
