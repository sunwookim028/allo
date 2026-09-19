# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register item A (docs/source/developer/limitations.rst): the dataflow simulator cannot lower the
`math` dialect, so `allo.exp` / `allo.log` (the documented workaround for
item 4) fail on `target="simulator"` with
`cannot be converted to LLVM IR: ... for op: math.exp`.

The plain LLVM backend (`allo.customize(...).build()`) lowers the same op via
`lower_allo_to_llvm` (populateMathToLLVMConversionPatterns); the simulator's
own pipeline in `LLVMOMPModule.__init__` has no math pass."""
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = "new-sim-math"
import allo
import allo.dataflow as df
from allo.ir.types import float32


@df.region()
def top(A: float32[4], B: float32[4]):
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: float32[4], b: float32[4]):
        for i in range(4):
            b[i] = allo.exp(a[i])


def plain(a: float32[4]) -> float32[4]:
    b: float32[4]
    for i in range(4):
        b[i] = allo.exp(a[i])
    return b


def main():
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    res = {}
    try:
        res["llvm backend"] = "ok" if np.allclose(allo.customize(plain).build()(a), np.exp(a), rtol=1e-5) else "WRONG"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        res["llvm backend"] = f"{type(e).__name__}"
    try:
        mod = df.build(top, target="simulator")
        b = np.zeros(4, dtype=np.float32)
        mod(a, b)
        res["simulator"] = "ok" if np.allclose(b, np.exp(a), rtol=1e-5) else "WRONG"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        res["simulator"] = f"{type(e).__name__}: {str(e)[:60]}"
    print(res)
    verdict(ITEM, res["simulator"] != "ok", str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
