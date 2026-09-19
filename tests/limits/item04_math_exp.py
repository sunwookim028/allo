# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 4: `math.exp` / `math.log` are not recognized by the AST builder.

Original claim: inside `@df.kernel` bodies `math.exp(x)` raises
`KeyError: 'exp'`; `allo.exp` must be used instead.

Checked with the plain LLVM backend so the result isolates the FRONTEND: the
dataflow simulator cannot run `allo.exp` either, for an unrelated reason (see
new_sim_math_lowering.py)."""
import math  # noqa: F401  (used inside the kernel bodies)
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 4
import allo
import allo.dataflow as df
from allo.ir.types import float32


def with_math(a: float32[4]) -> float32[4]:
    b: float32[4]
    for i in range(4):
        b[i] = math.exp(a[i]) + math.log(a[i])
    return b


def with_allo(a: float32[4]) -> float32[4]:
    b: float32[4]
    for i in range(4):
        b[i] = allo.exp(a[i]) + allo.log(a[i])
    return b


@df.region()
def df_math(A: float32[4], B: float32[4]):
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: float32[4], b: float32[4]):
        for i in range(4):
            b[i] = math.exp(a[i])


def run_plain(fn):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        out = allo.customize(fn).build()(a)
        return "ok" if np.allclose(out, np.exp(a) + np.log(a), rtol=1e-5) else "WRONG"
    except (Exception, SystemExit) as e:  # noqa: BLE001  (customize() sys.exit(1)s on frontend errors)
        traceback.print_exc()
        return f"{type(e).__name__}"


def main():
    res = {"allo.exp (llvm)": run_plain(with_allo), "math.exp (llvm)": run_plain(with_math)}
    try:
        df.customize(df_math)
        res["math.exp (df.kernel, frontend)"] = "ok"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        res["math.exp (df.kernel, frontend)"] = f"{type(e).__name__}"
    print(res)
    verdict(ITEM, res["math.exp (llvm)"] != "ok", str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
