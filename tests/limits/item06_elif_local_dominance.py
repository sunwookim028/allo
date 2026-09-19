# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 6: a fresh local declared inside an `elif` arm.

Original claim: `eff_d: int32 = rs1_lo + d_off` inside one `elif` arm and
`new_idx: int32 = iter_idx[0] + 1` inside another produced MLIR that did not
verify (dominance / null-Value errors). Exercised in the three shapes the
claim could mean: (a) fresh local used only in its own arm, one per arm;
(b) the same name declared in two arms; (c) a name declared in one arm and
READ in a later arm (the "referencing it in another branch" wording)."""
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 6
import allo.dataflow as df
from allo.ir.types import Stateful, int32


def build_a():
    @df.region()
    def top(cmd: int32[4], out: int32[2]):
        iter_idx: int32[1] @ Stateful = 0

        @df.kernel(mapping=[1], args=[cmd, out])
        def dec(c: int32[4], o: int32[2]):
            for t in range(4):
                funct7: int32 = c[t]
                if funct7 == 0:
                    o[0] = o[0] + 1
                elif funct7 == 1:
                    eff_d: int32 = t + 10
                    o[0] = o[0] + eff_d
                elif funct7 == 2:
                    new_idx: int32 = iter_idx[0] + 1
                    iter_idx[0] = new_idx
                    o[1] = new_idx
    return top, np.array([0, 1, 2, 2], dtype=np.int32), [12, 2]


def build_b():
    @df.region()
    def top(cmd: int32[4], out: int32[2]):
        @df.kernel(mapping=[1], args=[cmd, out])
        def dec(c: int32[4], o: int32[2]):
            for t in range(4):
                funct7: int32 = c[t]
                if funct7 == 1:
                    tmp: int32 = t + 10
                    o[0] = o[0] + tmp
                elif funct7 == 2:
                    tmp: int32 = t * 100
                    o[1] = o[1] + tmp
    return top, np.array([1, 2, 1, 2], dtype=np.int32), [10 + 12, 100 + 300]


def build_c():
    @df.region()
    def top(cmd: int32[4], out: int32[2]):
        @df.kernel(mapping=[1], args=[cmd, out])
        def dec(c: int32[4], o: int32[2]):
            for t in range(4):
                funct7: int32 = c[t]
                if funct7 == 1:
                    eff: int32 = t + 10
                    o[0] = eff
                elif funct7 == 2:
                    o[1] = eff  # noqa: F821  -- declared in another arm
    return top, np.array([1, 2, 1, 2], dtype=np.int32), None


def run(builder):
    top, cmd, expect = builder()
    mod = df.build(top, target="simulator")
    out = np.zeros(2, dtype=np.int32)
    mod(cmd, out)
    return out.tolist(), expect


def main():
    res = {}
    for name, b in (("a", build_a), ("b", build_b), ("c", build_c)):
        try:
            got, exp = run(b)
            res[name] = "ok" if exp is None or got == exp else f"WRONG {got} != {exp}"
            if exp is None:
                res[name] = f"built+ran, out={got} (Python would raise NameError/UnboundLocal)"
        except (Exception, SystemExit) as e:  # noqa: BLE001
            traceback.print_exc()
            res[name] = f"{type(e).__name__}: {str(e)[:150]}"
    print(res)
    # The claim is about (a)/(b): legal Python that should compile.
    bad = res["a"] != "ok" or res["b"] != "ok"
    verdict(6, bad, str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
