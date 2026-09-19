# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 17: "Frontend constraints worth documenting" -- five bullets, each
checked on the simulator.

 a. a runtime stream-array subscript fails with "Fail to resolve the
    expression as symbolic expression"
 b. nested `meta_for` over a 2-D stream array fails where 1-D is fine
 c. names bound inside `meta_if` are not visible after it
 d. runtime loop bounds work in a df.kernel, with stream ops in the body
    (a capability, claimed undocumented)
 e. `df.build` is `customize(func)` + `s.build(...)`, so schedule primitives
    are reachable (a capability, claimed undocumented)"""
import os
import subprocess
import sys
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 17
THIS = os.path.abspath(__file__)
import allo
import allo.dataflow as df
from allo.ir.types import Stream, int32

T = 2


def case_a():
    @df.region()
    def top(sel: int32[1], out: int32[1]):
        s: Stream[int32, 2][2]

        @df.kernel(mapping=[1], args=[sel])
        def p(x: int32[1]):
            s[x[0]].put(5)  # runtime subscript

        @df.kernel(mapping=[1], args=[out])
        def c(o: int32[1]):
            o[0] = s[1].get()

    mod = df.build(top, target="simulator")
    o = np.zeros(1, dtype=np.int32)
    mod(np.array([1], dtype=np.int32), o)
    return o[0] == 5


def case_b():
    @df.region()
    def top(A: int32[T * T], B: int32[T * T]):
        s: Stream[int32, 2][T, T]

        @df.kernel(mapping=[1], args=[A])
        def p(a: int32[T * T]):
            with allo.meta_for(T) as i:
                with allo.meta_for(T) as j:
                    s[i, j].put(a[i * T + j])

        @df.kernel(mapping=[1], args=[B])
        def c(b: int32[T * T]):
            with allo.meta_for(T) as i:
                with allo.meta_for(T) as j:
                    b[i * T + j] = s[i, j].get() + 1

    mod = df.build(top, target="simulator")
    a = np.arange(T * T, dtype=np.int32)
    b = np.zeros(T * T, dtype=np.int32)
    mod(a, b)
    return (b == a + 1).all()


def case_c():
    @df.region()
    def top(out: int32[T]):
        @df.kernel(mapping=[T], args=[out])
        def k(o: int32[T]):
            i = df.get_pid()
            with allo.meta_if(i == 0):
                a: int32 = 10
            with allo.meta_else():
                a: int32 = 20
            o[i] = a  # bound inside meta_if, used after

    mod = df.build(top, target="simulator")
    o = np.zeros(T, dtype=np.int32)
    mod(o)
    return o.tolist() == [10, 20]


def case_d():
    @df.region()
    def top(n: int32[1], A: int32[8], B: int32[8]):
        s: Stream[int32, 2]

        @df.kernel(mapping=[1], args=[n, A])
        def p(nn: int32[1], a: int32[8]):
            for i in range(nn[0]):
                s.put(a[i])

        @df.kernel(mapping=[1], args=[n, B])
        def c(nn: int32[1], b: int32[8]):
            for i in range(nn[0]):
                b[i] = s.get() * 2

    mod = df.build(top, target="simulator")
    a = np.arange(8, dtype=np.int32)
    b = np.zeros(8, dtype=np.int32)
    mod(np.array([5], dtype=np.int32), a, b)
    return b.tolist() == [0, 2, 4, 6, 8, 0, 0, 0]


def case_e():
    import inspect

    @df.region()
    def top(A: int32[8], B: int32[8]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: int32[8], b: int32[8]):
            buf: int32[8] = 0
            for i in range(8):
                buf[i] = a[i]
            for i in range(8):
                b[i] = buf[i] + 1

    s = df.customize(top)
    s.partition("k_0:buf", dim=1, factor=2)  # a schedule primitive on the df path
    code = str(s.build(target="vhls"))
    src = inspect.getsource(df.build)
    return ("array_partition" in code and "s = customize(func" in src
            and "s.build(" in src)


CASES = {"a": case_a, "b": case_b, "c": case_c, "d": case_d, "e": case_e}
# a/b/c are constraints (True == the constraint is gone); d/e are capabilities.
KIND = {"a": "constraint", "b": "constraint", "c": "constraint",
        "d": "capability", "e": "capability"}


def main():
    if len(sys.argv) > 1:
        try:
            print("CASE", "ok" if CASES[sys.argv[1]]() else "WRONG")
        except (Exception, SystemExit) as e:  # noqa: BLE001
            traceback.print_exc()
            print("CASE", f"{type(e).__name__}: {str(e).splitlines()[0][:110] if str(e) else ''}")
        return
    res = {}
    for k in CASES:
        p = subprocess.run([sys.executable, THIS, k], capture_output=True, text=True,
                           check=False, timeout=600)
        line = next((l for l in p.stdout.splitlines() if l.startswith("CASE")), None)
        if line is None:
            errs = [l for l in p.stderr.splitlines() if "Error" in l]
            line = "CASE " + (errs[-1][:110] if errs else f"died rc={p.returncode}")
        res[k] = line[5:]
        print(f"  17{k} ({KIND[k]}): {res[k]}")
    still = [k for k in "abc" if res[k] != "ok"]
    verdict(ITEM, bool(still), f"constraints still present: {still}; capabilities d={res['d']} e={res['e']}")


if __name__ == "__main__":
    main()
