# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 11: the simulator deadlocks when kernel instances outnumber
OMP_NUM_THREADS (recorded as FIXED in f193c057).

A 16-stage relay chain over depth-1 streams, run under OMP_NUM_THREADS=2 in a
child with a timeout. Without the fix every stage blocks and the team of two
never starts the stages that would unblock them."""
import os
import subprocess
import sys
import _worktree
from _worktree import verdict

ITEM = 11
THIS = os.path.abspath(__file__)
P = 16


def child():
    import numpy as np
    import allo.dataflow as df
    from allo.ir.types import Stream, int32

    @df.region()
    def top(A: int32[8], B: int32[8]):
        pipe: Stream[int32, 1][P + 1]

        @df.kernel(mapping=[1], args=[A])
        def src(a: int32[8]):
            for i in range(8):
                pipe[0].put(a[i])

        @df.kernel(mapping=[P])
        def stage():
            s = df.get_pid()
            for i in range(8):
                pipe[s + 1].put(pipe[s].get() + 1)

        @df.kernel(mapping=[1], args=[B])
        def sink(b: int32[8]):
            for i in range(8):
                b[i] = pipe[P].get()

    mod = df.build(top, target="simulator")
    a = np.arange(8, dtype=np.int32)
    b = np.zeros(8, dtype=np.int32)
    mod(a, b)
    assert (b == a + P).all(), b
    print("CHILD-OK")


def main():
    if len(sys.argv) > 1:
        child()
        return
    env = dict(os.environ, OMP_NUM_THREADS="2", ALLO_SIM_TIMEOUT="0")
    try:
        p = subprocess.run([sys.executable, THIS, "child"], env=env,
                           capture_output=True, text=True, timeout=int(os.environ.get("ITEM11_TIMEOUT", 180)), check=False)
        ok = "CHILD-OK" in p.stdout
        detail = "18 processes at OMP_NUM_THREADS=2 finished" if ok else p.stderr[-300:]
    except subprocess.TimeoutExpired:
        ok, detail = False, "hung past the timeout at OMP_NUM_THREADS=2"
    verdict(ITEM, not ok, detail)


if __name__ == "__main__":
    main()
