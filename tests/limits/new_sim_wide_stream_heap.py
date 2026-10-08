# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track A, finding T-1 (dev/records/minitpu/u4_track_a_2026-10-08.rst): a
Stream whose element is wider than 128 bits corrupts the simulator's heap.

The data arrive intact (4,096/4,096), but the process then dies or hangs
("corrupted size vs. prev_size", "free(): invalid pointer", SIGSEGV in a later
malloc, or a deadlock in malloc's lock inside LLVM's signal handler). At 128
bits every run is clean; at 129-160 bits about half are not. Each width runs
in its own child, several times, since the corruption is intermittent."""
import os
import subprocess
import sys

import numpy as np
import _worktree
from _worktree import verdict

ITEM = "new-sim-wide-stream-heap"
N = 4096


def child(w):
    import allo.dataflow as df
    from allo.ir.types import Stream, UInt

    @df.region()
    def top(A: UInt(32)[N, 5], B: UInt(32)[N, 5]):
        s: Stream[UInt(w), 4]

        @df.kernel(mapping=[1], args=[A])
        def prod(a: UInt(32)[N, 5]):
            for i in range(N):
                v: UInt(w) = 0
                v[0:32] = a[i, 0]
                v[32:64] = a[i, 1]
                v[64:96] = a[i, 2]
                v[96:128] = a[i, 3]
                if w > 128:
                    v[128:w] = a[i, 4]
                s.put(v)

        @df.kernel(mapping=[1], args=[B])
        def cons(b: UInt(32)[N, 5]):
            for i in range(N):
                v: UInt(w) = s.get()
                b[i, 0] = v[0:32]
                b[i, 1] = v[32:64]
                b[i, 2] = v[64:96]
                b[i, 3] = v[96:128]

    mod = df.build(top, target="simulator")
    a = np.random.default_rng(0).integers(0, 1 << 32, (N, 5), dtype=np.uint64).astype(np.uint32)
    b = np.zeros((N, 5), dtype=np.uint32)
    mod(a, b)
    print("INTACT" if (a[:, :4] == b[:, :4]).all() else "WRONG", flush=True)


def main():
    bad = {}
    for w in (128, 141):
        bad[w] = 0
        for _ in range(5):
            try:
                r = subprocess.run([sys.executable, __file__, "--child", str(w)], capture_output=True, text=True,
                                   timeout=90, env=dict(os.environ, ALLO_LIMITS_CHILD="1"))
            except subprocess.TimeoutExpired:  # a hang in malloc's lock is one of the failure modes
                bad[w] += 1
                continue
            err = r.stderr + r.stdout
            if r.returncode != 0 or "corrupted" in err or "invalid pointer" in err or "INTACT" not in r.stdout:
                bad[w] += 1
    print({f"UInt({w})": f"{k}/5 runs abnormal" for w, k in bad.items()})
    verdict(ITEM, bad[141] > 0 and bad[128] == 0, str(bad))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(int(sys.argv[2]))
    else:
        main()
