# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item P3: a `Stream` whose element is wider than 64 bits and not a multiple
of 8 corrupts the dataflow simulator's heap.

Original claim (branch `tinytpu-align`, increment 2): with the aligned design
at T=8, a `Stream` of `UInt(65)` was exact on its first invocation and then
aborted with glibc's "corrupted size vs. prev_size", or segfaulted, on the
next; 72 and 128 bits ran clean and 96 hung. It was recorded as not reduced to
a minimal repro.

This is the reduction: one region, one producer, one consumer, one stream of
`UInt(W)`, called repeatedly in one process. The simulator lowers a stream to a
`memref<(depth+1) x iW>` (``allo/backend/simulator.py``), and a memref of a
non-byte-multiple integer is allocated by its rounded-down byte size while the
stores are wider, so each `put` writes past the buffer.

Each width is swept in its own child process, because the failure mode is an
abort. Prints one verdict line for the whole sweep.
"""
import os
import subprocess
import sys

import numpy as np
import _worktree
from _worktree import verdict

ITEM = "P3"
import allo.dataflow as df  # noqa: E402
from allo.ir.types import UInt, int32, Stream  # noqa: E402

WIDTHS = [64, 65, 72, 96, 128]
N = 16
CALLS = 8


def run_one(width):
    """Build a region with a Stream[UInt(width)] and call it CALLS times."""
    W = UInt(width)

    @df.region()
    def top(src: int32[N], dst: int32[N]):
        fifo: Stream[W, 4]

        @df.kernel(mapping=[1], args=[src])
        def prod(a: int32[N]):
            for i in range(N):
                v: W = 0
                v[0:32] = a[i]
                fifo.put(v)

        @df.kernel(mapping=[1], args=[dst])
        def cons(b: int32[N]):
            for i in range(N):
                v: W = fifo.get()
                b[i] = v[0:32]

    mod = df.build(top, target="simulator")
    a = np.arange(1, N + 1, dtype=np.int32)
    for c in range(CALLS):
        b = np.zeros(N, dtype=np.int32)
        mod(a, b)
        if not np.array_equal(a, b):
            print(f"WIDTH {width}: WRONG on call {c}: {b.tolist()}", flush=True)
            return
    print(f"WIDTH {width}: OK over {CALLS} calls", flush=True)


def main():
    if os.environ.get("ALLO_P3_WIDTH"):
        run_one(int(os.environ["ALLO_P3_WIDTH"]))
        return
    results = {}
    for w in WIDTHS:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(os.environ, ALLO_P3_WIDTH=str(w), ALLO_LIMITS_CHILD="1"),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        line = [l for l in p.stdout.splitlines() if l.startswith(f"WIDTH {w}:")]
        blob = p.stdout + p.stderr
        glibc = next(
            (
                l.strip()
                for l in blob.splitlines()
                if l.startswith(("corrupted ", "malloc(): ", "free(): ", "double free"))
                or "Segmentation fault" in l
            ),
            None,
        )
        if line and p.returncode == 0:
            results[w] = line[0].split(": ", 1)[1]
        elif line:
            results[w] = (
                f"values exact, then rc={p.returncode}"
                + (f" ({glibc})" if glibc else "")
            )
        else:
            tail = [l for l in p.stderr.splitlines() if l.strip()][-1:] or ["<silent>"]
            results[w] = f"died rc={p.returncode}: {glibc or tail[0][:90]}"
        print(f"  {w:4d} bits -> {results[w]}")
    bad = [w for w, r in results.items() if not r.startswith("OK")]
    verdict(
        ITEM,
        bool(bad),
        "widths that do not survive "
        + f"{CALLS} calls: {bad}; "
        + ", ".join(f"{w}={r}" for w, r in results.items()),
    )


if __name__ == "__main__":
    main()
