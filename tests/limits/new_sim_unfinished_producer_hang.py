# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track B (dev/records/minitpu/u4_track_b_2026-10-08.rst, F-B4): the
dataflow simulator hangs, with no diagnostic, when a consumer finishes while a
producer is still blocked on a full stream.

A consumer that polls ``empty()`` for a fixed number of iterations (the
self-timed write-port merge of D-24, in miniature) ends; the producer still
has tokens to ``put`` into a depth-2 stream nobody reads any more, blocks, and
the region call never returns. SystemC csim of the same region terminates
(its testbench stops at the last output: a hung producer is invisible there),
so the two backends disagree on what the region means; Allo checks nothing
(``deadlock_free_because`` is not asked for: the netlist is acyclic).

The run is made in a child process with a timeout; REPRODUCES when it times
out. Expected after a fix: either the simulator returns (a finished region
whose producers are blocked is reported as an error naming the stream) or the
front end refuses a consumer whose trip count cannot drain its inputs."""
import multiprocessing as mp

import numpy as np
import _worktree
from _worktree import verdict

ITEM = "new-sim-unfinished-producer-hang"
TIMEOUT = 60


def child(q):
    import allo.dataflow as df
    from allo.ir.types import Stream, int32

    @df.region()
    def top(A: int32[8], B: int32[2]):
        s: Stream[int32, 2]

        @df.kernel(mapping=[1], args=[A])
        def prod(a: int32[8]):
            for i in range(8):
                s.put(a[i])  # 8 tokens

        @df.kernel(mapping=[1], args=[B])
        def cons(b: int32[2]):
            for i in range(2):  # reads at most 2: the producer blocks on the 5th put
                x: int32 = 0
                if not s.empty():
                    x = s.get()
                b[i] = x

    mod = df.build(top, target="simulator")
    b = np.zeros(2, dtype=np.int32)
    mod(np.arange(8, dtype=np.int32), b)
    q.put(f"returned, B={b.tolist()}")


def main():
    q = mp.Queue()
    p = mp.Process(target=child, args=(q,))
    p.start()
    p.join(TIMEOUT)
    hung = p.is_alive()
    if hung:
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
    detail = f"no return after {TIMEOUT} s (killed)" if hung else (q.get() if not q.empty() else f"exit {p.exitcode}")
    verdict(ITEM, hung, detail)


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
