"""Minimal repros for the U3 composed-FIFO record (Allo simulator).

    python repros.py [case ...]

From the worktree root after ``source examples/minitpu/harness/env-zhang21.sh``.
Each case is a two-kernel region with one ``Stream[int32, 4]`` between a
producer and a consumer; a case that does not finish in 45 s is a HANG.
"""
import os, signal, sys, time

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

import allo.dataflow as df  # noqa: E402
from allo.ir.types import Stream, int32, uint1  # noqa: E402


def _handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _handler)


def region(n, poll, pop_every):
    """``prod`` puts a[t] every iteration; ``cons`` gets every ``pop_every``-th
    iteration (so with ``pop_every > 1`` the producer must block on full)."""

    @df.region()
    def top(A: int32[n], B: int32[n], F: uint1[n], E: uint1[n]):
        q: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[A, F])
        def prod(a: int32[n], f: uint1[n]):
            for t in range(n):
                x: int32 = a[t]
                fl: uint1 = 0
                if poll == 1:
                    fl = q.full()
                f[t] = fl
                q.put(x)

        @df.kernel(mapping=[1], args=[B, E])
        def cons(b: int32[n], e: uint1[n]):
            for t in range(n):
                em: uint1 = 0
                if poll == 1:
                    em = q.empty()
                e[t] = em
                y: int32 = 0
                if t % pop_every == 0:
                    y = q.get()
                b[t] = y

    return top


def region_try(n):
    """B8 probe: 6 ``try_put`` into a depth-4 stream before the consumer runs
    (the consumer is held on a second stream the producer releases after its
    puts), then the consumer ``try_get``s 6 times."""

    @df.region()
    def top(A: int32[n], B: int32[n], OK: uint1[n], GOT: uint1[n]):
        q: Stream[int32, 4]
        go: Stream[int32, 1]

        @df.kernel(mapping=[1], args=[A, OK])
        def prod(a: int32[n], ok: uint1[n]):
            for t in range(n):
                x: int32 = a[t]
                o: uint1 = q.try_put(x)
                ok[t] = o
            go.put(1)

        @df.kernel(mapping=[1], args=[B, GOT])
        def cons(b: int32[n], got: uint1[n]):
            g0: int32 = go.get()
            for t in range(n):
                y: int32
                o: uint1
                y, o = q.try_get()
                b[t] = y * o + g0 - 1
                got[t] = o

    return top


def run(label, top, arrays):
    mod = df.build(top, target="simulator")
    signal.alarm(45)
    t0 = time.time()
    try:
        mod(*arrays)
        signal.alarm(0)
        return f"{label}: ok in {time.time() - t0:.1f}s"
    except TimeoutError:
        return f"{label}: HANG (>45 s)"


CASES = {}


def _blocking(n, poll, pop_every):
    m = n if pop_every == 1 else n  # the consumer gets ceil(n / pop_every) words
    A = np.arange(n, dtype=np.int32)
    B, F, E = np.zeros(n, np.int32), np.zeros(n, np.uint8), np.zeros(n, np.uint8)
    # with pop_every > 1 the producer has more puts than the consumer gets:
    # bound the producer to the consumer's count so the case can finish
    nput = (n + pop_every - 1) // pop_every if pop_every > 1 else n
    return A[:nput] if pop_every == 1 else A, B, F, E


def case_put6(_):
    n = 6
    A, B, F, E = _blocking(n, 0, 1)
    r = run("6 put / 6 get, no polling, depth 4", region(n, 0, 1), (A, B, F, E))
    return r + f"; got {B.tolist()}"


def case_put6_poll(_):
    n = 6
    A, B, F, E = _blocking(n, 1, 1)
    r = run("6 put / 6 get, full()/empty() polled, depth 4", region(n, 1, 1), (A, B, F, E))
    return r + f"; got {B.tolist()} full {F.tolist()} empty {E.tolist()}"


def case_put300_poll(_):
    n = 300
    A, B, F, E = _blocking(n, 1, 1)
    r = run("300 put / 300 get, polled", region(n, 1, 1), (A, B, F, E))
    return r + f"; got ok={bool((B == A).all())} full {int(F.sum())}/{n} empty {int(E.sum())}/{n}"


def case_try6(_):
    n = 6
    A = np.arange(1, n + 1, dtype=np.int32)
    B, OK, GOT = np.zeros(n, np.int32), np.zeros(n, np.uint8), np.zeros(n, np.uint8)
    r = run("B8: 6 try_put into depth 4 before any get, then 6 try_get", region_try(n), (A, B, OK, GOT))
    return r + f"; try_put ok {OK.tolist()} (expect [1,1,1,1,0,0]); try_get ok {GOT.tolist()} got {B.tolist()} (expect 4 words then 0)"


CASES = {"put6": case_put6, "put6_poll": case_put6_poll, "put300_poll": case_put300_poll, "try6": case_try6}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(CASES):
        print(CASES[name](None), flush=True)
