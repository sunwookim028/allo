"""Minimal repros of the ``vpu_fifo`` pilot's new findings (U2, D-9).

    source examples/minitpu/harness/env-zhang21.sh
    $ALLO_PYTHON dev/records/minitpu/u2_fifo_2026-10-02/repros.py [B7|S8|S9|M3 ...]

Each prints one line per backend with the observed and the expected values.
Nothing under ``allo/`` or ``mlir/`` is changed; these are evidence for the
proposed fixes in ``u2_fifo_2026-10-02.rst``.
"""
import os, sys, traceback
import numpy as np

sys.path.insert(0, os.getcwd())
import allo.dataflow as df  # noqa: E402
from allo.ir.types import Stream, UInt, int32, uint1  # noqa: E402

PRJ = os.environ.get("REPRO_PRJ", "/tmp/u2_fifo_repros")


def build(top, backend, name):
    if backend == "simulator":
        return df.build(top, target="simulator")
    return df.build(top, target="systemc", mode="csim", project=os.path.join(PRJ, name))


def B7():
    """simulator, silent: a ``try_put`` whose result is unused is dropped."""
    def make(use_result):
        @df.region()
        def top(X: int32[4], E: uint1[4]):
            s: Stream[int32, 4]

            @df.kernel(mapping=[1], args=[X, E])
            def k(x: int32[4], e: uint1[4]):
                for i in range(4):
                    if use_result == 1:
                        ok: uint1 = s.try_put(x[i])
                        e[i] = s.empty() + ok - ok
                    else:
                        junk: uint1 = s.try_put(x[i])
                        e[i] = s.empty()
        return top

    for backend in ("simulator", "systemc"):
        for u in (1, 0):
            e = np.zeros(4, np.uint8)
            build(make(u), backend, f"b7_{u}")(np.arange(4, dtype=np.int32), e)
            print(f"B7 {backend} try_put result {'used' if u else 'UNUSED'}: empty() after each put = {e.tolist()} (expected [0, 0, 0, 0])")


def S8():
    """SystemC csim, silent: a 64-bit port value >= 2^63 and every value after
    it on that port read back as 0x7fffffffffffffff (the testbench parses the
    data file with ``std::ifstream >> long long``: overflow stores LLONG_MAX
    and sets failbit, which sticks)."""
    @df.region()
    def top(X: UInt(64)[4], Y: UInt(64)[4]):
        @df.kernel(mapping=[1], args=[X, Y])
        def k(x: UInt(64)[4], y: UInt(64)[4]):
            for i in range(4):
                y[i] = x[i]

    x = np.array([1, 2**63 - 1, 2**63, 3], dtype=np.uint64)
    for backend in ("simulator", "systemc"):
        y = np.zeros(4, np.uint64)
        build(top, backend, "s8")(x, y)
        print(f"S8 {backend}: in {[hex(int(v)) for v in x]} out {[hex(int(v)) for v in y]} (expected equal)")


def S9():
    """SystemC csim: a self-FIFO's ``full()`` reads a synchronous counter that
    ``try_put`` advances even when the put is refused, so after one refused
    put the FIFO never reports full again (the simulator keeps full = 1)."""
    @df.region()
    def top(F: uint1[6]):
        s: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[F])
        def k(f: uint1[6]):
            for i in range(6):
                ok: uint1 = s.try_put(i)
                f[i] = s.full() + ok - ok

    for backend in ("simulator", "systemc"):
        f = np.zeros(6, np.uint8)
        build(top, backend, "s9")(f)
        print(f"S9 {backend}: full() after try_put #1..6 into Stream[int32, 4] = {f.tolist()} (expected [0, 0, 0, 1, 1, 1])")


def M3():
    """Semantic mismatch, both backends: push and pop together on an EMPTY
    Stream (pop first in program order, blocking-safe) keeps the word; the
    RTL loses it (U2 Phase 0, H12). Shown as the count after the cycle."""
    @df.region()
    def top(E: uint1[2]):
        s: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[E])
        def k(e: uint1[2]):
            for i in range(2):
                em: uint1 = s.empty()
                if em == 0:  # pop if not empty (nothing: it is empty)
                    junk: int32 = s.get()
                v: int32 = i
                s.put(v)  # the push of the same cycle
                e[i] = s.empty()

    for backend in ("simulator", "systemc"):
        e = np.zeros(2, np.uint8)
        build(top, backend, "m3")(e)
        print(f"M3 {backend}: empty() after push+pop on empty = {e.tolist()} (Stream keeps the word: 0; the RTL loses it: 1)")


if __name__ == "__main__":
    want = sys.argv[1:] or ["B7", "S8", "S9", "M3"]
    for name in want:
        try:
            globals()[name]()
        except Exception as ex:  # noqa: BLE001
            print(f"{name} FAILED: {type(ex).__name__}: {str(ex)[:300]}")
            traceback.print_exc(limit=2)
