import os; os.environ.setdefault("OMP_NUM_THREADS", "16")
import sys
import allo
from allo.ir.types import int32, int1, Stream, Wire, Channel, valid_ready
import allo.dataflow as df
import numpy as np

# =====================================================================================
# THE MODULARITY TAX, AND WHETHER A WIRE REMOVES IT.
#
# One computation -- a running dot product, C[i] = sum_{j<=i} A[j]*B[j] -- expressed
# three ways.  The arithmetic is identical in all three; only the STRUCTURE differs:
#
#   mono    one kernel does multiply AND accumulate           (no boundary at all)
#   wire    TWO kernels, multiply -> Wire -> accumulate       (combinational boundary)
#   stream  TWO kernels, multiply -> Stream[int32,2] -> acc   (buffered boundary)
#
# THE QUESTION.  Before Wire and Channel existed, splitting a PE into submodules ALWAYS
# cost a FIFO: every kernel boundary in Allo had to become a Stream, so modularity was
# taxed in both area (a buffer you did not ask for) and latency (at least one cycle).
# That tax is why our fused router beat the split one -- 25 crossbar FIFOs existed purely
# because a kernel boundary is a link.
#
# A Wire is supposed to make the boundary free: combinational, no storage, no handshake,
# no cycle of latency.  If that holds, `wire` should be indistinguishable from `mono` in
# results while being two independently-written modules -- which is exactly the
# modularity fix we want.  `stream` is the control showing what the boundary used to cost.
#
# WHAT TO COMPARE.  All three must produce the SAME C.  The interesting differences are
# structural: number of kernels, whether a buffer exists, and (once synthesis works) area
# and latency.  Same-results-different-structure is the whole point -- if the wire version
# disagreed with mono, the boundary would not be free after all.
#
# NOTE ON RUNNABILITY.  mono and stream run on the JIT simulator AND on SystemC csim;
# `wire` is SystemC-only, because the simulator has no Wire support whatsoever.  So the
# three-way comparison has to be done on csim.
#
# HARNESS SHAPE MATTERS.  Each kernel owns AT MOST ONE host array.  A first attempt gave
# `mono` three arrays and `mul` two, and both HUNG under csim while the JIT simulator ran
# them fine -- every shape proven to work on csim has one array per kernel (drv owns inj,
# col owns dlv).  So A and B are packed into one 2xN array fed by a `feed` kernel, results
# leave via a `sink` kernel, and the compute kernels take args=[] -- links only, which is
# the router shape that works.  All three variants share that harness and differ ONLY at
# the mul->acc boundary, which is what makes the comparison controlled.
#
# WHY THIS SHAPE SHOULD DODGE THE WIRE-LOOP HANG.  A wire-ONLY loop body currently gets no
# wait() from the emitter and deadlocks in zero simulated time (see switch_comb.py).  Here
# every kernel also touches a host array, and memory-port access is blocking, so it
# advances simulated time on its own -- the same accident that let router_rvn_wire.py run.
# =====================================================================================

N = 8


# ── VARIANT 1: monolithic compute. feed -> [pe does mul AND acc] -> sink ──
@df.region()
def pe_mono(AB: int32[2, N], C: int32[N]):
    fa: Stream[int32, 2]
    fb: Stream[int32, 2]
    res: Stream[int32, 2]

    @df.kernel(mapping=[1], args=[AB])
    def feed(ab: int32[2, N]):
        for i in range(N):
            fa.put(ab[0, i])
            fb.put(ab[1, i])

    @df.kernel(mapping=[1], args=[])
    def pe():
        s: int32 = 0
        for i in range(N):
            x: int32 = fa.get()
            y: int32 = fb.get()
            p: int32 = x * y          # multiply
            s += p                    # accumulate -- SAME kernel, no boundary
            res.put(s)

    @df.kernel(mapping=[1], args=[C])
    def sink(c: int32[N]):
        for i in range(N):
            c[i] = res.get()


# ── VARIANT 2: mul and acc are SEPARATE modules, joined by a WIRE ──
@df.region()
def pe_wire(AB: int32[2, N], C: int32[N]):
    fa: Stream[int32, 2]
    fb: Stream[int32, 2]
    prod: Wire[int32]                 # <- the boundary under test: 0 storage
    res: Stream[int32, 2]

    @df.kernel(mapping=[1], args=[AB])
    def feed(ab: int32[2, N]):
        for i in range(N):
            fa.put(ab[0, i])
            fb.put(ab[1, i])

    @df.kernel(mapping=[1], args=[])
    def mul():
        for i in range(N):
            x: int32 = fa.get()
            y: int32 = fb.get()
            p: int32 = x * y
            prod.put(p)               # written every cycle -- a wire cannot be skipped

    @df.kernel(mapping=[1], args=[])
    def acc():
        s: int32 = 0
        for i in range(N):
            p: int32 = prod.get()     # read every cycle, unconditionally
            s += p
            res.put(s)

    @df.kernel(mapping=[1], args=[C])
    def sink(c: int32[N]):
        for i in range(N):
            c[i] = res.get()


# ── VARIANT 3: the same split, but the boundary costs a FIFO ──
@df.region()
def pe_stream(AB: int32[2, N], C: int32[N]):
    fa: Stream[int32, 2]
    fb: Stream[int32, 2]
    prod: Stream[int32, 2]            # <- the tax: storage nobody asked for
    res: Stream[int32, 2]

    @df.kernel(mapping=[1], args=[AB])
    def feed(ab: int32[2, N]):
        for i in range(N):
            fa.put(ab[0, i])
            fb.put(ab[1, i])

    @df.kernel(mapping=[1], args=[])
    def mul():
        for i in range(N):
            x: int32 = fa.get()
            y: int32 = fb.get()
            p: int32 = x * y
            prod.put(p)

    @df.kernel(mapping=[1], args=[])
    def acc():
        s: int32 = 0
        for i in range(N):
            p: int32 = prod.get()
            s += p
            res.put(s)

    @df.kernel(mapping=[1], args=[C])
    def sink(c: int32[N]):
        for i in range(N):
            c[i] = res.get()


# ── VARIANT 4: the same split over a CHANNEL. Handshake, but no buffer. ──
# The third point on the storage axis: Stream has a FIFO, Wire has neither buffer nor
# handshake, Channel has the handshake WITHOUT the buffer. That middle point is the one
# worth having -- it keeps the synchronisation that makes `stream` correct while dropping
# the storage that makes it expensive, which is exactly what `wire` threw away too much of.
@df.region()
def pe_channel(AB: int32[2, N], C: int32[N]):
    fa: Stream[int32, 2]
    fb: Stream[int32, 2]
    prod: Channel[int32, valid_ready]   # <- handshake, zero storage
    res: Stream[int32, 2]

    @df.kernel(mapping=[1], args=[AB])
    def feed(ab: int32[2, N]):
        for i in range(N):
            fa.put(ab[0, i])
            fb.put(ab[1, i])

    @df.kernel(mapping=[1], args=[])
    def mul():
        for i in range(N):
            x: int32 = fa.get()
            y: int32 = fb.get()
            p: int32 = x * y
            prod.put(p)                 # blocking: the handshake DOES synchronise

    @df.kernel(mapping=[1], args=[])
    def acc():
        s: int32 = 0
        for i in range(N):
            p: int32 = prod.get()       # blocks until mul pushed -- unlike a Wire
            s += p
            res.put(s)

    @df.kernel(mapping=[1], args=[C])
    def sink(c: int32[N]):
        for i in range(N):
            c[i] = res.get()


VARIANTS = {"mono": pe_mono, "wire": pe_wire, "stream": pe_stream,
            "channel": pe_channel}
KERNELS = {"mono": 3, "wire": 4, "stream": 4, "channel": 4}
BOUNDARY = {"mono": "none (mul+acc in one kernel)",
            "wire": "Wire (0 storage, NO handshake)",
            "stream": "Stream[int32,2] (FIFO, 2 slots)",
            "channel": "Channel[valid_ready] (0 storage, handshake)"}


def golden(a, b):
    return np.cumsum(a.astype(np.int64) * b.astype(np.int64)).astype(np.int32)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    target = sys.argv[2] if len(sys.argv) > 2 else "systemc"
    names = list(VARIANTS) if which == "all" else [which]

    a = np.arange(1, N + 1, dtype=np.int32)
    b = np.arange(2, 2 * N + 2, 2, dtype=np.int32)
    ab = np.stack([a, b]).astype(np.int32)
    want = golden(a, b)
    print(f"target={target}   A={a}   B={b}\nexpected C={want}\n", flush=True)

    results = {}
    for nm in names:
        if target == "simulator" and nm == "wire":
            print(f"  {nm:<7s} SKIPPED (Wire has no JIT-simulator support)", flush=True)
            continue
        c = np.zeros(N, dtype=np.int32)
        if target == "simulator":
            mod = df.build(VARIANTS[nm], target="simulator")
        else:
            prj = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "csim_out", f"pe_{nm}")
            os.makedirs(prj, exist_ok=True)
            mod = df.build(VARIANTS[nm], target="systemc", mode="csim", project=prj)
        mod(ab, c)
        ok = bool((c == want).all())
        results[nm] = ok
        print(f"  {nm:<7s} kernels={KERNELS[nm]}  boundary={BOUNDARY[nm]:<34s} "
              f"C={c}  {'PASS' if ok else 'FAIL'}", flush=True)

    if len(results) > 1:
        print(f"\n  All variants match the golden result: "
              f"{'YES' if all(results.values()) else 'NO'}")
        print("  => same arithmetic, same harness; only the mul->acc boundary differs.")
        print("     If `wire` matches `mono`, that boundary cost nothing.")
    raise SystemExit(0 if all(results.values()) else 1)
