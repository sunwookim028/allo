# H15 at the front end: a dangling Stream in a plain region, and in a @df.unit netlist
import time, traceback
import allo.dataflow as df
from allo.ir.types import int32, Stream, UInt
import numpy as np
N = 8
# (a) declared, untouched
@df.region()
def untouched(A: int32[N], B: int32[N]):
    dangling: Stream[int32, 4]
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: int32[N], b: int32[N]):
        for i in range(N):
            b[i] = a[i] + 1
try:
    m = df.build(untouched, target="simulator"); a = np.arange(N, dtype=np.int32); b = np.zeros(N, np.int32); m(a, b)
    print("H15(a) plain region, Stream declared and untouched: BUILT AND RAN silently", b.tolist())
except Exception as e:
    print("H15(a) plain region untouched stream: refused:", type(e).__name__, str(e)[:200])
# (b) written, never read
@df.region()
def written_only(A: int32[N], B: int32[N]):
    orphan: Stream[int32, 4]
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: int32[N], b: int32[N]):
        for i in range(N):
            b[i] = a[i] + 1
            orphan.put(a[i])
try:
    m = df.build(written_only, target="simulator")
    print("H15(b) plain region, Stream written never read: BUILT silently (running would block once full: depth 4, 8 puts)")
except Exception as e:
    print("H15(b) plain region written-only stream: refused:", type(e).__name__, str(e)[:200])
# (c) the SystemC emitter on (b)
try:
    s = df.customize(written_only)
    import allo
    code = allo.backend.systemc.emit(s) if hasattr(allo.backend, "systemc") else None
    print("H15(c) systemc emit of written-only:", "emitted" if code else "no emitter api found")
except Exception as e:
    print("H15(c) systemc emit of written-only: refused:", type(e).__name__, str(e)[:200])
