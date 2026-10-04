import allo.dataflow as df
from allo.ir.types import int32, Stream
import numpy as np
N = 8
@df.unit()
def produce(dst: Stream[int32, 4], mem: int32[N]):
    for i in range(N):
        dst.put(mem[i])
@df.unit()
def consume(src: Stream[int32, 4], mem: int32[N]):
    for i in range(N):
        mem[i] = src.get() + 1
# (d) @df.unit netlist: a declared stream nothing wires
try:
    @df.region()
    def unit_dangling(A: int32[N], B: int32[N]):
        a: Stream[int32, 4]
        dangling: Stream[int32, 4]
        produce(dst=a, mem=A)
        consume(src=a, mem=B)
    print("H15(d) @df.unit region, Stream declared and unwired: ACCEPTED silently (bug)")
except Exception as e:
    print("H15(d) @df.unit region unwired stream: refused:", type(e).__name__, str(e).splitlines()[0][:160])
# (e) mixed: a unit netlist plus a nested kernel that writes a stream nobody reads
try:
    @df.region()
    def unit_orphan(A: int32[N], B: int32[N]):
        a: Stream[int32, 4]
        orphan: Stream[int32, 4]
        produce(dst=a, mem=A)
        consume(src=a, mem=B)
        @df.kernel(mapping=[1], args=[A])
        def side(x: int32[N]):
            for i in range(N):
                orphan.put(x[i])
    print("H15(e) @df.unit region, nested kernel writes an unread stream: ACCEPTED silently (bug)")
except Exception as e:
    print("H15(e) @df.unit region orphan writer: refused:", type(e).__name__, str(e).splitlines()[0][:160])
# (f) SystemC emission of the plain written-only region
@df.region()
def written_only(A: int32[N], B: int32[N]):
    orphan: Stream[int32, 4]
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: int32[N], b: int32[N]):
        for i in range(N):
            b[i] = a[i] + 1
            orphan.put(a[i])
import os, tempfile
try:
    d = tempfile.mkdtemp(prefix="u3e_sc_", dir=os.environ.get("U3E_SCRATCH", "/tmp"))
    mod = df.build(written_only, target="systemc", project=d, mode="csim" if False else None) if False else None
    import inspect
    sig = inspect.signature(df.build)
    print("df.build signature:", sig)
except Exception as e:
    print("H15(f) prep:", type(e).__name__, str(e)[:160])
