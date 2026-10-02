"""Closer-shaped expressions of vpu_bf16_add for synthesis (synth_top=add_0)."""
import allo.dataflow as df
from allo.ir.types import bfloat16, Stream, Wire, Channel, valid_ready


def stream(n):
    @df.region()
    def top(A: bfloat16[n], B: bfloat16[n], C: bfloat16[n]):
        sa: Stream[bfloat16, 2]
        sb: Stream[bfloat16, 2]
        sc: Stream[bfloat16, 2]

        @df.kernel(mapping=[1], args=[A, B])
        def src(a: bfloat16[n], b: bfloat16[n]):
            for i in range(n):
                sa.put(a[i])
                sb.put(b[i])

        @df.kernel(mapping=[1], args=[])
        def add():
            for _ in range(n):
                sc.put(sa.get() + sb.get())

        @df.kernel(mapping=[1], args=[C])
        def sink(c: bfloat16[n]):
            for i in range(n):
                c[i] = sc.get()
    return top


def wire(n):
    @df.region()
    def top(A: bfloat16[n], B: bfloat16[n], C: bfloat16[n]):
        wa: Wire[bfloat16]
        wb: Wire[bfloat16]
        wc: Wire[bfloat16]

        @df.kernel(mapping=[1], args=[A, B])
        def src(a: bfloat16[n], b: bfloat16[n]):
            for i in range(n):
                wa.put(a[i])
                wb.put(b[i])

        @df.kernel(mapping=[1], args=[])
        def add():
            for _ in range(n):
                wc.put(wa.get() + wb.get())

        @df.kernel(mapping=[1], args=[C])
        def sink(c: bfloat16[n]):
            for i in range(n):
                c[i] = wc.get()
    return top


def channel(n):
    @df.region()
    def top(A: bfloat16[n], B: bfloat16[n], C: bfloat16[n]):
        ca: Channel[bfloat16, valid_ready]
        cb: Channel[bfloat16, valid_ready]
        cc: Channel[bfloat16, valid_ready]

        @df.kernel(mapping=[1], args=[A, B])
        def src(a: bfloat16[n], b: bfloat16[n]):
            for i in range(n):
                ca.put(a[i])
                cb.put(b[i])

        @df.kernel(mapping=[1], args=[])
        def add():
            for _ in range(n):
                cc.put(ca.get() + cb.get())

        @df.kernel(mapping=[1], args=[C])
        def sink(c: bfloat16[n]):
            for i in range(n):
                c[i] = cc.get()
    return top
