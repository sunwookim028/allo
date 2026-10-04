"""Repro: several outputs per iteration; is the LAST iteration's store to the 2-D array kept in SystemC csim?"""
import sys
import numpy as np
import allo.dataflow as df
from allo.ir.types import uint16, uint8
n = 40
def mk(order):
    @df.region()
    def top(X: uint8[n], A: uint8[n], B: uint16[n], C: uint8[n], D: uint16[n, 4]):
        @df.kernel(mapping=[1], args=[X, A, B, C, D])
        def k(x: uint8[n], a: uint8[n], b: uint16[n], c: uint8[n], d: uint16[n, 4]):
            pipe: uint16[3, 4]
            for s in range(4):
                pipe[0, s] = 0
                pipe[1, s] = 0
                pipe[2, s] = 0
            for t in range(n):
                v: uint8 = x[t]
                for s in range(4):
                    pipe[2, s] = pipe[1, s]
                    pipe[1, s] = pipe[0, s]
                    pipe[0, s] = (t + 1) * 4 + s + v
                if order == 0:
                    a[t] = v
                    b[t] = t
                    c[t] = v
                    for s in range(4):
                        d[t, s] = pipe[2, s]
                else:
                    for s in range(4):
                        d[t, s] = pipe[2, s]
                    a[t] = v
                    b[t] = t
                    c[t] = v
    return top
for order in (0, 1):
    m = df.build(mk(order), target="systemc", mode="csim", project=f"{sys.argv[1]}/u3a_repro_multi_{order}")
    x = np.zeros(n, dtype=np.uint8); a = np.zeros(n, np.uint8); b = np.zeros(n, np.uint16); c = np.zeros(n, np.uint8); d = np.zeros((n, 4), np.uint16)
    m(x, a, b, c, d)
    want = np.zeros((n, 4), np.uint16)
    for t in range(2, n): want[t] = (t - 1) * 4 + np.arange(4)
    bad = np.flatnonzero((d != want).any(axis=1))
    print(f"2-D written {'last' if order == 0 else 'first'}: {'ok' if len(bad) == 0 else f'{len(bad)} bad rows {bad.tolist()[:4]}; last row got {d[-1].tolist()} want {want[-1].tolist()}'}; b[-1]={b[-1]}", flush=True)
