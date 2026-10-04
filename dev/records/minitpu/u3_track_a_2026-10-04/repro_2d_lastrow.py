"""Repro: a 2-D uint16 output's last row read back from SystemC csim."""
import sys
import numpy as np
import allo.dataflow as df
from allo.ir.types import uint16, uint8
for n in (16, 60660, 19861):
    @df.region()
    def top(X: uint8[n], O: uint16[n, 4]):
        @df.kernel(mapping=[1], args=[X, O])
        def k(x: uint8[n], o: uint16[n, 4]):
            for t in range(n):
                v: uint8 = x[t]
                for s in range(4):
                    o[t, s] = (t + 1) * 4 + s + v
    m = df.build(top, target="systemc", mode="csim", project=f"{sys.argv[1]}/u3a_repro_2d_{n}")
    x = np.zeros(n, dtype=np.uint8); o = np.zeros((n, 4), dtype=np.uint16)
    m(x, o)
    want = ((np.arange(n)[:, None] + 1) * 4 + np.arange(4)[None, :]).astype(np.uint16)
    bad = np.flatnonzero((o != want).any(axis=1))
    print(f"n={n}: {'ok' if len(bad) == 0 else f'{len(bad)} bad rows, first {bad[:3].tolist()}, last row got {o[-1].tolist()} want {want[-1].tolist()}'}", flush=True)
