"""Repro: a user constant array whose name starts with 'gelu' is erased on the simulator path."""
import numpy as np
import allo.dataflow as df
from allo.ir.types import int32
ROM = (np.arange(64, dtype=np.int32) * 3) % 17
N = 8
def top_gelu():
    @df.region()
    def top(X: int32[N], R: int32[N]):
        @df.kernel(mapping=[1], args=[X, R])
        def k(x: int32[N], r: int32[N]):
            gelu_tab: int32[64] = ROM
            for i in range(N):
                r[i] = gelu_tab[x[i]]
    return top
def top_other():
    @df.region()
    def top(X: int32[N], R: int32[N]):
        @df.kernel(mapping=[1], args=[X, R])
        def k(x: int32[N], r: int32[N]):
            tab_gelu: int32[64] = ROM
            for i in range(N):
                r[i] = tab_gelu[x[i]]
    return top
for name, mk in (("tab_gelu", top_other), ("gelu_tab", top_gelu)):
    try:
        m = df.build(mk(), target="simulator")
        x = np.arange(N, dtype=np.int32); r = np.zeros(N, dtype=np.int32); m(x, r)
        print(f"constant named {name}: ok" if (r == ROM[x]).all() else f"{name}: WRONG {r}")
    except Exception as e:
        print(f"constant named {name}: FAIL {type(e).__name__}: {' | '.join(l for l in str(e).splitlines() if l.strip())[:260]}")
