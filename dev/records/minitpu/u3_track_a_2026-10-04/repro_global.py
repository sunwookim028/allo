"""Minimal repro: a module-level numpy constant array read inside a df.kernel."""
import sys, traceback
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import int32

ROM = (np.arange(64, dtype=np.int32) * 3) % 17
N = 8

def case_customize():
    def k(X: int32[N], R: int32[N]):
        rom: int32[64] = ROM
        for i in range(N):
            R[i] = rom[X[i]]
    s = allo.customize(k)
    m = s.build(target="llvm")
    x = np.arange(N, dtype=np.int32); r = np.zeros(N, dtype=np.int32)
    m(x, r); assert (r == ROM[x]).all(), r; return "ok"

def case_df_kernel_local(target):
    @df.region()
    def top(X: int32[N], R: int32[N]):
        @df.kernel(mapping=[1], args=[X, R])
        def k(x: int32[N], r: int32[N]):
            rom: int32[64] = ROM
            for i in range(N):
                r[i] = rom[x[i]]
    kw = dict(mode="csim", project=f"{sys.argv[1]}/u3a_repro_global_{target}") if target == "systemc" else {}
    m = df.build(top, target=target, **kw)
    x = np.arange(N, dtype=np.int32); r = np.zeros(N, dtype=np.int32)
    m(x, r); assert (r == ROM[x]).all(), r; return "ok"

def case_df_helper(target):
    def lut(i: int32) -> int32:
        rom: int32[64] = ROM
        return rom[i]
    @df.region()
    def top(X: int32[N], R: int32[N]):
        @df.kernel(mapping=[1], args=[X, R])
        def k(x: int32[N], r: int32[N]):
            for i in range(N):
                r[i] = lut(x[i])
    kw = dict(mode="csim", project=f"{sys.argv[1]}/u3a_repro_global_h_{target}") if target == "systemc" else {}
    m = df.build(top, target=target, **kw)
    x = np.arange(N, dtype=np.int32); r = np.zeros(N, dtype=np.int32)
    m(x, r); assert (r == ROM[x]).all(), r; return "ok"

for name, fn in [("customize/llvm", case_customize),
                 ("df kernel-local/simulator", lambda: case_df_kernel_local("simulator")),
                 ("df helper/simulator", lambda: case_df_helper("simulator")),
                 ("df kernel-local/systemc csim", lambda: case_df_kernel_local("systemc")),
                 ("df helper/systemc csim", lambda: case_df_helper("systemc"))]:
    try:
        print(f"{name}: {fn()}", flush=True)
    except Exception as e:
        msg = str(e).splitlines()
        print(f"{name}: FAIL {type(e).__name__}: " + " | ".join(l for l in msg if l.strip())[:400], flush=True)
