# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""bf16 on Allo's paths: LLVM and the dataflow simulator work; the Vivado emitter aborts."""
import sys, numpy as np, ml_dtypes, allo
from allo.ir.types import bfloat16, float32, int32
import allo.dataflow as df
from allo.ir.types import Stream

N = 8
# 1. plain (non-dataflow) kernel: add, mul, extf/truncf
def k1(a: bfloat16[N], b: bfloat16[N]) -> bfloat16[N]:
    c: bfloat16[N] = 0.0
    for i in range(N):
        p: float32 = a[i]
        q: float32 = b[i]
        c[i] = a[i] * b[i] + a[i]
    return c

s = allo.customize(k1)
#print(s.module)
a = (np.random.randn(N)).astype(ml_dtypes.bfloat16)
b = (np.random.randn(N)).astype(ml_dtypes.bfloat16)
try:
    mod = s.build()
    out = mod(a, b)
    print("LLVM out", out, "gold", (a * b + a))
except Exception as e:
    print("LLVM build/run FAILED:", type(e).__name__, str(e)[:400])
try:
    if "--vhls" not in sys.argv:
        raise RuntimeError("skipped: the emitter ABORTS the process on bf16; pass --vhls to see it")
    code = str(s.build(target="vhls"))
    print("---- VHLS ----"); print(code[-2500:])
except Exception as e:
    print("VHLS FAILED:", type(e).__name__, str(e)[:400])

# 2. dataflow region with a bf16 stream, on the simulator
@df.region()
def top(A: bfloat16[N], B: bfloat16[N]):
    q: Stream[bfloat16, 4]
    @df.kernel(mapping=[1], args=[A])
    def p(lA: bfloat16[N]):
        for i in range(N):
            q.put(lA[i])
    @df.kernel(mapping=[1], args=[B])
    def c(lB: bfloat16[N]):
        for i in range(N):
            v: bfloat16 = q.get()
            lB[i] = v * v
try:
    sim = df.build(top, target="simulator")
    B2 = np.zeros(N, ml_dtypes.bfloat16)
    sim(a, B2)
    print("SIM out", B2, "gold", a * a)
except Exception as e:
    print("SIM FAILED:", type(e).__name__, str(e)[:600])
