import os
import sys, time, numpy as np, ml_dtypes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../examples/minitpu/harness"))
import stimulus, ref
from allo import kernel
from allo.lang import bf16

N = int(sys.argv[1]); CHUNKS = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9

@kernel
def bfadd_n(a: bf16[N], b: bf16[N], out: bf16[N]):
    for k in range(N, name="k"):
        out[k] = a[k] + b[k]

rtl = bfadd_n.schedule().export("rtl")
st = stimulus.binary_bf16()
m = len(st); pad = (-m) % N
st = np.concatenate([st, np.zeros((pad, 2), np.uint16)])
got = np.zeros(len(st), np.uint16)
t0 = time.time(); cyc = []
for c in range(min(len(st) // N, CHUNKS)):
    a = st[c*N:(c+1)*N, 0].copy().view(ml_dtypes.bfloat16)
    b = st[c*N:(c+1)*N, 1].copy().view(ml_dtypes.bfloat16)
    o = np.zeros(N, ml_dtypes.bfloat16)
    r = rtl.cosim(a, b, o)
    cyc.append(r); got[c*N:(c+1)*N] = o.view(np.uint16)
n = min(len(st), CHUNKS * N); n = min(n, m)
print("cosim results (first 2):", cyc[:2], "time %.1fs" % (time.time() - t0))
A, B, G = st[:n, 0], st[:n, 1], got[:n]
want = ref.vpu_bf16_add(A, B); ieee = ref.ieee_bf16_add(A, B)
mm = G != want
print(f"n={n} match_vs_minitpu={n-mm.sum()} mismatch={mm.sum()} match_vs_ieee={(G==ieee).sum()}")
idx = np.nonzero(mm)[0]
from collections import Counter
def cls(x):
    e=(x>>7)&0xFF; f=x&0x7F
    return "nan" if e==0xFF and f else "inf" if e==0xFF else "zero" if (x&0x7FFF)==0 else "sub" if e==0 else "norm"
print(Counter((cls(A[i]),cls(B[i]),hex(G[i]),hex(want[i])) for i in idx).most_common(12))
np.save("native_got.npy", got[:n])
