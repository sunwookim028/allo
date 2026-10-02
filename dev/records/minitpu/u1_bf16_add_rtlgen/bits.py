import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../examples/minitpu/harness"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[1]); os.environ["BF_N"] = str(N)
CHUNKS = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
import stimulus, ref
from bits_kernel import bf16_add_bits
rtl = bf16_add_bits.schedule().export("rtl")
res = rtl.schedule(); f = res.func("bf16_add_bits")
print("SCHED latency:", f.latency, "| II:", [r.interval for r in res.cyclic()])
open(f"bits_N{N}.sv", "w").write(rtl.verilog)
q = rtl.estimation
print("QOR: lat", q.latency, "fmax %.1f" % q.fmax, "area", q.area)
print("CRIT:", q.critical_paths[0] if q.critical_paths else None)
st = stimulus.binary_bf16(); m = len(st); pad = (-m) % N
st = np.concatenate([st, np.zeros((pad, 2), np.uint16)])
got = np.zeros(len(st), np.uint16); cyc = []; t0 = time.time()
nch = min(len(st) // N, CHUNKS)
for c in range(nch):
    a = st[c*N:(c+1)*N, 0].copy(); b = st[c*N:(c+1)*N, 1].copy(); o = np.zeros(N, np.uint16)
    cyc.append(rtl.cosim(a, b, o).cycles); got[c*N:(c+1)*N] = o
n = min(nch * N, m)
A, B, G = st[:n, 0], st[:n, 1], got[:n]
want = ref.vpu_bf16_add(A, B)
mm = np.nonzero(G != want)[0]
print(f"COSIM cycles/chunk={set(cyc)} time={time.time()-t0:.1f}s n={n} match={n-len(mm)} mismatch={len(mm)}")
for i in mm[:10]: print("  ", hex(A[i]), hex(B[i]), "got", hex(G[i]), "want", hex(want[i]))
np.save(f"bits_got.npy", G)
