"""The D-12 memories through AMC's FRONTEND (allocation infers the ports):
a trace kernel owning the memory as a local array, one host write and one
read per iteration (the fetch / replay side), as f1_d12 / l1_d12 use them.

    python mem_frontend_amc.py <mem: iram|loopbuf> <tgt> <sched> [N]

Builds, dumps AMC's port allocation (``<tag>_<sched>.alloc.txt``) and the
emitted SV, and runs a directed check: rows written then read back, a
same-cycle write+read of one address (the read must return the OLD word for
``collision`` "old"; D-12 declares "refuse" for both, i.e. the composition
promises it never happens), on ``llvm`` and ``amc``.
"""
import sys
import numpy as np
from common import load, build, guarded

mem, tgt, sched = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 64
ROWS = {"iram": 4096, "loopbuf": 24}[mem]
src = f'''from allo.ir.types import UInt, uint1, uint32, int32
N = {N}
ROWS = {ROWS}

def {mem}_k(WE: uint32[N], WA: uint32[N], W0: uint32[N], W1: uint32[N], W2: uint32[N], W3: uint32[N],
            RA: uint32[N], R0: uint32[N], R1: uint32[N], R2: uint32[N], R3: uint32[N]):
    m: UInt(128)[ROWS] = 0
    for t in range(N):
        ra: int32 = RA[t]
        word: UInt(128) = m[ra]
        R0[t] = word[0:32]
        R1[t] = word[32:64]
        R2[t] = word[64:96]
        R3[t] = word[96:128]
        if WE[t] == 1:
            wa: int32 = WA[t]
            nw: UInt(128) = W3[t]
            nw = (nw << 32) | W2[t]
            nw = (nw << 32) | W1[t]
            nw = (nw << 32) | W0[t]
            m[wa] = nw
'''
K = load(src, f"mem_{mem}_{N}")
rng = np.random.default_rng(7)
we = np.zeros(N, np.uint32); wa = np.zeros(N, np.uint32); ra = np.zeros(N, np.uint32)
W = rng.integers(0, 2**32, (4, N), dtype=np.uint64).astype(np.uint32)
half = N // 2
we[:half] = 1; wa[:half] = np.arange(half) % ROWS          # fill
ra[half:] = np.arange(N - half) % min(half, ROWS)            # read back
we[N - 4] = 1; wa[N - 4] = 3; ra[N - 4] = 3                  # same-cycle write + read of row 3
# reference: the read in iteration t sees the memory before t's write ("old")
m = {}
want = np.zeros((4, N), np.uint32)
for t in range(N):
    w = m.get(int(ra[t]), (0, 0, 0, 0))
    want[:, t] = w
    if we[t]:
        m[int(wa[t])] = tuple(int(W[k, t]) for k in range(4))
f = guarded(build, getattr(K, f"{mem}_k"), tgt, sched, ("S_t_0", "t"), tag=f"mem_{mem}")
if f is not None:
    outs = [np.zeros(N, np.uint32) for _ in range(4)]
    r = guarded(f, we, wa, *W, ra, *outs)
    got = np.stack(outs)
    defined = np.ones(N, bool); defined[:half] = False       # rows read before any write: unreset contents
    ok = (got[:, defined] == want[:, defined]).all()
    cyc = f.rpt["cycles"] if tgt == "amc" else None
    print(f"{'UNIT-MATCH' if ok else 'UNIT-DIFF '} mem {mem} {tgt} {sched}: read-back {int(defined.sum())} rows, "
          f"same-cycle row 3 got {[hex(x) for x in got[:, N - 4]]} old {[hex(x) for x in want[:, N - 4]]}; cycles={cyc}")
