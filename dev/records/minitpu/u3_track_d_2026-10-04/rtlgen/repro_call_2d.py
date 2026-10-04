"""RTLGen probe (the tree's `call` form gave zero payload): a 2-D port read
per lane, with and without a nested-kernel call on the lanes, and a carried
pipe behind the call."""
import numpy as np
from allo import kernel
from allo.lang import i32, u16
N = 16

@kernel
def addmod(a: i32, b: i32) -> i32:
    s: i32 = (a + b) & 0xFFFF
    return s

@kernel
def twod_inline(D: u16[N, 4], O: u16[N]):
    for t in range(N, name="t"):
        n0: i32 = D[t, 0]
        n1: i32 = D[t, 1]
        n2: i32 = D[t, 2]
        n3: i32 = D[t, 3]
        O[t] = (((n0 + n1) & 0xFFFF) + ((n2 + n3) & 0xFFFF)) & 0xFFFF

@kernel
def twod_call(D: u16[N, 4], O: u16[N]):
    for t in range(N, name="t"):
        n0: i32 = D[t, 0]
        n1: i32 = D[t, 1]
        n2: i32 = D[t, 2]
        n3: i32 = D[t, 3]
        n4: i32 = addmod(n0, n1)
        n5: i32 = addmod(n2, n3)
        n6: i32 = addmod(n4, n5)
        O[t] = n6

@kernel
def twod_call_pipe(D: u16[N, 4], O: u16[N]):
    q0: i32 = 0
    q1: i32 = 0
    for t in range(N, name="t"):
        n0: i32 = D[t, 0]
        n1: i32 = D[t, 1]
        n2: i32 = D[t, 2]
        n3: i32 = D[t, 3]
        n4: i32 = addmod(n0, n1)
        n5: i32 = addmod(n2, n3)
        n6: i32 = addmod(n4, n5)
        q1 = q0
        q0 = n6
        O[t] = q1

@kernel
def oned_call_pipe(A: u16[N], B: u16[N], O: u16[N]):
    q0: i32 = 0
    q1: i32 = 0
    for t in range(N, name="t"):
        a: i32 = A[t]
        b: i32 = B[t]
        n6: i32 = addmod(a, b)
        q1 = q0
        q0 = n6
        O[t] = q1

rng = np.random.default_rng(1)
D = rng.integers(0, 16384, (N, 4)).astype(np.uint16)
s = D.astype(int).sum(axis=1) & 0xFFFF
want = {"twod_inline": s, "twod_call": s, "twod_call_pipe": np.concatenate([[0, 0], s[:-2]]),
        "oned_call_pipe": np.concatenate([[0, 0], ((D[:, 0].astype(int) + D[:, 1]) & 0xFFFF)[:-2]])}
for name, k in (("twod_inline", twod_inline), ("twod_call", twod_call), ("twod_call_pipe", twod_call_pipe), ("oned_call_pipe", oned_call_pipe)):
    rtl = k.schedule().export("rtl"); res = rtl.schedule()
    o = np.zeros(N, np.uint16)
    c = rtl.cosim(D, o).cycles if name.startswith("twod") else rtl.cosim(D[:, 0].copy(), D[:, 1].copy(), o).cycles
    ok = (o.astype(int) == want[name]).all()
    print(f"{name:15s} cycles={c:4d} II={[r.interval for r in res.cyclic()]} {'OK' if ok else 'WRONG'} got={o[:6].tolist()} want={want[name][:6].tolist()}")
