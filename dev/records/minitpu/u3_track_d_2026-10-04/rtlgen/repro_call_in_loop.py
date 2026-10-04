"""RTLGen probe: a nested @kernel returning i32, called inside a loop whose
result feeds a loop-carried scalar (the tree's `call` form). Checks (a) a
plain call per iteration stored to an output, (b) the call result carried to
the next iteration through a scalar, (c) two chained calls."""
import numpy as np
from allo import kernel
from allo.lang import i32, u16
N = 16

@kernel
def addmod(a: i32, b: i32) -> i32:
    s: i32 = (a + b) & 0xFFFF
    return s

@kernel
def plain(A: u16[N], B: u16[N], O: u16[N]):
    for t in range(N, name="t"):
        a: i32 = A[t]
        b: i32 = B[t]
        y: i32 = addmod(a, b)
        O[t] = y

@kernel
def carried(A: u16[N], B: u16[N], O: u16[N]):
    q: i32 = 0
    for t in range(N, name="t"):
        a: i32 = A[t]
        b: i32 = B[t]
        y: i32 = addmod(a, b)
        O[t] = q
        q = y

@kernel
def chained(A: u16[N], B: u16[N], O: u16[N]):
    for t in range(N, name="t"):
        a: i32 = A[t]
        b: i32 = B[t]
        y: i32 = addmod(a, b)
        z: i32 = addmod(y, a)
        O[t] = z

@kernel
def carried_inline(A: u16[N], B: u16[N], O: u16[N]):
    q: i32 = 0
    for t in range(N, name="t"):
        a: i32 = A[t]
        b: i32 = B[t]
        y: i32 = (a + b) & 0xFFFF
        O[t] = q
        q = y

A = (np.arange(N) * 1000 + 7).astype(np.uint16); B = (np.arange(N) * 333 + 5).astype(np.uint16)
want = {"plain": (A.astype(int) + B) & 0xFFFF, "chained": (((A.astype(int) + B) & 0xFFFF) + A) & 0xFFFF}
want["carried"] = np.concatenate([[0], want["plain"][:-1]]); want["carried_inline"] = want["carried"]
for name, k in (("plain", plain), ("carried", carried), ("chained", chained), ("carried_inline", carried_inline)):
    rtl = k.schedule().export("rtl")
    res = rtl.schedule()
    o = np.zeros(N, np.uint16); c = rtl.cosim(A, B, o).cycles
    ok = (o.astype(int) == want[name]).all()
    print(f"{name:15s} cycles={c:4d} II={[r.interval for r in res.cyclic()]} {'OK' if ok else 'WRONG'} got={o[:6].tolist()} want={want[name][:6].tolist()}")
