"""RTLGen: a 2-stage pipe of loop-carried scalars loses a stage when the loop
body holds a nested-kernel call (oned_call_pipe in repro_call_2d.py). Is it
the call, or the imperfect-nest decomposition? Same pipe behind an inner
constant-trip loop, and behind a call placed AFTER the pipe shift."""
import numpy as np
from allo import kernel
from allo.lang import i32, u16
N = 16

@kernel
def addmod(a: i32, b: i32) -> i32:
    s: i32 = (a + b) & 0xFFFF
    return s

@kernel
def pipe_innerloop(A: u16[N], B: u16[N], O: u16[N]):
    q0: i32 = 0
    q1: i32 = 0
    for t in range(N, name="t"):
        a: i32 = A[t]
        b: i32 = B[t]
        acc: i32 = 0
        for j in range(4, name="j"):
            acc = acc + (a >> j)
        n6: i32 = (acc + b) & 0xFFFF
        q1 = q0
        q0 = n6
        O[t] = q1

@kernel
def pipe_before_call(A: u16[N], B: u16[N], O: u16[N]):
    q0: i32 = 0
    q1: i32 = 0
    for t in range(N, name="t"):
        a: i32 = A[t]
        b: i32 = B[t]
        O[t] = q1
        q1 = q0
        n6: i32 = addmod(a, b)
        q0 = n6

@kernel
def pipe3_call(A: u16[N], B: u16[N], O: u16[N]):
    q0: i32 = 0
    q1: i32 = 0
    q2: i32 = 0
    for t in range(N, name="t"):
        a: i32 = A[t]
        b: i32 = B[t]
        n6: i32 = addmod(a, b)
        q2 = q1
        q1 = q0
        q0 = n6
        O[t] = q2

rng = np.random.default_rng(1)
A = rng.integers(0, 16384, N).astype(np.uint16); B = rng.integers(0, 16384, N).astype(np.uint16)
acc = sum((A.astype(int) >> j) for j in range(4))
want = {"pipe_innerloop": np.concatenate([[0, 0], ((acc + B) & 0xFFFF)[:-2]]),
        "pipe_before_call": np.concatenate([[0, 0], ((A.astype(int) + B) & 0xFFFF)[:-2]]),
        "pipe3_call": np.concatenate([[0, 0, 0], ((A.astype(int) + B) & 0xFFFF)[:-3]])}
for name, k in (("pipe_innerloop", pipe_innerloop), ("pipe_before_call", pipe_before_call), ("pipe3_call", pipe3_call)):
    rtl = k.schedule().export("rtl"); res = rtl.schedule()
    o = np.zeros(N, np.uint16); c = rtl.cosim(A, B, o).cycles
    ok = (o.astype(int) == want[name]).all()
    print(f"{name:16s} cycles={c:4d} II={[r.interval for r in res.cyclic()]} {'OK' if ok else 'WRONG'} got={o[:6].tolist()} want={want[name][:6].tolist()}")
