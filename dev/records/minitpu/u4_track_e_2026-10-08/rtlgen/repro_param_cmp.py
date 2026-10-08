"""RTLGen E-R1, minimal: a narrow unsigned PARAMETER of a nested @kernel
compared with a literal whose top bit (at the parameter's width) is set."""
import numpy as np
from allo import kernel
from allo.lang import u3, u32
N = 8

@kernel
def is4_param(op: u3) -> u32:
    r: u32 = 0
    if op == 4:
        r = 1
    return r

@kernel
def is3_param(op: u3) -> u32:
    r: u32 = 0
    if op == 3:
        r = 1
    return r

@kernel
def is4_local(x: u32) -> u32:
    op: u3 = x
    r: u32 = 0
    if op == 4:
        r = 1
    return r

@kernel
def k4p(A: u32[N], O: u32[N]):
    for t in range(N, name="t"):
        op: u3 = A[t]
        O[t] = is4_param(op)

@kernel
def k3p(A: u32[N], O: u32[N]):
    for t in range(N, name="t"):
        op: u3 = A[t]
        O[t] = is3_param(op)

@kernel
def k4l(A: u32[N], O: u32[N]):
    for t in range(N, name="t"):
        O[t] = is4_local(A[t])

@kernel
def k4top(A: u32[N], O: u32[N]):
    for t in range(N, name="t"):
        op: u3 = A[t]
        r: u32 = 0
        if op == 4:
            r = 1
        O[t] = r

A = np.arange(N, dtype=np.uint32)
for nm, k, v in (("top-level local == 4", k4top, 4), ("callee param u3 == 3", k3p, 3), ("callee local u3 == 4", k4l, 4),
                 ("callee param u3 == 4", k4p, 4)):
    want = (A == v).astype(np.uint32)
    g = np.zeros(N, np.uint32); k.schedule().export("cpu")(A, g)
    o = np.zeros(N, np.uint32); k.schedule().export("rtl").cosim(A, o)
    print(f"RESULT {nm:24s} cpu {'OK' if (g == want).all() else 'WRONG ' + str(g.tolist())} | rtl "
          f"{'OK' if (o == want).all() else 'WRONG ' + str(o.tolist())}", flush=True)
