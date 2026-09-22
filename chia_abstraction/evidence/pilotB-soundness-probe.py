import sys, allo
from allo.ir.types import int32
PN = 16
def ports_kernel(A: int32[PN], B: int32[PN]):
    buf: int32[PN] = 0
    for i in range(PN // 2):
        buf[2 * i] = A[2 * i] + 1
        buf[2 * i + 1] = A[2 * i + 1] + 1
    for j in range(PN):
        B[j] = buf[j]

def trial(name, sched):
    s = allo.customize(ports_kernel)
    sched(s)
    try:
        s.memory_ports("buf", 1)
        return f"{name}: ACCEPTED"
    except Exception as e:
        return f"{name}: refused -- {type(e).__name__}: {str(e)[:90]}"

print(trial("cyclic-2 (needs 1 port/bank: CORRECT accept)",
            lambda s: (s.pipeline("i"), s.partition(s.buf, partition_type=2, dim=1, factor=2))))
print(trial("BLOCK-2  (both stores land in bank 0: needs 2 ports)",
            lambda s: (s.pipeline("i"), s.partition(s.buf, partition_type=1, dim=1, factor=2))))
print(trial("unpipelined, 2 stores (sequential: 1 port suffices)",
            lambda s: None))
