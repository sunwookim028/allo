"""Probe: which of the back kernel's operations loses the data on the simulator."""
import numpy as np
import allo, allo.dataflow as df
from allo.ir.types import UInt, uint16, uint32, uint8

def probe(name, body_sel):
    n = 4
    @df.region()
    def top(A: uint16[n], O1: uint16[n * 4], O2: uint16[n * 4], O3: uint16[n]):
        @df.kernel(mapping=[1], args=[A, O1, O2, O3])
        def k(a: uint16[n], o1: uint16[n * 4], o2: uint16[n * 4], o3: uint16[n]):
            mem: UInt(64)[4] = 0
            gq: UInt(64)[1] = 0
            for t in range(n):
                bf: UInt(16) = a[t]
                gnext: UInt(64) = gq[0]
                gi: UInt(2) = t
                with allo.meta_for(4) as sub:
                    if gi == sub:
                        gnext[16 * sub:16 * (sub + 1)] = bf
                gq[0] = gnext
                mem[t] = gnext
                head: UInt(64) = mem[t]
                with allo.meta_for(4) as sub:
                    o1[t * 4 + sub] = gnext[16 * sub:16 * (sub + 1)]
                    o2[t * 4 + sub] = head[16 * sub:16 * (sub + 1)]
                o3[t] = gnext[0:16]
    mod = df.build(top, target="simulator")
    a = np.array([0x1111, 0x2222, 0x3333, 0x4444], dtype=np.uint16)
    o1 = np.zeros(16, np.uint16); o2 = np.zeros(16, np.uint16); o3 = np.zeros(4, np.uint16)
    mod(a, o1, o2, o3)
    print("gnext slices", [hex(x) for x in o1])
    print("mem   slices", [hex(x) for x in o2])
    print("gnext[0:16] ", [hex(x) for x in o3])
probe("a", 0)
