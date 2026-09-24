# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tiled systolic GEMM (C = A @ B) on the SystemC backend. This is the standard
# Allo dataflow example (tests/dataflow/test_tiled_systolic.py) run through
# target="systemc" — it exercises, together:
#   - a mapping=[P0,P1] GRID of PEs (feeders / body / drains via meta_if),
#   - 2-D random-access boundary arrays A, B (reads) and C (writes) -> internal
#     memory ports (2x AlloMem + 1x AlloMemW), and
#   - int accumulation that widens past 64 bits (the ap_int<65> GEMM accumulator).
#
# NOTE: this works with Mt=Nt=1 (each boundary array is touched by exactly ONE
# grid instance = single memory-port client). For Mt,Nt>1, several feeder PEs
# read the same A/B array -> multiple clients on one memory channel, which needs
# arbitration and is not supported yet.

import numpy as np
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

M, N, K = 4, 4, 4
Mt, Nt = 1, 1
P0, P1 = Mt + 2, Nt + 2


@df.region()
def top(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
    fifo_A: Stream[int32, 4][P0, P1]
    fifo_B: Stream[int32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: int32[M, K], local_B: int32[K, N], local_C: int32[M, N]):
        i, j = df.get_pid()
        for m in range(M // Mt):
            for n in range(N // Nt):
                with allo.meta_if(i in {0, Mt + 1} and j in {0, Nt + 1}):
                    pass
                with allo.meta_elif(j == 0):
                    for k in range(K):
                        fifo_A[i, j + 1].put(local_A[m * Mt + i - 1, k])
                with allo.meta_elif(i == 0):
                    for k in range(K):
                        fifo_B[i + 1, j].put(local_B[k, n * Nt + j - 1])
                with allo.meta_elif(i == Mt + 1):
                    for k in range(K):
                        b: int32 = fifo_B[i, j].get()
                with allo.meta_elif(j == Nt + 1):
                    for k in range(K):
                        a: int32 = fifo_A[i, j].get()
                with allo.meta_else():
                    c: int32 = 0
                    for k in range(K):
                        a: int32 = fifo_A[i, j].get()
                        b: int32 = fifo_B[i, j].get()
                        c += a * b
                        fifo_A[i, j + 1].put(a)
                        fifo_B[i + 1, j].put(b)
                    local_C[m * Mt + i - 1, n * Nt + j - 1] = c


if __name__ == "__main__":
    import os

    code = df.build(top, target="systemc").hls_code
    open("tiled_systolic.cpp", "w").write(code)
    print("wrote tiled_systolic.cpp")
    assert code.count("AlloMem<") == 2 and code.count("AlloMemW<") == 1
    print("A,B -> 2x AlloMem (read ports), C -> AlloMemW (write port)")

    if os.environ.get("MGC_HOME"):
        A = np.random.randint(0, 10, (M, K)).astype(np.int32)
        B = np.random.randint(0, 10, (K, N)).astype(np.int32)
        C = np.zeros((M, N), dtype=np.int32)
        mod = df.build(top, target="systemc", mode="csim", project="tiled_systolic.prj")
        mod(A, B, C)
        print("C =\n", C)
        print("PASS" if np.array_equal(C, A @ B) else "FAIL")
