# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import re
import allo
from allo.ir.types import float32, int32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


# M, N, K = 512, 512, 512
# Mt, Nt = 16, 16
M, N, K = 8, 8, 8
Mt, Nt = 4, 4
# M, N, K = 4, 4, 4
# Mt, Nt = 1, 1
P0, P1 = Mt + 2, Nt + 2
NI = (M // Mt) * (N // Nt)


@df.region()
def top(imem: int32[NI], A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    fifo_A: Stream[float32, 4][P0, P1]
    fifo_B: Stream[float32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[imem, A, B, C])
    def gemm(li: int32[NI], local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        # A[Mt, K] * B[K, Nt] = C[Mt, Nt]
        i, j = df.get_pid()
        for m in range(M // Mt):
            for n in range(N // Nt):
                # peripheral kernels
                op: int32 = li[m * (N // Nt) + n]
                with allo.meta_if(i in {0, Mt + 1} and j in {0, Nt + 1}):
                    pass
                with allo.meta_elif(j == 0):
                    # i > 0
                    for k in range(K):
                        fifo_A[i, j + 1].put(local_A[m * Mt + i - 1, k])
                with allo.meta_elif(i == 0):
                    # j > 0
                    for k in range(K):
                        fifo_B[i + 1, j].put(local_B[k, n * Nt + j - 1])
                # drain
                with allo.meta_elif(i == Mt + 1):
                    for k in range(K):
                        b: float32 = fifo_B[i, j].get()
                with allo.meta_elif(j == Nt + 1):
                    for k in range(K):
                        a: float32 = fifo_A[i, j].get()
                # main body
                with allo.meta_else():
                    c: float32 = 0.0
                    for k in range(K):
                        a: float32 = fifo_A[i, j].get()
                        b: float32 = fifo_B[i, j].get()
                        c += a * b
                        fifo_A[i, j + 1].put(a)
                        fifo_B[i + 1, j].put(b)
                    o: float32 = c
                    if op == 999999:
                        o = max(c, 0.0)
                    local_C[m * Mt + i - 1, n * Nt + j - 1] = o



if __name__ == "__main__":
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), np.float32)
    sim = df.build(top, target="simulator")
    sim(np.zeros(NI, np.int32), A, B, C)
    print("mm      correct=", np.allclose(C, A @ B, atol=1e-4))
    C2 = np.zeros((M, N), np.float32)
    sim(np.ones(NI, np.int32), A, B, C2)
    print("mm.relu correct=", np.allclose(C2, np.maximum(A @ B, 0), atol=1e-4))
