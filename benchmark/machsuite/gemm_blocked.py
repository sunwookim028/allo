# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integer matrix multiply blocked into SxS tiles with runtime tile bounds."""

import numpy as np

from allo.lang import i32, kernel

from ..spec import Benchmark

M, N, K = 32, 32, 32
S = 8


def build():
    @kernel
    def bbgemm(A: i32[M, K], B: i32[K, N], C: i32[M, N]):
        i_max: i32 = 0
        j_max: i32 = 0
        k_max: i32 = 0
        sum_value: i32 = 0
        for i in range(0, M, S):
            i_max = i + S if i + S < M else M
            for j in range(0, N, S):
                j_max = j + S if j + S < N else N
                for k in range(0, K, S):
                    k_max = k + S if k + S < K else K
                    for ii in range(i, i_max):
                        for jj in range(j, j_max):
                            sum_value = 0
                            for kk in range(k, k_max):
                                sum_value += A[ii, kk] * B[kk, jj]
                            C[ii, jj] += sum_value

    return {"top": bbgemm}


def _none(parts):
    return parts["top"].schedule()


def inputs(rng):
    A = rng.integers(0, 4, (M, K)).astype(np.int32)
    B = rng.integers(0, 4, (K, N)).astype(np.int32)
    return A, B, np.zeros((M, N), np.int32)


def reference(A, B, C):
    return ((A.astype(np.int64) @ B.astype(np.int64)).astype(np.int32),)


BENCHMARK = Benchmark(
    suite="machsuite",
    name="bbgemm",
    build=build,
    schedules={"none": _none},
    inputs=inputs,
    reference=reference,
    outputs=(2,),
)
