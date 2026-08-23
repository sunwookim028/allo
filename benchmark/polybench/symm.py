# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric matrix multiply: a triangular partial sum plus a row-wise update."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

M, N = 20, 30
ALPHA, BETA = 1.5, 1.2


def build():
    @kernel
    def compute_sum(A: f32[M, M], B: f32[M, N], summ: f32[M, N]):
        for i1 in range(M):
            for j1 in range(N):
                for k1 in range(M):
                    if k1 < i1:
                        summ[i1, j1] += B[k1, j1] * A[i1, k1]

    @kernel
    def update_C(A: f32[M, M], B: f32[M, N], summ: f32[M, N], C: f32[M, N]):
        for i in range(M):
            for k in range(i):
                for j in range(N):
                    C[k, j] = C[k, j] + ALPHA * B[i, j] * A[i, k]
            for j2 in range(N):
                C[i, j2] = (
                    BETA * C[i, j2] + ALPHA * B[i, j2] * A[i, i] + ALPHA * summ[i, j2]
                )

    @kernel
    def symm(A0: f32[M, M], A1: f32[M, M], B0: f32[M, N], B1: f32[M, N], C: f32[M, N]):
        summ: f32[M, N] = 0.0
        compute_sum(A0, B0, summ)
        update_C(A1, B1, summ, C)

    return {"top": symm, "compute_sum": compute_sum, "update_C": update_C}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    cs = parts["compute_sum"].schedule()
    cs.pipeline(cs.loop("j1"), ii=1)
    uc = parts["update_C"].schedule()
    uc.pipeline(uc.loop("j"), ii=1)
    uc.pipeline(uc.loop("j2"), ii=1)
    top = parts["top"].schedule()
    top.compose(cs, uc)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (M, M)).astype(np.float32)
    B = rng.uniform(0.01, 0.25, (M, N)).astype(np.float32)
    C = rng.uniform(0.01, 0.25, (M, N)).astype(np.float32)
    return A, A.copy(), B, B.copy(), C


def reference(A0, A1, B0, B1, C):
    summ = np.zeros((M, N), np.float32)
    for i in range(M):
        for j in range(N):
            for k in range(M):
                if k < i:
                    summ[i, j] += B0[k, j] * A0[i, k]
    out = C.copy()
    for i in range(M):
        for k in range(i):
            for j in range(N):
                out[k, j] = out[k, j] + np.float32(ALPHA) * B1[i, j] * A1[i, k]
        for j in range(N):
            out[i, j] = (
                np.float32(BETA) * out[i, j]
                + np.float32(ALPHA) * B1[i, j] * A1[i, i]
                + np.float32(ALPHA) * summ[i, j]
            )
    return (out,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="symm",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(4,),
    tolerance=(2e-3, 2e-3),
)
