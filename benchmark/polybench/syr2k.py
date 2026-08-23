# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric rank-2k update: two guarded outer-product accumulations into C."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

M, N = 20, 30
ALPHA, BETA = 1.5, 1.2


def build():
    @kernel
    def update_C(Cin: f32[N, N], Cout: f32[N, N]):
        for i0 in range(N):
            for j0 in range(N):
                if j0 <= i0:
                    Cout[i0, j0] = BETA * Cin[i0, j0]
                else:
                    Cout[i0, j0] = Cin[i0, j0]

    @kernel
    def compute_sum(
        A: f32[N, M],
        A_copy: f32[N, M],
        B: f32[N, M],
        B_copy: f32[N, M],
        Cin: f32[N, N],
        Cout: f32[N, N],
    ):
        buffer: f32[N, N] = 0.0
        for i0 in range(N):
            for j0 in range(N):
                buffer[i0, j0] = Cin[i0, j0]
        for i1 in range(N):
            for k1 in range(M):
                for j1 in range(N):
                    if j1 <= i1:
                        buffer[i1, j1] += (
                            A[j1, k1] * ALPHA * B[i1, k1]
                            + B_copy[j1, k1] * ALPHA * A_copy[i1, k1]
                        )
        for i2 in range(N):
            for j2 in range(N):
                Cout[i2, j2] = buffer[i2, j2]

    @kernel
    def syr2k(
        A: f32[N, M],
        A_copy: f32[N, M],
        B: f32[N, M],
        B_copy: f32[N, M],
        Cin: f32[N, N],
        Cout: f32[N, N],
    ):
        C: f32[N, N] = 0.0
        update_C(Cin, C)
        compute_sum(A, A_copy, B, B_copy, C, Cout)

    return {"top": syr2k, "update_C": update_C, "compute_sum": compute_sum}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    uc = parts["update_C"].schedule()
    uc.pipeline(uc.loop("j0"), ii=1)
    cs = parts["compute_sum"].schedule()
    cs.pipeline(cs.loop("k1"), ii=1)
    top = parts["top"].schedule()
    top.compose(uc, cs)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (N, M)).astype(np.float32)
    B = rng.uniform(0.01, 0.25, (N, M)).astype(np.float32)
    Cin = rng.uniform(0.01, 0.25, (N, N)).astype(np.float32)
    return A, A.copy(), B, B.copy(), Cin, np.zeros((N, N), np.float32)


def reference(A, A_copy, B, B_copy, Cin, Cout):
    out = Cin.copy()
    lower = np.tril(np.ones((N, N), bool))
    out[lower] = np.float32(BETA) * Cin[lower]
    prod = np.float32(ALPHA) * (B @ A.T + A_copy @ B_copy.T)
    out[lower] += prod[lower]
    return (out,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="syr2k",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(5,),
    tolerance=(2e-3, 2e-3),
)
