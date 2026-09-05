# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix-transpose-times-vector chained onto a matrix-times-vector."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

M, N = 38, 42


def build():
    @kernel
    def stage_M(A: f32[M, N], x: f32[N], out_Ax: f32[M]):
        for m in range(M):
            for r in range(N):
                out_Ax[m] += A[m, r] * x[r]

    @kernel
    def stage_N(A: f32[M, N], out_Ax: f32[M], y: f32[N]):
        for n in range(N):
            for k in range(M):
                y[n] += A[k, n] * out_Ax[k]

    @kernel
    def atax(A: f32[M, N], x: f32[N], y: f32[N]):
        out_Ax: f32[M] = 0.0
        stage_M(A, x, out_Ax)
        stage_N(A, out_Ax, y)

    return {"top": atax, "stage_M": stage_M, "stage_N": stage_N}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    sm = parts["stage_M"].schedule()
    sm.pipeline(sm.loop("m"), ii=1)
    sn = parts["stage_N"].schedule()
    sn.pipeline(sn.loop("n"), ii=1)
    top = parts["top"].schedule()
    top.compose(sm, sn)
    return top


# Two variants only: the stages read A transposed relative to each other, so no
# one partition of it resolves for both, and the II is A's ports either way.


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (M, N)).astype(np.float32)
    x = rng.uniform(0.01, 0.25, N).astype(np.float32)
    return A, x, np.zeros(N, np.float32)


def reference(A, x, y):
    return (A.T @ (A @ x),)


BENCHMARK = Benchmark(
    suite="polybench",
    name="atax",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(2,),
    tolerance=(2e-3, 2e-3),
)
