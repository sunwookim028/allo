# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rank-2 update of A followed by two matrix-vector products chained through x."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

N = 30
ALPHA, BETA = 0.1, 0.1


def build():
    @kernel
    def gemver(
        A: f32[N, N],
        u1: f32[N],
        u2: f32[N],
        v1: f32[N],
        v2: f32[N],
        x: f32[N],
        y: f32[N],
        w: f32[N],
        z: f32[N],
    ):
        for i0 in range(N):
            for j0 in range(N):
                A[i0, j0] = A[i0, j0] + u1[i0] * v1[j0] + u2[i0] * v2[j0]

        for i1 in range(N):
            for j1 in range(N):
                x[i1] = x[i1] + BETA * A[j1, i1] * y[j1]

        for i2 in range(N):
            x[i2] = x[i2] + z[i2]

        for i3 in range(N):
            for j3 in range(N):
                w[i3] = w[i3] + ALPHA * A[i3, j3] * x[j3]

    return {"top": gemver}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("i0"), ii=1)
    s.pipeline(s.loop("i1"), ii=1)
    s.pipeline(s.loop("i2"), ii=1)
    s.pipeline(s.loop("i3"), ii=1)
    return s


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (N, N)).astype(np.float32)
    vecs = [rng.uniform(0.01, 0.25, N).astype(np.float32) for _ in range(6)]
    u1, u2, v1, v2, x, y = vecs
    return (
        A,
        u1,
        u2,
        v1,
        v2,
        x,
        y,
        np.zeros(N, np.float32),
        rng.uniform(0.01, 0.25, N).astype(np.float32),
    )


def reference(A, u1, u2, v1, v2, x, y, w, z):
    a = A + np.outer(u1, v1) + np.outer(u2, v2)
    xx = x + np.float32(BETA) * (a.T @ y)
    xx = xx + z
    return a, xx, w + np.float32(ALPHA) * (a @ xx)


BENCHMARK = Benchmark(
    suite="polybench",
    name="gemver",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0, 5, 7),
    tolerance=(2e-3, 2e-3),
)
