# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two matrix-vector updates of the same matrix, one walking it transposed."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

N = 40


def build():
    @kernel
    def stageA(x1_in: f32[N], x1_out: f32[N], A: f32[N, N], y1: f32[N]):
        for i0 in range(N):
            x: f32 = x1_in[i0]
            for j0 in range(N):
                x += A[i0, j0] * y1[j0]
            x1_out[i0] = x

    @kernel
    def stageB(x2_in: f32[N], x2_out: f32[N], A: f32[N, N], y2: f32[N]):
        for i1 in range(N):
            x: f32 = x2_in[i1]
            for j1 in range(N):
                x += A[j1, i1] * y2[j1]
            x2_out[i1] = x

    @kernel
    def mvt(
        A: f32[N, N],
        A_copy: f32[N, N],
        y1: f32[N],
        y2: f32[N],
        x1: f32[N],
        x2: f32[N],
        x1_out: f32[N],
        x2_out: f32[N],
    ):
        stageA(x1, x1_out, A, y1)
        stageB(x2, x2_out, A_copy, y2)

    return {"top": mvt, "stageA": stageA, "stageB": stageB}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    sa = parts["stageA"].schedule()
    sa.pipeline(sa.loop("i0"), ii=1)
    sb = parts["stageB"].schedule()
    sb.pipeline(sb.loop("i1"), ii=1)
    top = parts["top"].schedule()
    top.compose(sa, sb)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (N, N)).astype(np.float32)
    y1 = rng.uniform(0.01, 0.25, N).astype(np.float32)
    y2 = rng.uniform(0.01, 0.25, N).astype(np.float32)
    x1 = rng.uniform(0.01, 0.25, N).astype(np.float32)
    x2 = rng.uniform(0.01, 0.25, N).astype(np.float32)
    return (
        A,
        A.copy(),
        y1,
        y2,
        x1,
        x2,
        np.zeros(N, np.float32),
        np.zeros(N, np.float32),
    )


def reference(A, A_copy, y1, y2, x1, x2, x1_out, x2_out):
    return x1 + A @ y1, x2 + A_copy.T @ y2


BENCHMARK = Benchmark(
    suite="polybench",
    name="mvt",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(6, 7),
    tolerance=(2e-3, 2e-3),
)
