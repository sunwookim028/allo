# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two independent matrix-vector products over the same matrix, one transposed."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

M, N = 38, 42


def build():
    @kernel
    def stageS(A: f32[N, M], r: f32[N], s: f32[M]):
        for i0 in range(N):
            local_r: f32 = r[i0]
            for j0 in range(M):
                s[j0] += local_r * A[i0, j0]

    @kernel
    def stageQ(A: f32[N, M], p: f32[M], q: f32[N]):
        for i1 in range(N):
            for j1 in range(M):
                q[i1] += A[i1, j1] * p[j1]

    @kernel
    def bicg(
        A: f32[N, M], A_copy: f32[N, M], p: f32[M], r: f32[N], q: f32[N], s: f32[M]
    ):
        stageS(A, r, s)
        stageQ(A_copy, p, q)

    return {"top": bicg, "stageS": stageS, "stageQ": stageQ}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    ss = parts["stageS"].schedule()
    ss.pipeline(ss.loop("j0"), ii=1)
    sq = parts["stageQ"].schedule()
    sq.pipeline(sq.loop("i1"), ii=1)
    top = parts["top"].schedule()
    top.compose(ss, sq)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (N, M)).astype(np.float32)
    p = rng.uniform(0.01, 0.25, M).astype(np.float32)
    r = rng.uniform(0.01, 0.25, N).astype(np.float32)
    return (A, A.copy(), p, r, np.zeros(N, np.float32), np.zeros(M, np.float32))


def reference(A, A_copy, p, r, q, s):
    return A_copy @ p, A.T @ r


BENCHMARK = Benchmark(
    suite="polybench",
    name="bicg",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(4, 5),
    tolerance=(2e-3, 2e-3),
)
