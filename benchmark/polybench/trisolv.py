# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward substitution: a triangular solve whose inner trip count grows with i."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

N = 40


def build():
    @kernel
    def trisolv(L: f32[N, N], b: f32[N], x: f32[N]):
        for i in range(N):
            x[i] = b[i]
            for j in range(i):
                x[i] -= L[i, j] * x[j]
            x[i] /= L[i, i]

    return {"top": trisolv}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("j"), factor=4)
    s.pipeline(s.loop("j"), ii=1)
    return s


def inputs(rng):
    L = rng.uniform(-0.01, 0.01, (N, N)).astype(np.float32)
    L += np.eye(N, dtype=np.float32) * np.float32(2.0)
    L = np.tril(L).astype(np.float32)
    b = rng.uniform(0.01, 0.25, N).astype(np.float32)
    return L, b, np.zeros(N, np.float32)


def reference(L, b, x):
    out = np.zeros(N, np.float32)
    for i in range(N):
        out[i] = b[i]
        for j in range(i):
            out[i] -= L[i, j] * out[j]
        out[i] /= L[i, i]
    return (out,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="trisolv",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(2,),
    tolerance=(2e-3, 2e-3),
)
