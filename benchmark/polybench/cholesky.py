# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-place Cholesky factorisation over a triangular iteration space."""

import numpy as np

from allo.lang import f32, kernel
from allo.operators import math as amath

from ..spec import Benchmark

N = 30


def build():
    @kernel
    def cholesky(A: f32[N, N]):
        for i in range(N):
            for j in range(i):
                for k in range(j):
                    A[i, j] = A[i, j] - A[i, k] * A[j, k]
                A[i, j] = A[i, j] / A[j, j]
            for k2 in range(i):
                A[i, i] = A[i, i] - A[i, k2] * A[i, k2]
            A[i, i] = amath.sqrt(A[i, i] * 1.0)

    return {"top": cholesky}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("k"), ii=1)
    s.pipeline(s.loop("k2"), ii=1)
    return s


def inputs(rng):
    base = rng.uniform(-0.05, 0.05, (N, N)).astype(np.float32)
    A = (base @ base.T + np.eye(N, dtype=np.float32)).astype(np.float32)
    return (A,)


def reference(A):
    out = A.copy()
    for i in range(N):
        for j in range(i):
            for k in range(j):
                out[i, j] = out[i, j] - out[i, k] * out[j, k]
            out[i, j] = out[i, j] / out[j, j]
        for k in range(i):
            out[i, i] = out[i, i] - out[i, k] * out[i, k]
        out[i, i] = np.sqrt(out[i, i])
    return (out,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="cholesky",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0,),
    tolerance=(5e-3, 5e-3),
)
