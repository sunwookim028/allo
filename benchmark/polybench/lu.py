# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-place LU decomposition: a lower sweep then an upper sweep per row."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

N = 30


def build():
    @kernel
    def lu(A: f32[N, N]):
        for i in range(N):
            for j in range(i):
                for k in range(j):
                    A[i, j] -= A[i, k] * A[k, j]
                A[i, j] /= A[j, j]
            for j2 in range(i, N):
                for k2 in range(i):
                    A[i, j2] -= A[i, k2] * A[k2, j2]

    return {"top": lu}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("k"), factor=2)
    s.unroll(s.loop("k2"), factor=2)
    return s


def inputs(rng):
    A = rng.uniform(-0.01, 0.01, (N, N)).astype(np.float32)
    A += np.eye(N, dtype=np.float32) * np.float32(2.0)
    return (A.astype(np.float32),)


def reference(A):
    out = A.copy()
    for i in range(N):
        for j in range(i):
            for k in range(j):
                out[i, j] -= out[i, k] * out[k, j]
            out[i, j] /= out[j, j]
        for j in range(i, N):
            for k in range(i):
                out[i, j] -= out[i, k] * out[k, j]
    return (out,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="lu",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0,),
    tolerance=(5e-3, 5e-3),
)
