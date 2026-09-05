# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A 9-point Gauss-Seidel stencil updated in place, so every step is carried."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

TSTEPS, N = 10, 30


def build():
    @kernel
    def seidel_2d(A: f32[N, N]):
        for t in range(TSTEPS):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    A[i, j] = (
                        A[i - 1, j - 1]
                        + A[i - 1, j]
                        + A[i - 1, j + 1]
                        + A[i, j - 1]
                        + A[i, j]
                        + A[i, j + 1]
                        + A[i + 1, j - 1]
                        + A[i + 1, j]
                        + A[i + 1, j + 1]
                    ) / 9.0

    return {"top": seidel_2d}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("j"), factor=2)
    return s


def inputs(rng):
    return (rng.uniform(0.01, 0.25, (N, N)).astype(np.float32),)


def reference(A):
    a = A.copy()
    for _ in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                a[i, j] = (
                    a[i - 1, j - 1]
                    + a[i - 1, j]
                    + a[i - 1, j + 1]
                    + a[i, j - 1]
                    + a[i, j]
                    + a[i, j + 1]
                    + a[i + 1, j - 1]
                    + a[i + 1, j]
                    + a[i + 1, j + 1]
                ) / np.float32(9.0)
    return (a,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="seidel_2d",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0,),
    tolerance=(5e-3, 5e-3),
)
