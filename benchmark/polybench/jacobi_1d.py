# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A 3-point 1-D stencil, two half-sweeps per time step, ping-ponging A and B."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

TSTEPS, N = 20, 30
W = 0.33333


def build():
    @kernel
    def jacobi_1d(A: f32[N], B: f32[N]):
        for m in range(TSTEPS):
            for i0 in range(1, N - 1):
                B[i0] = W * (A[i0 - 1] + A[i0] + A[i0 + 1])
            for i1 in range(1, N - 1):
                A[i1] = W * (B[i1 - 1] + B[i1] + B[i1 + 1])

    return {"top": jacobi_1d}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("i0"), factor=4)
    s.unroll(s.loop("i1"), factor=4)
    return s


def inputs(rng):
    A = rng.uniform(0.01, 0.25, N).astype(np.float32)
    B = rng.uniform(0.01, 0.25, N).astype(np.float32)
    return A, B


def reference(A, B):
    a, b = A.copy(), B.copy()
    for _ in range(TSTEPS):
        b[1:-1] = np.float32(W) * (a[:-2] + a[1:-1] + a[2:])
        a[1:-1] = np.float32(W) * (b[:-2] + b[1:-1] + b[2:])
    return a, b


BENCHMARK = Benchmark(
    suite="polybench",
    name="jacobi_1d",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0, 1),
    tolerance=(2e-3, 2e-3),
)
