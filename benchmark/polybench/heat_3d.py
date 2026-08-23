# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A 7-point 3-D heat stencil with both half-sweeps fused into one loop body."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

TSTEPS, N = 6, 10
C0, C1 = 0.125, 2.0


def build():
    @kernel
    def heat_3d(A: f32[N, N, N], B: f32[N, N, N]):
        const0: f32 = C0
        const1: f32 = C1
        for m in range(TSTEPS):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    for k in range(1, N - 1):
                        B[i, j, k] = (
                            const0
                            * (A[i + 1, j, k] - const1 * A[i, j, k] + A[i - 1, j, k])
                            + const0
                            * (A[i, j + 1, k] - const1 * A[i, j, k] + A[i, j - 1, k])
                            + const0
                            * (A[i, j, k + 1] - const1 * A[i, j, k] + A[i, j, k - 1])
                            + A[i, j, k]
                        )
                        A[i, j, k] = (
                            const0
                            * (B[i + 1, j, k] - const1 * B[i, j, k] + B[i - 1, j, k])
                            + const0
                            * (B[i, j + 1, k] - const1 * B[i, j, k] + B[i, j - 1, k])
                            + const0
                            * (B[i, j, k + 1] - const1 * B[i, j, k] + B[i, j, k - 1])
                            + B[i, j, k]
                        )

    return {"top": heat_3d}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("j"), ii=1)
    return s


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (N, N, N)).astype(np.float32)
    B = rng.uniform(0.01, 0.25, (N, N, N)).astype(np.float32)
    return A, B


def reference(A, B):
    a, b = A.copy(), B.copy()
    c0, c1 = np.float32(C0), np.float32(C1)
    for _ in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    b[i, j, k] = (
                        c0 * (a[i + 1, j, k] - c1 * a[i, j, k] + a[i - 1, j, k])
                        + c0 * (a[i, j + 1, k] - c1 * a[i, j, k] + a[i, j - 1, k])
                        + c0 * (a[i, j, k + 1] - c1 * a[i, j, k] + a[i, j, k - 1])
                        + a[i, j, k]
                    )
                    a[i, j, k] = (
                        c0 * (b[i + 1, j, k] - c1 * b[i, j, k] + b[i - 1, j, k])
                        + c0 * (b[i, j + 1, k] - c1 * b[i, j, k] + b[i, j - 1, k])
                        + c0 * (b[i, j, k + 1] - c1 * b[i, j, k] + b[i, j, k - 1])
                        + b[i, j, k]
                    )
    return a, b


BENCHMARK = Benchmark(
    suite="polybench",
    name="heat_3d",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0, 1),
    tolerance=(5e-3, 5e-3),
)
