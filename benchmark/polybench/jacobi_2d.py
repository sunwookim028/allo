# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A 5-point 2-D stencil, two half-sweeps per time step, ping-ponging A and B."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

TSTEPS, N = 10, 24


def build():
    @kernel
    def compute_A(A0: f32[N, N], B0: f32[N, N]):
        for i0 in range(N - 2):
            for j0 in range(N - 2):
                B0[i0 + 1, j0 + 1] = 0.2 * (
                    A0[i0, j0 + 1]
                    + A0[i0 + 1, j0]
                    + A0[i0 + 1, j0 + 1]
                    + A0[i0 + 1, j0 + 2]
                    + A0[i0 + 2, j0 + 1]
                )

    @kernel
    def compute_B(B1: f32[N, N], A1: f32[N, N]):
        for i1 in range(N - 2):
            for j1 in range(N - 2):
                A1[i1 + 1, j1 + 1] = 0.2 * (
                    B1[i1, j1 + 1]
                    + B1[i1 + 1, j1]
                    + B1[i1 + 1, j1 + 1]
                    + B1[i1 + 1, j1 + 2]
                    + B1[i1 + 2, j1 + 1]
                )

    @kernel
    def jacobi_2d(A: f32[N, N], B: f32[N, N]):
        for m in range(TSTEPS):
            compute_A(A, B)
            compute_B(B, A)

    return {"top": jacobi_2d, "compute_A": compute_A, "compute_B": compute_B}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    ca = parts["compute_A"].schedule()
    ca.pipeline(ca.loop("i0"), ii=1)
    cb = parts["compute_B"].schedule()
    cb.pipeline(cb.loop("i1"), ii=1)
    top = parts["top"].schedule()
    top.compose(ca, cb)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (N, N)).astype(np.float32)
    B = rng.uniform(0.01, 0.25, (N, N)).astype(np.float32)
    return A, B


def reference(A, B):
    a, b = A.copy(), B.copy()
    for _ in range(TSTEPS):
        b[1:-1, 1:-1] = np.float32(0.2) * (
            a[:-2, 1:-1] + a[1:-1, :-2] + a[1:-1, 1:-1] + a[1:-1, 2:] + a[2:, 1:-1]
        )
        a[1:-1, 1:-1] = np.float32(0.2) * (
            b[:-2, 1:-1] + b[1:-1, :-2] + b[1:-1, 1:-1] + b[1:-1, 2:] + b[2:, 1:-1]
        )
    return a, b


BENCHMARK = Benchmark(
    suite="polybench",
    name="jacobi_2d",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0, 1),
    tolerance=(2e-3, 2e-3),
)
