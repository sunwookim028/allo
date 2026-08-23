# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LU decomposition followed by forward and backward substitution."""

import numpy as np

from allo.lang import f32, index, kernel

from ..spec import Benchmark

N = 30


def build():
    @kernel
    def ludcmp(A: f32[N, N], b: f32[N], x: f32[N], y: f32[N]):
        for i in range(N):
            for j in range(i):
                w_lower: f32 = A[i, j]
                for k in range(j):
                    w_lower -= A[i, k] * A[k, j]
                A[i, j] = w_lower / A[j, j]
            for j2 in range(i, N):
                w_upper: f32 = A[i, j2]
                for k2 in range(i):
                    w_upper -= A[i, k2] * A[k2, j2]
                A[i, j2] = w_upper

        for i2 in range(N):
            alpha_y: f32 = b[i2]
            for j3 in range(i2):
                alpha_y -= A[i2, j3] * y[j3]
            y[i2] = alpha_y

        for i_inv in range(N):
            i3: index = N - 1 - i_inv
            alpha_x: f32 = y[i3]
            for j4 in range(i3 + 1, N):
                alpha_x -= A[i3, j4] * x[j4]
            x[i3] = alpha_x / A[i3, i3]

    return {"top": ludcmp}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("k"), factor=2)
    s.unroll(s.loop("k2"), factor=2)
    s.unroll(s.loop("j3"), factor=2)
    s.unroll(s.loop("j4"), factor=2)
    return s


def inputs(rng):
    A = rng.uniform(-0.01, 0.01, (N, N)).astype(np.float32)
    A += np.eye(N, dtype=np.float32) * np.float32(2.0)
    b = rng.uniform(0.01, 0.25, N).astype(np.float32)
    return (A.astype(np.float32), b, np.zeros(N, np.float32), np.zeros(N, np.float32))


def reference(A, b, x, y):
    a = A.copy()
    for i in range(N):
        for j in range(i):
            w = a[i, j]
            for k in range(j):
                w -= a[i, k] * a[k, j]
            a[i, j] = w / a[j, j]
        for j in range(i, N):
            w = a[i, j]
            for k in range(i):
                w -= a[i, k] * a[k, j]
            a[i, j] = w
    yy = np.zeros(N, np.float32)
    for i in range(N):
        acc = b[i]
        for j in range(i):
            acc -= a[i, j] * yy[j]
        yy[i] = acc
    xx = np.zeros(N, np.float32)
    for i in range(N - 1, -1, -1):
        acc = yy[i]
        for j in range(i + 1, N):
            acc -= a[i, j] * xx[j]
        xx[i] = acc / a[i, i]
    return a, xx, yy


BENCHMARK = Benchmark(
    suite="polybench",
    name="ludcmp",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0, 2, 3),
    tolerance=(5e-3, 5e-3),
)
