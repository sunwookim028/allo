# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gram-Schmidt QR: normalise a column, then project it out of every later one."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

M, N = 20, 30


def build():
    @kernel
    def gramschmidt(A: f32[M, N], Q: f32[M, N], R: f32[N, N]):
        for k in range(N):
            nrm: f32 = 0.0
            for i in range(M):
                nrm += A[i, k] * A[i, k]
            R[k, k] = nrm

            for i2 in range(M):
                Q[i2, k] = A[i2, k] / R[k, k]

            for j in range(k + 1, N):
                R[k, j] = 0.0
                for i3 in range(M):
                    R[k, j] += Q[i3, k] * A[i3, j]
                for i4 in range(M):
                    A[i4, j] -= Q[i4, k] * R[k, j]

    return {"top": gramschmidt}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("i2"), factor=2)
    s.unroll(s.loop("i4"), factor=2)
    return s


def inputs(rng):
    A = rng.uniform(0.1, 0.5, (M, N)).astype(np.float32)
    return A, np.zeros((M, N), np.float32), np.zeros((N, N), np.float32)


def reference(A, Q, R):
    a = A.copy()
    q = np.zeros((M, N), np.float32)
    r = np.zeros((N, N), np.float32)
    for k in range(N):
        nrm = np.float32(0.0)
        for i in range(M):
            nrm += a[i, k] * a[i, k]
        r[k, k] = nrm
        for i in range(M):
            q[i, k] = a[i, k] / r[k, k]
        for j in range(k + 1, N):
            r[k, j] = np.float32(0.0)
            for i in range(M):
                r[k, j] += q[i, k] * a[i, j]
            for i in range(M):
                a[i, j] -= q[i, k] * r[k, j]
    return a, q, r


BENCHMARK = Benchmark(
    suite="polybench",
    name="gramschmidt",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0, 1, 2),
    tolerance=(5e-3, 5e-3),
)
