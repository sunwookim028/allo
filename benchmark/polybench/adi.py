# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alternating-direction implicit solver: a Thomas sweep forward then backward."""

import numpy as np

from allo.lang import f32, index, kernel

from ..spec import Benchmark

TSTEPS, N = 4, 12

DX = 1.0 / N
DY = 1.0 / N
DT = 1.0 / TSTEPS
MUL1 = 2.0 * DT / (DX * DX)
MUL2 = 1.0 * DT / (DY * DY)

A = -MUL1 / 2.0
B = 1.0 + MUL1
C = A
D = -MUL2 / 2.0
E = 1.0 + MUL2
F = D


def build():
    @kernel
    def adi(u: f32[N, N], v: f32[N, N], p: f32[N, N], q: f32[N, N]):
        for t in range(1, TSTEPS + 1):
            for i0 in range(1, N - 1):
                v[0, i0] = 1.0
                p[i0, 0] = 0.0
                q[i0, 0] = v[0, i0]
                for j0 in range(1, N - 1):
                    p[i0, j0] = -C / (A * p[i0, j0 - 1] + B)
                    q[i0, j0] = (
                        -D * u[j0, i0 - 1]
                        + (1.0 + 2.0 * D) * u[j0, i0]
                        - F * u[j0, i0 + 1]
                        - A * q[i0, j0 - 1]
                    ) / (A * p[i0, j0 - 1] + B)
                v[N - 1, i0] = 1.0
                for j1_rev in range(N - 1):
                    j1: index = N - 2 - j1_rev
                    v[j1, i0] = p[i0, j1] * v[j1 + 1, i0] + q[i0, j1]

            for i1 in range(1, N - 1):
                u[i1, 0] = 1.0
                p[i1, 0] = 0.0
                q[i1, 0] = u[i1, 0]
                for j2 in range(1, N - 1):
                    p[i1, j2] = -F / (D * p[i1, j2 - 1] + E)
                    q[i1, j2] = (
                        -A * v[i1 - 1, j2]
                        + (1.0 + 2.0 * A) * v[i1, j2]
                        - C * v[i1 + 1, j2]
                        - D * q[i1, j2 - 1]
                    ) / (D * p[i1, j2 - 1] + E)
                u[i1, N - 1] = 1.0
                for j3_rev in range(N - 1):
                    j3: index = N - 2 - j3_rev
                    u[i1, j3] = p[i1, j3] * u[i1, j3 + 1] + q[i1, j3]

    return {"top": adi}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("j1_rev"), factor=2)
    s.unroll(s.loop("j3_rev"), factor=2)
    return s


def inputs(rng):
    u = rng.uniform(0.01, 0.25, (N, N)).astype(np.float32)
    v = rng.uniform(0.01, 0.25, (N, N)).astype(np.float32)
    return u, v, np.zeros((N, N), np.float32), np.zeros((N, N), np.float32)


def reference(u, v, p, q):
    uu, vv = u.copy(), v.copy()
    pp, qq = p.copy(), q.copy()
    a, b, c = np.float32(A), np.float32(B), np.float32(C)
    d, e, f = np.float32(D), np.float32(E), np.float32(F)
    for _ in range(1, TSTEPS + 1):
        for i in range(1, N - 1):
            vv[0, i] = np.float32(1.0)
            pp[i, 0] = np.float32(0.0)
            qq[i, 0] = vv[0, i]
            for j in range(1, N - 1):
                pp[i, j] = -c / (a * pp[i, j - 1] + b)
                qq[i, j] = (
                    -d * uu[j, i - 1]
                    + (np.float32(1.0) + np.float32(2.0) * d) * uu[j, i]
                    - f * uu[j, i + 1]
                    - a * qq[i, j - 1]
                ) / (a * pp[i, j - 1] + b)
            vv[N - 1, i] = np.float32(1.0)
            for j in range(N - 2, -1, -1):
                vv[j, i] = pp[i, j] * vv[j + 1, i] + qq[i, j]
        for i in range(1, N - 1):
            uu[i, 0] = np.float32(1.0)
            pp[i, 0] = np.float32(0.0)
            qq[i, 0] = uu[i, 0]
            for j in range(1, N - 1):
                pp[i, j] = -f / (d * pp[i, j - 1] + e)
                qq[i, j] = (
                    -a * vv[i - 1, j]
                    + (np.float32(1.0) + np.float32(2.0) * a) * vv[i, j]
                    - c * vv[i + 1, j]
                    - d * qq[i, j - 1]
                ) / (d * pp[i, j - 1] + e)
            uu[i, N - 1] = np.float32(1.0)
            for j in range(N - 2, -1, -1):
                uu[i, j] = pp[i, j] * uu[i, j + 1] + qq[i, j]
    return uu, vv, pp, qq


BENCHMARK = Benchmark(
    suite="polybench",
    name="adi",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0, 1, 2, 3),
    tolerance=(5e-3, 5e-3),
)
