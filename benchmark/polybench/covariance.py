# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column means, then every pairwise centred inner product of the columns."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

M, N = 20, 26


def build():
    @kernel
    def covariance(data: f32[N, M], mean: f32[M], cov: f32[M, M]):
        for x in range(M):
            total: f32 = 0.0
            for k in range(N):
                total += data[k, x]
            mean[x] = total / N

        for i in range(M):
            for j in range(M):
                acc: f32 = 0.0
                for p in range(N):
                    acc += (data[p, i] - mean[i]) * (data[p, j] - mean[j])
                cov[i, j] = acc / (N - 1)

    return {"top": covariance}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("x"), ii=1)
    s.pipeline(s.loop("j"), ii=1)
    return s


def inputs(rng):
    data = rng.uniform(0.01, 0.25, (N, M)).astype(np.float32)
    return data, np.zeros(M, np.float32), np.zeros((M, M), np.float32)


def reference(data, mean, cov):
    m = data.sum(axis=0) / np.float32(N)
    centred = data - m
    return m, (centred.T @ centred) / np.float32(N - 1)


BENCHMARK = Benchmark(
    suite="polybench",
    name="covariance",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(1, 2),
    tolerance=(2e-3, 2e-3),
)
