# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column means and standard deviations, then the centred correlation matrix."""

import numpy as np

from allo.lang import f32, kernel
from allo.operators import math as amath

from ..spec import Benchmark

M, N = 20, 26
N_FLOAT = float(N)
EPSILON = 1e-5


def build():
    @kernel
    def compute_mean(data: f32[N, M], mean: f32[M]):
        for x in range(M):
            total: f32 = 0.0
            for k in range(N):
                total += data[k, x]
            mean[x] = total / N

    @kernel
    def compute_stddev(data: f32[N, M], mean: f32[M], mean_out: f32[M], stddev: f32[M]):
        for x2 in range(M):
            variance: f32 = 0.0
            for m in range(N):
                variance += (data[m, x2] - mean[x2]) * (data[m, x2] - mean[x2])
            stddev[x2] = amath.sqrt(variance / N_FLOAT)
            mean_out[x2] = mean[x2]
            if stddev[x2] <= EPSILON:
                stddev[x2] = 1.0

    @kernel
    def center_reduce(
        data: f32[N, M], data_out: f32[N, M], mean: f32[M], stddev: f32[M]
    ):
        for x3 in range(N):
            for y3 in range(M):
                d: f32 = data[x3, y3]
                d -= mean[y3]
                d /= amath.sqrt(N_FLOAT) * stddev[y3]
                data_out[x3, y3] = d

    @kernel
    def compute_corr(data: f32[N, M], corr: f32[M, M]):
        for i in range(M - 1):
            corr[i, i] = 1.0
            for j in range(M):
                if j > i:
                    corr_v: f32 = 0.0
                    for k4 in range(N):
                        corr_v += data[k4, i] * data[k4, j]
                    corr[j, i] = corr_v
                    corr[i, j] = corr_v
        corr[M - 1, M - 1] = 1.0

    @kernel
    def correlation(
        data_mean: f32[N, M],
        data_stddev: f32[N, M],
        data_for_center: f32[N, M],
        corr: f32[M, M],
    ):
        mean: f32[M] = 0.0
        mean_passed_on: f32[M] = 0.0
        stddev: f32[M] = 0.0
        compute_mean(data_mean, mean)
        compute_stddev(data_stddev, mean, mean_passed_on, stddev)
        data_centered: f32[N, M] = 0.0
        center_reduce(data_for_center, data_centered, mean_passed_on, stddev)
        compute_corr(data_centered, corr)

    return {
        "top": correlation,
        "compute_mean": compute_mean,
        "compute_stddev": compute_stddev,
        "center_reduce": center_reduce,
        "compute_corr": compute_corr,
    }


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    cm = parts["compute_mean"].schedule()
    cm.pipeline(cm.loop("x"), ii=1)
    cr = parts["center_reduce"].schedule()
    cr.pipeline(cr.loop("x3"), ii=1)
    top = parts["top"].schedule()
    top.compose(cm, cr)
    return top


def inputs(rng):
    data = rng.uniform(0.01, 0.25, (N, M)).astype(np.float32)
    return data, data.copy(), data.copy(), np.zeros((M, M), np.float32)


def reference(data_mean, data_stddev, data_for_center, corr):
    mean = data_mean.sum(axis=0) / np.float32(N)
    var = ((data_stddev - mean) ** 2).sum(axis=0) / np.float32(N_FLOAT)
    stddev = np.sqrt(var).astype(np.float32)
    stddev[stddev <= np.float32(EPSILON)] = np.float32(1.0)
    centred = (data_for_center - mean) / (np.float32(np.sqrt(N_FLOAT)) * stddev)
    out = (centred.T @ centred).astype(np.float32)
    np.fill_diagonal(out, np.float32(1.0))
    return (out,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="correlation",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(3,),
    tolerance=(5e-3, 5e-3),
    skip={
        # sqrt now lowers (none passes); v1 hits a separate exact-scheduler crash
        # (IndexError: absl::btree_map::at) placing center_reduce at II=20 under
        # exact only (heuristic compiles it). Unrelated to sqrt.
        "v1": "exact scheduler crashes placing center_reduce (II=20); heuristic ok",
    },
)
