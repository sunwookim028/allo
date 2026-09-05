# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""All-pairs shortest paths: a guarded min-relax over every intermediate node."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

N = 40


def build():
    @kernel
    def floyd_warshall(path: f32[N, N]):
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    path_: f32 = path[i, k] + path[k, j]
                    if path[i, j] >= path_:
                        path[i, j] = path_

    return {"top": floyd_warshall}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("i"), ii=1)
    return s


def inputs(rng):
    path = rng.uniform(0.1, 4.0, (N, N)).astype(np.float32)
    np.fill_diagonal(path, np.float32(0.0))
    return (path,)


def reference(path):
    out = path.copy()
    for k in range(N):
        cand = out[:, k, None] + out[None, k, :]
        out = np.where(out >= cand, cand, out).astype(np.float32)
    return (out,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="floyd_warshall",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0,),
    tolerance=(2e-3, 2e-3),
)
