# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse matrix-vector product in ELLPACK form: fixed row width, guarded gather."""

import numpy as np

from allo.lang import f64, i32, kernel

from ..spec import Benchmark

N = 32
L = 4


def build():
    @kernel
    def ellpack(NZ: f64[N * L], cols: i32[N * L], vec: f64[N], out: f64[N]):
        for i in range(N):
            for j in range(L):
                idx: i32 = j + i * L
                if cols[idx] != -1:
                    out[i] += NZ[idx] * vec[cols[idx]]

    return {"top": ellpack}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("i"), ii=1)
    return s


def inputs(rng):
    cols = rng.integers(0, N, N * L).astype(np.int32)
    cols[rng.random(N * L) < 0.1] = np.int32(-1)
    NZ = rng.uniform(0.01, 0.25, N * L).astype(np.float64)
    vec = rng.uniform(0.01, 0.25, N).astype(np.float64)
    return NZ, cols, vec, np.zeros(N, np.float64)


def reference(NZ, cols, vec, out):
    res = np.zeros(N, np.float64)
    for i in range(N):
        for j in range(L):
            idx = j + i * L
            if cols[idx] != -1:
                res[i] += NZ[idx] * vec[cols[idx]]
    return (res,)


BENCHMARK = Benchmark(
    suite="machsuite",
    name="ellpack",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(3,),
    tolerance=(1e-9, 1e-9),
)
