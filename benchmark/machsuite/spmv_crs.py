# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse matrix-vector product in CRS form: runtime trip counts and gathers."""

import numpy as np

from allo.lang import f64, i32, kernel

from ..spec import Benchmark

N = 32
NNZ = 96


def build():
    @kernel
    def crs(val: f64[NNZ], cols: i32[NNZ], row: i32[N + 1], vec: f64[N], out: f64[N]):
        for i in range(N):
            tmp_begin: i32 = row[i]
            tmp_end: i32 = row[i + 1]
            for j in range(tmp_begin, tmp_end):
                out[i] += val[j] * vec[cols[j]]

    return {"top": crs}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("j"), factor=2)
    return s


def inputs(rng):
    per_row = NNZ // N
    row = (np.arange(N + 1) * per_row).astype(np.int32)
    cols = rng.integers(0, N, NNZ).astype(np.int32)
    val = rng.uniform(0.01, 0.25, NNZ).astype(np.float64)
    vec = rng.uniform(0.01, 0.25, N).astype(np.float64)
    return val, cols, row, vec, np.zeros(N, np.float64)


def reference(val, cols, row, vec, out):
    res = np.zeros(N, np.float64)
    for i in range(N):
        for j in range(row[i], row[i + 1]):
            res[i] += val[j] * vec[cols[j]]
    return (res,)


BENCHMARK = Benchmark(
    suite="machsuite",
    name="crs",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(4,),
    tolerance=(1e-9, 1e-9),
)
