# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integer matrix-vector product, one dot product per row."""

import numpy as np

from allo.lang import i32, kernel

from ..spec import Benchmark

SIZE = 32


def build():
    @kernel
    def matrix_vector(M: i32[SIZE, SIZE], v_in: i32[SIZE], v_out: i32[SIZE]):
        for i in range(SIZE):
            acc: i32 = 0
            for j in range(SIZE):
                acc += v_in[j] * M[i, j]
            v_out[i] = acc

    return {"top": matrix_vector}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("i"), ii=1)
    return s


def _v2(parts):
    s = parts["top"].schedule()
    s.partition(s.buffer("M"), dim=2, kind=s.Cyclic, factor=4)
    s.partition(s.buffer("v_in"), dim=1, kind=s.Cyclic, factor=4)
    s.pipeline(s.loop("i"), ii=1)
    return s


def inputs(rng):
    M = rng.integers(-8, 8, (SIZE, SIZE), dtype=np.int32)
    v_in = rng.integers(-8, 8, SIZE, dtype=np.int32)
    return M, v_in, np.zeros(SIZE, np.int32)


def reference(M, v_in, v_out):
    return (M @ v_in,)


BENCHMARK = Benchmark(
    suite="pp4fpgas",
    name="matrix_vector",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(2,),
)
