# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inclusive prefix sum, whose every iteration reads the one before it."""

import numpy as np

from allo.lang import i32, kernel

from ..spec import Benchmark

SIZE = 128


def build():
    @kernel
    def prefixsum(x_in: i32[SIZE], out: i32[SIZE]):
        out[0] = x_in[0]
        for i in range(1, SIZE):
            out[i] = out[i - 1] + x_in[i]

    return {"top": prefixsum}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("i"), factor=2)
    return s


def _v2(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("i"), factor=4)
    return s


def inputs(rng):
    return rng.integers(-8, 8, SIZE, dtype=np.int32), np.zeros(SIZE, np.int32)


def reference(x_in, out):
    return (np.cumsum(x_in).astype(np.int32),)


BENCHMARK = Benchmark(
    suite="pp4fpgas",
    name="prefixsum",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(1,),
)
