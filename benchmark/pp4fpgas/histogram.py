# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Histogram accumulation, where consecutive bins collide through memory."""

import numpy as np

from allo.lang import i32, kernel

from ..spec import Benchmark

INPUT_SIZE = 64
VALUE_SIZE = 256


def build():
    @kernel
    def histogram(x_in: i32[INPUT_SIZE], hist: i32[VALUE_SIZE]):
        for v in range(VALUE_SIZE):
            hist[v] = 0
        for i in range(INPUT_SIZE):
            val: i32 = x_in[i]
            hist[val] = hist[val] + 1

    return {"top": histogram}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("i"), factor=2)
    return s


def inputs(rng):
    # The reference states the precondition in[x] != in[x+1], which is what lets
    # the RAW distance be relaxed; generate a stream that respects it.
    values = np.empty(INPUT_SIZE, np.int32)
    prev = -1
    for i in range(INPUT_SIZE):
        pick = int(rng.integers(0, VALUE_SIZE))
        while pick == prev:
            pick = int(rng.integers(0, VALUE_SIZE))
        values[i] = pick
        prev = pick
    return values, np.zeros(VALUE_SIZE, np.int32)


def reference(x_in, hist):
    return (np.bincount(x_in, minlength=VALUE_SIZE).astype(np.int32),)


BENCHMARK = Benchmark(
    suite="pp4fpgas",
    name="histogram",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(1,),
)
