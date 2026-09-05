# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FIR filter over a block of samples, carrying a shift-register delay line."""

import numpy as np

from allo.lang import i32, kernel

from ..spec import Benchmark

NUM_TAPS = 16
N = 128


def build():
    @kernel
    def block_fir(x_in: i32[N], taps: i32[NUM_TAPS], output: i32[N]):
        delay_line: i32[NUM_TAPS] = 0
        for j in range(N):
            result: i32 = 0
            for i in range(NUM_TAPS - 1):
                delay_line[NUM_TAPS - 1 - i] = delay_line[NUM_TAPS - 2 - i]
            delay_line[0] = x_in[j]
            for k in range(NUM_TAPS):
                result += delay_line[k] * taps[k]
            output[j] = result

    return {"top": block_fir}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("j"), ii=1)
    return s


def _v2(parts):
    s = parts["top"].schedule()
    s.partition(s.buffer("delay_line"), dim=1, kind=s.Complete)
    s.partition(s.buffer("taps"), dim=1, kind=s.Cyclic, factor=4)
    s.pipeline(s.loop("j"), ii=1)
    return s


def inputs(rng):
    x_in = rng.integers(-16, 16, N, dtype=np.int32)
    taps = rng.integers(-4, 4, NUM_TAPS, dtype=np.int32)
    return x_in, taps, np.zeros(N, np.int32)


def reference(x_in, taps, output):
    padded = np.concatenate([np.zeros(NUM_TAPS - 1, np.int32), x_in])
    out = np.array(
        [np.dot(padded[j : j + NUM_TAPS][::-1], taps) for j in range(N)], np.int32
    )
    return (out,)


BENCHMARK = Benchmark(
    suite="pp4fpgas",
    name="block_fir",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(2,),
)
