# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CORDIC rotation: sine and cosine from a chain of shifts and table subtractions."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

NUM_ITERATIONS = 16
N = 32

PHASE = np.arctan(2.0 ** -np.arange(NUM_ITERATIONS, dtype=np.float64)).astype(
    np.float32
)
GAIN = float(np.prod(1.0 / np.sqrt(1.0 + 4.0 ** -np.arange(NUM_ITERATIONS))))


def build():
    @kernel
    def cordic(theta_in: f32[N], sin_out: f32[N], cos_out: f32[N]):
        phase: f32[NUM_ITERATIONS] = PHASE
        for n in range(N):
            theta: f32 = theta_in[n]
            current_cos: f32 = GAIN
            current_sin: f32 = 0.0
            factor: f32 = 1.0
            for j in range(NUM_ITERATIONS):
                sigma: f32 = 1.0
                if theta < 0.0:
                    sigma = -1.0
                cos_shift: f32 = current_cos * sigma * factor
                sin_shift: f32 = current_sin * sigma * factor
                current_cos = current_cos - sin_shift
                current_sin = current_sin + cos_shift
                theta = theta - sigma * phase[j]
                factor = factor / 2.0
            sin_out[n] = current_sin
            cos_out[n] = current_cos

    return {"top": cordic}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("n"), ii=1)
    return s


def inputs(rng):
    theta = rng.uniform(-0.7, 0.7, N).astype(np.float32)
    return theta, np.zeros(N, np.float32), np.zeros(N, np.float32)


def reference(theta_in, sin_out, cos_out):
    return np.sin(theta_in).astype(np.float32), np.cos(theta_in).astype(np.float32)


BENCHMARK = Benchmark(
    suite="pp4fpgas",
    name="cordic",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(1, 2),
    tolerance=(1e-3, 1e-3),
)
