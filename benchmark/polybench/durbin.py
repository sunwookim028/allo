# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durbin's recurrence for a Toeplitz system: each step reads the whole prefix."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

N = 30


def build():
    @kernel
    def durbin(r: f32[N], y: f32[N]):
        y[0] = -r[0]
        beta: f32 = 1.0
        alpha: f32 = -r[0]

        for k in range(1, N):
            beta = (1.0 - alpha * alpha) * beta
            sum_: f32 = 0.0
            z: f32[N] = 0.0
            for i in range(k):
                sum_ = sum_ + r[k - i - 1] * y[i]
            alpha = -1.0 * (r[k] + sum_)
            for i2 in range(k):
                z[i2] = y[i2] + alpha * y[k - i2 - 1]
            for i3 in range(k):
                y[i3] = z[i3]
            y[k] = alpha

    return {"top": durbin}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("i2"), factor=2)
    s.unroll(s.loop("i3"), factor=2)
    return s


def inputs(rng):
    r = rng.uniform(0.01, 0.25, N).astype(np.float32)
    return r, np.zeros(N, np.float32)


def reference(r, y):
    out = np.zeros(N, np.float32)
    out[0] = -r[0]
    beta = np.float32(1.0)
    alpha = np.float32(-r[0])
    for k in range(1, N):
        beta = (np.float32(1.0) - alpha * alpha) * beta
        s = np.float32(0.0)
        for i in range(k):
            s = s + r[k - i - 1] * out[i]
        alpha = np.float32(-1.0) * (r[k] + s)
        z = out.copy()
        for i in range(k):
            z[i] = out[i] + alpha * out[k - i - 1]
        out[:k] = z[:k]
        out[k] = alpha
    return (out,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="durbin",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(1,),
    tolerance=(5e-3, 5e-3),
)
