# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-resolution analysis: every (r,q) fibre of a 3-D tensor times one matrix."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

Q, R, P = 8, 10, 12


def build():
    @kernel
    def doitgen(A: f32[R, Q, P], x: f32[P, P], sum_: f32[P]):
        for r in range(R):
            for q in range(Q):
                for p in range(P):
                    sum_[p] = 0.0
                    for s in range(P):
                        sum_[p] = sum_[p] + A[r, q, s] * x[s, p]
                for p1 in range(P):
                    A[r, q, p1] = sum_[p1]

    return {"top": doitgen}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("p"), ii=1)
    s.pipeline(s.loop("p1"), ii=1)
    return s


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (R, Q, P)).astype(np.float32)
    x = rng.uniform(0.01, 0.25, (P, P)).astype(np.float32)
    return A, x, np.zeros(P, np.float32)


def reference(A, x, sum_):
    out = A.copy()
    for r in range(R):
        for q in range(Q):
            out[r, q, :] = out[r, q, :] @ x
    return (out,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="doitgen",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0,),
    tolerance=(2e-3, 2e-3),
)
