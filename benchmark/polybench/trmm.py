# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triangular matrix multiply in place, then a scalar scaling of the result."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

M, N = 20, 30
ALPHA = 1.5


def build():
    @kernel
    def S0(A: f32[M, M], B: f32[M, N]):
        for i1 in range(M):
            for j1 in range(N):
                for k1 in range(M):
                    if k1 > i1:
                        B[i1, j1] += A[k1, i1] * B[k1, j1]

    @kernel
    def S1(B: f32[M, N]):
        for i0 in range(M):
            for j0 in range(N):
                B[i0, j0] = B[i0, j0] * ALPHA

    @kernel
    def trmm(A: f32[M, M], B: f32[M, N]):
        S0(A, B)
        S1(B)

    return {"top": trmm, "S0": S0, "S1": S1}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s0 = parts["S0"].schedule()
    s0.pipeline(s0.loop("j1"), ii=1)
    s1 = parts["S1"].schedule()
    s1.pipeline(s1.loop("j0"), ii=1)
    top = parts["top"].schedule()
    top.compose(s0, s1)
    return top


def _v2(parts):
    s0 = parts["S0"].schedule()
    s0.pipeline(s0.flatten(s0.loops("i1", "j1")), ii=1)
    s1 = parts["S1"].schedule()
    s1.pipeline(s1.flatten(s1.loops("i0", "j0")), ii=1)
    top = parts["top"].schedule()
    top.compose(s0, s1)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (M, M)).astype(np.float32)
    B = rng.uniform(0.01, 0.25, (M, N)).astype(np.float32)
    return A, B


def reference(A, B):
    out = B.copy()
    for i in range(M):
        for j in range(N):
            for k in range(M):
                if k > i:
                    out[i, j] += A[k, i] * out[k, j]
    return (out * np.float32(ALPHA),)


BENCHMARK = Benchmark(
    suite="polybench",
    name="trmm",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(1,),
    tolerance=(2e-3, 2e-3),
    skip={
        "v2": "flattening i1 x j1 leaves the triangular predicate a real "
        "floordiv-30 cone far past the 300 MHz period; the scheduler would "
        "derate the clock to fit it whole, so the variant is not comparable "
        "at the bed's frequency"
    },
)
