# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two chained matrix multiplies feeding a scaled elementwise combine."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

P, R, Q, S = 16, 18, 22, 24
ALPHA, BETA = 0.1, 0.5


def build():
    @kernel
    def mm1(A: f32[P, Q], B: f32[Q, R], out_AB: f32[P, R]):
        for i0 in range(P):
            for j0 in range(R):
                for k0 in range(Q):
                    out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    @kernel
    def mm2(out_AB: f32[P, R], C: f32[R, S], out_ABC: f32[P, S]):
        for i1 in range(P):
            for j1 in range(S):
                for k1 in range(R):
                    out_ABC[i1, j1] += out_AB[i1, k1] * C[k1, j1]

    @kernel
    def ele_add(out_ABC: f32[P, S], D: f32[P, S], output: f32[P, S]):
        for i2 in range(P):
            for j2 in range(S):
                output[i2, j2] = out_ABC[i2, j2] * BETA + D[i2, j2] * ALPHA

    @kernel
    def two_mm(
        A: f32[P, Q], B: f32[Q, R], C: f32[R, S], D: f32[P, S], output: f32[P, S]
    ):
        out_AB: f32[P, R] = 0.0
        out_ABC: f32[P, S] = 0.0
        mm1(A, B, out_AB)
        mm2(out_AB, C, out_ABC)
        ele_add(out_ABC, D, output)

    return {"top": two_mm, "mm1": mm1, "mm2": mm2, "ele_add": ele_add}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    m1 = parts["mm1"].schedule()
    m1.pipeline(m1.loop("j0"), ii=1)
    m2 = parts["mm2"].schedule()
    m2.pipeline(m2.loop("j1"), ii=1)
    ele = parts["ele_add"].schedule()
    ele.pipeline(ele.loop("j2"), ii=1)
    top = parts["top"].schedule()
    top.compose(m1, m2, ele)
    return top


def _v2(parts):
    m1 = parts["mm1"].schedule()
    m1.pipeline(m1.flatten(m1.loops("i0", "j0")), ii=1)
    m2 = parts["mm2"].schedule()
    m2.pipeline(m2.flatten(m2.loops("i1", "j1")), ii=1)
    ele = parts["ele_add"].schedule()
    ele.pipeline(ele.flatten(ele.loops("i2", "j2")), ii=1)
    top = parts["top"].schedule()
    top.compose(m1, m2, ele)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (P, Q)).astype(np.float32)
    B = rng.uniform(0.01, 0.25, (Q, R)).astype(np.float32)
    C = rng.uniform(0.01, 0.25, (R, S)).astype(np.float32)
    D = rng.uniform(0.01, 0.25, (P, S)).astype(np.float32)
    return A, B, C, D, np.zeros((P, S), np.float32)


def reference(A, B, C, D, output):
    return (((A @ B) @ C) * BETA + D * ALPHA,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="two_mm",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(4,),
    tolerance=(2e-3, 2e-3),
)
