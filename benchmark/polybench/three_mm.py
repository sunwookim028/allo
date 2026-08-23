# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two independent matrix multiplies whose results feed a third."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

P, R, Q, T, S = 16, 18, 20, 22, 24


def build():
    @kernel
    def mm1(A: f32[P, Q], B: f32[Q, R], out_AB: f32[P, R]):
        for i0 in range(P):
            for j0 in range(R):
                for k0 in range(Q):
                    out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    @kernel
    def mm2(C: f32[R, S], D: f32[S, T], out_CD: f32[R, T]):
        for i1 in range(R):
            for j1 in range(T):
                for k1 in range(S):
                    out_CD[i1, j1] += C[i1, k1] * D[k1, j1]

    @kernel
    def mm3(out_AB: f32[P, R], out_CD: f32[R, T], out_ABC: f32[P, T]):
        for i2 in range(P):
            for j2 in range(T):
                for k2 in range(R):
                    out_ABC[i2, j2] += out_AB[i2, k2] * out_CD[k2, j2]

    @kernel
    def three_mm(
        A: f32[P, Q], B: f32[Q, R], C: f32[R, S], D: f32[S, T], output: f32[P, T]
    ):
        out_AB: f32[P, R] = 0.0
        out_CD: f32[R, T] = 0.0
        mm1(A, B, out_AB)
        mm2(C, D, out_CD)
        mm3(out_AB, out_CD, output)

    return {"top": three_mm, "mm1": mm1, "mm2": mm2, "mm3": mm3}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    m1 = parts["mm1"].schedule()
    m1.pipeline(m1.loop("j0"), ii=1)
    m2 = parts["mm2"].schedule()
    m2.pipeline(m2.loop("j1"), ii=1)
    m3 = parts["mm3"].schedule()
    m3.pipeline(m3.loop("j2"), ii=1)
    top = parts["top"].schedule()
    top.compose(m1, m2, m3)
    return top


def _v2(parts):
    m1 = parts["mm1"].schedule()
    m1.pipeline(m1.flatten(m1.loops("i0", "j0")), ii=1)
    m2 = parts["mm2"].schedule()
    m2.pipeline(m2.flatten(m2.loops("i1", "j1")), ii=1)
    m3 = parts["mm3"].schedule()
    m3.pipeline(m3.flatten(m3.loops("i2", "j2")), ii=1)
    top = parts["top"].schedule()
    top.compose(m1, m2, m3)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (P, Q)).astype(np.float32)
    B = rng.uniform(0.01, 0.25, (Q, R)).astype(np.float32)
    C = rng.uniform(0.01, 0.25, (R, S)).astype(np.float32)
    D = rng.uniform(0.01, 0.25, (S, T)).astype(np.float32)
    return A, B, C, D, np.zeros((P, T), np.float32)


def reference(A, B, C, D, output):
    return ((A @ B) @ (C @ D),)


BENCHMARK = Benchmark(
    suite="polybench",
    name="three_mm",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(4,),
    tolerance=(2e-3, 2e-3),
)
