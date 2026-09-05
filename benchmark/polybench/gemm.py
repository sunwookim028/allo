# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense matrix multiply followed by a scaled elementwise add."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

P, R, Q = 20, 25, 30
BETA = 0.1


def build():
    @kernel
    def mm1(A: f32[P, Q], B: f32[Q, R], out_AB: f32[P, R]):
        for i0 in range(P):
            for j0 in range(R):
                for k0 in range(Q):
                    out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    @kernel
    def ele_add(out_AB: f32[P, R], C: f32[P, R], output: f32[P, R]):
        for i2 in range(P):
            for j2 in range(R):
                output[i2, j2] = BETA * C[i2, j2] + out_AB[i2, j2]

    @kernel
    def gemm(A: f32[P, Q], B: f32[Q, R], C: f32[P, R], output: f32[P, R]):
        out_AB: f32[P, R] = 0.0
        mm1(A, B, out_AB)
        ele_add(out_AB, C, output)

    return {"top": gemm, "mm1": mm1, "ele_add": ele_add}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    mm1 = parts["mm1"].schedule()
    mm1.pipeline(mm1.loop("j0"), ii=1)
    ele = parts["ele_add"].schedule()
    ele.pipeline(ele.loop("j2"), ii=1)
    top = parts["top"].schedule()
    top.compose(mm1, ele)
    return top


def _v2(parts):
    mm1 = parts["mm1"].schedule()
    mm1.partition(mm1.buffer("A"), dim=2, kind=mm1.Cyclic, factor=4)
    mm1.partition(mm1.buffer("B"), dim=1, kind=mm1.Cyclic, factor=4)
    mm1.pipeline(mm1.flatten(mm1.loops("i0", "j0")), ii=1)
    ele = parts["ele_add"].schedule()
    ele.pipeline(ele.flatten(ele.loops("i2", "j2")), ii=1)
    top = parts["top"].schedule()
    top.compose(mm1, ele)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (P, Q)).astype(np.float32)
    B = rng.uniform(0.01, 0.25, (Q, R)).astype(np.float32)
    C = rng.uniform(0.01, 0.25, (P, R)).astype(np.float32)
    return A, B, C, np.zeros((P, R), np.float32)


def reference(A, B, C, output):
    return (BETA * C + A @ B,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="gemm",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(3,),
    tolerance=(2e-3, 2e-3),
)
