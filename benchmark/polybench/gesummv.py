# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two matrix-vector products over one vector, fused into a single loop nest."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

N = 30
ALPHA, BETA = 0.1, 0.1


def build():
    @kernel
    def compute_tmp(
        y_in: f32[N], y_out: f32[N], A: f32[N, N], B: f32[N, N], x: f32[N], tmp: f32[N]
    ):
        tt: f32[N] = 0.0
        yy: f32[N]
        for i0 in range(N):
            yy[i0] = y_in[i0]
        for i in range(N):
            for j in range(N):
                tt[i] += A[i, j] * x[j]
                yy[i] += B[i, j] * x[j]
        for i1 in range(N):
            tmp[i1] = tt[i1]
            y_out[i1] = yy[i1]

    @kernel
    def compute_y(y_in: f32[N], y_out: f32[N], tmp: f32[N]):
        for i2 in range(N):
            y_out[i2] = ALPHA * tmp[i2] + BETA * y_in[i2]

    @kernel
    def gesummv(A: f32[N, N], B: f32[N, N], x: f32[N], y: f32[N]):
        y_init: f32[N] = 0.0
        y_fifo: f32[N]
        tmp: f32[N]
        compute_tmp(y_init, y_fifo, A, B, x, tmp)
        compute_y(y_fifo, y, tmp)

    return {"top": gesummv, "compute_tmp": compute_tmp, "compute_y": compute_y}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    ct = parts["compute_tmp"].schedule()
    ct.pipeline(ct.loop("i"), ii=1)
    cy = parts["compute_y"].schedule()
    cy.pipeline(cy.loop("i2"), ii=1)
    top = parts["top"].schedule()
    top.compose(ct, cy)
    return top


def inputs(rng):
    A = rng.uniform(0.01, 0.25, (N, N)).astype(np.float32)
    B = rng.uniform(0.01, 0.25, (N, N)).astype(np.float32)
    x = rng.uniform(0.01, 0.25, N).astype(np.float32)
    return A, B, x, np.zeros(N, np.float32)


def reference(A, B, x, y):
    return (np.float32(ALPHA) * (A @ x) + np.float32(BETA) * (B @ x),)


BENCHMARK = Benchmark(
    suite="polybench",
    name="gesummv",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(3,),
    tolerance=(2e-3, 2e-3),
)
