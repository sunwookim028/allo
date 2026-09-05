# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deriche edge detector: four IIR passes, each a scalar-carried shift register."""

import math

import numpy as np

from allo.lang import f32, index, kernel

from ..spec import Benchmark

W, H = 48, 32

ALPHA = 0.25
K = (
    (1.0 - math.exp(-ALPHA))
    * (1.0 - math.exp(-ALPHA))
    / (1.0 + 2.0 * ALPHA * math.exp(-ALPHA) - math.exp(2.0 * ALPHA))
)
A1 = K
A2 = K * math.exp(-ALPHA) * (ALPHA - 1.0)
A3 = K * math.exp(-ALPHA) * (ALPHA + 1.0)
A4 = -K * math.exp(-2.0 * ALPHA)
A5 = K
A6 = K * math.exp(-ALPHA) * (ALPHA - 1.0)
A7 = K * math.exp(-ALPHA) * (ALPHA + 1.0)
A8 = -K * math.exp(-2.0 * ALPHA)
B1 = 2.0 ** (-ALPHA)
B2 = -math.exp(-2.0 * ALPHA)
C1 = 1.0
C2 = 1.0


def build():
    @kernel
    def deriche(imgIn: f32[W, H], imgOut: f32[W, H], y1: f32[W, H], y2: f32[W, H]):
        for i0 in range(W):
            ym1: f32 = 0.0
            ym2: f32 = 0.0
            xm1: f32 = 0.0
            for j0 in range(H):
                y1[i0, j0] = A1 * imgIn[i0, j0] + A2 * xm1 + B1 * ym1 + B2 * ym2
                xm1 = imgIn[i0, j0]
                ym2 = ym1
                ym1 = y1[i0, j0]

        for i1 in range(W):
            yp1: f32 = 0.0
            yp2: f32 = 0.0
            xp1: f32 = 0.0
            xp2: f32 = 0.0
            for j1_inv in range(H):
                j1: index = H - 1 - j1_inv
                y2[i1, j1] = A3 * xp1 + A4 * xp2 + B1 * yp1 + B2 * yp2
                xp2 = xp1
                xp1 = imgIn[i1, j1]
                yp2 = yp1
                yp1 = y2[i1, j1]

        for i2 in range(W):
            for j2 in range(H):
                imgOut[i2, j2] = C1 * (y1[i2, j2] + y2[i2, j2])

        for j3 in range(H):
            tm1: f32 = 0.0
            ym1c: f32 = 0.0
            ym2c: f32 = 0.0
            for i3 in range(W):
                y1[i3, j3] = A5 * imgOut[i3, j3] + A6 * tm1 + B1 * ym1c + B2 * ym2c
                tm1 = imgOut[i3, j3]
                ym2c = ym1c
                ym1c = y1[i3, j3]

        for j4 in range(H):
            tp1: f32 = 0.0
            tp2: f32 = 0.0
            yp1c: f32 = 0.0
            yp2c: f32 = 0.0
            for i4_inv in range(W):
                i4: index = W - 1 - i4_inv
                y2[i4, j4] = A7 * tp1 + A8 * tp2 + B1 * yp1c + B2 * yp2c
                tp2 = tp1
                tp1 = imgOut[i4, j4]
                yp2c = yp1c
                yp1c = y2[i4, j4]

        for i5 in range(W):
            for j5 in range(H):
                imgOut[i5, j5] = C2 * (y1[i5, j5] + y2[i5, j5])

    return {"top": deriche}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("i2"), ii=1)
    s.pipeline(s.loop("i5"), ii=1)
    return s


def inputs(rng):
    imgIn = rng.uniform(0.01, 0.25, (W, H)).astype(np.float32)
    return (
        imgIn,
        np.zeros((W, H), np.float32),
        np.zeros((W, H), np.float32),
        np.zeros((W, H), np.float32),
    )


def reference(imgIn, imgOut, y1, y2):
    f = np.float32
    yy1 = np.zeros((W, H), np.float32)
    yy2 = np.zeros((W, H), np.float32)
    for i in range(W):
        ym1 = ym2 = xm1 = f(0.0)
        for j in range(H):
            yy1[i, j] = f(A1) * imgIn[i, j] + f(A2) * xm1 + f(B1) * ym1 + f(B2) * ym2
            xm1, ym2, ym1 = imgIn[i, j], ym1, yy1[i, j]
    for i in range(W):
        yp1 = yp2 = xp1 = xp2 = f(0.0)
        for j in range(H - 1, -1, -1):
            yy2[i, j] = f(A3) * xp1 + f(A4) * xp2 + f(B1) * yp1 + f(B2) * yp2
            xp2, xp1, yp2, yp1 = xp1, imgIn[i, j], yp1, yy2[i, j]
    out = (f(C1) * (yy1 + yy2)).astype(np.float32)
    for j in range(H):
        tm1 = ym1c = ym2c = f(0.0)
        for i in range(W):
            yy1[i, j] = f(A5) * out[i, j] + f(A6) * tm1 + f(B1) * ym1c + f(B2) * ym2c
            tm1, ym2c, ym1c = out[i, j], ym1c, yy1[i, j]
    for j in range(H):
        tp1 = tp2 = yp1c = yp2c = f(0.0)
        for i in range(W - 1, -1, -1):
            yy2[i, j] = f(A7) * tp1 + f(A8) * tp2 + f(B1) * yp1c + f(B2) * yp2c
            tp2, tp1, yp2c, yp1c = tp1, out[i, j], yp1c, yy2[i, j]
    return (f(C2) * (yy1 + yy2)).astype(np.float32), yy1, yy2


BENCHMARK = Benchmark(
    suite="polybench",
    name="deriche",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(1, 2, 3),
    tolerance=(5e-3, 5e-3),
)
