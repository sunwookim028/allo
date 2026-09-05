# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""2-D finite-difference time-domain: four field sweeps per time step."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

TMAX, NX, NY = 10, 20, 24


def build():
    @kernel
    def fdtd_2d(ex: f32[NX, NY], ey: f32[NX, NY], hz: f32[NX, NY], fict: f32[TMAX]):
        for m in range(TMAX):
            for j0 in range(NY):
                ey[0, j0] = fict[m]
            for i1 in range(1, NX):
                for j1 in range(NY):
                    ey[i1, j1] = ey[i1, j1] - 0.5 * (hz[i1, j1] - hz[i1 - 1, j1])
            for i2 in range(NX):
                for j2 in range(1, NY):
                    ex[i2, j2] = ex[i2, j2] - 0.5 * (hz[i2, j2] - hz[i2, j2 - 1])
            for i3 in range(NX - 1):
                for j3 in range(NY - 1):
                    hz[i3, j3] = hz[i3, j3] - 0.7 * (
                        ex[i3, j3 + 1] - ex[i3, j3] + ey[i3 + 1, j3] - ey[i3, j3]
                    )

    return {"top": fdtd_2d}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("i1"), ii=1)
    s.pipeline(s.loop("i2"), ii=1)
    s.pipeline(s.loop("i3"), ii=1)
    return s


def inputs(rng):
    ex = rng.uniform(0.01, 0.25, (NX, NY)).astype(np.float32)
    ey = rng.uniform(0.01, 0.25, (NX, NY)).astype(np.float32)
    hz = rng.uniform(0.01, 0.25, (NX, NY)).astype(np.float32)
    fict = rng.uniform(0.01, 0.25, TMAX).astype(np.float32)
    return ex, ey, hz, fict


def reference(ex, ey, hz, fict):
    e_x, e_y, h_z = ex.copy(), ey.copy(), hz.copy()
    half, sev = np.float32(0.5), np.float32(0.7)
    for m in range(TMAX):
        e_y[0, :] = fict[m]
        e_y[1:, :] = e_y[1:, :] - half * (h_z[1:, :] - h_z[:-1, :])
        e_x[:, 1:] = e_x[:, 1:] - half * (h_z[:, 1:] - h_z[:, :-1])
        h_z[:-1, :-1] = h_z[:-1, :-1] - sev * (
            e_x[:-1, 1:] - e_x[:-1, :-1] + e_y[1:, :-1] - e_y[:-1, :-1]
        )
    return e_x, e_y, h_z


BENCHMARK = Benchmark(
    suite="polybench",
    name="fdtd_2d",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(0, 1, 2),
    tolerance=(5e-3, 5e-3),
)
