# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lennard-Jones force on each atom over a neighbour list: a gather-heavy reduction."""

import numpy as np

from allo.lang import f64, i32, kernel

from ..spec import Benchmark

N_ATOMS = 32
MAX_NEIGHBORS = 8
LJ1 = 1.5
LJ2 = 2.0
DOMAIN_EDGE = 20.0


def build():
    @kernel
    def md_x(
        position_x: f64[N_ATOMS],
        position_y: f64[N_ATOMS],
        position_z: f64[N_ATOMS],
        NL: i32[N_ATOMS * MAX_NEIGHBORS],
        force_x: f64[N_ATOMS],
    ):
        i_x: f64 = 0.0
        i_y: f64 = 0.0
        i_z: f64 = 0.0
        jidx: i32 = 0
        j_x: f64 = 0.0
        j_y: f64 = 0.0
        j_z: f64 = 0.0
        delx: f64 = 0.0
        dely: f64 = 0.0
        delz: f64 = 0.0
        r2inv: f64 = 0.0
        r6inv: f64 = 0.0
        potential: f64 = 0.0
        force: f64 = 0.0
        fx: f64 = 0.0

        for i in range(N_ATOMS):
            i_x = position_x[i]
            i_y = position_y[i]
            i_z = position_z[i]
            fx = 0.0
            for j in range(MAX_NEIGHBORS):
                jidx = NL[i * MAX_NEIGHBORS + j]
                j_x = position_x[jidx]
                j_y = position_y[jidx]
                j_z = position_z[jidx]
                delx = i_x - j_x
                dely = i_y - j_y
                delz = i_z - j_z
                if (delx * delx + dely * dely + delz * delz) == 0:
                    r2inv = (DOMAIN_EDGE * DOMAIN_EDGE * 3.0) * 1000
                else:
                    r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
                r6inv = r2inv * r2inv * r2inv
                potential = r6inv * (LJ1 * r6inv - LJ2)
                force = r2inv * potential
                fx += delx * force
            force_x[i] = fx

    return {"top": md_x}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("i"), ii=1)
    return s


def inputs(rng):
    px = rng.uniform(0.0, 20.0, N_ATOMS).astype(np.float64)
    py = rng.uniform(0.0, 20.0, N_ATOMS).astype(np.float64)
    pz = rng.uniform(0.0, 20.0, N_ATOMS).astype(np.float64)
    nl = rng.integers(0, N_ATOMS, N_ATOMS * MAX_NEIGHBORS).astype(np.int32)
    return px, py, pz, nl, np.zeros(N_ATOMS, np.float64)


def reference(position_x, position_y, position_z, NL, force_x):
    out = np.zeros(N_ATOMS, np.float64)
    for i in range(N_ATOMS):
        fx = 0.0
        for j in range(MAX_NEIGHBORS):
            jidx = NL[i * MAX_NEIGHBORS + j]
            delx = position_x[i] - position_x[jidx]
            dely = position_y[i] - position_y[jidx]
            delz = position_z[i] - position_z[jidx]
            d = delx * delx + dely * dely + delz * delz
            if d == 0:
                r2inv = (DOMAIN_EDGE * DOMAIN_EDGE * 3.0) * 1000
            else:
                r2inv = 1.0 / d
            r6inv = r2inv * r2inv * r2inv
            potential = r6inv * (LJ1 * r6inv - LJ2)
            fx += delx * (r2inv * potential)
        out[i] = fx
    return (out,)


BENCHMARK = Benchmark(
    suite="machsuite",
    name="md_x",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(4,),
    tolerance=(1e-9, 1e-9),
)
