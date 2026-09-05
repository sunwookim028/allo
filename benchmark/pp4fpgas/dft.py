# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct discrete Fourier transform against a precomputed twiddle table."""

import numpy as np

from allo.lang import f32, i32, kernel

from ..spec import Benchmark

N = 64

_ANGLE = 2.0 * np.pi * np.arange(N) / N
COS_TABLE = np.cos(_ANGLE).astype(np.float32)
SIN_TABLE = (-np.sin(_ANGLE)).astype(np.float32)


def build():
    @kernel
    def dft(sample_real: f32[N], sample_imag: f32[N]):
        cos_table: f32[N] = COS_TABLE
        sin_table: f32[N] = SIN_TABLE
        temp_real: f32[N] = 0.0
        temp_imag: f32[N] = 0.0

        for i in range(N):
            for j in range(N):
                w: i32 = (i * j) % N
                c: f32 = cos_table[w]
                s: f32 = sin_table[w]
                temp_real[i] += sample_real[j] * c - sample_imag[j] * s
                temp_imag[i] += sample_real[j] * s + sample_imag[j] * c

        for k in range(N):
            sample_real[k] = temp_real[k]
            sample_imag[k] = temp_imag[k]

    return {"top": dft}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("j"), factor=4)
    return s


def _v2(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("j"), factor=8)
    return s


def inputs(rng):
    real = rng.uniform(-1.0, 1.0, N).astype(np.float32)
    imag = rng.uniform(-1.0, 1.0, N).astype(np.float32)
    return real, imag


def reference(sample_real, sample_imag):
    spectrum = np.fft.fft(sample_real + 1j * sample_imag)
    return spectrum.real.astype(np.float32), spectrum.imag.astype(np.float32)


BENCHMARK = Benchmark(
    suite="pp4fpgas",
    name="dft",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(0, 1),
    tolerance=(1e-3, 1e-3),
)
