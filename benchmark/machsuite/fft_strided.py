# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Radix-2 in-place FFT with a strided butterfly, driven by two while loops."""

import math

import numpy as np

from allo.lang import f64, i32, kernel

from ..spec import Benchmark

FFT_SIZE = 64
FFT_SIZE_HALF = FFT_SIZE // 2


def build():
    @kernel
    def fft(
        real: f64[FFT_SIZE],
        img: f64[FFT_SIZE],
        real_twid: f64[FFT_SIZE_HALF],
        img_twid: f64[FFT_SIZE_HALF],
    ):
        span: i32 = FFT_SIZE >> 1
        log: i32 = 0
        even: i32 = 0
        odd: i32 = 0
        rootindex: i32 = 0
        temp: f64 = 0.0

        while span > 0:
            odd = span
            while odd < FFT_SIZE:
                odd |= span
                even = odd ^ span
                temp = real[even] + real[odd]
                real[odd] = real[even] - real[odd]
                real[even] = temp
                temp = img[even] + img[odd]
                img[odd] = img[even] - img[odd]
                img[even] = temp
                rootindex = (even << log) & (FFT_SIZE - 1)
                if rootindex > 0:
                    temp = (
                        real_twid[rootindex] * real[odd]
                        - img_twid[rootindex] * img[odd]
                    )
                    img[odd] = (
                        real_twid[rootindex] * img[odd]
                        + img_twid[rootindex] * real[odd]
                    )
                    real[odd] = temp
                odd += 1
            span >>= 1
            log += 1

    return {"top": fft}


def _none(parts):
    return parts["top"].schedule()


def inputs(rng):
    real = rng.uniform(0.01, 0.25, FFT_SIZE).astype(np.float64)
    img = rng.uniform(0.01, 0.25, FFT_SIZE).astype(np.float64)
    real_twid = np.array(
        [math.cos(2.0 * math.pi * i / FFT_SIZE) for i in range(FFT_SIZE_HALF)],
        np.float64,
    )
    img_twid = np.array(
        [math.sin(2.0 * math.pi * i / FFT_SIZE) for i in range(FFT_SIZE_HALF)],
        np.float64,
    )
    return real, img, real_twid, img_twid


def reference(real, img, real_twid, img_twid):
    re, im = real.copy(), img.copy()
    span = FFT_SIZE >> 1
    log = 0
    while span > 0:
        odd = span
        while odd < FFT_SIZE:
            odd |= span
            even = odd ^ span
            temp = re[even] + re[odd]
            re[odd] = re[even] - re[odd]
            re[even] = temp
            temp = im[even] + im[odd]
            im[odd] = im[even] - im[odd]
            im[even] = temp
            rootindex = (even << log) & (FFT_SIZE - 1)
            if rootindex > 0:
                temp = real_twid[rootindex] * re[odd] - img_twid[rootindex] * im[odd]
                im[odd] = real_twid[rootindex] * im[odd] + img_twid[rootindex] * re[odd]
                re[odd] = temp
            odd += 1
        span >>= 1
        log += 1
    return re, im


BENCHMARK = Benchmark(
    suite="machsuite",
    name="fft",
    build=build,
    schedules={"none": _none},
    inputs=inputs,
    reference=reference,
    outputs=(0, 1),
    tolerance=(1e-9, 1e-9),
)
