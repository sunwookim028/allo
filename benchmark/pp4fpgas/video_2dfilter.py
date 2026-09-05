# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""3x3 Gaussian blur over an RGB image, with the border left black."""

import numpy as np

from allo.lang import i32, kernel

from ..spec import Benchmark

HEIGHT, WIDTH = 20, 20
WEIGHTS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.int32)


def build():
    @kernel
    def video_2dfilter(
        pixel_in: i32[HEIGHT, WIDTH, 3], pixel_out: i32[HEIGHT, WIDTH, 3]
    ):
        h: i32[3, 3] = WEIGHTS
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if row == 0 or col == 0 or row == HEIGHT - 1 or col == WIDTH - 1:
                    for ch0 in range(3):
                        pixel_out[row, col, ch0] = 0
                else:
                    for ch1 in range(3):
                        acc: i32 = 0
                        for i in range(3):
                            for j in range(3):
                                acc += pixel_in[row + i - 1, col + j - 1, ch1] * h[i, j]
                        pixel_out[row, col, ch1] = acc // 16

    return {"top": video_2dfilter}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("col"), ii=1)
    return s


def _v2(parts):
    s = parts["top"].schedule()
    s.partition(s.buffer("pixel_in"), dim=3, kind=s.Cyclic, factor=3)
    s.partition(s.buffer("pixel_out"), dim=3, kind=s.Cyclic, factor=3)
    s.pipeline(s.loop("col"), ii=1)
    return s


def inputs(rng):
    image = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.int32)
    return image, np.zeros((HEIGHT, WIDTH, 3), np.int32)


def reference(pixel_in, pixel_out):
    out = np.zeros((HEIGHT, WIDTH, 3), np.int32)
    for row in range(1, HEIGHT - 1):
        for col in range(1, WIDTH - 1):
            window = pixel_in[row - 1 : row + 2, col - 1 : col + 2]
            out[row, col] = (window * WEIGHTS[:, :, None]).sum(axis=(0, 1)) // 16
    return (out,)


BENCHMARK = Benchmark(
    suite="pp4fpgas",
    name="video_2dfilter",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(1,),
)
