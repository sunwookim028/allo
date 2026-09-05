# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insertion sort, whose inner shift loop runs a data-dependent number of times."""

import numpy as np

from allo.lang import f32, i32, kernel

from ..spec import Benchmark

SIZE = 32


def build():
    @kernel
    def insertion_sort(A: f32[SIZE]):
        for i in range(1, SIZE):
            item: f32 = A[i]
            j: i32 = i
            shifting: i32 = 1
            while shifting == 1:
                if j > 0:
                    if A[j - 1] > item:
                        A[j] = A[j - 1]
                        j -= 1
                    else:
                        shifting = 0
                else:
                    shifting = 0
            A[j] = item

    return {"top": insertion_sort}


def _none(parts):
    return parts["top"].schedule()


def inputs(rng):
    return (rng.uniform(-1.0, 1.0, SIZE).astype(np.float32),)


def reference(A):
    return (np.sort(A),)


BENCHMARK = Benchmark(
    suite="pp4fpgas",
    name="insertion_sort",
    build=build,
    schedules={"none": _none},
    inputs=inputs,
    reference=reference,
    outputs=(0,),
)
