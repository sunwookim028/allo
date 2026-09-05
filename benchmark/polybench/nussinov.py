# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RNA folding by dynamic programming over a triangular table."""

import numpy as np

from allo.lang import f32, index, kernel

from ..spec import Benchmark

N = 40


def build():
    @kernel
    def nussinov(seq: f32[N], table: f32[N, N]):
        for i_inv in range(N):
            i: index = N - 1 - i_inv
            for j in range(i + 1, N):
                if j - 1 >= 0:
                    if table[i, j] < table[i, j - 1]:
                        table[i, j] = table[i, j - 1]

                if i + 1 < N:
                    if table[i, j] < table[i + 1, j]:
                        table[i, j] = table[i + 1, j]

                if j - 1 >= 0 and i + 1 < N:
                    if i < j - 1:
                        w: f32 = seq[i] + seq[j]
                        match: f32 = 0.0
                        if w == 3.0:
                            match = 1.0
                        s2: f32 = table[i + 1, j - 1] + match
                        if table[i, j] < s2:
                            table[i, j] = s2
                    else:
                        if table[i, j] < table[i + 1, j - 1]:
                            table[i, j] = table[i + 1, j - 1]

                for k in range(i + 1, j):
                    s3: f32 = table[i, k] + table[k + 1, j]
                    if table[i, j] < s3:
                        table[i, j] = s3

    return {"top": nussinov}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.unroll(s.loop("k"), factor=2)
    return s


def inputs(rng):
    seq = rng.integers(0, 4, N).astype(np.float32)
    return seq, np.zeros((N, N), np.float32)


def reference(seq, table):
    t = table.copy()
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0 and t[i, j] < t[i, j - 1]:
                t[i, j] = t[i, j - 1]
            if i + 1 < N and t[i, j] < t[i + 1, j]:
                t[i, j] = t[i + 1, j]
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    m = np.float32(1.0) if seq[i] + seq[j] == 3.0 else np.float32(0.0)
                    s2 = t[i + 1, j - 1] + m
                    if t[i, j] < s2:
                        t[i, j] = s2
                elif t[i, j] < t[i + 1, j - 1]:
                    t[i, j] = t[i + 1, j - 1]
            for k in range(i + 1, j):
                s3 = t[i, k] + t[k + 1, j]
                if t[i, j] < s3:
                    t[i, j] = s3
    return (t,)


BENCHMARK = Benchmark(
    suite="polybench",
    name="nussinov",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(1,),
    tolerance=(2e-3, 2e-3),
)
