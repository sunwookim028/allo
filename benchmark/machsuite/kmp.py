# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Knuth-Morris-Pratt substring search: two data-dependent while loops."""

import numpy as np

from allo.lang import index, kernel, u8

from ..spec import Benchmark

P, S = 4, 128


def build():
    @kernel
    def kmp(pattern: u8[P], input_str: u8[S], kmp_next: u8[P], matches: u8[1]):
        k: index = 0
        x: index = 1
        for i in range(P - 1):
            while k > 0 and pattern[k] != pattern[x]:
                k = kmp_next[k - 1]
            if pattern[k] == pattern[x]:
                k += 1
            kmp_next[x] = k
            x += 1

        q: index = 0
        for i2 in range(S):
            while q > 0 and pattern[q] != input_str[i2]:
                q = kmp_next[q - 1]
            if pattern[q] == input_str[i2]:
                q += 1
            if q >= P:
                matches[0] += 1
                q = kmp_next[q - 1]

    return {"top": kmp}


def _none(parts):
    return parts["top"].schedule()


def inputs(rng):
    pattern = rng.integers(0, 3, P).astype(np.uint8)
    input_str = rng.integers(0, 3, S).astype(np.uint8)
    return pattern, input_str, np.zeros(P, np.uint8), np.zeros(1, np.uint8)


def reference(pattern, input_str, kmp_next, matches):
    nxt = np.zeros(P, np.uint8)
    k, x = 0, 1
    for _ in range(P - 1):
        while k > 0 and pattern[k] != pattern[x]:
            k = int(nxt[k - 1])
        if pattern[k] == pattern[x]:
            k += 1
        nxt[x] = k
        x += 1
    count = 0
    q = 0
    for i in range(S):
        while q > 0 and pattern[q] != input_str[i]:
            q = int(nxt[q - 1])
        if pattern[q] == input_str[i]:
            q += 1
        if q >= P:
            count += 1
            q = int(nxt[q - 1])
    return nxt, np.array([count], np.uint8)


BENCHMARK = Benchmark(
    suite="machsuite",
    name="kmp",
    build=build,
    schedules={"none": _none},
    inputs=inputs,
    reference=reference,
    outputs=(2, 3),
)
