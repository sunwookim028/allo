# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viterbi decoding: a forward min-cost trellis then a backward traceback."""

import numpy as np

from allo.lang import f32, i32, kernel

from ..spec import Benchmark

N_OBS, N_STATES, N_TOKENS = 16, 8, 8


def build():
    @kernel
    def viterbi(
        obs: i32[N_OBS],
        init: f32[N_STATES],
        transition: f32[N_STATES, N_STATES],
        emission: f32[N_STATES, N_TOKENS],
        path: i32[N_OBS],
    ):
        llike: f32[N_OBS, N_STATES]

        for s0 in range(N_STATES):
            llike[0, s0] = init[s0] + emission[s0, obs[0]]

        for t in range(1, N_OBS):
            for curr in range(N_STATES):
                min_p: f32 = (
                    llike[t - 1, 0] + transition[0, curr] + emission[curr, obs[t]]
                )
                for prev in range(1, N_STATES):
                    p: f32 = (
                        llike[t - 1, prev]
                        + transition[prev, curr]
                        + emission[curr, obs[t]]
                    )
                    if p < min_p:
                        min_p = p
                llike[t, curr] = min_p

        min_s: i32 = 0
        best: f32 = llike[N_OBS - 1, 0]
        for s1 in range(1, N_STATES):
            p1: f32 = llike[N_OBS - 1, s1]
            if p1 < best:
                best = p1
                min_s = s1

        path[N_OBS - 1] = min_s

        for t2 in range(N_OBS - 1):
            actual_t: i32 = N_OBS - 2 - t2
            min_s = 0
            best = llike[actual_t, 0] + transition[0, path[actual_t + 1]]
            for s2 in range(1, N_STATES):
                p2: f32 = llike[actual_t, s2] + transition[s2, path[actual_t + 1]]
                if p2 < best:
                    best = p2
                    min_s = s2
            path[actual_t] = min_s

    return {"top": viterbi}


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    s = parts["top"].schedule()
    s.pipeline(s.loop("curr"), ii=1)
    return s


def inputs(rng):
    obs = rng.integers(0, N_TOKENS, N_OBS).astype(np.int32)
    init = rng.uniform(0.01, 0.25, N_STATES).astype(np.float32)
    transition = rng.uniform(0.01, 0.25, (N_STATES, N_STATES)).astype(np.float32)
    emission = rng.uniform(0.01, 0.25, (N_STATES, N_TOKENS)).astype(np.float32)
    return obs, init, transition, emission, np.zeros(N_OBS, np.int32)


def reference(obs, init, transition, emission, path):
    llike = np.zeros((N_OBS, N_STATES), np.float32)
    for s in range(N_STATES):
        llike[0, s] = init[s] + emission[s, obs[0]]
    for t in range(1, N_OBS):
        for curr in range(N_STATES):
            best = llike[t - 1, 0] + transition[0, curr] + emission[curr, obs[t]]
            for prev in range(1, N_STATES):
                p = llike[t - 1, prev] + transition[prev, curr] + emission[curr, obs[t]]
                if p < best:
                    best = p
            llike[t, curr] = best
    out = np.zeros(N_OBS, np.int32)
    min_s, best = 0, llike[N_OBS - 1, 0]
    for s in range(1, N_STATES):
        if llike[N_OBS - 1, s] < best:
            best, min_s = llike[N_OBS - 1, s], s
    out[N_OBS - 1] = min_s
    for t in range(N_OBS - 1):
        at = N_OBS - 2 - t
        min_s = 0
        best = llike[at, 0] + transition[0, out[at + 1]]
        for s in range(1, N_STATES):
            p = llike[at, s] + transition[s, out[at + 1]]
            if p < best:
                best, min_s = p, s
        out[at] = min_s
    return (out,)


BENCHMARK = Benchmark(
    suite="machsuite",
    name="viterbi",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(4,),
)
