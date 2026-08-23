# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""K-nearest-neighbour digit classification over Hamming distances."""

import numpy as np

from allo.lang import i32, kernel, u32

from ..spec import Benchmark

NUM_TRAINING = 180
CLASS_SIZE = 18
NUM_TEST = 8
DIGIT_WIDTH = 8
K_CONST = 3
NUM_CLASSES = 10
MAX_DIST = 256


def build():
    @kernel
    def popcount(x: u32) -> i32:
        v: u32 = x
        v = v - ((v >> 1) & 0x55555555)
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
        v = (v + (v >> 4)) & 0x0F0F0F0F
        v = v + (v >> 8)
        v = v + (v >> 16)
        return v & 0x3F

    @kernel
    def update_knn(
        training_set: u32[NUM_TRAINING, DIGIT_WIDTH],
        test_set: u32[NUM_TEST, DIGIT_WIDTH],
        t: i32,
        n: i32,
        dists: i32[K_CONST],
        labels: i32[K_CONST],
        label: i32,
    ):
        dist: i32 = 0
        for i in range(DIGIT_WIDTH):
            diff: u32 = test_set[t, i] ^ training_set[n, i]
            dist += popcount(diff)

        max_dist: i32 = 0
        max_dist_id: i32 = K_CONST
        for k in range(K_CONST):
            if dists[k] > max_dist:
                max_dist = dists[k]
                max_dist_id = k

        if dist < max_dist:
            dists[max_dist_id] = dist
            labels[max_dist_id] = label

    @kernel
    def knn_vote(labels: i32[K_CONST]) -> i32:
        votes: i32[NUM_CLASSES] = 0
        for i in range(K_CONST):
            votes[labels[i]] += 1
        max_vote: i32 = 0
        max_label: i32 = 0
        for c in range(NUM_CLASSES):
            if votes[c] > max_vote:
                max_vote = votes[c]
                max_label = c
        return max_label

    @kernel
    def digit_recognition(
        training_set: u32[NUM_TRAINING, DIGIT_WIDTH],
        test_set: u32[NUM_TEST, DIGIT_WIDTH],
        results: i32[NUM_TEST],
    ):
        dists: i32[K_CONST] = 0
        labels: i32[K_CONST] = 0
        for t in range(NUM_TEST):
            for k in range(K_CONST):
                dists[k] = MAX_DIST
                labels[k] = 0
            for n in range(NUM_TRAINING):
                update_knn(training_set, test_set, t, n, dists, labels, n // CLASS_SIZE)
            results[t] = knn_vote(labels)

    return {
        "top": digit_recognition,
        "update_knn": update_knn,
        "knn_vote": knn_vote,
        "popcount": popcount,
    }


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    knn = parts["update_knn"].schedule()
    knn.unroll(knn.loop("i"))
    top = parts["top"].schedule()
    top.compose(knn)
    return top


def _v2(parts):
    knn = parts["update_knn"].schedule()
    knn.unroll(knn.loop("i"))
    knn.unroll(knn.loop("k"))
    top = parts["top"].schedule()
    top.compose(knn)
    return top


def inputs(rng):
    training = rng.integers(0, 2**32, (NUM_TRAINING, DIGIT_WIDTH), dtype=np.uint32)
    test = rng.integers(0, 2**32, (NUM_TEST, DIGIT_WIDTH), dtype=np.uint32)
    return training, test, np.zeros(NUM_TEST, np.int32)


def reference(training_set, test_set, results):
    out = np.zeros(NUM_TEST, np.int32)
    for t in range(NUM_TEST):
        dists = [MAX_DIST] * K_CONST
        labels = [0] * K_CONST
        for n in range(NUM_TRAINING):
            dist = int(
                sum(
                    int(x).bit_count()
                    for x in (test_set[t] ^ training_set[n]).astype(np.uint32)
                )
            )
            max_dist, max_id = 0, K_CONST
            for k in range(K_CONST):
                if dists[k] > max_dist:
                    max_dist, max_id = dists[k], k
            if dist < max_dist:
                dists[max_id] = dist
                labels[max_id] = n // CLASS_SIZE
        votes = [0] * NUM_CLASSES
        for label in labels:
            votes[label] += 1
        out[t] = int(np.argmax(votes))
    return (out,)


BENCHMARK = Benchmark(
    suite="rosetta",
    name="digit_recognition",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(2,),
)
