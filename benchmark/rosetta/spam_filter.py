# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logistic regression trained by stochastic gradient descent, one sample at a time."""

import numpy as np

from allo.lang import i32, i64, kernel

from ..spec import Benchmark

NUM_FEATURES = 128
NUM_TRAINING = 32
NUM_EPOCHS = 2
LUT_SIZE = 512

# Fixed point throughout, as in the accelerator, which carries ap_fixed features
# and reads the sigmoid out of a BRAM. Both choices are load-bearing here: the
# table index falls out of a shift, so no float-to-int conversion is needed, and
# the reference can reproduce the arithmetic exactly rather than within a
# tolerance.
FRAC = 16
ONE = 1 << FRAC
FOUR = 4 << FRAC
LUT_SHIFT = 10  # (8 << FRAC) >> LUT_SHIFT == LUT_SIZE
STEP_SIZE = int(0.6 * ONE)

SIGMOID_LUT = np.array(
    [int(ONE / (1.0 + np.exp(-(-4.0 + 8.0 * i / LUT_SIZE)))) for i in range(LUT_SIZE)],
    np.int32,
)


def build():
    @kernel
    def dot_product(param: i32[NUM_FEATURES], feature: i32[NUM_FEATURES]) -> i32:
        acc: i64 = 0
        for d in range(NUM_FEATURES):
            acc += param[d] * feature[d]
        result: i32 = acc >> FRAC
        return result

    @kernel
    def sigmoid(exponent: i32, lut: i32[LUT_SIZE]) -> i32:
        out: i32 = 0
        if exponent >= FOUR:
            out = ONE
        elif exponent < -FOUR:
            out = 0
        else:
            idx: i32 = (exponent + FOUR) >> LUT_SHIFT
            out = lut[idx]
        return out

    @kernel
    def compute_gradient(
        grad: i32[NUM_FEATURES], feature: i32[NUM_FEATURES], scale: i32
    ):
        for g in range(NUM_FEATURES):
            prod: i64 = scale * feature[g]
            grad[g] = prod >> FRAC

    @kernel
    def update_parameter(param: i32[NUM_FEATURES], grad: i32[NUM_FEATURES], scale: i32):
        for u in range(NUM_FEATURES):
            prod: i64 = scale * grad[u]
            param[u] += prod >> FRAC

    @kernel
    def spam_filter(
        data: i32[NUM_TRAINING, NUM_FEATURES],
        label: i32[NUM_TRAINING],
        lut: i32[LUT_SIZE],
        theta: i32[NUM_FEATURES],
    ):
        feature: i32[NUM_FEATURES]
        gradient: i32[NUM_FEATURES]
        for e in range(NUM_EPOCHS):
            for tid in range(NUM_TRAINING):
                for r in range(NUM_FEATURES):
                    feature[r] = data[tid, r]
                dot: i32 = dot_product(theta, feature)
                prob: i32 = sigmoid(dot, lut)
                compute_gradient(gradient, feature, prob - label[tid])
                update_parameter(theta, gradient, -STEP_SIZE)

    return {
        "top": spam_filter,
        "dot_product": dot_product,
        "sigmoid": sigmoid,
        "compute_gradient": compute_gradient,
        "update_parameter": update_parameter,
    }


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    stages = []
    for name, loop in (
        ("dot_product", "d"),
        ("compute_gradient", "g"),
        ("update_parameter", "u"),
    ):
        s = parts[name].schedule()
        s.unroll(s.loop(loop), factor=4)
        stages.append(s)
    top = parts["top"].schedule()
    top.compose(*stages)
    return top


def _v2(parts):
    stages = []
    for name, loop, buffers in (
        ("dot_product", "d", ("param", "feature")),
        ("compute_gradient", "g", ("grad", "feature")),
        ("update_parameter", "u", ("param", "grad")),
    ):
        s = parts[name].schedule()
        for buf in buffers:
            s.partition(s.buffer(buf), dim=1, kind=s.Cyclic, factor=4)
        s.unroll(s.loop(loop), factor=4)
        stages.append(s)
    top = parts["top"].schedule()
    # A partitioned array has to carry the same banking on both sides of a call,
    # and the copy loop that feeds it has to be unrolled to address one bank.
    for buf in ("theta", "feature", "gradient"):
        top.partition(top.buffer(buf), dim=1, kind=top.Cyclic, factor=4)
    top.unroll(top.loop("r"), factor=4)
    top.compose(*stages)
    return top


def inputs(rng):
    data = (rng.uniform(-0.5, 0.5, (NUM_TRAINING, NUM_FEATURES)) * ONE).astype(np.int32)
    label = (rng.integers(0, 2, NUM_TRAINING) * ONE).astype(np.int32)
    return data, label, SIGMOID_LUT.copy(), np.zeros(NUM_FEATURES, np.int32)


def reference(data, label, lut, theta):
    param = [int(v) for v in theta]
    for _ in range(NUM_EPOCHS):
        for tid in range(NUM_TRAINING):
            feature = [int(v) for v in data[tid]]
            dot = sum(p * f for p, f in zip(param, feature)) >> FRAC
            if dot >= FOUR:
                prob = ONE
            elif dot < -FOUR:
                prob = 0
            else:
                prob = int(lut[(dot + FOUR) >> LUT_SHIFT])
            scale = prob - int(label[tid])
            grad = [(scale * f) >> FRAC for f in feature]
            param = [p + ((-STEP_SIZE * g) >> FRAC) for p, g in zip(param, grad)]
    return (np.array(param, np.int32),)


BENCHMARK = Benchmark(
    suite="rosetta",
    name="spam_filter",
    build=build,
    schedules={"none": _none, "v1": _v1, "v2": _v2},
    inputs=inputs,
    reference=reference,
    outputs=(3,),
)
