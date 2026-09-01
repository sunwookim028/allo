# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional simulation of hand-written CornellTPU assembly via @tpu.oracle."""

import numpy as np

from examples.accelerator.cornell_tpu.isa import (
    bram,
    matmul,
    tpu,
    vadd,
    vload,
    vreg,
    vstore,
)


def test_vadd_inspect_vreg():
    """vload two slots, vadd them, inspect the destination register."""
    a = np.arange(8, dtype=np.float32)
    b = np.arange(8, dtype=np.float32) * 10.0

    @tpu.oracle(init={bram: np.concatenate([a, b])})
    def prog():
        vload(s=0, d=0)  # bram[0:8]  -> vreg[0]
        vload(s=8, d=1)  # bram[8:16] -> vreg[1]
        vadd(a=0, b=1, d=2)  # vreg[2] = vreg[0] + vreg[1]
        tpu.inspect(vreg, label="out")

    out = prog()["out"]  # whole vreg file, shape (8, 8)
    np.testing.assert_allclose(out[2], a + b)


def test_matmul_roundtrip():
    """Z = X @ W^T on the 4x4 systolic matmul (the weight is consumed transposed),
    read back from bram."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 4)).astype(np.float32)
    X = rng.standard_normal((4, 4)).astype(np.float32)
    # bram layout: W at 0..15, X at 16..31, Z at 32..47
    init = np.zeros(48, np.float32)
    init[0:16] = W.reshape(-1)
    init[16:32] = X.reshape(-1)

    @tpu.oracle(init={bram: init})
    def prog():
        matmul(x=16, w=0, z=32)
        tpu.inspect(bram[32:48], label="z")

    z = prog()["z"].reshape(4, 4)
    np.testing.assert_allclose(z, X @ W.T, rtol=1e-5, atol=1e-5)


def test_reference_diff():
    """The built-in differential check passes when the program matches."""
    a = np.arange(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)

    @tpu.oracle(
        init={bram: np.concatenate([a, b])},
        reference=lambda: {"r": (a + b)},
    )
    def prog():
        vload(s=0, d=0)
        vload(s=8, d=1)
        vadd(a=0, b=1, d=2)
        vstore(s=2, d=64)  # vreg[2] -> bram[64:72]
        tpu.inspect(bram[64:72], label="r")

    prog()  # raises if the differential check fails
