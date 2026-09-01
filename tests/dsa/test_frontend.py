# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 frontend: the CornellTPU catalog verifies and lowers."""

from examples.accelerator.cornell_tpu.isa import tpu


def test_catalog_verifies():
    assert tpu.catalog().operation.verify()


def test_catalog_structure():
    text = str(tpu.catalog())
    assert (
        "allo.buffer @dram extents(65536) : !allo.scalar<f32>" in text
    )  # off-chip I/O
    assert "allo.buffer @bram extents(8192) : !allo.scalar<f32>" in text  # on-chip
    assert "!allo.vector<8xf32>" in text
    for name in (
        "dma_load",
        "dma_store",
        "vload",
        "vstore",
        "vadd",
        "vsub",
        "vmul",
        "vrelu",
        "matmul",
    ):
        assert f"allo.define @{name}" in text
    assert "allo.patterns.strided" in text
    assert "allo.patterns.expand_shape" in text
    assert "tosa.add" in text
    assert "tosa.matmul" in text  # matmul lowers via tosa (batched 3-D) too
    assert "tosa.clamp" in text  # relu


def test_visible_shape_inference():
    """Compute block args reflect the access patterns' visible shapes."""
    text = str(tpu.catalog())
    assert "tensor<8xf32>" in text  # vector<8> slot, counts=1 -> rank-reduced
    assert "tensor<1x4x4xf32>" in text  # strided(16) -> expand to batched 1x4x4


def test_lowers_via_oracle():
    """Each define inlines through -lower-instructions (the oracle path)."""
    from allo.backend.base import run_pipeline  # registers allo passes

    module = tpu.catalog()
    run_pipeline(module, "builtin.module(lower-instructions)")
    text = str(module)
    assert "allo.define" not in text
    assert "memref.global" in text
