# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""mininpu ISA 0.5.0: the modeled instruction set, hand-written and compiled.

Guards the two things the model has to get right: the splatting reductions
(``vredsum`` / ``vredmax``, whose destination is a full 16-lane register) and the
MXU modeled as state buffers -- a source matmul must still compile into the
three-instruction ``vmatload`` / ``vmatpush`` / ``vmatpop`` sequence, with the two
MXU operands supplied by ordinary routing.
"""

from collections import Counter

import numpy as np
import pytest

# allo is imported here (module load) before torch_mlir (lazily, inside the test
# below), so allo's LLVM binds the JIT -- co-loading torch_mlir LLVM first segfaults.
from examples.accelerator.mininpu import oracle
from examples.accelerator.mininpu.isa import (
    VEC_LANES,
    dram,
    npu,
    vld,
    vmemld,
    vmemst,
    vredmax,
    vredsum,
    vst,
)


def test_catalog_verifies():
    text = str(npu.catalog())
    assert "allo.buffer @vr extents(32) : !allo.vector<16xf32>" in text
    # the MXU state: one stationary 16x16 tile + the result queue
    assert "allo.buffer @mxu_w extents(256) : !allo.scalar<f32>" in text
    assert "allo.buffer @mxu_q extents(16) : !allo.vector<16xf32>" in text
    for name in ("vmemld", "vld", "vadd", "vredsum", "vmatload", "vmatpush", "vmatpop"):
        assert f"allo.define @{name}" in text
    assert npu.catalog().operation.verify()


@pytest.mark.parametrize("example", oracle.EXAMPLES, ids=lambda f: f.__name__)
def test_oracle_examples(example):
    example()  # each asserts against NumPy internally


def test_reductions_splat_all_lanes():
    """A reduction writes a whole vector register: every lane holds the result."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(VEC_LANES).astype(np.float32)

    @npu.oracle(init={dram: x})
    def kernel():
        vmemld(d=0, s=0, n=VEC_LANES)
        vld(d=0, s=0)
        vredsum(d=1, a=0)
        vredmax(d=2, a=0)
        vst(s=1, d=16)
        vst(s=2, d=32)
        vmemst(d=64, s=16, n=2 * VEC_LANES)
        npu.inspect(dram[64 : 64 + 2 * VEC_LANES], label="reduced")

    got = np.asarray(kernel()["reduced"], np.float32)
    np.testing.assert_allclose(got[:VEC_LANES], np.full(VEC_LANES, x.sum()), rtol=1e-5)
    np.testing.assert_allclose(got[VEC_LANES:], np.full(VEC_LANES, x.max()), rtol=1e-5)


def test_compiled_programs():
    """Every compiled example: each asserts its own numerics, and the last one
    asserts that the three out-of-model sources are still rejected."""
    pytest.importorskip("torch_mlir.fx")
    from examples.accelerator.mininpu import program

    for example in program.EXAMPLES:
        example()


def test_compile_matmul_uses_mxu_sequence():
    """A source matmul compiles into the three-step MXU sequence: only ``vmatpush``
    is selected by the matcher, and its stationary-tile and result-queue operands
    are then routed in/out by ``vmatload`` / ``vmatpop``."""
    pytest.importorskip("torch_mlir.fx")
    from examples.accelerator.mininpu import program

    import torch

    class M(torch.nn.Module):
        def forward(self, x, w):
            return x @ w.T

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, VEC_LANES)).astype(np.float32)
    w = rng.standard_normal((VEC_LANES, VEC_LANES)).astype(np.float32)
    prog = npu.compile_program(program._tosa(M(), x, w))

    counts = Counter(e.name for e in prog.emits)
    assert counts == Counter(
        vmemld=2, vld=1, vmatload=1, vmatpush=1, vmatpop=1, vst=1, vmemst=1
    ), counts
    got = np.asarray(prog(x, w), np.float32).reshape(1, VEC_LANES)
    np.testing.assert_allclose(got, x @ w.T, rtol=1e-4, atol=1e-4)
