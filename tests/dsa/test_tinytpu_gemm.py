# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU's declarative whole-GEMM lowering uses only physical instructions."""

import numpy as np

from examples.accelerator.tinytpu.isa import tpu
from examples.accelerator.tinytpu.verify import _gemm_source


def test_tiled_gemm_uses_vpu_for_k_accumulation():
    """A 2-tile K reduction expands to 4x4 MXU ops plus two VPU additions/tile."""
    M = K = N = 8
    prog = tpu.compile_program(_gemm_source(M, K, N))
    names = [emit.name for emit in prog.emits]
    assert "gemm" not in names
    assert set(names) <= {
        "dma_load",
        "dma_store",
        "matmul",
        "vload",
        "vadd",
        "vstore",
    }
    assert names.count("matmul") == 8
    assert names.count("vadd") == 8  # 4 output tiles * two 8-lane halves

    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, M, K)).astype(np.float32)
    b = rng.standard_normal((1, K, N)).astype(np.float32)
    np.testing.assert_allclose(prog(a, b), a @ b, rtol=1e-4, atol=1e-4)
