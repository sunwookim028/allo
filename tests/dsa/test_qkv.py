# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""QKV accelerator: end-to-end attention + the bf16 / broadcast paths it relies on.

Also serves as the regression guard for the two core frontend capabilities QKV
introduced: bf16 oracle round-trips and TOSA-style elementwise broadcast (the
softmax divide).
"""

from collections import Counter

import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")

# allo is imported here (module load) before torch_mlir (lazily, inside the test
# below), so allo's LLVM binds the JIT — co-loading torch_mlir LLVM first segfaults.
from examples.accelerator.qkv.attention import N, attention_ref, run_attention
from examples.accelerator.qkv.isa import d2, qkv, softmax


def _attention_tosa() -> str:
    """torch attention (S=Q@Kᵀ, naive row softmax, O=P@V) exported to TOSA text."""
    import torch

    fx = pytest.importorskip("torch_mlir.fx")

    class Attn(torch.nn.Module):
        def forward(self, q, k, v):
            s = q @ k.transpose(0, 1)
            e = torch.exp(s)
            p = e / e.sum(dim=1, keepdim=True)
            return p @ v

    tensors = [torch.randn(N, N) for _ in range(3)]
    return str(
        fx.export_and_import(Attn().eval(), *tensors, output_type=fx.OutputType.TOSA)
    )


def test_attention_matches_reference():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((N, N)).astype(np.float32) * 0.25
    k = rng.standard_normal((N, N)).astype(np.float32) * 0.25
    v = rng.standard_normal((N, N)).astype(np.float32)
    got = run_attention(q, k, v)
    ref = attention_ref(q, k, v)
    assert np.max(np.abs(got - ref)) < 0.1


def test_softmax_row_broadcast_bf16():
    """The softmax instruction alone: bf16 + reduce_sum([n,1]) + broadcast divide."""
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((N, N)) * 0.5).astype(ml_dtypes.bfloat16)
    init = np.zeros((N, N), ml_dtypes.bfloat16)
    init[:] = x

    @qkv.oracle(init={d2: init})
    def kernel():
        softmax(addr=0, n=N)
        qkv.inspect(d2[0:N], label="P")

    got = np.asarray(kernel()["P"], np.float32)
    e = np.exp(np.asarray(x, np.float32))
    ref = e / e.sum(axis=1, keepdims=True)
    assert np.max(np.abs(got - ref)) < 0.05


def test_compile_attention_from_torch():
    """End-to-end: torch attention -> TOSA -> instruction selection -> run.

    Exercises the two matcher fixes QKV forced: folding softmax (a compute DAG with
    internal fan-out — shared ``exp``) into one instruction, and solving a move's
    shape param from the moved value (``load_rm`` n = rows, not the word count)."""
    prog = qkv.compile_program(_attention_tosa())

    # The selected sequence is ACT's: 2 loads + transpose-load, 2 gemms, softmax,
    # 2 movs, store. softmax==1 proves the fan-out fold; load_cm==1 the transpose.
    counts = Counter(e.name for e in prog.emits)
    assert counts == Counter(
        load_cm=1, load_rm=2, gemm=2, softmax=1, mov=2, store_rm=1
    ), counts

    rng = np.random.default_rng(0)
    q = (rng.standard_normal((N, N)) * 0.25).astype(np.float32)
    k = (rng.standard_normal((N, N)) * 0.25).astype(np.float32)
    v = rng.standard_normal((N, N)).astype(np.float32)
    got = np.asarray(prog(q, k, v), np.float32).reshape(N, N)
    assert np.max(np.abs(got - attention_ref(q, k, v))) < 0.1
