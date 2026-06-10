# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiling external TOSA programs onto QKV via ``qkv.compile_program``.

The source program is an external TOSA-dialect module string -- here built with
torch_mlir's TOSA backend -- which the search backend covers with QKV instructions,
allocates onto d0/d1/d2, and inserts data movement for. The result is a
``CompiledProgram``; ``print(prog)`` shows the chosen instruction sequence (with the
allocated offsets), and ``prog(*inputs)`` runs it on the bf16 functional simulator.

These examples lean on the parts of selection QKV stresses: a ``tosa.transpose``
feeding a matmul is matched as a *standalone* ``load_cm`` (then a plain ``gemm``);
a chained matmul forces a ``mov`` to stage a GEMM result (in d2) back as a GEMM
operand (in d1); and ``softmax`` -- whose lowered form shares one ``exp`` between the
reduction and the divide -- is folded into the single ``softmax`` instruction. The
softmax is written in the naive ``exp / sum`` form the instruction implements (not
torch's max-subtracted form). Everything is 64x64 bf16, so the golden is diffed with
a bf16-scale tolerance.

Run all examples:  ``python -m example.accelerator.qkv.program``
"""

import ml_dtypes
import numpy as np

from .isa import qkv


def _bf16(a):
    return np.asarray(a, ml_dtypes.bfloat16)


def _f32(a):
    return np.asarray(a, np.float32)


def _randn(rng, *shape, scale=0.25) -> np.ndarray:
    """A bf16-rounded normal tile (held as f32), so the sim ingests it exactly."""
    return _f32(_bf16(rng.standard_normal(shape) * scale))


def _tosa(model, *inputs) -> str:
    """Export a torch model to a TOSA module string (the source-program contract).
    torch is imported lazily, after allo, to avoid a dual-LLVM JIT clash."""
    import torch
    import torch_mlir.fx as fx

    tensors = [torch.from_numpy(np.asarray(x, np.float32)) for x in inputs]
    return str(
        fx.export_and_import(model.eval(), *tensors, output_type=fx.OutputType.TOSA)
    )


def _softmax(x):  # naive row softmax, matching the QKV instruction
    e = np.exp(_f32(x))
    return e / e.sum(axis=1, keepdims=True)


def _run(prog, inputs, want, atol=1e-1, rtol=5e-2) -> None:
    """print() the compiled sequence, run it, and diff against NumPy."""
    print(prog)
    out = _f32(prog(*inputs)).reshape(np.shape(want))
    err = float(np.max(np.abs(out - _f32(want))))
    np.testing.assert_allclose(out, _f32(want), rtol=rtol, atol=atol)
    print(f"  -> {len(prog.emits)} instructions; matches NumPy (max|Δ|={err:.4f})\n")


# ==========================================================================#
# 1. Full attention  (the headline: transpose-load + 2 gemms + softmax + mov)
# ==========================================================================#
def attention() -> None:
    """O = softmax(Q @ Kᵀ) @ V. Selection lowers this to ACT's sequence: ``load_cm``
    (Kᵀ), ``load_rm`` (Q), ``gemm``, ``softmax``, ``mov``, ``load_rm`` (V), ``gemm``,
    ``mov``, ``store_rm`` -- the transpose, the softmax fold, and the GEMM->GEMM mov
    all chosen automatically."""
    import torch

    class M(torch.nn.Module):
        def forward(self, q, k, v):
            e = torch.exp(q @ k.transpose(0, 1))
            return (e / e.sum(dim=1, keepdim=True)) @ v

    rng = np.random.default_rng(10)
    q, k, v = _randn(rng, 64, 64), _randn(rng, 64, 64), _randn(rng, 64, 64)
    want = _softmax(_f32(q) @ _f32(k).T) @ _f32(v)
    _run(qkv.compile_program(_tosa(M(), q, k, v)), [q, k, v], want)


# ==========================================================================#
# 2. Scores + softmax  (softmax fold, no value matmul)
# ==========================================================================#
def scores_softmax() -> None:
    """P = softmax(Q @ Kᵀ). The score half of attention: ``load_cm`` + ``gemm`` +
    the folded ``softmax``, then store -- exercises the matcher's internal-fan-out
    fold (the shared ``exp``) without the trailing P @ V."""
    import torch

    class M(torch.nn.Module):
        def forward(self, q, k):
            e = torch.exp(q @ k.transpose(0, 1))
            return e / e.sum(dim=1, keepdim=True)

    rng = np.random.default_rng(11)
    q, k = _randn(rng, 64, 64), _randn(rng, 64, 64)
    _run(qkv.compile_program(_tosa(M(), q, k)), [q, k], _softmax(_f32(q) @ _f32(k).T))


# ==========================================================================#
# 3. Chained matmul  (mov auto-inserted between gemms)
# ==========================================================================#
def chained_matmul() -> None:
    """Y = (A @ B) @ C. The inner product lands in d2; since the outer GEMM reads its
    operands from d1, the planner inserts a ``mov`` (d2 -> d1) between the two
    ``gemm`` instructions -- the routing QKV needs for matmul chains."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a, b, c):
            return (a @ b) @ c

    rng = np.random.default_rng(12)
    a, b, c = _randn(rng, 64, 64), _randn(rng, 64, 64), _randn(rng, 64, 64)
    _run(
        qkv.compile_program(_tosa(M(), a, b, c)),
        [a, b, c],
        (_f32(a) @ _f32(b)) @ _f32(c),
    )


# ==========================================================================#
# 4. Gram matrix  (one input matched both row- and column-major)
# ==========================================================================#
def gram_matrix() -> None:
    """G = A @ Aᵀ. The single input A is matched as both GEMM operands -- once
    directly (``load_rm``) and once through the transpose (``load_cm``) -- so it is
    loaded on-chip twice, two ways, from the one d0 tile."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a):
            return a @ a.transpose(0, 1)

    rng = np.random.default_rng(13)
    a = _randn(rng, 64, 64)
    _run(qkv.compile_program(_tosa(M(), a)), [a], _f32(a) @ _f32(a).T)


# ==========================================================================#
# 5. Transposed matmul  (gemm + load_cm, no softmax)
# ==========================================================================#
def transposed_matmul() -> None:
    """Z = A @ Bᵀ. A ``tosa.transpose`` feeding the matmul is matched as ``load_cm``
    (the transpose is *not* fused into the GEMM here, unlike CornellTPU), leaving a
    plain ``gemm``."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a, b):
            return a @ b.transpose(0, 1)

    rng = np.random.default_rng(14)
    a, b = _randn(rng, 64, 64), _randn(rng, 64, 64)
    _run(qkv.compile_program(_tosa(M(), a, b)), [a, b], _f32(a) @ _f32(b).T)


# ==========================================================================#
# 6. Plain matmul  (simplest: one gemm, both operands row-major)
# ==========================================================================#
def plain_matmul() -> None:
    """Z = A @ B. No transpose: both operands load row-major (``load_rm``) and a
    single ``gemm`` produces the result -- the minimal QKV program."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    rng = np.random.default_rng(15)
    a, b = _randn(rng, 64, 64), _randn(rng, 64, 64)
    _run(qkv.compile_program(_tosa(M(), a, b)), [a, b], _f32(a) @ _f32(b))


EXAMPLES = [
    attention,
    scores_softmax,
    chained_matmul,
    gram_matrix,
    transposed_matmul,
    plain_matmul,
]


def main() -> None:
    for fn in EXAMPLES:
        print(f"=== {fn.__name__} ===")
        print(f"    {fn.__doc__.strip().splitlines()[0]}")
        fn()
    print(f"All {len(EXAMPLES)} programs compiled, ran, and matched NumPy.")


if __name__ == "__main__":
    main()
