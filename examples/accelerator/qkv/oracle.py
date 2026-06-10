# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-written QKV assembly, functionally simulated via ``@qkv.oracle``.

Each example traces a sequence of bare instruction calls (``load_rm``, ``load_cm``,
``gemm``, ``softmax``, ``mov``, ``store_rm`` / ``store_cm``) into an ``@qkv.oracle``
body, JIT-runs the bf16 functional simulator, and checks the result against a NumPy
reference. They run from complex (full attention) down to simple (a transpose
round-trip), and together exercise all seven instructions.

``qkv.inspect(buffer[slice], label=...)`` snapshots buffer state *at that point* in
the stream -- not at the end -- so in a long kernel it acts like a probe, letting us
read out and diff every intermediate (the attention scores, the softmax probs, ...).

Everything is bf16 over 64x64 tiles, so the references are diffed with a bf16-scale
tolerance; inputs are pre-rounded to bf16 so only the on-chip arithmetic error shows.

Run all examples:  ``python -m example.accelerator.qkv.oracle``
"""

import ml_dtypes
import numpy as np

from .isa import (
    N,
    d0,
    d2,
    gemm,
    load_cm,
    load_rm,
    mov,
    qkv,
    softmax,
    store_cm,
    store_rm,
)


def _bf16(a):
    return np.asarray(a, ml_dtypes.bfloat16)


def _f32(a):
    return np.asarray(a, np.float32)


def _randn(rng, *shape, scale=0.25) -> np.ndarray:
    """A bf16-rounded normal tile (held as f32) so the sim ingests it exactly and the
    only discrepancy from NumPy is the on-chip bf16 arithmetic."""
    return _f32(_bf16(rng.standard_normal(shape) * scale))


def _d0_init(regions) -> np.ndarray:
    """Flat bf16 d0 pool with each ``(offset, tile)`` written at its element offset."""
    size = max(off + np.asarray(a).size for off, a in regions)
    buf = np.zeros(size, ml_dtypes.bfloat16)
    for off, a in regions:
        buf[off : off + np.asarray(a).size] = _bf16(a).reshape(-1)
    return buf


def _softmax(x):  # naive row softmax, matching the QKV instruction
    e = np.exp(_f32(x))
    return e / e.sum(axis=1, keepdims=True)


def _check(label, got, want, atol=1e-1, rtol=5e-2) -> None:
    """Diff an inspected snapshot against NumPy and print a compact summary."""
    got = _f32(got).reshape(np.shape(want))
    want = _f32(want)
    err = float(np.max(np.abs(got - want)))
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
    print(f"    [ok] {label}  shape={got.shape}  max|Δ|={err:.4f}")


# ==========================================================================#
# 1. Full attention  (most complex: every stage, probed)
# ==========================================================================#
def attention() -> None:
    """Scaled-dot-product attention O = softmax(Q @ Kᵀ) @ V over 64x64 tiles -- the
    whole QKV pipeline. K is brought on-chip transposed by ``load_cm``; the scores
    and the softmax probabilities are snapshotted mid-stream before the value matmul.

    d0:  Q@0, K@4096, V@8192      d1:  Kᵀ@[0:64], Q/P/V@[64:128]      d2:  S/P/O@[0:64]
    """
    rng = np.random.default_rng(0)
    Q, K, V = _randn(rng, N, N), _randn(rng, N, N), _randn(rng, N, N)
    init = _d0_init([(0, Q), (4096, K), (8192, V)])

    @qkv.oracle(init={d0: init})
    def prog():
        load_rm(addr_in=0, addr_out=64, n=N)  # Q  -> d1[64:128]
        load_cm(addr_in=4096, addr_out=0)  # Kᵀ -> d1[0:64]
        gemm(addr_1=64, addr_2=0, addr_out=0)  # S = Q @ Kᵀ -> d2
        qkv.inspect(d2[0:N], label="scores")
        softmax(addr=0, n=N)  # P = softmax(S) -> d2
        qkv.inspect(d2[0:N], label="probs")
        mov(addr_in=0, addr_out=64, n=N)  # P  -> d1[64:128]
        load_rm(addr_in=8192, addr_out=0, n=N)  # V  -> d1[0:64]
        gemm(addr_1=64, addr_2=0, addr_out=0)  # O = P @ V -> d2
        qkv.inspect(d2[0:N], label="out")

    res = prog()
    scores = _f32(Q) @ _f32(K).T
    probs = _softmax(scores)
    _check("scores = Q @ Kᵀ", res["scores"], scores)
    _check("probs = softmax(scores)", res["probs"], probs)
    _check("out = probs @ V", res["out"], probs @ _f32(V))


# ==========================================================================#
# 2. Chained GEMM  (matmul -> mov -> matmul)
# ==========================================================================#
def chained_gemm() -> None:
    """Y = (A @ B) @ C over 64x64 tiles. The first product lands in d2, so ``mov``
    stages it back into d1 to be the next GEMM's operand -- the move that lets GEMM
    results feed GEMMs. Both the intermediate and the final matrix are probed."""
    rng = np.random.default_rng(1)
    A, B, C = _randn(rng, N, N), _randn(rng, N, N), _randn(rng, N, N)
    init = _d0_init([(0, A), (4096, B), (8192, C)])

    @qkv.oracle(init={d0: init})
    def prog():
        load_rm(addr_in=0, addr_out=0, n=N)  # A  -> d1[0:64]
        load_rm(addr_in=4096, addr_out=64, n=N)  # B  -> d1[64:128]
        gemm(addr_1=0, addr_2=64, addr_out=0)  # AB = A @ B -> d2
        qkv.inspect(d2[0:N], label="AB")
        mov(addr_in=0, addr_out=0, n=N)  # AB -> d1[0:64]
        load_rm(addr_in=8192, addr_out=64, n=N)  # C  -> d1[64:128]
        gemm(addr_1=0, addr_2=64, addr_out=0)  # Y = AB @ C -> d2
        qkv.inspect(d2[0:N], label="Y")

    res = prog()
    AB = _f32(A) @ _f32(B)
    _check("AB = A @ B", res["AB"], AB)
    _check("Y = (A @ B) @ C", res["Y"], AB @ _f32(C))


# ==========================================================================#
# 3. Gram matrix  (one input, loaded both row- and column-major)
# ==========================================================================#
def gram_matrix() -> None:
    """G = A @ Aᵀ over a 64x64 tile. The same d0 tile is read once row-major
    (``load_rm``) and once column-major (``load_cm``, transposing), so a single
    input feeds both GEMM operands."""
    rng = np.random.default_rng(2)
    A = _randn(rng, N, N)
    init = _d0_init([(0, A)])

    @qkv.oracle(init={d0: init})
    def prog():
        load_rm(addr_in=0, addr_out=0, n=N)  # A  -> d1[0:64]
        load_cm(addr_in=0, addr_out=64)  # Aᵀ -> d1[64:128]
        gemm(addr_1=0, addr_2=64, addr_out=0)  # G = A @ Aᵀ -> d2
        qkv.inspect(d2[0:N], label="G")

    _check("G = A @ Aᵀ", prog()["G"], _f32(A) @ _f32(A).T)


# ==========================================================================#
# 4. Row softmax of a product  (logits -> probabilities)
# ==========================================================================#
def softmax_rows() -> None:
    """P = softmax(A @ B) over 64x64 tiles: a GEMM produces the logits in d2, then
    ``softmax`` normalizes each row in place. Probes the logits and the probs."""
    rng = np.random.default_rng(3)
    A, B = _randn(rng, N, N), _randn(rng, N, N)
    init = _d0_init([(0, A), (4096, B)])

    @qkv.oracle(init={d0: init})
    def prog():
        load_rm(addr_in=0, addr_out=0, n=N)  # A -> d1[0:64]
        load_rm(addr_in=4096, addr_out=64, n=N)  # B -> d1[64:128]
        gemm(addr_1=0, addr_2=64, addr_out=0)  # L = A @ B -> d2
        qkv.inspect(d2[0:N], label="logits")
        softmax(addr=0, n=N)  # P = softmax(L) -> d2
        qkv.inspect(d2[0:N], label="probs")

    res = prog()
    logits = _f32(A) @ _f32(B)
    _check("logits = A @ B", res["logits"], logits)
    _check("probs = softmax(logits)", res["probs"], _softmax(logits))


# ==========================================================================#
# 5. Attention scores  (transpose-load + GEMM)
# ==========================================================================#
def scores() -> None:
    """S = Q @ Kᵀ over 64x64 tiles: ``load_cm`` brings K on-chip transposed, then a
    single ``gemm`` -- the score half of attention on its own."""
    rng = np.random.default_rng(4)
    Q, K = _randn(rng, N, N), _randn(rng, N, N)
    init = _d0_init([(0, Q), (4096, K)])

    @qkv.oracle(init={d0: init})
    def prog():
        load_rm(addr_in=0, addr_out=64, n=N)  # Q  -> d1[64:128]
        load_cm(addr_in=4096, addr_out=0)  # Kᵀ -> d1[0:64]
        gemm(addr_1=64, addr_2=0, addr_out=0)  # S = Q @ Kᵀ -> d2
        qkv.inspect(d2[0:N], label="scores")

    _check("scores = Q @ Kᵀ", prog()["scores"], _f32(Q) @ _f32(K).T)


# ==========================================================================#
# 6. Transpose round-trip  (the two transposing data movers, simplest)
# ==========================================================================#
def transpose_roundtrip() -> None:
    """Compute Mᵀ two ways and check they agree: ``store_cm`` (load row-major, store
    transposed) and ``load_cm`` (load transposed, store row-major). Pure data
    movement -- the clearest look at the column-major / transposing movers."""
    rng = np.random.default_rng(5)
    M = _randn(rng, N, N, scale=1.0)
    init = _d0_init([(0, M)])

    @qkv.oracle(init={d0: init})
    def prog():
        load_rm(addr_in=0, addr_out=0, n=N)  # M  -> d1[0:64]
        store_cm(addr_in=0, addr_out=4096)  # Mᵀ -> d0[4096:] (column-major store)
        qkv.inspect(d0[4096:8192], label="via_store_cm")
        load_cm(addr_in=0, addr_out=64)  # Mᵀ -> d1[64:128] (column-major load)
        store_rm(addr_in=64, addr_out=8192, n=N)  # -> d0[8192:] (row-major store)
        qkv.inspect(d0[8192:12288], label="via_load_cm")

    res = prog()
    _check("Mᵀ via store_cm", res["via_store_cm"], _f32(M).T)
    _check("Mᵀ via load_cm", res["via_load_cm"], _f32(M).T)


EXAMPLES = [
    attention,
    chained_gemm,
    gram_matrix,
    softmax_rows,
    scores,
    transpose_roundtrip,
]


def main() -> None:
    for fn in EXAMPLES:
        print(f"\n=== {fn.__name__} ===")
        print(f"    {fn.__doc__.strip().splitlines()[0]}")
        fn()
    print(f"\nAll {len(EXAMPLES)} examples passed NumPy verification.")


if __name__ == "__main__":
    main()
