# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-written attention on QKV, validated against a NumPy reference.

Runs the canonical 9-instruction QKV attention kernel (the same sequence ACT's
generated backend produces) through the functional oracle and diffs the result
against ``softmax(Q @ Kᵀ) @ V``. This exercises every instruction (both load
forms, gemm, softmax, mov, store) and the bf16 + broadcast paths end-to-end.

Run from the repo root: ``python -m example.accelerator.qkv.attention``
"""

import ml_dtypes
import numpy as np

from .isa import (
    N,
    d0,
    gemm,
    load_cm,
    load_rm,
    mov,
    qkv,
    softmax,
    store_rm,
)

# d0 layout (bf16 words): Q | K | V | O, each an N*N block.
Q_OFF, K_OFF, V_OFF, O_OFF = 0, N * N, 2 * N * N, 3 * N * N


def _bf16(a):
    return np.asarray(a, ml_dtypes.bfloat16)


def _f32(a):
    return np.asarray(a, np.float32)


def attention_ref(q, k, v):
    """Naive (no max-subtract) row-softmax attention, matching the QKV semantics."""
    s = _f32(q) @ _f32(k).T
    e = np.exp(s)
    p = e / e.sum(axis=1, keepdims=True)
    return p @ _f32(v)


def run_attention(q, k, v):
    pool = np.zeros(qkv.buffers["d0"].size, ml_dtypes.bfloat16)
    pool[Q_OFF : Q_OFF + N * N] = _bf16(q).reshape(-1)
    pool[K_OFF : K_OFF + N * N] = _bf16(k).reshape(-1)
    pool[V_OFF : V_OFF + N * N] = _bf16(v).reshape(-1)

    @qkv.oracle(init={d0: pool})
    def kernel():
        load_rm(addr_in=Q_OFF, addr_out=64, n=N)  # Q  -> d1[64:128]
        load_cm(addr_in=K_OFF, addr_out=0)  # Kᵀ -> d1[0:64]
        gemm(addr_1=64, addr_2=0, addr_out=0)  # S = Q @ Kᵀ -> d2[0:64]
        softmax(addr=0, n=N)  # P = softmax(S) -> d2[0:64]
        mov(addr_in=0, addr_out=64, n=N)  # P  -> d1[64:128]
        load_rm(addr_in=V_OFF, addr_out=0, n=N)  # V  -> d1[0:64]
        gemm(addr_1=64, addr_2=0, addr_out=0)  # O = P @ V -> d2[0:64]
        mov(addr_in=0, addr_out=0, n=N)  # O  -> d1[0:64]
        store_rm(addr_in=0, addr_out=O_OFF, n=N)  # O  -> d0[O_OFF:]
        qkv.inspect(d0[O_OFF : O_OFF + N * N], label="O")

    return _f32(kernel()["O"]).reshape(N, N)


def main():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((N, N)).astype(np.float32) * 0.25
    k = rng.standard_normal((N, N)).astype(np.float32) * 0.25
    v = rng.standard_normal((N, N)).astype(np.float32)

    got = run_attention(q, k, v)
    ref = attention_ref(q, k, v)
    max_diff = float(np.max(np.abs(got - ref)))
    print(f"attention max|Δ| = {max_diff:.4f}  (bf16 sim vs f32 reference)")
    assert max_diff < 0.1, f"attention mismatch: max|Δ|={max_diff}"
    print("QKV attention matches the NumPy reference (within bf16 tolerance).")


if __name__ == "__main__":
    main()
