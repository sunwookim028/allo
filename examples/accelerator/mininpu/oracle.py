# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-written mininpu assembly, functionally simulated via ``@npu.oracle``.

Each example traces a sequence of bare instruction calls into an ``@npu.oracle``
body, JIT-runs the functional simulator, and checks the result against a NumPy
reference. They run from complex (a matrix-vector product over the three-step MXU
sequence, staged through the DMA) down to simple (a 16-lane fused multiply-add).

Program I/O lives in ``dram``, so every example starts with a ``vmemld`` and ends
with a ``vmemst`` -- the mininpu memory hierarchy in full. ``npu.inspect(dram[...])``
snapshots buffer state *at that point* in the stream, so intermediate stages can be
diffed independently.

Run all examples:  ``python -m examples.accelerator.mininpu.oracle``
"""

import numpy as np

from .isa import (
    VEC_LANES,
    MXU_TILE,
    dram,
    npu,
    vadd,
    vld,
    vmatload,
    vmatpop,
    vmatpush,
    vmemld,
    vmemst,
    vmov,
    vmul,
    vrecip,
    vredmax,
    vredsum,
    vrsqrt,
    vst,
    vsub,
)


def _dram_init(regions) -> np.ndarray:
    """Build a flat fp32 image with each ``(offset, array)`` written at its offset
    (everything else zero) -- the host-side layout of the dram inputs."""
    size = max(off + np.asarray(a).size for off, a in regions)
    buf = np.zeros(size, np.float32)
    for off, a in regions:
        flat = np.asarray(a, np.float32).reshape(-1)
        buf[off : off + flat.size] = flat
    return buf


def _check(label, got, want, rtol=1e-4, atol=1e-4) -> None:
    """Diff an inspected snapshot against NumPy and print the captured value."""
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=rtol, atol=atol)
    flat = np.array2string(
        got.reshape(-1), precision=2, suppress_small=True, max_line_width=100
    )
    print(f"    [ok] {label}\n         inspected = {flat}")


# ==========================================================================#
# 1. Matrix path: two activation rows through the MXU  (most complex)
# ==========================================================================#
def matmul_rows() -> None:
    """Z = X @ W^T for a 2x16 X against the 16x16 stationary tile -- the full
    three-step MXU sequence. ``vmatload`` fills the array once, then each row is
    pushed (``vmatpush``, one enqueued multiply) and its result popped
    (``vmatpop``) back into a vector register. The weight tile and the activations
    are DMA'd in from dram and the result DMA'd back out."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((VEC_LANES, VEC_LANES)).astype(np.float32)
    X = rng.standard_normal((2, VEC_LANES)).astype(np.float32)

    # dram: W@0 (256 words), X@256 (32 words) | Z@512
    init = _dram_init([(0, W), (MXU_TILE, X)])

    @npu.oracle(init={dram: init})
    def prog():
        vmemld(d=0, s=0, n=MXU_TILE)  # vmem[0:256] = W
        vmatload(s=0)  # stationary tile <- vmem[0:256]
        vmemld(d=MXU_TILE, s=MXU_TILE, n=2 * VEC_LANES)  # vmem[256:288] = X
        vld(d=0, s=MXU_TILE)  # vr0 = X[0]
        vld(d=1, s=MXU_TILE + VEC_LANES)  # vr1 = X[1]
        vmatpush(x=0, q=0)  # enqueue X[0] @ W^T
        vmatpush(x=1, q=1)  # enqueue X[1] @ W^T
        vmatpop(d=2, q=0)  # vr2 = Z[0]
        vmatpop(d=3, q=1)  # vr3 = Z[1]
        vst(s=2, d=288)
        vst(s=3, d=288 + VEC_LANES)
        vmemst(d=512, s=288, n=2 * VEC_LANES)
        npu.inspect(dram[512 : 512 + 2 * VEC_LANES], label="Z")

    _check("Z = X @ W^T", prog()["Z"].reshape(2, VEC_LANES), X @ W.T)


# ==========================================================================#
# 2. L2 normalize: reduction feeding a reciprocal square root
# ==========================================================================#
def l2_normalize() -> None:
    """y = x / ||x||, i.e. ``x * vrsqrt(vredsum(x * x))``. ``vredsum`` splats its
    scalar result across all 16 lanes, which is exactly what makes it usable as the
    right operand of a lane-wise ``vmul`` -- no broadcast instruction needed."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(VEC_LANES).astype(np.float32)
    init = _dram_init([(0, x)])

    @npu.oracle(init={dram: init})
    def prog():
        vmemld(d=0, s=0, n=VEC_LANES)
        vld(d=0, s=0)
        vmul(d=1, a=0, b=0)  # vr1 = x * x
        vredsum(d=1, a=1)  # vr1 = sum(x*x), splatted
        vrsqrt(d=1, a=1)  # vr1 = 1 / ||x||
        vmul(d=2, a=0, b=1)  # vr2 = x / ||x||
        vst(s=2, d=16)
        vmemst(d=64, s=16, n=VEC_LANES)
        npu.inspect(dram[64 : 64 + VEC_LANES], label="y")

    _check("y = x / ||x||", prog()["y"], x / np.linalg.norm(x))


# ==========================================================================#
# 3. Mean-center: the divisor is a constant *in memory*
# ==========================================================================#
def mean_center() -> None:
    """y = x - mean(x). The ISA has no immediate operand and the compute vocabulary
    no constant, so ``1/16`` is built from data: the host stages a vector of ones,
    ``vredsum`` turns it into a splatted 16, and ``vrecip`` inverts it. This is the
    same trick the legacy ISA's ``ADDR_CONST`` used -- constants live in memory."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(VEC_LANES).astype(np.float32)
    ones = np.ones(VEC_LANES, np.float32)
    init = _dram_init([(0, x), (VEC_LANES, ones)])

    @npu.oracle(init={dram: init})
    def prog():
        vmemld(d=0, s=0, n=2 * VEC_LANES)  # vmem[0:16] = x, vmem[16:32] = ones
        vld(d=0, s=0)
        vld(d=1, s=VEC_LANES)
        vredsum(d=2, a=0)  # vr2 = sum(x), splatted
        vredsum(d=3, a=1)  # vr3 = 16, splatted
        vrecip(d=3, a=3)  # vr3 = 1/16
        vmul(d=2, a=2, b=3)  # vr2 = mean(x)
        vst(s=2, d=32)
        vmemst(d=64, s=32, n=VEC_LANES)
        npu.inspect(dram[64 : 64 + VEC_LANES], label="mean")
        vsub(d=4, a=0, b=2)  # vr4 = x - mean(x)
        vst(s=4, d=48)
        vmemst(d=96, s=48, n=VEC_LANES)
        npu.inspect(dram[96 : 96 + VEC_LANES], label="centered")

    res = prog()
    _check("mean(x), splatted", res["mean"], np.full(VEC_LANES, x.mean()))
    _check("y = x - mean(x)", res["centered"], x - x.mean())


# ==========================================================================#
# 4. Max-shift: the numerically-stable softmax prefix
# ==========================================================================#
def max_shift() -> None:
    """y = x - max(x) -- the stabilizing prefix of a softmax, and as far as this
    model goes: the ``vexp`` that follows is base-2 and needs a ``log 2`` constant
    the closed compute vocabulary cannot express, so it is not modeled."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(VEC_LANES).astype(np.float32)
    init = _dram_init([(0, x)])

    @npu.oracle(init={dram: init})
    def prog():
        vmemld(d=0, s=0, n=VEC_LANES)
        vld(d=0, s=0)
        vredmax(d=1, a=0)  # vr1 = max(x), splatted
        vsub(d=2, a=0, b=1)
        vst(s=2, d=16)
        vmemst(d=64, s=16, n=VEC_LANES)
        npu.inspect(dram[64 : 64 + VEC_LANES], label="shifted")

    _check("y = x - max(x)", prog()["shifted"], x - x.max())


# ==========================================================================#
# 5. Fused multiply-add  (simplest)
# ==========================================================================#
def vector_madd() -> None:
    """y = a * b + c over 16 lanes, with a ``vmov`` staging the product into the
    register the accumulate writes."""
    rng = np.random.default_rng(4)
    a, b, c = (rng.standard_normal(VEC_LANES).astype(np.float32) for _ in range(3))
    init = _dram_init([(0, a), (VEC_LANES, b), (2 * VEC_LANES, c)])

    @npu.oracle(init={dram: init})
    def prog():
        vmemld(d=0, s=0, n=3 * VEC_LANES)
        vld(d=0, s=0)
        vld(d=1, s=VEC_LANES)
        vld(d=2, s=2 * VEC_LANES)
        vmul(d=3, a=0, b=1)
        vmov(d=4, s=3)
        vadd(d=5, a=4, b=2)
        vst(s=5, d=48)
        vmemst(d=64, s=48, n=VEC_LANES)
        npu.inspect(dram[64 : 64 + VEC_LANES], label="y")

    _check("y = a * b + c", prog()["y"], a * b + c)


EXAMPLES = [
    matmul_rows,
    l2_normalize,
    mean_center,
    max_shift,
    vector_madd,
]


def main() -> None:
    for fn in EXAMPLES:
        print(f"\n=== {fn.__name__} ===")
        print(f"    {fn.__doc__.strip().splitlines()[0]}")
        fn()
    print(f"\nAll {len(EXAMPLES)} examples passed NumPy verification.")


if __name__ == "__main__":
    main()
