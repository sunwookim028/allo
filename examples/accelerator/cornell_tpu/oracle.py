# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-written CornellTPU assembly, functionally simulated via ``@tpu.oracle``.

Each example below traces a sequence of bare instruction calls (``matmul``,
``vload``, ``vadd``, ...) into an ``@tpu.oracle`` body, JIT-runs the functional
simulator, and checks the result against a NumPy reference. The examples run
from complex (K-tiled matmul with VPU accumulation) down to simple (a single
8-lane vector add).

``tpu.inspect(buffer[slice], label=...)`` snapshots buffer state *at that point*
in the instruction stream -- not at the end -- so in a long sequence it acts like
a probe, letting us read out every intermediate result and diff each stage
independently. The multi-stage examples lean on this heavily.

Run all examples:  ``python -m example.accelerator.cornell_tpu.oracle``
"""

import numpy as np

from .isa import bram, matmul, tpu, vadd, vload, vmul, vrelu, vstore


def _bram_init(regions) -> np.ndarray:
    """Build a flat fp32 vector with each ``(offset, array)`` written at its
    offset (everything else zero) -- the host-side layout of bram inputs."""
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
# 1. K-tiled matmul with VPU accumulation  (most complex)
# ==========================================================================#
def tiled_matmul_accumulate() -> None:
    """A 4x8 @ 8x4 matmul reduced over K by *tiling* it into two 4x4 systolic
    matmuls whose partial products are accumulated on the VPU -- the "software
    tiling with VPU accumulation" from todos/isa.md.

        Z = X @ W^T,  X = [X0 | X1],  W = [W0 | W1]  (split along K)
                      Z = X0 @ W0^T + X1 @ W1^T

    inspect() snapshots each 4x4 partial product mid-stream, then the final sum.
    """
    rng = np.random.default_rng(0)
    X0, X1, W0, W1 = (rng.standard_normal((4, 4)).astype(np.float32) for _ in range(4))

    # bram: X0@0, X1@16, W0@32, W1@48 | partials P0@64, P1@80 | Z@96
    init = _bram_init([(0, X0), (16, X1), (32, W0), (48, W1)])

    @tpu.oracle(init={bram: init})
    def prog():
        matmul(w=32, x=0, z=64)  # P0 = X0 @ W0^T
        tpu.inspect(bram[64:80], label="partial0")
        matmul(w=48, x=16, z=80)  # P1 = X1 @ W1^T
        tpu.inspect(bram[80:96], label="partial1")
        # Z = P0 + P1  (16 elems = two 8-lane halves)
        vload(s=64, d=0)
        vload(s=72, d=1)
        vload(s=80, d=2)
        vload(s=88, d=3)
        vadd(a=0, b=2, d=0)
        vadd(a=1, b=3, d=1)
        vstore(s=0, d=96)
        vstore(s=1, d=104)
        tpu.inspect(bram[96:112], label="Z")

    res = prog()
    _check("partial0 = X0 @ W0^T", res["partial0"].reshape(4, 4), X0 @ W0.T)
    _check("partial1 = X1 @ W1^T", res["partial1"].reshape(4, 4), X1 @ W1.T)
    _check("Z = X0@W0^T + X1@W1^T", res["Z"].reshape(4, 4), X0 @ W0.T + X1 @ W1.T)


# ==========================================================================#
# 2. Two-layer perceptron  (matmul -> activation -> matmul)
# ==========================================================================#
def two_layer_mlp() -> None:
    """A two-layer perceptron over 4x4 tiles: Y = relu(X @ W1^T) @ W2^T -- the
    PyTorch nn.Linear convention the systolic array implements natively. The
    three inspects read out the pre-activation, the hidden activation, and the
    final output across a long matmul/VPU/matmul sequence."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 4)).astype(np.float32)
    W1 = rng.standard_normal((4, 4)).astype(np.float32)
    W2 = rng.standard_normal((4, 4)).astype(np.float32)

    # bram: X@0, W1@16, W2@32 | hidden H@48 | Y@64
    init = _bram_init([(0, X), (16, W1), (32, W2)])

    @tpu.oracle(init={bram: init})
    def prog():
        matmul(w=16, x=0, z=48)  # H_pre = X @ W1^T
        tpu.inspect(bram[48:64], label="pre_act")
        vload(s=48, d=0)  # relu(H_pre) in place over two 8-lane halves
        vload(s=56, d=1)
        vrelu(a=0, d=0)
        vrelu(a=1, d=1)
        vstore(s=0, d=48)
        vstore(s=1, d=56)
        tpu.inspect(bram[48:64], label="hidden")
        matmul(w=32, x=48, z=64)  # Y = H @ W2^T
        tpu.inspect(bram[64:80], label="Y")

    res = prog()
    H = np.maximum(X @ W1.T, 0)
    _check("pre-activation = X @ W1^T", res["pre_act"].reshape(4, 4), X @ W1.T)
    _check("hidden = relu(X @ W1^T)", res["hidden"].reshape(4, 4), H)
    _check("Y = hidden @ W2^T", res["Y"].reshape(4, 4), H @ W2.T)


# ==========================================================================#
# 3. Dense layer with bias + activation
# ==========================================================================#
def dense_layer() -> None:
    """A dense layer Z = relu(X @ W^T + bias), all 4x4. Mixes the systolic matmul
    with VPU elementwise ops and reuses vector registers in place -- the bias sum
    sitting in V0/V1 is relu'd without reloading it from bram."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 4)).astype(np.float32)
    W = rng.standard_normal((4, 4)).astype(np.float32)
    bias = rng.standard_normal((4, 4)).astype(np.float32)

    # bram: X@0, W@16, bias@32 | Z@48
    init = _bram_init([(0, X), (16, W), (32, bias)])

    @tpu.oracle(init={bram: init})
    def prog():
        matmul(w=16, x=0, z=48)  # Z = X @ W^T
        tpu.inspect(bram[48:64], label="matmul")
        vload(s=48, d=0)  # Z halves
        vload(s=56, d=1)
        vload(s=32, d=2)  # bias halves
        vload(s=40, d=3)
        vadd(a=0, b=2, d=0)  # V0, V1 = Z + bias
        vadd(a=1, b=3, d=1)
        vstore(s=0, d=48)
        vstore(s=1, d=56)
        tpu.inspect(bram[48:64], label="biased")
        vrelu(a=0, d=0)  # reuse V0, V1 in place
        vrelu(a=1, d=1)
        vstore(s=0, d=48)
        vstore(s=1, d=56)
        tpu.inspect(bram[48:64], label="out")

    res = prog()
    pre = X @ W.T
    _check("matmul = X @ W^T", res["matmul"].reshape(4, 4), pre)
    _check("biased = X @ W^T + bias", res["biased"].reshape(4, 4), pre + bias)
    _check(
        "out = relu(X @ W^T + bias)",
        res["out"].reshape(4, 4),
        np.maximum(pre + bias, 0),
    )


# ==========================================================================#
# 4. Vector compute chain  (probe each stage with inspect)
# ==========================================================================#
def vector_pipeline() -> None:
    """An 8-lane vector chain held in one register: out = relu((a + b) * c). An
    inspect() after each stage shows how the value evolves down the sequence --
    the clearest illustration of inspect-as-probe."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    c = rng.standard_normal(8).astype(np.float32)

    # bram: a@0, b@8, c@16 | out@24
    init = _bram_init([(0, a), (8, b), (16, c)])

    @tpu.oracle(init={bram: init})
    def prog():
        vload(s=0, d=0)
        vload(s=8, d=1)
        vload(s=16, d=2)
        vadd(a=0, b=1, d=3)  # V3 = a + b
        vstore(s=3, d=24)
        tpu.inspect(bram[24:32], label="sum")
        vmul(a=3, b=2, d=3)  # V3 = (a + b) * c
        vstore(s=3, d=24)
        tpu.inspect(bram[24:32], label="prod")
        vrelu(a=3, d=3)  # V3 = relu((a + b) * c)
        vstore(s=3, d=24)
        tpu.inspect(bram[24:32], label="out")

    res = prog()
    _check("sum  = a + b", res["sum"], a + b)
    _check("prod = (a + b) * c", res["prod"], (a + b) * c)
    _check("out  = relu((a + b) * c)", res["out"], np.maximum((a + b) * c, 0))


# ==========================================================================#
# 5. Elementwise fused multiply-add
# ==========================================================================#
def elementwise_madd() -> None:
    """An 8-lane fused multiply-add: y = a * b + c."""
    rng = np.random.default_rng(4)
    a = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    c = rng.standard_normal(8).astype(np.float32)
    init = _bram_init([(0, a), (8, b), (16, c)])

    @tpu.oracle(init={bram: init})
    def prog():
        vload(s=0, d=0)
        vload(s=8, d=1)
        vload(s=16, d=2)
        vmul(a=0, b=1, d=3)  # V3 = a * b
        vadd(a=3, b=2, d=3)  # V3 = a * b + c
        vstore(s=3, d=24)
        tpu.inspect(bram[24:32], label="y")

    _check("y = a * b + c", prog()["y"], a * b + c)


# ==========================================================================#
# 6. Single systolic matmul tile
# ==========================================================================#
def matmul_tile() -> None:
    """A single 4x4 systolic matmul: Z = X @ W^T. The weight is consumed
    transposed by the array (modeled with primitive.transpose in the ISA semantics),
    so the NumPy reference transposes W."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 4)).astype(np.float32)
    W = rng.standard_normal((4, 4)).astype(np.float32)
    init = _bram_init([(0, W), (16, X)])  # ADDR_A=W, ADDR_B=X

    @tpu.oracle(init={bram: init})
    def prog():
        matmul(w=0, x=16, z=32)
        tpu.inspect(bram[32:48], label="Z")

    _check("Z = X @ W^T", prog()["Z"].reshape(4, 4), X @ W.T)


# ==========================================================================#
# 7. Vector add  (simplest)
# ==========================================================================#
def vector_add() -> None:
    """Load two 8-lane vectors, add, store, read back: c = a + b."""
    a = np.arange(8, dtype=np.float32)
    b = np.arange(8, dtype=np.float32) * 10.0
    init = _bram_init([(0, a), (8, b)])

    @tpu.oracle(init={bram: init})
    def prog():
        vload(s=0, d=0)
        vload(s=8, d=1)
        vadd(a=0, b=1, d=2)
        vstore(s=2, d=16)
        tpu.inspect(bram[16:24], label="c")

    _check("c = a + b", prog()["c"], a + b)


EXAMPLES = [
    tiled_matmul_accumulate,
    two_layer_mlp,
    dense_layer,
    vector_pipeline,
    elementwise_madd,
    matmul_tile,
    vector_add,
]


def main() -> None:
    for fn in EXAMPLES:
        print(f"\n=== {fn.__name__} ===")
        print(f"    {fn.__doc__.strip().splitlines()[0]}")
        fn()
    print(f"\nAll {len(EXAMPLES)} examples passed NumPy verification.")


if __name__ == "__main__":
    main()
