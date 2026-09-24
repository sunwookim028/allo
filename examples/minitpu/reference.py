# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical reference for the MiniTPU model, in MiniTPU's own arithmetic.

This is not an idealised GEMM. It reproduces the accumulation order and the
rounding points that ``~/core/minitpu/docs/ARITHMETIC.md`` records, so that a
mismatch against the Allo model means the model is wrong rather than that
numpy is more accurate:

* the BF16 x BF16 product is exact (16-bit significand) and is never rounded,
* row 0 of the array adds into a zero psum through the adder's zero bypass,
  so the first term of every column is exact,
* rows 1..15 round to acc24 (1 sign + 8 exp + 15 fraction) after every add,
  strictly in ascending contraction index -- a chain, never a tree,
* the column result is rounded to BF16 exactly once, at the array edge,
* a contraction deeper than 16 is summed across 16-deep tiles by the vector
  ALU in BF16, ascending (``vadd``), which is where the error actually comes
  from.

acc24 is emulated by rounding a float32 to 15 fraction bits. That is **exact**
for the range: ``MXU_ACC_W = 1 + 8 + 15`` gives acc24 the *same 8-bit exponent
field as float32*, so the subnormal and overflow thresholds are identical, not
inherited approximations. It holds because the accumulator is float; a
dtype-parametrised variant with an integer accumulator would break it silently.

The one measured difference from the RTL: a float32 add followed by an acc24
round can double-round where the RTL rounds the exact sum once. Measured over
24 million random acc24 operand pairs, the two disagree on **6,933 of them
(0.03 %)**, always by one ulp of acc24 (2^-16 relative), which usually
disappears at the final BF16 rounding. MiniTPU's own host reference
(``tools/accum_precision.py``) takes the same float32 path, so this model and
that one agree; neither can be held to silicon until the golden vector set
MiniTPU's owner is building exists.
"""

import numpy as np
import ml_dtypes

bf16 = ml_dtypes.bfloat16

#: Array edge length: ``NUM_LANES`` (``vpu_pkg.sv:25``).
DIM = 16
#: Rows per VREG: ``NUM_SUBLANES`` (``vpu_pkg.sv:28``).
SUB = 4
#: ``MXU_ACC_FRAC_W`` (``vpu_pkg.sv:100``): acc24 keeps 15 fraction bits.
ACC_FRAC_W = 15
_DROP = 23 - ACC_FRAC_W  # float32 has 23; drop 8, as ``ACC_DROP_W`` does.


def acc24(x):
    """Round a float32 to the 24-bit accumulator, round-to-nearest-even.

    The same add-half-plus-lsb rule the RTL uses in
    ``mxu_acc24_add_pipe.sv`` and in ``pack_bf16``.
    """
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    half = np.uint32((1 << _DROP) >> 1) - np.uint32(1)
    lsb = (u >> np.uint32(_DROP)) & np.uint32(1)
    mask = np.uint32((0xFFFFFFFF << _DROP) & 0xFFFFFFFF)
    r = (u + half + lsb) & mask
    return r.view(np.float32)


def tile_gemm(a_tile, w_tile):
    """One MXU tile: ``a_tile`` (rows x 16) against ``w_tile`` (16 x 16).

    Returns BF16, as ``vmatpop`` delivers it.
    """
    a = np.asarray(a_tile, dtype=bf16).astype(np.float32)
    w = np.asarray(w_tile, dtype=bf16).astype(np.float32)
    depth = w.shape[0]
    # Row 0: the zero bypass returns the product bit-for-bit -- exact.
    psum = a[:, 0, None] * w[None, 0, :]
    for k in range(1, depth):
        psum = acc24(psum + a[:, k, None] * w[None, k, :])
    return psum.astype(bf16)


def gemm(a, b, dim=DIM, sub=SUB):
    """A full GEMM the way MiniTPU runs it.

    ``a`` is M x K (tokens x depth), ``b`` is K x N.  K is split into 16-deep
    tiles; each tile is one weight load and its result is one BF16 value per
    output element; the tiles are summed in BF16 by the vector ALU, ascending.
    """
    a = np.asarray(a, dtype=bf16)
    b = np.asarray(b, dtype=bf16)
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    assert k % dim == 0 and n % dim == 0 and m % sub == 0
    out = None
    for k0 in range(0, k, dim):
        step = np.concatenate(
            [
                tile_gemm(a[:, k0 : k0 + dim], b[k0 : k0 + dim, n0 : n0 + dim])
                for n0 in range(0, n, dim)
            ],
            axis=1,
        )
        # Cross-tile accumulation is a BF16 ``vadd`` into an ordinary VREG.
        out = (
            step
            if out is None
            else (out.astype(np.float32) + step.astype(np.float32)).astype(bf16)
        )
    return out


def fp64_gemm(a, b):
    """The oracle MiniTPU measures against: the same operands in float64."""
    return np.asarray(a, dtype=bf16).astype(np.float64) @ np.asarray(
        b, dtype=bf16
    ).astype(np.float64)


def rel_error(got, want):
    """Relative Frobenius-norm error, MiniTPU's acceptance measure.

    ``board_package/gpt2_kernels.py`` sets the bar at 1 %.
    """
    got = np.asarray(got, dtype=bf16).astype(np.float64)
    want = np.asarray(want, dtype=np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))
