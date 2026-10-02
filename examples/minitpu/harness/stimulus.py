# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stimulus for float units: corner values, crossed, plus seeded random.

Exhaustive is the RTL side's job where it is cheap (MiniTPU's own ``tb/``
does 2^32 for the multiplier); the Allo side runs these finite sets.
"""

import numpy as np


def bf16_corners():
    """Every bf16 class at its edges, both signs (44 values)."""
    mags = [
        0x0000,  # zero
        0x0001,  # smallest subnormal
        0x0002,
        0x0040,  # mid subnormal
        0x007F,  # largest subnormal
        0x0080,  # smallest normal
        0x0081,
        0x00FF,
        0x3F80,  # 1.0
        0x3F81,  # 1 + ulp
        0x3FFF,  # just under 2
        0x4000,  # 2.0
        0x3B80,  # 2^-8: half-ulp of 1.0, the rounding-tie scale
        0x3C00,
        0x7F00,
        0x7F7E,
        0x7F7F,  # largest finite
        0x7F80,  # inf
        0x7FC0,  # quiet NaN, canonical
        0x7F81,  # signalling-pattern NaN
        0x7FFF,  # NaN, all payload
        0x0100,
    ]
    return np.array(mags + [m | 0x8000 for m in mags], dtype=np.uint16)


def bf16_ties(n, seed):
    """Pairs whose exact sum lands on, or one ulp around, a rounding tie."""
    rng = np.random.default_rng(seed)
    e = rng.integers(1, 250, n)
    fa = rng.integers(0, 128, n)
    d = rng.integers(1, 9, n)  # exponent gap that puts b's bits at the round point
    fb = rng.integers(0, 128, n)
    sa = rng.integers(0, 2, n)
    sb = rng.integers(0, 2, n)
    a = (sa << 15) | (e << 7) | fa
    b = (sb << 15) | (np.clip(e - d, 1, 254) << 7) | fb
    return a.astype(np.uint16), b.astype(np.uint16)


def binary_bf16(n_random=200_000, seed=0xBF16):
    """``uint16[m, 2]``: corners crossed with corners, ties, then random."""
    c = bf16_corners()
    a, b = np.meshgrid(c, c, indexing="ij")
    parts = [np.stack([a.ravel(), b.ravel()], axis=1)]
    ta, tb = bf16_ties(n_random // 4, seed + 1)
    parts.append(np.stack([ta, tb], axis=1))
    rng = np.random.default_rng(seed)
    parts.append(rng.integers(0, 1 << 16, (n_random, 2), dtype=np.uint64).astype(np.uint16))
    return np.concatenate(parts).astype(np.uint16)
