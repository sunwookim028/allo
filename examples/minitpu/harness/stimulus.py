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


def bf16_sweep_a():
    """``uint16[2**16 * 44, 2]``: every bf16 ``a`` crossed with every corner
    ``b``, both orders. Exhaustive in one operand, for the units whose
    special-case tests read one operand's bit pattern."""
    a = np.arange(1 << 16, dtype=np.uint16)
    c = bf16_corners()
    x, y = np.meshgrid(a, c, indexing="ij")
    ab = np.stack([x.ravel(), y.ravel()], axis=1)
    return np.concatenate([ab, ab[:, ::-1]]).astype(np.uint16)


# acc24: MiniTPU's MXU accumulator, 1 sign + 8 exponent (bias 127) + 15
# fraction bits, with subnormals (vpu_pkg MXU_ACC_FRAC_W = 15).
ACC24_FRAC = 15


def acc24_corners():
    """Every acc24 class at its edges, both signs (46 values)."""
    F = ACC24_FRAC
    mags = [
        0x000000,  # zero
        0x000001,  # smallest subnormal
        0x000002,
        0x004000,  # mid subnormal
        0x007FFF,  # largest subnormal
        0x008000,  # smallest normal
        0x008001,
        0x00FFFF,
        0x010000,
        127 << F,  # 1.0
        (127 << F) | 1,  # 1 + ulp
        (128 << F) - 1,  # just under 2
        128 << F,  # 2.0
        111 << F,  # 2^-16: half-ulp of 1.0, the rounding-tie scale
        (111 << F) | 1,
        110 << F,
        (254 << F),
        (254 << F) | 0x7FFE,
        (254 << F) | 0x7FFF,  # largest finite
        0x7F8000,  # inf
        0x7FC000,  # quiet NaN, canonical
        0x7F8001,  # signalling-pattern NaN
        0x7FFFFF,  # NaN, all payload
    ]
    return np.array(mags + [m | 0x800000 for m in mags], dtype=np.uint32)


def acc24_ties(n, seed, max_gap=20):
    """Pairs whose exact sum lands on, or near, an acc24 rounding tie, and
    near-cancellations (gap 0, opposite signs) that normalize far."""
    F = ACC24_FRAC
    rng = np.random.default_rng(seed)
    e = rng.integers(1, 255, n)
    fa = rng.integers(0, 1 << F, n)
    d = rng.integers(0, max_gap + 1, n)
    fb = rng.integers(0, 1 << F, n)
    near = rng.integers(0, 2, n) == 1  # half of them: b's fraction close to a's
    fb = np.where(near, (fa + rng.integers(-4, 5, n)) & ((1 << F) - 1), fb)
    sa = rng.integers(0, 2, n)
    sb = rng.integers(0, 2, n)
    a = (sa << 23) | (e << F) | fa
    b = (sb << 23) | (np.clip(e - d, 0, 254) << F) | fb
    return a.astype(np.uint32), b.astype(np.uint32)


def acc24_edges(n, seed):
    """Random pairs with exponents at the subnormal and overflow ends, as
    MiniTPU's ``tb/tb_acc24_add_pipe.cpp`` biases its draws."""
    F = ACC24_FRAC
    rng = np.random.default_rng(seed)
    ends = np.array([0, 1, 2, 3, 252, 253, 254])
    e = ends[rng.integers(0, len(ends), (n, 2))]
    f = rng.integers(0, 1 << F, (n, 2))
    s = rng.integers(0, 2, (n, 2))
    return ((s << 23) | (e << F) | f).astype(np.uint32)


def binary_acc24(n_random=200_000, seed=0xACC24):
    """``uint32[m, 2]``: corners crossed, ties, the range ends, then random."""
    c = acc24_corners()
    a, b = np.meshgrid(c, c, indexing="ij")
    parts = [np.stack([a.ravel(), b.ravel()], axis=1)]
    ta, tb = acc24_ties(n_random // 2, seed + 1)
    parts.append(np.stack([ta, tb], axis=1))
    parts.append(acc24_edges(n_random // 4, seed + 2))
    rng = np.random.default_rng(seed)
    parts.append(rng.integers(0, 1 << 24, (n_random, 2), dtype=np.uint64).astype(np.uint32))
    return np.concatenate(parts).astype(np.uint32)


def alu_ops(ops, n_random=200_000, seed=0xA1F):
    """``uint16[m, 3]`` of ``(op, a, b)``: ``binary_bf16`` under each op code
    in ``ops``, plus every ``a`` (exhaustive) against a random ``b`` per op."""
    ab = binary_bf16(n_random, seed)
    rng = np.random.default_rng(seed + 7)
    every = np.stack(
        [np.arange(1 << 16), rng.integers(0, 1 << 16, 1 << 16)], axis=1
    ).astype(np.uint16)
    parts = []
    for op in ops:
        for x in (ab, every):
            parts.append(np.concatenate([np.full((len(x), 1), op, np.uint16), x], axis=1))
    return np.concatenate(parts)
