# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every nest a target's intrinsic admits: factorings x permutations."""

import itertools

from allo.act.nest import INTRINSIC, OUTER, Loop


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def factorings(n, slots):
    if slots == 1:
        return [(n,)]
    return [(d,) + rest
            for d in divisors(n)
            for rest in factorings(n // d, slots - 1)]


def residual(extents, intrinsic):
    out = {}
    for rank, extent in extents.items():
        tile = intrinsic.get(rank, 1)
        if extent % tile:
            return None
        out[rank] = extent // tile
    return out


def nests(extents, intrinsics, slots=2):
    ranks = tuple(extents)
    seen = set()
    for intrinsic in intrinsics:
        left = residual(extents, intrinsic)
        if left is None:
            continue
        tail = tuple(Loop(r, intrinsic[r], INTRINSIC)
                     for r in ranks if intrinsic.get(r, 1) != 1)
        for combo in itertools.product(*(factorings(left[r], slots)
                                         for r in ranks)):
            outer = [Loop(r, f, OUTER)
                     for r, fs in zip(ranks, combo) for f in fs if f != 1]
            for perm in itertools.permutations(outer):
                nest = tuple(perm) + tail
                if nest not in seen:
                    seen.add(nest)
                    yield nest


def size(extents, intrinsics, slots=2):
    return sum(1 for _ in nests(extents, intrinsics, slots))
