# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the enumerator promises: exact coverage, a pinned intrinsic, no repeats."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from act import mapspace  # noqa: E402
from act.nest import (  # noqa: E402
    INTRINSIC, Refused, check_coverage, covered, peel_intrinsic,
)

EXTENTS = {"M": 16, "K": 16, "N": 8}
INTRINSICS = [{"M": rows, "K": 4, "N": 4} for rows in (1, 2, 4, 8, 16)]


def every_nest(extents=None, intrinsics=None, slots=2):
    return list(mapspace.nests(extents or EXTENTS, intrinsics or INTRINSICS,
                               slots))


def test_factorings_are_exactly_the_ordered_factorisations():
    assert mapspace.factorings(1, 2) == [(1, 1)]
    assert sorted(mapspace.factorings(4, 2)) == [(1, 4), (2, 2), (4, 1)]
    for n in (6, 12, 16):
        for parts in (1, 2, 3):
            got = mapspace.factorings(n, parts)
            assert all(math.prod(f) == n for f in got)
            assert len(got) == len(set(got))


def test_every_nest_covers_the_extents_exactly():
    for nest in every_nest():
        check_coverage(nest, EXTENTS)
        assert covered(nest) == EXTENTS


def test_no_nest_is_enumerated_twice():
    nests = every_nest()
    assert len(nests) == len(set(nests))


def test_the_intrinsic_is_the_innermost_band_of_every_nest():
    for nest in every_nest():
        outer, intrinsic = peel_intrinsic(nest)
        assert all(loop.level != INTRINSIC for loop in outer)
        assert intrinsic["K"] == 4 and intrinsic["N"] == 4


def test_an_intrinsic_that_does_not_divide_is_dropped():
    assert every_nest(intrinsics=[{"M": 5, "K": 4, "N": 4}]) == []


def test_slots_only_grow_the_space():
    one = set(every_nest(slots=1))
    two = set(every_nest(slots=2))
    assert one and one <= two and len(two) > len(one)


def test_coverage_names_the_rank_that_is_wrong():
    nest = every_nest()[0]
    with pytest.raises(Refused) as caught:
        check_coverage(nest, dict(EXTENTS, M=15))
    assert caught.value.cause == "coverage"
    assert "M" in caught.value.detail
