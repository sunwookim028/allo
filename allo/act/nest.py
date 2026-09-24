# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A loop nest, in ACT's four fields, and the legality every target shares."""

from collections import namedtuple

OUTER, INTRINSIC = "outer", "intrinsic"

Loop = namedtuple("Loop", "rank factor level spatial")
Loop.__new__.__defaults__ = (OUTER, False)


class Refused(Exception):
    """A nest this target cannot compile, tagged by the cause that refused it."""

    def __init__(self, cause, detail, also=()):
        super().__init__(f"{cause}: {detail}")
        self.cause = cause
        self.detail = detail
        self.also = tuple(dict.fromkeys((cause,) + tuple(also)))


def refuse(refusals):
    if refusals:
        cause, detail = refusals[0]
        raise Refused(cause, detail, also=[c for c, _ in refusals])


def covered(nest):
    out = {}
    for loop in nest:
        out[loop.rank] = out.get(loop.rank, 1) * loop.factor
    return out


def check_coverage(nest, extents):
    got = covered(nest)
    for rank, extent in extents.items():
        if got.get(rank, 1) != extent:
            raise Refused(
                "coverage",
                f"rank {rank} factors to {got.get(rank, 1)}, extent {extent}")
    for rank in got:
        if rank not in extents:
            raise Refused("coverage", f"unknown rank {rank!r}")


def check_sequential(nest):
    for loop in nest:
        if loop.spatial:
            raise Refused(
                "spatial",
                f"loop {loop.rank!r} fans out across instances, and this "
                f"target has one instance and no instance index")


def peel_intrinsic(nest):
    if not nest:
        raise Refused("intrinsic", "empty nest")
    if nest[-1].level != INTRINSIC:
        raise Refused(
            "intrinsic",
            f"the innermost loop is at level {nest[-1].level!r}, not "
            f"{INTRINSIC!r}")
    cut = len(nest)
    while cut and nest[cut - 1].level == INTRINSIC:
        cut -= 1
    for loop in nest[:cut]:
        if loop.level == INTRINSIC:
            raise Refused(
                "intrinsic",
                f"{order(nest)} puts an intrinsic loop outside the innermost "
                f"band")
    return tuple(nest[:cut]), covered(nest[cut:])


def order(nest):
    return ">".join(f"{l.rank}{l.factor}" for l in nest) or "-"


def emitted_order(nest):
    return order(tuple(l for l in nest if l.level != INTRINSIC))


def positions(nest, rank):
    return [i for i, l in enumerate(nest) if l.rank == rank]
