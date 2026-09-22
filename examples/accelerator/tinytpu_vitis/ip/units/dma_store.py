# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulator -> DRAM, sole writer of the result matrix. Composed last,
because Vitis csim runs a region's processes in declaration order and this one
consumes what the accumulator produces (limitations register item 15)."""

from __future__ import annotations

from allo.customize import Partition

from ..compose import unit


def dma_store_directives(s, ctx):
    # T lanes per cycle into the result port.
    s.partition(ctx.memory("C"), Partition.Cyclic, dim=2,
                factor=ctx.parameters["T"])


@unit(
    memories=("C",),
    reads=("c_dst", "ac2sp"),
    parameters=("MAXDIM", "T", "VW"),
    directives=dma_store_directives,
)
def dma_st(lC: int8[MAXDIM * MAXDIM]):
    nw: UInt(64) = c_dst.get()
    n_row: int32 = nw[0:16]
    f1: int32 = 0
    f2: int32 = 0
    cnt: int32 = 0
    r: int32 = -1               # advanced at the TOP: see the II note
    for x in range(n_row):
        r += 1
        if r >= cnt:
            # Only `mvout` reaches this queue, so there is no opcode test.
            w0: UInt(64) = c_dst.get()
            f1 = w0[18:30]
            f2 = w0[30:42]
            cnt = w0[54:62]
            r = 0
        qw: UInt(VW) = ac2sp.get()
        with allo.meta_for(T) as e:
            ov: int8 = qw[8 * e : 8 * (e + 1)]
            lC[(f1 + r) * MAXDIM + f2 * T + e] = ov
