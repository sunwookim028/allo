# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulator -> DRAM, sole writer of the result matrix. Composed last,
because Vitis csim runs a region's processes in declaration order and this one
consumes what the accumulator produces (limitations register item 15)."""

from __future__ import annotations

from allo.customize import Partition

from allo.compose import unit


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
def dma_st(dram_c: int8[MAXDIM * MAXDIM]):
    count_word: UInt(64) = c_dst.get()
    n_row: int32 = count_word[0:16]
    dram_row0: int32 = 0
    col_block: int32 = 0
    instr_rows: int32 = 0
    row: int32 = -1             # advanced at the top: hoisting this holds II=1
    for work in range(n_row):
        row += 1
        if row >= instr_rows:
            # Only `mvout` reaches this queue, so there is no opcode test.
            word: UInt(64) = c_dst.get()
            dram_row0 = word[18:30]
            col_block = word[30:42]
            instr_rows = word[54:62]
            row = 0
        clipped_word: UInt(VW) = ac2sp.get()
        with allo.meta_for(T) as lane:
            lane_value: int8 = clipped_word[8 * lane : 8 * (lane + 1)]
            dram_c[(dram_row0 + row) * MAXDIM + col_block * T + lane] = lane_value
