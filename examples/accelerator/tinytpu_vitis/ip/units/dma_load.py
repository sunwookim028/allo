# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DRAM -> scratchpad or operand vector registers, sole reader of the operand
matrices. See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from allo.customize import Partition

from examples.accelerator.tinytpu_vitis.ip.compose import unit


def dma_load_directives(s, ctx):
    # T lanes per cycle out of each operand port.
    factor = ctx.parameters["T"]
    s.partition(ctx.memory("A"), Partition.Cyclic, dim=2, factor=factor)
    s.partition(ctx.memory("B"), Partition.Cyclic, dim=2, factor=factor)


@unit(
    memories=("A", "B"),
    reads=("c_dld",),
    writes=("dma2sp", "dma2vr"),
    parameters=("MAXDIM", "WPR", "T", "VW"),
    isa=("DMA_SRC_B", "DMA_TO_VR"),
    directives=dma_load_directives,
)
def dma_ld(dram_a: int8[MAXDIM * MAXDIM], dram_b: int8[MAXDIM * MAXDIM]):
    count_word: UInt(64) = c_dld.get()
    n_row: int32 = count_word[0:16]
    span_word: UInt(64) = c_dld.get()
    a_rows: int32 = span_word[0:16]
    b_rows: int32 = span_word[16:32]

    # One variable-length burst per matrix, covering exactly the DRAM rows the
    # program names: a per-row strided read is a separate four-beat AXI
    # transaction per row (`[HLS 214-115] Multiple burst reads of length 4`,
    # II=4). Merging the two into one loop bounded by max(a_rows, b_rows)
    # halves the burst time and was measured to move nothing -- the bursts are
    # already hidden behind the sequencer's prefetch.
    a_onchip: UInt(VW)[MAXDIM * WPR]
    b_onchip: UInt(VW)[MAXDIM * WPR]
    for a_word in range(a_rows * WPR):
        packed_a: UInt(VW) = 0
        with allo.meta_for(T) as a_lane:
            a_value: int8 = dram_a[a_word * T + a_lane]
            packed_a[8 * a_lane : 8 * (a_lane + 1)] = a_value
        a_onchip[a_word] = packed_a
    for b_word in range(b_rows * WPR):
        packed_b: UInt(VW) = 0
        with allo.meta_for(T) as b_lane:
            b_value: int8 = dram_b[b_word * T + b_lane]
            packed_b[8 * b_lane : 8 * (b_lane + 1)] = b_value
        b_onchip[b_word] = packed_b

    route: int32 = 0
    dram_row0: int32 = 0
    col_block: int32 = 0
    instr_rows: int32 = 0
    row: int32 = -1             # advanced at the top: hoisting this holds II=1
    for work in range(n_row):
        row += 1
        if row >= instr_rows:
            # Only `dma_ld` reaches this queue, so there is no opcode test.
            word: UInt(64) = c_dld.get()
            route = word[6:18]
            dram_row0 = word[18:30]
            col_block = word[30:42]
            instr_rows = word[54:62]
            row = 0
        packed: UInt(VW) = 0
        if (route & DMA_SRC_B) == 0:
            packed = a_onchip[(dram_row0 + row) * WPR + col_block]
        else:
            packed = b_onchip[(dram_row0 + row) * WPR + col_block]
        if route >= DMA_TO_VR:
            dma2vr.put(packed)
        else:
            dma2sp.put(packed)
