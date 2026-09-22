# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SIMD scratchpad, its sole owner, and the array's weight port.

One row is one packed word of T int8 lanes and there is no way to address a
lane, which is what lets a single-ported memory feed T lanes per cycle.
See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from ..compose import unit


@unit(
    reads=("c_spm", "dma2sp"),
    writes=("sp2vr", "wcol"),
    parameters=("SPAD_ROWS", "VW"),
    isa=("OP_MM", "OP_DMA_LD", "OP_VLD"),
)
def spm():
    spad: UInt(VW)[SPAD_ROWS]
    count_word: UInt(64) = c_spm.get()
    n_row: int32 = count_word[0:16]
    array_counts_word: UInt(64) = c_spm.get()
    array_counts: UInt(VW) = 0
    array_counts[0:32] = array_counts_word[0:32]   # mm count | mm rows << 16
    wcol[0].put(array_counts)
    op: int32 = 0
    f1: int32 = 0               # mm: the array's row count; vld: the spad row
    spad_base: int32 = 0
    instr_rows: int32 = 0
    row: int32 = -1             # advanced at the top: hoisting this holds II=1
    for work in range(n_row):
        row += 1
        if row >= instr_rows:
            word: UInt(64) = c_spm.get()
            op = word[0:6]
            f1 = word[18:30]
            spad_base = word[42:54]
            instr_rows = word[54:62]
            row = 0
        read_row: int32 = f1 + row
        if op == OP_MM:
            # row 0 is the header; rows 1..T are weight rows 0..T-1, row 0
            # first, so PE row i keeps the i-th word it sees.
            read_row = spad_base + row - 1
            if row == 0:
                read_row = spad_base
        if op == OP_DMA_LD:
            spad[spad_base + row] = dma2sp.get()
        else:
            # ONE spad read per iteration at a muxed address, which a dual-port
            # BRAM holds at II=1.
            loaded: UInt(VW) = spad[read_row]
            if op == OP_VLD:
                sp2vr.put(loaded)
            else:
                weight_word: UInt(VW) = loaded
                if row == 0:
                    header: UInt(VW) = 0
                    header[0:12] = f1
                    weight_word = header
                wcol[0].put(weight_word)
