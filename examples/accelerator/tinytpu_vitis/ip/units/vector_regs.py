# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operand vector registers, their sole owner, and the array's activation
port. Written by `vld` from the scratchpad and by a `dma_ld` addressed here,
read by `mm`. See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from examples.accelerator.tinytpu_vitis.ip.compose import unit


@unit(
    reads=("c_vru", "sp2vr", "dma2vr"),
    writes=("acol",),
    parameters=("NVR", "VW"),
    isa=("OP_MM", "OP_DMA_LD", "OP_VLD"),
)
def vru():
    count_word: UInt(64) = c_vru.get()
    n_word: int32 = count_word[0:16]
    vr: UInt(VW)[NVR]
    op: int32 = 0
    vr_base: int32 = 0          # mm: the activation rows; vld: the destination
    dma_base: int32 = 0
    instr_rows: int32 = 0
    row: int32 = -1             # advanced at the top: hoisting this holds II=1
    for work in range(n_word):
        row += 1
        if row >= instr_rows:
            word: UInt(64) = c_vru.get()
            op = word[0:6]
            vr_base = word[6:18]
            dma_base = word[42:54]
            instr_rows = word[54:62]
            row = 0
        # ONE vr read and ONE vr write per iteration, each at a muxed address
        # and the write from a muxed source: that is what holds II=1 in the
        # unit that used to be the critical one.
        if op == OP_MM:
            activation: UInt(VW) = vr[vr_base + row]
            acol[0].put(activation)
        else:
            write_row: int32 = vr_base + row
            if op == OP_DMA_LD:
                write_row = dma_base + row
            write_word: UInt(VW) = 0
            if op == OP_VLD:
                write_word = sp2vr.get()
            else:
                write_word = dma2vr.get()
            vr[write_row] = write_word
