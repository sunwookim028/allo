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
    # A widened burst writes DMA_WORDS words of each buffer per iteration.
    # Banked, every bank has one writer; left alone, Vitis emits a
    # two-write-port RAM, which standard cells cannot build (ELAB-366).
    # docs/source/designs/benchmarks.rst has the measurement.
    words = ctx.parameters["DMA_WORDS"]
    if words > 1:
        for buffer in ("a_onchip", "b_onchip"):
            s.partition(f"{ctx.instance('dma_ld')}:{buffer}", Partition.Cyclic,
                        dim=1, factor=words)


@unit(
    memories=("A", "B"),
    reads=("c_dld",),
    writes=("dma2sp", "dma2vr"),
    parameters=("MAXDIM", "WPR", "T", "VW", "DMA_WORDS"),
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
    # program names. Merging the two into one loop bounded by
    # max(a_rows, b_rows) halves the burst time and was measured to move
    # nothing -- the bursts are already hidden behind the sequencer's prefetch.
    #
    # THE "STRIDED COSTS 4x" JUSTIFICATION FOR STAGING IS STALE, and it is the
    # reason this mirror exists. It was measured before `align_value(64)` and
    # `-m_axi_max_widen_bitwidth 512` were in the build; with them gmem1/gmem2
    # are 32 bits -- exactly one packed word at T=4 -- so a row read is one
    # beat, Vitis emits NO `[HLS 214-115]` note for the operand ports at all
    # (the only one left is gmem0, the 512-bit instruction port), and a flat
    # per-row `m_axi` loop closes at `Final II = 1, Depth = 17`. Measured on
    # `dma_ld_0_1_Pipeline_VITIS_LOOP_645_1` of a T=4 MAXDIM=16 QD=16 build
    # on 2026-09-24, and independently in docs/source/backends/vitis.rst
    # ("two burst loops from II=4 to II=1").
    #
    # So the mirror must be defended on what it still buys -- amortising
    # re-read of a row the program names more than once -- not on burst shape.
    # dev/records/tinytpu/big_shapes_settlement.rst has the cycle cost of
    # removing it: +6 / 0 / -2 / -6 / +10 on the published five.
    #
    # The burst WIDTH is a parameter. At DMA_WORDS=1 this is the shipped loop,
    # one packed word an iteration; above 1 each iteration reads DMA_WORDS
    # whole words -- up to the 64-byte beat `align_value(64)` lets Vitis widen
    # the port to -- so the burst costs a factor of DMA_WORDS fewer iterations.
    # The `meta_for` unrolls inside a runtime loop, so the trip count stays
    # runtime data. It is parametric rather than landed because it was found at
    # `-m_axi_latency 0`, and a change whose whole benefit is wider bursts is
    # exactly the kind whose advantage can grow or vanish with memory latency.
    a_onchip: UInt(VW)[MAXDIM * WPR + DMA_WORDS]
    b_onchip: UInt(VW)[MAXDIM * WPR + DMA_WORDS]
    a_groups: int32 = (a_rows * WPR + (DMA_WORDS - 1)) // DMA_WORDS
    for a_group in range(a_groups):
        with allo.meta_for(DMA_WORDS) as a_word:
            packed_a: UInt(VW) = 0
            with allo.meta_for(T) as a_lane:
                a_value: int8 = dram_a[(a_group * DMA_WORDS + a_word) * T + a_lane]
                packed_a[8 * a_lane : 8 * (a_lane + 1)] = a_value
            a_onchip[a_group * DMA_WORDS + a_word] = packed_a
    b_groups: int32 = (b_rows * WPR + (DMA_WORDS - 1)) // DMA_WORDS
    for b_group in range(b_groups):
        with allo.meta_for(DMA_WORDS) as b_word:
            packed_b: UInt(VW) = 0
            with allo.meta_for(T) as b_lane:
                b_value: int8 = dram_b[(b_group * DMA_WORDS + b_word) * T + b_lane]
                packed_b[8 * b_lane : 8 * (b_lane + 1)] = b_value
            b_onchip[b_group * DMA_WORDS + b_word] = packed_b

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
