# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DRAM -> scratchpad or operand vector registers, sole reader of the operand
matrices. See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from allo.customize import Partition

from ..compose import unit


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
def dma_ld(lA: int8[MAXDIM * MAXDIM], lB: int8[MAXDIM * MAXDIM]):
    nw: UInt(64) = c_dld.get()
    n_row: int32 = nw[0:16]
    sw: UInt(64) = c_dld.get()  # the DRAM row span of each source matrix
    na: int32 = sw[0:16]
    nb: int32 = sw[16:32]

    # One burst per matrix, each covering exactly that matrix's own span.
    # Merging the two into one loop bounded by `max(na, nb)` was measured
    # and moved nothing: the bursts are hidden behind the prefetch.
    rbA: UInt(VW)[MAXDIM * WPR]
    rbB: UInt(VW)[MAXDIM * WPR]
    for ia in range(na * WPR):
        pa: UInt(VW) = 0
        with allo.meta_for(T) as e:
            av: int8 = lA[ia * T + e]
            pa[8 * e : 8 * (e + 1)] = av
        rbA[ia] = pa
    for ic in range(nb * WPR):
        pb: UInt(VW) = 0
        with allo.meta_for(T) as e2:
            bv: int8 = lB[ic * T + e2]
            pb[8 * e2 : 8 * (e2 + 1)] = bv
        rbB[ic] = pb

    f0: int32 = 0
    f1: int32 = 0
    f2: int32 = 0
    cnt: int32 = 0
    r: int32 = -1               # advanced at the TOP: see the II note
    for x in range(n_row):
        r += 1
        if r >= cnt:
            # Only `dma_ld` reaches this queue, so there is no opcode test.
            w0: UInt(64) = c_dld.get()
            f0 = w0[6:18]
            f1 = w0[18:30]
            f2 = w0[30:42]
            cnt = w0[54:62]
            r = 0
        pw: UInt(VW) = 0
        if (f0 & DMA_SRC_B) == 0:
            pw = rbA[(f1 + r) * WPR + f2]
        else:
            pw = rbB[(f1 + r) * WPR + f2]
        if f0 >= DMA_TO_VR:
            dma2vr.put(pw)
        else:
            dma2sp.put(pw)
