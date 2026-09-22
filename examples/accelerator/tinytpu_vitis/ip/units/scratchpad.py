# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SIMD scratchpad, its sole owner, and the array's weight port. See
``docs/source/designs/tinytpu_isa.rst``."""

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
    nw: UInt(64) = c_spm.get()
    n_row: int32 = nw[0:16]
    mw: UInt(64) = c_spm.get()
    nmo: UInt(VW) = 0
    nmo[0:32] = mw[0:32]        # mm count | wavefront rows << 16
    wcol[0].put(nmo)
    op: int32 = 0
    f1: int32 = 0
    f3: int32 = 0
    cnt: int32 = 0
    r: int32 = -1               # advanced at the TOP: see the II note
    for x in range(n_row):
        r += 1
        if r >= cnt:
            w0: UInt(64) = c_spm.get()
            op = w0[0:6]
            f1 = w0[18:30]
            f3 = w0[42:54]
            cnt = w0[54:62]
            r = 0
        ra: int32 = f1 + r
        if op == OP_MM:
            # iteration 0 is the header; iterations 1..T are weight rows
            # 0..T-1, row 0 first: PE row i keeps the i-th word it sees.
            ra = f3 + r - 1
            if r == 0:
                ra = f3
        if op == OP_DMA_LD:
            spad[f3 + r] = dma2sp.get()
        else:
            lw: UInt(VW) = spad[ra]
            if op == OP_VLD:
                sp2vr.put(lw)
            else:
                ow: UInt(VW) = lw
                if r == 0:
                    hdr: UInt(VW) = 0
                    hdr[0:12] = f1
                    ow = hdr
                wcol[0].put(ow)
        # No write-back branch: the scratchpad is input-only. Results
        # leave through the accumulator, which is a separate memory.
