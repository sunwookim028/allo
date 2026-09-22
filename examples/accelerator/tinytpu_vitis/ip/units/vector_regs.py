# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operand vector registers, their sole owner, and the array's activation
port. See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from ..compose import unit


@unit(
    reads=("c_vru", "sp2vr", "dma2vr"),
    writes=("acol",),
    parameters=("NVR", "VW"),
    isa=("OP_MM", "OP_DMA_LD", "OP_VLD"),
)
def vru():
    nw: UInt(64) = c_vru.get()
    n_word: int32 = nw[0:16]
    vr: UInt(VW)[NVR]
    op: int32 = 0
    f0: int32 = 0
    f3: int32 = 0
    cnt: int32 = 0
    r: int32 = -1               # advanced at the TOP: see the II note
    for x in range(n_word):
        r += 1
        if r >= cnt:
            w0: UInt(64) = c_vru.get()
            op = w0[0:6]
            f0 = w0[6:18]
            f3 = w0[42:54]
            cnt = w0[54:62]
            r = 0
        if op == OP_MM:
            vv: UInt(VW) = vr[f0 + r]
            acol[0].put(vv)
        else:
            wa: int32 = f0 + r
            if op == OP_DMA_LD:
                wa = f3 + r
            wv: UInt(VW) = 0
            if op == OP_VLD:
                wv = sp2vr.get()
            else:
                wv = dma2vr.get()
            vr[wa] = wv
