# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The weight half of a processing element, one per PE: walks the header and
weight chain and hands its PE a shadow weight register. Decodes no instruction.
See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from ..compose import unit


@unit(
    instances=("T", "T"),
    reads=("wcol", "wrow"),
    writes=("wcol", "wrow", "wq"),
    parameters=("T", "VW"),
)
def wld():
    i, j = df.get_pid()
    nmw: UInt(VW) = 0
    with allo.meta_if(j == 0):
        nmw = wcol[i].get()
        with allo.meta_if(i != T - 1):
            wcol[i + 1].put(nmw)
    with allo.meta_else():
        nmw = wrow[i, j - 1].get()
    with allo.meta_if(j != T - 1):
        wrow[i, j].put(nmw)
    nmm: int32 = nmw[0:16]
    tq: UInt(32) = 0
    tq[0:16] = nmw[16:32]       # the PE's own trip count: wavefront rows
    wq[i, j].put(tq)
    for c in range(nmm):
        hdr: UInt(VW) = 0
        with allo.meta_if(j == 0):
            hdr = wcol[i].get()
            with allo.meta_if(i != T - 1):
                wcol[i + 1].put(hdr)
        with allo.meta_else():
            hdr = wrow[i, j - 1].get()
        with allo.meta_if(j != T - 1):
            wrow[i, j].put(hdr)
        ww: UInt(VW) = 0
        with allo.meta_if(j == 0):
            ww = wcol[i].get()
            with allo.meta_for(T - 1 - i) as _f:
                wcol[i + 1].put(wcol[i].get())
        with allo.meta_else():
            ww = wrow[i, j - 1].get()
        with allo.meta_if(j != T - 1):
            wrow[i, j].put(ww)
        q: UInt(32) = 0
        q[0:8] = ww[8 * j : 8 * (j + 1)]
        q[8:20] = hdr[0:12]
        wq[i, j].put(q)
