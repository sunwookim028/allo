# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The weight half of a processing element, one per PE.

Walks the header and weight chain -- down column 0, then east along the row --
and hands its own PE one `(weight lane, row count)` word per `mm` through a
depth-4 FIFO, which is the shadow weight register: this process latches `mm`
n+1's weight while the PE computes `mm` n. Decodes no instruction.
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
    counts_word: UInt(VW) = 0
    with allo.meta_if(j == 0):
        counts_word = wcol[i].get()
        with allo.meta_if(i != T - 1):
            wcol[i + 1].put(counts_word)
    with allo.meta_else():
        counts_word = wrow[i, j - 1].get()
    with allo.meta_if(j != T - 1):
        wrow[i, j].put(counts_word)
    n_mm: int32 = counts_word[0:16]
    trip_word: UInt(32) = 0
    trip_word[0:16] = counts_word[16:32]
    wq[i, j].put(trip_word)
    for mm in range(n_mm):
        header: UInt(VW) = 0
        with allo.meta_if(j == 0):
            header = wcol[i].get()
            with allo.meta_if(i != T - 1):
                wcol[i + 1].put(header)
        with allo.meta_else():
            header = wrow[i, j - 1].get()
        with allo.meta_if(j != T - 1):
            wrow[i, j].put(header)
        weight_word: UInt(VW) = 0
        with allo.meta_if(j == 0):
            weight_word = wcol[i].get()
            with allo.meta_for(T - 1 - i) as _forward:
                wcol[i + 1].put(wcol[i].get())
        with allo.meta_else():
            weight_word = wrow[i, j - 1].get()
        with allo.meta_if(j != T - 1):
            wrow[i, j].put(weight_word)
        pe_word: UInt(32) = 0
        pe_word[0:8] = weight_word[8 * j : 8 * (j + 1)]
        pe_word[8:20] = header[0:12]
        wq[i, j].put(pe_word)
