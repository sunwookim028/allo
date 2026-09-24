# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One weight-stationary processing element: tap a lane, take the partial sum
from the north, multiply-add, pass both on.

Decodes no instruction, and the MAC carries no loop-carried value, so the
multiplier and adder latencies are pipeline depth rather than initiation
interval. One flat loop over every wavefront row of every `mm`, so consecutive
`mm`s stream back to back. See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from allo.compose import unit


@unit(
    instances=("T", "T"),
    reads=("wq", "acol", "a_fwd", "p_fwd", "cw"),
    writes=("acol", "a_fwd", "p_fwd", "cw"),
    parameters=("T", "VW", "AW"),
)
def pe():
    i, j = df.get_pid()
    trip_word: UInt(32) = wq[i, j].get()
    n_wavefront_row: int32 = trip_word[0:16]
    weight: int8 = 0
    mm_rows: int32 = 0
    row: int32 = -1             # advanced at the top: hoisting this holds II=1
    for work in range(n_wavefront_row):
        row += 1
        if row >= mm_rows:
            pe_word: UInt(32) = wq[i, j].get()
            weight = pe_word[0:8]
            mm_rows = pe_word[8:20]
            row = 0
        activation: int8 = 0
        with allo.meta_if(j == 0):
            activation_word: UInt(VW) = acol[i].get()
            with allo.meta_if(i != T - 1):
                acol[i + 1].put(activation_word)
            activation = activation_word[8 * i : 8 * (i + 1)]
        with allo.meta_else():
            activation = a_fwd[i, j - 1].get()
        psum_north: int32 = 0
        with allo.meta_if(i > 0):
            psum_north = p_fwd[i - 1, j].get()
        # int8 x int8 -> int16 keeps this a narrow multiply; the operands bound
        # the product at 128*128 = 16384.
        activation16: int16 = activation
        weight16: int16 = weight
        psum: int32 = psum_north + activation16 * weight16
        with allo.meta_if(i != T - 1):
            p_fwd[i, j].put(psum)
        with allo.meta_else():
            # The bottom row assembles the packed result word as it travels
            # east, so the accumulator sees whole words and there is no T-way
            # fan-in.
            result_word: UInt(AW) = 0
            with allo.meta_if(j > 0):
                result_word = cw[j - 1].get()
            result_word[32 * j : 32 * (j + 1)] = psum
            cw[j].put(result_word)
        with allo.meta_if(j != T - 1):
            a_fwd[i, j].put(activation)
