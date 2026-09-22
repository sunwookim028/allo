# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One weight-stationary processing element: tap a lane, take the partial sum
from the north, multiply-add, pass both on. Decodes no instruction, and the MAC
carries no loop-carried value. See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from ..compose import unit


@unit(
    instances=("T", "T"),
    reads=("wq", "acol", "a_fwd", "p_fwd", "cw"),
    writes=("acol", "a_fwd", "p_fwd", "cw"),
    parameters=("T", "VW", "AW"),
)
def pe():
    i, j = df.get_pid()
    tq: UInt(32) = wq[i, j].get()
    nt: int32 = tq[0:16]
    w: int8 = 0
    cnt: int32 = 0
    r: int32 = -1               # advanced at the TOP: see the II note
    for x in range(nt):
        r += 1
        if r >= cnt:
            q: UInt(32) = wq[i, j].get()
            w = q[0:8]
            cnt = q[8:20]
            r = 0
        a: int8 = 0
        with allo.meta_if(j == 0):
            aw: UInt(VW) = acol[i].get()
            with allo.meta_if(i != T - 1):
                acol[i + 1].put(aw)
            a = aw[8 * i : 8 * (i + 1)]
        with allo.meta_else():
            a = a_fwd[i, j - 1].get()
        p: int32 = 0
        with allo.meta_if(i > 0):
            p = p_fwd[i - 1, j].get()
        # int8 x int8 -> int16 keeps this a narrow multiply; the
        # operands bound the product at 128*128 = 16384.
        av: int16 = a
        wv: int16 = w
        o: int32 = p + av * wv
        with allo.meta_if(i != T - 1):
            p_fwd[i, j].put(o)
        with allo.meta_else():
            # The bottom row assembles the packed result word as it
            # travels east, so the accumulator sees whole words and
            # there is no T-way fan-in.
            cv: UInt(AW) = 0
            with allo.meta_if(j > 0):
                cv = cw[j - 1].get()
            cv[32 * j : 32 * (j + 1)] = o
            cw[j].put(cv)
        with allo.meta_if(j != T - 1):
            a_fwd[i, j].put(a)
