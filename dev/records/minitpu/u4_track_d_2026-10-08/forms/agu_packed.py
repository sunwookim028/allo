# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D form of ``agu_resolve.c1`` for Catapult: the eight ``iv_by_level``
lanes as one ``UInt(256)`` port per row (U3 track C C1: a ``[n, 8]`` lane array
is RAM pins). ``resolve`` and the runner (``agu_resolve.run_c1``) unchanged."""
import allo.dataflow as df
from allo.ir.types import UInt, uint1

from examples.minitpu.units import agu_resolve as U
from examples.minitpu.units.agu_resolve import U3, U4, U12, U32, resolve  # noqa: F401


def make(n, w=0, inst="base"):
    @df.region()
    def top(IV: UInt(256)[n], LIT: U12[n], AV: uint1[n], LVL: U3[n], SH: U4[n], OUT: U12[n]):
        @df.kernel(mapping=[1], args=[IV, LIT, AV, LVL, SH, OUT])
        def agu(iv: UInt(256)[n], lit: U12[n], av: uint1[n], lvl: U3[n], sh: U4[n], out: U12[n]):
            for t in range(n):
                ivw: UInt(256) = iv[t]
                row: U32[8]
                row[0] = ivw[0:32]
                row[1] = ivw[32:64]
                row[2] = ivw[64:96]
                row[3] = ivw[96:128]
                row[4] = ivw[128:160]
                row[5] = ivw[160:192]
                row[6] = ivw[192:224]
                row[7] = ivw[224:256]
                out[t] = resolve(row, lit[t], av[t], lvl[t], sh[t])

    return top


run = U.run_c1
