# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D form of ``seq_decoder.c1`` for Catapult: the bundle and the two-lane
slot records as one wide port per row (U3 track C C1: a ``[n, k]`` lane array is
RAM pins). The per-slot functions and the runner (``seq_decoder.run_c1``)
unchanged; each output port is ``32 x lanes`` bits, as the lanes it replaces."""
import allo.dataflow as df
from allo.ir.types import UInt

from examples.minitpu.units import seq_decoder as U
from examples.minitpu.units.seq_decoder import (U32, decode_c, decode_d, decode_l, decode_m,  # noqa: F401
                                                decode_s, decode_v, decode_x)


def make(n, w=0, inst="base"):
    @df.region()
    def top(B: UInt(128)[n], V: UInt(64)[n], D: UInt(64)[n], L: UInt(64)[n], S: UInt(64)[n],
            M: U32[n], X: U32[n], C: U32[n], DL: U32[n]):
        @df.kernel(mapping=[1], args=[B, V, D, L, S, M, X, C, DL])
        def decoder(b: UInt(128)[n], v: UInt(64)[n], d: UInt(64)[n], lp: UInt(64)[n], s: UInt(64)[n],
                    m: U32[n], x: U32[n], c: U32[n], dl: U32[n]):
            for t in range(n):
                w: UInt(128) = b[t]
                fv: UInt(46) = decode_v(w)
                fd: UInt(57) = decode_d(w)
                fl: UInt(45) = decode_l(w)
                fs: UInt(36) = decode_s(w)
                v[t] = fv
                d[t] = fd
                lp[t] = fl
                s[t] = fs
                m[t] = decode_m(w)
                x[t] = decode_x(w)
                c[t] = decode_c(w)
                dl[t] = w[15:22]

    return top


run = U.run_c1
