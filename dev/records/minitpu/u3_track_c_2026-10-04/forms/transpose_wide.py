# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track C form of ``xlu_transpose`` X1 ``trace``: the landed body with the
``uint16[n, 64]`` lane arrays packed into ``UInt(1024)[n]`` words (the RTL's
``write_data_i``/``read_data_o``), so the ports are Connections streams and not
RAM pins (as for ``tree_wide``). ``make(n, inst, unreset=False)``: the tile is
``@ Stateful`` (reset) as the landed X1; ``unreset=True`` declares
``Stateful(reset=False)`` (P-9) -- expected refused by the emitter on a kernel
with non-Wire ports (track A finding A6), recorded.
"""
import allo
import allo.dataflow as df
from allo.ir.types import Stateful, UInt, int32, uint8, uint16

from examples.minitpu.units import xlu_transpose as X

LANES, SUB, NF, W = X.LANES, X.SUB, X.NF, X.W
WW = UInt(W)


def make(n, inst="t16x4", unreset=False):
    @df.region()
    def top(RST: uint8[n], WV: uint8[n], WI: uint8[n], WD: WW[n],
            RV: uint8[n], RI: uint8[n], RVO: uint8[n], RDO: WW[n]):
        @df.kernel(mapping=[1], args=[RST, WV, WI, WD, RV, RI, RVO, RDO])
        def tx(rst: uint8[n], wv: uint8[n], wi: uint8[n], wd: WW[n],
               rv: uint8[n], ri: uint8[n], rvo: uint8[n], rdo: WW[n]):
            tile: uint16[LANES, LANES] @ Stateful
            for t in range(n):
                r: uint8 = rst[t]
                v: uint8 = rv[t]
                q: int32 = ri[t]
                e: uint8 = wv[t]
                wq: int32 = wi[t]
                w: WW = wd[t]
                o: WW = 0
                with allo.meta_for(SUB) as s:
                    with allo.meta_for(LANES) as l:
                        o[16 * (s * LANES + l) : 16 * (s * LANES + l) + 16] = tile[l, 4 * q + s]
                rdo[t] = o
                if r == 0:
                    rvo[t] = 0
                else:
                    rvo[t] = v
                if e == 1:
                    with allo.meta_for(SUB) as s:
                        with allo.meta_for(LANES) as l:
                            tile[4 * wq + s, l] = w[16 * (s * LANES + l) : 16 * (s * LANES + l) + 16]

    return top


def make_unreset(n, inst="t16x4"):
    @df.region()
    def top(RST: uint8[n], WV: uint8[n], WI: uint8[n], WD: WW[n],
            RV: uint8[n], RI: uint8[n], RVO: uint8[n], RDO: WW[n]):
        @df.kernel(mapping=[1], args=[RST, WV, WI, WD, RV, RI, RVO, RDO])
        def tx(rst: uint8[n], wv: uint8[n], wi: uint8[n], wd: WW[n],
               rv: uint8[n], ri: uint8[n], rvo: uint8[n], rdo: WW[n]):
            tile: uint16[LANES, LANES] @ Stateful(reset=False)
            for t in range(n):
                r: uint8 = rst[t]
                v: uint8 = rv[t]
                q: int32 = ri[t]
                e: uint8 = wv[t]
                wq: int32 = wi[t]
                w: WW = wd[t]
                o: WW = 0
                with allo.meta_for(SUB) as s:
                    with allo.meta_for(LANES) as l:
                        o[16 * (s * LANES + l) : 16 * (s * LANES + l) + 16] = tile[l, 4 * q + s]
                rdo[t] = o
                if r == 0:
                    rvo[t] = 0
                else:
                    rvo[t] = v
                if e == 1:
                    with allo.meta_for(SUB) as s:
                        with allo.meta_for(LANES) as l:
                            tile[4 * wq + s, l] = w[16 * (s * LANES + l) : 16 * (s * LANES + l) + 16]

    return top
