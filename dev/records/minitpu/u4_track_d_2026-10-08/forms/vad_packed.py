# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D form of ``vpu_adapter.slots`` for Catapult: the V record as one
``UInt(64)`` port per row (U3 track C C1), and ``iss[t]`` read once into a local
(``slots`` reads it in three calls: three accesses of one port make it RAM pins,
U3 track C C2). ``adapt_v/x/m`` and the runner (``vpu_adapter.run_slots``)
unchanged."""
import allo.dataflow as df
from allo.ir.types import UInt, uint1

from examples.minitpu.units import vpu_adapter as U
from examples.minitpu.units.vpu_adapter import U32, adapt_m, adapt_v, adapt_x  # noqa: F401


def make(n, w=0, inst="base"):
    @df.region()
    def top(VI: UInt(64)[n], MI: UInt(8)[n], XI: U32[n], ISS: uint1[n], ROW: UInt(12)[n],
            VC: UInt(64)[n], XC: U32[n], MC: U32[n]):
        @df.kernel(mapping=[1], args=[VI, MI, XI, ISS, ROW, VC, XC, MC])
        def adapter(vi: UInt(64)[n], mi: UInt(8)[n], xi: U32[n], iss: uint1[n], row: UInt(12)[n],
                    vc: UInt(64)[n], xc: U32[n], mc: U32[n]):
            for t in range(n):
                vw: UInt(64) = vi[t]
                v: UInt(46) = vw[0:46]
                x: UInt(27) = xi[t]
                i1: uint1 = iss[t]
                r12: UInt(12) = row[t]
                m8: UInt(8) = mi[t]
                cv: UInt(47) = adapt_v(v, i1)
                cx: UInt(20) = adapt_x(x, i1, r12)
                cm: UInt(18) = adapt_m(m8, i1)
                vc[t] = cv
                xc[t] = cx
                mc[t] = cm

    return top


run = U.run_slots
