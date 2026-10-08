# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D form of ``dma_desc_adapter.bits`` (plan A1) for Catapult: the two
column arrays ``IN: UInt(64)[n, 6]`` / ``OUT: UInt(64)[n, 9]`` (RAM pins, U3
track C C1) become one ``UInt(384)`` / ``UInt(576)`` port per row, column ``j``
at bits ``[64 j, 64 j + 64)``; the body unchanged. Runner
``dma_desc_adapter._run_bits`` unchanged."""
import allo.dataflow as df
from allo.ir.types import UInt, int32

from examples.minitpu.units import dma_desc_adapter as U


def make(n, w=0, inst="base"):
    G = U.GEOMETRY
    G.legality()
    SL = G.SUBLANE_SEL_W
    VA = G.VMEM_ADDR_W
    RW = G.DESC_WORD_ROWS_W
    BW = G.ROW_BITS

    @df.region()
    def top(IN: UInt(384)[n], OUT: UInt(576)[n]):
        @df.kernel(mapping=[1], args=[IN, OUT])
        def adapter(cin: UInt(384)[n], cout: UInt(576)[n]):
            issue: int32 = 0
            st_q: UInt(1) = 0
            ch_q: UInt(1) = 0
            va_q: UInt(VA) = 0
            rows_q: UInt(RW) = 0
            base_q: UInt(32) = 0
            stride_q: UInt(32) = 0
            for t in range(n):
                cw: UInt(384) = cin[t]
                rst: UInt(1) = cw[0:1]
                start: UInt(1) = cw[64:65]
                d: UInt(64) = cw[128:192]
                sb: UInt(32) = cw[192:224]
                ss: UInt(32) = cw[256:288]
                acc: UInt(1) = cw[320:321]
                if rst == 0:
                    issue = 0
                    st_q = 0
                    ch_q = 0
                    va_q = 0
                    rows_q = 0
                    base_q = 0
                    stride_q = 0
                vrow: UInt(BW) = va_q
                vrow = vrow << SL
                brow: UInt(BW) = rows_q
                brow = (brow << SL) | ((1 << SL) - 1)
                done: UInt(1) = 0
                if issue == 1:
                    done = acc
                ow: UInt(576) = 0
                ow[0:64] = issue
                ow[64:128] = st_q
                ow[128:192] = ch_q
                ow[192:256] = vrow
                ow[256:320] = brow
                ow[448:512] = stride_q
                ow[384:448] = base_q
                ow[512:576] = done
                cout[t] = ow
                if rst == 1:
                    if issue == 0:
                        if start == 1:
                            st_q = d[55]
                            ch_q = d[53]
                            va_q = d[41:53]
                            rows_q = d[29:41]
                            disp: UInt(32) = d[4:28]
                            if d[27] == 1:
                                disp[24:32] = 0xFF
                            if d[28] == 1:
                                base_q = sb + disp
                            else:
                                base_q = sb
                            stride_q = ss
                            issue = 1
                    elif acc == 1:
                        issue = 0

    return top


run = U._run_bits
