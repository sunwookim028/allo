# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D form of ``scalar_agu.s1`` (plan S1) for Catapult.

The landed body unchanged in logic; its ports changed the way U3 track C's
forms did: the kernel-arg CSR (``ka``, 4 x 32), ``iv_by_level`` (8 x 32) and the
sreg view (``sreg_o``, 4 x 32) are one wide port per row instead of ``[n, k]``
lane arrays (C1: RAM pins), and every port is read once, unconditionally, at
the top of the iteration (``siv``/``slv``/``simm``/``ka`` were read under ``if``
and ``ka`` at a data-dependent lane: C2). The ``S_LAT``-deep write pipe stays
data, so the I/O pin is 1 and a write is visible ``S_LAT`` rows later.
Runner: ``scalar_agu.run_s1`` unchanged."""
import allo.dataflow as df
from allo.ir.types import UInt, int32, uint1

from examples.minitpu.units import scalar_agu as U
from examples.minitpu.units.scalar_agu import LATS, U32, scalar_result  # noqa: F401


def make(n, w=0, inst="lat2"):
    SL = LATS[inst]

    @df.region()
    def top(RST: uint1[n], SV: uint1[n], SOP: UInt(3)[n], SRD: UInt(2)[n], SRS: UInt(2)[n], SIV: uint1[n],
            SLV: UInt(3)[n], SIMM: U32[n], KA: UInt(128)[n], IV: UInt(256)[n], SB: UInt(2)[n], SS: UInt(2)[n],
            LA: uint1[n], SLB: UInt(2)[n],
            SREG: UInt(128)[n], OB: U32[n], OS: U32[n], OL: U32[n], OW: U32[n]):
        @df.kernel(mapping=[1], args=[RST, SV, SOP, SRD, SRS, SIV, SLV, SIMM, KA, IV, SB, SS, LA, SLB,
                                      SREG, OB, OS, OL, OW])
        def agu(rst: uint1[n], sv: uint1[n], sop: UInt(3)[n], srd: UInt(2)[n], srs: UInt(2)[n],
                siv: uint1[n], slv: UInt(3)[n], simm: U32[n], ka: UInt(128)[n], iv: UInt(256)[n],
                sb: UInt(2)[n], ss: UInt(2)[n], la: uint1[n], slb: UInt(2)[n],
                sreg_o: UInt(128)[n], ob: U32[n], os: U32[n], ol: U32[n], ow: U32[n]):
            sreg: U32[4] = 0
            written: UInt(4) = 0
            pv: uint1[SL] = 0
            prd: UInt(2)[SL] = 0
            pdata: U32[SL] = 0
            pprod: U32[SL] = 0
            pmac: uint1[SL] = 0
            for t in range(n):
                # every port once, unconditionally (C2)
                live: uint1 = rst[t]
                x_sv: uint1 = sv[t]
                x_sop: UInt(3) = sop[t]
                x_srd: UInt(2) = srd[t]
                x_srs: UInt(2) = srs[t]
                x_siv: uint1 = siv[t]
                x_slv: UInt(3) = slv[t]
                x_simm: U32 = simm[t]
                kaw: UInt(128) = ka[t]
                ivw: UInt(256) = iv[t]
                x_sb: UInt(2) = sb[t]
                x_ss: UInt(2) = ss[t]
                x_la: uint1 = la[t]
                x_slb: UInt(2) = slb[t]
                kal: U32[4]
                kal[0] = kaw[0:32]
                kal[1] = kaw[32:64]
                kal[2] = kaw[64:96]
                kal[3] = kaw[96:128]
                ivl: U32[8]
                ivl[0] = ivw[0:32]
                ivl[1] = ivw[32:64]
                ivl[2] = ivw[64:96]
                ivl[3] = ivw[96:128]
                ivl[4] = ivw[128:160]
                ivl[5] = ivw[160:192]
                ivl[6] = ivw[192:224]
                ivl[7] = ivw[224:256]
                if live == 0:  # async reset: the row shows the reset state
                    written = 0
                    for k in range(4):
                        sreg[k] = 0
                    for k in range(SL):
                        pv[k] = 0
                        prd[k] = 0
                        pdata[k] = 0
                        pprod[k] = 0
                        pmac[k] = 0
                # ---- "pre" outputs: committed sreg only (no bypass) ----
                so: UInt(128) = 0
                so[0:32] = sreg[0]
                so[32:64] = sreg[1]
                so[64:96] = sreg[2]
                so[96:128] = sreg[3]
                sreg_o[t] = so
                ib: int32 = x_sb
                ob[t] = sreg[ib]
                i_s: int32 = x_ss
                os[t] = sreg[i_s]
                il: int32 = x_slb
                raw: U32 = sreg[il]
                if x_la:
                    raw = kal[il]
                bound: U32 = raw[0:16]
                if raw[16:32] != 0:
                    bound = 0xFFFF
                ol[t] = bound
                ow[t] = written
                # ---- rising edge ----
                if live:
                    irs: int32 = x_srs
                    ilv: int32 = x_slv
                    imm: UInt(24) = x_simm
                    operand: U32 = sreg[irs]
                    if x_siv:
                        operand = ivl[ilv]
                    prod: U32 = operand * imm
                    pre: U32 = scalar_result(x_sop, sreg[irs], imm, kal[irs])
                    for j in range(SL - 1):
                        k: int32 = SL - 1 - j
                        pv[k] = pv[k - 1]
                        prd[k] = prd[k - 1]
                        pdata[k] = pdata[k - 1]
                        pprod[k] = pprod[k - 1]
                        pmac[k] = pmac[k - 1]
                    pv[0] = x_sv
                    prd[0] = x_srd
                    pdata[0] = pre
                    pprod[0] = prod
                    pmac[0] = x_sop == 3
                    written = 0
                    if pv[SL - 1]:
                        fin: U32 = pdata[SL - 1]
                        if pmac[SL - 1]:
                            fin = fin + pprod[SL - 1]
                        iw: int32 = prd[SL - 1]
                        sreg[iw] = fin
                        written[iw] = 1

    return top


run = U.run_s1
