# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D form of ``loop_ctrl.l1`` (plan L1) for Catapult.

The landed body unchanged in logic; its ports changed as U3 track C's forms
did: ``capture_data`` (4 x 32), ``iv_by_level`` (8 x 32) and ``replay_data``
(4 x 32) are one wide port per row instead of ``[n, k]`` lane arrays (C1: RAM
pins), and every port is read once, unconditionally, at the top of the
iteration (``ms_i``/``hi_i``/``bs_i``/``skip_i``/``step_i``/``capd`` were read
under conditions: C2). The loop buffer is ``UInt(128)[LB_CAP]`` (the same
bits as ``U32[LB_CAP, 4]``). Runner ``loop_ctrl.run_l1`` unchanged."""
import allo.dataflow as df
from allo.ir.types import UInt, int32, uint1

from examples.minitpu.units import loop_ctrl as U

U32 = UInt(32)
W128 = UInt(128)


def make(n, w=128, inst="shipped"):
    from examples.minitpu.template.control_geometry import SHIPPED as G

    SD, CAP = G.STACK_DEPTH, G.LB_CAP
    A = UInt(G.INSTR_ADDR_W)
    SPW = UInt(G.SP_W)
    CW = UInt(G.LB_COUNT_W)
    IW = UInt(G.LB_IDX_W)

    @df.region()
    def top(RST: uint1[n], BV: uint1[n], BS: A[n], LO: UInt(8)[n], HI: UInt(16)[n], STEP: UInt(8)[n],
            MS: uint1[n], SKIP: A[n], EV: uint1[n], ISS: uint1[n], CAPD: W128[n],
            IVL: UInt(256)[n], RD: W128[n], TAKEN: uint1[n], TARGET: A[n], IVT: U32[n], LVT: UInt(3)[n],
            LBR: uint1[n], CAPEN: uint1[n], REPEN: uint1[n], RIDX: IW[n], CAPOVF: uint1[n], DEPTH: SPW[n],
            OVF: uint1[n], UNF: uint1[n]):
        @df.kernel(mapping=[1], args=[RST, BV, BS, LO, HI, STEP, MS, SKIP, EV, ISS, CAPD,
                                      IVL, RD, TAKEN, TARGET, IVT, LVT, LBR, CAPEN, REPEN, RIDX, CAPOVF,
                                      DEPTH, OVF, UNF])
        def loop(rst: uint1[n], bv_i: uint1[n], bs_i: A[n], lo_i: UInt(8)[n], hi_i: UInt(16)[n],
                 step_i: UInt(8)[n], ms_i: uint1[n], skip_i: A[n], ev_i: uint1[n], iss_i: uint1[n],
                 capd: W128[n],
                 ivl_o: UInt(256)[n], rd_o: W128[n], taken_o: uint1[n], target_o: A[n], ivt_o: U32[n],
                 lvt_o: UInt(3)[n], lbr_o: uint1[n], capen_o: uint1[n], repen_o: uint1[n], ridx_o: IW[n],
                 capovf_o: uint1[n], depth_o: SPW[n], ovf_o: uint1[n], unf_o: uint1[n]):
            fr_bs: A[SD] = 0
            fr_iv: U32[SD] = 0
            fr_hi: UInt(16)[SD] = 0
            fr_step: UInt(8)[SD] = 0
            sp: SPW = 0
            warm: uint1 = 0
            cap_count: CW = 0
            body_len: CW = 0
            ridx: IW = 0
            invalid: uint1 = 0
            lbmem: W128[CAP] = 0  # sequencer_loop_buffer.mem (unreset LUTRAM)
            wr_ptr: CW = 0
            for t in range(n):
                # every port once, unconditionally (C2)
                r: uint1 = rst[t]
                bv: uint1 = bv_i[t]
                bs: A = bs_i[t]
                lo8: UInt(8) = lo_i[t]
                hi16: UInt(16) = hi_i[t]
                st8: UInt(8) = step_i[t]
                ms: uint1 = ms_i[t]
                sk: A = skip_i[t]
                endv: uint1 = ev_i[t]
                iss: uint1 = iss_i[t]
                cw: W128 = capd[t]
                if r == 0:
                    for k in range(SD):
                        fr_bs[k] = 0
                        fr_iv[k] = 0
                        fr_hi[k] = 0
                        fr_step[k] = 0
                    sp = 0
                    warm = 0
                    cap_count = 0
                    body_len = 0
                    ridx = 0
                    invalid = 0
                    wr_ptr = 0
                # ---- combinational: sequencer_loop_ctrl ----
                tos: SPW = 0
                if sp != 0:
                    tos = sp - 1
                it: int32 = tos
                step32: U32 = fr_step[it]
                iv_next: U32 = fr_iv[it] + step32
                hi32: U32 = fr_hi[it]
                live: uint1 = endv & (sp != 0)
                cont: uint1 = 0
                if live and iv_next < hi32:
                    cont = 1
                exit_: uint1 = live & (1 - cont)
                lo16: UInt(16) = lo8
                skip: uint1 = 0
                if bv and ms and hi16 <= lo16:
                    skip = 1
                push: uint1 = bv & (1 - skip)
                lb_reset: uint1 = push | exit_ | skip
                cap_en: uint1 = 0
                if sp != 0 and warm == 0 and invalid == 0:
                    cap_en = 1
                rep_en: uint1 = 0
                if sp != 0 and warm == 1:
                    rep_en = 1
                cap_ovf: uint1 = 0
                if cap_en and iss and wr_ptr == CAP:
                    cap_ovf = 1
                completes: uint1 = cap_en & endv & (1 - cap_ovf)
                cold_cont: uint1 = cont & (1 - warm)
                warm_exit: uint1 = exit_ & warm
                target: A = fr_bs[it] + body_len + 1
                if skip:
                    target = bs + sk
                elif cold_cont:
                    target = fr_bs[it]
                ivw: UInt(256) = 0
                ivw[0:32] = fr_iv[0]
                ivw[32:64] = fr_iv[1]
                ivw[64:96] = fr_iv[2]
                ivw[96:128] = fr_iv[3]
                ivw[128:160] = fr_iv[4]
                ivw[160:192] = fr_iv[5]
                ivw[192:224] = fr_iv[6]
                ivw[224:256] = fr_iv[7]
                ivl_o[t] = ivw
                ir: int32 = ridx
                word: W128 = 0
                if ir < CAP:  # an index at or above LB_CAP reads nothing defined
                    word = lbmem[ir]
                rd_o[t] = word
                taken_o[t] = cold_cont | warm_exit | skip
                target_o[t] = target
                ivt_o[t] = fr_iv[it]
                lvt_o[t] = tos
                lbr_o[t] = lb_reset
                capen_o[t] = cap_en
                repen_o[t] = rep_en
                ridx_o[t] = ridx
                capovf_o[t] = cap_ovf
                depth_o[t] = sp
                ovf_o[t] = push & (sp == SD)
                unf_o[t] = endv & (sp == 0)
                # ---- rising edge ----
                if r:
                    isp: int32 = sp
                    if push and sp != SD:
                        fr_bs[isp] = bs
                        fr_iv[isp] = lo8
                        fr_hi[isp] = hi16
                        fr_step[isp] = st8
                        sp = sp + 1
                    elif endv and sp != 0:
                        if cont:
                            fr_iv[it] = iv_next
                        else:
                            sp = sp - 1
                    if push:
                        warm = 0
                        cap_count = 0
                        ridx = 0
                        invalid = 0
                    elif exit_ or skip:
                        warm = 0
                        cap_count = 0
                        ridx = 0
                        invalid = 1
                    else:
                        old_count: CW = cap_count
                        if cap_en and iss:
                            cap_count = cap_count + 1
                        if completes:
                            body_len = old_count
                            warm = 1
                        if rep_en and iss:
                            if endv:
                                ridx = 0
                            else:
                                ridx = ridx + 1
                    if lb_reset:
                        wr_ptr = 0
                    elif cap_en and iss and wr_ptr < CAP:
                        iw: int32 = wr_ptr
                        lbmem[iw] = cw
                        wr_ptr = wr_ptr + 1

    return top


run = U.run_l1
