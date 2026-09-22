# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch, decode, resolve addresses, dispatch: the unit with the PC and the
loop stack. See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from allo.customize import Partition

from ..compose import unit


def sequencer_directives(s, ctx):
    # 8 words per cycle into the program buffer, matching the 512-bit gmem0.
    s.partition(f"{ctx.instance('sequencer')}:ib", Partition.Cyclic, dim=1,
                factor=8)


@unit(
    memories=("imem",),
    writes=("c_dld", "c_spm", "c_vru", "c_acc", "c_dst"),
    parameters=("IMEM_SIZE", "T"),
    isa=("LOOP_DEPTH", "NHDR", "IWORDS", "AGU_TERMS", "AGU_F0", "AGU_F1", "AGU_F2", "AGU_F3",
         "OP_LOOP", "OP_ENDLOOP", "OP_DMA_LD", "OP_VLD", "OP_MM", "OP_VADD",
         "OP_VRELU", "OP_MVOUT", "DMA_TO_VR"),
    directives=sequencer_directives,
)
def sequencer(l_imem: UInt(64)[IMEM_SIZE]):
    # --- the program comes on-chip first, 8 words per cycle ---
    # `l_imem` is an `m_axi` port (the design is `wrap_io=False`). Fetched
    # in place, `l_imem[NHDR + pc * IWORDS]` is a data-dependent address
    # Vitis can only burst two words at a time -- `Final II = 13` on the
    # fetch loop. A constant-trip contiguous copy is one real AXI burst
    # instead, and every fetch after it is an on-chip read.
    #
    # gmem0 is 512 bits wide (`m_axi_max_widen_bitwidth 512` with
    # `align_value(64)`), so each iteration moves 8 words into an `ib`
    # cyclically partitioned by 8 (`schedule()`): the 56-word prefix is 7
    # iterations, not 56. One word per iteration cost 52 cycles at every
    # shape (the `v_imem8` variant, gap attribution).
    ib: UInt(64)[IMEM_SIZE]
    for i in range(IMEM_SIZE // 8):
        with allo.meta_for(8) as e8:
            ib[8 * i + e8] = l_imem[8 * i + e8]

    iw: UInt(64) = ib[0]
    n_instr: int32 = iw[0:16]

    c_dld.put(ib[1])
    c_dld.put(ib[7])
    c_spm.put(ib[2])
    c_spm.put(ib[4])
    c_vru.put(ib[3])
    c_acc.put(ib[5])
    c_dst.put(ib[6])

    lp_start: int32[LOOP_DEPTH] = 0
    lp_iv: int32[LOOP_DEPTH] = 0
    lp_trip: int32[LOOP_DEPTH] = 0
    iv_now: int32[LOOP_DEPTH] = 0
    sp: int32 = 0
    pc: int32 = 0
    running: int32 = 1

    while running == 1:
        w0: UInt(64) = ib[NHDR + pc * IWORDS]
        w1: UInt(64) = ib[NHDR + pc * IWORDS + 1]
        op: int32 = w0[0:6]
        nr: int32 = w0[54:62]

        if op == OP_LOOP:
            lp_start[sp] = pc + 1
            lp_iv[sp] = 0
            lp_trip[sp] = nr
            iv_now[sp] = 0
            sp += 1
            pc += 1
        elif op == OP_ENDLOOP:
            nxt: int32 = lp_iv[sp - 1] + 1
            if nxt < lp_trip[sp - 1]:
                lp_iv[sp - 1] = nxt
                iv_now[sp - 1] = nxt
                pc = lp_start[sp - 1]
            else:
                sp -= 1
                pc += 1
        else:
            # --- resolve the address terms against the live IVs ---
            f0: int32 = w0[6:18]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            with allo.meta_for(AGU_TERMS) as _t:
                tw: int32 = w1[19 * _t : 19 * _t + 4]
                lw: int32 = w1[19 * _t + 4 : 19 * _t + 7]
                sw: int32 = w1[19 * _t + 7 : 19 * _t + 19]
                d: int32 = iv_now[lw] * sw
                if tw == AGU_F0:
                    f0 = f0 + d
                if tw == AGU_F1:
                    f1 = f1 + d
                if tw == AGU_F2:
                    f2 = f2 + d
                if tw == AGU_F3:
                    f3 = f3 + d

            rw: UInt(64) = w0
            rw[6:18] = f0
            rw[18:30] = f1
            rw[30:42] = f2
            rw[42:54] = f3

            if op == OP_DMA_LD:
                c_dld.put(rw)
                if f0 >= DMA_TO_VR:
                    c_vru.put(rw)
                else:
                    c_spm.put(rw)
            if op == OP_VLD:
                c_spm.put(rw)
                c_vru.put(rw)
            if op == OP_MM:
                ws: UInt(64) = rw
                ws[54:62] = T + 1
                ws[18:30] = nr
                c_spm.put(ws)
                c_vru.put(rw)
                c_acc.put(rw)
            if op == OP_VADD:
                wv: UInt(64) = rw
                wv[54:62] = nr * 2
                c_acc.put(wv)
            if op == OP_VRELU:
                c_acc.put(rw)
            if op == OP_MVOUT:
                c_acc.put(rw)
                c_dst.put(rw)
            pc += 1

        if pc >= n_instr:
            running = 0
