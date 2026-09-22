# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch, decode, resolve addresses, dispatch: the unit with the program
counter and the loop stack.

Addresses resolve here and nowhere else, so every other unit receives a word
already in the format it decodes, and two units receive a rewritten copy whose
`nr` field is their own work count -- a row count that depends on the decoded
opcode inside a flat loop is a carried dependence on the counter and closes
that loop at II=2. Dispatch order is dataflow order, and it is load-bearing:
dispatching downstream-first deadlocks.
See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from allo.customize import Partition

from ..compose import unit


def sequencer_directives(s, ctx):
    # 8 words per cycle out of the program buffer, matching the 512-bit gmem0
    # that `align_value(64)` plus `m_axi_max_widen_bitwidth 512` gives the port.
    s.partition(f"{ctx.instance('sequencer')}:program", Partition.Cyclic,
                dim=1, factor=8)


@unit(
    memories=("imem",),
    writes=("c_dld", "c_spm", "c_vru", "c_acc", "c_dst"),
    parameters=("IMEM_SIZE", "T"),
    isa=("LOOP_DEPTH", "NHDR", "IWORDS", "AGU_TERMS", "AGU_F0", "AGU_F1",
         "AGU_F2", "AGU_F3", "OP_LOOP", "OP_ENDLOOP", "OP_DMA_LD", "OP_VLD",
         "OP_MM", "OP_VADD", "OP_VRELU", "OP_MVOUT", "DMA_TO_VR"),
    directives=sequencer_directives,
)
def sequencer(dram_imem: UInt(64)[IMEM_SIZE]):
    # The whole program comes on-chip in one constant-trip contiguous burst,
    # and every fetch after it is an on-chip read. Fetched in place,
    # `dram_imem[NHDR + pc * IWORDS]` is a data-dependent address that Vitis
    # can only burst two words at a time: II=13 on the fetch loop.
    program: UInt(64)[IMEM_SIZE]
    for group in range(IMEM_SIZE // 8):
        with allo.meta_for(8) as word_in_group:
            program[8 * group + word_in_group] = dram_imem[8 * group + word_in_group]

    header: UInt(64) = program[0]
    n_instr: int32 = header[0:16]

    c_dld.put(program[1])
    c_dld.put(program[7])
    c_spm.put(program[2])
    c_spm.put(program[4])
    c_vru.put(program[3])
    c_acc.put(program[5])
    c_dst.put(program[6])

    loop_body: int32[LOOP_DEPTH] = 0
    loop_iter: int32[LOOP_DEPTH] = 0
    loop_trip: int32[LOOP_DEPTH] = 0
    live_iv: int32[LOOP_DEPTH] = 0
    loop_sp: int32 = 0
    pc: int32 = 0
    running: int32 = 1

    while running == 1:
        control_word: UInt(64) = program[NHDR + pc * IWORDS]
        agu_word: UInt(64) = program[NHDR + pc * IWORDS + 1]
        op: int32 = control_word[0:6]
        nr: int32 = control_word[54:62]

        if op == OP_LOOP:
            loop_body[loop_sp] = pc + 1
            loop_iter[loop_sp] = 0
            loop_trip[loop_sp] = nr
            live_iv[loop_sp] = 0
            loop_sp += 1
            pc += 1
        elif op == OP_ENDLOOP:
            next_iter: int32 = loop_iter[loop_sp - 1] + 1
            if next_iter < loop_trip[loop_sp - 1]:
                loop_iter[loop_sp - 1] = next_iter
                live_iv[loop_sp - 1] = next_iter
                pc = loop_body[loop_sp - 1]
            else:
                loop_sp -= 1
                pc += 1
        else:
            f0: int32 = control_word[6:18]
            f1: int32 = control_word[18:30]
            f2: int32 = control_word[30:42]
            f3: int32 = control_word[42:54]
            with allo.meta_for(AGU_TERMS) as term:
                target: int32 = agu_word[19 * term : 19 * term + 4]
                level: int32 = agu_word[19 * term + 4 : 19 * term + 7]
                stride: int32 = agu_word[19 * term + 7 : 19 * term + 19]
                offset: int32 = live_iv[level] * stride
                if target == AGU_F0:
                    f0 = f0 + offset
                if target == AGU_F1:
                    f1 = f1 + offset
                if target == AGU_F2:
                    f2 = f2 + offset
                if target == AGU_F3:
                    f3 = f3 + offset

            resolved: UInt(64) = control_word
            resolved[6:18] = f0
            resolved[18:30] = f1
            resolved[30:42] = f2
            resolved[42:54] = f3

            if op == OP_DMA_LD:
                c_dld.put(resolved)
                if f0 >= DMA_TO_VR:
                    c_vru.put(resolved)
                else:
                    c_spm.put(resolved)
            if op == OP_VLD:
                c_spm.put(resolved)
                c_vru.put(resolved)
            if op == OP_MM:
                # `spm` pushes one header word and T weight rows per `mm`, and
                # the array's row count travels in f1.
                spm_copy: UInt(64) = resolved
                spm_copy[54:62] = T + 1
                spm_copy[18:30] = nr
                c_spm.put(spm_copy)
                c_vru.put(resolved)
                c_acc.put(resolved)
            if op == OP_VADD:
                # `accu` takes two steps per `vadd` row.
                accu_copy: UInt(64) = resolved
                accu_copy[54:62] = nr * 2
                c_acc.put(accu_copy)
            if op == OP_VRELU:
                c_acc.put(resolved)
            if op == OP_MVOUT:
                c_acc.put(resolved)
                c_dst.put(resolved)
            pc += 1

        if pc >= n_instr:
            running = 0
