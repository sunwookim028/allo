# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The accumulator vector registers, their sole owner, and the vector ALU.

`mm` deposits or accumulates one k-tile's partial sums, `vadd` and `vrelu` are
elementwise, and `mvout` clips to int8 on the way out. Every arm is muxed down
to ONE `ar` read and ONE `ar` write per iteration; `vadd` needs two reads, so
it takes two steps per row, which is why the sequencer sends this unit twice
its row count for a `vadd`. Its II=1 rests on a dependence claim the assembler
makes true. See ``docs/source/designs/tinytpu_isa.rst``."""

from __future__ import annotations

from examples.accelerator.tinytpu_vitis.ip.compose import unit


def accumulator_directives(s, ctx):
    # `#pragma HLS dependence variable=ar inter false` on the work loop. The
    # row index is a carried register, so Vitis cannot prove that step n's
    # store and step n+1's load touch different rows, and closes the loop at
    # II=3. The claim is true only for programs that meet the assembler's
    # accumulator distance contract, which `Assembler.check` enforces.
    s.dependence(
        f"{ctx.instance('accu')}:work", "ar", dep_type="inter",
        dependent=False,
        because=(
            f"check_program() rejects any program that reads an ar row within "
            f"AR_RAW_DIST={ctx.parameters['AR_RAW_DIST']} accu iterations of "
            f"writing it (THE ACCUMULATOR DISTANCE CONTRACT); assemble() "
            f"enforces it, the hardware does not, and only TPU_TB=stress cosim "
            f"can see a breach"),
    )


@unit(
    reads=("c_acc", "cw"),
    writes=("ac2sp",),
    parameters=("NAR", "AW", "VW", "T"),
    isa=("OP_MM", "OP_VADD", "OP_VRELU", "OP_MVOUT"),
    directives=accumulator_directives,
)
def accu():
    ar: UInt(AW)[NAR]
    count_word: UInt(64) = c_acc.get()
    n_step: int32 = count_word[0:16]
    op: int32 = 0
    f0: int32 = 0               # mm/vadd/vrelu: destination; mvout: source
    f1: int32 = 0               # mm: destination; vadd/vrelu: first source
    f2: int32 = 0               # mm: accumulate flag; vadd: second source
    instr_steps: int32 = 0
    step: int32 = -1            # advanced at the top: hoisting this holds II=1
    vadd_first: UInt(AW) = 0    # vadd's first operand, held for one step
    for work in range(n_step):
        step += 1
        if step >= instr_steps:
            word: UInt(64) = c_acc.get()
            op = word[0:6]
            f0 = word[6:18]
            f1 = word[18:30]
            f2 = word[30:42]
            instr_steps = word[54:62]
            step = 0
        row: int32 = step
        phase: int32 = 0
        if op == OP_VADD:
            row = step >> 1
            phase = step - (row << 1)
        read_row: int32 = f1 + row
        write_row: int32 = f0 + row
        if op == OP_MM:
            write_row = f1 + row
        if op == OP_MVOUT:
            read_row = f0 + row
        if op == OP_VADD:
            if phase == 1:
                read_row = f2 + row
        read_word: UInt(AW) = ar[read_row]
        write_word: UInt(AW) = 0
        do_write: int32 = 1
        if op == OP_MM:
            array_word: UInt(AW) = cw[T - 1].get()
            base: UInt(AW) = 0
            if f2 == 1:
                base = read_word
            with allo.meta_for(T) as lane:
                base_lane: int32 = base[32 * lane : 32 * (lane + 1)]
                array_lane: int32 = array_word[32 * lane : 32 * (lane + 1)]
                summed: int32 = base_lane + array_lane
                write_word[32 * lane : 32 * (lane + 1)] = summed
        elif op == OP_VADD:
            if phase == 0:
                vadd_first = read_word
                do_write = 0
            else:
                with allo.meta_for(T) as add_lane:
                    first: int32 = vadd_first[32 * add_lane : 32 * (add_lane + 1)]
                    second: int32 = read_word[32 * add_lane : 32 * (add_lane + 1)]
                    added: int32 = first + second
                    write_word[32 * add_lane : 32 * (add_lane + 1)] = added
        elif op == OP_VRELU:
            with allo.meta_for(T) as relu_lane:
                before: int32 = read_word[32 * relu_lane : 32 * (relu_lane + 1)]
                rectified: int32 = before
                if rectified < 0:
                    rectified = 0
                write_word[32 * relu_lane : 32 * (relu_lane + 1)] = rectified
        else:
            do_write = 0
            clipped_word: UInt(VW) = 0
            with allo.meta_for(T) as clip_lane:
                retiring: int32 = read_word[32 * clip_lane : 32 * (clip_lane + 1)]
                if retiring > 127:
                    retiring = 127
                if retiring < -128:
                    retiring = -128
                clipped: int8 = retiring
                clipped_word[8 * clip_lane : 8 * (clip_lane + 1)] = clipped
            ac2sp.put(clipped_word)
        if do_write == 1:
            ar[write_row] = write_word
