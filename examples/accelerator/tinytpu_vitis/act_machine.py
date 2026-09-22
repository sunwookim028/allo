# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU-isa as an `act.machine.Machine`: the units `assemble()` promises."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from act.machine import Machine, Opcode, Space  # noqa: E402
from act.schedule import Region  # noqa: E402
from act.schedule import Step  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    AGU_TERMS, DMA_SRC_B, DMA_TO_VR, LOOP_DEPTH, NAR, NVR, OP_DMA_LD,
    OP_ENDLOOP, OP_LOOP, OP_MM, OP_MVOUT, OP_VADD, OP_VLD, OP_VRELU,
    SPAD_ROWS, T, expand,
)

# The fields of `allo.encoding.Encoding` on branch `act-abstractions`
# (65ae98c6), by name, so the two sides state one budget rather than two.
ENCODING = {
    "name": "tinytpu-isa",
    "address_terms": AGU_TERMS,
    "loop_depth": LOOP_DEPTH,
    "has_predicated_fields": False,
    "requires_static_trip_counts": True,
    "requires_affine_addressing": True,
}

SPAD, VR, AR = "spad", "vr", "ar"

SEQUENCER_II = 5
ISSUE = ("sequencer", lambda **_: SEQUENCER_II)
ISSUE_COST = ("sequencer", SEQUENCER_II)


def dram(tensor, col_block):
    return f"dram.{tensor}:{col_block}"


DMA_LD = Opcode(
    code=OP_DMA_LD, name="dma_ld",
    loads=(ISSUE, ("dma_ld", lambda nr, **_: nr),
           ("spm", lambda nr, f0, **_: 0 if f0 & DMA_TO_VR else nr),
           ("vru", lambda nr, f0, **_: nr if f0 & DMA_TO_VR else 0)),
    reads=(lambda nr, f0, f1, f2, **_: Region(
        dram("B" if f0 & DMA_SRC_B else "A", f2), f1, nr),),
    writes=(lambda nr, f0, f3, **_: Region(
        VR if f0 & DMA_TO_VR else SPAD, f3, nr),))

VLD = Opcode(
    code=OP_VLD, name="vld",
    loads=(ISSUE, ("spm", lambda nr, **_: nr),
           ("vru", lambda nr, **_: nr)),
    reads=(lambda nr, f1, **_: Region(SPAD, f1, nr),),
    writes=(lambda nr, f0, **_: Region(VR, f0, nr),))

MM = Opcode(
    code=OP_MM, name="mm",
    loads=(ISSUE, ("spm", lambda **_: T + 1), ("vru", lambda nr, **_: nr),
           ("accu", lambda nr, **_: nr)),
    reads=(lambda nr, f0, **_: Region(VR, f0, nr),
           lambda f3, **_: Region(SPAD, f3, T),
           lambda nr, f1, f2, **_: Region(AR, f1, nr) if f2 else None),
    writes=(lambda nr, f1, **_: Region(AR, f1, nr),))

VADD = Opcode(
    code=OP_VADD, name="vadd",
    loads=(ISSUE, ("accu", lambda nr, **_: 2 * nr)),
    reads=(lambda nr, f1, **_: Region(AR, f1, nr),
           lambda nr, f2, **_: Region(AR, f2, nr)),
    writes=(lambda nr, f0, **_: Region(AR, f0, nr),))

VRELU = Opcode(
    code=OP_VRELU, name="vrelu",
    loads=(ISSUE, ("accu", lambda nr, **_: nr)),
    reads=(lambda nr, f1, **_: Region(AR, f1, nr),),
    writes=(lambda nr, f0, **_: Region(AR, f0, nr),))

MVOUT = Opcode(
    code=OP_MVOUT, name="mvout",
    loads=(ISSUE, ("accu", lambda nr, **_: nr),
           ("dma_st", lambda nr, **_: nr)),
    reads=(lambda nr, f0, **_: Region(AR, f0, nr),),
    writes=(lambda nr, f1, f2, **_: Region(dram("C", f2), f1, nr),))

MACHINE = Machine(
    name="tinytpu-isa",
    spaces=(Space(SPAD, SPAD_ROWS), Space(VR, NVR), Space(AR, NAR)),
    opcodes=(DMA_LD, VLD, MM, VADD, VRELU, MVOUT))


def is_control(prog):
    """One bool per sequencer fetch: True for `loop`/`endloop`, else False.

    The sequencer fetches these and the units never see them, so `expand` drops
    them -- but the loop stack is what holds the sequencer at `SEQUENCER_II`, so
    a cost model that does not charge them undercharges a deep nest. The walk
    mirrors `microarch_isa._trace`'s control flow and nothing else; the AGU
    resolution stays there, and the data fetches are paired with `expand`.
    """
    pc, stack = 0, []
    while pc < len(prog):
        op = prog[pc][0] & 0x3F
        if op == OP_LOOP:
            stack.append([pc + 1, 0, (prog[pc][0] >> 54) & 0xFF])
            yield True
            pc += 1
        elif op == OP_ENDLOOP:
            frame = stack[-1]
            frame[1] += 1
            yield True
            if frame[1] < frame[2]:
                pc = frame[0]
            else:
                stack.pop()
                pc += 1
        else:
            yield False
            pc += 1


def steps_of(prog):
    """Every sequencer fetch as a step, in issue order."""
    data = iter(expand(prog))
    out = []
    for index, control in enumerate(is_control(prog)):
        if control:
            out.append(Step(index=index, loads=(ISSUE_COST,)))
        else:
            out.append(MACHINE.step(index, next(data)))
    return tuple(out)
