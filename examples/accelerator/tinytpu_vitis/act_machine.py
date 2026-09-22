# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU-isa as an `act.machine.Machine`: the units `assemble()` promises."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from act.machine import Machine, Opcode, Space  # noqa: E402
from act.schedule import Region  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    DMA_SRC_B, DMA_TO_VR, NAR, NVR, OP_DMA_LD, OP_MM, OP_MVOUT, OP_VADD,
    OP_VLD, OP_VRELU, SPAD_ROWS, T,
)

SPAD, VR, AR = "spad", "vr", "ar"

SEQUENCER_II = 5
ISSUE = ("sequencer", lambda **_: SEQUENCER_II)


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
