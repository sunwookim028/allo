# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The harness glue, instance-aware: ``examples/tinytpu/microarch_isa.py``
with one switch.

``TPU_INSTANCE=template`` (the default here) builds TinyTPU-isa as an
instance of the template (``instances/tinytpu/instance.py``);
``TPU_INSTANCE=frozen`` builds the frozen ``ip/tinytpu.py`` design through
this same module, which is the control that the mount itself changes
nothing. Every other name -- the parameter set from the ``TPU_*``
environment, the memory map, the assembler, the programs -- is the frozen
glue's, so a gate that imports ``examples.tinytpu.microarch_isa`` sees the
same module-level names whichever build is under test.

This file is only ever reached through ``run_gates.py``'s mount; it is not
``examples/tinytpu/microarch_isa.py`` and does not replace it (README D-4).
"""

import os

from examples.tinytpu.ip.params import (  # noqa: F401
    BUS_BYTES, TEST_WINDOW, TpuParams)
from examples.tinytpu.ip.assembler import (  # noqa: F401
    AR_RAW_DIST, ProgramError)
from examples.tinytpu.ip.isa import (  # noqa: F401
    AGU_F0, AGU_F1, AGU_F2, AGU_F3, AGU_TERMS, DMA_SRC_B, DMA_TO_VR, IWORDS,
    LOOP_DEPTH, MAXROWS, NHDR, OP_DMA_LD, OP_DMA_ST, OP_ENDLOOP, OP_LOOP,
    OP_MM, OP_MVOUT, OP_NOP, OP_VADD, OP_VADDRELU, OP_VLD, OP_VRELU, enc,
    enc_agu)

INSTANCE = os.environ.get("TPU_INSTANCE", "template")
assert INSTANCE in ("template", "frozen"), f"TPU_INSTANCE={INSTANCE!r}"

_MAX_STATIC = 24

T = int(os.environ.get("TPU_T", 4))
MAXDIM = int(os.environ.get("TPU_MAXDIM", 64))
_WIDEN = TpuParams.widest_burst(T, MAXDIM) if os.environ.get("TPU_DMA_WIDEN") == "1" else 1
DMA_WORDS = int(os.environ.get("TPU_DMA_WORDS", _WIDEN))


def _override(name):
    value = os.environ.get(name)
    return int(value) if value else None


PARAMS = TpuParams(
    T=T,
    MAXDIM=MAXDIM,
    SPAD_ROWS=_override("TPU_SPAD"),
    NVR=_override("TPU_NVR"),
    NAR=_override("TPU_NAR"),
    QD=int(os.environ.get("TPU_QD", 16)),
    IMEM_SIZE=int(os.environ.get("TPU_IMEM", NHDR + IWORDS * _MAX_STATIC)),
    DMA_WORDS=DMA_WORDS,
)

if INSTANCE == "template":
    from examples.minitpu.template.instances.tinytpu.geometry import TinyTpuGeometry
    from examples.minitpu.template.instances.tinytpu.instance import TinyTpuInstance
    GEOMETRY = TinyTpuGeometry(
        T=T, MAXDIM=MAXDIM, QD=PARAMS.QD, IMEM_SIZE=PARAMS.IMEM_SIZE,
        DMA_WORDS=DMA_WORDS, spad_rows=_override("TPU_SPAD"),
        nvr=_override("TPU_NVR"), nar=_override("TPU_NAR"))
    TPU = TinyTpuInstance(GEOMETRY)
    assert TPU.params == PARAMS, (TPU.params, PARAMS)
else:
    from examples.tinytpu.ip.tinytpu import TinyTPU
    GEOMETRY = None
    TPU = TinyTPU(PARAMS)

VW = PARAMS.VW
AW = PARAMS.AW
WPR = PARAMS.WPR
QD = PARAMS.QD
IMEM_SIZE = PARAMS.IMEM_SIZE
OPERAND_ROWS = PARAMS.OPERAND_ROWS
SPAD_ROWS = PARAMS.SPAD_ROWS
NVR = PARAMS.NVR
NAR = PARAMS.NAR

A_VR = TPU.memory_map.A_VR
B_SP = TPU.memory_map.B_SP
AR_C = TPU.memory_map.AR_C
AR_P = TPU.memory_map.AR_P

tinytpu_isa = TPU.region
schedule = TPU.schedule

expand = TPU.assembler.expand
check_program = TPU.assembler.check
assemble = TPU.assembler.assemble

gemm_program_handwritten = TPU.programs.looped
gemm_program_flat = TPU.programs.flat
vadd_program = TPU.programs.vector
