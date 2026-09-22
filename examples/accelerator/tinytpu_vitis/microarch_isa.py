# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU-isa: the shipped instantiation of the `ip` unit library.

The hardware is no longer here. Each of the eight units is a module of its own
under `ip/units/`, declaring the channels, memories, parameters and ISA names
it needs and closing over nothing; `ip/tinytpu.py` wires them into an
architecture, and `ip/compose.py` emits the `@df.region()` that nests them --
which Allo requires, because it reaches a `@df.kernel` only as a nested
`ast.FunctionDef` inside its region.

This module is the one build the fork ships and measures: the parameter set,
read from the environment so a sweep can change it, and the module-level names
the harness (`bench_isa`, `stress_isa`, `cosim`, `isa_dsl`, `isa_ref`,
`kpn_model`, `mutate`) imports.

Design notes: `docs/source/designs/tinytpu_isa.rst` (the architecture and the
ISA), `tinytpu_library.rst` (the decomposition, and what the front end refuses),
`tinytpu_history.rst` (how the numbers were reached).
"""

import os

from examples.accelerator.tinytpu_vitis.ip.params import TpuParams
from examples.accelerator.tinytpu_vitis.ip.tinytpu import TinyTPU
from examples.accelerator.tinytpu_vitis.ip.assembler import (  # noqa: F401
    AR_RAW_DIST, ProgramError)
from examples.accelerator.tinytpu_vitis.ip.isa import (  # noqa: F401
    AGU_F0, AGU_F1, AGU_F2, AGU_F3, AGU_TERMS, DMA_SRC_B, DMA_TO_VR, IWORDS,
    LOOP_DEPTH, MAXROWS, NHDR, OP_DMA_LD, OP_DMA_ST, OP_ENDLOOP, OP_LOOP,
    OP_MM, OP_MVOUT, OP_NOP, OP_VADD, OP_VLD, OP_VRELU, enc, enc_agu)

# Instruction slots. Sized to the longest program shipped (the stress harness's
# random programs), not to a round number: the sequencer's prefetch is
# `IMEM_SIZE` words long whatever the program, so every unused slot is startup
# time. With control flow the program is O(nesting), not O(tiles).
_MAX_STATIC = 24               # longest program shipped, plus headroom

# T and MAXDIM are defined here, at module level, each exactly once and as an
# environment parameter: one RTL build runs every shape, and the parametricity
# gate rebuilds the design at other values of them.
T = int(os.environ.get("TPU_T", 4))
MAXDIM = int(os.environ.get("TPU_MAXDIM", 16))
SPAD_ROWS = int(os.environ.get("TPU_SPAD", 512))
NVR = int(os.environ.get("TPU_NVR", 256))
NAR = int(os.environ.get("TPU_NAR", 128))
QD = int(os.environ.get("TPU_QD", 8))
IMEM_SIZE = int(os.environ.get("TPU_IMEM", NHDR + IWORDS * _MAX_STATIC))

PARAMS = TpuParams(T=T, MAXDIM=MAXDIM, SPAD_ROWS=SPAD_ROWS, NVR=NVR, NAR=NAR,
                   QD=QD, IMEM_SIZE=IMEM_SIZE)
TPU = TinyTPU(PARAMS)

VW = PARAMS.VW
AW = PARAMS.AW
WPR = PARAMS.WPR

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
