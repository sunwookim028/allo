# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The instruction set as numpy: what a TinyTPU-isa program MEANS.

`bench_isa.py` can only check GEMM, because its gold is `A @ B`. Any other
program -- and so any unit field a GEMM never varies (a nonzero DRAM row, a
`vadd` whose destination is not its source) -- had no reference at all. This
is that reference: one function per opcode, over the same resolved dynamic
stream the sequencer issues (`microarch_isa.expand`), written from the ISA
comments and the unit docstrings, not from the unit bodies.

    C_expected = run(prog, A, B, C_before)

It refuses a program `check_program` rejects, so it never has to invent a
value for an unwritten row. `C_before` is carried through untouched outside
what the program's `mvout`s name, which is how a sentinel check falls out of
comparing the whole of `C`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    OP_DMA_LD, OP_VLD, OP_VMATLOAD, OP_VMATPUSH, OP_VMATPOP,
    OP_VADD, OP_VRELU, OP_VST, OP_VMEMST,
    DMA_SRC_B,
    MAXDIM, T, VMEM_ROWS, NVREG, check_program, expand,
)


def _wrap32(x):
    return ((np.asarray(x, np.int64) + (1 << 31)) % (1 << 32) - (1 << 31))


def _low8(x):
    """The low 8 bits of each int32 lane, as int8: what the array sees."""
    return (np.asarray(x, np.int64) + 128) % 256 - 128


def run(prog, A, B, C):
    """Execute `prog` on flat int8 `A`, `B`, `C` (MAXDIM*MAXDIM each).
    Returns the new `C`; the arguments are not modified."""
    check_program(prog)
    A = np.asarray(A, np.int8).reshape(MAXDIM, MAXDIM)
    B = np.asarray(B, np.int8).reshape(MAXDIM, MAXDIM)
    C = np.array(C, np.int8).reshape(MAXDIM, MAXDIM)
    vmem = np.zeros((VMEM_ROWS, T), np.int64)   # one row = T int8 lanes
    vr = np.zeros((NVREG, T), np.int64)         # ONE file: T int32 lanes a row
    W = None                                    # the weights in the array
    queue = []                                  # pushed, un-popped results
    for op, nr, f0, f1, f2, f3 in expand(prog):
        if op == OP_VMATLOAD:
            # PE(i, j) holds W[i][j] = lane j of weight row i; they take
            # effect from the next push (check_program: every load is
            # pushed before the next one), so Y = X W. The array takes the
            # low 8 bits of each lane.
            W = _low8(vr[f0:f0 + T])
            continue
        for r in range(nr):
            if op == OP_DMA_LD:
                src = B if f0 & DMA_SRC_B else A
                vmem[f3 + r] = src[f1 + r, f2 * T:(f2 + 1) * T]
            elif op == OP_VLD:
                vr[f0 + r] = vmem[f1 + r]           # int8 sign-extended
            elif op == OP_VMATPUSH:
                # column j is sum_i act[i] * W[i][j], T deep, from 0
                queue.append(_wrap32(_low8(vr[f0 + r]) @ W))
            elif op == OP_VMATPOP:
                vr[f0 + r] = queue.pop(0)
            elif op == OP_VADD:
                vr[f0 + r] = _wrap32(vr[f1 + r] + vr[f2 + r])
            elif op == OP_VRELU:
                vr[f0 + r] = np.maximum(vr[f1 + r], 0)
            elif op == OP_VST:
                vmem[f1 + r] = np.clip(vr[f0 + r], -128, 127)   # saturated
            elif op == OP_VMEMST:
                C[f1 + r, f2 * T:(f2 + 1) * T] = vmem[f3 + r]
    return C.reshape(-1)
