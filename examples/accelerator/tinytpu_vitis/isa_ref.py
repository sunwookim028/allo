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
    OP_DMA_LD, OP_VLD, OP_MM, OP_VADD, OP_VRELU, OP_MVOUT,
    MAXDIM, T, SPAD_ROWS, NVR, NAR, check_program, expand,
)


def _wrap32(x):
    return ((np.asarray(x, np.int64) + (1 << 31)) % (1 << 32) - (1 << 31))


def run(prog, A, B, C):
    """Execute `prog` on flat int8 `A`, `B`, `C` (MAXDIM*MAXDIM each).
    Returns the new `C`; the arguments are not modified."""
    check_program(prog)
    A = np.asarray(A, np.int8).reshape(MAXDIM, MAXDIM)
    B = np.asarray(B, np.int8).reshape(MAXDIM, MAXDIM)
    C = np.array(C, np.int8).reshape(MAXDIM, MAXDIM)
    spad = np.zeros((SPAD_ROWS, T), np.int64)   # one row = T int8 lanes
    vr = np.zeros((NVR, T), np.int64)
    ar = np.zeros((NAR, T), np.int64)           # one row = T int32 lanes
    for op, nr, f0, f1, f2, f3 in expand(prog):
        for r in range(nr):
            if op == OP_DMA_LD:
                src = A if f0 == 0 else B
                spad[f3 + r] = src[f1 + r, f2 * T:(f2 + 1) * T]
            elif op == OP_VLD:
                vr[f0 + r] = spad[f1 + r]
            elif op == OP_MM:
                # PE(i, j) holds lane j of weight row i and taps lane i of the
                # activation word, so column j is sum_i act[i] * W[i][j].
                W = vr[f3:f3 + T]
                psum = vr[f0 + r] @ W
                base = ar[f1 + r] if f2 == 1 else 0
                ar[f1 + r] = _wrap32(base + psum)
            elif op == OP_VADD:
                ar[f0 + r] = _wrap32(ar[f1 + r] + ar[f2 + r])
            elif op == OP_VRELU:
                ar[f0 + r] = np.maximum(ar[f1 + r], 0)
            elif op == OP_MVOUT:
                C[f1 + r, f2 * T:(f2 + 1) * T] = np.clip(ar[f0 + r], -128, 127)
    return C.reshape(-1)
