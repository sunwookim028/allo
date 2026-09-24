# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The instruction set as numpy: what a TinyTPU-isa program MEANS.

`bench_isa.py` can only check GEMM, because its gold is `A @ B`. Any other
program -- and so any unit field a GEMM never varies (a nonzero DRAM row, a
`vadd` whose destination is not its source) -- had no reference at all. This
is that reference: one function per opcode, over the same resolved dynamic
stream the sequencer issues.

    C_expected = run(prog, A, B, C_before)

**It is built on the spec, not on the design.** Every ISA fact it uses --
opcode numbers, which field carries which operand, the bit layout, the loop
stack, the AGU, the accumulator width, the output conversion -- comes from
`isa_encoding`, which `gen_isa.py` generates from `isa_spec.json`. It names
operands the way the spec names them (`i["spad_w"]`, never `f3`), so it cannot
inherit a bit position or a numeric width from `microarch_isa.py`. It used to
import all of those from the design and was written from the design's own
comments, which made it a restatement rather than an oracle: it agreed with
everything because it shared the design's assumptions.

What it still takes from the design is the program VALIDATOR, `check_program`.
That is a guard, not an oracle: it refuses a program the hardware would run to
a wrong answer, so this model never has to invent a value for an unwritten row.
`C_before` is carried through untouched outside what the program's `mvout`s
name, which is how a sentinel check falls out of comparing the whole of `C`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from examples.tinytpu.isa_encoding import (  # noqa: E402
    MAXDIM, NAR, NVR, SPAD_ROWS, T,
    OP_DMA_LD, OP_MM, OP_MVOUT, OP_VADD, OP_VLD, OP_VRELU,
    ACC_DTYPE, OPERAND_DTYPE,
    acc, dma_dest_is_vr, dma_source_is_b, expand, operands, to_operand,
)
from examples.tinytpu.microarch_isa import (  # noqa: E402
    check_program,
)


def run(prog, A, B, C):
    """Execute `prog` on flat operand-format `A`, `B`, `C` (MAXDIM*MAXDIM
    each). Returns the new `C`; the arguments are not modified."""
    check_program(prog)
    A = np.asarray(A, OPERAND_DTYPE).reshape(MAXDIM, MAXDIM)
    B = np.asarray(B, OPERAND_DTYPE).reshape(MAXDIM, MAXDIM)
    C = np.array(C, OPERAND_DTYPE).reshape(MAXDIM, MAXDIM)
    spad = np.zeros((SPAD_ROWS, T), np.int64)   # one row = T operand lanes
    vr = np.zeros((NVR, T), np.int64)
    ar = np.zeros((NAR, T), np.int64)           # one row = T accumulator lanes
    for op, nr, f0, f1, f2, f3 in expand(prog):
        i = operands(op, f0, f1, f2, f3, nr)
        for r in range(nr):
            if op == OP_DMA_LD:
                src = B if dma_source_is_b(i["mode"]) else A
                dst = vr if dma_dest_is_vr(i["mode"]) else spad
                blk = i["col_block"]
                dst[i["dst_row0"] + r] = src[i["dram_row0"] + r,
                                             blk * T:(blk + 1) * T]
            elif op == OP_VLD:
                vr[i["vr0"] + r] = spad[i["spad0"] + r]
            elif op == OP_MM:
                # PE(i, j) holds lane j of weight row i and taps lane i of the
                # activation word, so column j is sum_i act[i] * W[i][j].
                W = spad[i["spad_w"]:i["spad_w"] + T]
                # One `acc()` at the end, although the array accumulates
                # sequentially down the column: wraparound addition is a ring
                # homomorphism, so the order the spec fixes cannot be observed
                # in THIS configuration. A configuration whose accumulate
                # rounds would have to fold in the spec's order term by term.
                psum = vr[i["vr_a"] + r] @ W
                base = ar[i["ar0"] + r] if i["acc"] == 1 else 0
                ar[i["ar0"] + r] = acc(base + psum)
            elif op == OP_VADD:
                ar[i["ar_d"] + r] = acc(ar[i["ar_s1"] + r] + ar[i["ar_s2"] + r])
            elif op == OP_VRELU:
                ar[i["ar_d"] + r] = np.maximum(ar[i["ar_s"] + r], 0)
            elif op == OP_MVOUT:
                blk = i["col_block"]
                C[i["dram_row0"] + r, blk * T:(blk + 1) * T] = to_operand(
                    ar[i["ar0"] + r])
    return C.reshape(-1)


# The lanes above are int64 so that no intermediate wraps before `acc()` folds
# the value into the spec's accumulator format. That is only sound while the
# spec's accumulator fits in an int64.
assert np.iinfo(np.int64).min < np.iinfo(ACC_DTYPE).min and \
    np.iinfo(ACC_DTYPE).max < np.iinfo(np.int64).max, (
        "the spec's accumulator no longer fits inside this model's int64 lanes")
