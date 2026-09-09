# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU-grid: an instruction-programmable TPU whose array is a *grid*.

This is the upstream-syntax counterpart to the `chia-codesign` branch's
``examples/accelerator/tinytpu/microarch_v2.py``, and the difference is the one
that matters architecturally.

On that branch there is no language-level spatial construct: a PE is a point in
a hand-unrolled ``for i: for j:`` nest with ``unroll`` applied, so a DIM x DIM
array exists only because the schedule flattens the nest into DIM*DIM copies of
one body. That works -- the emitted Verilog has DIM*DIM real multipliers -- but
it costs three things that show up in `FINDINGS_v2.md`: the compiler's
if-conversion pass segfaults on conditionals inside the unrolled body (so the
array's edges had to be re-expressed as extra register columns), the whole array
is one scheduling region (so its II is a single global number rather than a
per-PE property), and the array is a `func.call` from the dispatch loop, which
the compiler will not pipeline across -- measured at exactly zero overlap
between consecutive instructions.

Upstream has the construct: ``@df.kernel(mapping=[P0, P1])`` instantiates a
2-D grid of kernel *instances*, each reading its own coordinates from
``df.get_pid()`` and communicating with its neighbours over ``Stream``s. A PE is
then a process, not a loop iteration. `tests/dataflow/test_tiled_systolic.py`
is the plain-GEMM form of this; what this file adds is the thing the headline
claim needs, which that test does not have: **the grid runs a program.**

Shape of the machine:

    imem --(broadcast)--> [ P0 x P1 grid: feed edges, T x T PEs, drain edges ]
    A, B --------------->   west and north edges, one FIFO per PE link
    C   <----------------   each PE writes the output element it owns

Every instance runs the same body and loops over the instruction stream,
decoding each record and branching on the opcode, so one program steers the
whole array and adding an opcode does not change the array. That is what makes it an accelerator with an
ISA rather than a fixed-function GEMM block: ``mm`` and ``mm.relu`` differ only
in what the drain PEs do with the finished value.

Operands reach the array as *streams*, not through a shared scratchpad, which
is deliberate: a shared random-access scratchpad is exactly what the Vitis
dataflow checker rejects ("it can only have a single reader and a single
writer"), and it is also the wrong shape for a grid -- the west and north edges
want a row and a column per cycle, which is what a FIFO per edge gives.

**Status, and the ceiling this currently runs into.** Verified in the dataflow
simulator at one tile (4x4x4), both opcodes, `mm` and `mm.relu` correct. It
*hangs* from two tiles on, and `repro/` isolates why with single-variable runs:
not the shape, not the dtype, not the branch, but the fact that every grid
instance reads the shared `imem` argument. The alternative -- decoding once and
forwarding the opcode through the grid -- deadlocks for a different and
understood reason, also recorded there.

So on this branch the grid construct expresses the *array* well and the
*programmability* not yet, which is the exact complement of `chia-codesign`,
where the array had to be hand-unrolled but the ISA works. Reading `repro/` is
the fastest way to see what has to be fixed for a grid-based TPU to run a
multi-tile program.

Note also that on this branch the instruction stream is written by hand
(``program()`` below). The `@I.expand` compiler backend that lowers a TOSA
matmul into instructions lives only on `chia-codesign`, so there is no
auto-generated assembly here.
"""

import allo
from allo.ir.types import float32, int32, Stream
import allo.dataflow as df

import os

T = 4                       # the systolic array is T x T
P0, P1 = T + 2, T + 2       # + a feed row/column and a drain row/column

# Problem shape, overridable so the same design can be smoke-tested at one tile
# and measured at the benchmark shape. Multiples of T.
M = int(os.environ.get("TPU_M", 16))
K = int(os.environ.get("TPU_K", 16))
N = int(os.environ.get("TPU_N", 16))
Mt, Kt, Nt = M // T, K // T, N // T

IWIDTH = 4                  # instruction record: [opcode, m, n, unused]
NINSTR = Mt * Nt            # one `mm` per output tile; K contracts inside it
IMEM = NINSTR * IWIDTH

OP_MM = 0                   # C[m, n] = A[m, :] @ B[:, n]
OP_MM_RELU = 1              # ... through ReLU on the way out


@df.region()
def tinytpu_grid(
    imem: int32[IMEM],
    A: float32[M, K],
    B: float32[K, N],
    C: float32[M, N],
):
    # Operand edges: one FIFO per PE link. The array reads its west and north
    # neighbours, so these are the PE-to-PE channels, and the [0, *] / [*, 0]
    # entries are the feed edges.
    fifo_A: Stream[float32, 4][P0, P1]
    fifo_B: Stream[float32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[imem, A, B, C])
    def pe(l_imem: int32[IMEM], lA: float32[M, K], lB: float32[K, N],
           lC: float32[M, N]):
        """One processing element -- or, on the border, one feed/drain unit.

        The body is written once and instantiated P0*P1 times; ``meta_if`` on the
        coordinates is resolved at build time, so each instance keeps only the
        role its position gives it, so a PE is a process rather than a loop
        iteration.
        """
        i, j = df.get_pid()

        # Nested tile loops, matching tests/dataflow/test_tiled_systolic.py's
        # shape exactly, so that the only difference from that (working) design
        # is `op`: the activation comes from the program, the geometry does not.
        for tm in range(Mt):
            for tn in range(Nt):
                op: int32 = l_imem[(tm * Nt + tn) * IWIDTH]

                with allo.meta_if(i in {0, T + 1} and j in {0, T + 1}):
                    pass
                with allo.meta_elif(j == 0):
                    for k in range(K):
                        fifo_A[i, j + 1].put(lA[tm * T + i - 1, k])
                with allo.meta_elif(i == 0):
                    for k in range(K):
                        fifo_B[i + 1, j].put(lB[k, tn * T + j - 1])
                with allo.meta_elif(i == T + 1):
                    for k in range(K):
                        b_drain: float32 = fifo_B[i, j].get()
                with allo.meta_elif(j == T + 1):
                    for k in range(K):
                        a_drain: float32 = fifo_A[i, j].get()
                with allo.meta_else():
                    acc: float32 = 0.0
                    for k in range(K):
                        a: float32 = fifo_A[i, j].get()
                        b: float32 = fifo_B[i, j].get()
                        acc += a * b
                        fifo_A[i, j + 1].put(a)
                        fifo_B[i + 1, j].put(b)
                    out: float32 = acc
                    if op == OP_MM_RELU:
                        out = max(acc, 0.0)
                    lC[tm * T + i - 1, tn * T + j - 1] = out


def program(relu=False):
    """The instruction stream: one `mm` per output tile."""
    op = OP_MM_RELU if relu else OP_MM
    prog = []
    for m in range(Mt):
        for n in range(Nt):
            prog += [op, m, n, 0]
    return prog
