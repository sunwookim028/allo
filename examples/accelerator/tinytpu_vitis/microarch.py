# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU-vitis: an instruction-programmable systolic TPU that Vitis accepts.

`examples/accelerator/tinytpu_grid/` put the whole machine in one
`@df.kernel(mapping=[P0, P1])` grid, giving every PE instance the arrays it
needed as ordinary arguments. That is how upstream's
`tests/dataflow/test_tiled_systolic.py` is written, and for a fixed-function
GEMM it is fine. For an accelerator with an ISA it fails twice over, and
`../tinytpu_grid/BACKEND_CHOICE.md` has the evidence:

  * the dataflow simulator **hangs** from two tiles on, silently;
  * Vitis **rejects** it outright --
      `[HLS 200-779] Non-shared array 'v730' failed dataflow checking: it can
      only have a single reader and a single writer` (that is `imem`, and `A`
      and `B`), and
      `[HLS 200-979] Argument 'v731' failed dataflow checking: it can only be
      written in one process function` (that is `C`, written by 16 PEs).

Both are the same construct: one array fanned out to many processes. The fix is
not a pragma, it is the structure a real accelerator already has -- **an owning
process per memory**. Nothing here reads an array that another process also
touches:

    imem --> [sequencer] --cmd[p]---->  +-----------------------+
                                        |  T x T PE grid, each  |
    A,B  --> [loader]  --a_in[i]------> |  PE owning one C      |
                       --b_in[j]------> |  element and its own   |
                                        |  neighbour links      |
    C    <-- [drainer] <--c_out[p]----  +-----------------------+

Why this is also the *faster* shape, not just the legal one. Vitis emits a
dataflow region as **persistent concurrent processes**, and persistence is the
one property the chia RTLGen backend lacks: there a unit is a `func.call`, the
compiler will not pipeline a loop across one, and back-to-back instructions were
measured to overlap by exactly zero -- 712 of 1448 executor cycles at 16x16x16,
37% of the whole run, spent filling and draining a pipeline once per instruction
(`FINDINGS_v2.md` G.4). Here every process is a `while`-style server that never
restarts, so that cost is structural rather than per-instruction.

Three things make it deeply pipelined rather than merely concurrent:

1. **The PE inner loop is the pipeline.** One iteration is one MAC: two stream
   reads, a multiply-add into a local accumulator, two stream writes to the
   east and south neighbours. Nothing in it is loop-carried except the
   accumulator, so it holds II=1 and the array runs a wavefront per cycle.
2. **Nobody waits for a whole instruction.** The sequencer issues commands
   ahead into per-PE FIFOs, the loader streams operands ahead into the edge
   FIFOs, and the drainer collects behind -- so instruction *n+1*'s operands are
   already in flight while *n* is still draining. The queue depth is the
   run-ahead, and it is one constant (`QD`).
3. **No spill processes.** The last column does not forward east and the last
   row does not forward south, so the array has no border ring to schedule --
   `T*T` instances rather than `(T+2)*(T+2)`, and no process whose only job is
   to discard data.

The ISA is unchanged from the grid version: one `mm` per output tile, with the
opcode choosing whether the finished element passes through ReLU. Instruction
records are `IWIDTH` words and the stream is written by hand (`program()`), since
the `@I.expand` compiler backend that lowers a TOSA matmul into instructions
lives only on `chia-codesign`.
"""

import os

import allo
from allo.ir.types import float32, int32, Stream
import allo.dataflow as df

T = 4                       # the systolic array is T x T

M = int(os.environ.get("TPU_M", 8))
K = int(os.environ.get("TPU_K", 8))
N = int(os.environ.get("TPU_N", 8))
Mt, Nt = M // T, N // T

IWIDTH = 2                  # instruction record: [opcode, unused]
NI = Mt * Nt                # one `mm` per output tile
IMEM = NI * IWIDTH

QD = 8                      # queue depth == how far a stage may run ahead

OP_MM = 0                   # C[m, n] = A[m, :] @ B[:, n]
OP_MM_RELU = 1              # ... through ReLU on the way out


@df.region()
def tinytpu_vitis(
    imem: int32[IMEM],
    A: float32[M, K],
    B: float32[K, N],
    C: float32[M, N],
):
    # Every channel below is point-to-point: one writing process, one reading
    # process. That is what `HLS 200-779` requires, and it is why this design
    # synthesizes where the single-grid version does not.
    cmd: Stream[int32, QD][T * T]       # sequencer -> each PE
    a_in: Stream[float32, QD][T]        # loader -> the west edge of row i
    b_in: Stream[float32, QD][T]        # loader -> the north edge of column j
    a_fwd: Stream[float32, QD][T, T]    # PE(i, j) -> PE(i, j + 1)
    b_fwd: Stream[float32, QD][T, T]    # PE(i, j) -> PE(i + 1, j)
    c_out: Stream[float32, QD][T * T]   # PE -> drainer

    @df.kernel(mapping=[1], args=[imem])
    def sequencer(l_imem: int32[IMEM]):
        """Fetch and decode; issue one command to every PE and move on.

        The sequencer consumes nothing, so it cannot be in a dependence cycle
        with the operand streams -- which is exactly how the earlier
        command-broadcast design deadlocked (`../tinytpu_grid/repro/`). It runs
        ahead by the queue depth, so the array is never waiting on decode.

        The fan-out is `meta_for`, not `range`: a stream-array subscript names a
        *physical* FIFO, so it must be a compile-time constant. A runtime index
        fails in the frontend with "Fail to resolve the expression as symbolic
        expression" -- the right error, but it does not say that stream indices
        are the reason. `cmd` and `c_out` are also flat `[T*T]` rather than
        `[T, T]`, because a *nested* `meta_for` over a 2-D stream array hits the
        same error while a single `meta_for` over a 1-D one is fine. The PEs
        index them as `i * T + j` from their own `get_pid`, which is constant
        per instance."""
        for c in range(NI):
            op: int32 = l_imem[c * IWIDTH]
            with allo.meta_for(T * T) as p:
                cmd[p].put(op)

    @df.kernel(mapping=[1], args=[A, B])
    def loader(lA: float32[M, K], lB: float32[K, N]):
        """DRAM -> the array's west and north edges.

        Sole reader of `A` and `B`. One instruction streams the whole K
        contraction: row `m*T+i` of A into row i's west edge, column `n*T+j` of
        B into column j's north edge. The edge FIFOs are what a shared
        scratchpad would otherwise be, and they are the reason no scratchpad is
        needed: the array wants a row and a column per cycle, which is exactly
        what one FIFO per edge delivers."""
        for c in range(NI):
            tm: int32 = c // Nt
            tn: int32 = c % Nt
            for k in range(K):
                with allo.meta_for(T) as i:
                    a_in[i].put(lA[tm * T + i, k])
                with allo.meta_for(T) as j:
                    b_in[j].put(lB[k, tn * T + j])

    @df.kernel(mapping=[T, T])
    def pe():
        """One processing element. Weight- and activation-agnostic: it multiplies
        whatever arrives from the west by whatever arrives from the north.

        The body is the pipeline. One `k` iteration is one MAC -- read west, read
        north, multiply-add into `acc`, forward both on -- and the only
        loop-carried value is `acc`, so it holds II=1 and the array advances one
        wavefront per cycle.

        The last column does not forward east and the last row does not forward
        south, so there is no border ring to schedule and no process whose only
        job is to discard operands."""
        i, j = df.get_pid()
        for c in range(NI):
            op: int32 = cmd[i * T + j].get()
            acc: float32 = 0.0
            for k in range(K):
                # Declared before the meta branches, assigned inside: a name
                # bound inside a `meta_if` is not visible after it ("Unsupported
                # Name `a`").
                a: float32 = 0.0
                b: float32 = 0.0
                with allo.meta_if(j == 0):
                    a = a_in[i].get()          # the west edge
                with allo.meta_else():
                    a = a_fwd[i, j - 1].get()  # the PE to the west
                with allo.meta_if(i == 0):
                    b = b_in[j].get()          # the north edge
                with allo.meta_else():
                    b = b_fwd[i - 1, j].get()  # the PE to the north
                acc += a * b
                with allo.meta_if(j != T - 1):
                    a_fwd[i, j].put(a)
                with allo.meta_if(i != T - 1):
                    b_fwd[i, j].put(b)
            out: float32 = acc
            if op == OP_MM_RELU:
                out = max(acc, 0.0)
            c_out[i * T + j].put(out)

    @df.kernel(mapping=[1], args=[C])
    def drainer(lC: float32[M, N]):
        """The array -> DRAM. Sole writer of `C`.

        Collects one tile per instruction in a fixed order, so it needs no
        addressing state of its own beyond the tile counter."""
        for c in range(NI):
            tm: int32 = c // Nt
            tn: int32 = c % Nt
            with allo.meta_for(T) as i:
                with allo.meta_for(T) as j:
                    lC[tm * T + i, tn * T + j] = c_out[i * T + j].get()


def program(relu=False):
    """The instruction stream, written by hand: one `mm` per output tile."""
    op = OP_MM_RELU if relu else OP_MM
    prog = []
    for _ in range(Mt):
        for _ in range(Nt):
            prog += [op, 0]
    return prog
