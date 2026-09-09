# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU-ws: the weight-stationary (WS) TPU, in grid syntax.

`microarch.py` in this directory is output-stationary: PE(i, j) owns C[i, j] and
contracts over k in place, `acc += a * b`. That is the shape every tutorial
systolic example uses, and Vitis priced it exactly:

    WARNING: [HLS 200-880] Unable to enforce a carried dependence constraint
      (II = 1, distance = 1) between 'store' and 'load' on 'v121'   <- acc
    ... Final II = 7

Seven is the fp32 adder's latency. The contraction is a loop-carried dependence
of *distance 1* with the adder inside it, so II = latency, and this is not a
scheduler weakness -- three backends independently reported the same number
(`RESULTS.md`). Rotating P partial sums would paper over it.

The real fix is architectural, and it is what Gemmini and the TPU actually do:
**do not accumulate in place -- move the partial sum.**

    weights   stay:   PE(i, j) latches W[i, j] for the whole instruction
    A         flows:  west -> east,   a[m, i] enters row i
    psum      flows:  north -> south, PE(i, j) emits p_in + a * w
    C[m, j]   falls out of the bottom of column j

Now one PE iteration is: two stream reads, one multiply-add, two stream writes.
**Nothing is loop-carried.** The adder's 7 cycles become pipeline *depth*
instead of initiation interval, which is the whole point of a systolic array --
the array is deep so that it can be fast. Expected PE II is 1.

The contraction over k did not vanish, it moved outward: k is tiled by T, one
instruction per (kt, tn) weight tile, and the psums of successive kt are summed
in the drainer's accumulator buffer. That is Gemmini's split too -- its adds
live in `AccumulatorMem`'s write path, not in the mesh.

Two consequences worth noting, both good:

  * **The PEs no longer decode anything.** In a weight-stationary array the opcode
    only matters to whoever fetches operands and whoever retires results, so
    `cmd` fans out to the loader/wloader/drainer, not to T*T PEs. Gemmini is the
    same: its PEs are dumb, `ExecuteController` decodes.
  * **One instruction is a whole M x T x T panel**, not one T x T tile: a weight
    tile is loaded in T cycles and then all M rows of A stream through it. So
    weight load is amortized M/T-fold instead of paying T*T per T*T MACs, and
    the instruction count drops from Mt*Nt*Kt to Nt*Kt.

The ISA. One instruction, `mm`, with three flag bits packed into the opcode --
extracted by honest integer arithmetic since that is what the frontend supports:

    zero = op % 2          start a fresh accumulation for this output column
    out  = (op // 2) % 2   retire the accumulator to C
    relu = (op // 4) % 2   ... through ReLU on the way out

so a Kt-deep contraction is `zero` on the first instruction, `out` on the last,
and plain accumulate in between. Written by hand in `program()`: the `@I.expand`
backend that lowers a TOSA matmul to instructions lives only on `chia-codesign`.

Feed bandwidth is a separate, real problem and it is *not* fixed by the shape.
The array wants T A-values and T weights per cycle; one process reading one
2-port memory delivers two, hence the `[HLS 200-885] Final II = 2` on the
loader. `schedule()` below cyclic-partitions A, B and C by T so those T reads
land in T banks -- reachable because `df.build` is just `customize(func)` plus
`s.build(...)`, so the schedule primitives are available on the Vitis path.
"""

import os

import allo
from allo.ir.types import float32, int8, int16, int32, Stream
from allo.customize import Partition
from allo.ir.utils import MockBuffer
import allo.dataflow as df

# --- data type -------------------------------------------------------------
# Gemmini's *default* config is `inputType = SInt(8.W)`, `accType = SInt(32.W)`
# (`Configs.scala`), and every tapeout/lean config pins `dataflow = WS`. The
# fp32 configs are the off-default ones and they lean on FPU IP, which on an
# FPGA is soft-float: an fp32 add is ~7 cycles where an int32 add is 1. Holding
# the dtype at fp32 therefore measures FPGA soft-float latency, not the quality
# of the generated architecture, so int8/int32 is both the fair comparison and
# Gemmini's own default. `TPU_DTYPE=fp32` keeps the old configuration for
# a single-variable comparison.
DTYPE = os.environ.get("TPU_DTYPE", "int8")
if DTYPE == "int8":
    IN_T, PROD_T, ACC_T = int8, int16, int32
    OUT_T = int8            # Gemmini mvouts elem_t, so the result is clipped
    CLIP_LO, CLIP_HI = -128, 127
elif DTYPE == "fp32":
    IN_T, PROD_T, ACC_T = float32, float32, float32
    OUT_T = float32
    CLIP_LO, CLIP_HI = 0, 0  # unused
else:
    raise ValueError(f"TPU_DTYPE must be int8 or fp32, got {DTYPE!r}")

T = 4                       # the systolic array is T x T

M = int(os.environ.get("TPU_M", 8))
K = int(os.environ.get("TPU_K", 8))
N = int(os.environ.get("TPU_N", 8))
Kt, Nt = K // T, N // T

IWIDTH = 4                  # instruction record: [op, kt, tn, unused]
NI = Nt * Kt                # one weight tile per instruction; M rows stream
IMEM = NI * IWIDTH

QD = int(os.environ.get("TPU_QD", 16))   # queue depth == how far a stage may run ahead
# Per-class depths, so which channel actually needs the buffering can be
# measured rather than guessed. Default to QD.
QI = int(os.environ.get("TPU_QI", QD))   # sequencer -> units
QW = int(os.environ.get("TPU_QW", QD))   # weight shift-in
QA = int(os.environ.get("TPU_QA", QD))   # activations, west -> east
QP = int(os.environ.get("TPU_QP", QD))   # partial sums, north -> south
QC = int(os.environ.get("TPU_QC", QD))   # bottom of the array -> drainer


@df.region()
def tinytpu_ws(
    imem: int32[IMEM],
    A: IN_T[M, K],
    B: IN_T[K, N],
    C: OUT_T[M, N],
):
    # Every channel is point-to-point: one writer, one reader. That is what
    # `HLS 200-779` requires of a dataflow region.
    q_ld: Stream[int32, QI]             # sequencer -> loader
    q_wl: Stream[int32, QI]             # sequencer -> weight loader
    q_dr: Stream[int32, QI]             # sequencer -> drainer

    w_top: Stream[IN_T, QW][T]       # wloader -> top of column j
    w_fwd: Stream[IN_T, QW][T, T]    # PE(i, j) -> PE(i + 1, j), weights
    a_in: Stream[IN_T, QA][T]        # loader -> west edge of row i
    a_fwd: Stream[IN_T, QA][T, T]    # PE(i, j) -> PE(i, j + 1), activations
    p_fwd: Stream[ACC_T, QP][T, T]    # PE(i, j) -> PE(i + 1, j), partial sums
    c_out: Stream[ACC_T, QC][T]       # bottom of column j -> drainer

    @df.kernel(mapping=[1], args=[imem])
    def sequencer(l_imem: int32[IMEM]):
        """Fetch, decode, dispatch the fields each unit needs.

        Consumes nothing, so it cannot be in a dependence cycle with the
        operand streams -- which is how the earlier command-broadcast design
        deadlocked (`../tinytpu_grid/repro/`). It runs ahead by the queue depth,
        so no unit ever waits on decode."""
        for c in range(NI):
            op: int32 = l_imem[c * IWIDTH + 0]
            kt: int32 = l_imem[c * IWIDTH + 1]
            tn: int32 = l_imem[c * IWIDTH + 2]
            q_ld.put(kt)
            q_wl.put(kt)
            q_wl.put(tn)
            q_dr.put(op)
            q_dr.put(tn)

    @df.kernel(mapping=[1], args=[B])
    def wloader(lB: IN_T[K, N]):
        """Sole reader of B. Shifts one weight tile into the top of the array.

        Column j receives W[T-1, j] first and W[0, j] last, so that after each
        PE has forwarded the rows below it, the value it keeps is its own -- the
        standard weight shift-in. T cycles for a T x T tile, at T words per
        cycle, which is why B is cyclic-partitioned by T."""
        for c in range(NI):
            kt: int32 = q_wl.get()
            tn: int32 = q_wl.get()
            for st in range(T):
                with allo.meta_for(T) as j:
                    w_top[j].put(lB[kt * T + T - 1 - st, tn * T + j])

    @df.kernel(mapping=[1], args=[A])
    def loader(lA: IN_T[M, K]):
        """Sole reader of A. Streams all M rows through the latched weight tile.

        Row i of the array is fed a[m, kt*T + i]: the row index selects the
        *contraction* coordinate, because in a weight-stationary array it is k
        that is spatial and m that is temporal. One instruction is therefore M
        cycles of feed against T cycles of weight load."""
        for c in range(NI):
            kt: int32 = q_ld.get()
            for m in range(M):
                with allo.meta_for(T) as i:
                    a_in[i].put(lA[m, kt * T + i])

    @df.kernel(mapping=[T, T])
    def pe():
        """One processing element. It decodes nothing -- in a weight-stationary array
        the opcode only concerns the units that fetch and retire.

        The compute loop is the pipeline, and it has **no loop-carried value**:
        read west, read north, multiply-add, write east, write south. The
        adder's latency is depth, not II."""
        i, j = df.get_pid()
        for c in range(NI):
            # --- weight shift-in: forward the rows below me, then keep mine ---
            with allo.meta_for(T - 1 - i) as _s:
                with allo.meta_if(i == 0):
                    w_fwd[i, j].put(w_top[j].get())
                with allo.meta_else():
                    w_fwd[i, j].put(w_fwd[i - 1, j].get())
            w: IN_T = 0
            with allo.meta_if(i == 0):
                w = w_top[j].get()
            with allo.meta_else():
                w = w_fwd[i - 1, j].get()

            # --- compute: one wavefront per cycle, M wavefronts ---
            for m in range(M):
                a: IN_T = 0
                with allo.meta_if(j == 0):
                    a = a_in[i].get()
                with allo.meta_else():
                    a = a_fwd[i, j - 1].get()
                p: ACC_T = 0
                with allo.meta_if(i > 0):
                    p = p_fwd[i - 1, j].get()
                with allo.meta_else():
                    p = 0
                # Widen to the product type before multiplying, so the MAC is
                # int8 x int8 -> int16 rather than a 32x32 multiply. int8
                # operands bound the product at 127*127 = 16129, which fits.
                av: PROD_T = a
                wv: PROD_T = w
                o: ACC_T = p + av * wv
                # Emit south, then forward east. Swapping these two was tried
                # as a deadlock fix (see RESULTS_WS.md section 6) and measured
                # to change nothing -- the minimum workable queue depth was 32
                # either way -- so the order here is not load-bearing and the
                # deadlock is elsewhere. Kept in this order only because it
                # reads in dataflow order.
                with allo.meta_if(i != T - 1):
                    p_fwd[i, j].put(o)
                with allo.meta_else():
                    c_out[j].put(o)
                with allo.meta_if(j != T - 1):
                    a_fwd[i, j].put(a)

    @df.kernel(mapping=[1], args=[C])
    def drainer(lC: OUT_T[M, N]):
        """Sole writer of C, and the accumulator memory.

        The k-contraction that the PEs no longer do happens here, across
        instructions: `acc[m, j]` sums the psums of successive kt. Consecutive
        cycles touch different m, so there is no recurrence and the adder is
        again depth rather than II -- the same reason Gemmini can put its adds
        in the accumulator's write path."""
        acc: ACC_T[M, T] = 0
        for c in range(NI):
            op: int32 = q_dr.get()
            tn: int32 = q_dr.get()
            zero: int32 = op % 2
            out: int32 = (op // 2) % 2
            relu: int32 = (op // 4) % 2
            for m in range(M):
                with allo.meta_for(T) as j:
                    v: ACC_T = c_out[j].get()
                    base: ACC_T = 0
                    if zero == 0:
                        base = acc[m, j]
                    s: ACC_T = base + v
                    acc[m, j] = s
                    if out == 1:
                        r: ACC_T = s
                        if relu == 1:
                            r = max(s, 0)
                        # Gemmini's mvout writes `elem_t` with
                        # ACC_SCALE_IDENTITY and shift 0, i.e. a bare clip.
                        with allo.meta_if(DTYPE == "int8"):
                            if r > CLIP_HI:
                                r = CLIP_HI
                            if r < CLIP_LO:
                                r = CLIP_LO
                        lC[m, tn * T + j] = r


def program(relu=False):
    """The instruction stream, by hand: one weight tile per instruction.

    `zero` on the first kt of an output column, `out` (and optionally `relu`) on
    the last, plain accumulate in between."""
    prog = []
    for tn in range(Nt):
        for kt in range(Kt):
            op = 0
            if kt == 0:
                op += 1                      # zero
            if kt == Kt - 1:
                op += 2                      # out
                if relu:
                    op += 4                  # relu
            prog += [op, kt, tn, 0]
    return prog


def schedule(s):
    """Give the feeders the ports they need.

    The array consumes T activations and T weights per cycle; one process
    reading one 2-port memory supplies two, which is the loader's
    `[HLS 200-885] Final II = 2`. Cyclic partitioning by T puts those T reads
    in T banks: A and C on their last dimension (indexed by i and j), B on its
    last dimension (indexed by j)."""
    top = s.top_func_name
    s.partition(f"{top}:A", Partition.Cyclic, dim=2, factor=T)
    s.partition(f"{top}:B", Partition.Cyclic, dim=2, factor=T)
    s.partition(f"{top}:C", Partition.Cyclic, dim=2, factor=T)
    # And the drainer's accumulator, on the column index. Without this the
    # drainer retires one element per cycle where the array produces T, and
    # Vitis says so in as many words:
    #   [HLS 200-885] Unable to schedule 'store' ... on array 'acc' due to
    #   limited memory ports (II = 31). Please consider ... partitioning the
    #   array 'acc'.
    s.partition(MockBuffer("drainer_0", "acc"), Partition.Complete, dim=2)
    return s
