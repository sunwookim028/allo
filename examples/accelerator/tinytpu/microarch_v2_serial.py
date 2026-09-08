# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CornellTPU v2: a decoupled access-execute TPU with a real systolic array.

``microarch.py`` is a *sequential* machine: one fetch-decode-dispatch loop calls
one unit at a time, so a DMA burst and a matmul never overlap, and its "systolic
array" is an unrolled dot product -- four multipliers producing one output
column per cycle -- rather than an array of processing elements. Both are fixed
here.

**True overlap.** The top is a dataflow region of four concurrent processes:

    imem -> [sequencer] --ld_q--> [loader]  --> spad --> [executor] --> acc
                        --ex_q--> [executor]      ^                      |
                        --st_q--> [storer]   <----+---- spad <-----------+

  The sequencer fetches and decodes, then *routes* each command to the unit that
  owns it and moves on -- it never waits for a unit to finish. Ordering that
  actually matters is re-imposed by tokens: every instruction carries a ``dep``
  field naming which token it waits on and which it posts (see ``DEP_*``), the
  same job Gemmini's ROB dependency bits do. ``free`` is seeded with ``NBUF``
  credits, so the loader may run at most ``NBUF`` weight tiles ahead of the
  executor and the two overlap without racing on the scratchpad.

  Dispatch is strictly in program order. Draining one queue before another
  deadlocks: the queues are finite and the executor is what returns the loader's
  credits, so run-ahead has to come from queue depth and credits, never from
  reordering issue.

**A real systolic array.** ``mmu`` is a DIM x DIM weight-stationary array in the
shape of ``~/core/npu/src/core/mxu/mxu_systolic_array.sv``: weights sit in the
PEs, the activation flows west->east along a row, and the partial sum flows
north->south down a column, so the contraction index *is* the row index and each
column's bottom edge emits one finished output element. It is written
time-stepped with an explicit next-state commit because that is the RTL
semantics -- every PE reads its neighbour's register (last cycle's value) and
writes its own, so there is no combinational path across the array and no
accumulator recurrence inside a PE. Fully unrolling the two spatial loops
therefore has to materialise DIM*DIM multipliers and DIM*DIM adders, and the
emitted Verilog is where that is checked, not here.

Three consequences of the array being real, each of which shapes the ISA:

  * *Weights are state.* ``wst`` persists across instructions, so latching a
    weight tile is its own instruction (``loadw``) and one latch is followed by a
    long streaming ``mm`` -- the fill/drain skew amortises over the rows instead
    of being paid per 4x4 tile. This is Gemmini's ``preload``/``matmul`` split,
    and it is why a systolic array is worth having at all: on a single 4x4 tile
    the skew makes it *slower* than an unrolled dot product.
  * *The array needs DIM activations per cycle*, one per PE row, so the
    scratchpad is DIM independent banks -- ``sp0..sp3``, written out as separate
    arrays because that is the only way to get separate memories (see
    "Banking", below). PE row ``i`` only ever reads bank ``i``, which is what
    makes the banked access conflict-free.
  * *Accumulation is a separate walk.* The array writes its own output buffer
    ``zb*`` store-only; a short ``ar`` loop then adds that into the accumulator
    ``ac*``. Folding the accumulate into the array loop instead is a
    read-modify-write the dependence test cannot separate, and it costs a factor
    of four (see ``mmu``).

**Banking.** ``Complete`` partitioning does not mean "one memory per bank": the
compiler says so directly (``ALLO-E0006``: bound to storage 'bram' *and*
completely partitioned "scatters it into registers; the two cannot both hold"),
and the emitted RTL confirms it -- a ``partition(dim=2, Complete)`` on
``f32[512, 4]`` produced 2048 flip-flops behind 2048-way address decoders, a
98k-line netlist, and a combinational loop Verilator rejects outright. So every
bank here is its own array. It is verbose in the signatures, and it is what a
banked scratchpad is.

**The guard band.** Edge handling is what keeps the array at II=1. A wavefront
is one diagonal, and every PE on it works on the same output row, so an
out-of-range wavefront needs discarding in exactly one place. Doing that with a
*guard* is the trap: a guarded store is if-converted into a store to a selected
address, the address stops being affine, the dependence test falls back to "may
alias", and the array drops to II=4 -- measured. So the reads are left
unguarded and the writes land in pad rows the ISA reserves: ``GUARD`` rows below
every streamed operand and ``SKEW`` past its end. Off the ends the array reads
and writes live-but-dead values, which costs nothing, because the wavefronts
they belong to are discarded anyway.

The same rule catches three other things, all of them measured rather than
reasoned about: an index held in an ``i32`` temporary is not raised to affine
even when the arithmetic is (so subscripts are written out in full); a
conditional load is predicated and loses affinity (so the load is unconditional
and the *value* is selected); and a store whose row index contains the unrolled
column index pairs with the next iteration's load at an assumed distance of 1
(so the array output is de-skewed first).

The ISA keeps ``microarch.py``'s opcode values and adds the ones the new array
needs. Instruction records are ``IWIDTH`` words, ``[opcode, a0, a1, a2, dep]``.
"""

import argparse
from pathlib import Path

from allo.exp.dsa.core import ISA
from allo.lang.core import f32, i32, range as arange
from allo.lang import Stream

tpu2 = ISA("CornellTPU-v2-serial")

DIM = 4                  # the systolic array is DIM x DIM (and the bank count)
PHOP = 1                 # iterations a partial sum takes to cross one PE row
GUARD = (DIM - 1) * PHOP        # rows the array reads below a streamed operand
SKEW = (DIM - 1) * (1 + PHOP)   # fill/drain clocks, and rows read past the end
APAD = SKEW                     # pad rows around the array's output buffer

IWIDTH = 5               # words per instruction: [opcode, a0, a1, a2, dep]
DRAM_SIZE = 8192
SPAD_ROWS = 512          # rows per scratchpad bank
IMEM_SIZE = 8192         # -> IMEM_SIZE / IWIDTH instructions
MAXROWS = 16             # longest activation panel one `mm` may stream
AROWS = MAXROWS + 2 * APAD   # sized for the output buffer, whose pad rows
                             # absorb the array's unguarded fill/drain writes;
                             # the accumulator proper uses rows [0, MAXROWS)
VEC_LANES = 8
VEC_REGS = 8
VEC_ROWS = VEC_LANES // DIM   # scratchpad rows one vector spans

QD = 16                  # command / token queue depth
NBUF = 2                 # weight tiles the loader may run ahead by

# --- Opcodes. 0..9 keep microarch.py's encoding. ---------------------------- #
OP_VADD = 0
OP_VSUB = 1
OP_VMUL = 2
OP_VRELU = 3
OP_VLOAD = 4
OP_VSTORE = 5
OP_DMA_LOAD = 6
OP_DMA_STORE = 7
OP_VNEG = 9
OP_LOADW = 10            # latch a DIM x DIM weight tile into the PEs
OP_MM0 = 11              # stream a2 rows through the array, starting a new sum
OP_MM = 12               # ... accumulating onto the existing sum
OP_ACCST = 13            # accumulator -> scratchpad
OP_ACCRELU = 14          # ... with ReLU fused into the drain

# --- Dependency bits. The consumer interprets these; the sequencer only
#     routes. ----------------------------------------------------------------- #
DEP_WAIT_LOAD = 1        # executor: wait for a loader token
DEP_SIG_LOAD = 2         # loader: post a token
DEP_WAIT_FREE = 4        # loader: take a buffer credit
DEP_SIG_FREE = 8         # executor: return a buffer credit
DEP_WAIT_EXEC = 16       # storer: wait for an executor token
DEP_SIG_EXEC = 32        # executor: post a token


# --- The systolic array ----------------------------------------------------- #
@tpu2.unit
def mmu(
    sp0: f32[SPAD_ROWS],
    sp1: f32[SPAD_ROWS],
    sp2: f32[SPAD_ROWS],
    sp3: f32[SPAD_ROWS],
    wst: f32[DIM, DIM],
    ac0: f32[AROWS],
    ac1: f32[AROWS],
    ac2: f32[AROWS],
    ac3: f32[AROWS],
    x_row: i32,
    rows: i32,
    zero_first: i32,
):
    """Stream ``rows`` rows of X through the array against the stationary
    weights, then add the pass into the accumulator.

    ``a_reg`` is each PE's activation register and ``ps_dl`` its partial-sum
    pipeline, so one iteration of ``t`` is one clock of the whole array.
    ``ps_dl`` is ``PHOP`` deep, matching the reference design's ``mxu_pe.sv``
    ("partial sums advance one physical row every four cycles") because the hop
    passes through an FP adder. ``PHOP = 1`` still reaches II=1 here, since the
    split output buffer leaves no memory recurrence for the adder latency to
    close; the parameter is kept because a slower adder or a larger DIM would
    need it. The activation hop east is a plain register copy, always one clock.

    Skew: PE[i][j] handles output row ``m = t - j - i*PHOP``, so row ``i``'s
    west edge is fed ``X[t - i*PHOP][i]`` and column ``j``'s bottom edge emits
    ``C[m][j]`` at ``t = m + j + (DIM-1)*PHOP``; the de-skew registers then hand
    the whole row to the output buffer at ``t = m + SKEW``. Hence ``rows + SKEW``
    clocks."""
    # The array's registers carry one extra edge each, so a PE's inputs are
    # plain indexed reads with no conditional in the body:
    #   a_reg[i, 0]      -- the west-edge feed for row i (column j reads j)
    #   ps_dl[0, j, :]   -- the north-edge zero for column j (row i reads i)
    # Besides being what the hardware is (the edges are registers like any
    # other), this is load-bearing: a `j == 0` / `i > 0` select inside the PE
    # body makes the body a hyperblock, and if-conversion over the unrolled
    # DIM*DIM copies of it segfaults the compiler outright.
    a_reg: f32[DIM, DIM + 1]
    a_nxt: f32[DIM, DIM + 1]
    ps_dl: f32[DIM + 1, DIM, PHOP]
    ps_new: f32[DIM + 1, DIM]
    odl: f32[DIM, DIM]      # output de-skew: column j delayed by DIM-1-j
    zb0: f32[AROWS]         # the array's own output buffer, one bank per column
    zb1: f32[AROWS]
    zb2: f32[AROWS]
    zb3: f32[AROWS]

    for zi in arange(DIM + 1, name="zi"):
        for zj in arange(DIM, name="zj"):
            a_reg[zi % DIM, zj] = 0.0
            odl[zi % DIM, zj] = 0.0
            ps_new[zi, zj] = 0.0
            for zd in arange(PHOP, name="zd"):
                ps_dl[zi, zj, zd] = 0.0
    for za in arange(DIM, name="za"):
        a_reg[za, DIM] = 0.0

    nstep: i32 = rows + SKEW
    for t in arange(nstep, name="t"):
        # --- west edge: PE row i reads scratchpad bank i, staggered by the
        #     skew. Hoisted into its own registers rather than selected inside
        #     the PE loop: a bank select nested under the `j == 0` test is a
        #     nested hyperblock the if-conversion pass crashes on (SIGSEGV in
        #     [PREP], reproducible). These registers are what the hardware has
        #     at the west edge anyway. Unguarded and with the subscript written
        #     out, so each stays affine in t.
        a_reg[0, 0] = sp0[x_row + t]
        a_reg[1, 0] = sp1[x_row + t - PHOP]
        a_reg[2, 0] = sp2[x_row + t - 2 * PHOP]
        a_reg[3, 0] = sp3[x_row + t - 3 * PHOP]

        # --- one clock of the array: DIM*DIM PEs, one multiply-add each ---
        for i in arange(DIM, name="pi"):
            for j in arange(DIM, name="pj"):
                av: f32 = a_reg[i, j]               # west neighbour, or the feed
                pv: f32 = ps_dl[i, j, PHOP - 1]     # north neighbour, or zero
                a_nxt[i, j + 1] = av
                ps_new[i + 1, j] = pv + av * wst[i, j]

        # --- output de-skew ---
        # Column j finishes output row m at t = m + j + (DIM-1)*PHOP, so the DIM
        # columns of one row leave the array on DIM consecutive cycles; `odl`
        # delays column j by DIM-1-j to line them back up. Any systolic design
        # has this register file at its output edge, and here it also keeps the
        # store address free of the column index -- written straight out, the
        # row would be `t - oj - c`, and iteration t column oj would name the
        # same row as iteration t+1 column oj+1. Different banks, but the
        # dependence test pairs them anyway: II=4, measured.
        for sj in arange(DIM, name="sj"):
            for sd in arange(DIM - 1, name="sd"):
                odl[sj, DIM - 1 - sd] = odl[sj, DIM - 2 - sd]
            odl[sj, 0] = ps_new[DIM, sj]

        zb0[APAD + t - SKEW] = odl[0, DIM - 1]
        zb1[APAD + t - SKEW] = odl[1, DIM - 2]
        zb2[APAD + t - SKEW] = odl[2, DIM - 3]
        zb3[APAD + t - SKEW] = odl[3, 0]

        # --- commit: every PE register advances one slot ---
        for ci in arange(DIM, name="ci"):
            for cj in arange(DIM, name="cj"):
                a_reg[ci, cj + 1] = a_nxt[ci, cj + 1]
                for cd in arange(PHOP - 1, name="cd"):
                    ps_dl[ci + 1, cj, PHOP - 1 - cd] = ps_dl[ci + 1, cj,
                                                             PHOP - 2 - cd]
                ps_dl[ci + 1, cj, 0] = ps_new[ci + 1, cj]

    # --- accumulate the pass into the accumulator ---
    # A separate walk, and that is structural rather than a workaround: the
    # array writes `zb*` store-only, so its inner loop carries no memory
    # recurrence at all. Reading and writing the accumulator *inside* the array
    # loop is a read-modify-write whose two accesses the dependence test cannot
    # separate -- it pairs the store with the next iteration's load at an
    # assumed distance of 1 and pins the array at II=4, unaffected by pipeline
    # depth, by de-skewing, or by writing the subscripts inline (all measured).
    # As its own II=1 loop a pass costs `rows + SKEW + rows` instead of
    # `4 * (rows + SKEW)`. Any real design has this buffer at the output edge
    # for the same reason.
    for ar in arange(rows, name="ar"):
        b0: f32 = 0.0
        b1: f32 = 0.0
        b2: f32 = 0.0
        b3: f32 = 0.0
        if zero_first == 0:
            b0 = ac0[ar]
            b1 = ac1[ar]
            b2 = ac2[ar]
            b3 = ac3[ar]
        ac0[ar] = b0 + zb0[APAD + ar]
        ac1[ar] = b1 + zb1[APAD + ar]
        ac2[ar] = b2 + zb2[APAD + ar]
        ac3[ar] = b3 + zb3[APAD + ar]


@tpu2.unit
def loadw(
    sp0: f32[SPAD_ROWS],
    sp1: f32[SPAD_ROWS],
    sp2: f32[SPAD_ROWS],
    sp3: f32[SPAD_ROWS],
    wst: f32[DIM, DIM],
    w_row: i32,
):
    """Latch a DIM x DIM weight tile into the stationary PE registers. The tile
    is stored as rows W[n][*] and consumed transposed, so PE[i][j] holds W[j][i];
    reading one W row per cycle puts the DIM accesses in DIM distinct banks."""
    for n in arange(DIM, name="wn"):
        wst[0, n] = sp0[w_row + n]
        wst[1, n] = sp1[w_row + n]
        wst[2, n] = sp2[w_row + n]
        wst[3, n] = sp3[w_row + n]


@tpu2.unit
def accst(
    ac0: f32[AROWS],
    ac1: f32[AROWS],
    ac2: f32[AROWS],
    ac3: f32[AROWS],
    sp0: f32[SPAD_ROWS],
    sp1: f32[SPAD_ROWS],
    sp2: f32[SPAD_ROWS],
    sp3: f32[SPAD_ROWS],
    sp_row: i32,
    rows: i32,
    do_relu: i32,
):
    """Drain ``rows`` accumulator rows to the scratchpad, optionally through
    ReLU -- the activation an MLP layer needs, fused into the drain."""
    for am in arange(rows, name="am"):
        v0: f32 = ac0[am]
        v1: f32 = ac1[am]
        v2: f32 = ac2[am]
        v3: f32 = ac3[am]
        if do_relu == 1:
            v0 = max(v0, 0.0)
            v1 = max(v1, 0.0)
            v2 = max(v2, 0.0)
            v3 = max(v3, 0.0)
        sp0[sp_row + am] = v0
        sp1[sp_row + am] = v1
        sp2[sp_row + am] = v2
        sp3[sp_row + am] = v3


# --- Vector unit and its register-file movers ------------------------------- #
@tpu2.unit
def vpu(opcode: i32, vreg: f32[VEC_REGS, VEC_LANES], a: i32, b: i32, d: i32):
    """``vreg[d] = op(vreg[a], vreg[b])`` lanewise; unary ops ignore ``b``."""
    for i in arange(VEC_LANES, name="lane"):
        x: f32 = vreg[a, i]
        y: f32 = vreg[b, i]
        if opcode == OP_VADD:
            vreg[d, i] = x + y
        elif opcode == OP_VSUB:
            vreg[d, i] = x - y
        elif opcode == OP_VMUL:
            vreg[d, i] = x * y
        elif opcode == OP_VNEG:
            vreg[d, i] = -x
        else:  # OP_VRELU
            vreg[d, i] = max(x, 0.0)


@tpu2.unit
def vload(
    sp0: f32[SPAD_ROWS],
    sp1: f32[SPAD_ROWS],
    sp2: f32[SPAD_ROWS],
    sp3: f32[SPAD_ROWS],
    vreg: f32[VEC_REGS, VEC_LANES],
    sp_row: i32,
    slot: i32,
):
    """One VEC_LANES vector spans VEC_ROWS scratchpad rows, DIM banks each."""
    for r in arange(VEC_ROWS, name="vr"):
        vreg[slot, r * DIM + 0] = sp0[sp_row + r]
        vreg[slot, r * DIM + 1] = sp1[sp_row + r]
        vreg[slot, r * DIM + 2] = sp2[sp_row + r]
        vreg[slot, r * DIM + 3] = sp3[sp_row + r]


@tpu2.unit
def vstore(
    vreg: f32[VEC_REGS, VEC_LANES],
    sp0: f32[SPAD_ROWS],
    sp1: f32[SPAD_ROWS],
    sp2: f32[SPAD_ROWS],
    sp3: f32[SPAD_ROWS],
    slot: i32,
    sp_row: i32,
):
    for r in arange(VEC_ROWS, name="sr"):
        sp0[sp_row + r] = vreg[slot, r * DIM + 0]
        sp1[sp_row + r] = vreg[slot, r * DIM + 1]
        sp2[sp_row + r] = vreg[slot, r * DIM + 2]
        sp3[sp_row + r] = vreg[slot, r * DIM + 3]


# --- Sequential top: one fetch-decode-dispatch loop ------------------------ #
# The measurement control for the pipelined design. Same units, same schedule,
# same instruction stream; the only difference is that this top calls one unit
# at a time and waits, instead of routing commands to four processes that run
# concurrently. It is `microarch.py`'s shape with `microarch_v2.py`'s datapath,
# so a cycle difference between the two isolates the decoupling from the array.
@tpu2.entry
def tinytpu2s(
    dmem: f32[DRAM_SIZE],
    imem: i32[IMEM_SIZE],
    n_instr: i32,
    n_ld: i32,
    n_ex: i32,
    n_st: i32,
):
    sp0: f32[SPAD_ROWS]
    sp1: f32[SPAD_ROWS]
    sp2: f32[SPAD_ROWS]
    sp3: f32[SPAD_ROWS]
    wst: f32[DIM, DIM]
    ac0: f32[AROWS]
    ac1: f32[AROWS]
    ac2: f32[AROWS]
    ac3: f32[AROWS]
    vreg: f32[VEC_REGS, VEC_LANES]
    for pc in arange(n_instr, name="pc"):
        base: i32 = pc * IWIDTH
        op: i32 = imem[base]
        a0: i32 = imem[base + 1]
        a1: i32 = imem[base + 2]
        a2: i32 = imem[base + 3]
        zf: i32 = 0
        if op == OP_MM0:
            zf = 1
        rl: i32 = 0
        if op == OP_ACCRELU:
            rl = 1
        if op == OP_DMA_LOAD:
            for r in arange(a2, name="r"):
                for e in arange(DIM, name="e"):
                    w: f32 = dmem[a0 + r * DIM + e]
                    if e == 0:
                        sp0[a1 + r] = w
                    elif e == 1:
                        sp1[a1 + r] = w
                    elif e == 2:
                        sp2[a1 + r] = w
                    else:
                        sp3[a1 + r] = w
        elif op == OP_DMA_STORE:
            for r in arange(a2, name="sr2"):
                for e in arange(DIM, name="se2"):
                    u: f32 = 0.0
                    if e == 0:
                        u = sp0[a0 + r]
                    elif e == 1:
                        u = sp1[a0 + r]
                    elif e == 2:
                        u = sp2[a0 + r]
                    else:
                        u = sp3[a0 + r]
                    dmem[a1 + r * DIM + e] = u
        elif op == OP_LOADW:
            loadw(sp0, sp1, sp2, sp3, wst, a0)
        elif op == OP_MM0 or op == OP_MM:
            mmu(sp0, sp1, sp2, sp3, wst, ac0, ac1, ac2, ac3, a0, a2, zf)
        elif op == OP_ACCST or op == OP_ACCRELU:
            accst(ac0, ac1, ac2, ac3, sp0, sp1, sp2, sp3, a1, a2, rl)
        elif op == OP_VLOAD:
            vload(sp0, sp1, sp2, sp3, vreg, a0, a1)
        elif op == OP_VSTORE:
            vstore(vreg, sp0, sp1, sp2, sp3, a0, a1)
        else:
            vpu(op, vreg, a0, a1, a2)


# ==========================================================================#
# Scheduling.
#
# The array is the point, so it gets the whole schedule: the PE registers are
# completely partitioned (they are registers, not memory), the two spatial loops
# are unrolled so DIM*DIM physical MACs exist, and the time loop is pipelined at
# II=1 -- one clock of the array per cycle. The banks need no partition
# directive: they are already separate arrays.
# ==========================================================================#
mmu_s = mmu.schedule()
for _b in ("a_reg", "a_nxt", "ps_dl", "ps_new", "odl"):
    mmu_s.partition(mmu_s.buffer(_b), kind=mmu_s.Complete)  # PE registers
for _ln in ("pi", "pj", "sj", "sd", "ci", "cj", "cd", "zi", "zj", "zd", "za"):
    mmu_s.unroll(_ln)                                        # -> DIM*DIM MACs
mmu_s.pipeline("t")                                          # one clock / cycle
mmu_s.pipeline("ar")                                         # the accumulate walk

lw_s = loadw.schedule()
lw_s.pipeline("wn")

as_s = accst.schedule()
as_s.pipeline("am")

vpu_s = vpu.schedule()
vpu_s.pipeline("lane")
vl_s = vload.schedule()
vl_s.pipeline("vr")
vs_s = vstore.schedule()
vs_s.pipeline("sr")

top_s = tinytpu2s.schedule()
top_s.partition(top_s.buffer("vreg"), dim=1, kind=top_s.Complete)
top_s.partition(top_s.buffer("wst"), kind=top_s.Complete)
top_s.pipeline("e")
top_s.pipeline("se2")
top_s.compose(mmu_s, lw_s, as_s, vpu_s, vl_s, vs_s)


def export_backend(backend: str, part: str | None = None, freq_mhz: float = 300.0):
    """Export the composed v2 schedule to one backend. Backend choice does not
    alter the kernel or the schedule."""
    assert backend in ("cpu", "vitis", "rtl"), f"unknown backend: {backend}"
    if backend == "cpu":
        return top_s.export("cpu")
    if backend == "rtl":
        return top_s.export("rtl", freq_mhz=freq_mhz)
    kwargs: dict = {"freq_mhz": freq_mhz}
    if part:
        kwargs["part"] = part
    return top_s.export("vitis", **kwargs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="rtl", choices=("cpu", "vitis", "rtl"))
    ap.add_argument("--out", default="./tinytpu2_prj")
    ap.add_argument("--freq", type=float, default=300.0)
    args = ap.parse_args()
    r = export_backend(args.backend, freq_mhz=args.freq)
    if args.backend == "rtl":
        Path(args.out).mkdir(parents=True, exist_ok=True)
        Path(args.out, "tinytpu2.sv").write_text(r.verilog)
        print("wrote", Path(args.out, "tinytpu2.sv"))
