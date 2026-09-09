# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU-isa: an instruction-programmable tiled-GEMM accelerator in grid Allo.

What the earlier files in this directory got and did not get:

  * `microarch.py`     -- output-stationary, Vitis-legal, but `acc += a*b` is a
                          loop-carried dependence so `Final II = 7`.
  * `microarch_ws.py`  -- weight-stationary, II=1 per MAC, array at 100% of
                          roofline -- but *one* opcode, no scratchpad, no vector
                          unit. Operands stream from DRAM straight into the
                          array. A fast fixed-function GEMM, not a programmable
                          accelerator.

This file is the machine both were aiming at. The requirements it is built to
meet, and where each one lands:

  1. **Instruction-programmable.** Nine opcodes, a decoded instruction word
     broadcast to every unit, and units that hold no knowledge of the program.
  2. **A vector unit that tiled GEMM cannot do without.** `mm` computes the
     psums of *one* k-tile and writes them to accumulator registers; summing
     across k-tiles is an explicit `vadd`. Remove `vadd` and tiled GEMM stops
     working -- it is load-bearing, not decoration.
  3. **A scratchpad with pure SIMD access.** One row of `spad` *is* one
     `UInt(T*8)` packed word of T int8 lanes. There is no per-lane addressing
     anywhere: one port, one row per cycle, which is what makes T lanes per
     cycle come out of a single-ported memory.
  4. **vld / vst move vreg <-> scratchpad**, and nothing else touches the
     scratchpad.
  5. **A streaming interface between the vregs and the array ports.** Packed
     words travel on daisy chains, and each PE taps its own lane.

## Why chains rather than T-way fan-out

`microarch_ws.py` had a `loader` writing `a_in[0..T-1]` and a `drainer` reading
`c_out[0..T-1]`, both in fixed order. That design needs stream depth
proportional to `M * NI` -- the entire run -- which means back-pressure never
engages (`RESULTS_WS.md` section 6). Upstream's `test_multi_cache_gemm.py` has
no process of that shape: `offchip_loadA` writes exactly one stream and the
border PEs daisy-chain it, `L2_A[i] -> L2_A[i+1]`. `test_tiled_systolic.py`
runs on depth-**4** FIFOs for the same reason.

So every distribution here is a chain, and every chain carries *packed words*:

    vru --hdr,W--> wcol[0] --v--> PE(i,0) --> wcol[i+1]      (down column 0)
                                    |
                                    +-------> wrow[i,0] --> PE(i,1) --> ...
                                                             (east, lane j)

    vru --A------> acol[0] --v--> PE(i,0) --> acol[i+1]      (down column 0)
                                    |  lane i
                                    +-------> a_fwd[i,0] --> east, scalar

    PE(T-1,0) --cw[0]--> PE(T-1,1) --cw[1]--> ... --cw[T-1]--> accu
              (the bottom row packs its psums into one word as it goes east)

One word per cycle per link, so T lanes per cycle, from single-ported memories.

## Why the PEs decode nothing

A header word leads every instruction down the same chain the weights use,
carrying just `is_mm` and `nrows`. A PE forwards it and acts on it; there is no
command fan-out to `T*T` PEs and no opcode in the array. Gemmini is the same --
its PEs are dumb and `ExecuteController` decodes -- and it also means the
instruction set can grow without touching the array.

## Dependences come from being in-order

Each unit consumes the instruction stream in order and every channel is
point-to-point, so `vld` before `mm` before `vadd` before `vst` is enforced by
construction: within a unit by program order, across units by stream order.
That is the whole of the hazard logic. Gemmini spends a 48-entry reservation
station (`ReservationStation.scala`) to get out-of-order issue on top of this;
this design is strictly in-order and does not pretend otherwise.

## Data type

int8 lanes, int32 accumulation, `vst` clipping to int8 -- Gemmini's default
config (`inputType = SInt(8.W)`, `accType = SInt(32.W)`) and its `mvout`
behaviour under `ACC_SCALE_IDENTITY` with shift 0, which is what `allo_cmp.c`
passes. Packing is what makes the SIMD scratchpad work, and packing needs
integers, so unlike `microarch_ws.py` there is no fp32 switch here.
"""

import os

import allo
from allo.ir.types import int8, int16, int32, UInt, Stream
from allo.customize import Partition
import allo.dataflow as df

T = 4                          # SIMD width == array dimension
VW = T * 8                     # packed operand word: T int8 lanes
AW = T * 32                    # packed accumulator word: T int32 lanes

M = int(os.environ.get("TPU_M", 8))
K = int(os.environ.get("TPU_K", 8))
N = int(os.environ.get("TPU_N", 8))
Mt, Kt, Nt = M // T, K // T, N // T

SPAD_ROWS = int(os.environ.get("TPU_SPAD", 512))   # rows, each one packed word
NVR = int(os.environ.get("TPU_NVR", 256))          # operand vector registers
NAR = int(os.environ.get("TPU_NAR", 128))          # accumulator vector registers
MAXROWS = 127                                      # the `nr` field is 7 bits
VLD_CHUNK = int(os.environ.get("TPU_CHUNK", 8))    # max rows per vld burst

# ---------------------------------------------------------------- the ISA ----
# One 64-bit instruction word: a 6-bit opcode and five fields. The fields are
# deliberately wide enough that the encoding is not the limit on problem size.
#   op [0:6]  f0 [6:18]  f1 [18:30]  f2 [30:42]  f3 [42:54]  nr [54:61]
#
# `nr` is the row count for *every* instruction that has one, and it is only 7
# bits wide (<= MAXROWS = 127). That width is load-bearing, not cosmetic: a
# synthesis tool bounds a runtime-bounded loop by the *range of the index*, so
# when the row count came out of a 12-bit field Vitis assumed up to 4095 rows
# per instruction and reported `Trip = 1023 / 2049` with a top-level latency of
# 91407 cycles -- a worst-case bound from the encoding, not a property of the
# design. Narrowing the field narrows the bound. Gemmini does the same thing:
# its mvin/mvout carry an explicit, bounded row count.
OP_NOP = 0
OP_DMA_LD = 1     # f0=src(0=A,1=B) f1=dram_row0 f2=col_block f3=spad0  nr=rows
OP_DMA_ST = 2     # (retired: results leave via OP_MVOUT)
OP_VLD = 3        # f0=vr0  f1=spad0                            nr=rows
OP_MM = 4         # f0=vr_a f1=ar0   f3=vr_w             nr=rows
OP_VADD = 5       # f0=ar_d f1=ar_s1 f2=ar_s2            nr=rows
OP_VRELU = 6      # f0=ar_d f1=ar_s                      nr=rows
OP_MVOUT = 7      # f0=ar0 f1=dram_row0 f2=col_block     nr=rows  acc -> DRAM


def enc(op, f0=0, f1=0, f2=0, f3=0, nr=0):
    """Assemble one instruction word. The compiler backend that lowers a TOSA
    matmul into these lives only on `chia-codesign`, so programs are written by
    hand -- `gemm_program()` below is the tiled-GEMM one."""
    for v, w in ((f0, 12), (f1, 12), (f2, 12), (f3, 12), (nr, 7)):
        assert 0 <= v < (1 << w), f"field {v} does not fit in {w} bits"
    return (
        (op & 0x3F)
        | (f0 << 6)
        | (f1 << 18)
        | (f2 << 30)
        | (f3 << 42)
        | (nr << 54)
    )


# ------------------------------------------------- scratchpad / vreg layout ---
# Chosen by the program, not the hardware: the hardware only knows that a
# scratchpad row and a vreg are both one packed word.
A_SP = 0                       # A word (m, kb) at kb*M + m
B_SP = Kt * M                  # B word (k, nb) at B_SP + nb*K + k
C_SP = B_SP + Nt * K           # C word (m, nb) at C_SP + nb*M + m
A_VR = 0                       # all A words live in vregs for the whole run
W_VR = Kt * M                  # T weight words, reloaded per (nb, kb)
AR_C = 0                       # the accumulator, M words
AR_P = M + 1                   # psums of one k-tile. The +1 is not cosmetic:
                               # `ar` is cyclic-partitioned by 2, and vadd
                               # reads ar_s1 and ar_s2 and writes ar_d in one
                               # cycle. With AR_C=0 the write and the s1 read
                               # are the same address (fine on a dual-ported
                               # bank), so s2 only needs to land in the *other*
                               # bank -- which an odd base guarantees.


def gemm_program(relu=False):
    """Tiled GEMM as an instruction stream.

    The shape of the loop nest is the point: `mm` only ever produces the psums
    of a single k-tile, so the sum over k-tiles is `vadd`. A machine without a
    vector add cannot run this program.
    """
    p = []
    # A and B into the scratchpad, one column-block at a time.
    for kb in range(Kt):
        p.append(enc(OP_DMA_LD, f0=0, f1=0, f2=kb, f3=A_SP + kb * M, nr=M))
    for nb in range(Nt):
        for kb in range(Kt):
            p.append(enc(OP_DMA_LD, f0=1, f1=kb * T, f2=nb,
                         f3=B_SP + nb * K + kb * T, nr=T))
    # Every A word into vregs once, and it stays there for the whole run --
    # but issued in bounded chunks. One `vld` of `Kt*M` rows is a *burst* of
    # that many words into `sp2vr`, and a channel shallower than the burst only
    # works if the consumer drains concurrently; measured, the design needed
    # depth >= Kt*M (64 words at 16x16x16, 128 at 32x16x16). Chunking is a
    # program change, not a hardware change -- the ISA already takes a row
    # count -- and it bounds every channel by a constant.
    for c0 in range(0, Kt * M, VLD_CHUNK):
        n = min(VLD_CHUNK, Kt * M - c0)
        p.append(enc(OP_VLD, f0=A_VR + c0, f1=A_SP + c0, nr=n))

    for nb in range(Nt):
        for kb in range(Kt):
            # This k-tile's weights: T words, rows kb*T .. kb*T+T of column nb.
            p.append(enc(OP_VLD, f0=W_VR, f1=B_SP + nb * K + kb * T, nr=T))
            if kb == 0:
                # First k-tile lands straight in the accumulator, so no vadd
                # and no zeroing instruction is needed.
                p.append(enc(OP_MM, f0=A_VR + kb * M, f1=AR_C, f3=W_VR, nr=M))
            else:
                p.append(enc(OP_MM, f0=A_VR + kb * M, f1=AR_P, f3=W_VR, nr=M))
                p.append(enc(OP_VADD, f0=AR_C, f1=AR_C, f2=AR_P, nr=M))
        if relu:
            p.append(enc(OP_VRELU, f0=AR_C, f1=AR_C, nr=M))
        # One instruction retires the tile: accumulator -> DRAM, clipped.
        p.append(enc(OP_MVOUT, f0=AR_C, f1=0, f2=nb, nr=M))
    return p


# The instruction count is a hardware parameter (every unit's outer loop bound),
# so it is fixed at build time from the longest program the design must run.
NPROG = max(len(gemm_program(False)), len(gemm_program(True)))

# Stream depth. A constant, and a small one: the design's channel graph needs
# depth 4, verified two ways -- a KPN model of the exact channel structure
# (`scratchpad/kpn_model.py`) completes at depth 4 for every shape, and the Allo
# simulator agrees once it has enough threads.
#
# That thread caveat is the whole story behind what looked for a long time like
# a design problem. The simulator appears to give each process an OMP thread and
# to block the thread on an empty/full stream, so with fewer threads than
# processes a blocked process can hold a thread its own producer needed, and the
# region deadlocks. This design has T*T + 6 = 22 processes. Measured at
# 16x16x16 with QD=16:
#
#     OMP_NUM_THREADS=8   hang        OMP_NUM_THREADS=24  pass
#     OMP_NUM_THREADS=16  hang        OMP_NUM_THREADS=32  pass
#
# and with 32 threads, 16x16x16 passes at QD=4 while 32x16x16 -- which had never
# passed at any depth -- passes at QD=8. Deep FIFOs were only ever masking it,
# by letting producers finish before anyone had to block. **Run this design with
# OMP_NUM_THREADS >= 22**, not the 8 in CLAUDE.md.
QD = int(os.environ.get("TPU_QD", 8))

# Instruction memory, sized to the program rather than to a round number.
# Allo's `wrap_io` copies each m_axi argument into a local buffer before the
# region runs, so imem's *declared* length is paid as startup latency whether or
# not the program uses it. Declared as [1024] it cost 1034 cycles of the
# measured 1449 at 8x8x8 -- 71% of the runtime spent copying 1002 unused NOPs
# (`load_buf0` in the csynth report). This is an artifact of the declaration,
# not of the architecture, and the fix is to declare what the program needs.
IMEM_SIZE = max(64, NPROG)


@df.region()
def tinytpu_isa(
    imem: UInt(64)[IMEM_SIZE],
    A: int8[M, K],
    B: int8[K, N],
    C: int8[M, N],
):
    # Control: the decoded instruction word, one point-to-point queue per unit.
    # Control is a *chain* in dataflow order, not a broadcast. Each unit takes
    # the word, forwards it downstream immediately, and only then executes.
    #
    # A broadcasting sequencer puts every unit in a cycle with it:
    #   sequencer -> q_spm -> spm -> sp2vr -> vru -> acol -> array -> cw
    #             -> accu -> (needs its word from) sequencer
    # so one stalled unit fills its queue, blocks the sequencer, and starves the
    # downstream unit whose progress would have cleared the stall. Deep queues
    # only hide it, by letting the whole program buffer -- measured, the design
    # then needed QD >= ~NPROG, which is not a real machine.
    #
    # Forwarding first makes the control path follow the data path, so a unit
    # always holds its instruction before the operands arrive, and queue depth
    # is bounded by pipeline skew rather than program length.
    c_dld: Stream[UInt(64), QD]         # sequencer -> dma_ld
    c_spm: Stream[UInt(64), QD]         # dma_ld    -> spm
    c_vru: Stream[UInt(64), QD]         # spm       -> vru
    c_acc: Stream[UInt(64), QD]         # vru       -> accu
    c_dst: Stream[UInt(64), QD]         # accu      -> dma_st

    # Data paths, all one packed word wide.
    dma2sp: Stream[UInt(VW), QD]        # dma_ld -> scratchpad
    sp2vr: Stream[UInt(VW), QD]         # scratchpad -> vregs  (vld)
    ac2sp: Stream[UInt(VW), QD]         # accumulator -> dma_st (clipped)

    # The array's streaming ports. Chains, never fan-out.
    wcol: Stream[UInt(VW), QD][T]       # header + weight words, down column 0
    wrow: Stream[UInt(VW), QD][T, T]    # ... then east along row i
    acol: Stream[UInt(VW), QD][T]       # activation words, down column 0
    a_fwd: Stream[int8, QD][T, T]       # one lane, east
    p_fwd: Stream[int32, QD][T, T]      # partial sums, south
    cw: Stream[UInt(AW), QD][T]         # bottom row packs T psums going east

    @df.kernel(mapping=[1], args=[imem])
    def sequencer(l_imem: UInt(64)[IMEM_SIZE]):
        """Fetch and broadcast the decoded word. Consumes nothing, so it cannot
        be in a dependence cycle with any data path.

        Four puts to four distinct FIFOs are four distinct ports, so this is one
        cycle per instruction -- the fetch/issue rate is not the bottleneck the
        way it was on `chia-codesign`, where a unit was a `func.call` and the
        compiler would not pipeline across one."""
        for c in range(NPROG):
            w0: UInt(64) = l_imem[c]
            c_dld.put(w0)

    @df.kernel(mapping=[1], args=[A, B])
    def dma_ld(lA: int8[M, K], lB: int8[K, N]):
        """DRAM -> scratchpad. Sole reader of A and B.

        Split from the store unit deliberately. A single unit doing both put
        `spm` and the DMA in a **two-process cycle** -- `dma -> dma2sp -> spm`
        and `spm -> sp2dma -> dma` -- and with bounded FIFOs that deadlocks: the
        round trip `dma_ld; dma_st` hung with every body variant tried,
        including compile-time trip counts on both sides. Two one-way units have
        no cycle. Gemmini splits the same way, into `LoadController.scala` and
        `StoreController.scala`.

        Packs T lanes per cycle, so the scratchpad only ever sees whole words.
        The T reads land in T banks because `schedule()` partitions A and B."""
        for c in range(NPROG):
            w0: UInt(64) = c_dld.get()
            c_spm.put(w0)          # forward before executing
            op: int32 = w0[0:6]
            f0: int32 = w0[6:18]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:61]
            if op == OP_DMA_LD:
                for r in range(nr):
                    pw: UInt(VW) = 0
                    with allo.meta_for(T) as e:
                        v: int8 = 0
                        if f0 == 0:
                            v = lA[f1 + r, f2 * T + e]
                        else:
                            v = lB[f1 + r, f2 * T + e]
                        pw[8 * e : 8 * (e + 1)] = v
                    dma2sp.put(pw)

    @df.kernel(mapping=[1])
    def spm():
        """The scratchpad, and the only unit that owns it.

        Pure SIMD access: an address names a whole `UInt(T*8)` row and there is
        no way to address a lane. That is what lets a single-ported memory feed
        T lanes per cycle, and it is why `HLS 200-779` (single reader, single
        writer) is satisfied without a pragma."""
        spad: UInt(VW)[SPAD_ROWS] = 0
        for c in range(NPROG):
            w0: UInt(64) = c_spm.get()
            c_vru.put(w0)          # forward before executing
            op: int32 = w0[0:6]
            f0: int32 = w0[6:18]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:61]
            if op == OP_DMA_LD:
                for r in range(nr):
                    spad[f3 + r] = dma2sp.get()
            if op == OP_VLD:
                for r in range(nr):
                    lw: UInt(VW) = spad[f1 + r]
                    sp2vr.put(lw)
            # No write-back branch: the scratchpad is input-only. Results
            # leave through the accumulator, which is a separate memory.

    @df.kernel(mapping=[1])
    def vru():
        """Operand vector registers, and the array's input port.

        `vld` fills them from the scratchpad; `mm` streams them into the array.
        The header word it emits ahead of every instruction is the array's only
        control input -- `is_mm` and `nrows`, nothing else."""
        vr: UInt(VW)[NVR] = 0
        for c in range(NPROG):
            w0: UInt(64) = c_vru.get()
            c_acc.put(w0)          # forward before executing
            op: int32 = w0[0:6]
            f0: int32 = w0[6:18]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:61]

            # The header leads every instruction, so the array's outer loop
            # stays in lockstep with the program without decoding it.
            hdr: UInt(VW) = 0
            if op == OP_MM:
                hdr[0:1] = 1
                hdr[8:15] = nr
            wcol[0].put(hdr)

            if op == OP_VLD:
                for r in range(nr):
                    vr[f0 + r] = sp2vr.get()
            if op == OP_MM:
                # T weight words, row 0 first: each PE keeps the first word it
                # sees and forwards the rest, so row i keeps word i.
                with allo.meta_for(T) as k:
                    kw: UInt(VW) = vr[f3 + k]
                    wcol[0].put(kw)
                for r in range(nr):
                    rw: UInt(VW) = vr[f0 + r]
                    acol[0].put(rw)

    @df.kernel(mapping=[T, T])
    def pe():
        """One processing element. Weight-stationary, and it decodes nothing.

        The compute loop has **no loop-carried value**: tap a lane, take the
        partial sum from the north, multiply-add, pass both on. The multiplier
        and adder latencies are pipeline *depth*, not initiation interval, which
        is the property `microarch.py` could not have and the reason a systolic
        array is built deep."""
        i, j = df.get_pid()
        w: int8 = 0
        for c in range(NPROG):
            # --- header: down column 0, then east along the row ---
            hdr: UInt(VW) = 0
            with allo.meta_if(j == 0):
                hdr = wcol[i].get()
                with allo.meta_if(i != T - 1):
                    wcol[i + 1].put(hdr)
            with allo.meta_else():
                hdr = wrow[i, j - 1].get()
            with allo.meta_if(j != T - 1):
                wrow[i, j].put(hdr)
            is_mm: int32 = hdr[0:1]
            nrows: int32 = hdr[8:15]

            if is_mm == 1:
                # --- latch my row's weight word, forward the rows below ---
                ww: UInt(VW) = 0
                with allo.meta_if(j == 0):
                    ww = wcol[i].get()
                    with allo.meta_for(T - 1 - i) as _f:
                        wcol[i + 1].put(wcol[i].get())
                with allo.meta_else():
                    ww = wrow[i, j - 1].get()
                with allo.meta_if(j != T - 1):
                    wrow[i, j].put(ww)
                w = ww[8 * j : 8 * (j + 1)]

                # --- one wavefront per cycle ---
                for m in range(nrows):
                    a: int8 = 0
                    with allo.meta_if(j == 0):
                        aw: UInt(VW) = acol[i].get()
                        with allo.meta_if(i != T - 1):
                            acol[i + 1].put(aw)
                        a = aw[8 * i : 8 * (i + 1)]
                    with allo.meta_else():
                        a = a_fwd[i, j - 1].get()
                    p: int32 = 0
                    with allo.meta_if(i > 0):
                        p = p_fwd[i - 1, j].get()
                    # int8 x int8 -> int16 keeps this a narrow multiply; the
                    # operands bound the product at 127*127 = 16129.
                    av: int16 = a
                    wv: int16 = w
                    o: int32 = p + av * wv
                    with allo.meta_if(i != T - 1):
                        p_fwd[i, j].put(o)
                    with allo.meta_else():
                        # The bottom row assembles the packed result word as it
                        # travels east, so the accumulator sees whole words and
                        # there is no T-way fan-in.
                        cv: UInt(AW) = 0
                        with allo.meta_if(j > 0):
                            cv = cw[j - 1].get()
                        cv[32 * j : 32 * (j + 1)] = o
                        cw[j].put(cv)
                    with allo.meta_if(j != T - 1):
                        a_fwd[i, j].put(a)

    @df.kernel(mapping=[1])
    def accu():
        """Accumulator vector registers and the vector ALU.

        This is where the k-contraction that the PEs no longer do actually
        happens, and it happens *as an instruction*: `mm` deposits one k-tile's
        psums, `vadd` sums them into the running total. Gemmini splits the same
        way -- adds in `AccumulatorMem`'s write path, not in the mesh.

        `vst` clips to int8 on the way out, which is Gemmini's `mvout` under
        ACC_SCALE_IDENTITY with shift 0."""
        ar: UInt(AW)[NAR] = 0
        for c in range(NPROG):
            w0: UInt(64) = c_acc.get()
            c_dst.put(w0)          # forward before executing
            op: int32 = w0[0:6]
            f0: int32 = w0[6:18]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:61]

            if op == OP_MM:
                for r in range(nr):
                    ar[f1 + r] = cw[T - 1].get()
            if op == OP_VADD:
                for r in range(nr):
                    x: UInt(AW) = ar[f1 + r]
                    y: UInt(AW) = ar[f2 + r]
                    z: UInt(AW) = 0
                    with allo.meta_for(T) as e:
                        xe: int32 = x[32 * e : 32 * (e + 1)]
                        ye: int32 = y[32 * e : 32 * (e + 1)]
                        se: int32 = xe + ye
                        z[32 * e : 32 * (e + 1)] = se
                    ar[f0 + r] = z
            if op == OP_VRELU:
                for r in range(nr):
                    u: UInt(AW) = ar[f1 + r]
                    zr: UInt(AW) = 0
                    with allo.meta_for(T) as e:
                        ue: int32 = u[32 * e : 32 * (e + 1)]
                        re: int32 = ue
                        if re < 0:
                            re = 0
                        zr[32 * e : 32 * (e + 1)] = re
                    ar[f0 + r] = zr
            if op == OP_MVOUT:
                for r in range(nr):
                    t: UInt(AW) = ar[f0 + r]
                    ow: UInt(VW) = 0
                    with allo.meta_for(T) as e:
                        te: int32 = t[32 * e : 32 * (e + 1)]
                        if te > 127:
                            te = 127
                        if te < -128:
                            te = -128
                        tc: int8 = te
                        ow[8 * e : 8 * (e + 1)] = tc
                    ac2sp.put(ow)




    @df.kernel(mapping=[1], args=[C])
    def dma_st(lC: int8[M, N]):
        """Accumulator -> DRAM. Sole writer of C.

        Declared **last on purpose**. Allo emits the process calls in
        declaration order, and Vitis `csim` executes a dataflow region in that
        order, so a consumer declared before its producer reads an empty stream:
            ERROR [HLS SIM]: an hls::stream is read while empty
        `dma_st` consumes what `accu` produces, so it has to come after it. The
        order has no effect on the generated hardware -- in RTL the processes
        are concurrent -- but it decides whether `csim` works, and `csim` is the
        fast functional check."""
        for c in range(NPROG):
            w0: UInt(64) = c_dst.get()
            op: int32 = w0[0:6]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:61]
            if op == OP_MVOUT:
                for r in range(nr):
                    qw: UInt(VW) = ac2sp.get()
                    with allo.meta_for(T) as e:
                        ov: int8 = qw[8 * e : 8 * (e + 1)]
                        lC[f1 + r, f2 * T + e] = ov

def schedule(s):
    """Ports where the design needs them.

    A and B are read T lanes per cycle by the dma unit; C is written T lanes per
    cycle. `ar` needs two reads and a write per cycle for `vadd`, and two banks
    are enough given the odd `AR_P` base (see the comment there).

    Reachable because `df.build` is `customize(func)` followed by `s.build(...)`,
    so the schedule primitives apply on the Vitis path."""
    top = s.top_func_name
    s.partition(f"{top}:A", Partition.Cyclic, dim=2, factor=T)
    s.partition(f"{top}:B", Partition.Cyclic, dim=2, factor=T)
    s.partition(f"{top}:C", Partition.Cyclic, dim=2, factor=T)
    return s
