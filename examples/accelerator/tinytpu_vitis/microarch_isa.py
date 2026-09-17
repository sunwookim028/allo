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

# ---------------------------------------------------------------- the ISA ----
# One 64-bit instruction word: a 6-bit opcode and five fields. The fields are
# deliberately wide enough that the encoding is not the limit on problem size.
#   op [0:6]  f0 [6:18]  f1 [18:30]  f2 [30:42]  f3 [42:54]  nr [54:62]
#
# **Every field carries one more bit than its value range needs, because a
# bit-slice is extracted into a *signed* `ap_int<N>` in the emitted HLS:**
#
#     ap_int<7> v268;  v268 = w02(60, 54);   // nr
#     int32_t nr = v268;                     // 64 -> 0b1000000 -> -64
#
# so a field whose top bit is set reads back negative, and a loop bounded by it
# runs zero times. This cost a real bug: `nr = 64` for a 64-row `vld` silently
# loaded nothing, and the design produced zeros. The Allo dataflow simulator
# treats the slice as unsigned and passed, so **only cosim/csim caught it** --
# a genuine simulator/RTL divergence, and the reason `cosim.py` is worth having
# in the loop rather than at the end.
#
# The rule this imposes: an N-bit field safely carries 0 .. 2^(N-1) - 1. `nr`
# is therefore 8 bits for MAXROWS = 127, and the address fields are 12 bits for
# a 2047 maximum, which is comfortably above SPAD_ROWS and NVR.
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
OP_MM = 4         # f0=vr_a f1=ar0 f2=acc f3=vr_w       nr=rows
OP_VADD = 5       # f0=ar_d f1=ar_s1 f2=ar_s2            nr=rows
OP_VRELU = 6      # f0=ar_d f1=ar_s                      nr=rows
OP_MVOUT = 7      # f0=ar0 f1=dram_row0 f2=col_block     nr=rows  acc -> DRAM


def enc(op, f0=0, f1=0, f2=0, f3=0, nr=0):
    """Assemble one instruction word. The compiler backend that lowers a TOSA
    matmul into these lives only on `chia-codesign`, so programs are written by
    hand -- `gemm_program()` below is the tiled-GEMM one."""
    # `< (1 << (w - 1))`, not `< (1 << w)`: the top bit is the sign bit once the
    # field is extracted, see the encoding note above.
    for v, w in ((f0, 12), (f1, 12), (f2, 12), (f3, 12), (nr, 8)):
        assert 0 <= v < (1 << (w - 1)), (
            f"field {v} does not fit in {w - 1} usable bits "
            f"(bit {w - 1} is the sign bit after extraction)")
    return (
        (op & 0x3F)
        | (f0 << 6)
        | (f1 << 18)
        | (f2 << 30)
        | (f3 << 42)
        | (nr << 54)
    )


# ---------------------------------------------------------------------------
# THE HARDWARE. Every constant below is fixed at build time and is *independent
# of the workload*: one RTL build runs every shape, with M, K and N arriving as
# instruction fields rather than as Python constants.
#
# This is the property the comparison needs. Gemmini's numbers come from one
# elaboration -- `allo_cmp.c` declares `elem_t A[MAXDIM][MAXDIM]` and passes
# MAXDIM as the stride for every shape it runs -- so a per-workload
# specialization on our side would not be measuring the same kind of object.
# Earlier revisions of this file did exactly that: M/K/N, the instruction count,
# and every unit's loop bound were compile-time constants, so 4x4x4 and
# 16x16x16 were *different accelerators*. They are now the same one.
# ---------------------------------------------------------------------------
T = int(os.environ.get("TPU_T", 4))   # SIMD width == array dimension
# T is the one parameter that changes the *shape* of the generated region:
# the array is T*T kernel instances and the chains are T and T*T stream
# arrays, so T=16 is 262 instances and ~800 streams. That was unrunnable
# until the simulator's OpenMP team was sized to the section count
# (notes/ALLO_SHORTCOMINGS.md #11); before that fix it hung with no output.
VW = T * 8                     # packed operand word: T int8 lanes
AW = T * 32                    # packed accumulator word: T int32 lanes

MAXDIM = int(os.environ.get("TPU_MAXDIM", 16))     # largest M, K, N supported
# A, B and C are **flat** at the region boundary, addressed `row * MAXDIM + col`.
# That is what DRAM is, and it is also what lets the design be built with
# `wrap_io=False`: with 2-D arguments Allo wraps each one in a bulk copy into a
# local buffer before the region starts, sized to the *declared* array rather
# than to the shape being run -- four copies of 256 words, 1024 cycles, which
# was the entire fixed-cost gap against Gemmini (which DMAs only the tiles it
# touches). `wrap_io=False` refuses multi-dimensional arguments to nested
# kernels ("Top-level multi-dimensional arrays are linearized to 1D pointers"),
# so flat is the shape that makes it legal.
SPAD_ROWS = int(os.environ.get("TPU_SPAD", 512))   # rows, each one packed word
NVR = int(os.environ.get("TPU_NVR", 256))          # operand vector registers
NAR = int(os.environ.get("TPU_NAR", 128))          # accumulator vector registers
QD = int(os.environ.get("TPU_QD", 8))              # stream depth

MAXROWS = 127                                      # `nr` is 8 bits, top bit signed
NHDR = 8                       # imem[0:NHDR] is the header, instructions follow

# Instruction slots. Sized to the longest program the built MAXDIM admits, not
# to a round number: with `wrap_io=True` Allo copies each argument into a local
# buffer before the region starts, and the copy is the *declared* length, so
# every unused slot is a wasted startup cycle. The worst case is
# gemm(MAXDIM,MAXDIM,MAXDIM) with relu -- Kt + Nt dma_lds, Kt vlds for A,
# Nt*(Kt vld + Kt mm) + Nt relu + Nt mvout -- plus the NHDR header.
_KB = MAXDIM // T
IMEM_SIZE = int(os.environ.get(
    "TPU_IMEM", NHDR + 2 * _KB + _KB + _KB * (2 * _KB) + 2 * _KB + 16))

# Scratchpad and vreg layout. Fixed offsets in a fixed memory, sized for the
# largest supported shape rather than for the shape being run.
KB_MAX = MAXDIM // T           # column blocks in the widest matrix
A_SP = 0                       # A words:  kb * MAXDIM + m
B_SP = KB_MAX * MAXDIM         # B words:  nb * MAXDIM + k
A_VR = 0                       # A vregs mirror the A scratchpad region
W_VR = KB_MAX * MAXDIM         # T weight words, reloaded per (nb, kb)
AR_C = 0                       # the accumulator, up to MAXDIM words
AR_P = MAXDIM + 1              # scratch region for vector-unit programs


@df.region()
def tinytpu_isa(
    imem: UInt(64)[IMEM_SIZE],
    A: int8[MAXDIM * MAXDIM],
    B: int8[MAXDIM * MAXDIM],
    C: int8[MAXDIM * MAXDIM],
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
        """Fetch, decode, and dispatch each instruction to the units that act
        on it -- and to no others.

        **The loop bounds come from the program, not from the build.** The
        header `imem[0:NHDR]` carries the instruction count and the per-unit
        counts, so this one RTL build runs any shape: the sequencer forwards
        each unit its own count as the first word on its queue, and each unit
        loops that many times. Runtime trip counts are what make the design
        workload-independent, and their cost is that synthesis can only report
        a worst-case bound (`RESULTS_ISA.md`), which is why cosim is the
        measurement that counts.

        Consumes nothing, so it cannot be in a dependence cycle with any data
        path. Its own loop is the only one Vitis pipelines (II=1), because its
        body is a decode and a few puts with no inner loop.

        **The dispatch order is dataflow order** (`dma_ld`, `spm`, `vru`,
        `accu`, `dma_st`) and that is load-bearing. If the sequencer blocks on a
        full queue, every unit *upstream* of that one already holds this
        instruction and can keep producing what the blocked unit waits for, so
        it drains and the block clears. Dispatching downstream-first deadlocks:
        a unit would wait for operands from an upstream unit that had not yet
        been given the instruction that produces them."""
        iw: UInt(64) = l_imem[0]
        n_instr: int32 = iw[0:16]
        # Each unit's count travels ahead of its instructions.
        c_dld.put(l_imem[1])
        c_spm.put(l_imem[2])
        c_vru.put(l_imem[3])
        c_vru.put(l_imem[4])   # ... and the `mm` count, for the array
        c_acc.put(l_imem[5])
        c_dst.put(l_imem[6])
        for c in range(n_instr):
            w0: UInt(64) = l_imem[NHDR + c]
            op: int32 = w0[0:6]
            if op == OP_DMA_LD:
                c_dld.put(w0)
                c_spm.put(w0)
            if op == OP_VLD:
                c_spm.put(w0)
                c_vru.put(w0)
            if op == OP_MM:
                c_vru.put(w0)
                c_acc.put(w0)
            if op == OP_VADD:
                c_acc.put(w0)
            if op == OP_VRELU:
                c_acc.put(w0)
            if op == OP_MVOUT:
                c_acc.put(w0)
                c_dst.put(w0)

    @df.kernel(mapping=[1], args=[A, B])
    def dma_ld(lA: int8[MAXDIM * MAXDIM], lB: int8[MAXDIM * MAXDIM]):
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
        nw: UInt(64) = c_dld.get()
        n_own: int32 = nw[0:16]     # 16-bit slice: a stream word is UInt(64),
                                    # and the top bit of a slice is its sign,
                                    # so this carries 0..32767 counts.
        for c in range(n_own):
            w0: UInt(64) = c_dld.get()
            op: int32 = w0[0:6]
            f0: int32 = w0[6:18]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:62]
            if op == OP_DMA_LD:
                for r in range(nr):
                    pw: UInt(VW) = 0
                    with allo.meta_for(T) as e:
                        v: int8 = 0
                        if f0 == 0:
                            v = lA[(f1 + r) * MAXDIM + f2 * T + e]
                        else:
                            v = lB[(f1 + r) * MAXDIM + f2 * T + e]
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
        nw: UInt(64) = c_spm.get()
        n_own: int32 = nw[0:16]     # 16-bit slice: a stream word is UInt(64),
                                    # and the top bit of a slice is its sign,
                                    # so this carries 0..32767 counts.
        for c in range(n_own):
            w0: UInt(64) = c_spm.get()
            op: int32 = w0[0:6]
            f0: int32 = w0[6:18]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:62]
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
        The only control the array ever receives is the `mm` count, once, then
        one header per `mm` carrying its row count -- no opcode reaches the
        array at all."""
        nw: UInt(64) = c_vru.get()
        n_own: int32 = nw[0:16]     # 16-bit slice: a stream word is UInt(64),
                                    # and the top bit of a slice is its sign,
                                    # so this carries 0..32767 counts.
        mw: UInt(64) = c_vru.get()
        n_mm_in: int32 = mw[0:16]
        vr: UInt(VW)[NVR] = 0
        # Hand the array its own count before anything else, on the chain it
        # already uses for headers and weights.
        nmo: UInt(VW) = n_mm_in
        wcol[0].put(nmo)
        for c in range(n_own):
            w0: UInt(64) = c_vru.get()
            op: int32 = w0[0:6]
            f0: int32 = w0[6:18]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:62]

            if op == OP_VLD:
                for r in range(nr):
                    vr[f0 + r] = sp2vr.get()
            if op == OP_MM:
                # The header carries the row count, and it is sent **only for
                # `mm`** -- the array's loop counts `mm` instructions, not
                # program instructions, so there is nothing for it to skip and
                # no opcode in the array at all.
                hdr: UInt(VW) = 0
                hdr[0:12] = nr
                wcol[0].put(hdr)
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

        Its loop counts **`mm` instructions**, not program instructions: the
        sequencer sends `mm` only to the units that execute it, so the array
        never sees -- and never spends a cycle decoding -- a `vld`, `vadd` or
        `mvout`. At 16x16x16 that is 16 iterations instead of 80.

        The compute loop has **no loop-carried value**: tap a lane, take the
        partial sum from the north, multiply-add, pass both on. The multiplier
        and adder latencies are pipeline *depth*, not initiation interval, which
        is the property `microarch.py` could not have and the reason a systolic
        array is built deep."""
        i, j = df.get_pid()
        w: int8 = 0
        # The `mm` count arrives on the same chain the headers use, so the
        # array learns how many matmuls the program holds without decoding it.
        nmw: UInt(VW) = 0
        with allo.meta_if(j == 0):
            nmw = wcol[i].get()
            with allo.meta_if(i != T - 1):
                wcol[i + 1].put(nmw)
        with allo.meta_else():
            nmw = wrow[i, j - 1].get()
        with allo.meta_if(j != T - 1):
            wrow[i, j].put(nmw)
        nmm: int32 = nmw[0:16]
        for c in range(nmm):
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
            nrows: int32 = hdr[0:12]

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
        nw: UInt(64) = c_acc.get()
        n_own: int32 = nw[0:16]     # 16-bit slice: a stream word is UInt(64),
                                    # and the top bit of a slice is its sign,
                                    # so this carries 0..32767 counts.
        for c in range(n_own):
            w0: UInt(64) = c_acc.get()
            op: int32 = w0[0:6]
            f0: int32 = w0[6:18]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:62]

            if op == OP_MM:
                # `f2` selects overwrite (0) or accumulate (1). Accumulating
                # here is what makes the k-contraction free: with it, a Kt-deep
                # contraction is Kt `mm`s and no `vadd` at all, where before it
                # was Kt `mm`s *and* Kt-1 `vadd`s -- and both ran in this one
                # unit, so `accu` was doing 2.25 row-ops for every wavefront the
                # array produced and was the critical unit in the design
                # (csynth interval 10772 against ~6000 for every other unit).
                #
                # This is Gemmini's structure, not a shortcut around the ISA:
                # `AccumulatorMem.scala` puts the add in the memory's write
                # path and the matmul carries an accumulate bit. `vadd` remains
                # a real instruction for elementwise work -- `vadd_program()`
                # exercises it -- it is just no longer on the GEMM inner loop.
                #
                # It also cannot recur: consecutive iterations touch different
                # addresses (`f1 + r` for successive r), so the add is pipeline
                # depth rather than initiation interval. That is the same reason
                # `microarch_ws.py`'s drainer accumulated at II=1.
                for r in range(nr):
                    v: UInt(AW) = cw[T - 1].get()
                    base: UInt(AW) = 0
                    if f2 == 1:
                        base = ar[f1 + r]
                    z: UInt(AW) = 0
                    with allo.meta_for(T) as e:
                        be: int32 = base[32 * e : 32 * (e + 1)]
                        ve: int32 = v[32 * e : 32 * (e + 1)]
                        se: int32 = be + ve
                        z[32 * e : 32 * (e + 1)] = se
                    ar[f1 + r] = z
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
    def dma_st(lC: int8[MAXDIM * MAXDIM]):
        """Accumulator -> DRAM. Sole writer of C.

        Declared **last on purpose**. Allo emits the process calls in
        declaration order, and Vitis `csim` executes a dataflow region in that
        order, so a consumer declared before its producer reads an empty stream:
            ERROR [HLS SIM]: an hls::stream is read while empty
        `dma_st` consumes what `accu` produces, so it has to come after it. The
        order has no effect on the generated hardware -- in RTL the processes
        are concurrent -- but it decides whether `csim` works, and `csim` is the
        fast functional check."""
        nw: UInt(64) = c_dst.get()
        n_own: int32 = nw[0:16]     # 16-bit slice: a stream word is UInt(64),
                                    # and the top bit of a slice is its sign,
                                    # so this carries 0..32767 counts.
        for c in range(n_own):
            w0: UInt(64) = c_dst.get()
            op: int32 = w0[0:6]
            f1: int32 = w0[18:30]
            f2: int32 = w0[30:42]
            f3: int32 = w0[42:54]
            nr: int32 = w0[54:62]
            if op == OP_MVOUT:
                for r in range(nr):
                    qw: UInt(VW) = ac2sp.get()
                    with allo.meta_for(T) as e:
                        ov: int8 = qw[8 * e : 8 * (e + 1)]
                        lC[(f1 + r) * MAXDIM + f2 * T + e] = ov

def gemm_program(M, K, N, relu=False):
    """Tiled GEMM for *any* shape, on fixed hardware.

    M, K and N are arguments, not build constants: this returns an instruction
    stream and the same RTL runs it. `mm` carries an accumulate bit, so a
    Kt-deep contraction is Kt `mm`s and no `vadd` -- Gemmini's split, with the
    add in the accumulator's write path (`AccumulatorMem.scala`).

    Operands sit in DRAM at the fixed `MAXDIM` stride, which is what
    `allo_cmp.c` does for Gemmini too (`MAXDIM, MAXDIM, 0, MAXDIM`), so one set
    of host buffers serves every shape.
    """
    assert M <= MAXDIM and K <= MAXDIM and N <= MAXDIM, (
        f"{M}x{K}x{N} exceeds the built MAXDIM={MAXDIM}")
    assert M % T == 0 and K % T == 0 and N % T == 0
    assert M <= MAXROWS and K <= MAXROWS
    Kt, Nt = K // T, N // T
    p = []
    for kb in range(Kt):
        p.append(enc(OP_DMA_LD, f0=0, f1=0, f2=kb, f3=A_SP + kb * MAXDIM, nr=M))
    for nb in range(Nt):
        p.append(enc(OP_DMA_LD, f0=1, f1=0, f2=nb, f3=B_SP + nb * MAXDIM, nr=K))
    total_a = Kt * MAXDIM
    for c0 in range(0, total_a, MAXROWS):
        n = min(MAXROWS, total_a - c0)
        p.append(enc(OP_VLD, f0=A_VR + c0, f1=A_SP + c0, nr=n))
    for nb in range(Nt):
        for kb in range(Kt):
            p.append(enc(OP_VLD, f0=W_VR, f1=B_SP + nb * MAXDIM + kb * T, nr=T))
            p.append(enc(OP_MM, f0=A_VR + kb * MAXDIM, f1=AR_C,
                         f2=(1 if kb else 0), f3=W_VR, nr=M))
        if relu:
            p.append(enc(OP_VRELU, f0=AR_C, f1=AR_C, nr=M))
        p.append(enc(OP_MVOUT, f0=AR_C, f1=0, f2=nb, nr=M))
    return p


def vadd_program(M, K, N):
    """A program for the vector unit itself, so `vadd`/`vrelu` stay exercised
    now that tiled GEMM no longer needs them on its inner loop.

    Computes A@B twice into two accumulator regions, adds them, ReLUs the sum
    and retires it: `relu(2 * (A @ B))` on the first output tile."""
    return [enc(OP_DMA_LD, f0=0, f1=0, f2=0, f3=A_SP, nr=M),
            enc(OP_DMA_LD, f0=1, f1=0, f2=0, f3=B_SP, nr=T),
            enc(OP_VLD, f0=A_VR, f1=A_SP, nr=M),
            enc(OP_VLD, f0=W_VR, f1=B_SP, nr=T),
            enc(OP_MM, f0=A_VR, f1=AR_C, f2=0, f3=W_VR, nr=M),
            enc(OP_MM, f0=A_VR, f1=AR_P, f2=0, f3=W_VR, nr=M),
            enc(OP_VADD, f0=AR_C, f1=AR_C, f2=AR_P, nr=M),
            enc(OP_VRELU, f0=AR_C, f1=AR_C, nr=M),
            enc(OP_MVOUT, f0=AR_C, f1=0, f2=0, nr=M)]


def assemble(prog):
    """Prepend the header the sequencer reads.

    The header is how one fixed build runs a variable program: each unit is
    told how many instructions *it* will receive, so every loop bound in the
    design is data rather than a build constant. The counts must match the
    sequencer's dispatch rules exactly -- a unit promised more than it is sent
    waits forever -- so they are derived here from the same opcode
    classification the sequencer applies.

        imem[0] instruction count      imem[4] mm  (the array)
        imem[1] dma_ld                 imem[5] accu
        imem[2] spm                    imem[6] dma_st
        imem[3] vru                    imem[7] unused
    """
    import collections
    c = collections.Counter(int(w) & 0x3F for w in prog)
    hdr = [len(prog),
           c[OP_DMA_LD],
           c[OP_DMA_LD] + c[OP_VLD],
           c[OP_VLD] + c[OP_MM],
           c[OP_MM],
           c[OP_MM] + c[OP_VADD] + c[OP_VRELU] + c[OP_MVOUT],
           c[OP_MVOUT],
           0]
    assert len(hdr) == NHDR
    words = hdr + [int(w) for w in prog]
    assert len(words) <= IMEM_SIZE, f"{len(words)} words > IMEM_SIZE={IMEM_SIZE}"
    return words


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
