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
MAXROWS = 127                                      # `nr` is 8 bits, top bit signed
VLD_CHUNK = int(os.environ.get("TPU_CHUNK", 127))  # max rows per vld burst

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
    # One `dma_ld` per column block of B, not one per (nb, kb) tile: rows
    # 0..K-1 of column block nb are contiguous at `B_SP + nb*K`, so `nr = K`
    # fetches the lot. Instruction count is the term that matters -- each one
    # costs ~18 cycles of unpipelined loop overhead on top of its rows -- so
    # folding Nt*Kt instructions into Nt is worth more than it looks.
    for nb in range(Nt):
        p.append(enc(OP_DMA_LD, f0=1, f1=0, f2=nb, f3=B_SP + nb * K, nr=K))
    # Every A word into vregs once, and it stays there for the whole run.
    # Chunked only by the row-count field's range: the burst-length theory that
    # motivated small chunks was disproved (the depth requirement was the
    # simulator's thread count, see RESULTS_ISA.md), so the chunk is now as
    # large as the ISA allows and this is one instruction at every shape here.
    for c0 in range(0, Kt * M, VLD_CHUNK):
        n = min(VLD_CHUNK, Kt * M - c0)
        p.append(enc(OP_VLD, f0=A_VR + c0, f1=A_SP + c0, nr=n))

    for nb in range(Nt):
        for kb in range(Kt):
            # This k-tile's weights: T words, rows kb*T .. kb*T+T of column nb.
            p.append(enc(OP_VLD, f0=W_VR, f1=B_SP + nb * K + kb * T, nr=T))
            # One instruction per k-tile: the first overwrites the
            # accumulator, the rest accumulate into it. No `vadd`, no zeroing
            # instruction, and nothing on the critical unit but the psums the
            # array actually produced.
            p.append(enc(OP_MM, f0=A_VR + kb * M, f1=AR_C, f2=(1 if kb else 0),
                         f3=W_VR, nr=M))
        if relu:
            p.append(enc(OP_VRELU, f0=AR_C, f1=AR_C, nr=M))
        # One instruction retires the tile: accumulator -> DRAM, clipped.
        p.append(enc(OP_MVOUT, f0=AR_C, f1=0, f2=nb, nr=M))
    return p


# The instruction count is a hardware parameter (every unit's outer loop bound),
# so it is fixed at build time from the longest program the design must run.
def vadd_program():
    """A program for the vector unit itself, so `vadd`/`vrelu` stay exercised
    now that tiled GEMM no longer needs them on its inner loop.

    Computes A@B twice into two accumulator regions, adds them, ReLUs the sum,
    and retires it -- so the result is `relu(2 * (A @ B))` on the first output
    tile, which the bench checks."""
    p = [enc(OP_DMA_LD, f0=0, f1=0, f2=0, f3=A_SP, nr=M),
         enc(OP_DMA_LD, f0=1, f1=0, f2=0, f3=B_SP, nr=T),
         enc(OP_VLD, f0=A_VR, f1=A_SP, nr=M),
         enc(OP_VLD, f0=W_VR, f1=B_SP, nr=T),
         enc(OP_MM, f0=A_VR, f1=AR_C, f2=0, f3=W_VR, nr=M),
         enc(OP_MM, f0=A_VR, f1=AR_P, f2=0, f3=W_VR, nr=M),
         enc(OP_VADD, f0=AR_C, f1=AR_C, f2=AR_P, nr=M),
         enc(OP_VRELU, f0=AR_C, f1=AR_C, nr=M),
         enc(OP_MVOUT, f0=AR_C, f1=0, f2=0, nr=M)]
    return p


NPROG = max(len(gemm_program(False)), len(gemm_program(True)),
            len(vadd_program()))

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


def _unit_counts():
    """How many instructions each unit actually acts on.

    Every unit used to see all `NPROG` instructions and decode each one, even
    when it had nothing to do with it -- the PEs saw 80 and needed 16, `dma_st`
    saw 80 and needed 4, and 76% of all decode work in the design was for
    instructions the unit would ignore. Since a unit's per-instruction loop is
    not pipelined (`RESULTS_ISA.md`), that decode is not free: it is the
    dominant term in the 24 cycles/instruction measured against Gemmini's 5.9
    (`COMPARISON.md`).

    So the sequencer now dispatches each instruction only to the units that
    must act on it, and each unit's loop runs for its own count. Taken over
    both programs, since the same hardware runs `gemm` and `gemm.relu`."""
    import collections
    n = collections.Counter()
    progs = [gemm_program(False), gemm_program(True), vadd_program()]
    for prog in progs:
        c = collections.Counter(int(w) & 0x3F for w in prog)
        for k, v in (
            ("dld", c[OP_DMA_LD]),
            ("spm", c[OP_DMA_LD] + c[OP_VLD]),
            ("vru", c[OP_VLD] + c[OP_MM]),
            ("mm",  c[OP_MM]),
            ("acc", c[OP_MM] + c[OP_VADD] + c[OP_VRELU] + c[OP_MVOUT]),
            ("dst", c[OP_MVOUT]),
        ):
            n[k] = max(n[k], v)
    return n


_N = _unit_counts()
N_DLD, N_SPM, N_VRU = _N["dld"], _N["spm"], _N["vru"]
N_MM, N_ACC, N_DST = _N["mm"], _N["acc"], _N["dst"]


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
        """Fetch, decode, and dispatch each instruction to the units that act
        on it -- and to no others.

        Consumes nothing, so it cannot be in a dependence cycle with any data
        path. Its own loop is the only one in the design Vitis pipelines (II=1),
        because its body is a decode and a few puts with no inner loop.

        **The dispatch order is dataflow order** (`dma_ld`, `spm`, `vru`,
        `accu`, `dma_st`) and that is load-bearing. If the sequencer blocks on a
        full queue, every unit *upstream* of that one already holds this
        instruction and can keep producing the data the blocked unit is waiting
        for, so it drains and the block clears. Dispatching downstream-first
        deadlocks: a unit would sit waiting for operands from an upstream unit
        that had not yet been given the instruction that produces them.

        The trailing pads matter for the same reason the counts do: `gemm` and
        `gemm.relu` contain different numbers of `vrelu`, so a queue is topped
        up with NOPs to the fixed count its unit loops over."""
        n_dld: int32 = 0
        n_spm: int32 = 0
        n_vru: int32 = 0
        n_acc: int32 = 0
        n_dst: int32 = 0
        for c in range(NPROG):
            w0: UInt(64) = l_imem[c]
            op: int32 = w0[0:6]
            if op == OP_DMA_LD:
                c_dld.put(w0)
                n_dld += 1
                c_spm.put(w0)
                n_spm += 1
            if op == OP_VLD:
                c_spm.put(w0)
                n_spm += 1
                c_vru.put(w0)
                n_vru += 1
            if op == OP_MM:
                c_vru.put(w0)
                n_vru += 1
                c_acc.put(w0)
                n_acc += 1
            if op == OP_VADD:
                c_acc.put(w0)
                n_acc += 1
            if op == OP_VRELU:
                c_acc.put(w0)
                n_acc += 1
            if op == OP_MVOUT:
                c_acc.put(w0)
                n_acc += 1
                c_dst.put(w0)
                n_dst += 1
        # Top each queue up to the count its unit loops over.
        for _p in range(N_DLD - n_dld):
            c_dld.put(0)
        for _p in range(N_SPM - n_spm):
            c_spm.put(0)
        for _p in range(N_VRU - n_vru):
            c_vru.put(0)
        for _p in range(N_ACC - n_acc):
            c_acc.put(0)
        for _p in range(N_DST - n_dst):
            c_dst.put(0)

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
        for c in range(N_DLD):
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
        for c in range(N_SPM):
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
        The header word it emits ahead of every instruction is the array's only
        control input -- `is_mm` and `nrows`, nothing else."""
        vr: UInt(VW)[NVR] = 0
        n_mm: int32 = 0
        for c in range(N_VRU):
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
                n_mm += 1
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

        # Pad the array to `N_MM`. The PEs count `mm` instructions, so a
        # program with fewer than the maximum would leave them waiting for
        # headers that never arrive. A pad is a complete but empty `mm`: a
        # header with nrows = 0 and the T weight words the shift-in consumes,
        # after which the wavefront loop runs zero times. That keeps the PE
        # body branch-free, which is the point of counting `mm`s there.
        for _q in range(N_MM - n_mm):
            zh: UInt(VW) = 0
            wcol[0].put(zh)
            with allo.meta_for(T) as k2:
                zw: UInt(VW) = 0
                wcol[0].put(zw)

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
        for c in range(N_MM):
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
        for c in range(N_ACC):
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
        for c in range(N_DST):
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
