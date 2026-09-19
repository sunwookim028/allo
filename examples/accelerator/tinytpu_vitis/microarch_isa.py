# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU-isa: an instruction-programmable tiled-GEMM accelerator in grid Allo.

What the earlier designs got and did not get. Both files have since been removed;
read them with `git show e2451b81:examples/accelerator/tinytpu_vitis/<file>`.
  * `microarch.py`     -- output-stationary, Vitis-legal, but `acc += a*b` is a
                          loop-carried dependence so `Final II = 7`.
  * `microarch_ws.py`  -- weight-stationary, II=1 per MAC, array at 100% of
                          roofline -- but *one* opcode, no VMEM, no vector
                          unit. Operands stream from DRAM straight into the
                          array. A fast fixed-function GEMM, not a programmable
                          accelerator.

This file is the machine both were aiming at. The requirements it is built to
meet, and where each one lands:

  1. **Instruction-programmable.** Nine opcodes, a decoded instruction word
     broadcast to every unit, and units that hold no knowledge of the program.
  2. **A vector unit.** `vadd` / `vrelu` operate on accumulator rows. This
     item used to say tiled GEMM cannot do without `vadd`, because summing
     across k-tiles was an explicit `vadd`. That is no longer true: `mm`'s
     `f2` field selects overwrite or accumulate, so the shipped GEMM program
     emits no `vadd` at all, and `vadd_program` exists to keep the vector unit
     exercised.
  3. **A VMEM with pure SIMD access.** One row of `vmem` *is* one
     `UInt(T*8)` packed word of T int8 lanes. There is no per-lane addressing
     anywhere: one port, one row per cycle, which is what makes T lanes per
     cycle come out of a single-ported memory.
  4. **Explicit data movement.** `dma_ld` moves DRAM rows into the VMEM
     or straight into the vregs, `vld` moves VMEM rows into the vregs,
     and `mvout` retires accumulator rows to DRAM.
  5. **A streaming interface between the memories and the array ports.**
     Activations stream from the vregs, weights from the VMEM; packed
     words travel on daisy chains, and each PE taps its own lane.

## Why chains rather than T-way fan-out

`microarch_ws.py` had a `loader` writing `a_in[0..T-1]` and a `drainer` reading
`c_out[0..T-1]`, both in fixed order. That design needs stream depth
proportional to `M * NI` -- the entire run -- which means back-pressure never
engages (`RESULTS_WS.md` section 6; both files at `e2451b81`). Upstream's
`test_multi_cache_gemm.py` has no process of that shape: `offchip_loadA` writes
exactly one stream and the border PEs daisy-chain it, `L2_A[i] -> L2_A[i+1]`.
`test_tiled_systolic.py` runs on depth-**4** FIFOs for the same reason.

So every distribution here is a chain, and every chain carries *packed words*:

    vmu --hdr,W--> wcol[0] --v--> wld(i,0) --> wcol[i+1]     (down column 0)
                                    |    +--wq--> PE(i,0)    (its own weight)
                                    +-------> wrow[i,0] --> wld(i,1) --> ...
                                                             (east, lane j)

    vru --A------> acol[0] --v--> PE(i,0) --> acol[i+1]      (down column 0)
                                    |  lane i
                                    +-------> a_fwd[i,0] --> east, scalar

    PE(T-1,0) --cw[0]--> PE(T-1,1) --cw[1]--> ... --cw[T-1]--> accu
              (the bottom row packs its psums into one word as it goes east)

One word per cycle per link, so T lanes per cycle, from single-ported memories.

## Why the PEs decode nothing

A header word leads every `mm` down the chain the weights use, carrying just its
row count. Each PE's weight loader (`wld`) forwards it and hands its PE one
`(weight, rows)` word; there is no command fan-out to `T*T` PEs and no opcode in
the array. Gemmini is the same --
its PEs are dumb and `ExecuteController` decodes -- and it also means the
instruction set can grow without touching the array.

## Dependences come from being in-order

Each unit consumes the instruction stream in order and every channel is
point-to-point, so `dma_ld` before `mm` before `vadd` before `mvout` is enforced by
construction: within a unit by program order, across units by stream order.
That is the whole of the hazard logic. Gemmini spends a 48-entry reservation
station (`ReservationStation.scala`) to get out-of-order issue on top of this;
this design is strictly in-order and does not pretend otherwise.

## ROW-FLATTENING: why every unit is one loop and not two

Each unit used to be a loop over *instructions* containing a loop over *rows*:

    for c in range(n_own):          # instructions this unit is sent
        w0 = c_unit.get(); <decode>
        if op == OP_A:
            for r in range(nr): <work>
        if op == OP_B:
            for r in range(nr): <work>

Vitis reported `Pipelined = no` on every one of those outer loops, with an
iteration latency of 69-74 cycles, and the reason is structural rather than a
tool defect: modulo scheduling needs a fixed II, an inner loop whose trip count
arrives in an instruction field cannot be unrolled to give one, and so the outer
loop has no II at all. An independent check found Catapult will not pipeline it
either. The sequencer was the only unit that pipelined at all, because its body
has no inner loop -- and even it closes at II=5, on the loop-stack recurrence.

The consequence was measured against Gemmini: 18.1 cycles per instruction
against its 10.8, with `vru` -- 33 instructions and 464 words at 16x16x16 --
as the critical process. A unit finished instruction *n* before starting *n+1*,
so every instruction boundary cost a pipeline fill and drain plus the call
overhead of the sub-function Vitis extracts each inner loop into.

**Every unit is now one flat loop over the work items it will process**, with
the instruction fetched on the iteration that needs it:

    r = -1
    for x in range(n_work):         # ROWS (or words), from the header
        r += 1
        if r >= cnt:                # this row starts a new instruction
            w0 = c_unit.get(); <decode>; cnt = <work items>; r = 0
        <straight-line body, indexed by r>

There is no inner loop left, so the body is straight-line and schedules with a
real II; the conditional stream read is legal in a pipelined region and simply
back-pressures. The header carries the dynamic *work* count per unit instead of
the dynamic instruction count -- `assemble()` already expanded the control flow
to compute the latter, so this is the same mechanism with `nr` summed rather
than counted.

**What could not be split, and why the arms stay in one process.** The obvious
alternative -- one process per opcode, fed by per-opcode queues -- founders on
one owner per memory. `vmem` lives in `vmu` and is written by `dma_ld` and read
by `vld` and `mm`; `vr` lives in `vru` and is written by `vld` and `dma_ld` and
read by `mm`; `ar` lives in `accu` and is touched by all four of its opcodes.
Allo enforces single reader and single writer (Vitis itself would share an array
between two processes under `stream type=unsync`; see the limitations
register), so those three units keep every arm. Flattening gets what the split
was wanted for without moving a memory: the opcode test survives as a mux
inside a pipelined body instead of as sub-loops the scheduler must serialize.

Four traps a naive flattening falls into, and what each unit does instead:

  * **T+1 writes to one FIFO in one iteration.** `vmu` emits a header word and
    T weight words per `mm`; in the fetch branch that would schedule the whole
    loop at II=T+1. It charges the prologue T+1 *iterations* instead -- the
    same cycles, II=1 everywhere else.
  * **the row counter incremented at the bottom.** Vitis put the add in the
    last stage and the next iteration's conditional queue read depends on it:
    `Final II = 2` on every unit. Hoisting the increment costs nothing.
  * **a row count that depends on the decoded opcode** (`cnt = nr`, or
    `T + 1` for an `mm`) is a carried dependence on the counter and closes the
    loop at `Final II = 2` -- measured in `vmu` and `accu`. The sequencer
    precomputes each unit's count and sends it in that unit's copy of `nr`.
  * **a read in each arm.** The obvious transcription synthesized to one read
    port per arm on one memory (`vru` II=2, `accu` II=3). Every unit reads its
    memory ONCE per iteration at an address the branch selects.

**`accu` was the unit that would not flatten**, for a reason the other units do
not have: its row index is a carried register, so Vitis cannot prove that
iteration n's store and iteration n+1's load of `ar` touch different rows, and
the flat loop closes at `Final II = 3` (II=2 with `ar` in registers, II=1 with a
write-behind rotation at 13.7x the flip-flops, reverted). It is flat now because
`schedule()` asserts the absence of that dependence with `s.dependence` -- a
claim the assembler makes true by enforcing a minimum read-after-write
distance on `ar` (THE ACCUMULATOR DISTANCE CONTRACT, below).

**The PEs are flat too, and their weights are double-buffered.** A PE used to
run one loop per `mm` with a serial prologue -- read the header, latch a weight,
forward the rest down the column -- so no wavefront entered the array between
two `mm`s until the next weight had walked the chain. The prologue now lives in
a separate weight loader per PE (`wld`) that hands its PE `(weight, rows)`
through a depth-4 FIFO while the PE computes the previous `mm` (Gemmini's c1/c2
double buffer), and the PE runs one flat loop over every wavefront row.

## THE DMA: `wrap_io=False`, and one burst per thing the program names

The design is built `wrap_io=False`, so nothing is hoisted for it and every
unit addresses `m_axi` itself. That was not a free switch: with `wrap_io=True`
Allo wraps each argument in a bulk copy before the region starts
(`wrap_data_movement`, `allo/ir/transform.py`), and the extent it copies is
`MemRefType(arg.type).shape` -- the STATIC type, with no offset and no length.
At MAXDIM=16 that is imem 56 + A 256 + B 256 + C 256 = **824 words copied
whether the program touches them or not**, at II=1, which cosim charges as a
fixed 907 cycles: 57% of 16x16x16 and 90% of 4x4x4, and it did not move for the
whole of the array's development. It is not a DMA; it is an argument-passing
convention.

Turning it off alone makes things worse, and an earlier measurement
(fixed 481, marginal 39.8 against 15.1) concluded from that that `m_axi` is
inherently slow. It is not. Two access patterns were put to Vitis and they
synthesize to completely different hardware:

  * **strided** -- `lA[(f1 + r) * MAXDIM + f2 * T + e]`, `e` unrolled over the
    T lanes, one row per iteration:
    `[HLS 214-115] Multiple burst reads of length 4 and bit width 8`.
    A separate four-beat AXI transaction per row, `Final II = 4` on the loop
    and nearer 9 cycles a row once request latency is counted.
  * **contiguous with a RUNTIME trip count** -- `for i in range(n): b[i] = lA[i]`:
    `[HLS 214-115] Multiple burst reads of variable length and bit width 8`.
    One real AXI burst, and the loop closes at the port's own limit.

So Allo can express a program-controlled burst DMA today; the old measurement
condemned the access pattern, not the configuration. Both of the design's
`m_axi` readers were restructured around the second pattern:

  * `sequencer` pulls the whole program on-chip with one `IMEM_SIZE`-word burst,
    8 words per cycle since `align_value` widened gmem0 to 512 bits, and
    fetches from on-chip memory afterwards. Fetched in place,
    `imem[NHDR + pc * IWORDS]` is a data-dependent address that Vitis can only
    burst **two words at a time**, and the fetch loop closes at II=13 instead
    of 5. This, not the operand path, was most of the 39.8.
  * `dma_ld` opens with one variable-length burst per operand matrix, covering
    exactly the DRAM rows the program will name -- the assembler resolves the
    control flow and the AGU and passes the two spans in the header. Every
    instruction afterwards is one BRAM read, and the flat row loop is back to
    II=1 from II=4.

Measured over the five shapes, one build each:

| shape | `wrap_io=True` | `wrap_io=False`, bursts |
|---|---|---|
| 4x4x4    | 1004 | **680**  (1.48x) |
| 8x8x8    | 1108 | **831**  (1.33x) |
| 12x12x12 | 1294 | **1066** (1.21x) |
| 16x16x8  | 1344 | **1139** (1.18x) |
| 16x16x16 | 1586 | **1457** (1.09x) |

Fixed cost 907 -> 557, marginal cost 15.12 -> 20.07 cycles/instruction. That
is a real trade and it is stated as one: the win is the fixed term and it is
largest where the fixed term dominates. The crossover is at 71 dynamic
instructions, which is past the longest program this MAXDIM admits (45), so the
burst build wins everywhere it can be run -- but it would not at a larger
MAXDIM without also fixing what the marginal term buys.

**What the remaining +5 cycles/instruction is.** `dma_st` still writes `C` with
the strided pattern -- `[HLS 214-115] Multiple burst writes of length 4 and bit
width 8`, II=4 -- because a contiguous write-back would have to either clobber
the columns the program never named or be deferred to the end of the run, where
it would serialize behind the last `mvout` instead of overlapping the compute
it currently overlaps. That is the next thing to measure, not an oversight.

**What the operand burst is NOT worth.** Merging the A and B bursts into one
loop bounded by `max(na, nb)` halves the burst time -- they are separate
bundles and Vitis attributes a variable-length burst on each to the one loop --
and it was built, measured, and changed the cosim count at all five shapes by
exactly **zero cycles**. The operand burst is already entirely hidden behind
the sequencer's dispatch and the units' pipeline fill, so the version that
reads the fewest bytes is the one kept. Halving something off the critical path
buys nothing, which is worth writing down.

**One toolchain avenue that was wrongly called closed, and is now open.**
`config_interface -m_axi_max_widen_bitwidth 512` lets Vitis widen the 8-bit
`m_axi` ports so a long burst moves 64 bytes a beat instead of one. This
docstring used to say it "does nothing here" and blame
`[HLS 214-307] Could not widen since type i8 size is greater than or equal to
alignment 1(bytes)`. **That error does not occur on this design** -- it came
from a standalone probe. Here the setting is simply accepted and the ports stay
8 bits with no diagnostic, which is worse, because nothing tells you.

The diagnosis underneath was right: Allo emitted the pointers with no alignment
attribute, so Vitis assumed 1 byte and declined. With `align_value` emitted
(`configs={"align_value": 64}`) the same setting gives gmem0 bit width 512,
gmem1/2 32, and takes both `dma_ld`'s burst loop and `dma_st` from II=4 to
II=1.

## Data type

int8 lanes, int32 accumulation, `mvout` clipping to int8 -- Gemmini's default
config (`inputType = SInt(8.W)`, `accType = SInt(32.W)`) and its `mvout`
behaviour under `ACC_SCALE_IDENTITY` with shift 0, which is what `allo_cmp.c`
passes. Packing is what makes the SIMD VMEM work, and packing needs
integers, so unlike `microarch_ws.py` (at `e2451b81`) there is no fp32 switch here.
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
# bit-slice USED TO BE extracted into a *signed* `ap_int<N>` in the emitted HLS:**
#
#     ap_int<7> v268;  v268 = w02(60, 54);   // nr
#     int32_t nr = v268;                     // 64 -> 0b1000000 -> -64
#
# so a field whose top bit was set read back negative, and a loop bounded by it
# ran zero times. This cost a real bug: `nr = 64` for a 64-row `vld` silently
# loaded nothing, and the design produced zeros. The Allo dataflow simulator
# treated the slice as unsigned and passed, so **only cosim/csim caught it** --
# a genuine simulator/RTL divergence, and the reason `cosim.py` is worth having
# in the loop rather than at the end.
#
# Fixed since: slices are emitted unsigned (fork 3de74846, upstream #612, merged
# in dc6b8fa6; limitations register item 12). The spare bit is kept as a
# conservative encoding rule, and `enc()` still asserts it.
#
# The rule this imposed: an N-bit field safely carries 0 .. 2^(N-1) - 1. `nr`
# is therefore 8 bits for MAXROWS = 127, and the address fields are 12 bits for
# a 2047 maximum, which is comfortably above VMEM_ROWS and NVREG.
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
OP_DMA_LD = 1     # f0=src f1=dram_row0 f2=col_block f3=vmem0          nr=rows
OP_DMA_ST = 2     # (retired: results leave via OP_MVOUT)
OP_VLD = 3        # f0=vr0  f1=vmem0                            nr=rows
OP_MM = 4         # (retired: split into vmatload / vmatpush / vmatpop)
OP_VADD = 5       # f0=ar_d f1=ar_s1 f2=ar_s2            nr=rows
OP_VRELU = 6      # f0=ar_d f1=ar_s                      nr=rows
OP_MVOUT = 7      # f0=ar0 f1=dram_row0 f2=col_block     nr=rows  acc -> DRAM
OP_LOOP = 8       # open a loop, body is the next instruction   nr=trip count
OP_ENDLOOP = 9    # close the innermost loop
# MiniTPU's M slot (alignment increment 2). The array computes Y = X W.
OP_VMATLOAD = 10  # f0=vr_w: T weight rows, W row i = vr[f0 + i]    nr=T
OP_VMATPUSH = 11  # f0=vr_a: activation rows                        nr=rows
OP_VMATPOP = 12   # f0=ar_d: the oldest un-popped result rows       nr=rows

# `dma_ld`'s f0 is the SOURCE matrix: 0 A, 1 B. The destination is always
# VMEM (MiniTPU: DMA moves only between DRAM and VMEM). TinyTPU-isa v1 also
# had a destination bit that sent A straight into the operand vregs (the A
# bypass); alignment increment 1 removed it, so activations reach the vregs by
# `vld` from VMEM, as on MiniTPU.
DMA_SRC_B = 1

# ---- THE WRITE-BEFORE-READ CONTRACT ----
# `vmem` and `vreg` are NOT cleared by the hardware. Their `= 0`
# initialisers were removed to delete a 514-cycle memset, so at `ap_start`
# each holds whatever the previous invocation (or power-up) left there. The
# guarantee moved from the hardware to the program: every vreg row that
# `vmatload`, `vmatpush`, `vadd`, `vrelu` or `mvout` reads must have been
# written earlier in the SAME program -- by a `vld` of a VMEM row a `dma_ld`
# wrote, a `vmatpop`, a `vadd` or a `vrelu`. `vld` is a pure copy and MAY copy
# an unwritten VMEM row, but the copy is then unwritten too.
#
# ---- THE VREG DISTANCE CONTRACT ----
# `vpu` runs at II=1 because `schedule()` tells Vitis there is no carried
# dependence through `vreg` (`s.dependence`, limitations register item 21).
# That is true only if no row is read too soon after it was written: counting
# `vpu` iterations -- one per row of every instruction it runs (`vld`,
# `vmatload`, `vmatpush`, `vmatpop`, `vrelu`, `mvout`), two per `vadd` row
# (first source on the even one; second source and the write on the odd one)
# -- a read of a vreg row must come at least VR_RAW_DIST iterations after the
# write it depends on. Every GEMM satisfies it with room to spare, and
# `check_program` rejects any program that does not, exactly as it rejects an
# unwritten read. The contract was measured on v1's `accu` (a read 1 or 2
# iterations after the write returned the OLD row in RTL; 3, 4, 5 were
# exact), and moved here with the file in alignment increment 3;
# `TPU_TB=stress` cosim runs `ar_distance_program(VR_RAW_DIST)` on every
# build, so a re-synthesis that widened the window fails there.
#
# `rbA`/`rbB` (dma_ld's burst buffers) and `ib` (the sequencer's program
# buffer) are also unzeroed but cannot be read early: `ib` is filled by an
# unconditional IMEM_SIZE-word burst, and the A/B spans are computed by
# `assemble()` from the same resolved trace that names the rows `dma_ld` reads.
#
# `check_program()` below enforces all of this statically, and `assemble()`
# calls it, so a program that violates the contract cannot be assembled.


LOOP_DEPTH = 4                 # nesting levels, as MiniTPU's loop stack
IWORDS = 2                     # an instruction is two 64-bit words


AGU_TERMS = 3                  # address terms per instruction
AGU_F0, AGU_F1, AGU_F2, AGU_F3 = 1, 2, 3, 4   # term targets (0 = unused)


def enc_agu(*terms):
    """The second instruction word: up to `AGU_TERMS` address terms.

    Each term is `(target, level, stride)` and resolves to
    `field[target] += iv[level] * stride`, so an address can be relative to any
    enclosing loop's induction variable. Terms name their target rather than
    being fixed one-per-field, because a single field often needs two: the
    weight `vld` inside the k loop is offset by both the n tile and the k tile,
    `B_VM + nb*MAXDIM + kb*T`, and a one-term-per-field encoding cannot say it.

    Without this a loop body would reissue identical addresses every iteration
    and simply redo the same work, which is why MiniTPU exports its induction
    variables to `sequencer_agu_resolve` instead of keeping them in the stack.

    Field widths carry a spare bit each: a slice used to extract to a signed
    `ap_int<N>` (see the encoding note above; fixed since, the rule is kept), so target is 4 bits for 0..4,
    level 3 bits for 0..3, stride 12 bits for 0..2047."""
    assert len(terms) <= AGU_TERMS, f"at most {AGU_TERMS} address terms"
    w = 0
    for i, (target, level, stride) in enumerate(terms):
        assert 0 <= target <= 4 and 0 <= level < LOOP_DEPTH
        assert 0 <= stride < (1 << 11), f"stride {stride} does not fit"
        base = 19 * i
        w |= (target << base) | (level << (base + 4)) | (stride << (base + 7))
    return w


def enc(op, f0=0, f1=0, f2=0, f3=0, nr=0):
    """Assemble one instruction word. The compiler backend that lowers a TOSA
    matmul into these lives only on `chia-codesign`, so programs are written
    against this encoder -- by hand in `gemm_program_handwritten()` below, or
    through the loop-nest generator in `isa_dsl.py`, which derives the AGU
    levels from nesting instead of having them typed."""
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
# T >= 4 so that a packed word has room for the two 16-bit counts `vru` sends
# down `wcol` to the array (`mm` count, wavefront rows).
assert T >= 4, "a packed operand word must be at least 32 bits"
# T is the one parameter that changes the *shape* of the generated region:
# the array is T*T kernel instances and the chains are T and T*T stream
# arrays, so T=16 is 262 instances and ~800 streams. That was unrunnable
# until the simulator's OpenMP team was sized to the section count
# (docs/source/developer/limitations.rst, item 11); before that fix it hung with no output.
VW = T * 8                     # packed operand word: T int8 lanes
AW = T * 32                    # packed accumulator word: T int32 lanes

MAXDIM = int(os.environ.get("TPU_MAXDIM", 16))     # largest M, K, N supported
# A, B and C are **flat** at the region boundary, addressed `row * MAXDIM + col`.
# That is what DRAM is, and it is what makes `wrap_io=False` legal -- it refuses
# multi-dimensional arguments to nested kernels ("Top-level multi-dimensional
# arrays are linearized to 1D pointers"). The design is built that way, and the
# reason is the DMA note at the top of this file: with `wrap_io=True` Allo
# hoists every argument into a local buffer before the region starts, sized to
# the *declared* array rather than to the shape being run, and that copy was
# 907 of the 1586 cycles at 16x16x16 and 90% of them at 4x4x4.
# Memory sizes grow with the problem the build admits: the GEMM layout needs
# 2 * MAXDIM^2 / T rows of VMEM and of vregs (A and B, one column block per
# MAXDIM rows), and the test programs up to about 8 * MAXDIM accumulator
# rows. The defaults are the v1 sizes wherever those suffice, so the T=4,
# MAXDIM=16 build is unchanged.
_LAYOUT = 2 * (MAXDIM // T) * MAXDIM
VMEM_ROWS = int(os.environ.get("TPU_VMEM", max(512, 2 * _LAYOUT)))
# one vreg file: the GEMM layout (A, B), the accumulator and a popped tile,
# and the test programs' regions (about 8 * MAXDIM rows past 10)
NVREG = int(os.environ.get("TPU_NVREG",
                           max(256, _LAYOUT + 2 * MAXDIM + 8, 10 + 8 * MAXDIM)))
QD = int(os.environ.get("TPU_QD", 8))              # stream depth
OUTQ = 64                      # the array's output FIFO, rows (MiniTPU's)

MAXROWS = 127                                      # `nr` is 8 bits, top bit spare
NHDR = 8                       # imem[0:NHDR] is the header, instructions follow

# Instruction slots. Sized to the longest program shipped, not to a round
# number: the sequencer's prefetch is `IMEM_SIZE` words long whatever the
# program, so every unused slot is startup time -- the same arithmetic as under
# `wrap_io=True`, which copied the declared length for the same reason.
_KB = MAXDIM // T
# With control flow the program is O(nesting), not O(tiles): the looped GEMM is
# at most 14 instructions (with relu) at every shape, where the unrolled one
# reaches 32 at 16x16x16. So imem is sized to the longest program shipped (the
# stress harness's random programs, up to `_MAX_STATIC`) rather than to the
# largest problem.
#
# This is not cosmetic. Measured when the prefetch moved one word per cycle:
# moving to a 2-word instruction format cost exactly +68 cycles at all five
# shapes, the 68 extra words it added -- the loop logic itself cost nothing.
# The prefetch now moves 8 words per cycle, so the 56 words cost 7 cycles.
_MAX_STATIC = 24               # longest program shipped, plus headroom
IMEM_SIZE = int(os.environ.get("TPU_IMEM", NHDR + IWORDS * _MAX_STATIC))

# VMEM and vreg layout. Fixed offsets in a fixed memory, sized for the
# largest supported shape rather than for the shape being run.
KB_MAX = MAXDIM // T           # column blocks in the widest matrix
assert MAXDIM % T == 0, "a DRAM row must be a whole number of packed words"
WPR = MAXDIM // T              # packed words per DRAM row
B_VM = 0                       # B words:  nb * MAXDIM + k  (weights)
A_VM = KB_MAX * MAXDIM         # A words in VMEM: kb * MAXDIM + m, after B
# ONE vreg file (increment 3): A, B, the accumulator and the popped k-tile
# are regions of it.
A_VR = 0                       # A rows:  kb * MAXDIM + m  (vld'd from VMEM)
B_VR = KB_MAX * MAXDIM         # B rows:  nb * MAXDIM + k  (vmatload's weights)
AR_C = 2 * KB_MAX * MAXDIM     # the accumulator, up to MAXDIM rows
AR_P = AR_C + MAXDIM + 1       # a popped k-tile, before its vadd
VR_RAW_DIST = 4                # see THE VREG DISTANCE CONTRACT
assert VR_RAW_DIST <= T, "a T-row GEMM must satisfy the vreg contract"

# The scored shapes, scaled with the array: T, 2T, 3T, 4T x 4T x 2T and 4T
# cubed, dropped where they exceed MAXDIM. At T=4 they are exactly the five
# shapes v1 and Gemmini were measured at; nothing in the harness names a
# shape by number, so a constant that only holds at T=4 cannot hide.
# Rows of `isa_dsl.vector_program` by default: 8 where MAXDIM admits it.
VEC_M = max(2, min(8, MAXDIM // 2) // 2 * 2)
SCORED_SHAPES = [s for s in ((T, T, T), (2 * T,) * 3, (3 * T,) * 3,
                             (4 * T, 4 * T, 2 * T), (4 * T,) * 3)
                 if max(s) <= MAXDIM]
assert IMEM_SIZE % 8 == 0, "the program prefetch moves 8 words per iteration"


@df.region()
def tinytpu_isa(
    imem: UInt(64)[IMEM_SIZE],
    A: int8[MAXDIM * MAXDIM],
    B: int8[MAXDIM * MAXDIM],
    C: int8[MAXDIM * MAXDIM],
):
    # Control: the decoded instruction word, one point-to-point queue per unit.
    # The sequencer sends each unit only the instructions it executes, in
    # dataflow order (see `sequencer`), so a unit always holds its instruction
    # before the operands arrive, and queue depth is bounded by pipeline skew
    # rather than program length. (A broadcast sequencer put every unit in a
    # cycle with it; measured, the design then needed QD >= ~NPROG.)
    c_dld: Stream[UInt(64), QD]         # sequencer -> dma_ld
    c_vmu: Stream[UInt(64), QD]         # sequencer -> vmu
    c_vpu: Stream[UInt(64), QD]         # sequencer -> vpu
    c_dst: Stream[UInt(64), QD]         # sequencer -> dma_st

    # Data paths, all one packed word wide.
    dma2vm: Stream[UInt(VW), QD]        # dma_ld -> VMEM
    vm2vr: Stream[UInt(VW), QD]         # VMEM -> vregs  (vld)
    ac2sp: Stream[UInt(VW), QD]         # accumulator -> dma_st (clipped)

    # The array's streaming ports. Chains, never fan-out.
    wcol: Stream[UInt(VW), QD][T]       # counts, then weight words, down column 0
    wrow: Stream[UInt(VW), QD][T, T]    # ... then east along row i
    acol: Stream[UInt(VW), QD][T]       # activation words, down column 0
    # The weight-switch flag, beside each activation word down column 0. A
    # separate chain rather than bit VW of `acol`: a Stream of UInt(VW + 1)
    # (65 bits at T=8) corrupts the dataflow simulator's heap (72 and 128 bits
    # run, 96 hangs) -- see docs/source/designs/alignment.rst.
    afl: Stream[UInt(8), QD][T]
    a_fwd: Stream[UInt(16), QD][T, T]   # one lane and the flag, east
    p_fwd: Stream[int32, QD][T, T]      # partial sums, south
    cw: Stream[UInt(AW), QD][T - 1]     # bottom row packs T psums going east
    # The array's OUTPUT FIFO: rows pushed and not yet popped wait here.
    # MiniTPU's holds 64 rows; `check_program` refuses a program that leaves
    # more than OUTQ rows un-popped, which under back-pressure would be a
    # deadlock where MiniTPU drops results.
    mxo: Stream[UInt(AW), OUTQ]
    # wld(i, j) -> pe(i, j): the weight lane per `vmatload`. Depth 4 holds
    # the pending weights -- the next load's weight is latched while this one
    # computes (MiniTPU's two banks, Gemmini's c1/c2 double buffer).
    wq: Stream[UInt(32), 4][T, T]

    @df.kernel(mapping=[1], args=[imem])
    def sequencer(l_imem: UInt(64)[IMEM_SIZE]):
        """Fetch, decode, resolve addresses, dispatch.

        **This unit has a program counter and a loop stack.** `loop`/`endloop`
        bound a body and the PC branches backward, so the *static* program is
        O(nesting) while the *dynamic* instruction stream is O(tiles). Modelled
        on MiniTPU's `sequencer_loop_ctrl`: a LIFO of `{body_start, iv, trip}`
        frames, `LOOP_DEPTH` levels deep.

        **Addresses resolve here, not in the units.** Each instruction's second
        word carries up to three `(level, stride)` terms, and this loop adds
        `iv[level] * stride` to the corresponding field before dispatching, so
        every other unit receives a resolved word in the format it decodes.

        **Unit-specific copies of a word.** Two units get a rewritten copy so
        that their own flat row loops read their work count straight out of
        `nr` -- a row count that depends on the decoded opcode inside a flat
        loop closes that loop at `Final II = 2` on the counter (a carried
        dependence), measured in both `vmu` and `accu`:

          * `vmu`'s copy of an `mm` carries `nr = T + 1` (one header word and
            T weight words) and the array's row count in `f1`;
          * `accu`'s copy of a `vadd` carries `nr = 2 * rows` (it takes two
            iterations per row, see `accu`). `nr` is 8 bits and a slice is
            unsigned, so 2 * 127 fits.

        The dispatch order is dataflow order and that is load-bearing: if this
        blocks on a full queue, every unit upstream of that one already holds
        the instruction and can keep producing what the blocked unit waits for.
        Dispatching downstream-first deadlocks."""
        # --- the program comes on-chip first, 8 words per cycle ---
        # `l_imem` is an `m_axi` port (the design is `wrap_io=False`). Fetched
        # in place, `l_imem[NHDR + pc * IWORDS]` is a data-dependent address
        # Vitis can only burst two words at a time -- `Final II = 13` on the
        # fetch loop. A constant-trip contiguous copy is one real AXI burst
        # instead, and every fetch after it is an on-chip read.
        #
        # gmem0 is 512 bits wide (`m_axi_max_widen_bitwidth 512` with
        # `align_value(64)`), so each iteration moves 8 words into an `ib`
        # cyclically partitioned by 8 (`schedule()`): the 56-word prefix is 7
        # iterations, not 56. One word per iteration cost 52 cycles at every
        # shape (the `v_imem8` variant, gap attribution).
        ib: UInt(64)[IMEM_SIZE]
        for i in range(IMEM_SIZE // 8):
            with allo.meta_for(8) as e8:
                ib[8 * i + e8] = l_imem[8 * i + e8]

        iw: UInt(64) = ib[0]
        n_instr: int32 = iw[0:16]

        c_dld.put(ib[1])
        c_dld.put(ib[7])
        c_vmu.put(ib[2])
        c_vpu.put(ib[3])
        c_vpu.put(ib[4])
        c_dst.put(ib[6])

        lp_start: int32[LOOP_DEPTH] = 0
        lp_iv: int32[LOOP_DEPTH] = 0
        lp_trip: int32[LOOP_DEPTH] = 0
        iv_now: int32[LOOP_DEPTH] = 0
        sp: int32 = 0
        pc: int32 = 0
        running: int32 = 1

        while running == 1:
            w0: UInt(64) = ib[NHDR + pc * IWORDS]
            w1: UInt(64) = ib[NHDR + pc * IWORDS + 1]
            op: int32 = w0[0:6]
            nr: int32 = w0[54:62]

            if op == OP_LOOP:
                lp_start[sp] = pc + 1
                lp_iv[sp] = 0
                lp_trip[sp] = nr
                iv_now[sp] = 0
                sp += 1
                pc += 1
            elif op == OP_ENDLOOP:
                nxt: int32 = lp_iv[sp - 1] + 1
                if nxt < lp_trip[sp - 1]:
                    lp_iv[sp - 1] = nxt
                    iv_now[sp - 1] = nxt
                    pc = lp_start[sp - 1]
                else:
                    sp -= 1
                    pc += 1
            else:
                # --- resolve the address terms against the live IVs ---
                f0: int32 = w0[6:18]
                f1: int32 = w0[18:30]
                f2: int32 = w0[30:42]
                f3: int32 = w0[42:54]
                with allo.meta_for(AGU_TERMS) as _t:
                    tw: int32 = w1[19 * _t : 19 * _t + 4]
                    lw: int32 = w1[19 * _t + 4 : 19 * _t + 7]
                    sw: int32 = w1[19 * _t + 7 : 19 * _t + 19]
                    d: int32 = iv_now[lw] * sw
                    if tw == AGU_F0:
                        f0 = f0 + d
                    if tw == AGU_F1:
                        f1 = f1 + d
                    if tw == AGU_F2:
                        f2 = f2 + d
                    if tw == AGU_F3:
                        f3 = f3 + d

                rw: UInt(64) = w0
                rw[6:18] = f0
                rw[18:30] = f1
                rw[30:42] = f2
                rw[42:54] = f3

                if op == OP_DMA_LD:
                    c_dld.put(rw)
                    c_vmu.put(rw)
                if op == OP_VLD:
                    c_vmu.put(rw)
                    c_vpu.put(rw)
                if op == OP_VMATLOAD:
                    c_vpu.put(rw)
                if op == OP_VMATPUSH:
                    c_vpu.put(rw)
                if op == OP_VMATPOP:
                    c_vpu.put(rw)
                if op == OP_VADD:
                    wv: UInt(64) = rw
                    wv[54:62] = nr * 2
                    c_vpu.put(wv)
                if op == OP_VRELU:
                    c_vpu.put(rw)
                if op == OP_MVOUT:
                    c_vpu.put(rw)
                    c_dst.put(rw)
                pc += 1

            if pc >= n_instr:
                running = 0

    @df.kernel(mapping=[1], args=[A, B])
    def dma_ld(lA: int8[MAXDIM * MAXDIM], lB: int8[MAXDIM * MAXDIM]):
        """DRAM -> VMEM or operand vregs. Sole reader of A and B.

        Split from the store unit deliberately: a single unit doing both put
        `vmu` and the DMA in a two-process cycle, which deadlocked with every
        body variant tried. Gemmini splits the same way (`LoadController` /
        `StoreController`).

        **Two variable-length bursts, then every instruction runs from BRAM.**
        A per-row strided read, `lA[(f1 + r) * MAXDIM + f2 * T + e]`, is a
        separate four-beat AXI transaction per row (`[HLS 214-115] Multiple
        burst reads of length 4`, `Final II = 4`); a contiguous sweep with a
        runtime trip count is one real burst. So the unit opens with one burst
        per operand matrix covering exactly the DRAM rows the program names --
        `na`/`nb` come from the assembler -- packing T lanes per word as it
        goes, and each row afterwards is one BRAM read.

        **Where a row goes is the instruction's.** `f0` bit 0 picks the source
        matrix, bit 1 the destination: the VMEM (`dma2vm`), or the
        operand vregs directly (`dma2vr`). The shipped GEMM sends A straight to
        the vregs, so activations skip the `vmem -> vld -> vr` double handling
        (-64 cycles at 16x16x16, the `v_design` variant); B goes to the
        VMEM, where `vmu` streams it into the array as weights.

        One flat loop over rows, the instruction fetched on the iteration that
        needs it (the ROW-FLATTENING note above); the burst loops sit outside
        it on purpose."""
        nw: UInt(64) = c_dld.get()
        n_row: int32 = nw[0:16]
        sw: UInt(64) = c_dld.get()  # the DRAM row span of each source matrix
        na: int32 = sw[0:16]
        nb: int32 = sw[16:32]

        # One burst per matrix, each covering exactly that matrix's own span.
        # Merging the two into one loop bounded by `max(na, nb)` was measured
        # and moved nothing: the bursts are hidden behind the prefetch.
        rbA: UInt(VW)[MAXDIM * WPR]
        rbB: UInt(VW)[MAXDIM * WPR]
        for ia in range(na * WPR):
            pa: UInt(VW) = 0
            with allo.meta_for(T) as e:
                av: int8 = lA[ia * T + e]
                pa[8 * e : 8 * (e + 1)] = av
            rbA[ia] = pa
        for ic in range(nb * WPR):
            pb: UInt(VW) = 0
            with allo.meta_for(T) as e2:
                bv: int8 = lB[ic * T + e2]
                pb[8 * e2 : 8 * (e2 + 1)] = bv
            rbB[ic] = pb

        f0: int32 = 0
        f1: int32 = 0
        f2: int32 = 0
        cnt: int32 = 0
        r: int32 = -1               # advanced at the TOP: see the II note
        for x in range(n_row):
            r += 1
            if r >= cnt:
                # Only `dma_ld` reaches this queue, so there is no opcode test.
                w0: UInt(64) = c_dld.get()
                f0 = w0[6:18]
                f1 = w0[18:30]
                f2 = w0[30:42]
                cnt = w0[54:62]
                r = 0
            pw: UInt(VW) = 0
            if (f0 & DMA_SRC_B) == 0:
                pw = rbA[(f1 + r) * WPR + f2]
            else:
                pw = rbB[(f1 + r) * WPR + f2]
            dma2vm.put(pw)

    @df.kernel(mapping=[1])
    def vmu():
        """VMEM, the only unit that owns it.

        Pure SIMD access: an address names a whole `UInt(T*8)` row and there is
        no way to address a lane. Written by `dma_ld` (the only place DMA
        writes) and read by `vld` (VMEM -> vregs). Since increment 2 it no
        longer feeds the array: MiniTPU's `vmatload` takes its weights from
        VREGs, so B travels VMEM -> `vld` -> vregs -> `vmatload` like A.

        One flat row loop, ONE `vmem` access per iteration."""
        vmem: UInt(VW)[VMEM_ROWS]
        nw: UInt(64) = c_vmu.get()
        n_row: int32 = nw[0:16]
        op: int32 = 0
        f1: int32 = 0
        f3: int32 = 0
        cnt: int32 = 0
        r: int32 = -1               # advanced at the TOP: see the II note
        for x in range(n_row):
            r += 1
            if r >= cnt:
                w0: UInt(64) = c_vmu.get()
                op = w0[0:6]
                f1 = w0[18:30]
                f3 = w0[42:54]
                cnt = w0[54:62]
                r = 0
            if op == OP_DMA_LD:
                vmem[f3 + r] = dma2vm.get()
            else:
                vm2vr.put(vmem[f1 + r])

    @df.kernel(mapping=[1])
    def vpu():
        """THE VREG FILE -- one file, one owner -- and every unit that reads or
        writes it: MiniTPU's V slot (`vadd`, `vrelu`), the VREG side of its M
        slot (`vmatload`/`vmatpush` read, `vmatpop` writes) and of its MEM
        slot (`vld` writes; `mvout`, until `vst` exists, reads).

        **One file, int32 lanes.** A row is T int32 lanes (`UInt(AW)`): wide
        enough for a popped result, and `vld` sign-extends an int8 VMEM row
        into it. `vmatload`/`vmatpush` send the array the low 8 bits of each
        lane, which is exact for anything a `vld` wrote. (MiniTPU has one
        type, BF16, so its file needs no such rule.)

        **MiniTPU has three read ports and one write port; this process does
        one row of one instruction per iteration**, with ONE `vreg` read and
        ONE write, so the ports are a budget, not concurrency: V, M and MEM
        work that MiniTPU overlaps in one bundle is serialised here. That is
        the cost this increment measures. `vadd` takes two iterations per row
        (first source on the even one, second source and the write on the odd
        one), as `accu` did.

        **This process is on a CYCLE**: it feeds the array and drains it
        (`acol`/`wcol` out, `mxo` in). A pop waits for rows an earlier push in
        the same pipelined loop put; under Vitis's default stall pipeline the
        blocked read freezes that put and the RTL deadlocks.
        `schedule()` pipelines this loop `style=flp` (flushable), which keeps
        earlier iterations draining (`align_probes/probe_cycle.py`), and
        `cosim.py` runs the C model threaded (`threaded_csim.py`).

        **The dependence claim moves here from `accu`**: `schedule()` tells
        Vitis there is no carried dependence through `vreg`, and
        `check_program` makes it true by keeping every read at least
        `VR_RAW_DIST` vpu iterations after the write it depends on -- now for
        every reader: `vmatload`, `vmatpush`, `vadd`, `vrelu`, `mvout`."""
        nw: UInt(64) = c_vpu.get()
        n_it: int32 = nw[0:16]
        mw: UInt(64) = c_vpu.get()
        nmo: UInt(VW) = 0
        nmo[0:32] = mw[0:32]        # load count | pushed rows << 16
        wcol[0].put(nmo)
        vreg: UInt(AW)[NVREG]
        op: int32 = 0
        f0: int32 = 0
        f1: int32 = 0
        f2: int32 = 0
        cnt: int32 = 0
        pend: int32 = 0             # a vmatload no push has used yet
        r: int32 = -1               # advanced at the TOP: see the II note
        xr: UInt(AW) = 0            # vadd's first operand, held for one row
        for x in range(n_it):
            r += 1
            if r >= cnt:
                w0: UInt(64) = c_vpu.get()
                op = w0[0:6]
                f0 = w0[6:18]
                f1 = w0[18:30]
                f2 = w0[30:42]
                cnt = w0[54:62]
                r = 0
            rr: int32 = r
            ph: int32 = 0
            if op == OP_VADD:
                rr = r >> 1
                ph = r - (rr << 1)
            ra: int32 = f1 + rr
            wa: int32 = f0 + rr
            if op != OP_VADD:
                if op != OP_VRELU:
                    ra = f0 + rr
            if op == OP_VADD:
                if ph == 1:
                    ra = f2 + rr
            rv: UInt(AW) = vreg[ra]
            z: UInt(AW) = 0
            dw: int32 = 1
            if op == OP_VLD:
                v8: UInt(VW) = vm2vr.get()
                with allo.meta_for(T) as e0:
                    b8: int8 = v8[8 * e0 : 8 * (e0 + 1)]
                    b32: int32 = b8
                    z[32 * e0 : 32 * (e0 + 1)] = b32
            elif op == OP_VMATPOP:
                z = mxo.get()
            elif op == OP_VADD:
                if ph == 0:
                    xr = rv
                    dw = 0
                else:
                    with allo.meta_for(T) as e2:
                        xe: int32 = xr[32 * e2 : 32 * (e2 + 1)]
                        ye: int32 = rv[32 * e2 : 32 * (e2 + 1)]
                        xy: int32 = xe + ye
                        z[32 * e2 : 32 * (e2 + 1)] = xy
            elif op == OP_VRELU:
                with allo.meta_for(T) as e3:
                    ue: int32 = rv[32 * e3 : 32 * (e3 + 1)]
                    re: int32 = ue
                    if re < 0:
                        re = 0
                    z[32 * e3 : 32 * (e3 + 1)] = re
            elif op == OP_MVOUT:
                dw = 0
                ow: UInt(VW) = 0
                with allo.meta_for(T) as e4:
                    te: int32 = rv[32 * e4 : 32 * (e4 + 1)]
                    if te > 127:
                        te = 127
                    if te < -128:
                        te = -128
                    tc: int8 = te
                    ow[8 * e4 : 8 * (e4 + 1)] = tc
                ac2sp.put(ow)
            else:
                # vmatload / vmatpush: the low 8 bits of each lane
                dw = 0
                nv: UInt(VW) = 0
                with allo.meta_for(T) as e5:
                    nv[8 * e5 : 8 * (e5 + 1)] = rv[32 * e5 : 32 * e5 + 8]
                if op == OP_VMATLOAD:
                    wcol[0].put(nv)
                    pend = 1
                else:
                    acol[0].put(nv)
                    fw: UInt(8) = pend
                    afl[0].put(fw)
                    pend = 0
            if dw == 1:
                vreg[wa] = z

    @df.kernel(mapping=[T, T])
    def wld():
        """The weight half of a processing element: one per PE.

        Walks the weight chain -- down column 0, then east along the row --
        and hands its own PE one weight lane per `vmatload` through `wq[i,
        j]`. PE row i keeps the i-th of the T words a `vmatload` sends, so PE
        (i, j) holds W[i][j] and the array computes Y = X W, as MiniTPU's
        does. `wq` is the pending-weight queue: `vmatload` n+1's weight is
        latched while the PE computes with n's (MiniTPU's two pending banks,
        Gemmini's c1/c2 double buffer)."""
        i, j = df.get_pid()
        nmw: UInt(VW) = 0
        with allo.meta_if(j == 0):
            nmw = wcol[i].get()
            with allo.meta_if(i != T - 1):
                wcol[i + 1].put(nmw)
        with allo.meta_else():
            nmw = wrow[i, j - 1].get()
        with allo.meta_if(j != T - 1):
            wrow[i, j].put(nmw)
        nld: int32 = nmw[0:16]
        tq: UInt(32) = 0
        tq[0:16] = nmw[16:32]       # the PE's own trip count: pushed rows
        wq[i, j].put(tq)
        for c in range(nld):
            ww: UInt(VW) = 0
            with allo.meta_if(j == 0):
                ww = wcol[i].get()
                with allo.meta_for(T - 1 - i) as _f:
                    wcol[i + 1].put(wcol[i].get())
            with allo.meta_else():
                ww = wrow[i, j - 1].get()
            with allo.meta_if(j != T - 1):
                wrow[i, j].put(ww)
            q: UInt(32) = 0
            q[0:8] = ww[8 * j : 8 * (j + 1)]
            wq[i, j].put(q)

    @df.kernel(mapping=[T, T])
    def pe():
        """One processing element: weight-stationary MAC, and it decodes
        nothing.

        ONE flat loop over every pushed row. A row whose flag bit is set is
        the first after a `vmatload`, and the PE takes its next weight from
        `wq` before using it (MiniTPU: the switch rides the wavefront).

        The MAC has **no loop-carried value**: tap a lane, take the partial
        sum from the north, multiply-add, pass both on. The partial sum starts
        at 0 in row 0, so nothing accumulates across pushes -- a k-tile's
        result leaves the array and deeper contractions are summed by `vadd`
        (MiniTPU's split)."""
        i, j = df.get_pid()
        tq: UInt(32) = wq[i, j].get()
        nt: int32 = tq[0:16]
        w: int8 = 0
        for x in range(nt):
            a: int8 = 0
            fl: UInt(1) = 0
            with allo.meta_if(j == 0):
                aw: UInt(VW) = acol[i].get()
                fw: UInt(8) = afl[i].get()
                with allo.meta_if(i != T - 1):
                    acol[i + 1].put(aw)
                    afl[i + 1].put(fw)
                a = aw[8 * i : 8 * (i + 1)]
                fl = fw[0:1]
            with allo.meta_else():
                af: UInt(16) = a_fwd[i, j - 1].get()
                a = af[0:8]
                fl = af[8:9]
            if fl == 1:
                q: UInt(32) = wq[i, j].get()
                w = q[0:8]
            p: int32 = 0
            with allo.meta_if(i > 0):
                p = p_fwd[i - 1, j].get()
            # int8 x int8 -> int16 keeps this a narrow multiply; the
            # operands bound the product at 128*128 = 16384.
            av: int16 = a
            wv: int16 = w
            o: int32 = p + av * wv
            with allo.meta_if(i != T - 1):
                p_fwd[i, j].put(o)
            with allo.meta_else():
                # The bottom row assembles the packed result word as it
                # travels east, so the pop sees whole words and there is no
                # T-way fan-in.
                cv: UInt(AW) = 0
                with allo.meta_if(j > 0):
                    cv = cw[j - 1].get()
                cv[32 * j : 32 * (j + 1)] = o
                with allo.meta_if(j == T - 1):
                    mxo.put(cv)
                with allo.meta_else():
                    cw[j].put(cv)
            with allo.meta_if(j != T - 1):
                ao: UInt(16) = 0
                ao[0:8] = a
                ao[8:9] = fl
                a_fwd[i, j].put(ao)

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
        n_row: int32 = nw[0:16]
        f1: int32 = 0
        f2: int32 = 0
        cnt: int32 = 0
        r: int32 = -1               # advanced at the TOP: see the II note
        for x in range(n_row):
            r += 1
            if r >= cnt:
                # Only `mvout` reaches this queue, so there is no opcode test.
                w0: UInt(64) = c_dst.get()
                f1 = w0[18:30]
                f2 = w0[30:42]
                cnt = w0[54:62]
                r = 0
            qw: UInt(VW) = ac2sp.get()
            with allo.meta_for(T) as e:
                ov: int8 = qw[8 * e : 8 * (e + 1)]
                lC[(f1 + r) * MAXDIM + f2 * T + e] = ov

def gemm_program_handwritten(M, K, N, relu=False):
    """Tiled GEMM as a program with **control flow**, hand-emitted.

    **Superseded as the shipped program by `isa_dsl.gemm_program`, and kept as
    its reference.** The two must emit bit-identical words at every shape --
    `isa_dsl.assert_matches_handwritten` checks every word of every instruction
    and `bench_isa.py` runs it. What the generated form removes is the AGU
    *levels* below: hand-typed integers that have to agree with where the
    `loop`/`endloop` pairs happen to sit.

    The program is MiniTPU's GEMM on this machine (alignment increment 2):
    A and B go DRAM -> VMEM -> `vld` -> vregs; each (n, k) tile is a
    `vmatload` of T weight rows, a `vmatpush` of the M activation rows and a
    `vmatpop` of the M result rows; k-tiles after the first are popped into a
    scratch region and summed with `vadd` (MiniTPU: the array accumulates only
    T deep, deeper contractions are `vadd`s in the VPU). The first k-tile is
    peeled so it can pop straight into the accumulator region.
    """
    assert M <= MAXDIM and K <= MAXDIM and N <= MAXDIM, (
        f"{M}x{K}x{N} exceeds the built MAXDIM={MAXDIM}")
    assert M % T == 0 and K % T == 0 and N % T == 0
    assert M <= MAXROWS and K <= MAXROWS
    Kt, Nt = K // T, N // T
    p = []
    ins = lambda w, agu=0: p.append((w, agu))

    # --- A: one dma_ld per column block into VMEM, then vld into the vregs ---
    ins(enc(OP_LOOP, nr=Kt))
    ins(enc(OP_DMA_LD, f0=0, f1=0, f2=0, f3=A_VM, nr=M),
        enc_agu((AGU_F2, 0, 1), (AGU_F3, 0, MAXDIM)))
    ins(enc(OP_VLD, f0=A_VR, f1=A_VM, nr=M),
        enc_agu((AGU_F0, 0, MAXDIM), (AGU_F1, 0, MAXDIM)))
    ins(enc(OP_ENDLOOP))
    # --- B: the same, one per column block ---
    ins(enc(OP_LOOP, nr=Nt))
    ins(enc(OP_DMA_LD, f0=DMA_SRC_B, f1=0, f2=0, f3=B_VM, nr=K),
        enc_agu((AGU_F2, 0, 1), (AGU_F3, 0, MAXDIM)))
    ins(enc(OP_VLD, f0=B_VR, f1=B_VM, nr=K),
        enc_agu((AGU_F0, 0, MAXDIM), (AGU_F1, 0, MAXDIM)))
    ins(enc(OP_ENDLOOP))

    # --- the output loop: level 0 is nb, level 1 is kb ---
    ins(enc(OP_LOOP, nr=Nt))
    #   peeled first k-tile: weights B_VR + nb*MAXDIM, popped into AR_C
    ins(enc(OP_VMATLOAD, f0=B_VR, nr=T), enc_agu((AGU_F0, 0, MAXDIM)))
    ins(enc(OP_VMATPUSH, f0=A_VR, nr=M))
    ins(enc(OP_VMATPOP, f0=AR_C, nr=M))
    if Kt > 1:
        ins(enc(OP_LOOP, nr=Kt - 1))
        #   f0 needs BOTH tiles: B_VR + nb*MAXDIM + (kb+1)*T
        ins(enc(OP_VMATLOAD, f0=B_VR + T, nr=T),
            enc_agu((AGU_F0, 0, MAXDIM), (AGU_F0, 1, T)))
        ins(enc(OP_VMATPUSH, f0=A_VR + MAXDIM, nr=M),
            enc_agu((AGU_F0, 1, MAXDIM)))
        ins(enc(OP_VMATPOP, f0=AR_P, nr=M))
        ins(enc(OP_VADD, f0=AR_C, f1=AR_C, f2=AR_P, nr=M))
        ins(enc(OP_ENDLOOP))
    if relu:
        ins(enc(OP_VRELU, f0=AR_C, f1=AR_C, nr=M))
    ins(enc(OP_MVOUT, f0=AR_C, f1=0, f2=0, nr=M),
        enc_agu((AGU_F2, 0, 1)))
    ins(enc(OP_ENDLOOP))
    return p


def gemm_program_flat(M, K, N, relu=False):
    """The fully unrolled form, kept as the differential reference.

    Every instruction carries absolute addresses and there is no control flow,
    so this is what the looped program must reproduce exactly. Keeping it is
    the cheap way to test a PC and an AGU: same shape, same result, two
    encodings."""
    assert M <= MAXDIM and K <= MAXDIM and N <= MAXDIM
    assert M % T == 0 and K % T == 0 and N % T == 0
    assert M <= MAXROWS and K <= MAXROWS
    Kt, Nt = K // T, N // T
    p = []
    for kb in range(Kt):
        p.append((enc(OP_DMA_LD, f0=0, f1=0, f2=kb,
                      f3=A_VM + kb * MAXDIM, nr=M), 0))
        p.append((enc(OP_VLD, f0=A_VR + kb * MAXDIM, f1=A_VM + kb * MAXDIM,
                      nr=M), 0))
    for nb in range(Nt):
        p.append((enc(OP_DMA_LD, f0=DMA_SRC_B, f1=0, f2=nb,
                      f3=B_VM + nb * MAXDIM, nr=K), 0))
        p.append((enc(OP_VLD, f0=B_VR + nb * MAXDIM, f1=B_VM + nb * MAXDIM,
                      nr=K), 0))
    for nb in range(Nt):
        for kb in range(Kt):
            p.append((enc(OP_VMATLOAD, f0=B_VR + nb * MAXDIM + kb * T,
                          nr=T), 0))
            p.append((enc(OP_VMATPUSH, f0=A_VR + kb * MAXDIM, nr=M), 0))
            p.append((enc(OP_VMATPOP, f0=(AR_P if kb else AR_C), nr=M), 0))
            if kb:
                p.append((enc(OP_VADD, f0=AR_C, f1=AR_C, f2=AR_P, nr=M), 0))
        if relu:
            p.append((enc(OP_VRELU, f0=AR_C, f1=AR_C, nr=M), 0))
        p.append((enc(OP_MVOUT, f0=AR_C, f1=0, f2=nb, nr=M), 0))
    return p


def vadd_program(M, K, N):
    """A program for the vector unit itself: `relu(2 * (A @ B))` on the first
    output tile, as two pushes of the same tile popped into two regions,
    added, ReLU'd and retired."""
    return [(w, 0) for w in [
        enc(OP_DMA_LD, f0=0, f1=0, f2=0, f3=A_VM, nr=M),
        enc(OP_VLD, f0=A_VR, f1=A_VM, nr=M),
        enc(OP_DMA_LD, f0=DMA_SRC_B, f1=0, f2=0, f3=B_VM, nr=T),
        enc(OP_VLD, f0=B_VR, f1=B_VM, nr=T),
        enc(OP_VMATLOAD, f0=B_VR, nr=T),
        enc(OP_VMATPUSH, f0=A_VR, nr=M),
        enc(OP_VMATPOP, f0=AR_C, nr=M),
        enc(OP_VMATLOAD, f0=B_VR, nr=T),
        enc(OP_VMATPUSH, f0=A_VR, nr=M),
        enc(OP_VMATPOP, f0=AR_P, nr=M),
        enc(OP_VADD, f0=AR_C, f1=AR_C, f2=AR_P, nr=M),
        enc(OP_VRELU, f0=AR_C, f1=AR_C, nr=M),
        enc(OP_MVOUT, f0=AR_C, f1=0, f2=0, nr=M)]]


def expand(prog):
    """Run the program's control flow at assembly time, yielding one
    `(opcode, row count, f0, f1, f2, f3)` tuple per instruction the sequencer
    will actually issue -- with the AGU resolved exactly as the sequencer
    resolves it.

    The units loop over *their own* work count, so the header must carry the
    DYNAMIC count -- how much work each unit is really sent -- not the static
    one. With a hardware loop those differ, and a unit promised more than it
    receives waits forever. This is the assembler's obligation, and it is the
    price of putting control flow in the sequencer.

    The row count comes along because the units now count ROWS rather than
    instructions (see the flattening note at the top of the file). `nr` is
    never an AGU target, so it is the same for every dynamic issue of a given
    static instruction and can be read straight off the encoding.

    **The resolved address fields come along because `dma_ld` now bursts.** It
    needs to know, before the first instruction arrives, how many DRAM rows of
    A and of B the program will name, and `f1` is an AGU target in the general
    case -- so the assembler has to run the same resolution the sequencer runs
    rather than read `f1` off the static encoding. Mirroring the sequencer here
    is also what makes `bench_isa`'s loop-vs-flat equivalence check strict: the
    two forms must now agree on every resolved address, not just on the opcode
    and row-count stream.
    """
    return [e[2:] for e in _trace(prog)]


def _trace(prog):
    """`expand`, with where each dynamic issue came from: yields
    `(pc, ivs, op, nr, f0, f1, f2, f3)`, `ivs` the live induction variables.
    The one place the assembler mirrors the sequencer's control flow."""
    pc = 0
    stack = []
    iv_now = [0] * LOOP_DEPTH
    guard = 0
    while pc < len(prog):
        guard += 1
        if guard > 1 << 22:
            raise AssertionError("program does not terminate")
        w0, w1 = prog[pc]
        op = w0 & 0x3F
        if op == OP_LOOP:
            trip = (w0 >> 54) & 0xFF
            iv_now[len(stack)] = 0
            stack.append([pc + 1, 0, trip])
            pc += 1
        elif op == OP_ENDLOOP:
            fr = stack[-1]
            fr[1] += 1
            if fr[1] < fr[2]:
                iv_now[len(stack) - 1] = fr[1]
                pc = fr[0]
            else:
                stack.pop()
                pc += 1
        else:
            f = [(w0 >> sh) & 0xFFF for sh in (6, 18, 30, 42)]
            for t in range(AGU_TERMS):
                base = 19 * t
                tw = (w1 >> base) & 0xF
                lw = (w1 >> (base + 4)) & 0x7
                st = (w1 >> (base + 7)) & 0xFFF
                if tw != 0:
                    f[tw - 1] += iv_now[lw] * st
            yield (pc, tuple(iv_now[:len(stack)]), op, (w0 >> 54) & 0xFF,
                   f[0], f[1], f[2], f[3])
            pc += 1


class ProgramError(ValueError):
    """A program the hardware would run to a wrong answer, or a hang."""


_OPNAME = {OP_NOP: "nop", OP_DMA_LD: "dma_ld", OP_DMA_ST: "dma_st",
           OP_VLD: "vld", OP_MM: "mm", OP_VADD: "vadd", OP_VRELU: "vrelu",
           OP_MVOUT: "mvout", OP_LOOP: "loop", OP_ENDLOOP: "endloop",
           OP_VMATLOAD: "vmatload", OP_VMATPUSH: "vmatpush",
           OP_VMATPOP: "vmatpop"}
_RETIRED = (OP_DMA_ST, OP_MM)


def check_program(prog):
    """Reject a program the hardware cannot run correctly. Called by `assemble`.

    **Static, and exact rather than conservative.** The program has no
    data-dependent control flow -- trip counts and AGU strides are instruction
    fields -- so the dynamic instruction stream is a pure function of the
    program text, and walking it (`_trace`, the same walk that computes the
    header) visits every issue the sequencer will make with the addresses it
    will resolve. There is no loop abstraction to get wrong.

    What it checks, per dynamic issue:

      * **write-before-read** on `vmem` and the vreg file -- the contract at
        the opcode table. Written-ness is tracked per row and propagated
        through `vld` (a copy of an unwritten row is unwritten); it is an
        error only where a value is *consumed*: `vmatload`, `vmatpush`,
        `vadd`, `vrelu` and `mvout` reading vregs.
      * **the vreg distance contract**: every vreg read comes at least
        `VR_RAW_DIST` `vpu` iterations after the write it depends on, which
        is what makes `schedule()`'s dependence claim on `vreg` true.
      * **bounds** on every memory and on `C`/`A`/`B`: an out-of-range row is
        silent corruption in RTL, not an exception.
      * **`nr >= 1`** on every data op. `dma_ld`, `vmu`, `vpu` and `dma_st` run
        one flat loop over the SUM of their rows and fetch an instruction
        whenever the row counter runs out, so a zero-row instruction is
        fetched as if it had one row -- it desynchronises the unit, it is not
        a no-op.
      * resolved fields below 2^11, the range `enc` admits (the encoding
        note at the top). The sequencer writes the resolved sum back into the
        12-bit field, so an AGU term can push a field past what `enc` would
        have accepted; this is the only place that can see it.
      * structure: balanced loops, depth <= LOOP_DEPTH, trip >= 1 (the
        sequencer is a do-while), AGU terms naming a loop that is open, no
        retired opcode, `dma_ld` f0 in {0, 1}.
      * **the array's queue**: a `vmatload` moves exactly T rows and its
        weights must be used by a `vmatpush` before the next `vmatload` (the
        PE switches on the first pushed row after a load); a `vmatpush` needs
        weights; a `vmatpop` may not pop more rows than are pushed and
        un-popped (it would wait forever); at most OUTQ rows may be
        un-popped (the output FIFO; MiniTPU drops the excess, this machine
        would deadlock); and every pushed row is popped by the end.

    Raises `ProgramError` naming the static instruction, the loop iteration,
    and the rows; returns None."""
    if not prog:
        raise ProgramError("empty program")
    depth = 0
    for pc, (w0, w1) in enumerate(prog):
        op = w0 & 0x3F
        name = _OPNAME.get(op)
        where = f"instruction {pc} ({name or f'opcode {op}'})"
        if name is None or op in _RETIRED:
            raise ProgramError(f"{where}: not an opcode this machine executes")
        if op == OP_LOOP:
            if depth >= LOOP_DEPTH:
                raise ProgramError(f"{where}: nesting exceeds LOOP_DEPTH={LOOP_DEPTH}")
            if (w0 >> 54) & 0xFF < 1:
                raise ProgramError(f"{where}: trip count 0 still runs the body once")
            depth += 1
        elif op == OP_ENDLOOP:
            if depth == 0:
                raise ProgramError(f"{where}: endloop with no open loop")
            depth -= 1
        for t in range(AGU_TERMS):
            tw = (w1 >> (19 * t)) & 0xF
            lw = (w1 >> (19 * t + 4)) & 0x7
            if tw == 0:
                continue
            if op in (OP_LOOP, OP_ENDLOOP, OP_NOP) or tw > 4 or lw >= depth:
                raise ProgramError(
                    f"{where}: AGU term {t} targets field {tw - 1} with loop "
                    f"level {lw}, but {depth} loop(s) are open here -- the "
                    f"sequencer would use a stale iv_now[{lw}]")
    if depth:
        raise ProgramError(f"{depth} loop(s) never closed")

    written = {"vmem": [False] * VMEM_ROWS, "vr": [False] * NVREG}
    vr_wrote = [-VR_RAW_DIST] * NVREG  # vpu iteration of each row's last write
    it = 0                            # vpu iterations issued so far
    q = {"loaded": False, "unused": False, "out": 0}   # the array's queue
    size = {"vmem": VMEM_ROWS, "vr": NVREG}

    for pc, ivs, op, nr, f0, f1, f2, f3 in _trace(prog):
        where = (f"instruction {pc} ({_OPNAME[op]}"
                 + (f", loop ivs {list(ivs)}" if ivs else "") + ")")

        def span(mem, base, n):
            if base < 0 or base + n > size[mem]:
                raise ProgramError(f"{where}: {mem} rows {base}..{base + n - 1} "
                                   f"outside 0..{size[mem] - 1}")
            return range(base, base + n)

        def need(mem, rows, what):
            bad = [r for r in rows if not written[mem][r]]
            if bad:
                raise ProgramError(
                    f"{where}: reads {mem} row(s) {bad} as {what} before any "
                    f"instruction wrote them. {mem} is not cleared by the "
                    f"hardware; see the write-before-read contract.")

        if op == OP_NOP:
            continue
        for v, fld in ((f0, "f0"), (f1, "f1"), (f2, "f2"), (f3, "f3")):
            if v >= 1 << 11:
                raise ProgramError(f"{where}: AGU-resolved {fld}={v} is "
                                   f"outside the 0..2047 range `enc` admits")
        if nr < 1:
            raise ProgramError(f"{where}: nr=0 desynchronises the unit's "
                               f"flat row loop; drop the instruction instead")

        def vr_read(row, at, what):
            # the distance contract: `at` is the vpu iteration of the read
            need("vr", [row], what)
            if at - vr_wrote[row] < VR_RAW_DIST:
                raise ProgramError(
                    f"{where}: reads vr row {row} as {what} "
                    f"{at - vr_wrote[row]} vpu iteration(s) after it was "
                    f"written; the vreg file's dependence claim needs "
                    f">= VR_RAW_DIST={VR_RAW_DIST} (see THE VREG "
                    f"DISTANCE CONTRACT)")

        def vr_write(row, at, ok=True):
            written["vr"][row] = ok
            vr_wrote[row] = at

        if op == OP_DMA_LD:
            if f0 not in (0, 1):
                raise ProgramError(f"{where}: f0={f0}, must be the source "
                                   f"(0 A, 1 B); DMA only writes VMEM")
            if f2 >= WPR or f1 + nr > MAXDIM:
                raise ProgramError(f"{where}: DRAM rows {f1}..{f1 + nr - 1}, "
                                   f"col block {f2} outside the {MAXDIM}x{MAXDIM} operand")
            for r in span("vmem", f3, nr):
                written["vmem"][r] = True
        elif op == OP_VLD:
            src = span("vmem", f1, nr)
            for i, (d, s_) in enumerate(zip(span("vr", f0, nr), src)):
                vr_write(d, it + i, written["vmem"][s_])   # a copy of unwritten
            it += nr                                       # is unwritten
        elif op == OP_VMATLOAD:
            if nr != T:
                raise ProgramError(f"{where}: nr={nr}; a vmatload moves "
                                   f"exactly T={T} weight rows")
            for i, r in enumerate(span("vr", f0, T)):
                vr_read(r, it + i, "weights")
            it += T
            if q["unused"]:
                raise ProgramError(f"{where}: the previous vmatload's weights "
                                   f"were never pushed; each PE switches on "
                                   f"the first pushed row after a load")
            q["loaded"] = q["unused"] = True
        elif op == OP_VMATPUSH:
            if not q["loaded"]:
                raise ProgramError(f"{where}: no vmatload before this push")
            for i, r in enumerate(span("vr", f0, nr)):
                vr_read(r, it + i, "activations")
            it += nr
            q["unused"] = False
            q["out"] += nr
            if q["out"] > OUTQ:
                raise ProgramError(f"{where}: {q['out']} rows pushed and not "
                                   f"popped; the output FIFO holds OUTQ={OUTQ}")
        elif op == OP_VMATPOP:
            if nr > q["out"]:
                raise ProgramError(f"{where}: pops {nr} rows but only "
                                   f"{q['out']} are pushed and un-popped; the "
                                   f"pop would wait forever")
            q["out"] -= nr
            for i, r in enumerate(span("vr", f0, nr)):   # row by row, as `vpu`
                vr_write(r, it + i)
            it += nr
        elif op == OP_VADD:
            s1, s2, dst = span("vr", f1, nr), span("vr", f2, nr), span("vr", f0, nr)
            for i in range(nr):              # two iterations per row
                vr_read(s1[i], it + 2 * i, "a source")
                vr_read(s2[i], it + 2 * i + 1, "a source")
                vr_write(dst[i], it + 2 * i + 1)
            it += 2 * nr
        elif op == OP_VRELU:
            src, dst = span("vr", f1, nr), span("vr", f0, nr)
            for i in range(nr):
                vr_read(src[i], it + i, "a source")
                vr_write(dst[i], it + i)
            it += nr
        elif op == OP_MVOUT:
            for i, r in enumerate(span("vr", f0, nr)):
                vr_read(r, it + i, "the value to retire")
            it += nr
            if f2 >= WPR or f1 + nr > MAXDIM:
                raise ProgramError(f"{where}: C rows {f1}..{f1 + nr - 1}, col "
                                   f"block {f2} outside the {MAXDIM}x{MAXDIM} result")
    if q["out"]:
        raise ProgramError(f"{q['out']} pushed row(s) are never popped")
    if q["unused"]:
        raise ProgramError("the last vmatload's weights are never pushed")


def assemble(prog, check=True):
    """Two words per instruction, behind a header of dynamic per-unit counts.

        imem[0] static instruction count   imem[4] loads | pushed rows << 16
        imem[1] dma_ld  rows               imem[5] (retired)
        imem[2] vmu     rows               imem[6] dma_st rows
        imem[3] vpu iterations             imem[7] A rows | B rows << 16

    imem[0] bounds the sequencer's fetch; every other count is dynamic, from
    `expand`. They must match the sequencer's dispatch rules exactly.

    **These are work counts, not instruction counts.** Each unit runs one flat
    loop over the rows (or words, or iterations) it will actually process, so
    what it is promised has to be the sum of `nr` over the instructions it is
    sent, with the per-unit adjustments the flattened bodies make:

      * `vpu` charges a `vadd` two iterations per row, and T per
        `vmatload` (its `nr` is T by rule).

    A unit promised the wrong number here does not produce a wrong answer, it
    hangs -- which is worth stating, because it is the one place where the
    assembler and the microarchitecture are coupled.

    Every program goes through `check_program` first. `check=False` exists
    only so a test can put a known-bad program on the machine and watch it
    fail; nothing that ships passes it.
    """
    if check:
        check_program(prog)
    ev = expand(prog)

    def rows(*ops):
        return sum(e[1] for e in ev if e[0] in ops)

    def count(*ops):
        return sum(1 for e in ev if e[0] in ops)

    def span(src):
        # The DRAM row span `dma_ld` must burst for one source matrix: the
        # highest row any of its `dma_ld`s names, after the AGU is resolved.
        return max([e[3] + e[1] for e in ev
                    if e[0] == OP_DMA_LD and (e[2] & DMA_SRC_B) == src] + [0])

    a_span, b_span = span(0), span(1)
    assert a_span <= MAXDIM and b_span <= MAXDIM, (
        f"dma_ld row span {a_span}/{b_span} exceeds MAXDIM={MAXDIM}")

    n_ld = count(OP_VMATLOAD)
    push_rows = rows(OP_VMATPUSH)
    assert n_ld < (1 << 15) and push_rows < (1 << 15), "array counts overflow"
    hdr = [len(prog),
           rows(OP_DMA_LD),
           rows(OP_DMA_LD, OP_VLD),
           rows(OP_VLD, OP_VMATLOAD, OP_VMATPUSH, OP_VMATPOP, OP_VRELU,
                OP_MVOUT) + 2 * rows(OP_VADD),
           n_ld | (push_rows << 16),
           0,                                  # retired (was accu's count)
           rows(OP_MVOUT),
           a_span | (b_span << 16)]
    assert len(hdr) == NHDR
    # Every count is read back through a 16-bit slice, which used to extract
    # to a signed ap_int<16>, so the usable range stops at 2^15 - 1 (see the
    # encoding note at the top of the file; the spare bit is kept).
    for h in hdr[1:4] + hdr[6:7]:
        assert 0 <= h < (1 << 15), f"header count {h} does not fit 15 bits"
    words = list(hdr)
    for w0, w1 in prog:
        words.append(int(w0))
        words.append(int(w1))
    assert len(words) <= IMEM_SIZE, f"{len(words)} words > IMEM_SIZE={IMEM_SIZE}"
    return words


def schedule(s):
    """The schedule the Vitis build applies. Reachable because `df.build` is
    `customize(func)` followed by `s.build(...)`, so the primitives apply to
    the dataflow region's kernels by their instance names.

      * A, B and C are cyclically partitioned by T: the DMA units move T lanes
        per cycle.
      * `ib` is cyclically partitioned by 8, so the sequencer's program
        prefetch writes 8 words per cycle (see `sequencer`).
      * **`vpu`'s row loop carries no dependence through `vreg`** -- the
        claim `s.dependence` emits as `#pragma HLS dependence variable=vreg
        inter false`. It is what holds the flat loop at II=1, and it is true
        because `check_program` enforces THE VREG DISTANCE CONTRACT on every
        program `assemble()` accepts.
      * **`vpu`'s row loop is a flushable pipeline** (`style=flp`, the
        `s.pipeline` option added for this): the unit pushes into the array
        and pops from it in one loop, and under the default stall pipeline a
        pop blocked on an earlier iteration's push deadlocks the RTL.
    """
    top = s.top_func_name
    s.partition(f"{top}:A", Partition.Cyclic, dim=2, factor=T)
    s.partition(f"{top}:B", Partition.Cyclic, dim=2, factor=T)
    s.partition(f"{top}:C", Partition.Cyclic, dim=2, factor=T)
    s.partition("sequencer_0:ib", Partition.Cyclic, dim=1, factor=8)
    s.dependence("vpu_0:x", "vreg", dep_type="inter", dependent=False)
    # vpu both feeds the array and drains it: a flushable pipeline, or a pop
    # that waits on an earlier iteration's push deadlocks the RTL
    s.pipeline("vpu_0:x", style="flp")
    # ... and so does every pipelined loop on the cycle it closes. A PE
    # blocked reading its NEXT activation row -- which the vpu will not push
    # until it has popped -- would otherwise freeze the previous row's
    # forwarding puts inside its own stall pipeline, and the pop never
    # completes (cosim deadlock at 16x16x16; 4x4x4 escaped only because the
    # last pushed rows end the PE loops, which drains them). The weight
    # loaders hold weights the same way.
    for i in range(T):
        for j in range(T):
            s.pipeline(f"pe_{i}_{j}:x", style="flp")
            s.pipeline(f"wld_{i}_{j}:c", style="flp")
    return s
