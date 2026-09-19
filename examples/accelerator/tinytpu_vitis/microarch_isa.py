# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU-isa: an instruction-programmable tiled-GEMM accelerator in grid Allo.

What the earlier designs got and did not get. Both files have since been removed;
read them with `git show e2451b81:examples/accelerator/tinytpu_vitis/<file>`.
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
  2. **A vector unit.** `vadd` / `vrelu` operate on accumulator rows. This
     item used to say tiled GEMM cannot do without `vadd`, because summing
     across k-tiles was an explicit `vadd`. That is no longer true: `mm`'s
     `f2` field selects overwrite or accumulate, so the shipped GEMM program
     emits no `vadd` at all, and `vadd_program` exists to keep the vector unit
     exercised.
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
engages (`RESULTS_WS.md` section 6; both files at `e2451b81`). Upstream's
`test_multi_cache_gemm.py` has no process of that shape: `offchip_loadA` writes
exactly one stream and the border PEs daisy-chain it, `L2_A[i] -> L2_A[i+1]`.
`test_tiled_systolic.py` runs on depth-**4** FIFOs for the same reason.

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

**Every unit but the accumulator is now one flat loop over the work items it
will process**, with the instruction fetched on the iteration that needs it:

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
one owner per memory. `spad` lives in `spm` and is written by `dma_ld` and read
by `vld`; `vr` lives in `vru` and is written by `vld` and read by `mm`; `ar`
lives in `accu` and is touched by all four of its opcodes. Allo enforces single
reader and single writer, and Vitis rejects the violation outright
(HLS 200-779 / 200-979), so those three units keep every arm. Flattening gets
what the split was wanted for without moving a memory: the opcode test survives
as a mux inside a pipelined body instead of as sub-loops the scheduler must
serialize. The two units that own no memory -- `dma_ld` and `dma_st` -- serve
one opcode each already, so there was nothing to split there either.

Two places where a naive flattening would have cost more than it saved, and
what each does instead:

  * `vru` emits a header word and T weight words per `mm`. In the fetch branch
    that is T+1 writes to one FIFO in one iteration, and a FIFO takes one write
    per cycle, so the whole loop would schedule at II=T+1. It charges the
    prologue T+1 *iterations* instead -- the same cycles, II=1 everywhere else.
  * the row counter is incremented at the TOP of the body. With `r += 1` at
    the bottom Vitis put the add in the last stage and the next iteration's
    conditional queue read depends on it, which is a distance-1 recurrence that
    does not close in one cycle: `Final II = 2` on every unit. Hoisting the
    increment is worth exactly a factor of two here and costs nothing.
  * the memory is read ONCE per iteration, at an address the branch selects.
    Written as a read in each arm -- the obvious transcription -- it
    synthesized to one read port per arm on one memory, and `vru` came back at
    II=2 and `accu` at II=3. Muxing the address rather than the data fixes it.

**`accu` did not flatten, and it is not flattened here.** Its four arms all
touch `ar`, and once `r` is a carried register rather than the loop's own
induction variable, Vitis can no longer prove that iteration *n*'s store and
iteration *n+1*'s load touch different rows: `Final II = 3`. Completely
partitioning `ar` into registers removes the aliasing question entirely and
gets II=2, no further -- read, add, write, and the next iteration may read what
this one wrote is a real recurrence. Both versions were built and both were
bit-exact; both were also slower than the nested loop, which keeps `mm` at II=1
and pays a pipeline fill per instruction instead. `accu`'s docstring has the
detail.

That is the general rule this pass ran into: **flattening trades a fixed
per-instruction cost for a permanent per-row II, and only pays where the II
stays at 1.** Four units kept the flat loop; one did not, and did not.

The array is left alone by this pass: a PE's loop counts `mm` instructions,
not rows, and its per-`mm` prologue (latch a weight, forward the rest down the
column) is a different shape from its MAC body, so folding the two would set
the II of the MAC path -- the one loop in the design that is already II=1 and
carries the actual arithmetic -- from the worst of the two. `vru` hands the
array its total wavefront-row count alongside the `mm` count so that a later
pass can flatten a PE without another header change.

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

  * `sequencer` pulls the whole program on-chip with one `IMEM_SIZE`-word burst
    and fetches from BRAM afterwards. Fetched in place,
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

int8 lanes, int32 accumulation, `vst` clipping to int8 -- Gemmini's default
config (`inputType = SInt(8.W)`, `accType = SInt(32.W)`) and its `mvout`
behaviour under `ACC_SCALE_IDENTITY` with shift 0, which is what `allo_cmp.c`
passes. Packing is what makes the SIMD scratchpad work, and packing needs
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
OP_LOOP = 8       # open a loop, body is the next instruction   nr=trip count
OP_ENDLOOP = 9    # close the innermost loop

# ---- THE WRITE-BEFORE-READ CONTRACT (`mm` acc, `vadd`, `vrelu`, `mvout`) ----
# `spad`, `vr` and `ar` are NOT cleared by the hardware. Their `= 0`
# initialisers were removed to delete a 514-cycle memset, so at `ap_start`
# each holds whatever the previous invocation (or power-up) left there. The
# guarantee moved from the hardware to the program:
#
#   * `mm` with f2=1 (accumulate), `vadd` (both sources), `vrelu` (source) and
#     `mvout` read `ar` rows, and every such row must have been written earlier
#     in the SAME program -- by an overwriting `mm` (f2=0), a `vadd` or a
#     `vrelu`. `ar` is the accumulator; reading it early is a wrong answer,
#     not a crash.
#   * `mm` reads `vr` (T weight rows at f3, `nr` activation rows at f0); those
#     rows must hold data that came from a `dma_ld` via `vld`.
#   * `vld` is a pure copy and MAY copy an unwritten `spad` row -- the shipped
#     GEMM does, when M < MAXDIM it moves whole MAXDIM-row blocks -- but the
#     copy is then unwritten too, and consuming it in an `mm` is an error.
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
    `B_SP + nb*MAXDIM + kb*T`, and a one-term-per-field encoding cannot say it.

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
# (notes/ALLO_SHORTCOMINGS.md #11); before that fix it hung with no output.
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
SPAD_ROWS = int(os.environ.get("TPU_SPAD", 512))   # rows, each one packed word
NVR = int(os.environ.get("TPU_NVR", 256))          # operand vector registers
NAR = int(os.environ.get("TPU_NAR", 128))          # accumulator vector registers
QD = int(os.environ.get("TPU_QD", 8))              # stream depth

MAXROWS = 127                                      # `nr` is 8 bits, top bit spare
NHDR = 8                       # imem[0:NHDR] is the header, instructions follow

# Instruction slots. Sized to the longest program the built MAXDIM admits, not
# to a round number: the sequencer's prefetch burst is `IMEM_SIZE` words long
# whatever the program, so every unused slot is a wasted startup cycle -- the
# same arithmetic as under `wrap_io=True`, which copied the declared length for
# the same reason. The worst case is
# gemm(MAXDIM,MAXDIM,MAXDIM) with relu -- Kt + Nt dma_lds, Kt vlds for A,
# Nt*(Kt vld + Kt mm) + Nt relu + Nt mvout -- plus the NHDR header.
_KB = MAXDIM // T
# With control flow the program is O(nesting), not O(tiles): the looped GEMM is
# 17 instructions at every shape where the unrolled one reaches 49. So imem is
# sized to the longest program shipped rather than to the largest problem.
#
# This is not cosmetic. The program is pulled on-chip by one burst of
# `IMEM_SIZE` words, so every imem word is a startup cycle whether the program
# uses it or not -- 56 cycles today, and it is the only fixed charge left that
# does not scale with the shape. Measured: moving to a
# 2-word instruction format cost exactly +68 cycles at all five shapes, which
# is exactly the 68 extra words it added -- the loop logic itself cost nothing.
# Sizing imem to the looped program is where the hardware loop actually pays.
_MAX_STATIC = 24               # longest program shipped, plus headroom
IMEM_SIZE = int(os.environ.get("TPU_IMEM", NHDR + IWORDS * _MAX_STATIC))

# Scratchpad and vreg layout. Fixed offsets in a fixed memory, sized for the
# largest supported shape rather than for the shape being run.
KB_MAX = MAXDIM // T           # column blocks in the widest matrix
assert MAXDIM % T == 0, "a DRAM row must be a whole number of packed words"
WPR = MAXDIM // T              # packed words per DRAM row
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
        """Fetch, decode, resolve addresses, dispatch.

        **This unit has a program counter and a loop stack.** Before, every
        tiling loop was unrolled in Python at assembly time, so the static
        program grew with the problem and the machine had no control flow at
        all -- which is not an instruction-programmable accelerator in any
        useful sense. Now `loop`/`endloop` bound a body and the PC branches
        backward, so the *static* program is O(nesting) while the *dynamic*
        instruction stream is O(tiles). Modelled on MiniTPU's
        `sequencer_loop_ctrl`: a LIFO of `{body_start, iv, trip}` frames,
        `LOOP_DEPTH` levels deep.

        **Addresses resolve here, not in the units.** Each instruction's second
        word carries up to three `(level, stride)` terms, and this loop adds
        `iv[level] * stride` to the corresponding field before dispatching. That
        is the whole reason the loop is worth having -- a body that reissued the
        same addresses every iteration would just redo the same work -- and it
        keeps every other unit unchanged, since what they receive is a resolved
        instruction word in the format they already decode.

        The dispatch order is dataflow order and that is load-bearing: if this
        blocks on a full queue, every unit upstream of that one already holds
        the instruction and can keep producing what the blocked unit waits for.
        Dispatching downstream-first deadlocks."""
        # --- ONE contiguous burst pulls the whole program on-chip ---
        # `l_imem` is an `m_axi` port: the design is built `wrap_io=False`, so
        # nothing is hoisted on our behalf. Fetched in place,
        # `l_imem[NHDR + pc * IWORDS]` is a data-dependent address and Vitis can
        # only turn it into a **two-word burst per instruction** -- measured,
        # `[HLS 214-115] Multiple burst reads of length 2 and bit width 64 ...
        # on bundle 'gmem0'` -- so the fetch loop closes at `Final II = 13`
        # instead of 5, and ~76 dynamic sequencer iterations each pay full bus
        # latency. That, not the operand path, was the whole of the marginal
        # cost that made an earlier `wrap_io=False` measurement look like a
        # loss.
        #
        # A constant-trip contiguous copy is the pattern Vitis turns into one
        # real AXI burst, so the program arrives as a single `IMEM_SIZE`-word
        # transfer and every fetch after it is a BRAM read. Instructions are
        # the one stream in this design whose access pattern is random, so they
        # are the one stream that has to be prefetched rather than streamed.
        ib: UInt(64)[IMEM_SIZE]
        for i in range(IMEM_SIZE):
            ib[i] = l_imem[i]

        iw: UInt(64) = ib[0]
        n_instr: int32 = iw[0:16]

        c_dld.put(ib[1])
        c_dld.put(ib[7])
        c_spm.put(ib[2])
        c_vru.put(ib[3])
        c_vru.put(ib[4])
        c_acc.put(ib[5])
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
                    c_spm.put(rw)
                if op == OP_VLD:
                    c_spm.put(rw)
                    c_vru.put(rw)
                if op == OP_MM:
                    c_vru.put(rw)
                    c_acc.put(rw)
                if op == OP_VADD:
                    c_acc.put(rw)
                if op == OP_VRELU:
                    c_acc.put(rw)
                if op == OP_MVOUT:
                    c_acc.put(rw)
                    c_dst.put(rw)
                pc += 1

            if pc >= n_instr:
                running = 0

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

        **TWO VARIABLE-LENGTH BURSTS, THEN EVERY INSTRUCTION RUNS FROM BRAM.**
        This unit reads `m_axi` directly -- the design is built
        `wrap_io=False`, so nothing is hoisted for it -- and *what* it reads is
        the whole of the fixed cost the design was carrying. Two access
        patterns were measured against Vitis:

          * the per-instruction strided read this unit used to do,
            `lA[(f1 + r) * MAXDIM + f2 * T + e]` with `e` unrolled over the T
            lanes, yields `[HLS 214-115] Multiple burst reads of length 4 and
            bit width 8` -- a separate four-beat AXI transaction per row, and
            the flat loop closes at `Final II = 4` with the real cost nearer
            9 cycles a row once request latency is counted.
          * a contiguous sweep whose trip count is a RUNTIME value yields
            `[HLS 214-115] ... of VARIABLE LENGTH`, i.e. one real AXI burst.

        So the unit opens with one burst per operand matrix, covering exactly
        the DRAM rows the program will name -- `na` and `nb` come from the
        assembler, which resolves the control flow and the AGU and takes the
        maximum `f1 + nr` over the `dma_ld`s of each source. Nothing is fetched
        for a shape that does not touch it, which is the property `wrap_io`
        could not have: its copy is the DECLARED length, always 256 words per
        argument, whatever is being run -- 824 words of argument copying before
        the region starts, which was 907 of the 1586 cycles at 16x16x16 and 90%
        of them at 4x4x4.

        The bytes arrive packed: the burst loop assembles each `UInt(T*8)` word
        as it sweeps, so the instruction loop after it does ONE BRAM read per
        row and no packing at all. `rbA` and `rbB` are separate memories, so
        the source mux costs one read port on each rather than two on one --
        the same rule `vru` and `accu` are written to (see the II note above),
        arrived at from the other direction.

        **One flat loop over rows, not a loop over instructions containing a
        loop over rows** -- see the ROW-FLATTENING note above. The instruction
        is fetched on the iteration that needs it. The burst loops sit OUTSIDE
        that loop on purpose: a conditional prefetch inside the body would put
        a runtime-bounded inner loop back into it and take the flat loop out of
        pipelining altogether, which is the trade that flattening bought."""
        nw: UInt(64) = c_dld.get()
        n_row: int32 = nw[0:16]     # 16-bit slice: a stream word is UInt(64),
                                    # and the top bit of a slice is its sign,
                                    # so this carries 0..32767 counts.
        sw: UInt(64) = c_dld.get()  # the DRAM row span of each source matrix
        na: int32 = sw[0:16]
        nb: int32 = sw[16:32]

        # One burst per matrix, each covering exactly that matrix's own span.
        # Merging the two into a single loop bounded by `max(na, nb)` was built
        # and measured: it does halve the burst time, because A and B are
        # separate `m_axi` bundles and their bursts then overlap (`[HLS
        # 214-115] ... variable length ... on gmem1` and `... on gmem2`, both
        # attributed to the one loop), and it changed the cosim count at all
        # five shapes by exactly ZERO cycles. The operand burst is already
        # entirely hidden behind the sequencer's dispatch and the units' fill,
        # so the version that reads the fewest bytes is the one to keep.
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
            if f0 == 0:
                pw = rbA[(f1 + r) * WPR + f2]
            else:
                pw = rbB[(f1 + r) * WPR + f2]
            dma2sp.put(pw)

    @df.kernel(mapping=[1])
    def spm():
        """The scratchpad, and the only unit that owns it.

        Pure SIMD access: an address names a whole `UInt(T*8)` row and there is
        no way to address a lane. That is what lets a single-ported memory feed
        T lanes per cycle, and it is why `HLS 200-779` (single reader, single
        writer) is satisfied without a pragma.

        **This unit cannot be split by opcode** -- `dma_ld` writes `spad` and
        `vld` reads it, and Allo enforces one owner per memory, so the two arms
        have to stay in one process. What they can do is share one flat row
        loop: the opcode test survives, but as a mux inside a pipelined body
        rather than as two sub-loops the scheduler has to serialize. One
        `spad` read and one `spad` write per iteration is what a dual-port
        BRAM gives, so this still schedules at II=1."""
        spad: UInt(VW)[SPAD_ROWS]
        nw: UInt(64) = c_spm.get()
        n_row: int32 = nw[0:16]     # 16-bit slice: a stream word is UInt(64),
                                    # and the top bit of a slice is its sign,
                                    # so this carries 0..32767 counts.
        op: int32 = 0
        f1: int32 = 0
        f3: int32 = 0
        cnt: int32 = 0
        r: int32 = -1               # advanced at the TOP: see the II note
        for x in range(n_row):
            r += 1
            if r >= cnt:
                w0: UInt(64) = c_spm.get()
                op = w0[0:6]
                f1 = w0[18:30]
                f3 = w0[42:54]
                cnt = w0[54:62]
                r = 0
            if op == OP_DMA_LD:
                spad[f3 + r] = dma2sp.get()
            else:
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
        array at all.

        **This is the unit the flattening was for.** It was the critical
        process in the design: at 16x16x16 it issues 33 instructions and 464
        words, so 33 un-overlapped instruction boundaries cost more than the
        words did. Like `spm` it cannot be split -- `vld` writes `vr` and `mm`
        reads it -- so it gets one flat loop instead.

        The `mm` prologue (one header word plus T weight words) is the reason
        this is a *word* loop and not a row loop. Emitting it inside the fetch
        branch would put T+1 writes to `wcol[0]` in one iteration, and a FIFO
        takes one write per cycle, so the whole loop would schedule at II=T+1.
        Charging the prologue T+1 *iterations* instead costs exactly the same
        T+1 cycles and leaves every other iteration at II=1."""
        nw: UInt(64) = c_vru.get()
        n_word: int32 = nw[0:16]    # 16-bit slice: a stream word is UInt(64),
                                    # and the top bit of a slice is its sign,
                                    # so this carries 0..32767 counts.
        mw: UInt(64) = c_vru.get()
        n_mm_in: int32 = mw[0:16]
        n_mr_in: int32 = mw[16:32]
        vr: UInt(VW)[NVR]
        # Hand the array its own counts before anything else, on the chain it
        # already uses for headers and weights: how many `mm`s, and how many
        # wavefront rows they carry between them. The second is what lets a PE
        # flatten its own loop the same way (see `pe`).
        nmo: UInt(VW) = 0
        nmo[0:16] = n_mm_in
        nmo[16:32] = n_mr_in
        wcol[0].put(nmo)
        op: int32 = 0
        f0: int32 = 0
        f3: int32 = 0
        nr: int32 = 0
        cnt: int32 = 0
        r: int32 = -1               # advanced at the TOP: see the II note
        for x in range(n_word):
            r += 1
            if r >= cnt:
                w0: UInt(64) = c_vru.get()
                op = w0[0:6]
                f0 = w0[6:18]
                f3 = w0[42:54]
                nr = w0[54:62]
                cnt = nr
                if op == OP_MM:
                    cnt = nr + T + 1
                r = 0
            if op == OP_VLD:
                vr[f0 + r] = sp2vr.get()
            elif r == 0:
                # The header carries the row count, and it is sent **only for
                # `mm`** -- the array's loop counts `mm` instructions, not
                # program instructions, so there is nothing for it to skip and
                # no opcode in the array at all.
                hdr: UInt(VW) = 0
                hdr[0:12] = nr
                wcol[0].put(hdr)
            else:
                # ONE read of `vr`, at an address the branch selects. Written
                # as two reads in two arms it synthesized to two read ports on
                # the same memory, and with the `vld` write that is three
                # accesses to a dual-port BRAM: measured `Final II = 2`, which
                # would have doubled the cost of the wavefront stream. Muxing
                # the address instead of the data costs an adder and holds the
                # loop at II=1.
                va: int32 = f0 + r - T - 1
                if r <= T:
                    # T weight words, row 0 first: each PE keeps the first
                    # word it sees and forwards the rest, so row i keeps
                    # word i.
                    va = f3 + r - 1
                vv: UInt(VW) = vr[va]
                if r <= T:
                    wcol[0].put(vv)
                else:
                    acol[0].put(vv)

    @df.kernel(mapping=[T, T])
    def pe():
        """One processing element. Weight-stationary, and it decodes nothing.

        Its loop counts **`mm` instructions**, not program instructions: the
        sequencer sends `mm` only to the units that execute it, so the array
        never sees -- and never spends a cycle decoding -- a `vld`, `vadd` or
        `mvout`. At 16x16x16 that is 16 iterations instead of 80.

        The compute loop has **no loop-carried value**: tap a lane, take the
        partial sum from the north, multiply-add, pass both on. The multiplier
        and adder latencies are pipeline *depth*, not initiation interval --
        the property `microarch.py` (at `e2451b81`) could not have, and the reason a
        systolic array is built deep."""
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
        ACC_SCALE_IDENTITY with shift 0.

        **THIS IS THE UNIT THAT WOULD NOT FLATTEN.** Four opcodes and one
        memory: `ar` is written by `mm`, read-modify-written by `vadd` and
        `vrelu`, and read by `mvout`. One owner per memory rules out splitting
        it, so the flat row loop the other four units got was the only route.
        It was built twice, was bit-exact both times, and was slower both
        times, so it is not here. The two attempts and what each hit:

          * **`ar` in BRAM: `Final II = 3`.** Muxing the four arms down to one
            `ar` read and one `ar` write fixed the port count, and hoisting the
            row counter fixed the queue recurrence -- both of which worked for
            the other units. What remained was a memory dependence:

                Unable to enforce a carried dependence constraint
                (II = 1, distance = 1) between 'store' on array 'ar'
                and 'load' ('rv') on array 'ar'

            In the nested form the index is `f1 + r` with `r` the inner loop's
            own induction variable, so Vitis proves iteration *n*'s store and
            iteration *n+1*'s load touch different rows and schedules `mm` at
            II=1. Flattening makes `r` a carried register, the index stops
            being affine in the loop index, and the proof fails. Deferring the
            write by one iteration -- the textbook fix -- only moves the
            violation onto the enable register.
          * **`ar` completely partitioned into registers: `Final II = 2`.** A
            register file has no port to arbitrate and nothing to prove about
            aliasing, and it did take the II from 3 to 2 (at 33k FF and 12k
            LUT, and 10 minutes of synthesis). The remainder is not an
            artifact: read `ar[ra]`, add, write `ar[wa]`, and the next
            iteration may read what this one wrote. That is a genuine
            read-modify-write recurrence through a register file and it does
            not close in one cycle.

          * **A write-behind rotation: `Final II = 1` -- built, measured,
            REVERTED.** II=2 is *not* the floor. Keeping the last two computed
            rows in registers, writing `ar` two iterations late and answering a
            read inside that window from the registers takes the carried path
            off the memory entirely (adder -> rotation register -> bypass mux
            -> adder), and it reaches `Final II = 1, Depth = 6`, bit-exact at
            all five shapes. It is not here because of what it cost: `accu` FF
            1270 -> 17450, 13.7x, for 2.3% end-to-end (1457 -> 1423 at
            16x16x16), and `ar` scales with T so the area grows with the array
            while the 2.3% does not. `#pragma HLS dependence variable=ar inter
            false` buys the same II for no area and Allo emits no such pragma;
            that gap is what this priced. Full numbers in `RESULTS_ISA.md`.

        So the nested loop is not the only thing that closes, but it is the one
        that is worth the area, and it keeps `mm` at II=1 where the GEMM
        actually lives. `vadd`'s own loop is II=2 (two
        `ar` reads and a write is three accesses to a dual-port BRAM), which is
        real but off the GEMM path.

        The general rule, and the reason four units flattened and this one did
        not: **flattening trades a fixed per-instruction cost for a permanent
        per-row II, and only pays where the II stays at 1.**"""
        ar: UInt(AW)[NAR]
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
                # `microarch_ws.py`'s drainer (at `e2451b81`) accumulated at II=1.
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
        n_row: int32 = nw[0:16]     # 16-bit slice: a stream word is UInt(64),
                                    # and the top bit of a slice is its sign,
                                    # so this carries 0..32767 counts.
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
    and `bench_isa.py` runs it. What the generated form removes is the last two
    lines of this docstring: the AGU *levels* below are hand-typed integers
    that have to agree with where the `loop`/`endloop` pairs happen to sit, and
    nothing here checks that they do.

    The static program is O(nesting), not O(tiles): the n and k loops are
    `loop`/`endloop` pairs and the addresses that used to be baked into each
    unrolled instruction are AGU terms against the induction variables. The
    instruction count stops growing with the problem, which is the property
    that makes this a programmable machine rather than a very long tape.

    The first k-tile is peeled out of the k loop deliberately: it carries
    `f2=0` (overwrite the accumulator) where the looped tiles carry `f2=1`
    (accumulate), and peeling is how you say that without a predicate on the
    induction variable.
    """
    assert M <= MAXDIM and K <= MAXDIM and N <= MAXDIM, (
        f"{M}x{K}x{N} exceeds the built MAXDIM={MAXDIM}")
    assert M % T == 0 and K % T == 0 and N % T == 0
    assert M <= MAXROWS and K <= MAXROWS
    Kt, Nt = K // T, N // T
    p = []
    ins = lambda w, agu=0: p.append((w, agu))

    # --- A: one dma_ld per column block, kb on the loop ---
    ins(enc(OP_LOOP, nr=Kt))
    ins(enc(OP_DMA_LD, f0=0, f1=0, f2=0, f3=A_SP, nr=M),
        enc_agu((AGU_F2, 0, 1), (AGU_F3, 0, MAXDIM)))
    ins(enc(OP_ENDLOOP))
    # --- B: one per column block, nb on the loop ---
    ins(enc(OP_LOOP, nr=Nt))
    ins(enc(OP_DMA_LD, f0=1, f1=0, f2=0, f3=B_SP, nr=K),
        enc_agu((AGU_F2, 0, 1), (AGU_F3, 0, MAXDIM)))
    ins(enc(OP_ENDLOOP))
    # --- every A word into vregs once ---
    total_a = Kt * MAXDIM
    assert total_a <= MAXROWS, "A vld exceeds the row-count field"
    ins(enc(OP_VLD, f0=A_VR, f1=A_SP, nr=total_a))

    # --- the output loop: level 0 is nb, level 1 is kb ---
    ins(enc(OP_LOOP, nr=Nt))
    #   peeled first k-tile: overwrite the accumulator
    ins(enc(OP_VLD, f0=W_VR, f1=B_SP, nr=T),
        enc_agu((AGU_F1, 0, MAXDIM)))
    ins(enc(OP_MM, f0=A_VR, f1=AR_C, f2=0, f3=W_VR, nr=M))
    if Kt > 1:
        ins(enc(OP_LOOP, nr=Kt - 1))
        #   f1 needs BOTH tiles: B_SP + nb*MAXDIM + (kb+1)*T
        ins(enc(OP_VLD, f0=W_VR, f1=B_SP + T, nr=T),
            enc_agu((AGU_F1, 0, MAXDIM), (AGU_F1, 1, T)))
        ins(enc(OP_MM, f0=A_VR + MAXDIM, f1=AR_C, f2=1, f3=W_VR, nr=M),
            enc_agu((AGU_F0, 1, MAXDIM)))
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
        p.append((enc(OP_DMA_LD, f0=0, f1=0, f2=kb, f3=A_SP + kb * MAXDIM,
                      nr=M), 0))
    for nb in range(Nt):
        p.append((enc(OP_DMA_LD, f0=1, f1=0, f2=nb, f3=B_SP + nb * MAXDIM,
                      nr=K), 0))
    total_a = Kt * MAXDIM
    for c0 in range(0, total_a, MAXROWS):
        n = min(MAXROWS, total_a - c0)
        p.append((enc(OP_VLD, f0=A_VR + c0, f1=A_SP + c0, nr=n), 0))
    for nb in range(Nt):
        for kb in range(Kt):
            p.append((enc(OP_VLD, f0=W_VR, f1=B_SP + nb * MAXDIM + kb * T,
                          nr=T), 0))
            p.append((enc(OP_MM, f0=A_VR + kb * MAXDIM, f1=AR_C,
                          f2=(1 if kb else 0), f3=W_VR, nr=M), 0))
        if relu:
            p.append((enc(OP_VRELU, f0=AR_C, f1=AR_C, nr=M), 0))
        p.append((enc(OP_MVOUT, f0=AR_C, f1=0, f2=nb, nr=M), 0))
    return p


def vadd_program(M, K, N):
    """A program for the vector unit itself, so `vadd`/`vrelu` stay exercised
    now that tiled GEMM no longer needs them on its inner loop.

    Computes A@B twice into two accumulator regions, adds them, ReLUs the sum
    and retires it: `relu(2 * (A @ B))` on the first output tile."""
    return [(w, 0) for w in [
        enc(OP_DMA_LD, f0=0, f1=0, f2=0, f3=A_SP, nr=M),
        enc(OP_DMA_LD, f0=1, f1=0, f2=0, f3=B_SP, nr=T),
        enc(OP_VLD, f0=A_VR, f1=A_SP, nr=M),
        enc(OP_VLD, f0=W_VR, f1=B_SP, nr=T),
        enc(OP_MM, f0=A_VR, f1=AR_C, f2=0, f3=W_VR, nr=M),
        enc(OP_MM, f0=A_VR, f1=AR_P, f2=0, f3=W_VR, nr=M),
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
           OP_MVOUT: "mvout", OP_LOOP: "loop", OP_ENDLOOP: "endloop"}


def check_program(prog):
    """Reject a program the hardware cannot run correctly. Called by `assemble`.

    **Static, and exact rather than conservative.** The program has no
    data-dependent control flow -- trip counts and AGU strides are instruction
    fields -- so the dynamic instruction stream is a pure function of the
    program text, and walking it (`_trace`, the same walk that computes the
    header) visits every issue the sequencer will make with the addresses it
    will resolve. There is no loop abstraction to get wrong.

    What it checks, per dynamic issue:

      * **write-before-read** on `spad`, `vr` and `ar` -- the contract at the
        opcode table. Written-ness is tracked per row and propagated through
        `vld` (a copy of an unwritten row is unwritten); it is an error only
        where a value is *consumed*: `mm` reading `vr`, and `mm`-acc / `vadd` /
        `vrelu` / `mvout` reading `ar`.
      * **bounds** on every memory and on `C`/`A`/`B`: an out-of-range row is
        silent corruption in RTL, not an exception.
      * **`nr >= 1`** on every data op. `dma_ld`, `spm`, `vru` and `dma_st` run
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
        retired opcode, `mm` f2 in {0, 1}, `dma_ld` source in {0, 1}.

    Raises `ProgramError` naming the static instruction, the loop iteration,
    and the rows; returns None."""
    if not prog:
        raise ProgramError("empty program")
    depth = 0
    for pc, (w0, w1) in enumerate(prog):
        op = w0 & 0x3F
        name = _OPNAME.get(op)
        where = f"instruction {pc} ({name or f'opcode {op}'})"
        if name is None or op == OP_DMA_ST:
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

    written = {"spad": [False] * SPAD_ROWS, "vr": [False] * NVR,
               "ar": [False] * NAR}
    size = {"spad": SPAD_ROWS, "vr": NVR, "ar": NAR}

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
        if op == OP_DMA_LD:
            if f0 not in (0, 1):
                raise ProgramError(f"{where}: source f0={f0}, must be 0 (A) or 1 (B)")
            if f2 >= WPR or f1 + nr > MAXDIM:
                raise ProgramError(f"{where}: DRAM rows {f1}..{f1 + nr - 1}, "
                                   f"col block {f2} outside the {MAXDIM}x{MAXDIM} operand")
            for r in span("spad", f3, nr):
                written["spad"][r] = True
        elif op == OP_VLD:
            src = span("spad", f1, nr)
            for d, s in zip(span("vr", f0, nr), src):
                written["vr"][d] = written["spad"][s]
        elif op == OP_MM:
            if f2 not in (0, 1):
                raise ProgramError(f"{where}: f2={f2}, must be 0 (overwrite) or 1 (accumulate)")
            need("vr", span("vr", f3, T), "weights")
            need("vr", span("vr", f0, nr), "activations")
            dst = span("ar", f1, nr)
            if f2 == 1:
                need("ar", dst, "the accumulate base")
            for r in dst:
                written["ar"][r] = True
        elif op in (OP_VADD, OP_VRELU):
            srcs = [span("ar", f1, nr)] + ([span("ar", f2, nr)] if op == OP_VADD else [])
            dst = span("ar", f0, nr)
            for i, d in enumerate(dst):      # row by row, as `accu` runs it
                need("ar", [s[i] for s in srcs], "a source")
                written["ar"][d] = True
        elif op == OP_MVOUT:
            need("ar", span("ar", f0, nr), "the value to retire")
            if f2 >= WPR or f1 + nr > MAXDIM:
                raise ProgramError(f"{where}: C rows {f1}..{f1 + nr - 1}, col "
                                   f"block {f2} outside the {MAXDIM}x{MAXDIM} result")


def assemble(prog, check=True):
    """Two words per instruction, behind a header of dynamic per-unit counts.

        imem[0] static instruction count   imem[4] mm count | mm rows << 16
        imem[1] dma_ld  rows               imem[5] accu   INSTRUCTIONS
        imem[2] spm     rows               imem[6] dma_st rows
        imem[3] vru     words              imem[7] A rows | B rows << 16

    `accu` is the exception, and the only one: it is the one unit still
    counting instructions, because it is the one unit that stayed nested (see
    its docstring).

    imem[0] bounds the sequencer's fetch; every other count is dynamic, from
    `expand`. They must match the sequencer's dispatch rules exactly.

    **These are work counts, not instruction counts.** Each unit runs one flat
    loop over the rows (or words, or iterations) it will actually process, so
    what it is promised has to be the sum of `nr` over the instructions it is
    sent, with the two per-unit adjustments the flattened bodies make:

      * `vru` charges an `mm` an extra `T + 1` words for the header and the T
        weight words it pushes down `wcol`, because they are iterations of the
        same loop rather than a prologue outside it;

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
                    if e[0] == OP_DMA_LD and e[2] == src] + [0])

    a_span, b_span = span(0), span(1)
    assert a_span <= MAXDIM and b_span <= MAXDIM, (
        f"dma_ld row span {a_span}/{b_span} exceeds MAXDIM={MAXDIM}")

    n_mm = count(OP_MM)
    mm_rows = rows(OP_MM)
    assert n_mm < (1 << 15) and mm_rows < (1 << 15), "array counts overflow"
    hdr = [len(prog),
           rows(OP_DMA_LD),
           rows(OP_DMA_LD, OP_VLD),
           rows(OP_VLD) + mm_rows + n_mm * (T + 1),
           n_mm | (mm_rows << 16),
           count(OP_MM, OP_VADD, OP_VRELU, OP_MVOUT),
           rows(OP_MVOUT),
           a_span | (b_span << 16)]
    assert len(hdr) == NHDR
    # Every count is read back through a 16-bit slice, which used to extract
    # to a signed ap_int<16>, so the usable range stops at 2^15 - 1 (see the
    # encoding note at the top of the file; the spare bit is kept).
    for h in hdr[1:4] + hdr[5:7]:
        assert 0 <= h < (1 << 15), f"header count {h} does not fit 15 bits"
    words = list(hdr)
    for w0, w1 in prog:
        words.append(int(w0))
        words.append(int(w1))
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
