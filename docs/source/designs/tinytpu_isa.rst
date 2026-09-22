..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

############################################
TinyTPU-isa: A Programmable GEMM Accelerator
############################################

TinyTPU-isa (``examples/accelerator/tinytpu_vitis/microarch_isa.py``) is an int8
instruction-programmable tiled-GEMM accelerator written in grid Allo
(``@df.region`` / ``@df.kernel``) and taken through the Vitis HLS dataflow path to
**RTL co-simulation**. It is the machine the project's requirements name: an ISA,
a vector unit, a SIMD scratchpad, vector registers streaming to the array's
ports, and tiled GEMM as a *program* rather than as a fixed-function datapath.

It is the only accelerator design left on ``main``. Its predecessors, the
optimisation history that produced the current numbers, and the superseded
readings of those numbers are on :doc:`tinytpu_history`. The data-type- and
mesh-matched comparison against Gemmini is on :doc:`gemmini_comparison`.

.. note::

   Headline, as of 2026-09-19 (``e24e433b``): one hardware build runs every
   shape as data, all five benchmark shapes are bit-exact in RTL
   co-simulation, and the design takes **172 / 262 / 418 / 484 / 686** cycles
   at 4x4x4 / 8x8x8 / 12x12x12 / 16x16x8 / 16x16x16 (Vitis ``cosim``,
   ``-m_axi_latency 0``). Measured over the same window on both sides, the
   design is **1.07-1.24x slower** than Gemmini at all five shapes; see
   :doc:`gemmini_comparison`.

   Until ``e24e433b`` the shipped design took **252 / 383 / 591 / 667 / 919**
   (1.55-1.8x behind Gemmini). The step between the two is the gap
   attribution's measured design stack, landed as the design
   (:ref:`tinytpu-isa-landing`, :ref:`gemmini-gap-attribution`).


Architecture
------------

.. code-block:: text

           imem ─► sequencer ──(decoded instruction)──► every unit
                   ┌─► spm[spad] ──(header, weights)──► wld ─┐
     A,B ──► dma_ld┤                                          ├─► 4x4 WS array ─► accu[ar] ─► dma_st ─► C
                   └─► vru[vr] ────(activations)──────────────┘

Eight kinds of unit, ``2*T*T + 6 = 38`` concurrent processes at ``T=4``:

.. list-table::
   :header-rows: 1

   * - unit
     - owns
     - does
   * - ``sequencer``
     - ``imem``
     - fetch, decode, broadcast; consumes nothing
   * - ``dma_ld``
     - ``A``, ``B``
     - DRAM -> scratchpad or operand vregs, packing T lanes/cycle
   * - ``spm``
     - ``spad``
     - the scratchpad; **pure SIMD access**; the array's weight port
   * - ``vru``
     - ``vr``
     - operand vregs; the array's activation port
   * - ``wld`` x16
     - --
     - one per PE: walks the weight chain, double-buffers its PE's weight
   * - ``pe`` x16
     - its lane
     - weight-stationary MAC, decodes nothing, one flat loop
   * - ``accu``
     - ``ar``
     - accumulator vregs **and the vector ALU**, one flat loop at II=1
   * - ``dma_st``
     - ``C``
     - accumulator -> DRAM, clipped (executes ``mvout``)

**A vector unit that tiled GEMM needed.** As designed, ``mm`` computed the psums
of *one* k-tile and wrote them to accumulator registers, and summing across
k-tiles was an explicit ``vadd`` -- load-bearing, not decoration (the module
docstring still states the requirement this way). This is Gemmini's split too:
the adds live in ``AccumulatorMem``'s write path, not in the mesh. In the
shipped program ``mm`` carries an overwrite/accumulate field (``f2``) and the
GEMM inner loop no longer uses ``vadd`` at all, so ``vadd_program`` exists to
keep ``vadd``/``vrelu`` exercised; see :ref:`tinytpu-isa-programs`.

**Pure SIMD scratchpad.** A row of ``spad`` *is* one ``UInt(T*8)`` packed word of
T int8 lanes; there is no way to address a lane. That is what lets a
single-ported memory feed T lanes per cycle, and it satisfies ``HLS 200-779``
(single reader, single writer) without a pragma. The diagnostic, as Vitis first
raised it on the single-grid predecessor, is quoted on :doc:`tinytpu_history`.

**The PEs decode nothing.** A header word leads every ``mm`` down the chain the
weights use, carrying just its row count. Each PE's weight loader (``wld``)
forwards it and hands its PE one ``(weight, rows)`` word through a depth-4 FIFO
-- the shadow register, so ``mm`` n+1's weight is latched while ``mm`` n
computes (Gemmini's c1/c2 double buffer); there is no command fan-out to
``T*T`` PEs and no opcode in the array. Gemmini is the same -- its PEs are dumb and ``ExecuteController``
decodes -- and it also means the instruction set can grow without touching the
array.

Chains rather than fan-out
~~~~~~~~~~~~~~~~~~~~~~~~~~

The weight-stationary predecessor (``microarch_ws.py``, removed) had a ``loader``
writing ``a_in[0..T-1]`` and a ``drainer`` reading ``c_out[0..T-1]``, both in fixed
order. That design needs stream depth proportional to ``M * NI`` -- the entire
run -- which means back-pressure never engages (``RESULTS_WS.md`` section 6; both
files at ``e2451b81``). Upstream's ``test_multi_cache_gemm.py`` has no process of
that shape: ``offchip_loadA`` writes exactly one stream and the border PEs
daisy-chain it, ``L2_A[i] -> L2_A[i+1]``. ``test_tiled_systolic.py`` runs on
depth-**4** FIFOs for the same reason.

So every distribution here is a chain, and every chain carries *packed words*:

.. code-block:: text

    spm --hdr,W--> wcol[0] --v--> wld(i,0) --> wcol[i+1]     (down column 0)
                                    |    +--wq--> PE(i,0)    (its own weight)
                                    +-------> wrow[i,0] --> wld(i,1) --> ...
                                                             (east, lane j)

    vru --A------> acol[0] --v--> PE(i,0) --> acol[i+1]      (down column 0)
                                    |  lane i
                                    +-------> a_fwd[i,0] --> east, scalar

    PE(T-1,0) --cw[0]--> PE(T-1,1) --cw[1]--> ... --cw[T-1]--> accu
              (the bottom row packs its psums into one word as it goes east)

One word per cycle per link, so T lanes per cycle, from single-ported memories.

Hazards come from being in-order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each unit consumes the instruction stream in order and every channel is
point-to-point, so ``dma_ld`` before ``mm`` before ``vadd`` before ``mvout`` is
enforced by construction: within a unit by program order, across units by
stream order. That is the whole of the hazard logic *between* units. Inside
``accu`` one more rule holds, and the assembler enforces it: an accumulator
row may not be read within ``AR_RAW_DIST`` rows of being written
(:ref:`tinytpu-isa-dependence`). Gemmini spends a 48-entry
reservation station (``ReservationStation.scala``) to get out-of-order issue on
top of this; this design is strictly in-order and does not pretend otherwise.

Two structural rules that fixed real deadlocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both fixes were the same shape, and both match Gemmini's structure:

1. **One unit doing DMA in both directions deadlocks.** A single ``dma`` unit put
   ``spm`` in a two-process cycle (``dma -> dma2sp -> spm`` *and*
   ``spm -> sp2dma -> dma``). The round trip ``dma_ld; dma_st`` hung with **every**
   body variant tried -- constant trip counts on both sides, no ``meta_for``, no
   store, a constant put. Splitting into one-way ``dma_ld`` and ``dma_st`` units
   fixed it immediately and exactly. Gemmini splits the same way
   (``LoadController.scala`` / ``StoreController.scala``).
2. **Writing results back into the input scratchpad deadlocks.** The original
   ``vst`` (accumulator -> scratchpad) hung even when its body was reduced to
   ``ac2sp.put(123)`` -- no ``ar`` read, no packing, no clipping -- and with
   compile-time trip counts on both sides, and at ``QD=256``. The put side alone
   ran; the get side alone ran; together they hung. It could not be reduced to
   a minimal repro, so the mechanism is still unexplained.

   The fix was architectural: **make the accumulator the output memory.**
   ``mvout`` reads ``ar`` and writes DRAM directly, the scratchpad becomes
   input-only, and the back edge disappears. This is exactly Gemmini:
   ``AccumulatorMem`` is a separate memory from the scratchpad, and
   ``StoreController`` reads it directly -- results never re-enter the input
   scratchpad. It also collapsed two instructions (``vst`` + ``dma_st``) into
   one.

That an architecture-level property (where the output memory lives) is what
makes a design runnable, while the symptom is an unexplained silent hang, is the
strongest argument for a deadlock report in the simulator: naming the blocked
processes and the full channels would have replaced a long bisection with one
run. (The simulator has since gained a tier-0 watchdog; the per-channel report
is still open -- see :ref:`limitation-11`.)

One owner per memory
~~~~~~~~~~~~~~~~~~~~

``spad`` lives in ``spm`` and is written by ``dma_ld`` and read by ``vld`` and
``mm`` (weights); ``vr`` lives in ``vru`` and is written by ``dma_ld`` and
``vld`` and read by ``mm`` (activations); ``ar`` lives in ``accu`` and is
touched by all four of its opcodes. Allo enforces single reader
and single writer, and Vitis rejects the violation outright
(HLS 200-779 / 200-979), so those three units keep every arm of their opcode
dispatch in one process. This rules out a one-process-per-opcode split; see
:ref:`tinytpu-history-rowflat`.

.. note::

   **Correction, 2026-09-19: the one-owner rule is Allo's, not Vitis's.** The
   paragraph above charges it to both; only the Allo half holds. A Vitis
   2023.2 probe (`impact/probe_shared/
   <https://github.com/sunwookim028/allo/tree/main/examples/accelerator/tinytpu_vitis/impact/probe_shared>`__)
   shows
   that ``#pragma HLS stream variable=buf type=unsync`` makes Vitis share an
   on-chip array between two processes, one per BRAM port (``HLS 200-824``,
   ``200-755``, ``200-634``); ``HLS 200-779`` applies only to *synchronized*
   arrays. Allo is what refuses: it rejects a region-scope ``Stateful`` shared
   by two kernels (``EmitVivadoHLS.cpp:3083-3122``) and never emits
   ``stream type=unsync``. Vitis does separately forbid one ``m_axi`` bundle
   read by two processes (``HLS 200-1013`` / ``200-984``).

   **Measured impact on this design: 0 cycles** -- every restructure the gap
   attribution needed was Allo-legal (:ref:`gemmini-gap-attribution`). An
   earlier claim in this project, that removing ``vru``'s double handling of B
   would need a second producer on a shared memory, **was wrong**: the
   ``v_wdirect`` / ``v_wdb`` variants remove it with ``spm`` still the one owner
   of ``spad`` and one writer of ``wcol[0]``, and that is how the shipped
   design has done it since ``e24e433b``. See :ref:`limitation-shared-memory`.

Data type
~~~~~~~~~

int8 lanes, int32 accumulation, clipping to int8 on the way out -- Gemmini's
default config (``inputType = SInt(8.W)``, ``accType = SInt(32.W)``) and its
``mvout`` behaviour under ``ACC_SCALE_IDENTITY`` with shift 0, which is what
``allo_cmp.c`` passes. Packing is what makes the SIMD scratchpad work, and
packing needs integers, so unlike ``microarch_ws.py`` (at ``e2451b81``) there is
no fp32 switch.


The ISA
-------

Instruction format
~~~~~~~~~~~~~~~~~~

An instruction is **two 64-bit words** (``IWORDS = 2``). The first carries a
6-bit opcode and five fields; the second carries up to three address-generation
(AGU) terms.

.. code-block:: text

   word 0:  op [0:6]  f0 [6:18]  f1 [18:30]  f2 [30:42]  f3 [42:54]  nr [54:62]
   word 1:  up to 3 terms of (target 4b, level 3b, stride 12b), 19 bits each

.. list-table:: Opcodes (``microarch_isa.py``)
   :header-rows: 1

   * - opcode
     - value
     - fields
   * - ``OP_NOP``
     - 0
     -
   * - ``OP_DMA_LD``
     - 1
     - ``f0`` = src | dst << 1 (src: 0 = A, 1 = B; dst: 0 = ``spad``, 1 = ``vr``),
       ``f1`` = dram_row0, ``f2`` = col_block, ``f3`` = spad0 or vr0,
       ``nr`` = rows
   * - ``OP_DMA_ST``
     - 2
     - retired: results leave via ``OP_MVOUT``
   * - ``OP_VLD``
     - 3
     - ``f0`` = vr0, ``f1`` = spad0, ``nr`` = rows
   * - ``OP_MM``
     - 4
     - ``f0`` = vr_a (activations), ``f1`` = ar0, ``f2`` = acc, ``f3`` =
       spad_w (T weight rows), ``nr`` = rows
   * - ``OP_VADD``
     - 5
     - ``f0`` = ar_d, ``f1`` = ar_s1, ``f2`` = ar_s2, ``nr`` = rows
   * - ``OP_VRELU``
     - 6
     - ``f0`` = ar_d, ``f1`` = ar_s, ``nr`` = rows
   * - ``OP_MVOUT``
     - 7
     - ``f0`` = ar0, ``f1`` = dram_row0, ``f2`` = col_block, ``nr`` = rows (acc -> DRAM)
   * - ``OP_LOOP``
     - 8
     - open a loop, body is the next instruction; ``nr`` = trip count
   * - ``OP_ENDLOOP``
     - 9
     - close the innermost loop

**Control flow.** ``LOOP``/``ENDLOOP`` drive a ``LOOP_DEPTH = 4`` hardware loop
stack, as MiniTPU's (:doc:`minitpu`). The sequencer's loop is do-while, so a
trip count of 0 would run **once**; the generator refuses it (below).

**Address generation.** Each AGU term is ``(target, level, stride)`` and resolves
to ``field[target] += iv[level] * stride``, so an address can be relative to any
enclosing loop's induction variable. Terms name their target rather than being
fixed one-per-field, because a single field often needs two: the ``mm`` inside
the k loop names its weights at an offset by both the n tile and the k tile,
``B_SP + nb*MAXDIM + kb*T``, and a one-term-per-field encoding cannot say it.
Without this a loop body would reissue identical addresses every iteration and
simply redo the same work.

The spare-bit rule, and the signed bit-slice bug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every field carries one more bit than its value range needs. When the design
was built, a bit-slice was extracted into a **signed** ``ap_int<N>`` in the
emitted HLS:

.. code-block:: cpp

   ap_int<7> v268;  v268 = w02(60, 54);   // the nr field
   int32_t nr = v268;                     // 64 -> 0b1000000 -> -64

So a field whose top bit is set read back negative, and a loop bounded by it ran
zero times. Consolidating the ``vld`` broke 16x16x16 in cosim -- **251 of 256
outputs wrong** -- while the Allo dataflow simulator passed it: ``nr = 64`` for a
64-row ``vld`` read back as -64, the loop ran zero times, and the design silently
produced zeros. It appeared exactly at the field's sign-bit boundary:
``nr <= 63`` worked, 64 did not.

The lesson for the methodology: **an N-bit ISA field safely carries
0 .. 2^(N-1) - 1**, and the dataflow simulator did not model this, so it was a
genuine simulator/RTL divergence. It is the argument for keeping cosim in the
loop rather than running it once at the end -- this bug was invisible to every
functional check that had been passing. ``nr`` is therefore 8 bits for
``MAXROWS = 127``, the address fields are 12 bits for a 2047 maximum, AGU
targets are 4 bits for 0..4, levels 3 bits for 0..3, strides 12 bits for
0..2047, and ``enc()`` asserts ``0 <= v < 2^(w-1)`` for every field.

.. note::

   The emitter has since been fixed: the fork's ``3de74846`` ("hls: emit bit
   slices as unsigned", 2026-09-18) and upstream's ``094ab413`` (PR #612,
   "Preserve unsigned bit-slice types during HLS codegen") fix the same bug;
   ``main`` took upstream's implementation in merge ``dc6b8fa6`` (2026-09-19).
   The design's spare-bit encoding predates the fix and is unchanged. See
   :ref:`limitation-12`.

The ``nr`` width is also load-bearing for synthesis, not only for correctness;
see :ref:`tinytpu-isa-csynth-bound`.

Header and program memory
~~~~~~~~~~~~~~~~~~~~~~~~~

``imem[0:NHDR]`` (``NHDR = 8``) is a header of **dynamic per-unit work counts**;
instructions follow, two words each (``assemble()``):

.. code-block:: text

   imem[0] static instruction count   imem[4] mm count | mm rows << 16
   imem[1] dma_ld  rows               imem[5] accu   iterations
   imem[2] spm     rows               imem[6] dma_st rows
   imem[3] vru     rows               imem[7] A rows | B rows << 16

``imem[0]`` bounds the sequencer's fetch; every other count is dynamic, from
``expand()``, which runs the program's control flow at assembly time and
resolves the AGU exactly as the sequencer does. With a hardware loop the static
and dynamic counts differ, and **a unit promised more work than it receives
does not produce a wrong answer, it hangs** -- the one place where the
assembler and the microarchitecture are coupled. These are *work* counts, not
instruction counts, because every unit runs one flat loop over its rows:
``spm`` charges an ``mm`` ``T + 1`` rows for the header and weight words it
pushes, ``accu`` charges a ``vadd`` two iterations per row, and a ``dma_ld``
counts for ``spm`` or ``vru`` by its destination. ``imem[7]`` is the DRAM row
span ``dma_ld`` bursts for each operand matrix.

``IMEM_SIZE`` is ``NHDR + IWORDS * 24`` (the longest program shipped, plus
headroom). The program is pulled on-chip by one burst of ``IMEM_SIZE`` words
before anything runs, so imem size is startup time whether the program uses it
or not. Measured when the prefetch moved one word per cycle: moving to the
2-word instruction format cost exactly +68 cycles at all five shapes, exactly
the 68 extra words it added -- the loop logic itself cost nothing. Since
``e24e433b`` the prefetch moves **8 words per cycle** (gmem0 is 512 bits wide;
``ib`` is cyclically partitioned by 8), so the 56 words cost 7 cycles, not 56
-- worth 52 cycles at every shape (``v_imem8``). With control flow the program
is O(nesting), not O(tiles): the looped GEMM is at most 14 instructions (with
``relu``) at every shape, where the unrolled one reaches 32.

.. _tinytpu-isa-programs:

Programs, and the loop-nest generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three program forms are kept, and ``bench_isa.py`` checks them against each
other before building anything:

* ``isa_dsl.gemm_program`` -- the shipped program, from the loop-nest generator:
  A ``dma_ld``'d straight into the vregs, B into the scratchpad, and every
  ``mm`` naming its T weight rows there.
* ``microarch_isa.gemm_program_handwritten`` -- the hand-emitted reference. The
  first k-tile is peeled out of the k loop deliberately: it carries ``f2=0``
  (overwrite the accumulator) where the looped tiles carry ``f2=1``
  (accumulate), and peeling is how you say that without a predicate on the
  induction variable.
* ``microarch_isa.gemm_program_flat`` -- the fully unrolled differential
  reference, absolute addresses and no control flow.
* ``vadd_program`` computes ``relu(2 * (A @ B))`` on the first output tile, so
  ``vadd``/``vrelu`` stay exercised now that the GEMM inner loop no longer needs
  them.
* ``isa_dsl.vector_program`` and ``isa_dsl.ar_distance_program`` are test
  programs: the first varies every field GEMM holds constant (both ``dma_ld``
  destinations, ``vld``, distinct accumulator regions), the second sits exactly
  on the accumulator distance contract (:ref:`tinytpu-isa-dependence`).

``isa_dsl.py`` is **a code generator, not a compiler**. It chooses nothing:

.. list-table::
   :widths: 25 75

   * - the programmer writes
     - the tiling, the loop order, the layout, which k-tile is peeled, and which
       loop an address walks
   * - the generator derives
     - every ``agu_level``, every AGU term, every loop open/close pair, and the
       encoded instruction word
   * - the generator refuses
     - a nest deeper than the hardware loop stack, an induction variable used
       outside its own loop, and more address terms than one instruction can
       carry
   * - nobody decides
     - what the best tiling is -- there is no search here

It ports MiniTPU's mechanism (``board_package/dsl.py:126-141``, its ``loop()``):
**nesting depth IS the AGU level**, and the only handle on a level is the
induction variable the ``with`` yields:

.. code-block:: python

   with k.loop(Nt, "n") as nb:                       # level 0, derived
       k.mm(A_VR, AR_C, Ref(B_SP).at(nb, MAXDIM), rows=M, acc=False)  # peeled
       with k.loop(Kt - 1, "k") as kb:               # level 1, derived
           k.mm(Ref(A_VR + MAXDIM).at(kb, MAXDIM), AR_C,
                Ref(B_SP + T).at(nb, MAXDIM).at(kb, T), rows=M, acc=True)

Swap two ``with`` statements and every address term follows, because the
induction variables carry their levels with them. It is generalised to this
AGU rather than copied: MiniTPU's AGU is one term on one field as a
power-of-two shift, this one is three ``(target, level, stride)`` terms with
arbitrary 11-bit strides, any number of which may land on the same field --
``B_SP + nb*MAXDIM + kb*T`` is two terms on ``f3`` and their encoding cannot say
it. So ``Ref`` is a base plus an ordered list of ``(iv, stride)`` terms and any
field may be one.

**It emits the identical instruction stream, and that is the whole test.**
``isa_dsl.assert_matches_handwritten`` compares both 64-bit words of every
instruction against ``gemm_program_handwritten`` at all five shapes and both
relu settings; ``bench_isa.py`` runs it before it builds anything. Bit-identical
means cosim cannot move, so it was not re-run.

**What it bought.** Three errors that were previously expressible are not: a
level that disagrees with the nest, an induction variable used after its loop
closed (the sequencer would resolve it against a stale ``iv_now[level]``), and a
``loop`` whose ``endloop`` was forgotten. Two more are now caught at the point
of writing rather than at ``enc_agu``: a nest deeper than ``LOOP_DEPTH``, named
("the nest is n > k > m > j > i"), and a trip count of 0, which the do-while
sequencer would run **once**.

**What it cost.** It is ~120 lines of machinery to remove four typed integers
from a 40-line program, and at this size the hand-written form was not actually
hard to keep right -- the trade only pays once there is more than one program.
It also trades one hand-maintained correspondence for another: the instruction
methods (``vld(vr, spad, rows)``) restate the field layout that the opcode table
comments give. The new correspondence is stated once per opcode instead of once
per instruction and the bit-identity assertion checks it, which is why it is the
better of the two, but it is not free. And the genuinely subtle part of the
program -- the peeled first k-tile, and the ``B_SP + T`` base that encodes "kb
starts at 1" -- is exactly as subtle as it was.

Programs are written against this encoder (by hand, or through ``isa_dsl.py``)
because the compiler backend that lowers a TOSA matmul into these instructions
lives only on the ``chia-codesign`` branch; see :doc:`/extensions/act`.


Hardware parameters
-------------------

Every constant is fixed at build time and **independent of the workload**: one
RTL build runs every shape, with M, K and N arriving as instruction fields. This
is the property the Gemmini comparison needs -- Gemmini's numbers come from one
elaboration, and ``allo_cmp.c`` passes ``MAXDIM`` as the stride for every shape.

.. list-table::
   :header-rows: 1

   * - constant
     - default
     - environment variable
     - meaning
   * - ``T``
     - 4
     - ``TPU_T``
     - SIMD width == array dimension; ``T >= 4``
   * - ``MAXDIM``
     - 16
     - ``TPU_MAXDIM``
     - largest M, K, N supported; A/B/C are flat ``int8[MAXDIM*MAXDIM]``
   * - ``SPAD_ROWS``
     - 512
     - ``TPU_SPAD``
     - scratchpad rows, each one packed word
   * - ``NVR``
     - 256
     - ``TPU_NVR``
     - operand vector registers
   * - ``NAR``
     - 128
     - ``TPU_NAR``
     - accumulator vector registers
   * - ``QD``
     - 8
     - ``TPU_QD``
     - stream depth
   * - ``IMEM_SIZE``
     - ``NHDR + 2*24`` = 56
     - ``TPU_IMEM``
     - instruction memory words

``T`` is the one parameter that changes the *shape* of the generated region: the
array is ``T*T`` kernel instances, so ``T=16`` is 262 instances and ~800 streams.
That was unrunnable until the simulator's OpenMP team was sized to the section
count (:ref:`limitation-11`); before that fix it hung with no output.

``A``, ``B`` and ``C`` are **flat** at the region boundary, addressed
``row * MAXDIM + col``. That is what DRAM is, and it is what makes
``wrap_io=False`` legal -- it refuses multi-dimensional arguments to nested
kernels (:ref:`limitation-14`). ``schedule()`` partitions ``A``, ``B`` and ``C``
cyclically by ``T``; this is reachable because ``df.build`` is
``customize(func)`` followed by ``s.build(...)`` (:ref:`limitation-17`).

The build is ``wrap_io=False`` with ``configs={"align_value": 64}`` and
``config_interface -m_axi_max_widen_bitwidth 512``; how those came to be the
build is on :doc:`tinytpu_history`, and the Vitis-side mechanics are in
:doc:`/backends/vitis`.


Building and running
--------------------

Files in ``examples/accelerator/tinytpu_vitis/``:

.. list-table::
   :widths: 25 75

   * - ``microarch_isa.py``
     - the design, the encoder, ``assemble()``/``expand()``, the reference
       programs, ``schedule()``
   * - ``isa_dsl.py``
     - the loop-nest generator (``gemm_program``)
   * - ``reproduce.sh``
     - one command from a clean checkout to the published cycle counts (see
       :ref:`tinytpu-isa-verify`)
   * - ``bench_isa.py``
     - functional sweep on the Allo dataflow simulator, one build -- the
       **performance** setup
   * - ``stress_isa.py``
     - the functional **correctness** gate (full-range operands, every shape,
       prefilled ``C``, non-GEMM and random programs, the validator)
   * - ``isa_ref.py``
     - the ISA as numpy: the reference for any program, not just GEMM
   * - ``mutate.py``
     - mutation testing of the verification harness
   * - ``cosim.py``
     - one csynth, then Vitis ``cosim`` per shape on the same RTL
   * - ``kpn_model.py``
     - a KPN model of the channel graph with bounded FIFOs and deadlock
       reporting
   * - ``logs/``
     - the csynth/cosim reports and sweep logs behind the numbers on these
       pages
   * - ``gemmini/``
     - the patches and benchmarks for the matched Gemmini baseline
       (:ref:`gemmini-reproduce`)
   * - ``impact/``
     - the gap attribution's variants (generated from the pre-landing
       baseline), their raw results and timelines, the Vitis shared-array
       probe, and the RTL probe of the accumulator's dependence claim
       (:ref:`gemmini-attribution-reproduce`)

.. code-block:: bash

   # One command, from a clean checkout: builds this checkout's MLIR bindings,
   # runs bench_isa + stress_isa, then the default cosim, and checks the five
   # cycle counts against 172 / 262 / 418 / 484 / 686. Exits nonzero otherwise.
   examples/accelerator/tinytpu_vitis/reproduce.sh            # ~6 min, incl. a fresh mlir build
   examples/accelerator/tinytpu_vitis/reproduce.sh --no-cosim # functional, ~1 min

   # Or by hand, from examples/accelerator/tinytpu_vitis (the env sets neither variable):
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
   python bench_isa.py                   # published functional setup: ALL EXACT
   python bench_isa.py 8 8 8             # one shape
   python stress_isa.py                  # correctness gate:          STRESS OK
   python kpn_model.py                   # channel protocol / deadlock model
   python cosim.py                       # default TB: the published cycle counts
   TPU_TB=stress TPU_SHAPES=4x4x4,16x16x16 python cosim.py   # correctness in RTL
   python mutate.py                      # does the harness catch a broken design? (~15 min)
   python mutate.py --no-rtl             # the same without the one cosim (~5 min)
   TPU_WRAP=1 python cosim.py            # the old hoisted-argument variant, for comparison

``bench_isa.py`` and ``cosim.py``'s default testbench use Gemmini's ``[-4, 4]``
operands so the comparison is like for like; they are **performance** checks
and are blind to an int16 accumulator, a wrong clip boundary, or a unit ignoring
a field GEMM never varies. ``stress_isa.py`` and ``TPU_TB=stress`` are the
**correctness** checks; ``mutate.py`` prints which level catches which bug. Any
``TPU_*`` variable left set changes what ``cosim.py`` measures.

These commands replaced older reproduce instructions (``d18de251``) that named
``TPU_M`` / ``TPU_K`` / ``TPU_N`` knobs, a ``simulator`` argument and
``OMP_NUM_THREADS=32``, none of which the current scripts use. Since the fix
recorded in :ref:`limitation-11` the simulator sizes its OpenMP team to the
section count itself, so ``OMP_NUM_THREADS=8`` runs the 38-process design; the
recorded numbers were produced at 32. ``kpn_model.py`` has been rewritten for the
row-flattened units and is driven by the assembled header (``ef112868``); every
shipped program completes in it at FIFO depth **1**. See
``dev/toolchains.rst`` for ``LLVM_BUILD_DIR`` and the Vitis install.

``bench_isa.py`` first asserts that the generated program is bit-identical to
the hand-written one, then that the looped and unrolled programs expand to the
identical dynamic stream (every resolved address field, not just the opcode and
row-count stream), then builds the simulator **once** and runs ``gemm`` and
``gemm.relu`` in both looped and flat forms at every shape, plus
``vadd_program``, against a numpy reference *including the int8 clip*. It ends
``ALL EXACT`` or ``FAILURES``.

``cosim.py`` knobs, all environment variables:

.. list-table::
   :widths: 30 70

   * - ``TPU_SHAPES=4x4x4,16x16x16``
     - restrict the sweep (default: every shape in 4x4x4, 8x8x8, 12x12x12,
       16x16x8, 16x16x16 that is a multiple of ``T``)
   * - ``TPU_AXI_LATENCY=<n>``
     - add ``config_interface -m_axi_latency <n>`` to the csynth script
       (default: unset, i.e. Vitis's default of 0)
   * - ``TPU_RANDOM_STALL=1``
     - pass ``-random_stall`` to ``cosim_design``
   * - ``TPU_WRAP=1``
     - build with ``wrap_io=True`` (the hoisted-argument variant)
   * - ``TPU_TB=stress``
     - the **correctness** testbench on the same RTL (below); unset is the
       default performance testbench, the only mode the published cycle counts
       come from

The toolchain fixes ``cosim.py`` applies (a plain C++ testbench, ``-B/usr/bin``,
explicit ``m_axi`` depths, ``alignas(64)`` arrays) are general to any Allo
dataflow design and are documented in :doc:`/backends/vitis`.

.. _tinytpu-isa-verify:

Verifying a change
~~~~~~~~~~~~~~~~~~

Added on ``main`` on 2026-09-19. From a clean checkout, one command builds the
checkout's own bindings, runs the functional gates, runs cosim, and checks the
published cycle counts (172 / 262 / 418 / 484 / 686), exiting nonzero if any
step fails or any number differs:

.. code-block:: bash

   examples/accelerator/tinytpu_vitis/reproduce.sh              # everything, with cosim
   examples/accelerator/tinytpu_vitis/reproduce.sh --no-cosim   # functional only, ~1 min

``reproduce.sh`` was run from a pristine worktree, including a fresh MLIR
build, and printed ``REPRODUCED`` with 172 / 262 / 418 / 484 / 686 and 0
mismatches in **5m49s** (``96c3aef6``, re-run at the docs commit); against the pre-landing
design it printed 252 / 383 / 591 / 667 / 919 in 5m48s (``d18de251``). It
unsets every ``TPU_*`` knob and checks that ``allo`` resolves to the checkout
it is run from (``92fb2f1b``).

**Performance vs. correctness.** ``bench_isa.py`` and ``cosim.py`` with its
default testbench are the **performance** setup: seed 0, operands in [-4, 4]
(Gemmini's ``allo_cmp.c`` distribution, kept so the comparison is like for
like), ``C`` zeroed, and only the ``M x N`` region compared. They miss real
bugs -- at T=4 a PE's partial sum never leaves 9 bits, so narrowing the int32
partial sum to int16 still prints ``ALL EXACT``. The correctness gates are
below. Run ``stress_isa.py`` after **any** change to ``microarch_isa.py``, and
``mutate.py`` after any change to the harness.

``stress_isa.py`` (``ef112868``)
   About 10 s on the Allo simulator, 492 runs: full-range int8 operands with
   -128 and 127 forced; clip and ReLU edge cases landing on 127 / 128 / -128 /
   -129 / 0 / -1; all 64 shapes; ``C`` prefilled with random bytes and compared
   in full; ``isa_dsl.vector_program``, which varies every field GEMM holds
   constant; ``isa_dsl.ar_distance_program`` at the accumulator's distance
   contract; and 200 random valid programs, generated to keep that contract.
   Everything is checked against ``isa_ref.py``, a numpy reference model of the
   ISA that is itself checked against numpy on every GEMM. It also runs the
   validator's negative controls (18 crafted bad programs rejected, 390
   generated ones accepted) and runs ``kpn_model.py`` on every distinct program
   first, so a simulator hang becomes a named report. It prints ``STRESS OK:
   n/n`` or lists each failing run.

``TPU_TB=stress python cosim.py`` (``1ac0d22f``)
   The same idea on the RTL: six calls per shape in one RTL simulation (corner,
   full-range, boundary and mid operands, ``vector_program``, and
   ``ar_distance_program(AR_RAW_DIST)`` -- the one case only RTL can fail),
   each inheriting the previous call's state, with ``C`` prefilled and compared
   in full. Its cycle column is the minimum over those calls, not the headline. The
   default testbench is unchanged byte for byte and is the only mode the
   published cycle counts come from.

**Proof that the split matters.** The int16 partial-sum mutant:

.. list-table::
   :header-rows: 1

   * - check
     - int16 partial-sum mutant
   * - default cosim (performance testbench)
     - **passes**: 0/16 and 0/256 mismatches, at the then-published 252 / 919
       cycles (measured on the pre-landing design)
   * - ``TPU_TB=stress`` cosim
     - **fails**: 3 wrong at 4x4x4, 82 at 16x16x16
   * - ``stress_isa.py``
     - **fails**: 247 of 486 runs (pre-landing); 241 of 492 on the landed design

``mutate.py`` (``c8089332``; re-anchored and extended in ``e24e433b``)
   34 single-point mutants of ``microarch_isa.py`` (plus the unmodified
   ``none`` control through the same loader), each run through ``bench_isa``
   and ``stress_isa``, and through the ``TPU_TB=stress`` cosim for the one
   RTL-only mutant (and for any mutant on request). **All 34 are caught**
   (``logs/mutate_landed.log``). 20 fail ``bench_isa``. 13 get past it and are
   caught **only** by ``stress_isa``: ``pe_psum_int16`` (int16 partial sum),
   ``clip_hi_off_by_one`` and ``clip_lo_off_by_one`` (both clip bounds),
   ``vadd_dst_is_src1`` and ``vrelu_dst_is_src`` (vadd/vrelu destination),
   ``vadd_src2_is_src1`` (vadd second source), ``vrelu_src_base_ignored``,
   ``mvout_src_base_ignored``, ``dma_ld_row_ignored`` and
   ``mvout_row_ignored`` (base / row fields), ``spm_vld_off_by_one`` (``vld``,
   which the shipped GEMM no longer issues), ``dma_st_accumulates_C``
   (``dma_st`` relying on a zeroed ``C``), and ``ar_contract_unenforced``
   (the assembler no longer enforcing the accumulator distance contract). One,
   ``ar_claim_false``, passes both functional levels by construction and is
   caught **only** by cosim; see :ref:`tinytpu-isa-dependence`.

   The mutants that target the landed code: ``pe_shadow_not_swapped`` and
   ``wld_rows_from_weight`` (the weight double buffer), ``spm_weight_off_by_one``
   (weights from the scratchpad), ``vru_dma_ignores_f3`` and ``vru_act_off_by_one``
   (the A path into and out of the vregs), ``prefetch_lane7_dup`` (the 8-wide
   prefetch), ``vadd_holds_stale_x`` (``vadd``'s two-iteration rows), and the
   two dependence mutants. The pre-landing mutants were re-anchored on the
   equivalent code; ``vrelu_loop_short`` became ``vrelu_rows_short`` (there is
   no per-instruction loop left to shorten). Each mutant's anchor must match
   exactly once, so a refactor of the design makes the script fail loudly
   instead of silently testing nothing.

**The arrays are not cleared by hardware.** ``spad``, ``vr`` and ``ar`` carry no
``= 0`` initialiser (it cost a runtime zero-fill loop; see
:ref:`limitation-g`), so ``assemble()`` enforces a write-before-read contract
(``4357ed59``): ``check_program()`` walks the exact dynamic trace -- the program
has no data-dependent control flow, so this is exact, not conservative -- and
rejects any read of an ``ar`` row (``mm`` accumulate, ``vadd``, ``vrelu``,
``mvout``), a ``vr`` row (``mm`` activations) or a ``spad`` row (``mm``
weights) that no earlier instruction wrote. Written-ness propagates through
``vld``: a ``vld`` may copy an unwritten ``spad`` row, but the copy then counts
as unwritten. The same walk enforces the accumulator distance contract
(:ref:`tinytpu-isa-dependence`). It also rejects out-of-range rows, bad loop nesting
(unbalanced or over-deep loops, trip 0, AGU terms naming a closed loop),
zero-row instructions, and the retired ``dma_st`` opcode.

Two design facts the hardening found:

* **Zero-row instructions hang the machine** instead of being no-ops: an
  ``nr=0`` data op desynchronises the flat row loops (confirmed on the
  simulator). The validator now rejects them.
* **The hung simulator ignores SIGTERM**; only SIGKILL stops it (it is inside
  a blocking C call; compare the watchdog note in :ref:`limitation-11`).

.. _tinytpu-isa-dependence:

The accumulator's dependence claim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``accu`` is one flat row loop at II=1 because ``schedule()`` tells Vitis that
its accumulator carries no dependence across iterations:

.. code-block:: python

   s.dependence("accu_0:x", "ar", dep_type="inter", dependent=False)
   # -> #pragma HLS dependence variable=ar inter false   (inside accu's row loop)

``s.dependence`` is the schedule primitive added for :ref:`limitation-21`
(``bbea2af0``). Without the claim the flat loop closes at ``Final II = 3``:
the row index is a carried register, so Vitis cannot prove that iteration n's
store and iteration n+1's load of ``ar`` touch different rows.

**The claim is not true of the hardware on its own**, and the branch that
priced it (``v_accudep`` / ``v_design_dep``, which injected the pragma into
``kernel.cpp``) never tested where it fails: only GEMM programs were cosimulated
there. In the synthesized loop (II=1, depth 6) the ``ar`` load issues in
pipeline state 5 and the store lands in state 7, so a row read one or two
iterations after it was written returns its **old** value. Measured in RTL
with ``isa_dsl.ar_distance_program(d)``, every read exactly ``d`` iterations
after its write (``logs/cosim_isa_ar_distance.log``,
``impact/ar_distance_probe.py``):

.. list-table::
   :header-rows: 1

   * - distance ``d`` (accu iterations)
     - 1
     - 2
     - 3
     - 4
     - 5
   * - cells of ``C`` wrong, RTL
     - **4**
     - **20**
     - 0
     - 0
     - 0
   * - cells wrong, C simulation
     - 0
     - 0
     - 0
     - 0
     - 0

So the claim is made true by the **assembler**, the way the write-before-read
contract is: ``check_program()`` counts ``accu`` iterations (one per ``mm``,
``vrelu`` or ``mvout`` row, two per ``vadd`` row) and rejects any read of an
``ar`` row fewer than ``AR_RAW_DIST = 4`` iterations after the write it
depends on -- the first safe distance plus one of margin, and exactly ``T``,
the row count of the smallest GEMM, so no GEMM is constrained by it. The
random-program generator keeps the contract, and the ``TPU_TB=stress`` cosim
runs ``ar_distance_program(AR_RAW_DIST)`` on every build, so a re-synthesis
that widened the window fails there.

That test has teeth. ``mutate.py``'s ``ar_claim_false`` sets ``AR_RAW_DIST =
1``: the pragma is then false for programs the assembler accepts. It passes
``bench_isa`` and ``stress_isa`` -- no simulator models a dependence pragma --
and the stress cosim catches it (``ar_distance(1)``: 4 cells wrong).
``ar_contract_unenforced`` (the check disabled) is caught by ``stress_isa``'s
validator controls.

The alternative claim, ``inter RAW distance=4 true``, would have Vitis honour a
distance of 4 itself rather than rely on the schedule; it was not measured.


Results
-------

Cycle counts (current build)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measured by Vitis ``cosim`` (xsim), one build, ``-m_axi_latency 0``
(``logs/cosim_isa_landed_sweep.log``, ``e24e433b``), against the pre-landing
build that was shipped until then (``logs/cosim_isa_widened_sweep.log``):

.. list-table::
   :header-rows: 1

   * - shape
     - dynamic instructions
     - cycles
     - mismatches
     - pre-landing (instructions, cycles)
     - saved
   * - 4x4x4
     - 4
     - **172**
     - 0/16
     - 6, 252
     - 80 (32%)
   * - 8x8x8
     - 10
     - **262**
     - 0/64
     - 15, 383
     - 121 (32%)
   * - 12x12x12
     - 18
     - **418**
     - 0/144
     - 28, 591
     - 173 (29%)
   * - 16x16x8
     - 16
     - **484**
     - 0/128
     - 25, 667
     - 183 (27%)
   * - 16x16x16
     - 28
     - **686**
     - 0/256
     - 45, 919
     - 233 (25%)

The two shapes the gap attribution measured (172 and 686, :ref:`gemmini-gap-attribution`)
reproduce exactly; the other three were first measured on this build.
``TPU_TB=stress`` cosim at 4x4x4 and 16x16x16: 0 wrong over 6 calls each
(``logs/cosim_isa_landed_stress.log``).

Utilization against the 4x4 array's peak is 2.3% / 12.2% / 25.8% / 26.4% /
37.3% (was 1.6% / 8.4% / 18.3% / 19.2% / 27.9%). Least squares against dynamic
instruction count gives fixed **74.5**, marginal **21.70** cycles per dynamic
instruction (was 151 and 17.28); the two fits are not comparable term by term,
because the landing removed instructions (the ``vld``\ s) as well as cycles,
so each remaining instruction carries more work. Against MACs instead: fixed
**192**, **7.95** MAC/cycle marginal, 49.7% of peak (was 289 and 6.17, 38.6%).

These numbers were measured with a memory that answers immediately; the
latency sweep (``-m_axi_latency`` 16 and 64: 214 / 702 and 358 / 894 at
4x4x4 / 16x16x16) and what a real memory system does to them is on
:doc:`gemmini_comparison`. **They are not faster than Gemmini**: 1.07-1.24x
slower like for like.

How the design got from 1017 cycles at 4x4x4 to 172 is on
:doc:`tinytpu_history`. What the last step was, where it came from, and what
deficit to Gemmini remains is on :ref:`gemmini-gap-attribution`.

.. _tinytpu-isa-landing:

What landed in ``e24e433b``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gap attribution priced the deficit to Gemmini on variants of the design
(:ref:`gemmini-gap-attribution`); its best bit-exact stack,
``v_design_dep_imem8``, is now the design, rebuilt from the sources rather than
patched:

* **Program prefetch, 8 words a cycle** into a cyclically partitioned ``ib``
  (gmem0 is 512 bits wide): 56 words in 7 cycles instead of 56 (``v_imem8``).
* **Weights by scratchpad address.** ``mm``'s ``f3`` names T scratchpad rows
  and ``spm`` streams the header and weights down the weight chain; the weight
  ``vld`` through the vregs is gone. A **per-PE weight loader** (``wld``)
  double-buffers each PE's weight, and the PE is one flat loop (``v_wdb``).
* **A straight into the vregs.** ``dma_ld``'s ``f0`` gained a destination bit;
  the shipped GEMM sends A to the vregs, so the A ``spad -> vld -> vr`` trip is
  gone (``v_design``). ``vld`` stays in the ISA.
* ``accu`` **flat at II=1**, held there by the ``s.dependence`` claim and the
  distance contract that makes it true (:ref:`tinytpu-isa-dependence`), not by
  patching ``kernel.cpp`` (``v_accudep``).
* **Per-unit work counts precomputed in the sequencer**: a flat loop whose row
  count depends on the decoded opcode closes at ``Final II = 2`` on its
  counter -- the trap the branch found in ``spm`` and ``accu``.

.. _tinytpu-isa-csynth-bound:

A programmable design's csynth number is not a cycle count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vitis_hls`` at 8x8x8: **0 errors**, ``dataflow`` at the top, all 16 PEs
instantiated as separate modules. But the first run reported a top-level latency
of **91407 cycles**, against 74 for the fixed-function ``microarch_ws.py``, and
the report says why:

.. code-block:: text

   o VITIS_LOOP_300_1   Trip = 1023   Pipelined = yes
   o VITIS_LOOP_474_4   Trip = 2049

Nothing is slow. **Vitis bounds a runtime-bounded loop by the range of its
index**, and the row count was arriving in a 12-bit instruction field, so it
assumed up to 4095 rows per instruction. The number is a worst-case bound
derived from the *encoding*, not a property of the machine.

For a fixed-function design, csynth's interval *is* the answer
(``microarch_ws.py``, at ``e2451b81``, hit its roofline exactly and could be
checked statically). For an instruction-programmable design, **trip counts are
data**, so static estimates become bounds and only ``cosim`` gives a cycle
count. Any comparison against Gemmini's measured ``rdcycle`` has to be a cosim
comparison.

The bound is still worth tightening, because it is the ISA's fault rather than
the tool's: ``nr`` became a dedicated row-count field (``MAXROWS = 127``) used by
every instruction, instead of a 12-bit general field, and the header that
carries the row count into the array was narrowed to match. Gemmini does the
same -- its mvin/mvout carry an explicit bounded row count. **Narrowing one
field, with no change to any datapath, moved the bound 22x:**

.. list-table::
   :header-rows: 1

   * -
     - before (12-bit count)
     - after (7-bit ``nr``)
   * - top-level latency
     - 91407
     - **4133**
   * - top-level interval
     - 90333
     - **3059**
   * - reported trip counts
     - 1023, 2049
     - **63**

The extreme case is the ``T=16`` build: csynth reports a top-level latency of
**2.259e+08**, cosim measures **1,176** -- a ratio of about 192,000x. A bound is
not a measurement. The same lesson, stated as a general rule, is
:ref:`limitation-16`.

Resources
~~~~~~~~~

What the landing cost, csynth on the xcu280 at the 3.33 ns target, the
pre-landing design re-synthesized with the same toolchain
(``logs/csynth_isa_prelanding.rpt``, ``logs/csynth_isa_landed.rpt``). The
estimated clock is **2.431 ns** for both, so the design still meets 3.33 ns
with the same margin. The last row is the remainder, and ``entry_proc`` is in
it (6,363 FF = 6,360 + 3; 7,809 LUT = 7,780 + 29), so a re-derivation from the
report's per-module rows matches.

.. list-table::
   :header-rows: 1

   * - unit
     - BRAM
     - DSP
     - FF
     - LUT
   * - ``sequencer``
     - 0 -> 0
     - 3 -> 2
     - 2,058 -> 2,275
     - 2,312 -> 3,313
   * - ``dma_ld``
     - 0 -> 0
     - 0 -> 0
     - 574 -> 575
     - 1,075 -> 1,094
   * - ``spm``
     - 1 -> 1
     - 0 -> 0
     - 147 -> 452
     - 417 -> 812
   * - ``vru``
     - 1 -> 1
     - 0 -> 0
     - 221 -> 484
     - 725 -> 544
   * - ``wld`` x16 (new)
     - 0
     - 0
     - 1,250
     - 4,472
   * - ``pe`` x16
     - 0 -> 0
     - 12 -> 12
     - 4,019 -> 3,996
     - 5,175 -> 6,264
   * - ``accu``
     - 4 -> 4
     - 0 -> 0
     - 1,250 -> 1,744
     - 2,393 -> 1,764
   * - ``dma_st``
     - 0 -> 0
     - 0 -> 0
     - 342 -> 342
     - 511 -> 511
   * - FIFOs, ``entry_proc`` and top level
     - 36 -> 36
     - 0 -> 0
     - 6,277 -> 6,363
     - 7,358 -> 7,809
   * - **total**
     - **42 -> 42**
     - **15 -> 14**
     - **14,888 -> 17,481** (+17%)
     - **19,966 -> 26,583** (+33%)

The speedup costs **+2,593 FF and +6,617 LUT**, about 0.1% and 0.5% of the
xcu280, and no BRAM or DSP. The largest item is the 16 weight loaders (1,250
FF, 4,472 LUT); ``accu``'s dependence claim costs 494 FF over the nested loop
it replaced -- against **16,188** for the write-behind rotation that reached
the same II (:ref:`limitation-21`).

The resource figures the sources record for earlier builds are listed with the
build they belong to on :doc:`tinytpu_history`: 8x8x8 csynth after the ``nr``
narrowing (BRAM 42, DSP 12, FF 16247, LUT 21918), the one-build row-flattened
design (BRAM 16, DSP 15, FF 12835, LUT 18968), and the ``T=16`` build (240 DSP,
111,776 FF, 144,329 LUT -- 11% of the xcu280 -- 34 BRAM, all 256 PEs
instantiated). DSP is low at ``T=4`` because int8 multiplies mapped into LUTs;
that is a mapping difference, not a missing array.

Cosim as a deadlock oracle
~~~~~~~~~~~~~~~~~~~~~~~~~~

Cosim compiles an ``AESL_deadlock_detect_unit`` into the RTL testbench, so
**Vitis cosim already has deadlock reporting** of the kind the Allo simulator
lacked, which makes it a useful oracle for the class of bug in
:ref:`limitation-11`. ``csim`` passing is the first functional check of the
*emitted HLS code* rather than of the Allo simulator's interpretation of the
design.
