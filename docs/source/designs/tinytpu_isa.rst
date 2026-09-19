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

   Headline, as of 2026-09-18: one hardware build runs every shape as data, all
   five benchmark shapes are bit-exact in RTL co-simulation, and the design
   takes **252 / 383 / 591 / 667 / 919** cycles at 4x4x4 / 8x8x8 / 12x12x12 /
   16x16x8 / 16x16x16 (Vitis ``cosim``, ``-m_axi_latency 0``). Measured over
   the same window as ours, Gemmini is **1.55-1.8x faster** at all five shapes;
   see :doc:`gemmini_comparison`.


Architecture
------------

.. code-block:: text

           imem ─► sequencer ──(decoded instruction)──► every unit
     A,B ──► dma_ld ─► spm[spad] ─► vru[vr] ─► 4x4 WS array ─► accu[ar] ─► dma_st ─► C

Seven units, ``T*T + 6 = 22`` concurrent processes at ``T=4``:

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
     - DRAM -> scratchpad, packing T lanes/cycle
   * - ``spm``
     - ``spad``
     - the scratchpad; **pure SIMD access**
   * - ``vru``
     - ``vr``
     - operand vregs; drives the array's ports
   * - ``pe`` x16
     - its lane
     - weight-stationary MAC, decodes nothing
   * - ``accu``
     - ``ar``
     - accumulator vregs **and the vector ALU**
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

**The PEs decode nothing.** A header word leads every instruction down the same
chain the weights use, carrying just ``is_mm`` and ``nrows``. A PE forwards it and
acts on it; there is no command fan-out to ``T*T`` PEs and no opcode in the
array. Gemmini is the same -- its PEs are dumb and ``ExecuteController``
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

Hazards come from being in-order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each unit consumes the instruction stream in order and every channel is
point-to-point, so ``vld`` before ``mm`` before ``vadd`` before ``mvout`` is
enforced by construction: within a unit by program order, across units by
stream order. That is the whole of the hazard logic. Gemmini spends a 48-entry
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

``spad`` lives in ``spm`` and is written by ``dma_ld`` and read by ``vld``; ``vr``
lives in ``vru`` and is written by ``vld`` and read by ``mm``; ``ar`` lives in
``accu`` and is touched by all four of its opcodes. Allo enforces single reader
and single writer, and Vitis rejects the violation outright
(HLS 200-779 / 200-979), so those three units keep every arm of their opcode
dispatch in one process. This rules out a one-process-per-opcode split; see
:ref:`tinytpu-history-rowflat`.

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
     - ``f0`` = src (0=A, 1=B), ``f1`` = dram_row0, ``f2`` = col_block, ``f3`` = spad0, ``nr`` = rows
   * - ``OP_DMA_ST``
     - 2
     - retired: results leave via ``OP_MVOUT``
   * - ``OP_VLD``
     - 3
     - ``f0`` = vr0, ``f1`` = spad0, ``nr`` = rows
   * - ``OP_MM``
     - 4
     - ``f0`` = vr_a, ``f1`` = ar0, ``f2`` = acc, ``f3`` = vr_w, ``nr`` = rows
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
fixed one-per-field, because a single field often needs two: the weight ``vld``
inside the k loop is offset by both the n tile and the k tile,
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
   imem[1] dma_ld  rows               imem[5] accu   INSTRUCTIONS
   imem[2] spm     rows               imem[6] dma_st rows
   imem[3] vru     words              imem[7] A rows | B rows << 16

``imem[0]`` bounds the sequencer's fetch; every other count is dynamic, from
``expand()``, which runs the program's control flow at assembly time and
resolves the AGU exactly as the sequencer does. With a hardware loop the static
and dynamic counts differ, and **a unit promised more work than it receives
does not produce a wrong answer, it hangs** -- the one place where the
assembler and the microarchitecture are coupled. These are *work* counts (rows
or words), not instruction counts, because every unit but ``accu`` runs one
flat loop over its rows; ``vru`` charges an ``mm`` an extra ``T + 1`` words for
the header and weight words it pushes. ``imem[7]`` is the DRAM row span
``dma_ld`` bursts for each operand matrix.

``IMEM_SIZE`` is ``NHDR + IWORDS * 24`` (the longest program shipped, plus
headroom). This is not cosmetic: the program is pulled on-chip by one burst of
``IMEM_SIZE`` words, so every imem word is a startup cycle whether the program
uses it or not -- 56 cycles today. Measured: moving to the 2-word instruction
format cost exactly +68 cycles at all five shapes, which is exactly the 68
extra words it added -- the loop logic itself cost nothing. With control flow
the program is O(nesting), not O(tiles): the looped GEMM is 17 instructions at
every shape where the unrolled one reaches 49.

.. _tinytpu-isa-programs:

Programs, and the loop-nest generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three program forms are kept, and ``bench_isa.py`` checks them against each
other before building anything:

* ``isa_dsl.gemm_program`` -- the shipped program, from the loop-nest generator.
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
       k.vld(W_VR, Ref(B_SP).at(nb, MAXDIM), rows=T)
       k.mm(A_VR, AR_C, W_VR, rows=M, acc=False)     # peeled: overwrite
       with k.loop(Kt - 1, "k") as kb:               # level 1, derived
           k.vld(W_VR, Ref(B_SP + T).at(nb, MAXDIM).at(kb, T), rows=T)

Swap two ``with`` statements and every address term follows, because the
induction variables carry their levels with them. It is generalised to this
AGU rather than copied: MiniTPU's AGU is one term on one field as a
power-of-two shift, this one is three ``(target, level, stride)`` terms with
arbitrary 11-bit strides, any number of which may land on the same field --
``B_SP + nb*MAXDIM + kb*T`` is two terms on ``f1`` and their encoding cannot say
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
   * - ``bench_isa.py``
     - functional sweep on the Allo dataflow simulator, one build
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

.. code-block:: bash

   source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build   # not set by the env
   export PYTHONPATH=<repo root>
   export OMP_NUM_THREADS=32   # >= 22 processes; the value the numbers were produced with
   cd examples/accelerator/tinytpu_vitis

   python bench_isa.py          # functional sweep, one build, every shape
   python bench_isa.py 8 8 8    # one shape
   python kpn_model.py          # channel-graph model
   python cosim.py              # one csynth, cosim per shape
   TPU_WRAP=1 python cosim.py   # the old hoisted-argument variant, for comparison

Since the fix recorded in :ref:`limitation-11` the simulator sizes its OpenMP
team to the section count itself, and the 22-process design runs every shape at
``OMP_NUM_THREADS=8``; 32 is kept above because it is the setting the recorded
numbers used. See :doc:`/developer/toolchains` for ``LLVM_BUILD_DIR`` and the
Vitis install.

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

The toolchain fixes ``cosim.py`` applies (a plain C++ testbench, ``-B/usr/bin``,
explicit ``m_axi`` depths, ``alignas(64)`` arrays) are general to any Allo
dataflow design and are documented in :doc:`/backends/vitis`.


Results
-------

Cycle counts (current build)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measured by Vitis ``cosim`` (xsim), one build, ``-m_axi_latency 0``
(``logs/cosim_isa_widened_sweep.log``):

.. list-table::
   :header-rows: 1

   * - shape
     - dynamic instructions
     - cycles
     - mismatches
   * - 4x4x4
     - 6
     - **252**
     - 0/16
   * - 8x8x8
     - 15
     - **383**
     - 0/64
   * - 12x12x12
     - 28
     - **591**
     - 0/144
   * - 16x16x8
     - 25
     - **667**
     - 0/128
   * - 16x16x16
     - 45
     - **919**
     - 0/256

Least squares against dynamic instruction count: fixed cost **151**, marginal
**17.28** cycles per dynamic instruction. Utilization against the 4x4 array's
peak is 1.6% / 8.4% / 18.3% / 19.2% / 27.9%.

These numbers were measured with a memory that answers immediately; the
latency sweep (``-m_axi_latency`` 16 and 64) and what a real memory system does
to them is on :doc:`gemmini_comparison`. **They are not faster than Gemmini** --
an earlier claim that they were is withdrawn there.

How the design got from 1017 cycles at 4x4x4 to 252 is on
:doc:`tinytpu_history`.

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

The resource figures the sources record are per build and are listed with the
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
