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
TinyTPU-isa vs Gemmini: A Matched Comparison
############################################

Both sides measured, both int8/int32 on a 4x4 array, same shapes, same operand
distribution. The design under test is :doc:`tinytpu_isa`.

.. important::

   **Result, 2026-09-18: TinyTPU-isa is 1.55-1.8x slower than Gemmini at all
   five shapes** when both sides are measured over the same window: the
   accelerator plus its dispatch, with near-zero memory latency on both.

   The earlier claim, "faster at all five shapes", compared our
   accelerator-only count with Gemmini's ``tiled_matmul_auto``, and about 395
   cycles of that call is Rocket driver software. **That claim is withdrawn**,
   as is the "3.2x fixed-cost win" that went with it. Ratios in the history
   sections (the I/O-trade and T=16 tables, and the marginal-cost fits) are
   against those end-to-end Gemmini numbers; they are kept, on this page and on
   :doc:`tinytpu_history`, as a record of our own progress, not as comparisons.


The like-for-like comparison
----------------------------

Gemmini measured over the same window ours is: ``rdcycle`` -> 5 ``config``\ s ->
one hardware ``loop_ws`` -> ``fence`` -> ``rdcycle``, with A and B refilled by the
CPU immediately before, exactly as ``allo_cmp.c`` does. Two trials per shape, all
measured in one ~90 s simulator run (``gemmini/allo_bare5.c``). Ours is Vitis
``cosim``, ``ap_start`` to ``ap_done``, with the program already in DRAM.

.. list-table::
   :header-rows: 1

   * - shape
     - driver's tile
     - ``loop_ws`` calls
     - Gemmini accel+dispatch
     - ours
     - **ours / Gemmini**
   * - 4x4x4
     - 1,1,1
     - 1
     - 161 / 144
     - 252
     - **1.6-1.8x slower**
   * - 8x8x8
     - 2,2,2
     - 1
     - 220 / 218
     - 383
     - **1.74x slower**
   * - 12x12x12
     - 3,3,3
     - 1
     - 347 / 344
     - 591
     - **1.70x slower**
   * - 16x16x8
     - 4,2,4
     - 1
     - 391 / 390
     - 667
     - **1.71x slower**
   * - 16x16x16
     - 4,4,4
     - 1
     - 593 / 590
     - 919
     - **1.55x slower**

**We are slower at all five shapes, by 1.55-1.8x, near-uniformly.** Every shape
uses exactly one ``loop_ws``, verified at runtime by replicating the driver's own
tiling search rather than assuming it.

The decomposition closes, which is the reason to trust this. ``allo_cmp.c``'s
total minus this window, per shape: **413, 396, 395, 394, 395**. A flat ~395
cycles of Rocket software across a 16x range of work is exactly what a per-call
driver overhead looks like, and it confirms the 72% figure (below)
independently.

Three caveats, none of which flatter us:

* **4x4x4 is noisy, 161 vs 144**, where the others are within +/-3. Published as
  a range. The 17-cycle spread was not chased.
* **This column is a lower bound on Gemmini's dispatch cost.** ``loop_ws``
  argument marshalling is compile-time-constant in the harness; the real driver
  computes those operands at runtime, and that cost sits in the ~395. So the
  true like-for-like number is at or above these, i.e. the conservative
  direction *for us*.
* **The intercepts are still not comparable** and are not paired here. This
  column's fitted intercept (~154 on a tile-count axis) is not our 151 on a
  dynamic-instruction axis, and pairing them would repeat the error the audit
  found.

History of our column, since earlier revisions quoted each in turn: 1004 / 1108
/ 1294 / 1344 / 1586 before the burst DMA; 680 / 831 / 1066 / 1139 / 1457 after
it, which is where the table sat for most of a day at 1.18x to 1.48x *behind*
(end-to-end Gemmini); 676 / 827 / 1062 / 1125 / 1423 with a flat accumulator
that was **reverted**, costing 13.7x the flip-flops in that unit for 2.3% and
priced as :ref:`limitation-21`. The step to the current numbers is the memset
and widening pass. All of these are on :doc:`tinytpu_history`.


.. _gemmini-gap-attribution:

Where the deficit comes from: forced by Allo/Vitis vs. our design
-----------------------------------------------------------------

.. important::

   **Provenance.** Everything in this section was measured on **variants** of
   TinyTPU-isa that live on branch ``impact-limits`` (commits ``f98c0dac`` and
   ``55405e00``, directory ``examples/accelerator/tinytpu_vitis/impact/``),
   cosimulated at **two shapes only** (4x4x4 and 16x16x16). They are **not the
   shipped design.** The shipped design's numbers remain
   **252 / 383 / 591 / 667 / 919** (the like-for-like table above). The 686 and
   172 below are what the variants reach with every change applied; they are
   not our result.

Of the **326-cycle deficit at 16x16x16** (919 against Gemmini's 593), **Allo
forces 35-95 cycles, all of it** :ref:`limitation-21` (no dependence pragma, so
``accu`` stays at II=2); **Vitis forces nothing measurable**; the rest, about
80%, is **our design**. With all the changes applied, the measured stack goes
**919 -> 686** at 16x16x16 and **252 -> 172** at 4x4x4, against Gemmini's 593
and 144-161.

.. list-table::
   :header-rows: 1
   :widths: 30 17 17 26 10

   * - cause
     - 16x16x16
     - 4x4x4
     - provenance
     - forced by
   * - program prefetch, one 64-bit word per iteration
     - 52
     - 52
     - measured (``v_imem8``)
     - design
   * - B through the vector registers, plus the serial per-``mm`` weight
       prologue
     - 22 alone, 82 once ``accu`` is II=1
     - ~10
     - measured
     - design
   * - A through ``spad`` -> ``vld`` -> ``vr``
     - 64
     - ~13
     - measured
     - design
   * - ``accu`` at II=2 (:ref:`limitation-21`)
     - 35 alone, 95 after the design fixes
     - 5
     - measured
     - Allo
   * - region start
     - ~0
     -
     -
     -
   * - residual
     - 93
     - 11-28
     - timeline **estimate**: operand staging ~84, serial DMA before the first
       weight, drain ~142 -- none of it built
     - design

The rows interact: the bottleneck moves from ``vru`` to the PE prologue to
``accu``. Taking the design rows first gives 919 to 833 (86) and then 95
forced; taking forced first gives 35 and then 146 design. The Allo row is
therefore quoted as a range, 35-95 (65 averaged over both orders).

Variants
~~~~~~~~

Every variant is the shipped ``microarch_isa.py`` plus asserted textual patches
(``make_variants.py``), so its diff is exactly the change being priced. A
variant counts only if the Allo simulator is ``ALL EXACT`` (``bench_variant.py``:
gemm and gemm.relu at five shapes, plus the vadd program) and cosim reports 0
mismatches. Measured by cosim (xsim, ``-m_axi_latency 0``), all bit-exact:

.. list-table::
   :header-rows: 1
   :widths: 20 44 9 9 9 9

   * - variant
     - what changes
     - 4x4x4
     - 16x16x16
     - accu FF
     - top FF
   * - base
     - shipped
     - 252
     - 919
     - 1,250
     - 14,888
   * - ``v_accu1``
     - accu II=1, write-behind rotation (``ca978b97``)
     - 248
     - 885
     - 17,438
     - 31,076
   * - ``v_accudep``
     - accu flat in BRAM + injected ``#pragma HLS dependence variable=ar inter
       false``
     - 247
     - 884
     - 1,744
     - 15,382
   * - ``v_wdirect``
     - ``mm`` reads its weights from ``spad`` via ``spm``, and the weight ``vld``
       is dropped
     - 247
     - 914
     -
     -
   * - ``v_wdb``
     - ``v_wdirect`` + per-PE weight loader and depth-4 FIFO (double buffer),
       flat PE
     - 242
     - 897
     - 1,250
     - 16,453
   * - ``v_wdb_accu1``
     - ``v_wdb`` + rotation
     - 238
     - 803
     -
     -
   * - ``v_wdb_accudep``
     - ``v_wdb`` + pragma
     - 237
     - 802
     -
     -
   * - ``v_design``
     - ``v_wdb`` + A DMA'd straight into ``vr``, so the A ``vld`` is dropped
     - 229
     - 833
     -
     -
   * - **v_design_dep**
     - ``v_design`` + pragma
     - **224**
     - **738**
     - 1,744
     - 17,780
   * - ``v_imem8``
     - program prefetch 8 words/iteration into cyclic-partitioned ``ib``
     - 200
     - 867
     -
     -
   * - **v_design_dep_imem8**
     - ``v_design_dep`` + imem8
     - **172**
     - **686**
     -
     -
   * - ``v_order``
     - program-only reorder: A ``vld`` before the B ``dma_ld``\ s
     - 252
     - 919
     -
     -
   * - ``v_dmadirect``
     - ``dma_ld`` without staging (strided rows, 32-bit beats)
     - 258
     - 926
     -
     -
   * - ``v_best``
     - wdb + rotation + order + dmadirect
     - 249
     - 800
     -
     -
   * - ``v_memset``
     - ``spad`` declared ``= 0`` again
     - 661
     - 1280
     -
     -
   * - ``v_memset6``
     - all six arrays ``= 0`` again
     - 661
     - 1280
     -
     -

Gemmini (accelerator + dispatch, same window): 161/144 at 4x4x4, 593 at
16x16x16.

The pragma form of the ``accu`` fix (``v_accudep``) reaches the same cycles as
the reverted rotation at **1,744 FF in** ``accu`` **against 17,438**. It is
injected by patching the emitted ``kernel.cpp`` between
``s.build(mode="csyn")`` and running Vitis (``cosim_variant.py``'s
``patch_kernel`` hook) -- an escape hatch that exists today, outside Allo.

``v_memset`` restores the ``= 0`` initialiser on ``spad`` and costs **+409**
cycles at 4x4x4 and **+361** at 16x16x16: Allo lowers an array initialiser to a
runtime zero-fill loop (:ref:`limitation-g`).

**A design trap found on the way.** A flat loop whose row count depends on the
decoded opcode closes at ``Final II = 2`` on the row counter (a carried
dependence). Precomputing the count upstream, in the sequencer, and carrying it
in the instruction word fixed it in both ``spm`` (``v_wdirect``) and ``accu``.

The shipped design's per-process timeline at 16x16x16
(``results/base.rle.txt``) reads, in monitor cycles: 0-47 region start
(``s_axilite`` programming, inside the cosim window), 47-120 program prefetch,
120-204 operand bursts, 204-334 128 DMA rows through ``spm``, 334-799 ``vru``
running 465 cycles back-to-back (its 464 words at II=1), and 799-921 the drain
(array, ``accu``, ``dma_st``).

The same work settled that the one-owner-per-array rule the design works under
is Allo's, not Vitis's, and that it cost this design **0 cycles**: every
restructure above was Allo-legal. See :ref:`limitation-shared-memory` and the
note on :doc:`tinytpu_isa` ("One owner per memory").

Reproducing the attribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

From ``examples/accelerator/tinytpu_vitis/impact/`` **on branch**
``impact-limits`` (the scripts are not on ``main``):

.. code-block:: bash

   source env.sh                    # conda allo, LLVM_BUILD_DIR, OMP, TMPDIR
   python make_variants.py          # writes v_*.py
   python pyrun.py bench_variant.py v_design_dep                     # simulator
   TPU_SHAPES=4x4x4,16x16x16 python pyrun.py cosim_variant.py v_design_dep runs/v_design_dep
   ./profile.sh runs/v_design_dep   # per-process timeline of the last shape

``pyrun.py`` strips the conda env's editable finder, so ``allo`` resolves to that
worktree. ``cosim_variant.py`` reuses ``../cosim.py`` unchanged (``align_value``
64, widen 512, ``-B/usr/bin``, ``m_axi`` depths) and adds only a
``patch_kernel(prj)`` hook for variants that edit the emitted C++.
``profile.sh`` re-runs cosim with ``-enable_dataflow_profiling``, then re-runs
the xsim snapshot so the monitor's CSVs survive (cosim deletes them).
``analyze_df.py`` and ``rle_df.py`` turn them into per-process
run/starve/block counts and run-length traces. Raw outputs, timelines and the
Vitis shared-array probe summaries are committed under ``results/`` and
``probe_shared/`` (``55405e00``).


Making the baseline matched
---------------------------

Neither stock Gemmini config is a fair reference:

.. list-table::
   :header-rows: 1

   * - config
     - mesh
     - dtype
     - matched?
   * - ``GemminiRocketConfig``
     - 16x16
     - int8/int32
     - no -- 16x the array
   * - ``FPGemminiRocketConfig``
     - 4x4
     - fp32
     - no -- FPU, and fp32 on an FPGA is soft-float
   * - ``Int8Dim4GemminiRocketConfig`` (**added**)
     - **4x4**
     - **int8/int32**
     - **yes**

``gemmini.GemminiCustomConfigs.int8Dim4Config`` is ``defaultConfig`` -- Gemmini's
own ``inputType = SInt(8.W)``, ``accType = SInt(32.W)`` -- with
``meshRows/Columns = 4``. Elaboration confirms it: ``GEMMINI DIM=4
elem_t_bytes=1``. ``allo_cmp.c`` needed no source change, being written against
``elem_t`` and filled with values in [-4, 4], which is also what
``bench_isa.py``/``cosim.py`` feed our design.

Both numbers are **measured on RTL**: ours by Vitis ``cosim`` (xsim), Gemmini's by
``rdcycle`` under Verilator.

One hardware build, workload swept as data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**This is the correction that matters most.** Earlier revisions compared
per-workload builds against Gemmini's single elaboration: ``M``, ``K``, ``N``, the
instruction count and every unit's loop bound were compile-time constants, so
4x4x4 and 16x16x16 were *different accelerators*. That is not a comparison an
instruction-programmable claim can rest on.

The design is now built **once** -- ``T=4``, ``MAXDIM=16``, fixed scratchpad,
vregs and imem -- and the shape arrives as data:

* the imem header carries the instruction count and a per-unit count, so every
  loop bound in the design is runtime data;
* ``A``, ``B``, ``C`` are flat ``int8[MAXDIM*MAXDIM]`` at the fixed ``MAXDIM``
  stride, exactly as ``allo_cmp.c`` does for Gemmini (``MAXDIM, MAXDIM, 0,
  MAXDIM``);
* ``gemm_program(M, K, N)`` assembles the stream; the RTL never changes.

Verified: **11 programs on one build, all bit-exact** -- ``gemm`` and
``gemm.relu`` at five shapes plus a vector-unit program. Two of those shapes
(12x12x12, 16x16x8) could not be run at all before, because each would have
needed its own accelerator.


The withdrawn comparison, and why it was wrong
----------------------------------------------

.. warning::

   **WITHDRAWN 2026-09-18: "faster than Gemmini at all five shapes", and the
   3.2x fixed-cost win.** Both stood in ``COMPARISON.md`` and both are wrong --
   not imprecise, wrong -- because the two columns measure different things.
   They are struck here rather than footnoted.

.. list-table::
   :header-rows: 1

   * - shape
     - ours (accelerator only)
     - Gemmini ``tiled_matmul_auto``
     - our util
   * - 4x4x4
     - **252**
     - 574
     - 1.6%
   * - 8x8x8
     - **383**
     - 615
     - 8.4%
   * - 12x12x12
     - **591**
     - 740
     - 18.3%
   * - 16x16x8
     - **667**
     - 784
     - 19.2%
   * - 16x16x16
     - **919**
     - 986
     - 27.9%

**Do not read a ratio off this table.** The right column is an accelerator plus
a RISC-V software driver; the left is an accelerator alone. At 4x4x4, **72% of
Gemmini's 574 cycles is Rocket driver code**.

**What was said, and why it was wrong.** The narrowing gap -- 2.28x at 4x4x4 down
to 1.07x at 16x16x16 -- was read as the signature of winning on fixed cost and
losing on marginal cost, and fixed 151 was quoted against Gemmini's 483 as a
3.2x cheaper start. The pattern was real; the explanation was not. Gemmini's
fixed term is mostly its software driver, which our number has no counterpart
for, so the gap narrows with size because the driver amortises -- not because
our machine starts cheaper.

The marginal comparison, 17.28 cycles per dynamic instruction against 10.81, is
**also not like-for-like** and is left standing only as our own figure. The two
fits are against different x-axes (ours dynamic instructions, Gemmini's shapes),
and the accelerator-only share of the 483 intercept is **unknown** -- estimated
140-170, not measured. It is left unknown here rather than given a number nobody
can defend.

**The result that made these numbers possible is a correction, not an
optimisation.** ``COMPARISON.md`` previously said the fixed cost was "essentially
closed" at 563 vs 483. It was not closed, it was *hidden*: ``spm`` opened with a
514-cycle zero-fill of ``spad`` and ``dma_ld``'s operand burst ran ~512 cycles
beside it, two serial prefixes of nearly equal length masking each other. That
is why fixing either alone measured as worthless, and why "merging the A and B
bursts changed cycles by exactly zero" was a true measurement supporting a false
conclusion. Details: :ref:`tinytpu-history-prefixes`.


What each number's window contains
----------------------------------

The audit that withdrew the claims above. Both sides re-measured on the
elaborated chipyard tree, not argued from documentation.

**The memory models match, so this is not the problem.** Gemmini's harness is
``WithBlackBoxSimMem(additionalLatency=0)`` (``AbstractConfig.scala:19`` ->
``HarnessBinders.scala:122-129``), confirmed in the elaborated
``TestHarness.sv``. That ``SimDRAM`` fork only instantiates DRAMSim2 given
``+dramsim``; otherwise it is ``mm_magic_t`` at ~1-2 cycle AXI latency.
Re-running the existing simulator and ELF both ways gives ``MLP 4.8.8.4`` = 1146
without and 1174 with, and ``logs/gemmini_int8_dim4.log`` records 1146 with no
DRAMSim banner. **All five headline shapes are bit-identical either way**,
because ``fill()`` plus the warm-up leaves A/B/C resident in L1/L2. Both sides
are idealised. Disclosed, not corrected for.

**The windows do not match, and that is the problem.** ``allo_cmp.c``'s
``rdcycle`` brackets exactly the ``tiled_matmul_auto`` call -- verified by
disassembling the ELF (``rdcycle s4`` / ``jal tiled_matmul_auto`` /
``rdcycle a0``). Inside that window sit the tile-size search, padding and
last-tile arithmetic, five ``config`` RoCC instructions, the ``loop_ws``
sequence, a closing full ``fence``, and cache coherence for operands the CPU
dirtied immediately before. Ours is ``ap_start`` to ``ap_done`` with the program
already in DRAM.

Measured on the existing simulator, operands refilled before every window
exactly as ``allo_cmp.c`` does:

.. list-table::
   :header-rows: 1

   * - window
     - cycles
   * - ``rdcycle`` pair alone
     - 1
   * - 5 configs + ``fence``
     - 44
   * - **5 configs + one hardware** ``loop_ws`` **+** ``fence``
     - **161**
   * - full ``tiled_matmul_auto(4,4,4)``
     - 536 / 544 (published 574)

Decomposition of the 574: **44** RoCC config dispatch and fence, **117** actual
accelerator work, **~413 (72%)** Rocket software driver. About 30 of that 413 is
L1 coherence from the pre-window ``fill()``.

So Gemmini's accelerator-plus-dispatch fixed cost is about **161** against our
**151** -- parity, not 3.2x. And because the same driver sits inside all five of
its numbers, the honest pairing at 4x4x4 is 252 against ~161: **we are roughly
1.6x slower**, not 2.28x faster.

**What a real comparison needed** was one of two things: a Gemmini window that
excludes the driver at every shape, or our number re-measured with an
equivalent host-side cost included. The first was the cheaper and it is done --
the like-for-like table above is what the 1.55-1.8x result rests on. The second
is still open, and is the honest way to measure what a *user* of each machine
would see rather than what each machine's hardware does.


The memory model, and our sensitivity to it
-------------------------------------------

**Everything above was measured with** ``-m_axi_latency 0`` -- a memory that
answers immediately. That is Vitis's default and it was never stated, so it is
stated here, along with what happens when it is not true.

Rebuilding the design at a given read latency and cosimulating it there
(``cosim.py`` with ``TPU_AXI_LATENCY``; ``logs/cosim_isa_axi_latency_sweep.log``):

.. list-table::
   :header-rows: 1

   * - ``m_axi_latency``
     - 4x4x4
     - 16x16x16
   * - **0** (the tables above)
     - 252
     - 919
   * - 16
     - 297 (+18%)
     - 935 (+2%)
   * - 64
     - 441 (+75%)
     - 1127 (+23%)

Bit-exact at every point (0/16 and 0/256 mismatches). An earlier revision set
these against Gemmini's end-to-end 574 and 986 as a margin of lead. Those
Gemmini numbers include about 395 cycles of Rocket driver software, so that
margin was withdrawn with the headline. Gemmini's own harness is also
near-zero latency (SimDRAM without ``+dramsim``), so latency 0 is the matched
setting. **The sweep is a property of our design, not a fairness correction.**

**The large shape amortises latency and the small one does not.** 16x16x16
moves 1.7% at latency 16, while 4x4x4 moves 18% at latency 16 and 75% at
latency 64. Latency lands on the fixed term, so a real memory system would
widen the like-for-like deficit above at small shapes.


Marginal cost across the sweep
------------------------------

.. note::

   The Gemmini column in this section is ``allo_cmp.c``'s end-to-end
   ``tiled_matmul_auto`` measurement, driver included. It characterises each
   machine's own scaling; it is not a like-for-like comparison.

Fitting only two points hides the shape. Successive marginal efficiency across
all five shapes (MAC/cycle between adjacent shapes, as a fraction of each
machine's own peak):

.. list-table::
   :header-rows: 1

   * - step
     - Gemmini
     - ours
   * - 4x4x4 -> 8x8x8
     - 10.93 MAC/cyc (**68.3%**)
     - 2.97 (**18.5%**)
   * - 8x8x8 -> 12x12x12
     - 9.73 (**60.8%**)
     - 5.17 (**32.3%**)
   * - 12x12x12 -> 16x16x8
     - 7.27 (45.5%)
     - 4.38 (27.4%)
   * - 16x16x8 -> 16x16x16
     - 10.14 (**63.4%**)
     - 6.44 (**40.3%**)
   * - least squares, all five
     - 9.71 (**60.7%**), fixed 566
     - 5.31 (**33.2%**), fixed 717

Our column is post-burst-DMA (the 680 ... 1457 build). It read 4.31 / 6.54 /
6.40 / 8.46 and 44.1% with fixed 1028 after row-flattening and before the burst
DMA, and 3.50 / 5.43 / 6.67 / 6.92 and 37.0% with fixed 1067 before that. **The
burst DMA took the marginal efficiency DOWN, 44.1% -> 33.2%, and the fixed cost
down further, 1028 -> 717**, which is the trade stated in its own terms and
which at every shape in this table the second term wins. The reverted flat
accumulator was the only change since that moved the marginal term back up, and
it moved it to 34.6% -- 0.75 cycles/instruction for 16k flip-flops.

The cube sweep varies all three dimensions at once and spans only 2.4x in
cycles, which makes each marginal a difference of two similar numbers. The
**tall sweep is the better-conditioned measurement** -- ``allo_cmp.c``'s
``gemm_tall`` (TALLM=64) holds K=N=16 and sweeps M, so exactly one dimension
moves, and it ran in the same int8 DIM=4 pass:

.. list-table::
   :header-rows: 1

   * - M
     - cycles
     - d cycles
     - d MACs
     - marginal
   * - 16
     - 986
     - --
     - --
     - --
   * - 32
     - 1,393
     - 407
     - 4,096
     - 10.06 MAC/cyc (**62.9%**)
   * - 64
     - 2,219
     - 826
     - 8,192
     - 9.92 MAC/cyc (**62.0%**)

Closed form: **cycles = 573 + 25.71 * M**, maximum error **0.18%** across the
three points. Marginal = 256 MAC/row / 25.71 cyc = **62.2% of peak**. Two
independent sweeps agreeing is the point: the 5-point cube fit gives 60.7% with
fixed 566, the clean 1-D sweep gives 62.2% with fixed 573. **Quote 62.2%** -- it
varies one dimension rather than three.

The residual is structural rather than noise (this simulator is deterministic):
it is ``tiled_matmul_auto`` re-deciding its tile split as M grows, plus RoCC
dispatch that does not divide evenly. Worth contrasting with the MiniTPU target
(:doc:`minitpu`), whose marginal is *exact* -- ``cycles = 432 + 2023 *
column_tiles``, zero deviation over a 39.7x range -- because nothing in it
arbitrates or backpressures, so there is no mechanism by which two runs could
differ. **A machine whose compiler carries the whole hazard burden is a machine
whose performance is a closed form.** Gemmini has a reservation station and a
tiling heuristic and both leave a trace.

The 12x12x12 -> 16x16x8 step dips for both machines because it changes aspect
ratio rather than growing uniformly; it is not a clean sweep point.

**Gemmini's marginal efficiency is roughly flat at ~61%. Ours still rises,
18.5% -> 40.3%, but far less steeply than it did.** The rise is the signature of
a fixed cost being amortised, and shrinking the fixed cost is exactly what
flattens the curve: the intercept fell 1028 -> 717 when the argument copying was
replaced by program-controlled bursts, and the curve came down with it. What is
left is closer to a constant factor on the work, which is the harder kind of
gap. Flattening the per-instruction loops had lifted the whole curve (21.9% ->
43.2% became 26.9% -> 52.9%) and left the intercept where it was; the burst DMA
did the opposite, and the burst DMA was worth more.

For contrast, the MiniTPU target's marginal efficiency is **flat at 19.0%** and
does not move with size. Measured on its own RTL over a 12x sweep of output
column tiles (``[32,192]@[192,16 / 64 / 192]``: 2,455 / 8,524 / 24,708 cycles for
98,304 / 393,216 / 1,179,648 useful MACs), it is 48.6 MAC/cycle = 19.0% at both
steps, to three digits. Two distinctions matter, and an earlier revision got the
first one wrong:

- **19.0% is the measured marginal; 36.4% is a bound.** One
  ``mxu_matrix_ctrl`` FSM cannot hold a ``vmatpush`` and a ``vmatpop`` at once,
  which permits 4 output rows per 11 cycles = 93.1 MAC/cycle = 36.4% of its 256
  peak. The emitter delivers 48.6 of those 93.1 -- **52% of the bound** -- so
  roughly half the distance to peak is the FSM and the other half is elsewhere
  in the schedule and is, as of this writing, unattributed. Quoting 36.4% as the
  machine's marginal conflates a bound with a measurement.
- **Flat versus rising is the real difference, not the number.** Our 37.0% and
  its 36.4% looked like the same quantity and are not: ours rises 26.9% ->
  52.9% and would keep climbing, theirs sits at 19.0% and does not move.

A ceiling and an overhead are different objects. Ours is an overhead: a fixed
charge the sweep is paying off. Theirs is a ceiling: no amount of amortisation
reaches past it without changing the emitter.

**A correction worth recording:** an earlier analysis put Gemmini's marginal
efficiency at "essentially 100% of peak" from the two-point difference
``986 - 574 = 412`` cycles. That is wrong: 412 cycles for the 4032 additional
MACs is 9.79 MAC/cycle, i.e. **61.2%**. The overhead-vs-ceiling distinction
survives the correction, but at 1.7x rather than 2.7x.


.. _gemmini-reproduce:

Reproducing the Gemmini baseline
--------------------------------

Every Gemmini number on this page came from a Chipyard tree whose changes were
**never committed anywhere**: four uncommitted diffs across three nested
repositories, plus a benchmark that was untracked even inside its own
submodule. They were captured 2026-09-18 into
``examples/accelerator/tinytpu_vitis/gemmini/``.

Pins
~~~~

.. list-table::
   :header-rows: 1

   * - repo
     - commit
   * - ``chipyard``
     - ``e0207441``
   * - ``generators/gemmini``
     - ``25809f7``
   * - ``.../software/gemmini-rocc-tests``
     - ``1a1a1c6``

Files
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - file
     - applies to
     - why it is needed
   * - ``gemmini_CustomConfigs.patch``
     - ``gemmini``
     - ``int8Dim4Config`` -- ``defaultConfig`` with ``meshRows/Columns = 4``. The
       matched baseline itself.
   * - ``chipyard_RoCCAcceleratorConfigs.patch``
     - ``chipyard``
     - ``Int8Dim4GemminiRocketConfig``, the name to pass as ``make CONFIG=``.
       Without it the documented repro fails on its first command.
   * - ``allo_cmp.c``
     - ``gemmini-rocc-tests/bareMetalC/``
     - the entire end-to-end cycle benchmark. **Was untracked**, so
       ``git stash`` in that submodule would have deleted it.
   * - ``allo_bare5.c``
     - ``gemmini-rocc-tests/bareMetalC/``
     - the accelerator-only window (below)
   * - ``roccTests_gemmini_h.patch``
     - same
     - the ``GEMMINI_ACC_SHR`` macro. Without it any FP config gives **318
       compile errors** and the whole suite fails to build.
   * - ``roccTests_gemmini_params_h.patch``
     - same
     - ``DIM`` 16 -> 4, ``BANK_ROWS``, ``ACC_ROWS``, ``ACC_READ_FULL_WIDTH``.
   * - ``roccTests_Makefile.patch``
     - same
     - builds ``allo_cmp``.

Applying and running
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   G=<repo>/examples/accelerator/tinytpu_vitis/gemmini
   cd ~/chipyard && git apply $G/chipyard_RoCCAcceleratorConfigs.patch
   cd generators/gemmini && git apply $G/gemmini_CustomConfigs.patch
   cd software/gemmini-rocc-tests
   git apply $G/roccTests_gemmini_h.patch
   git apply $G/roccTests_gemmini_params_h.patch
   git apply $G/roccTests_Makefile.patch
   cp $G/allo_cmp.c bareMetalC/

   # build the RTL and run
   cd ~/chipyard && source env.sh
   cd sims/verilator && make CONFIG=Int8Dim4GemminiRocketConfig -j16
   ./simulator-chipyard.harness-Int8Dim4GemminiRocketConfig +permissive +permissive-off \
     ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/allo_cmp-baremetal

``~/chipyard/env.sh`` activates Chipyard's own conda environment and so replaces
the ``allo`` one; source it in a separate shell (:doc:`/developer/toolchains`).
Raw Gemmini output is in ``logs/gemmini_int8_dim4.log``.

``allo_cmp.c`` measures what a user gets: ``tiled_matmul_auto``, driver and all.
``allo_bare5.c`` measures what the hardware does: ``rdcycle`` -> 5 ``config``\ s
-> one hardware ``loop_ws`` -> ``fence`` -> ``rdcycle``, operands refilled by the
CPU immediately before the window. Build it the same way and run it the same
way; a single ~90 s run covers all five shapes. It exists because the
comparison was wrong without it: the difference between the two benchmarks, per
shape, is a flat ~395 cycles, which is what a per-call driver overhead should
look like across a 16x range of work, and is the cross-check that the split is
real.

Things that cost time to learn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **The stock** ``matmul`` / ``matmul_ws`` **tests print no cycle counts** --
  their ``read_cycles()`` calls are commented out upstream. ``allo_cmp.c`` exists
  because of this; do not expect to get numbers from the shipped benchmarks.
* **One Verilator run takes about 22 minutes** at ~8.6 us/s simulated. Budget
  accordingly; the five-shape sweep is not interactive. (The ``allo_bare5.c``
  window run is the ~90 s one quoted above.)
* **Stale binaries silently produce a wrong comparison.** The 54 binaries found
  in ``build/`` were int8 artifacts from an earlier elaboration and would have
  been run against a differently-configured RTL without any error. Rebuild the
  benchmark whenever the config changes, and check ``GEMMINI DIM=`` in the boot
  banner -- ``gemmini_int8_dim4.log`` opens with ``GEMMINI DIM=4
  elem_t_bytes=1``, which is the line that proves which hardware produced the
  numbers.
* ``gemmini_counter.h`` exposes 8 hardware counters that were never read.
  Roughly an hour of work if a future comparison wants per-unit attribution
  rather than total cycles.
