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

   **Result, superseded in scope on 2026-09-22 — read both lines.**

   Over the five original shapes at ``MAXDIM=16`` (``e24e433b``), measured on
   both sides over the same window — the accelerator plus its dispatch, with
   near-zero memory latency on both — TinyTPU-isa is **1.07-1.24x slower than
   Gemmini**. That measurement stands and it is what the body of this page
   analyses.

   Over **ten** shapes at ``MAXDIM=64``, with Gemmini reported as a median of
   five trials, the deficit **converges**: 1.27x at 16x16x16, 1.26x at 32³,
   1.13x at 48³ and **1.09x at 64³**, with the design reaching **74.1 % of
   peak** and still climbing. At the two smallest shapes the difference **does
   not clear Gemmini's measurement spread** and is not a supportable claim
   either way. So the 1.07-1.24x range is a statement about pipeline depth at
   small shapes, not about steady-state efficiency, and a reader wanting "how
   far behind is this design" should take **1.09x at 64x64x64** — or **1.04x**
   with the burst-widening candidate, which is not landed and costs +123 %
   block RAM.

   An earlier version of this page claimed the design was faster than Gemmini
   at all five shapes; that claim, and the "3.2x fixed-cost win" that went with
   it, are withdrawn — see `Earlier measurements and corrections`_.


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
     - ours before ``e24e433b``
   * - 4x4x4
     - 1,1,1
     - 1
     - 161 / 144
     - 172
     - **1.07-1.19x slower**
     - 252 (1.6-1.8x)
   * - 8x8x8
     - 2,2,2
     - 1
     - 220 / 218
     - 262
     - **1.19x slower**
     - 383 (1.74x)
   * - 12x12x12
     - 3,3,3
     - 1
     - 347 / 344
     - 418
     - **1.20x slower**
     - 591 (1.70x)
   * - 16x16x8
     - 4,2,4
     - 1
     - 391 / 390
     - 484
     - **1.24x slower**
     - 667 (1.71x)
   * - 16x16x16
     - 4,4,4
     - 1
     - 593 / 590
     - 686
     - **1.16x slower**
     - 919 (1.55x)

**We are slower at all five shapes, by 1.07-1.24x** (ratios against the first
Gemmini trial; the 4x4x4 range spans both). Every shape uses exactly one
``loop_ws``, verified at runtime by replicating the driver's own tiling search
rather than assuming it.

The decomposition closes, which is the reason to trust this. ``allo_cmp.c``'s
total minus this window, per shape, **all from trial 1**: **413, 395, 393,
393, 393**. [#mixedtrials]_ A flat ~393 cycles of Rocket software across a 16x
range of work is exactly what a per-call driver overhead looks like, and it
confirms the 72% figure (below) independently.

Three caveats, none of which flatter us:

* **4x4x4 is noisy, 161 vs 144**, where the others are within +/-3. Published as
  a range. The 17-cycle spread was not chased.
* **This column is a lower bound on Gemmini's dispatch cost.** ``loop_ws``
  argument marshalling is compile-time-constant in the harness; the real driver
  computes those operands at runtime, and that cost sits in the ~395. So the
  true like-for-like number is at or above these, i.e. the conservative
  direction *for us*.
* **The intercepts are still not comparable** and are not paired here. This
  column's fitted intercept (~154 on a tile-count axis) is not our 74.5 (151
  before ``e24e433b``) on a dynamic-instruction axis, and pairing them would
  repeat the error the audit found.

Earlier values of our column, the measured attribution of the deficit that
produced the step to these numbers, and what remains of that deficit, are at
the foot of this page — see `Earlier measurements and corrections`_.


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

Earlier revisions of this page compared per-workload builds against Gemmini's
single elaboration; that is recorded in `Earlier measurements and
corrections`_.

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


What each number's window contains
----------------------------------

The audit that withdrew the claims recorded at the foot of this page
(`Earlier measurements and corrections`_). Both sides re-measured on the
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
its numbers, the honest pairing at 4x4x4 was 252 against ~161: **we were
roughly 1.6x slower**, not 2.28x faster (172 against ~161 since ``e24e433b``:
1.07x).

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
(``cosim.py`` with ``TPU_AXI_LATENCY``; ``logs/cosim_isa_landed_axi_latency.log``,
and ``logs/cosim_isa_axi_latency_sweep.log`` for the pre-landing design):

.. list-table::
   :header-rows: 1

   * - ``m_axi_latency``
     - 4x4x4
     - 16x16x16
     - 4x4x4, pre-landing
     - 16x16x16, pre-landing
   * - **0** (the tables above)
     - 172
     - 686
     - 252
     - 919
   * - 16
     - 214 (+24%)
     - 702 (+2%)
     - 297 (+18%)
     - 935 (+2%)
   * - 64
     - 358 (+108%)
     - 894 (+30%)
     - 441 (+75%)
     - 1127 (+23%)

Bit-exact at every point (0/16 and 0/256 mismatches). The latency costs the
landed design **the same absolute cycles** as the pre-landing one to within 3
(+42 / +186 at 4x4x4, +16 / +208 at 16x16x16), so it is a larger fraction of a
smaller total. An earlier revision set these against Gemmini's end-to-end 574
and 986 as a margin of lead. Those Gemmini numbers include about 395 cycles of
Rocket driver software, so that margin was withdrawn with the headline.
Gemmini's own harness is also near-zero latency (SimDRAM without
``+dramsim``), so latency 0 is the matched setting. **The sweep is a property
of our design, not a fairness correction.**

**The large shape amortises latency and the small one does not.** 16x16x16
moves 2% at latency 16, while 4x4x4 moves 24% at latency 16 and 108% at
latency 64 (18% and 75% before ``e24e433b``). Latency lands on the fixed term, so a real memory system would
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
     - ours (1457 build)
     - ours (landed, 686)
   * - 4x4x4 -> 8x8x8
     - 10.93 MAC/cyc (**68.3%**)
     - 2.97 (**18.5%**)
     - 4.98 (31.1%)
   * - 8x8x8 -> 12x12x12
     - 9.73 (**60.8%**)
     - 5.17 (**32.3%**)
     - 7.79 (48.7%)
   * - 12x12x12 -> 16x16x8
     - 7.27 (45.5%)
     - 4.38 (27.4%)
     - 4.85 (30.3%)
   * - 16x16x8 -> 16x16x16
     - 10.14 (**63.4%**)
     - 6.44 (**40.3%**)
     - 10.14 (**63.4%**)
   * - least squares, all five
     - 9.71 (**60.7%**), fixed 566
     - 5.31 (**33.2%**), fixed 717
     - 7.95 (**49.7%**), fixed 192

The landed column (``e24e433b``) is differences of our own cycle counts, so it
is as comparable with Gemmini's marginal as the driver-inclusive Gemmini
column allows: a per-call driver cost cancels in a difference. Its last step
equals Gemmini's to the digit (10.14 MAC/cycle): adding the second half of K at
16x16 costs both machines the same 202 cycles. Its fixed term, 192, is not
comparable with Gemmini's 566, which carries the driver.

The middle column is post-burst-DMA (the 680 ... 1457 build). It read 4.31 / 6.54 /
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

An earlier analysis of this page read Gemmini's marginal efficiency off a
two-point difference and put it at "essentially 100% of peak"; that reading was
corrected — see `Earlier measurements and corrections`_.


.. _gemmini-reproduce:

Reproducing the Gemmini baseline
--------------------------------

Our side of every table reproduces with one command from a clean checkout,
``examples/accelerator/tinytpu_vitis/reproduce.sh``, which checks the five cycle
counts (:ref:`tinytpu-isa-verify`); ``TPU_WRAP=1 python cosim.py`` builds the
old hoisted-argument variant for comparison. The Gemmini side follows.

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
     - adds ``allo_cmp`` **and** ``allo_bare5`` to ``tests``. It listed only
       ``allo_cmp``, so the recipe below could not produce
       ``allo_bare5-baremetal`` at all; the patch is the committed artefact, so
       it is what was extended rather than the recipe.

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
   cp $G/allo_cmp.c $G/allo_bare5.c bareMetalC/

   # build the RTL and run
   cd ~/chipyard && source env.sh
   cd sims/verilator && make CONFIG=Int8Dim4GemminiRocketConfig -j16
   ./simulator-chipyard.harness-Int8Dim4GemminiRocketConfig +permissive +permissive-off \
     ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/allo_cmp-baremetal

``~/chipyard/env.sh`` activates Chipyard's own conda environment and so replaces
the ``allo`` one; source it in a separate shell (``dev/toolchains.rst``).
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
* **One Verilator run takes about 80 s**, not the 22 minutes an earlier
  revision of this page claimed -- wrong by ~17x, and contradicted by this
  repository's own ``logs/gemmini_int8_dim4.log``, which records
  ``walltime 85.786 s; speed 13.101 us/s``. Re-measured: **79.6 s at
  14.1 us/s** for ``allo_cmp``. So the five-shape sweep *is* interactive and
  needs no special budgeting; the ``allo_bare5.c`` window run quoted above is
  the same order (~90 s), which is why the two are run together.
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

Counting host overhead symmetrically
------------------------------------

This page's window analysis established that a flat **~393-413 cycles** of
Rocket driver software sits inside Gemmini's end-to-end figure, independent of
shape — which is 72 % of the 4x4x4 number and nearly the whole gap between the
two windows. Our own figures are Vitis cosim counts from ``ap_start`` to
``ap_done``, and so contain no host at all.

MiniTPU's owner supplied the symmetric datum, unprompted, and it is the reason
to state this as a methodological rule rather than as a point in our favour:
**their per-launch host work is about 132 microseconds with a 44-microsecond
register-access floor, measured on board .187** — roughly **24,750 cycles at
187.5 MHz** — and their testbench numbers do not include it either.

So the rule for any comparison on this page, and for the benchmark set
generally:

- **Name the window for every figure.** What is inside it, where it starts, and
  where it stops. A cycle count without its window is not a measurement.
- **Count every machine's host, or none of them.** Counting Gemmini's driver
  while omitting our own or MiniTPU's host time is unfair to Gemmini, and a
  comparison that omits all of them is fine *provided it says so*. What is not
  acceptable is letting one machine's overhead count while another's vanishes.
- **Where both windows are cheap to report, report both.** For Gemmini this
  costs nothing, because the driver-inclusive and accelerator-only figures come
  out of the same run.
- **Report an integer invariant beside every timing.** Kernel invocations, DMA
  descriptors, burst iterations, instruction words, ``loop_ws`` calls — some
  count that cannot drift. A time can always be explained away as noise or a
  slow clock; an integer cannot, and it distinguishes *a changed measurement*
  from *a changed machine*.

  The worked example is MiniTPU's, offered against their own interest. A first
  measurement of an unchanged commit read a **2x regression**, and three things
  were wrong at once: provisioning had silently reprogrammed the part from a
  stale firmware directory, replacing the bitstream under the test; the host
  package was a stale copy missing its committed images, so it rebuilt work per
  launch; and provisioning had chowned a device node and locked the next user
  out. **What caught it was the launch count — 216 against 180, eighteen a layer
  instead of fifteen** — clock-independent, bitstream-independent and integer,
  pointing straight at the host. The wall-clock number alone would have sent
  them looking at the RTL.

  Their rule, now ours: read the identity of what you are measuring **after**
  provisioning, not only before.

This is the same failure mode as the withdrawn claim recorded at the foot of
this page (`Earlier measurements and corrections`_): a number that is true of
the measurement but not of the thing being measured. Two of those have now been
caught by comparing notes across designs rather than by inspection, which is an
argument for continuing to do so.

Gemmini at a matched array size is being built independently by both sides as of
2026-09-22 — deliberately twice, because neither side's Gemmini figure has ever
been reproduced by anyone, and a disagreement between two independent builds of
the same nominal configuration would be more informative than either number
alone. Reproducing a build means recording the config object, array dimensions,
datatypes, scratchpad and accumulator sizes, the chipyard and gemmini commits,
the harness, and exactly what is inside the counter window.

The memory-latency knob, and why the sweep outranks the value
-------------------------------------------------------------

Our cycle counts come from Vitis cosim, which has **no DRAM model**:
``TPU_AXI_LATENCY`` is a value we choose, and the published figures are taken at
0. Gemmini's side is idealised too — ``WithBlackBoxSimMem(additionalLatency=0)``
gives roughly a 1-2 cycle AXI unless ``+dramsim`` is used — so neither column
contains a memory system, and that is a disclosure rather than a defect. What
would be a defect is choosing a latency that makes our design look good.

MiniTPU's owner supplied their equivalent figure with the provenance that makes
it usable: **about 40 cycles at 187.5 MHz — 213 ns — for a contiguous
descriptor before its first beat lands, with up to 8 bursts in flight**
(``tools/round_trip.py`` in their tree holds it with the derivation). Carry the
**seconds**, not the cycles: 213 ns is about 88 cycles at our shipped design's
estimated 2.431 ns period and about 64 at a 3.33 ns target, so converting
through cycles imports their clock along with their memory system.

It is a **fitted** value, not a measured DRAM latency — fitted so that a
simulated schedule change matched a board A/B of the same commit — and its
history is the reason this section exists. The value it replaced, 92 cycles,
was also inferred, and at 92 their simulator **ranked GEMM templates backwards**
against the board: a 45 % simulated cut in GEMM cycles was 3.6 % on hardware.
At 0 it ranked them backwards as well. Two different wrong values, two
different wrong rankings, and the simulator was internally consistent and
confident in both cases; it was caught only by running the A/B on two boards.

The conclusion to carry, which is theirs and which we are adopting: **the
sensitivity sweep is worth more than any single value.** Ours runs 172 at 0
cycles, 214 at 16, and 358 at 64. So the question a design decision has to
answer is not "what is the right latency" but "does my ranking survive the
range" — if a variant wins at 0 and loses at 88, the knob chose the design. If
conclusions are stable from 0 to 100, the knob does not matter and can be
disclosed and forgotten. If they invert somewhere in that range, **the inversion
is the finding**.

And when a ranking does invert, **the inversion point in seconds is the
interesting quantity, not the inversion point in cycles**: seconds say which
real memory systems the candidate is good for, where cycles only say which
simulator settings it is good for. MiniTPU's ranking inverted somewhere between
40 and 92 cycles at 187.5 MHz — 213 to 490 ns — and it was the nanosecond
figure that let them say their board sits nearer the low end rather than merely
that one knob value beat another.

**Measured, and it changes what this knob is.** The grid was run — shipped
design against the burst-widened candidate at ``m_axi_latency`` 0, 16, 64, 88
and 100 — and the candidate's advantage is **exactly -720 cycles at 48x48x48 and
-960 at 64x64x64 at every one of the five points**, with the last two predicted
from the first three and returning to the cycle. There is no inversion and no
sensitivity: the knob did not choose the design. The mechanism is that the
widening removes burst iterations (1536 to 96, and 2048 to 128) and the saving
is exactly *half* of each, so the A and B bursts overlap and only one is ever
critical.

And the knob is **non-monotonic** — latency 16 beats latency 0, for both
variants. That is decisive about what it is: ``m_axi_latency`` is a **scheduling
directive to the HLS tool, not a memory latency model.** So **no row of that
sweep may be read as "what this design does against a memory of that
latency"**, and the sweep is not evidence about real memory systems. It bounds
how much this *directive* can move a conclusion, which is a narrower and much
less interesting claim than the one this page made before the grid was run,
which is kept at the foot (`Earlier measurements and corrections`_).

What survives is the discipline rather than the instrument: report the range
over which a conclusion holds, and do not let a knob you cannot interpret decide
a design. What does *not* survive is the idea that our sweep is a better
epistemic position than a fitted memory model — it is not a memory model at all,
and a fitted one at least attempts the right quantity. Our position is that we
have no memory model and should say so.

One caveat on transplanting the number at all: theirs is a ZCU104's memory
system seen through a descriptor-based DMA with two channels and 32-byte beats,
on an AXI port that is not on the memory controller's clock. Our AXI slave is a
different design on different silicon. 213 ns is a far better starting point
than zero and it is not a measurement of our machine.

Which Gemmini configuration is the honest opponent
--------------------------------------------------

Surveyed from Gemmini's Scala source on 2026-09-22
(``generators/gemmini/src/main/scala/gemmini/GemminiConfigs.scala`` has ~60
configuration fields). The conclusion is to **change almost nothing**, and the
reasoning is worth stating because every knob we touched would invite the
question of whether we tuned the opponent to lose.

The four axes pull in different directions and are settled differently:

- **Array size — match it.** The only axis where a mismatch is a pure
  multiplier on peak throughput, 16x for 16x16 against 4x4, with no
  counterargument. This is why stock ``GemminiRocketConfig`` is not a usable
  opponent.
- **Datatype — match it, and we do.** int8xint8 into int32 on both sides,
  which is Gemmini's *own* default. bf16 would be a step away from matched, not
  toward it.
- **Memory system — cannot be matched; enumerate it.** Gemmini is Rocket with a
  32 KiB L1 D$, an L2 and ``WithBlackBoxSimMem(additionalLatency=0)``; ours is
  Vitis cosim at zero AXI latency. Both idealised, neither is the other's
  memory, and equalising them means rebuilding one side's SoC.
- **Dispatch path — match it by *window*, not by configuration.** No knob
  addresses it; only the window does, which is exactly what fixed the withdrawn
  claim recorded at the foot of this page (`Earlier measurements and
  corrections`_).

So the config delta that survives review is **three fields, two of which are
the array size** — ``meshRows`` and ``meshColumns`` — with the third being
``has_training_convs = false``. (``tileRows`` and ``tileColumns`` also appear in
the ``copy()``, restated at the values they already hold, so a reviewer counting
changed fields finds three.) ``has_training_convs`` is read in **exactly one
place** in the whole generator, six conv-loop flags in ``LoopConv.scala``, and
nowhere in ``LoopMatmul``, the mesh, the scratchpad or the header generator — so
a GEMM provably cannot observe it, verifiable in one grep. Everything else is
named as unexercised. Two disclosures rather than fixes: ``dataflow =
Dataflow.BOTH`` leaves Gemmini carrying an output-stationary datapath we never
use, which favours it if anyone, so keeping it is the conservative choice; and
a DIM=16 taken from stock would also flip ``has_training_convs`` to true,
making the matched points differ in a second field.

A second matched point is being added at DIM=8, because ``T`` is now a working
parameter on our side and **a single matched point cannot separate "our design
is slower" from "our design is slower at this one size."**

The capacity asymmetry is cycle-neutral, and this is the finding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The obvious objection to this comparison is memory capacity: our design has a
4 KiB scratchpad, 4 KiB of vector registers and a 2.1 KiB accumulator, against
Gemmini's **256 KiB scratchpad and 64 KiB accumulator** — a 64x and 30x
asymmetry.

Stated precisely, that is **64x the scratchpad and about 30x the
accumulator**. It does not buy Gemmini a single cycle at any published shape. Re-running
``tiled_matmul_auto``'s own tiling search across capacities at DIM=4 gives the
``loop_ws`` call count below; every row is legal under the config's ``require``
clauses:

=========================== ======= ========== ========== ========== =========
Scratchpad / accumulator    4x4x4   16x16x16   32x32x32   64x64x64   64x32x64
=========================== ======= ========== ========== ========== =========
256 / 64 KB (what we ship)  1       1          1          1          1
64 / 32 KB (``chipConfig``) 1       1          1          1          1
32 / 8 KB                   1       1          1          4          4
8 / 4 KB (~2x ours)         1       1          4          12         12
4 / 4 KB (~matched to ours) 1       1          4          32         12
=========================== ======= ========== ========== ========== =========

Every capacity from 256 KB down to 4 KB issues exactly **one** ``loop_ws`` from
4x4x4 through 16x16x16, so shrinking Gemmini's memories to match ours would not
move any of the five published numbers. **The asymmetry is an area and power
disclosure, not a cycle correction** — which is a far stronger position than
"we could not match it". The binding limit is always the accumulator's half
capacity, never the scratchpad.

It stops being neutral at exactly 32x32x32, so as shapes grow this argument
needs restating rather than reusing. Keeping Gemmini's own default capacity is
also the choice that avoids hand-rolling a multi-tile nest and then arguing
about whether our tiling was fair.

Open: ``ex_accumulate`` in the window benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gemmini/allo_bare5.c`` passes ``ex_accumulate`` as a literal ``true``, where
the real driver computes ``!no_bias || D == NULL``, which is **false** for the
no-bias calls we make. In hardware it sets a bit on the preload's accumulator
address rather than issuing extra commands, so it adds no RoCC traffic, but it
turns the k=0 accumulator writes into read-modify-writes and is therefore **not
guaranteed cycle-neutral**.

**Measured both ways, and settled: the published figures stand.** The
driver-matched ``false`` makes Gemmini **0 to 4 cycles faster** at the five
shapes — so our deficit was very slightly understated, in the direction that
does not flatter us. The effect is below the measurement's own noise: the
``true`` column's trial-to-trial spread reaches 20 cycles at one shape, larger
than the effect itself. Its *sign is consistent* across all five shapes, so it
is a small systematic effect under the single-trial noise floor rather than
noise, and new work should use the driver-matched form. 161 / 220 / 347 / 391 /
593 need no revision on this account.

Two operational hazards found in the same survey, both of which have silently
produced wrong comparisons before: **every elaboration rewrites**
``gemmini_params.h`` **in place**, so elaborating a second configuration
destroys the first's header and the next C build silently targets the wrong
hardware — snapshot it, rebuild immediately, and check the ``GEMMINI DIM=`` boot
banner. And the committed ``roccTests_Makefile.patch`` adds only ``allo_cmp`` to
the ``tests`` list, **not** ``allo_bare5``, so ``build.sh`` will not build the
window benchmark; it needs its explicit make target.

Reproduced, once, independently
-------------------------------

On 2026-09-22 a second agent re-ran the untouched window benchmark on the same
hardware configuration and measured **161, 144 / 220, 218 / 347, 344 / 391, 390
/ 593, 590** — the five published figures exactly, including the 161/144 spread
at 4x4x4 that had been recorded and never explained.

This is the **first independent reproduction of the Gemmini column**, and it is
worth stating separately because until it happened neither this project's nor
MiniTPU's Gemmini figures had ever been reproduced by anyone. It is one
reproduction on the same host, not a second implementation, so it establishes
that the numbers are re-derivable rather than that they are right.

A cross-check fell out of the same work: a ``gemmini_params.h`` reconstructed
independently from stock ``HEAD`` plus the committed patch is **byte-identical**
to the header captured from the live tree.

The measurement noise floor, which constrains every claim on this page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same reproduction exposed something that had been visible in the recorded
numbers all along and never priced: **the trial-to-trial spread reaches 20
cycles at a shape whose total is 161** — identical hardware, identical binary,
consecutive runs. That is up to **12 % noise at the small shapes**, and it comes
from each measured window's sensitivity to the preceding call's cache and
scratchpad state.

Two consequences, both binding:

- **No single-trial number at 4x4x4 or 8x8x8.** Report a median and the spread.
  A 1.07-1.24x deficit measured at those shapes is, at the small end, partly
  inside the noise. It does not erase the result — the sign is consistent across
  all five shapes, and that consistency is the evidence — but any claim of a
  difference *smaller* than the spread is unsupportable.
- **The noise is the Gemmini column's, not ours.** This spread comes from a
  full SoC — Rocket, an L1 cache, a scratchpad whose state the previous call
  left behind. Our own figures are Vitis co-simulation of a fixed design on
  fixed inputs, which is **deterministic**: the five shapes reproduce exactly,
  run after run, and the alignment variant's independent re-measurement
  reproduced every one of its numbers to the cycle. So repeats are needed on
  the Gemmini side and are not needed on ours, and a design A/B measured only
  by our cosim — the latency sensitivity grid above, for instance — does not
  need them either. A *cross-machine comparison* is still limited by the
  noisier side of it.

  **That claim was incomplete, and the correction is measured.** A controlled
  count of nine runs of one configuration gives **8 successful co-simulations,
  8 identical cycle counts, 0 differing counts — and 1 run that yielded no
  number at all** (``cycles=None, no TB line`` at one shape while another
  returned its usual figure in the same run). So the statement has two halves
  and only the first was being made: **the simulation is deterministic; the
  pipeline can fail to produce a number.** A nondeterministic simulation gives
  a *different* number; a flaky pipeline gives *no* number.

  The operational consequence is the part that changes behaviour: **a single
  rejection by a cycle gate is not evidence and must be retried rather than
  believed.** Repeats are still not needed to establish a *value* on our side —
  that is what the 8-and-0 count says — but a *failure* on our side is not a
  measurement until it has been seen twice.

This is the cycle-domain analogue of the synthesis noise floor recorded in
:doc:`/designs/minitpu` — two builds of an identical netlist differing by 1,407
LUT and 0.046 ns. Both say the same thing: **state the noise before stating the
difference**, and say which measurement the noise belongs to.

**Verify determinism once per configuration rather than assuming it.** One extra
run per configuration — not per data point — and confirm the counts are
identical. The reason is not sampling: it is that a simulator figure which turns
out *not* to be deterministic would undermine every comparison drawn from it,
and that is much better discovered in a two-run check than in a disagreement
with someone else's table hours later. Our cosim has effectively passed this
already, through the alignment variant's independent re-measurement reproducing
every number to the cycle, but a new configuration has not.

And the trap on the far side of determinism, which is the more dangerous one: **a
simulator number can be perfectly reproducible and still wrong about hardware.**
Ours contains no memory system at all, and MiniTPU's contains a fitted one — the
92-cycle failure recorded above is exactly this, a confident, repeatable
simulator that ranked designs backwards against the board. Reproducibility is a
property of the measurement; agreement with hardware is a separate claim needing
separate evidence.

Do not mix the two benchmarks' columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are now two window benchmarks, and they do **not** produce interchangeable
latencies: the newer steady-state file reads **2 to 14 cycles lower** at the same
shapes on the same hardware. The cause is the call sequence rather than the
hardware — the files differ in how many trials they run per shape and in what
runs between them, and each measured window is sensitive to the preceding call's
cache and scratchpad state.

So a table must say which file produced it, and a comparison must take both
columns from the same file. This is the same class of error as the window
problem that produced the withdrawn claim: a real measurement, correctly taken,
that means something different from what it is being used to mean.

A third hazard for anyone reproducing at another array size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond the in-place ``gemmini_params.h`` rewrite and ``allo_bare5`` missing from
the patched ``tests`` list: **rocc-tests at its pinned HEAD ships a header that
no stock configuration generates.** It carries DIM 16 with 4096 bank rows and
1024 accumulator rows but *no* ``ACC_READ_FULL_WIDTH``, while the default config
sets ``acc_read_full_width = true`` and the generator emits that macro — so the
committed header came from a leaner config than the one it appears to describe.
A build that reuses the committed header instead of regenerating it will differ
from a build that regenerates, for a reason invisible in the Scala.

Two designs, the same shape of loss
-----------------------------------

On 2026-09-22 the MiniTPU side ran its own Gemmini evaluation, independently and
against its own interest, and the two results converged on a conclusion neither
project would have reached alone.

**Both designs lose to Gemmini, and in both cases the loss is not in the array.**

- Ours: the deficit **converges** with shape — 1.27x at 16x16x16 down to
  **1.09x at 64x64x64** — and the burst-widening candidate prices the remainder
  exactly, at 720 cycles at 48x48x48 and 960 at 64x64x64, which is 61 % and 55 %
  of the whole deficit. One identified prologue, not a mystery.
- Theirs: their general emitter's smallest expressible product takes 690 cycles
  against a 16x16 int8 Gemmini's 203-286 — but **the same 16x16x16 product
  hand-scheduled on their machine takes 168 cycles, 1.6x faster than that
  Gemmini.** Their array beats Gemmini's and their emitter loses to Gemmini's by
  roughly a factor of four. The loss is entirely in emitted code.

So the honest reading of both evaluations is that **Gemmini's advantage at these
shapes is its software**, not its datapath: a mature driver and a tiling search
that has been tuned against real workloads. That is a much more actionable
conclusion than a microarchitectural one, and it is the direction both projects'
remaining work should take.

Their steady-state figure, which they are re-running before standing behind it
(it currently rests on 3 of 16 sweep points): Gemmini converts its array at
86.4 % of peak against their 53.8 %, i.e. 1.61x more efficient per processing
element. Ours reaches **74.1 % of peak at 64x64x64** and is still climbing.

The capability gap, which is worse for us than the cycle deficit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gemmini runs **128x768x768 today**, through `tiled_matmul_auto`'s own tiling
search. We cannot address a matrix that size at all: our operands are a single
``int8[MAXDIM*MAXDIM]`` region, so the shape has to fit the addressable space
rather than being tiled into it. Unblocking it needs a runtime base and stride
on the DMA load and store paths, a fifth loop level, and possibly a fourth
address term.

This matters more than the 1.09x. A 9 % cycle deficit at a shape both machines
can run is a tuning result; being unable to express the shapes a real workload
uses is a capability result, and no amount of cycle-level work closes it.


.. _gemmini-parity:

The parity baseline
-------------------

A second, named configuration of our design, kept **beside** the shipped one,
whose purpose is to perform at parity with a matched Gemmini by changes that
are reasonable rather than contrived. The shipped design is unchanged.

The rules, fixed before any measurement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Written and committed on branch ``gemmini-parity`` before the first cosim of
any candidate, so the result cannot be fitted to them afterwards.

- **Parity, defined.** At a shape, ours is at parity when our cosim count is
  at most Gemmini's median plus its published spread (the ``+/-`` figure on
  :doc:`benchmarks`, which is the full min-max range over five trials), i.e.
  ``ours <= median + spread``. **Faster** means ``ours < median``. **Behind**
  means ``ours > median + spread``, and the report names every such shape.
- **Same windows, same shapes, same opponent.** Ours is Vitis cosim,
  ``ap_start`` to ``ap_done``, ``-m_axi_latency 0``, bit-exact, one run per
  shape (retried once on ``cycles=None``). Gemmini is the matched-array
  median of five from :doc:`benchmarks`: DIM=4 against T=4 over all ten
  shapes of that table, DIM=8 against T=8 over all eight of that table, both
  at MAXDIM=64. No shape is added or dropped after the fact.
- **A mechanism for every change**, in one sentence, and an integer invariant
  reported beside every timing (dynamic issues, per-unit work counts, burst
  iterations) so a changed measurement can be told from a changed machine.
- **Synthesisable at the same clock.** It must meet the 3.33 ns target with the
  estimated period reported, and use no FPGA-only structure (no RAM with two
  write ports) so it also maps to standard cells.
- **Preference order.** (1) a change to the program the generator emits, on the
  unchanged netlist; (2) the banked burst widening (``TPU_DMA_WIDEN=1``);
  (3) anything else, each with its mechanism and its price.

The configuration, and one command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``parity-t4`` and ``parity-t8`` in ``parity_sweep.py``: the shipped design
with **one** change, the banked burst widening, at the matched array sizes.

.. code-block:: bash

   cd examples/accelerator/tinytpu_vitis
   TPU_PARITY_CONFIG=parity-t4 python parity_sweep.py      # 10 shapes
   TPU_PARITY_CONFIG=parity-t8 python parity_sweep.py      #  8 shapes

=========================  ========================================
``T``                      4 (against Gemmini DIM=4) / 8 (DIM=8)
``MAXDIM``                 64, both sides
``DMA_WORDS``              16 (``TPU_DMA_WIDEN=1``), buffers banked
``TPU_PROGRAM``            ``shipped`` -- the published program order
``QD``                     8, unchanged
=========================  ========================================

One csynth per configuration; each (shape) cosim runs in its own copy of that
one synthesized solution, so every number in a column is the same netlist.

The change, and its mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The banked burst widening, and nothing else.** ``dma_ld``'s operand burst
moves ``DMA_WORDS`` packed words per iteration instead of one, and ``rbA`` /
``rbB`` are cyclically partitioned by ``DMA_WORDS`` so write ``w`` always lands
in bank ``w`` and every bank has exactly one writer (``ff7beaf1``).

That is the whole configuration. It is one sentence because the prologue is
what the measurement says the deficit was: at MAXDIM=64 the burst reads whole
64-byte DRAM rows, so it costs ``max(M,K) * MAXDIM/T`` iterations however
narrow the operand actually is -- 1,024 of the 22,123 cycles at 64x64x64 and
128 of the 424 at T=8 16x16x8 -- and widening divides that by ``DMA_WORDS``.

Two supporting measurements from the same apparatus, both on an unchanged
netlist, because they were the candidates that did **not** get used:

- **The A and B bursts already overlap.** At MAXDIM=16, growing A's burst span
  by 48 words costs **+45** cycles and then growing B's by 48 costs **+0**, so
  the burst is ``max(na, nb)``, not ``na + nb``. That is why merging them was
  once measured at exactly zero, and it is why widening is the only lever left
  on the burst.
- **``dma_ld``'s row loop costs exactly one cycle a row** (collapsing the
  program's operand blocks to one removed 96 rows and 96 cycles at
  16x16x16), which is what makes the program-order candidates below worth
  measuring at all.

Result: T=4 against Gemmini DIM=4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ours: Vitis cosim, ``ap_start`` to ``ap_done``, ``-m_axi_latency 0``,
bit-exact, one build. Gemmini: median of five, full min-max spread, from
:doc:`benchmarks`. ``csynth`` on xcu280 at the 3.33 ns target: **BRAM 100,
DSP 14, FF 25,026, LUT 33,799, estimated period 2.431 ns** -- the same
estimated period as the shipped control (BRAM 52, DSP 14, FF 17,488, LUT
26,554, 2.431 ns), so the widening is bought at +92 % BRAM, +43 % FF and
+27 % LUT and **no clock**.

.. list-table::
   :header-rows: 1

   * - shape
     - shipped
     - **parity-t4**
     - Gemmini
     - ratio
     - verdict
   * - 4x4x4
     - 218
     - **172**
     - 208 +/- 25
     - 0.83x
     - faster
   * - 8x8x8
     - 357
     - **262**
     - 324 +/- 36
     - 0.81x
     - faster
   * - 12x12x12
     - 563
     - **383**
     - 458 +/- 17
     - 0.84x
     - faster
   * - 16x16x8
     - 677
     - **437**
     - 527 +/- 44
     - 0.83x
     - faster
   * - 16x16x16
     - 879
     - **639**
     - 691 +/- 44
     - 0.93x
     - faster
   * - 32x32x32
     - 3 752
     - **3 272**
     - 2 977 +/- 34
     - 1.099x
     - **behind**
   * - 32x64x32
     - 6 824
     - **5 864**
     - 5 570 +/- 147
     - 1.053x
     - **behind**
   * - 48x48x48
     - 10 289
     - **9 569**
     - 9 100 +/- 35
     - 1.052x
     - **behind**
   * - 64x32x64
     - 12 907
     - **11 947**
     - 11 175 +/- 18
     - 1.069x
     - **behind**
   * - 64x64x64
     - 22 123
     - **21 163**
     - 20 287 +/- 34
     - 1.043x
     - **behind**

Result: T=8 against Gemmini DIM=8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``csynth``: **BRAM 138, DSP 58, FF 50,004, LUT 74,331, 2.431 ns**, against the
T=8 control's BRAM 62, DSP 58, FF 43,911, LUT 70,281, 2.431 ns.

.. list-table::
   :header-rows: 1

   * - shape
     - shipped
     - **parity-t8**
     - Gemmini
     - ratio
     - verdict
   * - 8x8x8
     - 285
     - **231**
     - 305 +/- 17
     - 0.76x
     - faster
   * - 16x16x8
     - 424
     - **312**
     - 500 +/- 16
     - 0.62x
     - faster
   * - 16x16x16
     - 493
     - **381**
     - 535 +/- 45
     - 0.71x
     - faster
   * - 32x32x32
     - 1 484
     - **1 260**
     - 1 280 +/- 18
     - 0.98x
     - faster
   * - 32x64x32
     - 2 508
     - **2 060**
     - 1 866 +/- 36
     - 1.104x
     - **behind**
   * - 48x48x48
     - 3 537
     - **3 201**
     - 3 028 +/- 35
     - 1.057x
     - **behind**
   * - 64x32x64
     - 4 523
     - **4 075**
     - 3 897 +/- 35
     - 1.046x
     - **behind**
   * - 64x64x64
     - 7 083
     - **6 635**
     - 6 033 +/- 35
     - 1.100x
     - **behind**

**Where parity is NOT reached, plainly.** Five shapes at T=4 (32x32x32,
32x64x32, 48x48x48, 64x32x64, 64x64x64) and four at T=8 (32x64x32, 48x48x48,
64x32x64, 64x64x64). Every one of them is a steady-state shape, every one is
**behind by 4.3 % to 10.4 %**, and every one of those margins clears Gemmini's
spread, so they are real deficits and not noise. Nine of the eighteen matched
points are faster than Gemmini; none is inside the spread without also being
faster. The honest summary is **comparable, not parity**: faster wherever the
shape is small enough to be dominated by fixed cost, and a few per cent behind
wherever it is not.

Where the cycles go, on our side
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measured with Vitis's dataflow profiler (``impact/prof_job.py`` then
``impact/profile.sh``'s steps), not modelled. Every unit's work count is an
integer the assembler writes into the imem header, so the run/starve/block
split can be read against it.

**Latency shapes are one pass of everything.** T=8 16x16x8, shipped, 424
cycles: 47 region start (``s_axilite``), 21 to the first dispatched
instruction, then ``dma_ld`` runs 196 cycles without a stall -- 128 burst
iterations and 48 operand rows, exactly its header counts -- ``spm`` waits
until row 48 arrives at cycle 249, ``vru`` runs its 64 words, ``accu`` runs its
48 iterations ending at 391, and ``dma_st`` retires 16 rows by 426. **Three
quarters of that shape is prologue and drain**, which is why widening the
burst takes it to 312.

**Steady-state shapes are a repeating per-n-tile loop.** T=4 32x32x32,
parity-t4, 3,272 cycles: ``dma_ld`` finishes all 544 of its work items by
cycle 632, i.e. the operand path is off the critical path after 19 % of the
run, and the rest is a loop of ``vru`` running 256 activation words and then
**blocking 65** while ``accu`` retires that tile -- 321 cycles per 256
mm rows, 1.25 cycles per row against a roofline of 1.00. The marginal rate
between 48x48x48 and 64x64x64 is **1.224 cycles per mm row for us against
Gemmini's 1.181**, which is the 4.3 % at 64x64x64 almost exactly.

**What that 0.224 is, and what is still unattributed.** ``accu`` is one flat
loop at II=1 over ``mm`` rows *plus* ``mvout`` rows, so the retire pass shares
the accumulator's single port with the accumulate pass and costs ``1/Kt`` of
the mm work: 12.5 % at 32x32x32, 8.3 % at 48x48x48, 6.3 % at 64x64x64. Our
T=4 deficit falls in the same order (9.9 %, 5.2 %, 4.3 %) and this is the
**explanation of the convergence** the benchmarks page measured -- deeper K
amortises one retire pass over more accumulate passes. It is not the whole of
it: the ratio of deficit to ``1/Kt`` is 0.79 / 0.63 / 0.69 rather than 1, and
at T=8 the correspondence fails outright (32x32x32 has ``Kt=4`` and we are
*faster* there). **So the retire pass is identified and the remainder is
not**, and no change has been built against either.

Why we are ahead at the small shapes, and why that stops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The owner's rule is that a win needs a mechanism or it cannot be defended.
For every shape where this baseline is ahead, the mechanism is the same one,
and it is a statement about **fixed cost**, not about the array:

- A latency shape is almost entirely prologue on both machines. Ours is
  47 cycles of ``s_axilite`` programming, one operand pass and one drain
  (measured above); Gemmini's window is five ``config`` RoCC instructions, the
  ``loop_ws`` dispatch, the mesh's own fill and drain, and a closing ``fence``
  -- of which the configs and fence alone were measured at 44 cycles at DIM=4.
- Widening the burst removes most of *our* prologue and cannot touch Gemmini's,
  so the machine with the shorter remaining fixed cost wins, and at these
  shapes that is now us by 7-38 %.
- It stops exactly where the marginal rate starts to dominate, because our
  marginal rate is the worse of the two (1.224 against 1.181 cycles per mm row
  at T=4). At T=8 the crossover sits one shape later -- 32x32x32 is still a
  win at 0.98x -- because doubling ``T`` quarters the number of tiles a shape
  contains, so a given shape stays fixed-cost-dominated for longer.

**The T=8 16x16x8 win, which was previously unexplained**, is the extreme case
of this: the shape is 2x2x1 tiles, so ``accu`` does 48 iterations of real work
inside a 424-cycle window, and 312 of those cycles are prologue and drain that
the widening then halves. It is the shape with the fewest tiles per unit of
window on either side, which is why its margin (0.62x) is the largest anywhere
in the two tables. The 1.33x standard-cell area that T=8 costs over T=4 at the
same MAXDIM is the price, at no clock penalty.

Two disclosures that belong with these numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **The FPGA resource line understates a build.** Operand space is nearly free
  on an FPGA and is not on standard cells: MAXDIM 16 -> 64 costs +64.1 %
  standard-cell area but only +2.4 % Vitis FF, a factor of about 27. Both
  parity configurations are at MAXDIM=64, which is the matched setting, but
  their BRAM and FF figures should not be read as what it would cost to build.
- **The windows contain different things** and always have: ours starts at
  ``ap_start`` with the program already in DRAM, Gemmini's is
  ``rdcycle``-bracketed around five configs, one hardware ``loop_ws`` and a
  ``fence``, with neither side's host counted. That is the same window pairing
  every number on this page uses.

.. _gemmini-parity-order:

The program order that was measured and not used
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first preference was a change to the emitted program on an unchanged
netlist, and two were built: ``isa_dsl.gemm_program_interleaved`` (every
operand column block loaded just before the first ``mm`` that reads it, so the
first ``mm`` waits behind ``M + K`` load rows instead of ``Kt*M + Nt*K``) and
``gemm_program_b_per_tile``. Both have **identical** dynamic issue counts and
per-unit work counts to the shipped order at every shape, so any cycle
difference is scheduling.

Measured at MAXDIM=16 on one netlist -- shipped / interleaved / b_per_tile /
k-outermost -- 172/167/169/169, 262/250/256/252, 418/392/394/418,
484/466/468/479, 686/636/638/672, all bit-exact. At MAXDIM=64 the interleaved
order is worth 2 to 50 cycles on the unchanged netlist and 16 to 50 on the
widened one (T=4 16x16x16: 879 -> 829, and 639 -> 589).

**It is not in the baseline, for two measured reasons.** Its whole mechanism
is removing a fixed prologue, so at the steady-state shapes where the deficit
actually is it does nothing -- 32x32x32 measured 3,355 against the shipped
order's 3,272 on the widened build, i.e. *worse* than simply widening -- and
at the latency shapes where it does help, the widened baseline is already
faster than Gemmini. And it **deadlocks in RTL** whenever ``Kt >= QD``: at
T=4 32x32x32 and above and at T=8 64x64x64 and 32x64x32, cosim runs 30 million
cycles without progress and Vitis's deadlock detector does not fire. Raising
``TPU_QD`` to 16 makes the identical program complete (3,355 cycles,
bit-exact), which is what identifies the cause: a ``dma_ld`` issued *between*
two ``mm``s is separated from the unit that consumes its rows by ``Kt``
instructions, the sequencer blocks mid-dispatch on a full depth-``QD`` queue,
and the consumer waits for a load the sequencer can no longer dispatch.

That the Allo simulator, ``stress_isa.py`` (640/640 exact for both orders) and
``kpn_model.py``'s bounded-FIFO deadlock model all accept a program the RTL
hangs on is a **gap in the verification stack**, not just a property of these
two orders: the KPN model runs the channel protocol at depth ``QD`` and
reports no deadlock, so it is missing the sequencer's mid-instruction blocking
across its five output queues. Anyone adding a program order should cosim it
at a shape with ``Kt >= QD`` before believing any of the three.

**This diagnosed a filed limitation.** :ref:`limitation-24` -- row-tiled
mappings that pass five checks and then never complete in cosim, un-diagnosed
after two independent investigations -- has the same signature, and it is the
same cause: run its whole ten-program family at ``TPU_QD=16`` and **all ten
complete, bit-exact**, where the three known cases do not complete at ``QD=8``
in the same tree on the same day. It also explains that item's "not monotone
in size" observation, which is what a threshold looks like from either side.
The measurement, its price (+9.3 % FF, no BRAM and no clock change) and what
is still open about the predicate are recorded there, not here.


Earlier measurements and corrections
------------------------------------

What the figures above used to be, and the claims this page has made and then
withdrawn. Nothing in this section describes the current state.

History of our column
~~~~~~~~~~~~~~~~~~~~~

Until ``e24e433b`` the like-for-like result was **1.55-1.8x** slower (252 / 383
/ 591 / 667 / 919 cycles).

History of our column, since earlier revisions quoted each in turn: 1004 / 1108
/ 1294 / 1344 / 1586 before the burst DMA; 680 / 831 / 1066 / 1139 / 1457 after
it, which is where the table sat for most of a day at 1.18x to 1.48x *behind*
(end-to-end Gemmini); 676 / 827 / 1062 / 1125 / 1423 with a flat accumulator
that was **reverted**, costing 13.7x the flip-flops in that unit for 2.3% and
priced as :ref:`limitation-21`; 252 / 383 / 591 / 667 / 919 after the memset
and widening pass, the shipped baseline until ``e24e433b``. The step to the
current numbers is the gap attribution's design stack
(:ref:`gemmini-gap-attribution`). All of these are on :doc:`tinytpu_history`.

.. _gemmini-gap-attribution:

Where the deficit came from, and what was landed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   **Provenance.** The attribution below was measured on **variants** of the
   design as it was shipped until ``e24e433b`` (``7a24c21e``: 252 / 383 / 591
   / 667 / 919), cosimulated at two shapes (4x4x4 and 16x16x16), on branch
   ``impact-limits`` (``f98c0dac``, ``55405e00``). The branch's scripts, raw
   results and timelines are now on ``main`` under
   `examples/accelerator/tinytpu_vitis/impact/
   <https://github.com/sunwookim028/allo/tree/main/examples/accelerator/tinytpu_vitis/impact>`__
   and the branch is deleted.

   **Its best stack is now the shipped design.** ``e24e433b`` landed
   ``v_design_dep_imem8`` -- every design row of the table below, plus the
   ``accu`` row through a real schedule primitive (``s.dependence``,
   ``bbea2af0``) rather than a ``kernel.cpp`` patch -- and measured it at all
   five shapes: **172 / 262 / 418 / 484 / 686**, bit-exact. At the two shapes
   the variants were measured on, the landed build reproduces them exactly
   (172 and 686); the other three shapes were first measured on the landed
   build. What landed is listed on :ref:`tinytpu-isa-landing`.

Of the **326-cycle deficit at 16x16x16** (919 against Gemmini's 593), **Allo
forced 35-95 cycles, all of it** :ref:`limitation-21` (no dependence pragma, so
``accu`` stayed at II=2); **Vitis forced nothing measurable**; the rest, about
80%, was **our design**. Both parts are now removed: the design rows by
rebuilding the units, the Allo row by adding the primitive. The stack took
**919 -> 686** at 16x16x16 and **252 -> 172** at 4x4x4, against Gemmini's 593
and 144-161.

.. list-table::
   :header-rows: 1
   :widths: 30 17 17 26 10

   * - cause (pre-landing design)
     - 16x16x16
     - 4x4x4
     - provenance
     - forced by
   * - program prefetch, one 64-bit word per iteration
     - 52
     - 52
     - measured (``v_imem8``); **landed**
     - design
   * - B through the vector registers, plus the serial per-``mm`` weight
       prologue
     - 22 alone, 82 once ``accu`` is II=1
     - ~10
     - measured; **landed**
     - design
   * - A through ``spad`` -> ``vld`` -> ``vr``
     - 64
     - ~13
     - measured; **landed**
     - design
   * - ``accu`` at II=2 (:ref:`limitation-21`)
     - 35 alone, 95 after the design fixes
     - 5
     - measured; **landed**, via ``s.dependence``
     - Allo (**fixed**)
   * - region start
     - ~0
     -
     -
     -
   * - residual
     - 93
     - 11-28
     - timeline **estimate**: operand staging ~84, serial DMA before the first
       weight, drain ~142 -- **none of it built**
     - design

The rows interact: the bottleneck moves from ``vru`` to the PE prologue to
``accu``. Taking the design rows first gives 919 to 833 (86) and then 95
forced; taking forced first gives 35 and then 146 design. The Allo row is
therefore quoted as a range, 35-95 (65 averaged over both orders).

**What remains.** The landed design is still **93 cycles** behind Gemmini at
16x16x16 (686 against 593) and **11-28** at 4x4x4 (172 against 144-161), and
between 42 and 93 cycles at the three shapes the branch did not measure (262
against 220, 418 against 347, 484 against 391). The branch estimated where the
16x16x16 residual sits from the stack's per-process timelines
(``impact/results/``): operand staging (~84 cycles), the serial DMA ahead of
the first weight, and the drain (~142). That split is an **estimate**, and no
change against it has been built or measured.

**The dependence claim needed a contract the branch never tested.** The
injected pragma is only true for programs that never read an accumulator row
within two ``accu`` iterations of writing it; the branch cosimulated GEMM
programs only, which never do. Landing it added the contract to the assembler
and a test at its edge (:ref:`tinytpu-isa-dependence`).

Variants
^^^^^^^^

Every variant is the pre-landing ``microarch_isa.py`` (``7a24c21e``, the
``base`` row) plus asserted textual patches (``make_variants.py``), so its diff
is exactly the change being priced. A
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
     - the design shipped until ``e24e433b``
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
     - ``v_design_dep`` + imem8 -- **shipped since** ``e24e433b``
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
the reverted rotation at **1,744 FF in** ``accu`` **against 17,438**. On the
branch it was injected by patching the emitted ``kernel.cpp`` between
``s.build(mode="csyn")`` and running Vitis (``cosim_variant.py``'s
``patch_kernel`` hook); the shipped design emits it through ``s.dependence``,
and the landed build's ``accu`` is 1,744 FF, as the variant's was.

``v_memset`` restores the ``= 0`` initialiser on ``spad`` and costs **+409**
cycles at 4x4x4 and **+361** at 16x16x16: Allo lowers an array initialiser to a
runtime zero-fill loop (:ref:`limitation-g`).

**A design trap found on the way.** A flat loop whose row count depends on the
decoded opcode closes at ``Final II = 2`` on the row counter (a carried
dependence). Precomputing the count upstream, in the sequencer, and carrying it
in the instruction word fixed it in both ``spm`` (``v_wdirect``) and ``accu``;
the landed sequencer does the same.

The pre-landing design's per-process timeline at 16x16x16
(``impact/results/base.rle.txt``) reads, in monitor cycles: 0-47 region start
(``s_axilite`` programming, inside the cosim window), 47-120 program prefetch,
120-204 operand bursts, 204-334 128 DMA rows through ``spm``, 334-799 ``vru``
running 465 cycles back-to-back (its 464 words at II=1), and 799-921 the drain
(array, ``accu``, ``dma_st``).

The same work settled that the one-owner-per-array rule the design works under
is Allo's, not Vitis's, and that it cost this design **0 cycles**: every
restructure above was Allo-legal, and the landed design is too. See
:ref:`limitation-shared-memory` and the note on :doc:`tinytpu_isa` ("One owner
per memory").

.. _gemmini-attribution-reproduce:

Reproducing the attribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

From ``examples/accelerator/tinytpu_vitis/impact/`` on ``main``:

.. code-block:: bash

   source env.sh                    # conda allo, LLVM_BUILD_DIR, OMP, TMPDIR
   python make_variants.py          # writes v_base.py and v_*.py from 7a24c21e
   python pyrun.py bench_variant.py v_design_dep                     # simulator
   TPU_SHAPES=4x4x4,16x16x16 python pyrun.py cosim_variant.py v_design_dep runs/v_design_dep
   ./profile.sh runs/v_design_dep   # per-process timeline of the last shape

``make_variants.py`` reads its baseline from git at ``7a24c21e`` -- the design
the table calls ``base`` -- not the shipped file, so the variants keep
measuring what the table says; it gives them the baseline's own hand-written
program, since ``isa_dsl`` now targets the landed ISA. ``cosim_variant.py
base`` builds the *shipped* design; the pre-landing one is ``v_base``.
``pyrun.py`` strips the conda env's editable finder, so ``allo`` resolves to
this checkout. ``cosim_variant.py`` reuses ``../cosim.py`` (``align_value`` 64,
widen 512, ``-B/usr/bin``, ``m_axi`` depths) and adds only a
``patch_kernel(prj)`` hook for variants that edit the emitted C++.
``profile.sh`` re-runs cosim with ``-enable_dataflow_profiling``, then re-runs
the xsim snapshot so the monitor's CSVs survive (cosim deletes them).
``analyze_df.py`` and ``rle_df.py`` turn them into per-process
run/starve/block counts and run-length traces. Raw outputs, timelines and the
Vitis shared-array probe summaries are under ``results/`` and
``probe_shared/``; ``ar_distance_probe.py`` is the RTL probe behind the
accumulator distance contract.

Per-workload builds
~~~~~~~~~~~~~~~~~~~

**This is the correction that matters most.** Earlier revisions compared
per-workload builds against Gemmini's single elaboration: ``M``, ``K``, ``N``, the
instruction count and every unit's loop bound were compile-time constants, so
4x4x4 and 16x16x16 were *different accelerators*. That is not a comparison an
instruction-programmable claim can rest on. What replaced it is `One hardware
build, workload swept as data`_.

The withdrawn comparison, and why it was wrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   **WITHDRAWN 2026-09-18: "faster than Gemmini at all five shapes", and the
   3.2x fixed-cost win.** Both stood in ``COMPARISON.md`` and both are wrong --
   not imprecise, wrong -- because the two columns measure different things.
   They are struck here rather than footnoted.

The earlier claim, "faster at all five shapes", compared our
accelerator-only count with Gemmini's ``tiled_matmul_auto``, and about 395
cycles of that call is Rocket driver software. **That claim is withdrawn**,
as is the "3.2x fixed-cost win" that went with it. Ratios in the history
sections (the I/O-trade and T=16 tables, and the marginal-cost fits) are
against those end-to-end Gemmini numbers; they are kept, on this page and on
:doc:`tinytpu_history`, as a record of our own progress, not as comparisons.

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

Gemmini's marginal efficiency, once read as 100% of peak
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A correction worth recording:** an earlier analysis put Gemmini's marginal
efficiency at "essentially 100% of peak" from the two-point difference
``986 - 574 = 412`` cycles. That is wrong: 412 cycles for the 4032 additional
MACs is 9.79 MAC/cycle, i.e. **61.2%**. The overhead-vs-ceiling distinction
survives the correction, but at 1.7x rather than 2.7x.

The latency knob, as read before the grid was run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This was written before the grid was run and is kept because the reasoning is
still right about fitted knobs, and wrong about ours: One argument for that
comes from the other side of the comparison, against their own interest: a
*fitted* knob invites belief — 92 cycles was carried for months, looked
authoritative, and inverted a ranking — whereas a latency of zero is so
obviously not a claim about silicon that nobody is tempted to quote it as one.
Having no memory model is not therefore better than having a fitted one; what
is better than either is reporting the range over which a conclusion holds.

A shared hypothesis died in the process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both projects had suspected a common mechanism: a single write port serialising
a step that a second port would relieve. It was set up as a falsifiable test and
**their half failed.** Banking their vector-register write port at 2 and 4 banks
changes their emitted GEMM by **zero cycles at every shape** — the deferrals are
on the *read* port, and a second write port was separately priced at +1.36 %
area while failing timing, which at a WNS of +0.053 ns is fatal. Relieving it
would move their ceiling from 61.5 % to 66.7 % and no further, because their
stream engine sits only 4 cycles behind it.

So the two are cousins rather than the same mechanism, and any case for a
second issue path on our side has to stand on our own measurement rather than
on the parallel. Recorded because a hypothesis that both sides liked and that
one side has now falsified is worth more than one nobody tested.

.. rubric:: Footnotes

.. [#mixedtrials] An earlier revision published 413 / 396 / 395 / 394 / 395,
   which mixed trials; the conclusion is unchanged.
