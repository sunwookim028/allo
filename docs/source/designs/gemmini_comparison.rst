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

A cycle comparison between TinyTPU-isa and a Gemmini elaborated to match it:
int8/int32, the same array size, the same shapes, the same operand
distribution, with both sides measured on RTL. It produces a per-shape cycle
count for each machine over a named measurement window, together with the
patches, pins and commands needed to rebuild either column. Our side needs the
``allo`` conda environment and Vitis HLS cosim; Gemmini's needs a Chipyard
checkout and Verilator. The design under test is :doc:`tinytpu_isa`; the
measurements, and every claim this page has withdrawn, are on
:doc:`gemmini_results`.

Quick start
-----------

Our column, from a clean checkout, in the ``allo`` environment with
``LLVM_BUILD_DIR`` exported:

.. code-block:: bash

   examples/accelerator/tinytpu_vitis/reproduce.sh   # checks the five cycle counts

What it checks is :ref:`tinytpu-isa-verify`. The parity configurations, which
are the ones measured against matched Gemmini across every shape of the
matched set:

.. code-block:: bash

   cd examples/accelerator/tinytpu_vitis
   TPU_PARITY_CONFIG=parity-t4 python parity_sweep.py      # 10 shapes
   TPU_PARITY_CONFIG=parity-t8 python parity_sweep.py      #  8 shapes

Gemmini's column, from the committed patches. ``~/chipyard/env.sh`` activates
Chipyard's own conda environment and so replaces the ``allo`` one; source it in
a separate shell (``dev/toolchains.rst``).

.. code-block:: bash

   G=<repo>/examples/accelerator/tinytpu_vitis/gemmini
   cd ~/chipyard && git apply $G/chipyard_RoCCAcceleratorConfigs.patch
   cd generators/gemmini && git apply $G/gemmini_CustomConfigs.patch
   cd software/gemmini-rocc-tests
   git apply $G/roccTests_gemmini_h.patch
   git apply $G/roccTests_gemmini_params_h.patch
   git apply $G/roccTests_Makefile.patch
   cp $G/allo_cmp.c $G/allo_bare5.c bareMetalC/

   cd ~/chipyard && source env.sh
   cd sims/verilator && make CONFIG=Int8Dim4GemminiRocketConfig -j16
   ./simulator-chipyard.harness-Int8Dim4GemminiRocketConfig +permissive +permissive-off \
     ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/allo_cmp-baremetal

One Verilator run takes about 80 s and covers all five shapes; the
accelerator-only window benchmark, ``allo_bare5``, is the same order (~90 s).
The run opens with a boot banner, and that banner is what proves which
hardware produced the numbers -- ``GEMMINI DIM=4 elem_t_bytes=1`` for the
matched DIM=4 build. Raw output from the recorded run is in
``dev/records/tinytpu/logs/gemmini_int8_dim4.log``.

``allo_bare5`` needs its own make target: the committed
``roccTests_Makefile.patch`` adds only ``allo_cmp`` to the ``tests`` list. The
full file-by-file account of the Gemmini side is in
:ref:`gemmini-reproduce`.

How it works
------------

Making the baseline matched
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _gemmini-one-build:

One hardware build, workload swept as data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Earlier revisions of this page compared per-workload builds against Gemmini's
single elaboration; that is recorded in :ref:`Earlier measurements and corrections <gemmini-history>`.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The audit that withdrew the claims recorded at the foot of this page
(:ref:`Earlier measurements and corrections <gemmini-history>`). Both sides re-measured on the
elaborated chipyard tree, not argued from documentation.

**The memory models match, so this is not the problem.** Gemmini's harness is
``WithBlackBoxSimMem(additionalLatency=0)`` (``AbstractConfig.scala:19`` ->
``HarnessBinders.scala:122-129``), confirmed in the elaborated
``TestHarness.sv``. That ``SimDRAM`` fork only instantiates DRAMSim2 given
``+dramsim``; otherwise it is ``mm_magic_t`` at ~1-2 cycle AXI latency.
Re-running the existing simulator and ELF both ways gives ``MLP 4.8.8.4`` = 1146
without and 1174 with, and ``dev/records/tinytpu/logs/gemmini_int8_dim4.log`` records 1146 with no
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Everything above was measured with** ``-m_axi_latency 0`` -- a memory that
answers immediately. That is Vitis's default and it was never stated, so it is
stated here, along with what happens when it is not true.

Rebuilding the design at a given read latency and cosimulating it there
(``cosim.py`` with ``TPU_AXI_LATENCY``; ``dev/records/tinytpu/logs/cosim_isa_landed_axi_latency.log``,
and ``dev/records/tinytpu/logs/cosim_isa_axi_latency_sweep.log`` for the pre-landing design):

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

Counting host overhead symmetrically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page's window analysis established that a flat **~393-413 cycles** of
Rocket driver software sits inside Gemmini's end-to-end figure, independent of
shape — which is 72 % of the 4x4x4 number and nearly the whole gap between the
two windows. Our own figures are Vitis cosim counts from ``ap_start`` to
``ap_done``, and so contain no host at all.

MiniTPU's owner supplied the symmetric datum, unprompted, and it is the reason
to state this as a methodological rule rather than as a point in our favour:
**their per-launch cost outside the fabric is about 143 microseconds**, which
their testbench numbers do not include either.

.. note::

   **Corrected 2026-09-23, by its own side.** This figure was first given, and
   first published here, as *host* work — and most of it is not. Decomposed:
   **~30 µs is the register path** -- of which, measured against a no-bus
   control, only **0.442 µs is the bus and ~29 µs is CPython** -- and **~52 µs
   plus
   10.8 µs per KiB is the device reloading its own instruction memory**
   (112.7 µs for a 5.62 KiB image). Of 46.76 ms saved by cutting launch count,
   **81% was the device refetching the same image** and only 8.88 ms was host
   work. It is therefore **not comparable to Gemmini's ~390-cycle software
   driver at all**: theirs is a driver, this is mostly on-device instruction
   fetch plus bus latency. The asymmetry is real and still large; *"they have a
   slow host"* is the wrong lesson.

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
this page (:ref:`Earlier measurements and corrections <gemmini-history>`): a number that is true of
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
which is kept at the foot (:ref:`Earlier measurements and corrections <gemmini-history>`).

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

.. _gemmini-parity:

The parity baseline
~~~~~~~~~~~~~~~~~~~

A second, named configuration of our design, kept **beside** the shipped one,
whose purpose is to perform at parity with a matched Gemmini by changes that
are reasonable rather than contrived. The shipped design is unchanged.

The rules, fixed before any measurement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``parity-t4`` and ``parity-t8`` in ``parity_sweep.py``: the shipped design
with **one** change, the banked burst widening, at the matched array sizes.

.. code-block:: bash

   cd examples/accelerator/tinytpu_vitis
   TPU_PARITY_CONFIG=parity-t4 python parity_sweep.py      # 10 shapes
   TPU_PARITY_CONFIG=parity-t8 python parity_sweep.py      #  8 shapes

=========================  ==========================================
``T``                      4 (against Gemmini DIM=4) / 8 (DIM=8)
``MAXDIM``                 64, both sides
``DMA_WORDS``              16 (``TPU_DMA_WIDEN=1``), buffers banked
``TPU_PROGRAM``            ``interleaved`` (T=4); ``shipped`` at T=8
``QD``                     32 at T=4; 8 at T=8
=========================  ==========================================

One csynth per configuration; each (shape) cosim runs in its own copy of that
one synthesized solution, so every number in a column is the same netlist.

Reference
---------

.. _gemmini-reproduce:

Reproducing the Gemmini baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
^^^^

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
^^^^^

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
^^^^^^^^^^^^^^^^^^^^

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
Raw Gemmini output is in ``dev/records/tinytpu/logs/gemmini_int8_dim4.log``.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **The stock** ``matmul`` / ``matmul_ws`` **tests print no cycle counts** --
  their ``read_cycles()`` calls are commented out upstream. ``allo_cmp.c`` exists
  because of this; do not expect to get numbers from the shipped benchmarks.
* **One Verilator run takes about 80 s**, not the 22 minutes an earlier
  revision of this page claimed -- wrong by ~17x, and contradicted by this
  repository's own ``dev/records/tinytpu/logs/gemmini_int8_dim4.log``, which records
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

Which Gemmini configuration is the honest opponent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
  claim recorded at the foot of this page (:ref:`Earlier measurements and corrections <gemmini-history>`).

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

.. _gemmini-capacity-neutral:

The capacity asymmetry is cycle-neutral, and this is the finding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The obvious objection to this comparison is memory capacity. Gemmini has a
**256 KiB scratchpad and a 64 KiB accumulator**; ours are smaller by one to
three orders of magnitude, depending on which of our builds is meant — and the
distinction matters enough that an earlier revision of this page got it wrong.

.. warning::

   **Corrected 2026-09-22.** This section used to say we have "a 4 KiB
   scratchpad, 4 KiB of vector registers and a 2.1 KiB accumulator", and put
   the asymmetry at 64x and 30x. Those figures are **``MAXDIM=64``**. The
   memories are derived from ``MAXDIM``, so at the **``MAXDIM=16`` baseline —
   the build every published cycle count and the published 1,136,598 cell area
   belong to** — the scratchpad and the vector registers are 64 rows of 32 bits
   each, **256 bytes apiece**, and the accumulator is 2.0 KiB. Measured out of
   the shipped RTL, not read off the source:

   .. list-table::
      :header-rows: 1
      :widths: 36 16 16 16 16

      * - variant
        - scratchpad
        - vregs
        - accumulator
        - all RAM modules
      * - ``T4_MAXDIM16_shipped_baseline``
        - 0.25 KiB
        - 0.25 KiB
        - 2.0 KiB
        - 3.50 KiB
      * - ``T4_MAXDIM64_shipped``
        - 4 KiB
        - 4 KiB
        - 2.1 KiB
        - 18.62 KiB
      * - ``T8_MAXDIM64``
        - 4 KiB
        - 4 KiB
        - 4.25 KiB
        - 20.75 KiB

   So the asymmetry against the baseline is **1024x the scratchpad and 32x the
   accumulator**, not 64x and 30x. The "all RAM modules" column adds the DMA
   read buffers and the sequencer's small RAMs, which are storage but not
   operand storage. The cycle-neutrality finding below is unaffected — it is a
   statement about Gemmini's tiling search, not about our capacities — but the
   *area* argument in :ref:`Area: the axis this page was missing <gemmini-area>` rests on these
   numbers and uses the corrected ones.

The asymmetry does not buy Gemmini a single cycle at any published shape. Re-running
``tiled_matmul_auto``'s own tiling search across capacities at DIM=4 gives the
``loop_ws`` call count below; every row is legal under the config's ``require``
clauses:

=========================== ======= ========== ========== ========== =========
Scratchpad / accumulator    4x4x4   16x16x16   32x32x32   64x64x64   64x32x64
=========================== ======= ========== ========== ========== =========
256 / 64 KB (what we ship)  1       1          1          1          1
64 / 32 KB (``chipConfig``) 1       1          1          1          1
32 / 8 KB                   1       1          1          4          4
8 / 4 KB (the area build)   1       1          4          12         12
4 / 4 KB (the floor tested) 1       1          4          32         12
=========================== ======= ========== ========== ========== =========

Every capacity from 256 KB down to 4 KB issues exactly **one** ``loop_ws`` from
4x4x4 through 16x16x16, so shrinking Gemmini's memories would not move any of
the five published numbers. That is what licenses the 8 / 4 KB elaboration the
area comparison uses. **The asymmetry is an area disclosure, not a cycle
correction** — which is a far stronger position than "we could not match it".
The binding limit is always the accumulator's half capacity, never the
scratchpad.

(The two lower rows used to be labelled "~2x ours" and "~matched to ours".
They are neither: against the ``MAXDIM=16`` baseline's 0.5 KiB of operand
storage, 8 KB is 16x and 4 KB is 8x. Against ``T8_MAXDIM64`` the 8 / 4 KB row
*is* matched, to 2 %.)

It stops being neutral at exactly 32x32x32, so as shapes grow this argument
needs restating rather than reusing. Keeping Gemmini's own default capacity is
also the choice that avoids hand-rolling a multi-tile nest and then arguing
about whether our tiling was fair.

Limits and known failures
-------------------------

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

One hazard, again
~~~~~~~~~~~~~~~~~

Elaborating these two configurations rewrote ``gemmini_params.h`` twice, as
every elaboration does. The header was restored to the committed ``DIM=4``
snapshot afterwards and verified by checksum. The exporter does not trust that
header at all: it reads ``DIM`` back out of the scratchpad's own geometry in
``.top.mems.conf`` — a bank is ``DIM`` int8 elements wide, an accumulator row is
``DIM`` int32 — and refuses an export whose RTL does not match the
configuration it claims. The C benchmark build still depends on the header, so
the snapshot-and-restore step is still required for anyone re-running cycles.

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

Power: declared absent, with the reason
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This project's own stated principle is that informed co-design needs timing
**and** power across an array of architectures, so power's absence here is a
real gap in the judge, and it is declared rather than papered over.

**No power number from this flow is publishable, and none will appear in any
table on this page.** DC's ``report_power`` output is a default-toggle-rate
estimate with no activity data. An earlier run's 57.1 mW was recorded and has
already been withdrawn once; it must not travel again, in any row, however
marked.

What closing the gap would actually take, and why it is not worth doing for
this evaluation:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - level
     - cost
     - what it still would not tell you
   * - RTL switching activity
     - 1-2 days of bring-up: scope a VCD to ``tinytpu_isa`` under xsim and to
       the ``Gemmini`` instance under Verilator, convert to SAIF, ``read_saif``
       before compile, plus the DC runs again
     - the two SAIFs describe **different windows** — ours is ``ap_start`` to
       ``ap_done``, Gemmini's necessarily includes the Rocket driver and the
       fences — so average toggle rates are diluted differently and the
       comparison is not like-for-like even though both numbers are real
   * - gate-level activity
     - the above, plus simulating the DC netlist against the NanGate45 Verilog
       models with the cosim testbench, which is C++ and drives AXI — a real
       bring-up on our side and a new harness on Gemmini's
     - still no wire capacitance
   * - the honest version
     - SRAM macros instead of flip-flop memories, **and** place-and-route
     - nothing — but it replaces the area methodology, so every published
       figure on this page would have to be re-run

The blocker is not effort, it is that under ``sram_mode='none'`` and without
P&R a power number is dominated by two artefacts larger than the effect being
measured. First, **clock power into flip-flop arrays**: 98,304 flops of
Gemmini memory and 200,561 sequential cells of ours are toggled on every edge
that is not gated, and a real SRAM's dynamic energy per access is nothing like
a flop array's — the same objection that makes the total-area column fragile,
except power is *more* sensitive to it, not less. Second, **no interconnect**:
with no P&R the wire capacitance is a wire-load-model guess, and at 45 nm
interconnect is a large share of dynamic power.

**Recommendation: do not pursue power for this evaluation.** The paper says
power is absent, and says this is why. That is a stronger position than an
indicative number quoted as a measurement.

Stated precisely, because a parallel effort is producing switching-activity
files and this must not read as an argument against it: there are **three**
objections to a power number from this flow, and activity annotation removes
exactly one of them. It removes the default-toggle-rate objection, which is the
one that makes the current figure unpublishable outright, and it is worth
having on its own terms. It does not touch the flip-flop-memory artefact or the
missing interconnect. So an annotated number from this flow would be
*compromised* rather than *meaningless* — a real improvement, and still not a
measurement of either design's power.

That is compatible with the methodology commitment in ``dev/paper_outline.md``
(publish power only if **both** sides are activity-annotated, otherwise report
none and say why): if only one side is annotated the answer is no on that
ground alone, and if both are, the answer is still no on these two. The
cheapest credible energy axis is SRAM macros plus place-and-route — the same
prerequisite that would replace the area methodology. If the project later
wants energy, that is the order to do it in: macros first, because they fix the
area artefact and the power artefact at once.

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

Results and history
-------------------

Every measurement this page's method produces, and every claim it has
withdrawn, are on :doc:`gemmini_results`: the like-for-like table, the
marginal-cost sweep, the area and frequency comparison, the parity baseline's
results at T=4 and T=8, the independent reproduction, and
:ref:`gemmini-history`.
