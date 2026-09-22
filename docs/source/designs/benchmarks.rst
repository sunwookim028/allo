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

#################
The Benchmark Set
#################

Three sets, answering three different questions. They are never blended into
one verdict, because the questions are not the same question.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - set
     - what it measures
     - what it cannot say
   * - **latency**
     - time to first result: 4x4x4 .. 16x16x16, where the fixed pipeline term
       dominates
     - nothing about throughput. The same shape is not the same work on two
       different array sizes.
   * - **steady state**
     - MACs/cycle as a fraction of the array's own peak: 16, 32, 48, 64 cubed
       plus two non-cubic shapes
     - nothing about latency, and nothing about a real layer's memory traffic
   * - **workload**
     - one GPT-2 projection, stated exactly
     - nothing yet: it is a **placeholder**. No build runs it (below).

.. important::

   **Which question a number answers, per the framing agreed with MiniTPU's
   owner.**

   * **TinyTPU-isa vs Gemmini is well posed**: matched 4x4 array, matched
     int8/int32, matched near-zero memory latency, matched measurement window.
     It is measured on both sets below.
   * **MiniTPU vs either is NOT well posed at these shapes**: a 16x16 BF16
     array has 16x the MAC/cycle peak and a different window, so a cycle
     ratio against it is a statement about array size. It becomes well posed
     only through fraction-of-peak, or through an equal-array-size build.
     Every MiniTPU figure carried elsewhere in these docs is **dated, not
     permanent** --- its board throughput moved on unchanged silicon within a
     day of being recorded, so the 79.16 / 129.70 tok/s pair is superseded and
     must not be quoted. **No figure on this page depends on it**; nothing
     here quotes a MiniTPU throughput at all. Its board clock is 187.5 MHz
     while its bitstream closes timing at 200 MHz, so a cycles-to-seconds
     conversion uses 187.5 and a timing-closure claim uses 200.

   A single blended verdict across the three machines is the one thing not to
   produce, because it will be quoted without its caveat.


.. important::

   **Every figure names its window, and no figure counts a host.** A cycle
   count without its window is not a measurement, so:

   * **Ours** is Vitis ``cosim`` (xsim), ``ap_start`` to ``ap_done``, with the
     program already in DRAM. There is **no host driver at all** --- not
     excluded, absent.
   * **Gemmini's** is ``rdcycle`` -> 5 ``config``\ s -> one hardware
     ``loop_ws`` -> ``fence`` -> ``rdcycle``, which **excludes** its ~395-cycle
     Rocket driver.
   * **MiniTPU's** testbench figures likewise exclude its per-launch host
     work, about 132 us with a 44 us register-access floor measured on board
     .187, i.e. roughly 24 750 cycles at 187.5 MHz.

   So: **no machine's host is counted anywhere on this page**, which is the
   consistent choice, and it is stated rather than assumed. What that
   accounting turned up on MiniTPU's side is in `Earlier measurements and
   corrections`_.

   **Both sides are idealised memory, and ours is a knob rather than a
   model.** Every cycle figure here is at ``TPU_AXI_LATENCY=0`` unless the row
   says otherwise; Vitis cosim has no DRAM model, so 0 is a value we chose.
   Gemmini's harness uses ``WithBlackBoxSimMem(additionalLatency=0)``, roughly
   1-2 cycle AXI. **Neither column contains a memory system.** What that costs
   us is measured in :ref:`benchmarks-latency-grid`.


.. _benchmarks-why-latency-is-not-throughput:

Why the latency set is not a throughput comparison
==================================================

The five shapes top out at 16x16x16. On a 4x4 array that is
:math:`(16/4) \times (16/4) = 16` weight loads and 16x16 = 256 wavefront rows;
on a 16x16 array it is **one** weight load and one pass. The per-machine work
accounting, for 16x16x16:

.. list-table::
   :header-rows: 1

   * - machine
     - array
     - tile-matmuls
     - wavefront rows
     - peak MAC/cycle
     - ideal cycles
   * - TinyTPU-isa, T=4
     - 4x4
     - 16
     - 256
     - 16
     - 256
   * - TinyTPU-isa, T=8
     - 8x8
     - 4
     - 32
     - 64
     - 64
   * - Gemmini, DIM=4
     - 4x4
     - 16
     - 256
     - 16
     - 256
   * - MiniTPU, 16x16
     - 16x16
     - 1
     - 16
     - 256
     - 16

"Ideal cycles" is :math:`(N/T)(K/T)M`, one wavefront row per cycle at T*T
MACs. A machine with a 16x deeper array needs 16x fewer cycles for the *same
shape* before any design difference is involved, so the shortest pipeline wins
this set by construction. That is why these five are labelled a **latency**
benchmark and why the steady-state set exists.


.. _benchmarks-steady:

The steady-state set
====================

Cubic 16, 32, 48, 64 plus **64x32x64** and **32x64x32**. The two non-cubic
shapes are not decoration: the cost model is
:math:`\text{fixed} + (N/T)(K/T) \cdot M`, so a shape that holds M and N and
halves K separates the wavefront-row term from the tile-count term, and
swapping M and K distinguishes the two.

64 is the largest cubic shape the build runs, and what bounds it is the
**instruction encoding**, not the datapath --- see
:ref:`benchmarks-raising-maxdim`.

TinyTPU-isa, T=4, MAXDIM=64
---------------------------

Vitis ``cosim`` (xsim), ``ap_start`` to ``ap_done``, one ``csynth`` and one RTL
build for the whole sweep, shapes swept as instruction data. Every shape
**bit-exact** (``mismatches = 0``). Peak is :math:`T^2 = 16` MAC/cycle.

.. list-table::
   :header-rows: 1
   :widths: 14 10 12 12 12 14 12

   * - shape
     - set
     - MACs
     - cycles
     - MAC/cycle
     - **% of peak**
     - ideal cycles
   * - 4x4x4
     - latency
     - 64
     - 218
     - 0.29
     - 1.8%
     - 4
   * - 8x8x8
     - latency
     - 512
     - 357
     - 1.43
     - 9.0%
     - 32
   * - 12x12x12
     - latency
     - 1 728
     - 563
     - 3.07
     - 19.2%
     - 108
   * - 16x16x8
     - latency
     - 2 048
     - 677
     - 3.02
     - 18.9%
     - 128
   * - 16x16x16
     - both
     - 4 096
     - 879
     - 4.66
     - 29.1%
     - 256
   * - 32x32x32
     - steady
     - 32 768
     - 3 752
     - 8.73
     - 54.6%
     - 2 048
   * - 48x48x48
     - steady
     - 110 592
     - 10 289
     - 10.75
     - 67.2%
     - 6 912
   * - 64x64x64
     - steady
     - 262 144
     - 22 123
     - 11.85
     - **74.1%**
     - 16 384
   * - 64x32x64
     - steady
     - 131 072
     - 12 907
     - 10.16
     - 63.5%
     - 8 192
   * - 32x64x32
     - steady
     - 65 536
     - 6 824
     - 9.60
     - 60.0%
     - 4 096

**This is the number that characterises the design, and it had never been
measured.** Utilisation rises monotonically with problem size and reaches
**74.1% of the array's peak at 64x64x64**, still climbing. The latency set
sits at 1.8-29% of peak, which is the quantitative statement of why it is not
a throughput benchmark.

TinyTPU-isa, T=8, MAXDIM=64
---------------------------

The same RTL flow at the second array size. Peak is :math:`T^2 = 64`
MAC/cycle, so **the fractions are against a four-times-larger peak and the
raw cycles are not comparable with the T=4 column** --- which is exactly why
the fraction is the column to read.

.. list-table::
   :header-rows: 1
   :widths: 14 12 12 14 16 14 16

   * - shape
     - MACs
     - cycles
     - MAC/cycle
     - **% of peak (64)**
     - ideal cycles
     - T=4 cycles / T=8
   * - 8x8x8
     - 512
     - 285
     - 1.80
     - 2.8%
     - 8
     - 1.25x
   * - 16x16x8
     - 2 048
     - 424
     - 4.83
     - 7.5%
     - 32
     - 1.60x
   * - 16x16x16
     - 4 096
     - 493
     - 8.31
     - 13.0%
     - 64
     - 1.78x
   * - 32x32x32
     - 32 768
     - 1 484
     - 22.08
     - 34.5%
     - 512
     - 2.53x
   * - 48x48x48
     - 110 592
     - 3 537
     - 31.27
     - 48.9%
     - 1 728
     - 2.91x
   * - 64x64x64
     - 262 144
     - 7 083
     - 37.01
     - **57.8%**
     - 4 096
     - **3.12x**
   * - 64x32x64
     - 131 072
     - 4 523
     - 28.98
     - 45.3%
     - 2 048
     - 2.85x

Two things to read off this, and they pull in opposite directions:

* **The array scales**: quadrupling the array gives 3.12x at 64x64x64, i.e.
  78% scaling efficiency, and the ratio is still rising with problem size.
  The design's T parameter works.
* **A bigger array needs a bigger problem.** T=8 reaches only 57.8% of *its*
  peak at 64x64x64 where T=4 reaches 74.1% of *its* peak, because 64x64x64 is
  a smaller problem relative to an 8x8 array. So 64 is a steady-state shape
  for T=4 and is still in the ramp for T=8 --- which is the same observation
  that makes the five small shapes a latency benchmark, one level up. A T=8
  steady-state set would need to start where this one ends.


.. _benchmarks-msweep:

Does utilisation move with M? Yes --- we amortise
=================================================

An M-sweep at fixed K=N=64, everything else held, run at both array sizes.
This answers a question **neither** project had measured, and it was asked
because MiniTPU's own M-sweep came out **flat**.

.. list-table:: M-sweep, K=N=64, one ``csynth`` per T, every shape bit-exact
   :header-rows: 1
   :widths: 8 10 12 12 12 14 16

   * - M
     - joint?
     - T=4 cycles
     - **% of 16**
     - T=8 cycles
     - **% of 64**
     - MACs
   * - 16
     - ours alone
     - 6 891
     - **59.4%**
     - 2 539
     - **40.3%**
     - 65 536
   * - 32
     - **joint**
     - 11 952
     - **68.5%**
     - 4 048
     - **50.6%**
     - 131 072
   * - 64
     - **joint**
     - 22 123
     - **74.1%**
     - 7 083
     - **57.8%**
     - 262 144

**Utilisation rises monotonically with M at both array sizes.** We amortise.

**The mechanism, and it is the predicted one.** The numbers are a straight
line in M with a single fixed intercept:

.. list-table::
   :header-rows: 1

   * -
     - fixed term
     - marginal, cycles per M row
     - fixed as % at M=16
     - at M=64
   * - T=4
     - **1 782**
     - 317.8
     - 25.9%
     - 8.1%
   * - T=8
     - **1 014**
     - 94.8
     - 39.9%
     - 14.3%

Fitted on M=32 and M=64, the model predicts M=16 to within 24 cycles (T=4)
and 8 cycles (T=8) --- so the fixed term is paid **once per call**, not once
per row block, which is exactly what "one program per call with M as an
internal loop bound" predicts. Tripling M does not triple the fixed cost; it
divides it.

.. important::

   **What this licenses, and what it does not.** MiniTPU is flat at 33.6%
   across M=32, 64 and 128, with an identified cause: an emitter issuing one
   launch per 32 rows, each re-paying the whole fixed term. We are not flat.
   **So flatness is not a property of how these machines re-enter a launch ---
   it is specific to that emitter**, and carrying the row-block loop inside
   the launch would make their structure ours.

   But the comparison is of an **axis, not of shapes**: our sweep is at
   K=N=64 and theirs at K=N=128, because 128 is past our ceiling
   (:ref:`benchmarks-workload`) and M=16 is below their 32-row floor. "Does
   utilisation move with M" is answerable on each machine independently and
   that is what makes the finding joint; the *numbers* are not comparable
   across the two columns. Only M ∈ {32, 64} exists on both machines at all,
   and even there the fixed K and N differ.

   For orientation and not as a comparison, Gemmini over its own M range goes
   80.7 -> 86.1 -> 95.1% at DIM=4 and 47.3 -> 58.3 -> 79.0% at DIM=16 --- it
   amortises too, and harder than we do.


.. _benchmarks-raising-maxdim:

Raising the build: what a bigger MAXDIM costs
=============================================

The design was built at MAXDIM=16 with a 512-row scratchpad, 256 operand
vector registers and 128 accumulator registers. Those were three typed-in
literals that happened to be big enough for MAXDIM=16, and **only** for
MAXDIM=16: the shipped GEMM lays A out at ``A_VR + kb*MAXDIM + m``, so the
highest operand row it names is

.. math::

   (\text{MAXDIM}/T - 1)\,\text{MAXDIM} + \text{MAXDIM}
   \;=\; \text{MAXDIM}^2 / T

which at MAXDIM=64 is 1024 rows against a 256-entry file. Raising MAXDIM alone
produced a build that assembled and gave wrong answers. The sizes are now
**that expression** (``microarch_isa.OPERAND_ROWS``), with the ``TPU_SPAD`` /
``TPU_NVR`` / ``TPU_NAR`` environment overrides kept only for probing a
deliberately undersized build. MAXDIM itself defaults to **64**.

Two independent ceilings, both in the encoding
----------------------------------------------

Found by raising MAXDIM until each fired:

.. list-table::
   :header-rows: 1
   :widths: 26 30 20 24

   * - limit
     - condition
     - ceiling at T=4
     - how it fails
   * - address field, 11 usable bits
     - :math:`\text{MAXDIM}^2/T \le 2047`
     - MAXDIM <= 88 (90 unrounded; MAXDIM is a multiple of T)
     - MAXDIM=96: ``check_program``, "AGU-resolved f3=2112 is outside the
       0..2047 range"
   * - header count, 15-bit slice
     - :math:`\text{MAXDIM}^3/T^2 + \text{MAXDIM}^2/T \le 32767`
       (``accu``'s iteration count)
     - MAXDIM <= 76 for a cubic shape
     - MAXDIM=80: ``assemble``, "header count 33600 does not fit 15 bits"

**MAXDIM=64 is the largest round value inside both**, and it is asserted at
import so an out-of-range configuration fails immediately rather than
silently. Neither ceiling is architectural --- widening the fields or the
header slices moves them --- but the field widths are load-bearing for a
different reason: a synthesis tool bounds a runtime-bounded loop by the *range
of the index*, which is why ``nr`` is 8 bits (see ``microarch_isa``'s encoding
note). Widening means re-measuring every loop bound Vitis derives.

At T=8 the same two expressions give MAXDIM <= 120, verified: the ceiling is a
property of :math:`\text{MAXDIM}^2/T`, so a wider array buys a longer edge.

The instruction memory does **not** need to grow
------------------------------------------------

``IMEM_SIZE`` stays at **56 words** at every shape up to 64x64x64, because
``isa_dsl``'s loop-nest generator already emits the tiled GEMM as
:math:`O(\text{nesting})` rather than :math:`O(\text{tiles})`:

.. list-table::
   :header-rows: 1

   * - shape
     - looped (shipped)
     - unrolled reference
     - dynamic instructions
   * - 16x16x16
     - 14 static / 36 words
     - 32 static / 72 words
     - 32
   * - 32x32x32
     - 14 static / 36 words
     - 96 static / 200 words
     - 96
   * - 64x64x64
     - 14 static / 36 words
     - 320 static / 648 words
     - 320

So the answer to "must imem grow, or must the program loop harder" is: the
program already loops, ``isa_dsl`` already handles it, and imem is untouched.
The unrolled form outgrows imem from 16x16x16 on, which is the point of having
control flow; ``bench_isa.py`` runs it where it fits and covers the rest with a
static equivalence check that the two forms issue the identical dynamic
instruction stream.

MAXDIM -> resources
-------------------

One ``csynth_design`` per configuration, ``xcu280-fsvh2892-2L-e``, 3.33 ns
target (``csynth_sweep.py``; reports kept under
``examples/accelerator/tinytpu_vitis/csynth_reports/``).

.. list-table::
   :header-rows: 1
   :widths: 8 10 10 10 10 10 10 10 12

   * - T
     - MAXDIM
     - spad
     - nvr
     - FF
     - LUT
     - BRAM
     - DSP
     - est. clock
   * - 4
     - 16
     - 64
     - 64
     - 17 074
     - 26 493
     - 40
     - 14
     - 2.431 ns (411 MHz)
   * - 4
     - 32
     - 256
     - 256
     - 17 455
     - 26 529
     - 44
     - 14
     - 2.431 ns (411 MHz)
   * - 4
     - 48
     - 576
     - 576
     - 17 467
     - 26 651
     - 48
     - 14
     - 2.431 ns (411 MHz)
   * - 4
     - 64
     - 1 024
     - 1 024
     - 17 487
     - 26 552
     - 48
     - 14
     - 2.431 ns (411 MHz)
   * - **8**
     - 64
     - 512
     - 512
     - 43 911
     - 70 281
     - 62
     - 58
     - 2.431 ns (411 MHz)

Read the two parameters as separate columns, because they cost completely
different things:

* **MAXDIM resizes memories only.** 16 -> 64 is +413 FF (+2.4%), +59 LUT
  (+0.2%), +8 BRAM, no clock change.
* **T changes the shape of the region** --- T*T PE instances, T and T*T stream
  arrays. T=4 -> T=8 at MAXDIM=64 is **2.51x the FF, 2.65x the LUT and 4.1x
  the DSP**, for 4x the peak and a measured **3.12x** at 64x64x64. The
  estimated clock does not move, so the array scales at close to constant
  frequency and the trade is favourable: 2.5-2.65x the logic for 3.12x the
  throughput. (Note T=8's *memories* are smaller than T=4's at the same
  MAXDIM, since ``OPERAND_ROWS`` = MAXDIM^2/T --- a wider array packs more
  lanes into each row.)

**Raising MAXDIM from 16 to 64 costs +413 FF (+2.4%), +59 LUT (+0.2%) and
+8 BRAM (+20%), and does not move the estimated clock.** That is the whole
price of being able to run a shape that reaches steady state, and it is the
reason the shipped default moved to 64 rather than staying at 16 with an
override. Note the MAXDIM=16 row is itself *cheaper* than the design as
previously shipped, because the derived sizes replace the literal
512/256 scratchpad and vreg files with the 64/64 that MAXDIM=16 actually
needs.

Verification per configuration
------------------------------

Every configuration below is exact on all four gates: ``bench_isa.py``
(``ALL EXACT``), ``stress_isa.py`` (``STRESS OK``, which includes the program
validator's controls --- crafted illegal programs rejected, every generated
one accepted), and RTL cosim bit-exact at every shape reported.

**Re-run in full after the rebase onto main**, which brought the ACT rebuild,
the ``Encoding`` primitive and the ``shapes.py`` deduplication under this
work. Verbatim:

.. code-block:: text

   bench_isa TPU_MAXDIM=16       ALL EXACT
   bench_isa (default, 64)       ALL EXACT
   bench_isa TPU_SET=all         ALL EXACT
   stress_isa TPU_MAXDIM=16      STRESS OK: 492/492 runs exact ... GEMM at 64 shapes
   stress_isa (default, 64)      STRESS OK: 640/640 runs exact ... GEMM at 96 shapes
   stress_isa T=8 MAXDIM=64      STRESS OK: 630/630 runs exact ... GEMM at 96 shapes
   param_check TPU_MAXDIM=8      PARAM OK: 69/69 runs exact  (3 seeds ungeneratable)
   param_check TPU_MAXDIM=12     PARAM OK: 186/186 runs exact
   param_check T=8 MAXDIM=32     PARAM OK: 408/408 runs exact
   mutate.py --no-rtl            MUTATE OK: all 33 mutants run were caught
                                 (1 RTL-only not run: ar_claim_false)
   isa_dsl.py                    generated == hand-written, word for word
   kpn_model.py                  KPN OK
   pytest tests/act/             98 passed, 4 skipped

.. note::

   **The first ASIC number, and it describes the current design.** DC on
   FreePDK-45nm, memories as flip-flops, synthesis only: the shipped baseline
   (T=4, MAXDIM=16) comes out at **1,136,598** standard-cell area, timing MET
   at +0.21 ns with zero violating and zero hold paths. That **replaces**
   1,271,692 from an earlier export of a superseded netlist --- **-10.6%** ---
   and the reason is the same memory-sizing change that took BRAM 42 -> 40 and
   one cycle off the fixed term (:ref:`benchmarks-one-cycle`). The design got
   smaller *and* slightly faster, and area and cycles now describe the same
   design, which they did not before.

   Read it only against our own variants synthesised identically. A 45 nm
   cell area has no relationship to a BRAM count, and with memories as
   registers it says as much about the memory treatment as about the datapath.

``mutate.py`` is the one that matters most here, because ``bench_isa`` and
``stress_isa`` both cite it as the evidence that they catch a broken design,
and both were edited by this work: **all 33 mutants still caught.** The
RTL-only mutant (``ar_claim_false``, a false ``#pragma HLS dependence``
claim, which no simulator can see) needs a cosim of its own and was not run.

.. list-table::
   :header-rows: 1
   :widths: 10 12 14 20 26 18

   * - T
     - MAXDIM
     - peak MAC/cyc
     - ``bench_isa``
     - ``stress_isa``
     - cosim
   * - 4
     - 16
     - 16
     - ALL EXACT (5 shapes)
     - STRESS OK 492/492, 64 shapes, 18 bad programs rejected
     - published 171/261/417/483/685
   * - 4
     - 64
     - 16
     - ALL EXACT (11 shapes)
     - STRESS OK 640/640, 96 shapes, 18 bad programs rejected
     - bit-exact, 10 shapes (table above)
   * - 8
     - 32
     - 64
     - ALL EXACT
     - STRESS OK 486/486, 64 shapes
     - not run (64 is the reported T=8 build)
   * - 8
     - 64
     - 64
     - ALL EXACT (8 shapes)
     - STRESS OK 630/630, 96 shapes
     - bit-exact, 8 shapes (T=8 table)

The published MAXDIM=16 row was 172/262/418/484/686 before this work; see
`Earlier measurements and corrections`_.

.. warning::

   **The sampled shape set is a WEAKENING of the stress gate at large MAXDIM,
   and that should be stated plainly rather than defended.**
   ``stress_isa``'s exhaustive set is :math:`(\text{MAXDIM}/T)^3` --- 64
   shapes at MAXDIM=16 but **4096** at MAXDIM=64 --- so above
   ``TPU_STRESS_SHAPES`` (default 96) it becomes a stratified sample. The
   sample is deterministic and seeded, so a failure reproduces, and it always
   keeps the scored shapes and every extreme (each dimension at its smallest
   and largest). **It is still a sample**, and a bug that lives only at, say,
   36x52x20 would now be missed at MAXDIM=64 where it would not have been at
   MAXDIM=16.

   What makes the trade acceptable is the second half: **at MAXDIM=16 the set
   remains exhaustive at 64 shapes, so the shipped configuration's gate is
   unchanged** (``STRESS OK 492/492``). ``TPU_STRESS_SHAPES=0`` restores the
   exhaustive set at any MAXDIM for anyone willing to pay for it.

**The memories carry a floor, not just the GEMM's footprint.** Sizing them
purely to :math:`\text{MAXDIM}^2/T` was wrong and the CHIA parametricity gate
caught it: ``PARAM_CONFIGS`` runs the design at MAXDIM=8 and 12, where that
expression gives 16 and 36 rows, and ``stress_isa.random_program`` addresses a
fixed **64-row window** in each memory regardless of MAXDIM. Every GEMM shape
stayed bit-exact while ``param_check`` went to "only 0 of 24 random programs
could be generated" --- a failure that reads as a harness complaint rather
than a design change, which is what makes it worth recording. The memories are
now ``max(TEST_WINDOW, OPERAND_ROWS)``, and the resource table above is
unaffected because the operand term dominates from MAXDIM=16 up.

Two harness limitations were found and fixed rather than worked around, both
of them hard-coded constants masquerading as design limits:

* ``isa_dsl.vector_program`` addressed accumulator and scratchpad rows by the
  literals 40 / 80 / 100, which fit only because the memories were themselves
  literals. Its regions are now derived from ``SPAD_ROWS`` / ``NVR`` / ``NAR``
  --- they only ever had to be distinct, non-zero and in range --- and it now
  states its three real requirements as asserts: ``M >= T`` (its third ``mm``
  reads T weight rows out of an M-row region), ``MAXDIM // T >= 4`` (it names
  column block 3), and room for its DRAM rows. **This, not the design, is what
  "T=4 only" was.**
* ``bench_isa``'s shape list was unfiltered, so at T=8 it asked for 4x4x4.
  Both sets now pass through ``runnable()``.
* The set a run sweeps is ``bench_isa.SWEEP``, **not** ``bench_isa.SHAPES``.
  ``SHAPES`` is re-exported from ``shapes.py`` (the one definition) and is
  read *positionally* by ``act_compile``, ``kpn_model``, ``isa_dsl``,
  ``tests/act/test_tinytpu.py`` and --- through ``accept.BASELINES`` against
  ``shapes.NAMES`` --- the CHIA harness. [#sweepknobs]_


.. _benchmarks-matched:

Matched against Gemmini
=======================

What is matched, and what is not
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 38 42

   * - axis
     - matched?
     - how
   * - array size
     - **yes, at two sizes**
     - T=4 vs ``Int8Dim4GemminiRocketConfig``; T=8 vs
       ``Int8Dim8GemminiRocketConfig``
   * - data type
     - **yes**
     - int8 x int8 -> int32 on both, Gemmini's own default type
   * - operand distribution
     - **yes**
     - [-4, 4], the distribution ``allo_cmp.c`` fills
   * - DRAM row stride
     - **yes**, and this is new
     - both sides run at MAXDIM=64 (below)
   * - measurement window
     - **yes, by construction**
     - see the window table
   * - memory latency
     - both idealised, not identical
     - ours ``-m_axi_latency 0``; Gemmini's Rocket SoC with
       ``additionalLatency=0``
   * - on-chip capacity
     - **no** --- disclosed, not corrected
     - Gemmini 256 KiB scratchpad + 64 KiB accumulator against our 4 + 4 +
       2.1 KiB. At every shape in either set **both machines hold the entire
       working set on chip** (64x64 int8 is 4 KiB), and Gemmini's tiling
       search returns one ``loop_ws`` at every capacity from 256 KiB down to
       4 KiB for 4x4x4 .. 16x16x16. So it is an area and power asymmetry, not
       a cycle one, at these shapes. It would become a cycle asymmetry above
       32x32x32 if Gemmini's memory were shrunk to ours, where the driver
       would split 64x64x64 into 32 calls.
   * - dispatch software
     - **no** --- excluded from both
     - Gemmini's window excludes its ~395-cycle Rocket driver; ours has no
       host driver at all

Gemmini's config delta from its own published ``defaultConfig`` is **two
fields, and both of them are the array size** (plus
``has_training_convs = false``, conv hardware neither benchmark touches and
which does not appear in the generated header). Everything else ---
``spad_read_delay`` = 4, ``tile_latency`` = 0, ``mesh_output_delay`` = 1,
``dataflow`` = ``BOTH``, both capacities --- is left at Gemmini's default and
is named here as **unexercised**, because tuning any of them is tuning the
opponent. ``dataflow = BOTH`` means Gemmini carries an output-stationary
datapath neither benchmark uses, which favours us if it favours anyone.

The Gemmini builds, in terms another team can reproduce
-------------------------------------------------------

Stated exhaustively because **neither side's Gemmini figure has ever been
reproduced by anyone**, and two independent builds of the same nominal
configuration are the only cross-check available. If ours and someone else's
disagree, that is chased rather than resolved by picking the prettier number.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * -
     - DIM=4 point
     - DIM=8 point
   * - chipyard config class
     - ``Int8Dim4GemminiRocketConfig``
     - ``Int8Dim8GemminiRocketConfig``
   * - Gemmini config object
     - ``GemminiCustomConfigs.int8Dim4Config``
     - ``GemminiCustomConfigs.int8Dim8Config``
   * - derived from
     - ``GemminiConfigs.defaultConfig``
     - ``GemminiConfigs.defaultConfig``
   * - delta from it
     - ``meshRows`` = ``meshColumns`` = 4, ``tileRows`` = ``tileColumns`` = 1,
       ``has_training_convs`` = false
     - ``meshRows`` = ``meshColumns`` = 8, ``tileRows`` = ``tileColumns`` = 1,
       ``has_training_convs`` = false
   * - ``inputType`` / ``accType``
     - ``SInt(8.W)`` / ``SInt(32.W)``
     - ``SInt(8.W)`` / ``SInt(32.W)``
   * - ``spatialArrayOutputType``
     - ``SInt(20.W)``
     - ``SInt(20.W)``
   * - scratchpad
     - 256 KiB, 4 banks, ``BANK_ROWS`` 16 384
     - 256 KiB, 4 banks, ``BANK_ROWS`` 8 192
   * - accumulator
     - 64 KiB, ``ACC_ROWS`` 4 096
     - 64 KiB, ``ACC_ROWS`` 2 048
   * - ``dataflow``
     - ``Dataflow.BOTH`` (benchmarks pass ``WS``)
     - ``Dataflow.BOTH`` (benchmarks pass ``WS``)
   * - pipeline depths
     - ``spad_read_delay`` 4, ``tile_latency`` 0, ``mesh_output_delay`` 1,
       ``acc_latency`` 2 --- **all at Gemmini's defaults, unexercised**
     - same
   * - SoC
     - ``WithNHugeCores(1)``, ``WithSystemBusWidth(128)``,
       ``AbstractConfig``; 32 KiB L1 D$
     - same
   * - memory model
     - ``WithBlackBoxSimMem(additionalLatency = 0)``
     - same
   * - simulator
     - Verilator 5.051, ``simulator-chipyard.harness-<config>``
     - same
   * - harness program
     - ``gemmini/allo_bare_steady.c``, ``-DMAXDIM=64``
     - same source, ``DIM``-guarded shape set
   * - ``loop_ws`` calls per window
     - **1**, printed at runtime for every shape
     - **1**, printed at runtime for every shape

**Pins**: chipyard ``e0207441``, ``generators/gemmini`` ``25809f7``,
``gemmini-rocc-tests`` ``1a1a1c6``. ``has_training_convs`` is read in exactly
one place in the whole generator (``LoopConv.scala:1358-1363``) and appears
nowhere in ``LoopMatmul.scala``, the mesh, the scratchpad or
``generateHeader``, so a GEMM provably cannot observe it. The complete
field-by-field delta --- all 5 written fields including the 2 that are no-ops,
all 55 inherited with where each is set, and all **28 derived** values, which
is where two builds can agree on the written config and still differ in
hardware --- is committed at ``gemmini/CONFIG_DELTA.txt``.

.. warning::

   **Elaboration rewrites** ``gemmini-rocc-tests/include/gemmini_params.h``
   **in place** (``Controller.scala:30``), so building any config destroys the
   previous one's header and every C binary built afterwards silently targets
   the wrong hardware. Both headers are therefore committed here as
   ``gemmini/gemmini_params.dim4.h`` and ``gemmini_params.dim8.h``, the DIM=4
   one is restored as the tree's baseline after any DIM=8 run, and every run
   log is checked for the ``GEMMINI DIM=`` boot banner. One trap found this
   way: a stale DIM=4 binary run against the DIM=8 simulator hit Gemmini's own
   ``"A single mvin instruction must load more than 0 bytes"`` assert, which is
   the only reason it did not pass silently.

   A second trap for anyone building DIM=16: ``gemmini-rocc-tests`` HEAD ships
   a committed header with ``DIM`` 16 / ``BANK_ROWS`` 4096 / ``ACC_ROWS`` 1024
   but **no** ``ACC_READ_FULL_WIDTH``, while ``defaultConfig`` sets
   ``acc_read_full_width = true`` and ``generateHeader`` emits that macro. That
   committed header was generated by some leaner config. A DIM=16 build that
   reuses it instead of regenerating will disagree with a regenerated one for a
   reason that is invisible in the Scala.


Why both sides run at MAXDIM=64
-------------------------------

MAXDIM is the DRAM row stride of every operand **on both machines**:
``allo_bare5.c`` passes it as the ``loop_ws`` stride and TinyTPU-isa's
``dma_ld`` addresses ``A[row * MAXDIM + col]``. So the same logical shape costs
differently on a build with a bigger MAXDIM, and a comparison is matched only
if the two sides use the same one. Both columns below are MAXDIM=64. The
effect is real and it is larger for us: at 16x16x16, Gemmini goes 593 -> 685
(+15.5%) and we go 686 -> 879 (+28%) --- diagnosed in
:ref:`benchmarks-diagnosis`.

The exact program each machine runs
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * -
     - TinyTPU-isa
     - Gemmini
   * - program
     - ``isa_dsl.gemm_program(M, K, N)``: a 13-instruction looped GEMM in the
       accelerator's own ISA, assembled into ``imem`` in DRAM. One RTL build
       runs every shape; the shape is instruction data.
     - 5 ``config`` RoCC instructions, then **one** hardware ``gemmini_loop_ws``
       covering the whole shape, then ``fence``
       (``gemmini/allo_bare_steady.c``). One elaboration runs every shape.
   * - window opens
     - ``ap_start``
     - ``rdcycle``, after the CPU has refilled A and B
   * - window closes
     - ``ap_done``
     - ``rdcycle``, after ``gemmini_fence()`` retires
   * - inside
     - the whole program: the imem burst, both operand bursts, every
       ``dma_ld`` / ``mm`` / ``mvout``, the write-back of C
     - the 5 configs, the ``loop_ws`` expansion into mvin/preload/compute/mvout,
       the fence, and L1 coherence for the operands the CPU just dirtied
   * - outside
     - nothing --- there is no host driver
     - ``tiled_matmul_auto``'s tile search, padding arithmetic, argument
       marshalling: the **~395 cycles** of Rocket software, measured flat
       across a 16x range of work
   * - tiles per call
     - ``(N/T)(K/T)`` ``mm`` instructions from one program
     - **one** ``loop_ws``, proven at runtime by replicating the driver's own
       tiling search and printing ``loop_ws_calls=1`` for every shape

The asymmetries, stated once
----------------------------

#. **Our window has no host driver**; Gemmini's excludes its ~395-cycle one.
   The doc-level consequence is that Gemmini's column is a **lower bound** on
   its dispatch cost, since the real driver computes the ``loop_ws`` operands
   at runtime and that cost sits in the excluded 395. Conservative against us.
#. **Both harnesses have near-zero memory latency.** Neither number is a
   statement about a real memory system.
#. **Gemmini's on-chip memory is 64x ours** and, at these shapes, provably
   cycle-neutral (above).
#. ``ex_accumulate``: ``allo_bare5.c`` hardcoded ``true`` where the driver
   computes ``false`` for a no-bias single-tile matmul. Both were measured at
   every shape (``BARE`` / ``BAREF`` in ``allo_bare_steady.c``). The
   difference is **within trial-to-trial noise** --- at 64x64x64, 20375/20377
   against 20384/20350 --- so the published numbers stand. Recorded because it
   sat inside all five of them and had not been checked.

T=4 vs Gemmini DIM=4, both at MAXDIM=64
---------------------------------------

Ours: Vitis cosim, bit-exact at every shape, deterministic (one run suffices,
verified). Gemmini: Verilator, **median of five trials**, every shape one
``loop_ws`` proven at runtime. Peak is 16 MAC/cycle on both.

.. list-table::
   :header-rows: 1
   :widths: 12 8 10 14 10 10 13 13

   * - shape
     - set
     - ours
     - Gemmini (median)
     - ours % peak
     - Gem % peak
     - **ours / Gem**
     - clears spread?
   * - 4x4x4
     - latency
     - 218
     - 208 +/- 25
     - 1.8%
     - 1.9%
     - 1.05x
     - **no (0.4x)**
   * - 8x8x8
     - latency
     - 357
     - 324 +/- 36
     - 9.0%
     - 9.9%
     - 1.10x
     - **no (0.9x)**
   * - 12x12x12
     - latency
     - 563
     - 458 +/- 17
     - 19.2%
     - 23.6%
     - 1.23x
     - yes (6.2x)
   * - 16x16x8
     - latency
     - 677
     - 527 +/- 44
     - 18.9%
     - 24.3%
     - 1.28x
     - yes (3.4x)
   * - 16x16x16
     - both
     - 879
     - 691 +/- 44
     - 29.1%
     - 37.0%
     - 1.27x
     - yes (4.3x)
   * - 32x32x32
     - steady
     - 3 752
     - 2 977 +/- 34
     - 54.6%
     - 68.8%
     - 1.26x
     - yes (23x)
   * - 48x48x48
     - steady
     - 10 289
     - 9 100 +/- 35
     - 67.2%
     - 75.9%
     - 1.13x
     - yes (34x)
   * - 64x64x64
     - steady
     - 22 123
     - 20 287 +/- 34
     - 74.1%
     - 80.8%
     - **1.09x**
     - yes (54x)

   * - 64x32x64
     - steady
     - 12 907
     - 11 175 +/- 18
     - 63.5%
     - 73.3%
     - 1.16x
     - yes (96x)
   * - 32x64x32
     - steady
     - 6 824
     - 5 570 +/- 147
     - 60.0%
     - 73.5%
     - 1.23x
     - yes (8.5x)

(``+/-`` is the full min-max spread over five trials, not a standard error.)

**Answer at this array size: we do not beat Gemmini at any shape, but the
deficit converges rather than persisting.** On the cubic sweep it goes 1.27x
(16) -> 1.26x (32) -> 1.13x (48) -> **1.09x (64)**, and in fraction-of-peak
terms the gap closes from 7.9 points at 16x16x16 to 6.7 at 64x64x64. It does
not invert.

That answers the honest open question directly: the 1.07-1.24x measured at the
five small shapes was **a statement about pipeline depth and issue overhead,
not about steady-state efficiency**, and at steady state the deficit shrinks
to about 9% without disappearing. Two caveats that belong with it: the two
smallest shapes do not clear the measurement noise at all, and the answer is
**different at the other matched array size** (next section).


T=8 vs Gemmini DIM=8, both at MAXDIM=64
---------------------------------------

The second matched point, and **it does not agree with the first**, which is
the whole reason for having two. Peak is 64 MAC/cycle on both sides.
``gemmini/allo_bare_steady.c`` at ``DIM=8``, Gemmini as **median of five
trials**, which drops 4x4x4 and 12x12x12 because they are not multiples of 8
--- the same ``runnable()`` rule our own side applies on T, so **both machines
drop the same shapes**. Our cycle column was re-measured on current ``main``
after the parametric ``DMA_WORDS`` refactor and is unchanged; see `Earlier
measurements and corrections`_.

.. list-table::
   :header-rows: 1
   :widths: 12 8 9 13 10 10 12 16

   * - shape
     - set
     - ours
     - Gemmini (median)
     - ours % peak
     - Gem % peak
     - **ours / Gem**
     - clears spread?
   * - 8x8x8
     - latency
     - 285
     - 305 +/- 17
     - 2.8%
     - 2.6%
     - 0.93x (ours faster)
     - no (1.2x)
   * - 16x16x8
     - latency
     - 424
     - 500 +/- 16
     - 7.5%
     - 6.4%
     - **0.85x (ours faster)**
     - **yes (4.8x)**
   * - 16x16x16
     - both
     - 493
     - 535 +/- 45
     - 13.0%
     - 12.0%
     - 0.92x (ours faster)
     - no (0.9x)
   * - 32x32x32
     - steady
     - 1 484
     - 1 280 +/- 18
     - 34.5%
     - 40.0%
     - 1.16x
     - yes (11x)
   * - 48x48x48
     - steady
     - 3 537
     - 3 028 +/- 35
     - 48.9%
     - 57.1%
     - 1.17x
     - yes (15x)
   * - 64x64x64
     - steady
     - 7 083
     - 6 033 +/- 35
     - 57.8%
     - 67.9%
     - **1.17x**
     - yes (30x)
   * - 64x32x64
     - steady
     - 4 523
     - 3 897 +/- 35
     - 45.3%
     - 52.6%
     - 1.16x
     - yes (18x)
   * - 32x64x32
     - steady
     - 2 508
     - 1 866 +/- 36
     - 40.8%
     - 54.8%
     - 1.34x
     - yes (18x)

**We beat Gemmini at one shape, supportably: 16x16x8, by 1.18x, clearing the
spread 4.8x.** At 8x8x8 and 16x16x16 the sign is also ours but the margins
(20 and 42 cycles) do not clear their spreads (17 and 45), so those two are
**level, not wins** --- and the earlier best-of-two reading of them as wins is
withdrawn. That is the first supportable win this design has over Gemmini at
any shape.

At steady state the deficit is a **flat 1.16-1.17x** with no convergence,
except 32x64x32 at 1.34x --- the one shape where a shallow K (32, four k-tiles
at T=8) leaves the weight chain least amortised.

.. note::

   **Settled at five trials.** The DIM=8 column above is the median of five,
   with the full min-max spread; the per-shape spread is 16-45 cycles, the
   same roughly-constant band measured at DIM=4. **One win is supportable,
   two are level.** [#bestoftwo]_

The reason we are ahead there rather than behind is the reason the latency set
exists: at T=8, 16x16x16 is only four tile-matmuls, so the machine with the
shorter pipeline is not penalised, and ours is shorter. 16x16x8 is the
clearest case because it is the shallowest of the three (two n-tiles), which
is why it is the one whose margin clears the noise.

At steady state the deficit is a **flat 1.17-1.18x** and does not converge,
where at T=4 it converged to 1.086x. Two matched points, two different
behaviours:

.. list-table::
   :header-rows: 1

   * -
     - latency shapes
     - 32^3
     - 48^3
     - 64^3
   * - T=4 / DIM=4
     - 1.06-1.30x slower
     - 1.25x
     - 1.13x
     - **1.086x**
   * - T=8 / DIM=8
     - **0.85x at 16x16x8** (a win); level at the other two
     - 1.16x
     - 1.17x
     - **1.17x**

**This is what a single matched point could not have told us**, and it is the
answer to "is the deficit a property of the design or of one configuration":
it is a property of *where the shape sits relative to the array*. At T=4,
64x64x64 is a steady-state shape (74.1% of peak) and we close to within 9%. At
T=8 the same shape is still in the ramp (57.8%) and we sit 18% behind, because
Gemmini's much deeper buffering fills an 8x8 array better than our 4 KiB
operand file does at this problem size. A T=8 steady-state set would have to
start around 128 on a side --- which is past the encoding ceiling, and is
therefore the concrete argument for widening the fields.

.. warning::

   **The uncertainty on every difference above is Gemmini's, not ours.**
   Gemmini is a whole SoC --- Rocket, an L1 cache, a scratchpad still holding
   the previous call's state --- so the same binary on the same hardware does
   not give the same count twice: over five trials the spread is **16-45
   cycles**, roughly constant in absolute terms, which is up to 12% of the
   total at the smallest shapes and under 0.2% at 64x64x64. Our side is a
   deterministic simulation of a fixed design on fixed inputs and reproduces
   to the cycle across independent runs.

   So **a claimed difference smaller than Gemmini's spread is unsupportable.**
   At T=4 that rules out 4x4x4 and 8x8x8; at T=8 it rules out 8x8x8 and
   16x16x16. In each case the **sign** is still informative --- it is the same
   across all ten shapes at T=4, and the same across all three latency shapes
   at T=8 --- but the magnitude at an individual sub-spread shape is not.
   Median and spread per shape are in :ref:`benchmarks-spread`.

   The two Gemmini benchmark files are also **not interchangeable**:
   ``allo_bare_steady.c`` reads 2-14 cycles below ``allo_bare5.c`` at the same
   shapes on the same hardware, because each measurement is sensitive to the
   preceding call's cache and scratchpad state and the two files have
   different call sequences. Every figure on this page names its file.


.. _benchmarks-latency-grid:

Does the ranking survive the memory-latency range?
==================================================

.. important::

   **The point of this grid is not which variant is faster. It is whether the
   knob chose the answer.** Every cycle count on this page is at
   ``TPU_AXI_LATENCY=0``, and Vitis cosim has no DRAM model, so 0 is a value
   we *chose*. A variant whose whole benefit is wider DMA bursts is exactly
   the kind whose advantage can grow or vanish with memory latency, so a win
   measured at 0 is a win at one arbitrary point of a knob.

   This is not hypothetical. MiniTPU's simulator ranked GEMM templates
   **backwards** against a board A/B at its inferred 92-cycle value --- a 45%
   simulated cut measured 3.6% on hardware --- and backwards again at 0. Two
   wrong values, two wrong rankings, and the simulator was internally
   consistent and confident both times. They found it only by running the A/B
   on two boards.

   **At zero latency the burst-widened candidate really is faster.** The
   measurement is sound. It is the inference from it to "this design is
   better" that the grid tests, and that is a much slipperier failure than a
   bad measurement.

The variant under test is a **parametric** burst width,
``microarch_isa.DMA_WORDS`` (``TPU_DMA_WIDEN=1``), not a patch: at 1 it is the
shipped loop, one packed word per iteration; at 16 each iteration reads a whole
64-byte beat, which is what ``align_value(64)`` lets Vitis widen the port to.
It is bit-exact on ``bench_isa`` and ``stress_isa`` (640/640) at both settings,
and **the parametric refactor is cycle-neutral at DMA_WORDS=1**: the shipped
path reproduces 10 289 and 22 123 exactly.

Latency in **ns as well as cycles**, because cycles say which simulator
settings a variant is good for and seconds say which real *memory systems* it
is good for. MiniTPU's inversion sat between 213 and 490 ns, and it was the
nanosecond figure that let them place their board inside that window. Our own
clock is not fixed either --- variants have estimated 2.431 and 3.782 ns ---
so a cycle-valued threshold moves when the frequency does and a time-valued
one does not. Both builds here estimate **2.431 ns**, so at least the two
columns share a clock.

For reference, MiniTPU's own fitted figure is **213 ns** (about 40 cycles at
187.5 MHz), which is **88 cycles at 2.431 ns** --- that is where the 88 comes
from, and it is carried as a time rather than as their cycle count so their
clock is not imported with it.

.. list-table:: 48x48x48 and 64x64x64, one csynth per grid point
   :header-rows: 1
   :widths: 11 10 12 12 10 12 12 10

   * - ``m_axi_latency``
     - = ns @ 2.431
     - shipped 48^3
     - widened 48^3
     - **delta**
     - shipped 64^3
     - widened 64^3
     - **delta**
   * - 0
     - 0
     - 10 289
     - 9 569
     - **-720**
     - 22 123
     - 21 163
     - **-960**
   * - 16
     - 39
     - 9 823
     - 9 103
     - **-720**
     - 21 175
     - 20 215
     - **-960**
   * - 64
     - 156
     - 10 430
     - 9 710
     - **-720**
     - 22 666
     - 21 706
     - **-960**
   * - 88
     - 214
     - 10 838
     - 10 118
     - **-720**
     - 23 554
     - 22 594
     - **-960**
   * - 100
     - 243
     - 11 042
     - 10 322
     - **-720**
     - 23 998
     - 23 038
     - **-960**

**Outcome 1, and more strongly than the question anticipated: the ranking does
not merely hold across the whole 0 to 243 ns range --- the advantage is
EXACTLY constant.** -720 cycles at 48x48x48 and -960 at 64x64x64 at **all
five** latency points, not approximately: the last two points were *predicted*
from the first three and came back to the cycle. **There is no inversion anywhere in the range, and no
sensitivity at all.** So the knob did not choose this design decision, and
what is left is purely the BRAM cost.

The invariance is not luck, and it has a mechanism, which is worth having
because an unexplained constant would be more suspicious than a varying one.
The widening removes loop *iterations* from the burst: at 48x48x48 the two
operands span 48 x WPR = 768 packed words each, so 1 536 iterations at
``DMA_WORDS=1`` become 96 at 16, a saving of 1 440; at 64x64x64, 2 048 become
128, a saving of 1 920. **The measured saving is exactly half of each**
(720 and 960), i.e. the A and B bursts overlap each other so precisely one of
the two is ever on the critical path. That is a fixed structural saving in the
burst prologue, which is why no amount of modelled latency moves it --- the
latency shifts *when* the prologue runs, not how many iterations it has.

.. warning::

   **The knob is NON-MONOTONIC, and that is the grid's other finding.**
   Latency 16 is *faster* than latency 0 --- 9 823 against 10 289 at
   48x48x48, and 21 175 against 22 123 at 64x64x64 --- for **both** variants.
   Cycles then rise monotonically from 16 through 100.

   A real memory system cannot get faster by being slower, so this is direct
   evidence that ``config_interface -m_axi_latency`` is a **scheduling
   directive, not a latency model**: telling Vitis to expect 16 cycles makes
   it schedule the burst more aggressively, and cosim's actual memory still
   answers immediately, so the design gets faster. It follows that **no row of
   this grid may be read as "what happens with a memory of that latency"**.
   What the grid legitimately shows is that the *ranking* of the two variants
   is invariant to the directive across its whole useful range --- which is
   the question it was built for, and all it answers.

What the widening costs, at every latency (it does not vary with the knob):

.. list-table::
   :header-rows: 1

   * - variant
     - FF
     - LUT
     - BRAM
     - DSP
     - estimated clock
   * - shipped
     - 17 488
     - 26 554
     - 52
     - 14
     - 2.431 ns
   * - burst-widened
     - 24 001
     - 31 396
     - **116**
     - 14
     - 2.431 ns (dual-ported form; see the warning below --- the shipped
       variant of this experiment is banked)
   * - delta
     - +6 513 (+37%)
     - +4 842 (+18%)
     - **+64 (+123%)**
     - 0
     - **unchanged**

The clock not moving is worth stating: widening the DMA datapath **does not
lengthen the critical path**, so the comparison is a pure cycles-for-area
trade and the two columns can be compared in cycles without converting. The
BRAM more than doubles, which is the real price.

.. warning::

   **The +123% BRAM and an ASIC elaboration failure are the same fact seen
   from two substrates, and that changes what the decision is.**

   The widened variant does not synthesise to standard cells. DC rejects it in
   two minutes --- ``ELAB-366: Net 'ram[0][31]' ... driven by more than one
   source``, across all 32 bits --- because Vitis satisfies the widened loop's
   ``DMA_WORDS`` writes per iteration by emitting ``rbA`` as a **true
   dual-write-port RAM**: two ``always @(posedge clk)`` blocks driving one
   array, while still naming the module ``_1R1W``. Every RAM in all four
   exported variants was checked and **it is the only two-write-port memory
   anywhere; it exists only in the widened build.**

   An FPGA block RAM has two independent write ports, so filling both per
   cycle is free. **Standard cells have no such primitive**, and with memories
   mapped to registers two unconditioned writers of one array is a genuine
   multi-driver. DC is right to refuse it, and the synthesis side is right to
   report it as not synthesisable rather than bodging a flip-flop dual port:
   that would be a number describing hardware nobody would build.

   So landing the widening is **not** "spend BRAM to buy cycles". It is
   "**commit to a dual-write-port memory**", which is a different commitment
   with different consequences for anyone carrying this design to another
   substrate. The BRAM number alone does not say that, which is why it is
   written here beside it.

   **Measured, and the widening survives without the primitive.** The buffers
   are now **cyclically banked by** ``DMA_WORDS``, so write ``w`` always lands
   in bank ``w`` and every bank has exactly one writer --- the same widening
   expressed in a way both substrates can build. The result is the strongest
   of the three possible outcomes:

   .. list-table::
      :header-rows: 1

      * - build
        - 48^3
        - 64^3
        - rbA write ports
        - FF
        - LUT
        - BRAM
      * - shipped (``DMA_WORDS=1``)
        - 10 289
        - 22 123
        - 1
        - 17 488
        - 26 554
        - 52
      * - widened, dual-ported
        - 9 569
        - 21 163
        - **2 (DC rejects)**
        - 24 001
        - 31 396
        - 116
      * - **widened, banked**
        - **9 569**
        - **21 163**
        - **1**
        - 25 026
        - 33 799
        - **100**

   **Identical cycles, to the cycle.** Banking keeps the entire -720 / -960,
   so **the gain was the widening and not the second write port** --- the
   optimisation is real and it is portable. Audited across every RAM module in
   the build: all seven have one write port, where the rejected export's
   ``rbA`` had two. The ``ELAB-366`` cause is removed rather than worked
   around.

   It also costs *less* block RAM than the dual-ported form (+92% over the
   shipped control rather than +123%), trading that for +43% FF and +27% LUT.
   So the decision is back on the table on its merits, and what it now reads
   as is: **-720 / -960 cycles for +43% FF, +27% LUT, +92% BRAM, at an
   unchanged clock, on a design that synthesises to standard cells.**

(The shipped row reads 52 BRAM here against 48 in the MAXDIM table. The
parametric burst buffers carry ``DMA_WORDS`` words of rounding headroom in
**both** variants, so the two differ only in the loop and not in the memory
they address --- which is what makes the cycle columns comparable. The price
is 4 BRAM on the shipped path against the pre-parametric build, at **zero
cycles**: 10 289 and 22 123 are the same numbers the pre-refactor build
produced.)


.. _benchmarks-spread:

Median and spread: which side carries the uncertainty
=====================================================

**Only one side has any.** This is not symmetric and it should not be reported
as if it were.

* **Ours is deterministic, and each new configuration was checked once.**
  Vitis cosim is a simulation of a fixed design on fixed inputs. The check is
  not a sample: a non-deterministic cosim would invalidate every comparison
  drawn from it, so it is worth finding now rather than in a disagreement with
  someone else's table later. **If a pair ever disagrees the right response is
  to stop and report it, not to average or re-run.** None did:

  .. list-table::
     :header-rows: 1

     * - configuration
       - run 1
       - run 2
       - agree?
     * - T=4 MAXDIM=64 shipped (48^3 / 64^3)
       - 10 289 / 22 123
       - 10 289 / 22 123
       - **to the cycle**
     * - burst-widened, latency 0 (48^3 / 64^3)
       - 9 569 / 21 163
       - 9 569 / 21 163
       - **to the cycle**
     * - T=8 MAXDIM=64 (16^3 / 64^3)
       - 493 / 7 083
       - 493 / 7 083
       - **to the cycle**

  Each pair is two separate processes, two separate ``csynth_design``
  invocations and two separate project directories --- not a re-read of one
  report.
* **Gemmini's is not.** It is a full SoC: a Rocket core, a 32 KiB L1 D-cache,
  and a scratchpad still holding the previous call's state. The same binary on
  the same hardware gives different counts, and the measured spread reaches
  **20 cycles at 16x16x8** against a ~520-cycle total.

Gemmini DIM=4 at MAXDIM=64, **five trials per shape**
(``allo_bare_steady.c``, ``BARE``):

.. list-table::
   :header-rows: 1
   :widths: 13 9 9 9 9 11 11 15

   * - shape
     - n
     - median
     - min
     - max
     - **spread**
     - % of total
     - deficit / spread
   * - 4x4x4
     - 5
     - 208
     - 208
     - 233
     - **25**
     - **12.0%**
     - 10 / 25 = **0.4x**
   * - 8x8x8
     - 5
     - 324
     - 324
     - 360
     - **36**
     - **11.1%**
     - 33 / 36 = **0.9x**
   * - 12x12x12
     - 5
     - 458
     - 458
     - 475
     - 17
     - 3.7%
     - 105 / 17 = 6.2x
   * - 16x16x8
     - 5
     - 527
     - 517
     - 561
     - 44
     - 8.3%
     - 150 / 44 = 3.4x
   * - 16x16x16
     - 5
     - 691
     - 681
     - 725
     - 44
     - 6.4%
     - 188 / 44 = 4.3x
   * - 32x32x32
     - 5
     - 2 977
     - 2 977
     - 3 011
     - 34
     - 1.1%
     - 775 / 34 = 23x
   * - 48x48x48
     - 5
     - 9 100
     - 9 100
     - 9 135
     - 35
     - 0.4%
     - 1 189 / 35 = 34x
   * - 64x64x64
     - 5
     - 20 287
     - 20 287
     - 20 321
     - 34
     - 0.17%
     - 1 836 / 34 = **54x**
   * - 64x32x64
     - 5
     - 11 175
     - 11 165
     - 11 183
     - 18
     - 0.16%
     - 1 732 / 18 = 96x
   * - 32x64x32
     - 5
     - 5 570
     - 5 474
     - 5 621
     - **147**
     - 2.6%
     - 1 254 / 147 = 8.5x

The spread is mostly **constant in absolute terms** (17-44 cycles) rather than
proportional, which is what a fixed-size cache and coherence effect looks like
--- so it matters enormously at 4x4x4 and not at all at 64x64x64. **32x64x32
is the exception at 147 cycles**, and it is the only steady-state shape whose
spread is not small; it still clears its own deficit by 8.5x, but a future
comparison at that shape should carry more than five trials.

.. warning::

   **Two shapes are inside the noise and must not be quoted as differences.**
   At 4x4x4 the deficit is 10 cycles against a 25-cycle spread (0.4x) and at
   8x8x8 it is 33 against 36 (0.9x). Five trials were needed to see this: the
   two-trial estimates understated the spread at 8x8x8 by 12x (3 against 36).
   **No single-trial number at 4x4x4 or 8x8x8 should be quoted again by
   either side.**

   Everything from 12x12x12 up clears the bar, and every steady-state
   conclusion clears it by one to two orders of magnitude: the 1 836-cycle
   deficit at 64x64x64 is **54x** its spread, and the -960-cycle burst saving
   is 28x it.

Consequence, stated once and applied everywhere: **the uncertainty on a
cross-machine difference is Gemmini's alone**, and a claimed difference
smaller than it is unsupportable. The earlier reading of that same bar is kept
in `Earlier measurements and corrections`_.

.. note::

   And the trap on the other side, which matters more for how all of this
   should be read: **a number can be perfectly deterministic and still wrong
   about the hardware.** Our cosim contains no memory system at all and
   ``TPU_AXI_LATENCY`` is a value we pick; MiniTPU's simulator contains a
   *fitted* one, and the 92-cycle version of that fit ranked GEMM templates
   backwards against the board --- a 45% simulated improvement that measured
   3.6% on silicon. Reproducibility is a property of the measurement.
   Agreement with hardware is a separate claim needing separate evidence.
   **We have the first and not the second**, on either side of this
   comparison.


.. _benchmarks-diagnosis:

Where the remaining cycles go
=============================

Three terms, in order of size at 64x64x64.

.. important::

   **The largest term is now measured, not estimated, and it accounts for most
   of the remaining deficit at T=4.** The burst-widened variant
   (:ref:`benchmarks-latency-grid`) is a bit-exact build that differs from the
   shipped one *only* in the burst loop, so the difference between them prices
   the burst prologue exactly:

   .. list-table::
      :header-rows: 1

      * - shape
        - shipped
        - Gemmini DIM=4
        - deficit
        - burst saves
        - **% of deficit**
        - widened vs Gemmini
      * - 48x48x48
        - 10 289
        - 9 102
        - 1 187
        - **720**
        - **61%**
        - 9 569 -> **1.051x**
      * - 64x64x64
        - 22 123
        - 20 375
        - 1 748
        - **960**
        - **55%**
        - 21 163 -> **1.039x**

   So **landing the burst widening would take the T=4 steady-state deficit
   from 1.086x to 1.039x at 64x64x64**, and it does so at every modelled
   latency. That is the single highest-value change available, it is already
   parametric and verified, and its price is +123% BRAM. The remaining
   ~790 cycles at 64x64x64 are terms 2 and 3 below.

**1. The operand burst reads whole DRAM rows (the MAXDIM-stride term).**
``dma_ld``'s burst loop runs ``na * WPR`` words, where ``WPR = MAXDIM / T`` is
the packed words in one DRAM row --- so it reads the **full MAXDIM-wide row**
whichever columns the program names. At MAXDIM=64 a 4x4x4 GEMM bursts
4 x 16 = 64 words to use 4, a 16x waste, and this is the whole of the
16x16x16 regression from 686 (MAXDIM=16) to 879 (MAXDIM=64): 193 cycles, against
Gemmini's 92 for the same stride change. At 64x64x64 the waste is zero,
because the full row *is* what the program needs --- which is why the deficit
converges. Fixing it means giving ``dma_ld`` a column extent so the burst
covers the named blocks only; it is the single largest item and it is worth
more at small shapes than large ones.

**2. The residual steady-state gap, ~5 700 cycles at 64x64x64** (22 123 against
an ideal 16 384; Gemmini is at 20 375 against the same ideal, so ~4 000 of it
is not ours specifically). Per the timeline attribution in
``impact/results/`` and ``chia_agent/evidence/``, the three components are
operand staging ahead of the first MAC, the serial DMA before the first
weight, and the drain. None has been built. The method is
``impact/profile.sh``: re-run cosim with ``-enable_dataflow_profiling``, re-run
the xsim snapshot so the monitor's CSVs survive, then ``impact/analyze_df.py``
for per-process timelines. At 64x64x64 this is the measurement to make next,
because every previous timeline was taken at 16x16x16, where term 1 dominates
and term 2 is invisible.

**3. Issue overhead, ~20 cycles per dynamic instruction** (fixed 557,
marginal 20.1, measured at MAXDIM=16). At 64x64x64 the program issues 320
dynamic instructions, so this is ~6 400 cycles of the 22 123 --- and it is the
term the loop nest already minimises: the *static* program is 14 instructions
at every shape, so nothing here is program size. It is the per-instruction
dispatch through the sequencer.

What would have to change, in order of **measured** value:

#. **Land the burst widening** (``TPU_DMA_WIDEN=1``, already parametric,
   bit-exact, latency-invariant). Worth 720 and 960 cycles at 48^3 and 64^3,
   i.e. 55-61% of the whole T=4 steady-state deficit, for +123% BRAM. This is
   the decision the owner has in front of them and the grid says the memory
   knob does not affect it.
#. **Give** ``dma_ld`` **a column extent** so the burst covers the named column
   blocks instead of the whole DRAM row. This is the *other* half of term 1 and
   it attacks it from the opposite side: widening makes the wasted reads
   cheaper, an extent stops reading them. It is worth most at the small
   shapes, where the waste is 16x, and nothing at 64^3, where the full row is
   what the program needs.
#. **Profile at 64x64x64 before touching term 2.** Every existing timeline was
   taken at 16x16x16, where term 1 dominates and term 2 is invisible, so the
   operand-staging/DMA/drain split in ``impact/results/`` is an estimate made
   in the wrong regime.
#. **For T=8, widen the encoding fields.** T=8's deficit does not converge
   because 64x64x64 is still in its ramp (57.8% of peak); reaching steady
   state needs ~128 on a side, which is past both ceilings in
   :ref:`benchmarks-raising-maxdim`. This is the first time a *measurement*
   has argued for widening them rather than a hypothetical.


.. _benchmarks-workload:

The workload set: a GPT-2 projection (placeholder)
==================================================

**Nothing here is measured.** The entry exists so that the shape and its cost
are written down rather than estimated in conversation.

**The shape.** GPT-2 small has :math:`d_\text{model} = 768`. The attention
**output projection** at sequence length 128 is

.. math::

   [128, 768] \times [768, 768] \;\longrightarrow\; [128, 768],
   \qquad 75\,497\,472 \text{ MACs}

i.e. **M=128, K=768, N=768**. (The fused QKV projection is the same M and K
with N=2304, three times the work; the MLP is 768->3072->768. The output
projection is chosen because it is the smallest square one.)

**Which dimension binds first, and at what value.** "We cannot run
128x768x768" is true and useless; what a reader needs is the binding dimension
and its number, because that is what says which field to widen. Derived from
the encoding rules, not estimated:

.. list-table:: Ceilings, largest legal value (MAXDIM must be a multiple of T)
   :header-rows: 1
   :widths: 30 22 16 16 16

   * - limit
     - what it bounds
     - T=4
     - T=8
     - T=16
   * - address field, 11 usable bits
       (:math:`\text{MAXDIM}^2/T \le 2047`)
     - **MAXDIM**, hence M, K and N together
     - **88**
     - **120**
     - **176**
   * - ``nr``, 7 usable bits (``MAXROWS``)
     - any single instruction's row count, so M and K
     - 127
     - 127
     - 127
   * - header count, 15 usable bits
     - ``accu`` iterations,
       :math:`(N/T)(K/T)M + (N/T)M`
     - shape-dependent
     - shape-dependent
     - shape-dependent

So **the address field binds first, through MAXDIM, at 88 (T=4) and 120
(T=8)** --- and note that ``nr``'s 127 is *not* a function of T, so it becomes
the wall as soon as the array is wide enough to push MAXDIM past it.

Against the shapes another team proposed, this is precise rather than
approximate:

* **32 x 512 x 128** (their recommended single shape). K=512 needs
  MAXDIM >= 512: we are short **5.8x** at T=4 and **4.3x** at T=8 on the
  address field, **4.0x** on ``nr``, and 4.0x / 1.0x on the header count.
* **K-sweep at M=32, N=128.** Raising MAXDIM to match K, the first failure is
  at **K=92** (T=4) and **K=128** (T=8), in both cases the address field;
  last good K is 88 and 120.
* **Tall sweep at K=N=128.** Blocked for *every* M, because N=128 alone
  exceeds the MAXDIM ceiling --- at T=8 by **eight** (120 against 128). And
  M=128 additionally exceeds ``nr`` by **one** (127 against 128).

Those last two are worth stating as margins rather than as failures: at T=8 we
miss their tall sweep by 8 in one dimension and by 1 in another.

**What our build would need to run the GPT-2 shape.** Not simply a bigger
MAXDIM --- 768 is past the address ceiling by 8.7x at T=4. The deeper blocker
is that the region's operands are declared ``int8[MAXDIM * MAXDIM]`` and
addressed ``row * MAXDIM + col``, so the machine cannot *address* a 128x768
matrix at all, whatever its on-chip capacity. Three changes, in dependency
order:

#. **A runtime base and row stride on** ``dma_ld`` **and** ``mvout``, so a
   MAXDIM=64 on-chip tile is a *window* into a larger DRAM matrix rather than
   the whole of it. Today both are MAXDIM-relative. This is the change that
   actually unblocks the entry.
#. **A deeper loop stack.** The GEMM already uses 2 levels; an outer tile nest
   over DRAM tiles (m, n, k) needs 3 more, so ``LOOP_DEPTH`` must grow from 4
   to at least 5 --- or the outer tiling must be issued by a host as repeated
   region invocations, which would put a driver inside our measurement window
   for the first time and change what the window means.
#. ``AGU_TERMS`` **may need to grow from 3**: a weight address already spends
   two terms (``B_SP + nb*MAXDIM + kb*T``), and an outer k-tile would want a
   third on the same field.

**What it would cost if it ran.** Projected from each build's *own* measured
fraction of peak at 64x64x64, which is the largest shape either can run:

.. list-table::
   :header-rows: 1

   * - build
     - peak MAC/cyc
     - measured % of peak at 64^3
     - projected cycles
     - at 411 MHz
   * - T=4, MAXDIM=64
     - 16
     - 74.1%
     - 6.37 M
     - 15.5 ms
   * - T=8, MAXDIM=64
     - 64
     - 57.8%
     - 2.04 M
     - 5.0 ms

Both are **projections from one point**, and they assume the windowed DMA that
does not exist yet would cost nothing --- which is exactly the assumption this
entry exists to stop us making. The T=8 figure is the weaker of the two, since
57.8% is a ramp point rather than a steady-state one.

.. important::

   **The asymmetry this entry exposes is the largest one on the page, and it
   is not in our favour: Gemmini can run this workload today and we cannot.**
   ``tiled_matmul_auto`` takes arbitrary M, K, N with runtime strides and
   splits them across as many ``loop_ws`` calls as its scratchpad needs --- at
   128x768x768 its own tiling search would use multiple calls, and that is
   ordinary operation for it, not an extension. Our machine cannot address the
   matrix.

   That is a real capability gap and no cycle count offsets it. It is also the
   reason this set is a placeholder rather than a measurement: a benchmark set
   that only contains shapes we can run is a set chosen to flatter us, and
   writing the gap down is the alternative to quietly omitting it.


.. _benchmarks-reproduce:

Reproducing
===========

.. code-block:: bash

   source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
   export PYTHONPATH=$PWD
   cd examples/accelerator/tinytpu_vitis

   # functional, seconds -- the gates every configuration must pass
   TPU_SET=all python bench_isa.py                 # ALL EXACT
   python stress_isa.py                            # STRESS OK
   TPU_T=8 TPU_MAXDIM=64 python stress_isa.py      # the second array size

   # resources: one csynth per configuration, project deleted, report kept
   python csynth_sweep.py 4:16 4:32 4:48 4:64
   python csynth_sweep.py --reparse                # re-read, no Vitis

   # cycles: one csynth, one cosim per shape, on one RTL build
   TPU_MAXDIM=64 TPU_PRJ=$PWD/cosim_T4_64.prj \
     TPU_SHAPES=4x4x4,8x8x8,12x12x12,16x16x8,16x16x16,32x32x32,48x48x48,64x64x64,64x32x64,32x64x32 \
     python cosim.py
   # the second array size
   TPU_T=8 TPU_MAXDIM=64 TPU_PRJ=$PWD/cosim_T8_64.prj \
     TPU_SHAPES=8x8x8,16x16x8,16x16x16,32x32x32,48x48x48,64x64x64,64x32x64,32x64x32 \
     python cosim.py

   # the memory-latency grid (one csynth per point; ~30 min each)
   python latency_grid.py                    # shipped, 0/16/64/88/100
   python latency_grid.py --widen            # the burst candidate

   # RTL for the ASIC synthesis handoff, into rtl_handoff/<config>/
   python export_rtl.py

**Every project is created inside this worktree via** ``TPU_PRJ`` **and deleted
as soon as its numbers are read.** ``csynth_sweep.py`` and ``latency_grid.py``
do this themselves and copy the report out first, so a parsing bug costs a
re-parse rather than a re-synthesis; ``latency_grid.py`` also refuses a point
outright if ``/home`` has less than 12 GB free rather than risking a full
filesystem shared with several other agents.

Gemmini, in a **separate shell** (``env.sh`` replaces the ``allo`` conda env):

.. code-block:: bash

   R=/home/sk3463/chipyard/generators/gemmini/software/gemmini-rocc-tests
   cp examples/accelerator/tinytpu_vitis/gemmini/allo_bare_steady.c $R/bareMetalC/
   cd /home/sk3463/chipyard && source env.sh
   make -C $R/build/bareMetalC -f $R/bareMetalC/Makefile \
        abs_top_srcdir=$R XLEN=64 src_dir=$R/bareMetalC allo_bare_steady-baremetal
   cd sims/verilator
   ./simulator-chipyard.harness-Int8Dim4GemminiRocketConfig +permissive +permissive-off \
     $R/build/bareMetalC/allo_bare_steady-baremetal

A shape change is C-only. A **config** change (DIM, dtype, capacity) needs
``make CONFIG=<X> -j16`` in ``sims/verilator``, and that **rewrites**
``gemmini-rocc-tests/include/gemmini_params.h`` in place as a side effect of
elaboration, so the DIM=4 header must be snapshotted first and the C rebuilt
immediately afterwards. Check the boot banner says the DIM you expect. The
config patches for both matched points are committed at
``examples/accelerator/tinytpu_vitis/gemmini/``.

.. note::

   **The published 171/261/417/483/685 are a MAXDIM=16 measurement**, and the
   shipped default is now 64, so ``reproduce.sh`` pins ``TPU_MAXDIM=16``
   explicitly --- it exists to reproduce those numbers and would otherwise
   measure 218/357/563/677/879 and report a difference that is the stride
   change, not a regression.

   **That row itself moved by one cycle when this work landed**, uniformly at
   all five shapes: 172/262/418/484/686 became 171/261/417/483/685. See
   :ref:`benchmarks-one-cycle`. Both sets are on this page and neither
   supersedes the other: the MAXDIM=16 five are the latency benchmark's
   provenance, the MAXDIM=64 sweep is the matched comparison.

.. seealso::

   :doc:`tinytpu_isa` for the design, :doc:`gemmini_comparison` for the
   latency-set comparison at MAXDIM=16 and the gap attribution,
   :doc:`minitpu` for MiniTPU.


Earlier measurements and corrections
====================================

Moved out of the sections above: superseded figures, readings that were
withdrawn, and the accounts of how particular numbers moved.

Why naming the window mattered
------------------------------

That accounting turned out to be worth more than fairness. MiniTPU's
per-launch host cost was volunteered by its own side so the comparison would
be honest, and putting ~24 750 cycles beside Gemmini's ~395 of driver is what
made the number look absurd rather than normal --- both of the subsequent wins
that doubled their board throughput came out of it. **Naming the window is not
only a reporting discipline; it is where the optimisations were hiding.**

.. _benchmarks-one-cycle:

The published row moved by one cycle, and why
---------------------------------------------

When this work landed, ``reproduce.sh`` printed ``DIFFERS``:

.. code-block:: text

   expected: 4x4x4=172  8x8x8=262  12x12x12=418  16x16x8=484  16x16x16=686
   got:      4x4x4=171  8x8x8=261  12x12x12=417  16x16x8=483  16x16x16=685

**Exactly one cycle faster at every shape**, every testbench bit-exact, and
reproduced independently by two separate runs. A delta that does not scale
with the work is a **fixed-cost** change, so it cannot be the burst loop's
per-iteration behaviour.

**The cause, measured rather than inferred, and isolated to one variable.**
Rebuilding on current ``main`` with the memory sizes the design used to carry
--- ``TPU_SPAD=512 TPU_NVR=256 TPU_NAR=128`` and *nothing else changed*, so
the derived-size expression, the two ceiling assertions, the test-window floor
and the parametric burst loop are all still present --- returns **every one of
the five published numbers exactly**:

.. code-block:: text

   TPU_MAXDIM=16 TPU_SPAD=512 TPU_NVR=256 TPU_NAR=128
     4x4x4 172   8x8x8 262   12x12x12 418   16x16x8 484   16x16x16 686

So **the memory sizing accounts for the entire shift and nothing else in that
work changed cycles at all** --- in particular the parametric burst loop is
cycle-neutral at ``DMA_WORDS=1``, which is the same thing it was shown to be
at MAXDIM=64 (10 289 and 22 123, unchanged). Specifically:

    the scratchpad and vreg files are now **derived** as
    :math:`\text{MAXDIM}^2/T`, which is 64 rows each at MAXDIM=16 against the
    literal 512 and 256 they replaced; a 64-row file is not implemented the
    way a 512-row one is (BRAM 42 -> 40 says two memories left block RAM), and
    the shorter operand read path takes one cycle out of the **fixed** term.

That is why the delta is uniform: it is one cycle of pipeline depth in the
operand path, paid once per run rather than once per work item.

It is a (very small) **improvement**, not a regression, and it changes no
conclusion on this page --- one cycle is 0.6% at 4x4x4 and 0.005% at
64x64x64, and both of the shapes where it is largest were already inside
Gemmini's measurement spread. Every comparison here was measured *after* the
change, so only the historical row needed restating.

.. note::

   **The gate is what caught it.** ``reproduce.sh`` carries the published
   numbers as expectations and refused to pass, minutes after the merge, on a
   one-cycle shift in a refactor whose functional gates were all green. That
   is the whole argument for wiring published numbers into a check rather than
   into prose: correctness is not cycles, and the resource counts moving
   (BRAM 42 -> 40, FF 17 481 -> 17 075, LUT 26 583 -> 26 558 at an unchanged
   2.431 ns) proved the netlist had changed without saying by how much.

The T=8 column, re-measured
---------------------------

**The T=8 cycle column was re-measured on current ``main`` and is
unchanged.** It had been taken before the parametric ``DMA_WORDS`` refactor,
which is the exposure that moved the published MAXDIM=16 row by one cycle
(:ref:`benchmarks-one-cycle`). Re-run from a fresh ``csynth``:

.. code-block:: text

   8x8x8     285   (was 285)      16x16x16   493   (was 493)
   16x16x8   424   (was 424)      64x64x64  7083   (was 7083)

every one bit-exact and identical. That is consistent with the mechanism
rather than merely reassuring: at T=8/MAXDIM=64 the memory *sizes* do not
change (``OPERAND_ROWS`` is 512 either way), only ``rbA``/``rbB`` grew by one
word --- which is what took BRAM 58 -> 62 and nothing else. **The burst loop
is now shown cycle-neutral at ``DMA_WORDS=1`` in three independent
configurations**: T=4/MAXDIM=16, T=4/MAXDIM=64 and T=8/MAXDIM=64. That is
what licenses the widening being a pure opt-in.

The entry most exposed was **16x16x8**, the only shape where this design beats
Gemmini on a supportable margin, and it returned 424 exactly.

An earlier reading of the same bar
----------------------------------

**Every steady-state conclusion on this page clears that bar by two to three
orders of magnitude.** The 1 748-cycle deficit at 64x64x64 is 874x the
2-cycle spread there; the -960-cycle burst saving is 480x it. **The one number
that does not clear it is 4x4x4**, where a 13-cycle deficit sits inside a
10-cycle spread --- so *the 1.06x at 4x4x4 is not a supportable claim of a
difference* and is reported only as part of a sign that is consistent across
all ten shapes. 16x16x8 has the largest absolute spread (20 cycles) but a
155-cycle deficit, so it survives.

.. rubric:: Footnotes

.. [#sweepknobs] An earlier revision of this work put the
   ``TPU_SET``/``TPU_SHAPES`` knobs on ``SHAPES`` itself, which would have
   silently changed what all of those measured.

.. [#bestoftwo] An earlier revision of this page read three shapes as wins
   from best-of-two and then downgraded all three to "level" as a precaution;
   the five-trial measurement shows the precaution was one shape too strong.
