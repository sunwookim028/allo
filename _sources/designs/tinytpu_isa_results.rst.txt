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

####################################
TinyTPU-isa: Results and History
####################################

What TinyTPU-isa (:doc:`tinytpu_isa`) measures: cycle counts, resources, the
one standard-cell synthesis figure, and the superseded figures and withdrawn
claims behind them. The design itself, how to build it and how to verify a
change are on :doc:`tinytpu_isa`; the benchmark set at ``MAXDIM=64`` and the
Gemmini comparison are on :doc:`benchmarks` and :doc:`gemmini_comparison`.

Headline
--------

.. note::

   Headline, as of 2026-09-24 (``63ee6ec7``): one hardware build runs every
   shape as data, all five benchmark shapes are bit-exact in RTL
   co-simulation, and the design takes **175 / 265 / 421 / 482 / 674** cycles
   at 4x4x4 / 8x8x8 / 12x12x12 / 16x16x8 / 16x16x16 (Vitis ``cosim``,
   ``-m_axi_latency 0``).

   **Those five are a ``T=4``, ``TPU_MAXDIM=16`` measurement**, which is the
   configuration ``reproduce.sh`` pins. The shipped default is ``MAXDIM=64``,
   and ``MAXDIM`` is the DRAM row stride of every operand, so the same shape
   costs more on the bigger build -- 4x4x4 is 222 cycles at ``MAXDIM=64``
   against 175 at 16. The ``MAXDIM=64`` sweep, and every shape large enough to
   reach steady state, are on :doc:`benchmarks`.

   Measured over the same window on both sides, the design was
   **1.07-1.24x slower** than Gemmini at these five shapes. Over ten shapes at
   ``MAXDIM=64`` the deficit **converged to 1.09x at 64x64x64** at
   **74.1 % of peak**, and the two smallest shapes did not clear Gemmini's
   measurement spread; see :doc:`gemmini_comparison`. **Those figures were
   measured at the previous channel depth** (``QD=8``), against the row that
   ended on 2026-09-24, and the sweep has **not** been re-run at ``QD=16``, so
   they are the last measured comparison and not a statement about the shipped
   design. Nothing should be restated from them until the sweep is re-run;
   what was checked before landing is that no margin *flips*
   (:ref:`limitation-24-price`), which is a check and not a measurement.

   Until ``e24e433b`` (2026-09-19) the shipped design took
   **252 / 383 / 591 / 667 / 919** (1.55-1.8x behind Gemmini). The step
   between the two is the gap attribution's measured design stack, landed as
   the design (:ref:`tinytpu-isa-landing`, :ref:`gemmini-gap-attribution`).
   The published row was **172 / 262 / 418 / 484 / 686** until 2026-09-22 and
   **171 / 261 / 417 / 483 / 685** from then until 2026-09-24; what moved it
   by one cycle at every shape, and what the channel depth then did to it, are
   in `Earlier measurements and corrections`_.

   **The price of the current row** is **+9.3 % flip-flops** on FPGA at the
   shipped ``T=4``/``MAXDIM=64`` (+9.5 % at the ``MAXDIM=16`` the five shapes
   are measured on), with BRAM, DSP and the 2.431 ns estimated period all
   unchanged. **The ASIC price is unmeasured**: what is recorded is a count
   from the RTL priced at this design's own area per sequential cell, not a DC
   run (:ref:`limitation-24-price`).

Results
-------

Cycle counts
~~~~~~~~~~~~

**Current row, 2026-09-24 (``63ee6ec7``), ``T=4``, ``TPU_MAXDIM=16``,
``-m_axi_latency 0``, channel depth ``QD=16``:**

.. code-block:: text

   4x4x4  175    8x8x8  265    12x12x12  421    16x16x8  482    16x16x16  674

That is what ``reproduce.sh`` checks. The ``MAXDIM=64`` sweep, which is the
shipped default and the one that reaches steady state, is on
:doc:`benchmarks` -- at ``QD=16`` it opens 222 / 361 / 567 / 676 / 868 at
these five shapes, the same +4 / +4 / +4 / -1 / -11 deltas.

The row reached this from 171 / 261 / 417 / 483 / 685, which is what it was
between 2026-09-22 and 2026-09-24, and 172 / 262 / 418 / 484 / 686 before
that; both moves are in `Earlier measurements and corrections`_.

The table below is the **``e24e433b`` measurement**, taken before either move:
the derived memory sizing that landed afterwards took one cycle out of the
fixed term at every shape, and the channel depth then moved it again by
+4 / +4 / +4 / -1 / -11 (`Earlier measurements and corrections`_). Every
other column -- instruction counts, mismatches, the pre-landing comparison --
is unaffected. Measured by Vitis ``cosim`` (xsim), one build,
``-m_axi_latency 0`` (``dev/records/tinytpu/logs/cosim_isa_landed_sweep.log``), against the
pre-landing build that was shipped until then
(``dev/records/tinytpu/logs/cosim_isa_widened_sweep.log``):

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
(``dev/records/tinytpu/logs/cosim_isa_landed_stress.log``).

Utilization against the 4x4 array's peak is 2.3% / 12.2% / 25.8% / 26.4% /
37.3% (was 1.6% / 8.4% / 18.3% / 19.2% / 27.9%). Least squares against dynamic
instruction count gives fixed **74.5**, marginal **21.70** cycles per dynamic
instruction (was 151 and 17.28); the two fits are not comparable term by term,
because the landing removed instructions (the ``vld``\ s) as well as cycles,
so each remaining instruction carries more work. Against MACs instead: fixed
**192**, **7.95** MAC/cycle marginal, 49.7% of peak (was 289 and 6.17, 38.6%).

Where a run's cycles go, in the shape the units impose:

.. code-block:: text

   one GEMM, time to the right (not to scale)

   sequencer  [prefetch][ dispatch, one instruction at a time ]
   dma_ld             [ burst A ][ burst B ][ one operand row/cycle ]
   spm                                     [ hdr + T weight rows / mm ]
   vru                                     [ nr activation rows / mm  ]
   wld + pe                                  [ one wavefront row/cycle ]
   accu                                        [ 1 iteration per row ]
   dma_st                                            [ mvout rows ] --> C

   |<------ prologue ------>|<----- steady state ----->|<-- drain -->|

* **Prefetch** is ``IMEM_SIZE / 8`` iterations -- 7 at the shipped 56 words,
  since gmem0 is 512 bits wide and ``ib`` is cyclically partitioned by 8 --
  and nothing else can start, because every unit blocks on its first control
  word.
* **The operand bursts** are ``na*WPR`` and ``nb*WPR`` packed words, one per
  cycle at ``DMA_WORDS=1``, where ``na`` and ``nb`` are the DRAM row spans
  ``assemble()`` computes. They are serial before the first weight reaches the
  array, and they scale with ``MAXDIM`` rather than with the shape being run:
  a 4x4x4 GEMM on a ``MAXDIM=64`` build bursts 4 x 16 = 64 words to use 4.
  That is the largest remaining term in the deficit to Gemmini, and the one
  the opt-in ``DMA_WORDS`` widening attacks (:doc:`benchmarks`).
* **Steady state** is one word per cycle per link, every unit concurrent.
* **Drain** is the last partial sums walking up to ``T-1`` links south and up
  to ``T-1`` east before ``accu`` sees them, then ``mvout``'s rows leaving
  through ``dma_st``.

These numbers were measured with a memory that answers immediately; the
latency sweep (``-m_axi_latency`` 16 and 64: 214 / 702 and 358 / 894 at
4x4x4 / 16x16x16) and what a real memory system does to them is on
:doc:`gemmini_comparison`. **They are not faster than Gemmini**: 1.07-1.24x
slower like for like.

How the design got from 1017 cycles at 4x4x4 to the current row is on
:doc:`tinytpu_history`. What the last step was, where it came from, and what
deficit to Gemmini remains is on :ref:`gemmini-gap-attribution`; the five
changes that step consisted of are listed in
`Earlier measurements and corrections`_.

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
(``dev/records/tinytpu/logs/csynth_isa_prelanding.rpt``,
``dev/records/tinytpu/logs/csynth_isa_landed.rpt``).
Both columns are the ``e24e433b`` netlist; the derived memory sizing that
landed on 2026-09-22 took BRAM 42 -> 40, FF 17,481 -> 17,075 and
LUT 26,583 -> 26,558 at an unchanged 2.431 ns
(`Earlier measurements and corrections`_). The
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


Standard-cell synthesis: one number, and what it is not
-------------------------------------------------------

The shipped configuration has been synthesised to standard cells for the first
time, on 2026-09-22, on a different host from the one every other figure here
comes from.

**FreePDK45 / NanGate, ``view-standard``, 3.33 ns on ``ap_clk``, topographical,
flatten effort 3, memories as flip-flops** (``sram_mode='none'``), Synopsys DC
``W-2024.09``, via mflowgen 0.8.0 at commit ``aee0e5d6``. 37 minutes of wall
time, synthesising the current shipped design
(``dev/records/tinytpu/rtl_handoff/``,
``T4_MAXDIM16_shipped_baseline``).

============================== ==========================================
Total cell area                **1,136,598** FreePDK45 area units
  non-combinational            906,098 — **79.7 %**
  combinational                230,501
  macro / black box            0
Cells                          389,399 (200,561 sequential, 1,662
                               hierarchical)
Timing                         **MET**, worst slack **+0.21 ns**, critical path
                               3.08 ns of 3.33, 62 logic levels, zero violating
                               and zero hold violations
Power                          not re-reported for this run; power from DC
                               here is **indicative only** in any case, at
                               default toggle rates with no activity data
============================== ==========================================

This replaces a first run of **1,271,692** that synthesised an export of the
design *before* its memories were made derived from ``MAXDIM``; that netlist no
longer exists. The current design is **10.6 % smaller** and one cycle faster,
from the same change that took FPGA block RAM from 42 to 40 — so area and cycles
now describe the same design.

**Across configurations**, synthesised identically and all closing at 3.33 ns
(full table and reports: ``asic_synthesis/``):

========================================= ============ ==========
comparison                                cell area    FPGA says
========================================= ============ ==========
``MAXDIM`` 16 → 64, T=4 fixed              **+64.1 %**  +2.4 % FF
``T`` 4 → 8, ``MAXDIM``\ =64 fixed         **1.33x**    --
========================================= ============ ==========

The first row is the finding: the two substrates disagree about operand space
by roughly **27x**. On the FPGA it disappears into block RAM and looks nearly
free; with memories as flip-flops it is most of the design. So off the FPGA,
**``MAXDIM`` is the expensive knob, not ``T``**. Never quote T=8 against the
shipped baseline (2.18x) -- that changes both at once.

**Four fifths of the cell area is flip-flops**, which is the predicted result
rather than a surprising one: the scratchpad, the vector registers and the
accumulator are block RAM on the FPGA and become registers when the flow is told
to use no memory macros.

That is the whole interpretive caveat, and it runs in a direction worth naming.
**This number says more about the memory treatment than about the datapath**, so
it cannot be compared against any flow that used provided SRAM macros without
giving that flow the same treatment. It also means a *variant* comparison —
which is the useful thing this run enables — will **overstate** the area cost of
any change that buys cycles with more on-chip memory, because memory is being
priced as registers rather than as macros. The burst-widening candidate, which
costs +123 % block RAM, is exactly such a change.

What the run gives up, stated plainly: relative cell area only, no place and
route and therefore no routed timing and no real area, no DRC or LVS, and power
without activity data. The one thing it does establish beyond relative area is
that the design **closes timing at 3.33 ns in 45 nm standard cells**, which no
FPGA figure could have told us.

Two flow defects were found and worked around rather than papered over, and both
would bite the next person:

- The RTL-collection step rejects a top module whose declaration carries an
  attribute, because it matches ``^\s*module\s+<name>`` and Vitis emits the
  ``CORE_GENERATION_INFO`` attribute and the ``module`` keyword on one line.
  ``sv2v`` itself converted all 146 files without complaint; the failure is
  downstream of it. Bypassing ``sv2v`` is correct on the merits here anyway —
  Vitis emits Verilog-2001 — and it keeps the RTL byte-identical.
- ``view-tiny`` is not usable for this: it has no technology file, no scan
  cells, and no driving cell, so DC runs and emits unmapped GTECH. Use
  ``view-standard``.



.. _tinytpu-isa-history:

Earlier measurements and corrections
------------------------------------

Superseded figures, withdrawn claims, and the accounts of how particular
numbers moved. Nothing here is the current state; the design's own history,
at greater length, is on :doc:`tinytpu_history`.

.. _tinytpu-isa-qd16:

The channel depth moved the row again, 2026-09-24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``63ee6ec7`` made ``QD=16`` the default channel depth, because three legal
tiled programs never complete at depth 8 and all ten of that family complete
bit-exact at 16 (:ref:`limitation-24-qd`). The published row moved from
171 / 261 / 417 / 483 / 685 to **175 / 265 / 421 / 482 / 674**.

**The delta is lumpy, and that is the finding**: +4 / +4 / +4 / **-1** /
**-11**. A uniform delta would have been a fixed-cost change, as the
2026-09-22 move below was. This one is not. The three smallest shapes pay the
pipeline skew of deeper FIFOs; the two largest **get faster**, because a
deeper queue lets the sequencer run further ahead of the units it dispatches
to. So the depth buys completion *and* throughput where the queues are the
constraint, and charges four cycles where they are not.

The same deltas reproduce independently at ``TPU_MAXDIM=64``:
218 / 357 / 563 / 677 / 879 becomes 222 / 361 / 567 / 676 / 868. Correctness
is unchanged -- ``stress_isa`` is 640/640 exact at the new default -- and
cosim reports ``COSIM OK``.

**What it costs, and what is not yet known.** The FPGA price is **+9.3 %
flip-flops** at the shipped ``T=4``/``MAXDIM=64`` (+9.5 % at ``MAXDIM=16``),
with BRAM, DSP and the 2.431 ns estimated period unchanged. **The ASIC price
is unmeasured**: the figure on record is a count from the RTL priced at this
design's own area per sequential cell, not a DC run. And **the Gemmini parity
sweep was measured at ``QD=8``**, so the deficit has to be re-measured before
any comparison is restated. The full accounting is :ref:`limitation-24-price`.

The published row moved by one cycle, 2026-09-22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The row moved by one cycle at every shape on 2026-09-22**, from
172 / 262 / 418 / 484 / 686 to **171 / 261 / 417 / 483 / 685**, every
testbench still bit-exact. It was caught by the design's own reproduction
gate reporting ``DIFFERS`` rather than passing, and confirmed by two
independent co-simulation runs in separate processes with separate syntheses,
which reproduced it exactly.

A delta that is **uniform across shapes** is a fixed-cost change rather than a
per-work one. It arrived with the design edits that came in with the benchmark
work -- sizing literals replaced by a derived ``OPERAND_ROWS``, two
encoding-ceiling assertions, and a test-window fix -- and this page recorded
for several hours that *which of those saves the cycle is not yet identified*,
on the grounds that a number whose mechanism nobody can state is a number on
probation however well it reproduces.

**It was then identified, by measurement rather than inference** (``05169938``,
recorded in ``reproduce.sh``): rebuilding the same configuration with the
memory sizes the design used to carry, ``TPU_SPAD=512 TPU_NVR=256
TPU_NAR=128`` and nothing else reverted, returns all five of the old numbers
exactly. The scratchpad and vreg files are now derived as ``MAXDIM^2/T``,
64 rows each at ``MAXDIM=16`` against the literals 512 and 256 they replaced;
a 64-row file is not implemented the way a 512-row one is (BRAM 42 -> 40 says
two memories left block RAM), and the shorter operand read path takes one
cycle out of the fixed term. That is why the delta is uniform. It is a small
improvement, not a regression, and it changes no conclusion on this page.

The same rebuild moved the csynth figures to BRAM 40, FF 17,075 and
LUT 26,558 at an unchanged 2.431 ns. The full account, including the
cycle-neutrality of the parametric burst loop, is at
:ref:`benchmarks-one-cycle`.

The signed bit-slice bug that the spare-bit rule came from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the design was built, a bit-slice was extracted into a **signed**
``ap_int<N>`` in the emitted HLS:

.. code-block:: cpp

   ap_int<7> v268;  v268 = w02(60, 54);   // the nr field
   int32_t nr = v268;                     // 64 -> 0b1000000 -> -64

So a field whose top bit is set read back negative, and a loop bounded by it
ran zero times. Consolidating the ``vld`` broke 16x16x16 in cosim -- **251 of
256 outputs wrong** -- while the Allo dataflow simulator passed it: ``nr = 64``
for a 64-row ``vld`` read back as -64, the loop ran zero times, and the design
silently produced zeros. It appeared exactly at the field's sign-bit boundary:
``nr <= 63`` worked, 64 did not. The emitter has since been fixed and the
spare-bit encoding kept; see :ref:`limitation-12` and the rule in
:ref:`tinytpu-isa-spare-bit`.

The one-owner rule was charged to Vitis as well as to Allo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Correction, 2026-09-19: the one-owner rule is Allo's, not Vitis's.**
:ref:`tinytpu-isa-one-owner` charged it to both until this date, citing
``HLS 200-779 / 200-979`` as a Vitis refusal; only the Allo half holds. A Vitis 2023.2 probe (`impact/probe_shared/
<https://github.com/sunwookim028/allo/tree/main/examples/tinytpu/impact/probe_shared>`__)
shows that ``#pragma HLS stream variable=buf type=unsync`` makes Vitis share
an on-chip array between two processes, one per BRAM port (``HLS 200-824``,
``200-755``, ``200-634``); ``HLS 200-779`` applies only to *synchronized*
arrays. Allo is what refuses: it rejects a region-scope ``Stateful`` shared by
two kernels (``EmitVivadoHLS.cpp:3083-3122``) and never emits
``stream type=unsync``. Vitis does separately forbid one ``m_axi`` bundle read
by two processes (``HLS 200-1013`` / ``200-984``).

An earlier claim in this project, that removing ``vru``'s double handling of B
would need a second producer on a shared memory, **was wrong**: the
``v_wdirect`` / ``v_wdb`` variants remove it with ``spm`` still the one owner
of ``spad`` and one writer of ``wcol[0]``, and that is how the shipped design
has done it since ``e24e433b``. See :ref:`limitation-shared-memory`.

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

Earlier reproduce instructions, and the runs of them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The commands in :ref:`tinytpu-isa-build` replaced older reproduce instructions
(``d18de251``) that named ``TPU_M`` / ``TPU_K`` / ``TPU_N`` knobs, a
``simulator`` argument and ``OMP_NUM_THREADS=32``, none of which the current
scripts use; the numbers recorded then were produced at 32 threads.
``kpn_model.py`` has since been rewritten for the row-flattened units.

``reproduce.sh`` was run from a pristine worktree, including a fresh MLIR
build, and printed ``REPRODUCED`` with 172 / 262 / 418 / 484 / 686 and 0
mismatches in **5m49s** (``96c3aef6``, re-run at the docs commit); against the
pre-landing design it printed 252 / 383 / 591 / 667 / 919 in 5m48s
(``d18de251``). Both predate the one-cycle move above.
