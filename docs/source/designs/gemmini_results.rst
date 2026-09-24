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

###########################################
TinyTPU-isa vs Gemmini: Results and History
###########################################

The measurements produced by the comparison set up in
:doc:`gemmini_comparison`, and the claims this page has made and then
withdrawn. Read the method there first: no number below is interpretable
without knowing which window it was taken over and which build produced it.

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

The parity baseline: results
----------------------------

The configuration, and the rules fixed before any of these numbers were
measured, are on :ref:`gemmini-parity`.

Result at T=4: faster than matched Gemmini at all ten shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   **One build, one configuration, every shape of the matched set.** Ours is
   Vitis cosim ``ap_start`` to ``ap_done``, ``-m_axi_latency 0``, bit-exact at
   every shape; Gemmini is the median of five with its full min-max spread.
   ``csynth`` on xcu280 at the 3.33 ns target: **BRAM 100, DSP 14, FF 27,730,
   LUT 35,661, estimated period 2.431 ns** -- the same estimated period as the
   shipped design, which is where the cost is *not*.

   .. list-table::
      :header-rows: 1

      * - shape
        - shipped
        - **parity-t4**
        - Gemmini
        - ratio
      * - 4x4x4
        - 218
        - **161**
        - 208 +/- 25
        - **0.774x**
      * - 8x8x8
        - 357
        - **232**
        - 324 +/- 36
        - **0.716x**
      * - 12x12x12
        - 563
        - **366**
        - 458 +/- 17
        - **0.799x**
      * - 16x16x8
        - 677
        - **419**
        - 527 +/- 44
        - **0.795x**
      * - 16x16x16
        - 879
        - **579**
        - 691 +/- 44
        - **0.838x**
      * - 32x32x32
        - 3 752
        - **2 820**
        - 2 977 +/- 34
        - **0.947x**
      * - 32x64x32
        - 6 824
        - **5 188**
        - 5 570 +/- 147
        - **0.931x**
      * - 48x48x48
        - 10 289
        - **8 384**
        - 9 100 +/- 35
        - **0.921x**
      * - 64x32x64
        - 12 907
        - **10 450**
        - 11 175 +/- 18
        - **0.935x**
      * - 64x64x64
        - 22 123
        - **19 186**
        - 20 287 +/- 34
        - **0.946x**

   Every margin clears Gemmini's spread by a wide factor (the tightest,
   64x32x64, by 40x). At 64x64x64 this is **85.4 % of peak against Gemmini's
   80.8 %** -- the first configuration of this design to convert its array
   better than Gemmini converts its own, and the answer to the question the
   benchmarks page left open when our column converged to 1.09x but did not
   invert.

**This is a win, so it owes a mechanism at every shape**, which is the rest of
this section. It is also bought with area, stated first so it is not buried:
against the shipped MAXDIM=64 control (BRAM 52, DSP 14, FF 17,488, LUT
26,554) it costs **+92 % BRAM, +58.6 % FF and +34.3 % LUT on the FPGA**, and
on standard cells the depth alone is about **+33,280 flip-flops, roughly
+16.6 % sequential cells** (:ref:`limitation-24-price`). No clock change.

The three changes, and their mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. The banked burst widening.** ``dma_ld``'s operand burst
moves ``DMA_WORDS`` packed words per iteration instead of one, and ``rbA`` /
``rbB`` are cyclically partitioned by ``DMA_WORDS`` so write ``w`` always lands
in bank ``w`` and every bank has exactly one writer (``ff7beaf1``).

The prologue is what the measurement says the deficit was: at MAXDIM=64 the
burst reads whole 64-byte DRAM rows, so it costs ``max(M,K) * MAXDIM/T``
iterations however narrow the operand actually is -- 1,024 of the 22,123
cycles at 64x64x64 and 128 of the 424 at T=8 16x16x8 -- and widening divides
that by ``DMA_WORDS``. Worth, alone: 22,123 -> 21,163 at 64x64x64.

**2. The interleaved program order**, on an unchanged netlist: every operand
column block is loaded just before the first ``mm`` that reads it, so the
first ``mm`` waits behind ``M + K`` load rows instead of ``Kt*M + Nt*K`` and
the remaining loads overlap the array instead of running ahead of it. At
64x64x64 that prefix is 2,048 rows of a 20,807-cycle run, and removing it is
worth **-1,621**. This is the change the measurement wanted from the start and
could not have: it deadlocks at ``Kt >= QD``, which is why it is listed after
the depth rather than before it.

**3. ``QD=32``.** Two things at once, which is why it earns its area. It makes
the interleaved order *legal*: the order hangs in RTL whenever ``Kt >= QD``
(:ref:`limitation-24`), and at T=4/MAXDIM=64 the largest ``Kt`` any
expressible shape can have is ``MAXDIM/T = 16``, so **QD=32 clears every shape
this build can express**, not merely the ten that were measured. And it is
worth cycles on its own, because a deeper queue lets the sequencer run further
ahead of the units it dispatches to: on the shipped order at MAXDIM=64,
16x16x16 goes 639 -> 628 and 64x64x64 goes 21,163 -> 20,807 from depth alone.

The three are separable and were measured separately, at 64x64x64:
22,123 (shipped) -> 21,163 (widening) -> 20,807 (depth) -> **19,186**
(interleaved order). Nothing here is a tuning constant: each is a structural
claim with its own cycle count.

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

The intermediate step, kept because it separates the changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The widening **alone**, at ``QD=8`` with the shipped program order (BRAM 100,
DSP 14, FF 25,026, LUT 33,799, 2.431 ns), which is what the first version of
this baseline shipped: 172 / 262 / 383 / 437 / 639 at the five latency shapes
-- already faster than Gemmini at all five -- and 3,272 / 5,864 / 9,569 /
11,947 / 21,163 at the five steady-state ones, which was **4.3 % to 9.9 %
behind**. That column is why the write-up said "comparable, not parity"; the
depth and the program order are what closed it, and keeping the intermediate
column is what lets a reader see which change did what.

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

**Where parity is NOT reached, plainly.** At T=4, nowhere: all ten shapes are
faster. **At T=8 this configuration has not been built**, and the column above
is the widening alone at ``QD=8``, which is behind at four steady-state shapes
by 4.6 % to 10.4 %. That is the one outstanding measurement of this baseline
and it should not be assumed to follow from T=4: the depth and the program
order both interact with ``Kt = K/T``, which is *halved* at T=8, so the
interleaved order is legal there at a smaller ``QD`` and buys less prefix. Do
not quote a T=8 parity result until it is run.

The honest summary is therefore **faster than matched Gemmini at every shape
of the T=4 set, at a real area cost, with T=8 unmeasured** -- not "comparable"
as an earlier revision of this section concluded from the widening alone.

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

Why we are ahead, at both ends of the shape set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A win needs a mechanism or it cannot be defended, and the two ends of this
table win for **different** reasons. That is the part to carry: an earlier
revision of this section could only explain the small shapes, and said so.

**The latency shapes: fixed cost.** A small shape is almost entirely prologue
on both machines. Ours is 47 cycles of ``s_axilite`` programming, one operand
pass and one drain (measured above); Gemmini's window is five ``config`` RoCC
instructions, the ``loop_ws`` dispatch, the mesh's own fill and drain and a
closing ``fence`` -- of which the configs and fence alone were measured at 44
cycles at DIM=4. Widening the burst removes most of *our* prologue and cannot
touch Gemmini's, so the machine with the shorter remaining fixed cost wins,
and at these shapes that is now us by 16-28 %.

**The steady-state shapes: the operand pass was never a fixed cost.** This is
the correction that turned "comparable" into a win. The serial load prefix is
``Kt*M + Nt*K`` rows, which **grows with the problem** -- 1,152 rows at
48x48x48 and 2,048 at 64x64x64 -- so it is a *marginal* cost of 0.095 cycles
per ``mm`` row, not an intercept, and reading it as a prologue is what hid it.
Interleaving the loads with the compute takes it off the critical path, and
the measured marginal rate between 48x48x48 and 64x64x64 moves accordingly:

.. list-table::
   :header-rows: 1

   * - build
     - cycles per ``mm`` row, 48^3 -> 64^3
   * - widening alone, ``QD=8``, shipped order
     - 1.224
   * - Gemmini DIM=4
     - **1.181**
   * - **parity-t4**
     - **1.140**

The 0.084 the change is worth is the 0.095 the load rows cost, within the
residual. So we are no longer merely amortising a fixed cost better: **our
marginal rate is now below Gemmini's**, which is why the win does not thin out
as the shapes grow (0.921x-0.947x across the five steady shapes) and why
64x64x64 reaches **85.4 % of peak against Gemmini's 80.8 %**.

**What this does to the earlier attribution.** The steady deficit was
attributed in part to ``accu``'s retire pass sharing the accumulator's single
port, worth ``1/Kt``, and that attribution explicitly failed at T=8. It is now
clear it was at best a co-factor: the retire pass is unchanged by anything in
this configuration, and the deficit went away regardless. **The operand pass,
not the retire pass, was the larger term.** The retire-pass paragraph above is
kept as measured, with this correction attached, because a superseded
attribution that was honest about its own failure is worth keeping visible.

**The T=8 16x16x8 win, which was previously unexplained**, is the extreme case
of the fixed-cost mechanism: the shape is 2x2x1 tiles, so ``accu`` does 48
iterations of real work inside a 424-cycle window, and 312 of those cycles are
prologue and drain that the widening then halves. It is the shape with the
fewest tiles per unit of window on either side, which is why its margin
(0.62x) is the largest anywhere in these tables. The 1.33x standard-cell area
that T=8 costs over T=4 at the same MAXDIM is the price, at no clock penalty.

Two disclosures that belong with these numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _gemmini-area:

Area: the axis this page was missing
------------------------------------

Until 2026-09-22 this project published standard-cell area for its own design
at three configurations — 1,136,598 at ``T=4, MAXDIM=16``, 1,865,314 at ``T=4,
MAXDIM=64``, 2,481,926 at ``T=8, MAXDIM=64`` — and **none for Gemmini**, while
comparing cycles against Gemmini at a matched array size. A cycle comparison
with an area column on only one side is not a comparison; it is an
advertisement. This section is the correction, and the RTL that makes it
possible is in ``examples/accelerator/tinytpu_vitis/gemmini_rtl/``.

The comparable unit is ``Gemmini``, not the SoC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chipyard elaborates a whole system-on-chip: a Rocket core, an L1 instruction
and data cache, an inclusive L2, the system and memory buses, the debug module,
a bootrom and a DRAM model. Our design has none of it — no host, no cache, no
core; its program is in DRAM and it starts on ``ap_start``. Synthesising the
chipyard top would compare a CPU with a matrix unit.

The unit that answers the same question on both sides is Gemmini's own
accelerator module, ``Gemmini`` (``Controller.scala:45``), whose ports are the
RoCC command and response interface, one TileLink master for its DMA, and a
page-table-walker port. Everything the accelerator is made of is inside it:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - subsystem
     - Gemmini modules
     - ours
   * - spatial array
     - ``Mesh``, ``Tile``, ``PE``, ``MacUnit``, ``MeshWithDelays``,
       ``TransposePreloadUnroller``
     - ``pe``, ``accu``
   * - scratchpad
     - ``Scratchpad``, ``ScratchpadBank``, ``mem``/``mem_ext``
     - ``spad`` and the vector registers
   * - accumulator
     - ``AccumulatorMem``, ``TwoPortSyncMem``, ``AccumulatorScale``,
       ``AccPipe``, ``ScalePipe``
     - ``ar``
   * - control
     - ``ExecuteController``, ``LoadController``, ``StoreController``,
       ``ReservationStation``, ``LoopMatmul*``
     - ``sequencer``
   * - memory interface
     - ``StreamReader``, ``StreamWriter``, ``BeatMerger``, ``XactTracker``,
       ``DMACommandTracker``, ``TLBuffer``, ``TLXbar``
     - ``dma_ld``, ``dma_st``

The boundary is not asserted, it is **computed**: the export takes the
transitive closure of the module instantiation graph from ``Gemmini`` over the
elaborated RTL, which gives 138 modules at DIM=4 and 136 at DIM=8 with **no
undefined module left over**. Anything outside that closure — Rocket and its
tile, the L1s, the L2's banks and directory, the SoC buses and peripherals, the
DRAM model and the test harness — is cut, and the closure is checkable by
anyone who disagrees with a particular cut.

Three things stay in although a GEMM never touches them, and all three make
Gemmini look **bigger**, which is the direction that does not flatter us:
the conv pipeline (``LoopConv*``, ``Im2Col``, ``PixelRepeater``,
``ZeroWriter``), the output-stationary datapath that ``dataflow = BOTH``
carries, and — the one worth naming in any quote, because it is a real block —
**an fp32 scaling pipeline**. ``defaultConfig``'s ``mvin_scale_args`` are
``Float``, so Gemmini's default int8 accelerator instantiates 8 recoded-float
multiply-adds at DIM=4 and 12 at DIM=8. We have none. Removing any of the three
would be tuning the opponent, so none is removed; naming them is the honest
alternative.

**One cut is genuinely ambiguous and is reported both ways.** ``FrontendTLB``
— Gemmini's private four-entry TLB, with ``DecoupledTLB``, ``DTLB_2``,
``PMAChecker`` and ``PMPChecker_s6`` — is *inside* the ``Gemmini`` module,
because Gemmini's DMA issues virtual addresses. Ours takes physical addresses
over AXI and has no translation at all. Cutting it would mean editing Gemmini's
RTL, so it is left in and its area is reported separately from DC's
``report_area -hierarchy``; both the with-TLB and the without-TLB figures
belong in any quote.

The memory treatment is the whole problem, and the choice made
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our three published areas are synthesised with ``sram_mode='none'``: the
scratchpad, vector registers and accumulator become flip-flops, which is why
~80 % of the cell area is non-combinational. Gemmini's scratchpad is 256 KiB
and its accumulator 64 KiB. Flip-flopping 320 KiB is **2,621,440 registers**,
against 200,561 sequential cells in our *entire* ``T=4`` design. The resulting
number would be a measurement of the memory treatment, not of either design,
and it would be roughly thirteen times our whole area before a single gate of
Gemmini's datapath was counted.

Four ways out, and why the fourth is not enough on its own:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - option
     - what it costs
   * - flip-flop both, caveat the number
     - the caveat cannot be read. A 13x ratio that is entirely an artefact of
       capacity is not improved by a sentence underneath it; readers quote
       numbers, not footnotes.
   * - SRAM macros on both sides
     - the honest ASIC implementation, and **it invalidates our three
       published figures**, which would all have to be re-run. It also needs
       macros for every geometry on both sides. Right answer eventually; not
       an answer today.
   * - match Gemmini's capacity to ours
     - changes Gemmini's hardware. Legitimate only if the change is provably
       cycle-neutral — which, at the published shapes, it is (below).
   * - report logic area excluding memories
     - the only figure invariant to all of this, but it deletes the axis the
       comparison most needs: a design that buys cycles with memory looks free.

**The choice is the third and the fourth together**, and it has to be both:
the fourth is the headline because it is the only figure invariant to the
memory treatment, and the third exists so that the full-memory figure beside it
is not absurd. Gemmini is elaborated at an 8 KiB scratchpad and a 4 KiB
accumulator — 98,304 bits, 12 KiB total — and each configuration ships two file
lists from one export, one including the memory arrays and one omitting them so
``mem``/``mem_0`` elaborate as empty black boxes.

Why 4 KiB of accumulator: it is the smallest capacity inside the envelope the
cycle-neutrality sweep already covers. 2 KiB is legal under Gemmini's
``require`` clauses at both array sizes, but a 16x16 int32 tile is 1 KiB and the
binding limit is the accumulator's *half* capacity, so 2 KiB sits exactly on the
edge where the tiling search has never been run.

Why 8 KiB of scratchpad needs a correction to this page's own arithmetic.
Measured out of the shipped RTL — every ``*_RAM_*`` module's depth times its
width times its hierarchical instance count — our operand and accumulator
storage is:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - our variant
     - scratchpad + vregs
     - accumulator
     - vs Gemmini's 12 KiB
   * - ``T4_MAXDIM16_shipped_baseline`` (1,136,598)
     - 0.5 KiB
     - 2.0 KiB
     - Gemmini has **4.8x**
   * - ``T4_MAXDIM64_shipped`` (1,865,314)
     - 8.0 KiB
     - 2.1 KiB
     - Gemmini has 1.19x
   * - ``T8_MAXDIM64`` (2,481,926)
     - 8.0 KiB
     - 4.25 KiB
     - **matched to 2 %**

**The "4 KiB scratchpad, 4 KiB of vector registers and a 2.1 KiB accumulator"
this page quotes elsewhere describes** ``MAXDIM=64``, **not the** ``MAXDIM=16``
**baseline whose area is published.** At ``MAXDIM=16`` both arrays are 64 rows
of 32 bits — 256 bytes apiece — because the memories are derived from
``MAXDIM``. So 8 KiB of Gemmini scratchpad is a near-exact match against
``T8_MAXDIM64`` and against ``T4_MAXDIM64_shipped``, and it hands Gemmini
**4.8x our storage** against the MAXDIM=16 baseline the DIM=4 cycle numbers
belong to. The total including the DMA read buffers and the sequencer's small
RAMs is 3.50 KiB, 18.62 KiB and 20.75 KiB respectively.

**That asymmetry is why the logic-only figure is the headline and the
full-memory figure is the secondary**, rather than the other way round. At the
DIM=8 pair the two agree; at the DIM=4 pair only the logic-only figure is
defensible without a paragraph of arithmetic attached.

**The capacity change is legitimate only because it is cycle-neutral, and that
was measured before this section existed.** ``tiled_matmul_auto``'s own tiling
search issues exactly one ``loop_ws`` at every published shape from 256/64 KB
down to 4/4 KB, so not one of 161 / 220 / 347 / 391 / 593 moves — see :ref:`The capacity asymmetry is cycle-neutral <gemmini-capacity-neutral>`. It stops being
neutral at 32x32x32, so **this section does not extend to the ten-shape sweep
at MAXDIM=64 without being re-measured**, and an area quote at 64x64x64 would
need its own capacity argument.

What the choice gives up, stated plainly: the 256 KiB scratchpad is a **real
capability** that this comparison deliberately removes. The area answers *what
does the same amount of local memory cost in each design*. It does not answer
*what does the shipped Gemmini cost*, and it must never be quoted against a
published Gemmini area, which will have been taken with SRAM macros at full
capacity and is not the same measurement at all. The logic-only figure is the
safer of the two to carry, being invariant to both the memory treatment and the
capacity choice — but only if our side omits its ``*_RAM_*`` modules in exactly
the same way, which is a second run on our side and not a free comparison.

Frequency is a column, not a footnote
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A cycle comparison between two designs at different achievable frequencies is
not a comparison**, and this page has never checked that the two are close. We
have a Vitis 2.431 ns estimate and a 3.33 ns ASIC constraint that our design
meets with +0.21 ns of slack; for Gemmini we have **nothing at all**. Until
that column is filled, every ratio on this page — 1.07-1.24x, 1.09x at 64³ —
is a statement about cycles that may or may not survive being restated in
seconds, and a reviewer will open that first.

The six DC runs give it for free: worst slack at the 3.33 ns constraint is
already in the QoR report, so

.. math::

   F_\text{max} \ge \frac{1}{3.33\,\text{ns} - \text{slack}}

That is a **floor, not Fmax**: DC stops optimising once the constraint is met,
so a design with comfortable slack is only shown to be *at least* that fast.
From the runs already done:

.. list-table::
   :header-rows: 1
   :widths: 40 18 22 20

   * - design
     - worst slack
     - violating paths
     - Fmax floor
   * - TinyTPU-isa ``T=4, MAXDIM=16``
     - +0.21 ns
     - 0 setup, 0 hold
     - >= 320.5 MHz
   * - TinyTPU-isa ``T=4, MAXDIM=64``
     - +0.21 ns
     - 0 setup, 0 hold
     - >= 320.5 MHz
   * - TinyTPU-isa ``T=8, MAXDIM=64``
     - +0.20 ns
     - 0 setup, 0 hold
     - >= 319.5 MHz
   * - Gemmini DIM=4
     - not yet run
     - —
     - unknown
   * - Gemmini DIM=8
     - not yet run
     - —
     - unknown

All four of our runs closed timing with zero violating setup paths and zero
hold violations. That is the baseline every Gemmini slack is compared against.

.. admonition:: If Gemmini misses 3.33 ns, this is how it gets written
   :class: important

   It may. ``defaultConfig`` sets ``tile_latency = 0``, so a tile is
   combinational through its PE and the mesh path is ``DIM`` MACs deep with no
   register between them — a path that grows with mesh width, so DIM=8 is the
   likelier miss. **The synthesis session will report a miss as a miss**, with
   total negative slack and a violating-path count, and will not loosen the
   constraint: relaxing it for one side would destroy the only thing making the
   two areas comparable.

   The sentence to write then is **"Gemmini's RTL was not targeted at this
   constraint"**, not "Gemmini is slower". Vitis emitted ours *for* 3.33 ns and
   pipelined it accordingly; Chisel emits Gemmini at no target at all, and
   Gemmini's own tape-outs pick their own period. A miss is a finding about the
   **two design flows** — one of which takes a frequency target as an input —
   and it says nothing about how fast a Gemmini pipelined for 3.33 ns would be.
   Written the other way it would be a claim this page cannot support.

   This paragraph exists so that sentence is not composed under pressure once
   the number is in.

If the two sides' slacks at the DIM=4 point differ by more than ~0.3 ns, real
Fmax needs bisecting — two or three tightened runs per design. That is not
requested on spec; the floors decide whether it is worth the slot.

Area results, pending
~~~~~~~~~~~~~~~~~~~~~

Pending. No DC run has completed at the time of writing; the ETA is in
`When this comparison will exist`_. The table below is the shape the answer has
to take, and **no ratio is quoted until a row has its frequency floor**, for
the reason above.

.. list-table::
   :header-rows: 1
   :widths: 28 12 14 14 14 18

   * - design
     - memory
     - Fmax floor
     - total area
     - logic only
     - less the TLB
   * - TinyTPU-isa ``T=4, MAXDIM=16``
     - 3.50 KiB
     - >= 320.5 MHz
     - 1,136,598
     - not yet run
     - n/a (no TLB)
   * - TinyTPU-isa ``T=4, MAXDIM=64``
     - 18.62 KiB
     - >= 320.5 MHz
     - 1,865,314
     - not yet run
     - n/a
   * - Gemmini DIM=4, 8/4 KiB
     - 12.0 KiB
     - not yet run
     - not yet run
     - not yet run
     - not yet run
   * - TinyTPU-isa ``T=8, MAXDIM=64``
     - 20.75 KiB
     - >= 319.5 MHz
     - 2,481,926
     - not yet run
     - n/a
   * - Gemmini DIM=8, 8/4 KiB
     - 12.0 KiB
     - not yet run
     - not yet run
     - not yet run
     - not yet run

Power is deliberately not a column. See above.

Both of our ``T=4`` rows are there because neither is a clean opponent on its
own: ``MAXDIM=16`` is the build the DIM=4 cycle numbers were measured on, and
``MAXDIM=64`` is the build whose memory capacity matches. A reader wanting one
number should take the **logic-only** column, where the difference between them
is our own ``+64.1 %`` operand-space step and not a property of Gemmini.

That split has a convenient resolution: :ref:`The parity baseline <gemmini-parity>` is at
``MAXDIM=64`` on both ``parity-t4`` and ``parity-t8``, so the parity
configurations are the ones whose local memory matches Gemmini's 12 KiB — to
19 % at T=4 and to 2 % at T=8 — *and* whose cycles are measured against matched
Gemmini across 18 points. **The area rows that belong beside the parity cycle
numbers are the ``MAXDIM=64`` rows, not the baseline.** The ``MAXDIM=16`` row
stays in the table because the five original cycle counts belong to it, not
because it is the right area to quote against parity.

The DIM=8 pair is matched on memory to 2 % but **not on operand space**: our
``T=8`` build is at ``MAXDIM=64``, a fourfold operand space against the
``MAXDIM=16`` baseline, and that difference is worth +64.1 % of our own cell
area on its own. Quoting DIM=8 against DIM=4 across designs without naming both
changes is the same error this project has already withdrawn once.

Three caveats travel with every number in that table, and none of them is
optional:

* **Total area is not comparable at DIM=4.** The capacity gap is 4.8x and no
  legal Gemmini configuration closes it; the logic-only column leads.
* **The memory-treatment figure is not a comparison.** 2,621,440 registers for
  stock Gemmini's memories against 200,561 sequential cells in our entire T=4
  design measures ``sram_mode='none'``, not either design, and must never
  appear as a ratio.
* **Logic-only omits the memory interface too, on both sides equally.** With
  the memory modules out of the file list DC infers nothing for them and their
  ports become dangling nets, so the figure excludes the port and address logic
  as well as the array. That is what makes it invariant to the memory
  treatment, and it is why a logic-only area may only ever be compared against
  another logic-only area — never against a full one on either side.
* **Both sides drop the same two things, by the same criterion.** The
  scratchpad and the accumulator, and nothing else: ``mem_ext``/``mem_0_ext``
  on Gemmini's side, ``*_spad_RAM_*`` and ``*_ar_RAM_*`` on ours — 98,304 bits
  against 20,480 at ``MAXDIM=16`` and 100,352 at ``T=8``. DMA buffers and
  control-path RAMs stay on both sides: ours because Gemmini's DMA buffering
  lives in ``BeatMerger``/``XactTracker`` as plain registers and stays, and its
  depth-2 queue RAMs stay. Both lists are generated — ``export_rtl.py
  --manifests`` and ``export_gemmini_rtl.py`` — because a hand-edited list on
  one side is exactly the difference that would stop the two being comparable.

When this comparison will exist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The blocking resource is a Design Compiler slot on zhang-21, which another
session owns and which is mid-way through this project's own variants. Six
runs close the table:

=====  =========================================  ==================
order  run                                        redone if stream depth lands?
=====  =========================================  ==================
1      Gemmini DIM=4, logic only                  no
2      TinyTPU ``T4_MAXDIM16``, logic only        **yes**
3      Gemmini DIM=4, full                        no
4      Gemmini DIM=8, logic only                  no
5      TinyTPU ``T8_MAXDIM64``, logic only        **yes**
6      Gemmini DIM=8, full                        no
=====  =========================================  ==================

Historic wall times for our own variants were 37-72 minutes each; the
logic-only runs are faster, having far fewer cells to map.

**Scheduled 2026-09-22.** Run 1 is in progress; first result within the hour,
all six within 4-6 hours. The session is taking **1, 3, 4, 6 first** — every
Gemmini run, all of them immune to the stream-depth change below — and then 2
and 5, which are ours and are the throw-away candidates.

**Sequencing around the stream-depth change.** The design may adopt a stream
depth of 8 -> 16, because three legal tiled programs otherwise never complete;
it costs +9.3 % flip-flops on FPGA and its ASIC price is being measured
separately. If it lands, our sequential-cell counts move and runs 2 and 5 —
plus the three already-published TinyTPU areas — need repeating. **Gemmini's
four runs are untouched by it.** So the decision is: **run now, redo ours
later**, because four of the six are permanent and the two that are not are the
cheap ones. Holding the whole batch for a decision that has not been made would
trade a certain delay for an uncertain saving of two short runs.

.. note::

   **Status 2026-09-22: running.** All six runs are expected within 4-6 hours,
   the first within the hour. Power is **parked, not cancelled**: no
   ``report_power`` figure will be quoted from this batch, and a separate
   effort is producing switching-activity files — which, if they arrive, move
   the first row of the power table above from "1-2 days of bring-up" to
   "already done", without touching the two artefacts that are the actual
   blocker.

   This note is the thing to update as results land.

The handoff
~~~~~~~~~~~

``gemmini_rtl/DIM4_int8_capmatched/`` and ``gemmini_rtl/DIM8_int8_capmatched/``,
written by ``export_gemmini_rtl.py``, in the shape ``rtl_handoff/`` already
uses: flat RTL, ``sv2v_manifest.f`` in dependency order, ``MANIFEST.json``
naming the top module and carrying the configuration, and a README stating the
elaborating commit and every cut. The exporter refuses an export whose named
top module is undefined and one whose resource record is empty — the two guards
``export_rtl.write_design`` makes, which caught a ``T8_MAXDIM64`` export that
shipped 223 self-consistent files with no top module in them — plus a third for
this comparison: an export carrying more than 32 KiB of memory array, which is
what a stock-capacity elaboration looks like.

Two differences from the Vitis handoff, both measured here:

* **sv2v IS needed**, unlike for Vitis output. "Chisel emits Verilog" is not
  true of firtool 1.75.0: ``CounterFile``, ``LoopMatmulStC`` and ``RRArbiter``
  use packed multidimensional arrays (``wire [7:0][6:0]``) and assignment
  patterns (``'{3'h5, 3'h0, …}``) **outside any** ``ifdef``. Verilator rejects
  every file list under ``--language 1364-2001`` with 6-7 syntax errors and
  accepts them all as SystemVerilog, so the files are named ``.sv`` and the
  flow needs ``normalize_rtl: True`` or ``analyze -format sverilog``.
* **``SYNTHESIS`` must be defined at read time**, or ``plusarg_reader`` arrives
  as ``$value$plusargs`` inside an ``initial`` block and 173 ``logic``
  declarations come back with it.

The top-module check that forced ``normalize_rtl: False`` on our own RTL is
*not* a problem here: ``module Gemmini(`` carries a trailing comment, not an
attribute, so the collector's ``^\s*module\s+<top>`` pattern matches.

All four file lists — two configurations, full and logic-only — pass
``verilator --lint-only -DSYNTHESIS --top-module Gemmini`` with no error, which
is the cheapest available evidence that the closure is complete before DC sees
it.

.. _gemmini-history:

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
(``dev/records/tinytpu/impact-results/``): operand staging (~84 cycles), the serial DMA ahead of
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
``base`` row) plus asserted textual patches, so its diff
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
(``dev/records/tinytpu/impact-results/base.rle.txt``) reads, in monitor cycles: 0-47 region start
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

The variant generator that produced these numbers, ``make_variants.py``, is
gone: it patched the pre-landing baseline textually and had gone stale
against the landed design (it named a scratchpad constant, ``A_SP``, that the
shipped ISA no longer defines), and the CHIA harness's own candidate/diff
pipeline (``chia_agent/evaluate.py``, ``chia_agent/accept.py``) is the
maintained way to measure a new variant today. The table above and the raw
per-run output it was measured from (``dev/records/tinytpu/impact-results/``)
are the durable record; they are not re-derived by re-running anything.

Once a variant's ``.py`` file exists (however it was produced), from
``examples/accelerator/tinytpu_vitis/impact/`` on ``main``:

.. code-block:: bash

   source env.sh                    # conda allo, LLVM_BUILD_DIR, OMP, TMPDIR
   python pyrun.py bench_variant.py v_design_dep                     # simulator
   TPU_SHAPES=4x4x4,16x16x16 python pyrun.py cosim_variant.py v_design_dep runs/v_design_dep
   ./profile.sh runs/v_design_dep   # per-process timeline of the last shape

``cosim_variant.py base`` builds the *shipped* design; the pre-landing one
would have been ``v_base``. ``pyrun.py`` strips the conda env's editable
finder, so ``allo`` resolves to this checkout. ``cosim_variant.py`` reuses ``../cosim.py`` (``align_value`` 64,
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
instruction-programmable claim can rest on. What replaced it is :ref:`One hardware build, workload swept as data <gemmini-one-build>`.

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
