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

###########################
TinyTPU-isa: Design History
###########################

This page records how :doc:`tinytpu_isa` reached its current numbers: the
predecessor designs it replaced, each optimisation pass with its measured
effect (including the ones that were reverted or measured slower), and the
readings of those numbers that were later corrected.

.. important::

   **Everything here is history.** Ratios against Gemmini on this page are
   against Gemmini's end-to-end ``tiled_matmul_auto`` numbers (574 / 615 / 740 /
   784 / 986), which include about 395 cycles of Rocket driver software. They
   are kept as a record of our own progress, **not as comparisons**. The
   like-for-like result -- **1.55-1.8x slower than Gemmini at all five shapes**
   -- is on :doc:`gemmini_comparison`.

Progression of the 16x16x16 cosim count, one build, bit-exact throughout:

.. list-table::
   :header-rows: 1

   * - build
     - 4x4x4
     - 16x16x16
     - section
   * - per-instruction loops nested
     - 1017
     - 1713
     - :ref:`tinytpu-history-rowflat`
   * - row-flattened (4 of 5 units)
     - 1004
     - 1586
     - :ref:`tinytpu-history-rowflat`
   * - ``wrap_io=False`` + burst DMA
     - 680
     - 1457
     - :ref:`tinytpu-history-burst`
   * - (+ flat accumulator, **reverted**)
     - 676
     - 1423
     - :ref:`tinytpu-history-lastloops`
   * - no memset + ``align_value`` widening (**current**)
     - **252**
     - **919**
     - :ref:`tinytpu-history-prefixes`


Removed predecessors
--------------------

``microarch_isa.py`` is the only design left in the tree. The designs it
superseded were removed on 2026-09-19 (``7a24c21e``, "prune main to one
accelerator design: TinyTPU-isa", 19 files); the last commit containing them is
``e2451b81``, and every file below is readable with
``git show e2451b81:examples/accelerator/<path>``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - path
     - what it was
   * - ``tinytpu_grid/microarch.py``, ``bench.py``
     - the first, single-grid machine: 36 PEs sharing ``A``, ``B``, ``imem`` and
       ``C``
   * - ``tinytpu_grid/repro/``
     - why it failed: the command-broadcast deadlock in the dataflow simulator
       (``README.md``, ``a_passes_no_imem.py``, ``b_hangs_with_imem.py``) and
       Vitis's refusal of the same fan-out (``vitis_csyn_errors.log``)
   * - ``tinytpu_grid/BACKEND_CHOICE.md``
     - the backend comparison (Vitis dataflow vs. chia RTLGen vs.
       SystemC/Catapult) that chose Vitis and forced the one-owner-per-array
       structure every later design keeps
   * - ``tinytpu_vitis/microarch.py``, ``bench.py``, ``RESULTS.md``,
       ``csyn_4x4x4.log``, ``csynth_4x4x4.rpt``
     - output-stationary feeder/drainer restructure; Vitis-legal, but
       ``acc += a*b`` is a loop-carried dependence so ``Final II = 7``
   * - ``tinytpu_vitis/microarch_ws.py``, ``bench_ws.py``, ``RESULTS_WS.md``,
       ``logs/csyn_int8_8x8x8.log``,
       ``logs/csynth_{int8_8x8x8,int8_16x16x16,fp32_8x8x8}.rpt``
     - weight-stationary, one opcode, II=1 per MAC at 100% of roofline
       (interval 74 at 8x8x8, down from 168 once the feeders and accumulator
       were partitioned) -- but *one* opcode, no scratchpad, no vector unit;
       operands stream from DRAM straight into the array. A fast fixed-function
       GEMM, not a programmable accelerator.

The Vitis diagnostic that every design since the grid satisfies by construction,
from ``tinytpu_grid/repro/vitis_csyn_errors.log`` on the 4x4x4 grid, where
``v728``/``v729``/``v730`` are ``A``, ``B``, ``imem`` (read by all 36 instances)
and ``v731`` is ``C`` (written by 16):

.. code-block:: text

   ERROR: [HLS 200-779] Non-shared array 'v730' failed dataflow checking:
                        it can only have a single reader and a single writer.
   ERROR: [HLS 200-979] Argument 'v731' failed dataflow checking:
                        it can only be written in one process function.


First working version
---------------------

Functional status
~~~~~~~~~~~~~~~~~

Before control flow was added (one-word instructions, a 6-bit opcode and five
fields, opcodes ``nop``, ``dma_ld``, ``vld``, ``mm``, ``vadd``, ``vrelu``,
``mvout``), with per-shape builds, on the Allo dataflow simulator:

.. list-table::
   :header-rows: 1

   * - shape
     - instrs
     - vadd runs
     - result
   * - 4x4x4 (Kt=1, Nt=1)
     - 6 / 7
     - no
     - **exact**, ``wrong=0/16``
   * - 8x8x8 (Kt=2, Nt=2)
     - 20 / 22
     - yes
     - **exact**, ``wrong=0/64``
   * - 8x16x8 (Kt=4, Nt=2)
     - 40 / 42
     - yes
     - **exact**, ``wrong=0/64``
   * - 16x16x16 (Kt=4, Nt=4)
     - 76 / 80
     - yes
     - **exact**, ``wrong=0/256``
   * - 32x16x16 (Kt=4, Nt=4)
     - 88
     - yes
     - **exact**, ``wrong=0/512``

Both ``gemm`` and ``gemm.relu`` were bit-exact against a numpy reference
*including the int8 clip* at every passing shape. Kt>=2 is the interesting case:
it is the first one where ``mm`` alone cannot finish a tile and ``vadd`` actually
runs. 8x8x8 is the first shape that exercises the whole ISA -- multiple k-tiles,
so ``vadd`` runs, and multiple n-tiles, so the accumulator is reused.

First synthesis and co-simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The csynth bound and the ``nr`` narrowing (91407 -> 4133) are on
:ref:`tinytpu-isa-csynth-bound`. Per unit at 8x8x8 after the narrowing
(``logs/csynth_isa_8x8x8.rpt``), 0 errors, ``dataflow``, 22 processes:

.. code-block:: text

   | + tinytpu_isa*  | latency 4133 | interval 3059 | dataflow | BRAM 42 | DSP 12 | FF 16247 | LUT 21918 |
   |  + sequencer_0  |         24   |               |          |
   |  + dma_ld_0     |       1519   |               |          |
   |  + spm_0        |       2034   |               |          |
   |  + vru_0        |       1866   |               |          |
   |  + pe_0_0       |       1629   |  ... 16 PEs, each its own module
   |  + accu_0       |       3058   |  <- the critical unit
   |  + dma_st_0     |       1497   |               |          |

``accu_0`` at 3058 was the whole top-level interval, so the vector unit -- not the
array -- was the thing to optimize next. DSP is 12 rather than the 96 of the
int8 ``microarch_ws.py`` build because int8 multiplies mapped into LUTs here;
that is a mapping difference, not a missing array (all 16 ``pe_i_j`` modules are
present in the report). The FIFOs are a constant depth 8, so the area figure is
one a real build would have.

The cosim flow (``cosim.py``) first passed at 8x8x8:

.. code-block:: text

   +----------+--------+------------------+
   |   RTL    | Status | Latency (cycles) |
   |  Verilog |  Pass  |       1449       |
   +----------+--------+------------------+
   TB: 8x8x8 gemm mismatches = 0 / 64
   C/RTL co-simulation finished: PASS

That 1449 was measured with ``imem`` declared ``[1024]``; sized to the program it
is **1003**. Measured numbers at that point, all exact: 4x4x4 **717**, 8x8x8
**1003**, 16x16x16 **2395**. Getting there needed three toolchain fixes, none of
them design changes (a real testbench, ``-B/usr/bin``, explicit ``m_axi``
depths), now documented in :doc:`/backends/vitis`.


The FIFO problem was the simulator's thread count
-------------------------------------------------

**Resolved, and it was not the design.** Earlier revisions called this "the
design's one unresolved problem" and sized ``QD`` from the program to work around
it. That was wrong.

**The design's channel graph needs depth 4.** ``kpn_model.py`` models the exact
channel structure -- every unit as a generator yielding blocking ``get``/``put``,
bounded FIFOs, a cooperative scheduler, and a deadlock report naming each
blocked process and the occupancy of the channel it waits on. It completes at
**depth 4 for every shape**, 4x4x4 through 16x16x16. So there is no circular
wait in the architecture at all.

**The Allo simulator needed one thread per process.** It gave each ``df.kernel``
instance an OMP thread and blocked that thread on an empty or full stream. With
fewer threads than processes, a blocked process could hold a thread that its own
producer needed, and the region wedged. This design has ``T*T + 6 = 22``
processes. At 16x16x16 with ``QD=16``:

.. list-table::
   :header-rows: 1

   * - ``OMP_NUM_THREADS``
     - result
   * - 8 (the value then in ``CLAUDE.md``)
     - **hang**
   * - 16
     - **hang**
   * - 24
     - pass
   * - 32
     - pass

The threshold sits exactly at the process count. With 32 threads the depth
requirement disappears: **16x16x16 passes at** ``QD=4`` -- matching the model --
and **32x16x16, which had never passed at any depth, passes at** ``QD=8``. So
the "required depth grows with the program" law was an artifact throughout. Deep
FIFOs were masking a thread-starvation deadlock by letting each producer run to
completion before anyone had to block. ``QD`` is now a constant **8**.

The simulator was fixed on 2026-09-17 (OpenMP team sized to the section count)
and gained a watchdog on 2026-09-18; see :ref:`limitation-11`.

**Four theories tested and disproved**, kept so they are not re-tried: the PE's
``put`` order (swapping changed nothing), a cycle in the process graph (two
repros, one with real traffic on every edge, both ran), the ``vld`` burst length
(chunking did not help), and the sequencer's control broadcast (rewritten as a
forwarding chain; exact, depth unchanged). The forwarding chain is kept because
control following the data path is the better structure, not because it fixed
anything.


Where the gap was, originally
-----------------------------

Fitting cycles against instruction count (6, 20, 76 instructions), per-shape
builds, against end-to-end Gemmini:

.. list-table::
   :header-rows: 1

   * -
     - fixed cost
     - marginal cost
   * - ours
     - 573 cycles
     - **24.0 cycles/instruction**
   * - Gemmini
     - 539 cycles
     - **5.9 cycles/instruction**

The reading then: the fixed costs are the same (573 vs 539), the entire gap is
marginal cost, and it is 4x. The mechanism is in the synthesis report. Every
unit's per-instruction loop:

.. code-block:: text

   o l_S_c_0_c    iter_latency=2   II=1   trip=80   pipelined=yes    <- sequencer
   o l_S_c_0_c1   iter_latency=69  II=-   trip=80   pipelined=no     <- dma_ld
   o l_S_c_0_c2   iter_latency=69  II=-   trip=80   pipelined=no     <- spm
   o l_S_c_0_c3   iter_latency=73  II=-   trip=80   pipelined=no     <- vru
   o l_S_c_0_c4+  iter_latency=74  II=-   trip=80   pipelined=no     <- the PEs

Only the sequencer's loop is pipelined. In every other unit the loop over
instructions is **not** pipelined, so a unit finishes instruction *n* before
starting *n+1*: **there is no inter-instruction overlap inside a unit.** The
units overlap with *each other* (that is what ``dataflow`` buys, and it is why the
measured 24 is far below the report's ~70), but within a unit instructions are
strictly serial.

This is the same root cause found on ``chia-codesign``, where a unit was a
``func.call`` the compiler would not pipeline across and consecutive
instructions overlapped by exactly zero (``FINDINGS_v2.md`` G.4). The structure
is much better here -- persistent processes, so the cost is 24 cycles rather
than a full pipeline fill and drain -- but the property is the same.

**Gemmini's answer is its reservation station.** ``ReservationStation.scala`` (48
entries: 8 ld / 16 ex / 4 st) exists precisely to have many instructions in
flight at once, issuing out of order when their operands do not overlap. Its
5.9 cycles/instruction is that hardware working. We are strictly in-order, by
construction, which is what makes the hazard logic free, and also what caps the
throughput.

The original "fairness note" of this period read: ours is the accelerator
alone; Gemmini's ``rdcycle`` figure includes RoCC dispatch from Rocket and its
own tiling loop, which "favours us slightly" and was noted rather than corrected
for. The 2026-09-18 window audit showed the included driver is about 413 of
574 cycles at 4x4x4 -- not slight -- and withdrew the comparison built on it
(:doc:`gemmini_comparison`).


T=16: scaling the array 16x bought 1.47x
----------------------------------------

The 16x16 build became possible only after the simulator's OpenMP team was sized
to the section count (:ref:`limitation-11`); before that fix a 262-instance
region hung silently. It builds and runs in **50 s**, 821 streams, all three
programs bit-exact, and cosim measures:

.. list-table::
   :header-rows: 1

   * -
     - cycles @16x16x16
     - PEs
     - roofline
     - utilization
   * - ours T=4
     - 1733
     - 16
     - 256 cyc
     - 14.8%
   * - **ours T=16**
     - **1176**
     - **256**
     - **16 cyc**
     - **1.36%**
   * - MiniTPU 16x16
     - 168
     - 256
     - 16 cyc
     - 9.5%
   * - Gemmini 4x4
     - 986
     - 16
     - 256 cyc
     - 26.0%

(Both "ours" rows predate the row-flattening pass, which took the T=4 number to
1586. T=16 has not been re-measured since; the argument this section makes is
about the fixed term, which flattening did not move.)

**16x the PEs bought 1.47x the speed, and utilization fell 14.8% -> 1.36%.**

- Array time fell from 256 cycles to 16. **Overhead went 1477 -> 1160, i.e.
  barely moved.** The overhead is the same absolute quantity at both array
  sizes, because it is data movement over a 16x16 operand set, which does not
  depend on T.
- At T=16 the program is **6 instructions**, not 45, so the marginal term --
  18.1 vs Gemmini's 10.8 cycles/instruction -- is almost entirely absent from
  this measurement. 1176 cycles for 6 instructions is not an issue-rate
  problem.
- Therefore **the fixed term was not one of two roughly equal problems; at a
  useful array size it was essentially the only problem.** An earlier revision
  listed "pipeline the per-instruction loop" and "program-controlled burst DMA"
  as comparable next steps. They were not. At T=4 the marginal term is visible
  because 45 instructions multiply it; at T=16 it nearly vanishes and 98.6% of
  the machine sits idle waiting for operands.

Resources at T=16: 240 DSP, 111,776 FF, 144,329 LUT (11% of the xcu280), 34
BRAM. All 256 PEs instantiated. csynth reports a top-level latency of
**2.259e+08** for this build, because the loop trip counts are runtime data and
it must bound them by the ISA's field widths. The real number is cosim's 1176.

This result is also one instance of a general trap, recorded on
:doc:`/developer/limitations`: an unused capability measures as a worthless one.
The bigger array was *unused*, not *worthless*.


.. _tinytpu-history-rowflat:

Row-flattening the per-instruction loop: 1.08x
----------------------------------------------

**The change.** Each unit was a loop over *instructions* containing a loop over
*rows*, and Vitis reported ``Pipelined = no`` on all five of those outer loops.
That is not a tool defect: modulo scheduling needs a fixed II, an inner loop
whose trip count arrives in an instruction field cannot be unrolled to give one,
and so the outer loop has no II at all. An independent check found Catapult will
not pipeline it either.

``dma_ld``, ``spm``, ``vru`` and ``dma_st`` became **one flat loop over rows**
(words, for ``vru``), with the instruction fetched on the iteration that needs it
and the opcode test surviving as a mux inside a pipelined body:

.. code-block:: python

   r = -1
   for x in range(n_work):         # ROWS (or words), from the header
       r += 1
       if r >= cnt:                # this row starts a new instruction
           w0 = c_unit.get(); <decode>; cnt = <work items>; r = 0
       <straight-line body, indexed by r>

The header carries a per-unit dynamic *work* count instead of an instruction
count; ``assemble()`` already expanded the control flow with ``expand()``, so this
is ``nr`` summed rather than counted.

**Measured, same flow, same testbenches, all five shapes bit-exact:**

.. list-table::
   :header-rows: 1

   * - shape
     - before
     - after
     - speedup
   * - 4x4x4
     - 1017
     - **1004**
     - 1.013x
   * - 8x8x8
     - 1145
     - **1108**
     - 1.033x
   * - 12x12x12
     - 1369
     - **1294**
     - 1.058x
   * - 16x16x8
     - 1417
     - **1344**
     - 1.054x
   * - 16x16x16
     - 1713
     - **1586**
     - **1.080x**

Fitting against dynamic instruction count (6, 15, 28, 25, 45): marginal
**18.07 -> 15.12** cycles/instruction, fixed **902 -> 907** (5 cycles is fit
noise), against Gemmini's (end-to-end) 10.8 / 483. Area essentially unchanged:
BRAM 16, DSP 15, FF 12432 -> 12835, LUT 18416 -> 18968.

Per-unit, before and after (csynth, the loop over instructions):

.. list-table::
   :header-rows: 1

   * - unit
     - before
     - after
   * - ``sequencer``
     - 1 loop, II=5
     - unchanged
   * - ``dma_ld``
     - iter latency 133, **Pipelined = no**
     - one loop, **yes, II=1**, iter latency 4
   * - ``spm``
     - iter latency 133, **no**
     - one loop, **yes, II=1**, iter latency 5
   * - ``vru``
     - iter latency 137, **no**
     - one loop, **yes, II=1**, iter latency 5
   * - ``accu``
     - iter latency 261, **no**
     - **unchanged -- see below**
   * - ``pe`` x16
     - iter latency 2055-2058, **no**
     - unchanged
   * - ``dma_st``
     - iter latency 132, **no**
     - one loop, **yes, II=1**, iter latency 3

**The marginal term moved 1.20x, not the 1.7x estimated, and the fixed term did
not move at all.** Two things account for the gap between the estimate and the
result:

1. **1.7x was the ratio of our marginal to Gemmini's, not the headroom in the
   change.** 18.1/10.8 = 1.68 is what closing the whole marginal gap would buy;
   pipelining the loop only removes the part of the marginal cost that is
   per-instruction pipeline fill, and the rest is real per-row work that no
   scheduling change touches.
2. **Amdahl.** At 16x16x16 the marginal term is 811 of 1713 cycles, 47%; at
   4x4x4 it is 108 of 1017, 11%. Even a marginal term driven to zero could not
   have given 1.7x overall at any shape in this sweep, and the measured speedups
   track that share exactly -- 1.3% at the smallest shape rising monotonically
   to 8.0% at the largest.

What would not split, and what would not flatten
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plan was per-opcode queues and one process per opcode. **One owner per
memory rules that out for exactly the units that matter** (``spad``, ``vr`` and
``ar`` each have arms on both sides; HLS 200-779 / 200-979). The two units that
own no memory, ``dma_ld`` and ``dma_st``, serve one opcode each, so there was
nothing to split there either -- the splitting half of the plan had no legal
instance anywhere in the design. Flattening turned out to get what the split was
wanted for without moving a memory, which is why it is the change that shipped.

.. note::

   **Correction, 2026-09-19.** The paragraph above is kept as the record of
   what was believed. The rule is Allo's, not Vitis's: with
   ``#pragma HLS stream variable=buf type=unsync`` Vitis shares an on-chip
   array between two processes, one per BRAM port (``HLS 200-824`` /
   ``200-755`` / ``200-634``), and ``HLS 200-779`` applies only to synchronized
   arrays. Allo rejects a region-scope ``Stateful`` shared by two kernels and
   never emits ``stream type=unsync``. The measured cost to this design is 0
   cycles. See :ref:`limitation-shared-memory` and
   :ref:`gemmini-gap-attribution`.

``accu`` **resisted the flattening.** It was built twice and was bit-exact both
times, and both times it was slower than the nested loop it replaced:

* With ``ar`` in BRAM, ``Final II = 3``. Once ``r`` is a carried register rather
  than the inner loop's induction variable, ``ar[f1+r]`` stops being affine in
  the loop index, and Vitis can no longer prove that iteration *n*'s store and
  iteration *n+1*'s load touch different rows. Deferring the write by one
  iteration, the textbook fix, only moves the violation onto the enable
  register.
* With ``ar`` completely partitioned into registers -- no ports to arbitrate, no
  aliasing to prove -- ``Final II = 2``, and no further. Read, add, write, and
  the next iteration may read what this one wrote is a genuine recurrence
  through a register file. It also cost 33k FF, 12k LUT and ten minutes of
  synthesis.

Two cycles a row for 384 rows is worse than 20 instruction boundaries plus 384
rows at one, so ``accu`` kept the nested loop and kept ``mm`` at II=1. The array
was left alone deliberately: a PE's per-``mm`` prologue is a different shape from
its MAC body, and folding them would set the II of the one loop in the design
that is already II=1 and carries the actual arithmetic.

**The general rule:** flattening trades a fixed per-instruction cost for a
permanent per-row II, and only pays where the II stays at 1. Four units kept it;
one did not, and did not.

Three Vitis-specific things were worth a factor of two each and are not obvious
from the source:

* ``vru`` emits a header word and T weight words per ``mm``. In the fetch branch
  that is T+1 writes to one FIFO in one iteration, and a FIFO takes one write
  per cycle, so the whole loop would schedule at II=T+1. It charges the prologue
  T+1 *iterations* instead -- the same cycles, II=1 everywhere else.
* **Increment the row counter at the TOP of the body.** With ``r += 1`` at the
  bottom, Vitis schedules the add in the last stage and the next iteration's
  conditional queue read depends on it -- a distance-1 recurrence that does not
  close in one cycle, ``Final II = 2`` on every unit. Hoisting it costs nothing.
* **Read the memory once, at an address the branch selects.** A read written
  into each arm -- the obvious transcription -- synthesizes to one read port per
  arm on one memory: ``vru`` came back at II=2 and ``accu`` at II=3.


.. _tinytpu-history-burst:

The burst DMA: the 907-cycle fixed charge, and what it actually was
-------------------------------------------------------------------

The fixed term had not moved all project: 907 cycles, 57% of 16x16x16 and 90% of
4x4x4, and row-flattening left it at 907 from 902. It was not a DMA cost at all
-- it was an argument-passing convention. ``wrap_io=True`` makes Allo hoist every
``m_axi`` argument into a local buffer before the region starts, via
``wrap_data_movement`` (``allo/ir/transform.py``), whose extent is
``MemRefType(arg.type).shape`` -- the STATIC type, with no offset and no length.
The csynth report names all four copies:

.. code-block:: text

   | m_axi_gmem0 | read  |  56 | 64 | l_S_load_buf0_load_buf0_l_0   |   # imem
   | m_axi_gmem1 | read  | 256 |  8 | l_S_load_buf1_load_buf1_l_0   |   # A
   | m_axi_gmem2 | read  | 256 |  8 | l_S_load_buf2_load_buf2_l_0   |   # B
   | m_axi_gmem3 | write | 256 |  8 | l_S_store_res3_store_res3_l_0 |   # C

824 words at II=1, copied whether the program touches them or not.

**The earlier verdict on** ``wrap_io=False`` **was wrong, and wrong in an
instructive way.** It measured fixed 481 / marginal 39.8 and concluded that
``m_axi`` is inherently slow. It was measured with the strided access pattern,
and the patterns synthesize to completely different hardware:

.. list-table::
   :header-rows: 1

   * - pattern
     - ``[HLS 214-115]`` says
     - loop II
   * - ``lA[(f1 + r) * MAXDIM + f2 * T + e]``, ``e`` unrolled
     - ``burst reads of length 4 and bit width 8``
     - 4
   * - ``imem[NHDR + pc * IWORDS]``, ``pc`` a register
     - ``burst reads of length 2 and bit width 64``
     - 13 (was 5)
   * - ``for i in range(n): b[i] = lA[i]``, ``n`` a runtime value
     - ``burst reads of variable length``
     - port-limited

So Allo could express a program-controlled burst DMA the whole time (this is the
retraction recorded as :ref:`limitation-13`). Two changes, both to access
patterns rather than to Allo:

* ``sequencer`` **prefetches the program.** One ``IMEM_SIZE``-word contiguous
  burst at start-up, then every fetch is a BRAM read. The fetch loop goes back to
  II=5. This was most of the old 39.8 -- ~76 dynamic sequencer iterations each
  paying full bus latency for two words.
* ``dma_ld`` **bursts each operand matrix once**, covering exactly the DRAM rows
  the program will name. The spans come from the assembler: ``expand()`` resolves
  the AGU exactly as the sequencer does and ``assemble()`` takes the maximum
  ``f1 + nr`` over the ``dma_ld``\ s of each source into ``imem[7]``. The bytes are
  packed into ``UInt(T*8)`` words as the burst sweeps, so the instruction loop
  does one BRAM read per row and no packing at all, back at II=1.

Resolving the AGU in ``expand()`` also made ``bench_isa``'s loop-vs-flat check
strict: the two program forms must agree on every resolved address field, not
just on the opcode and row-count stream.

Bursts inferred, which is the thing to check -- a change that does not change
the 214-115 message has not worked:

.. code-block:: text

   | m_axi_gmem0 | read  | 56       | 64 | l_S_i_0_i        |   # imem, one burst
   | m_axi_gmem1 | read  | variable |  8 | VITIS_LOOP_596_3 |   # A,    one burst
   | m_axi_gmem2 | read  | variable |  8 | VITIS_LOOP_656_4 |   # B,    one burst
   | m_axi_gmem3 | write | 4        |  8 |                  |   # C,    unchanged

Per-unit ``Pipelined`` / II, ``wrap_io=True`` before against ``wrap_io=False`` +
bursts after:

.. list-table::
   :header-rows: 1

   * - unit
     - loop
     - before
     - after
   * - ``sequencer``
     - fetch/dispatch
     - yes, II=5
     - yes, **II=5**
   * - ``sequencer``
     - imem prefetch
     - (hoisted, II=1)
     - yes, **II=1**, trip 56
   * - ``dma_ld``
     - operand prefetch
     - --
     - yes, **II=4** (8-bit port)
   * - ``dma_ld``
     - instruction rows
     - yes, II=1
     - yes, **II=1**
   * - ``spm``
     - rows
     - yes, II=1
     - yes, II=1
   * - ``vru``
     - words
     - yes, II=1
     - yes, II=1
   * - ``pe`` x16
     - per-``mm``
     - no, iter 2058
     - no, iter 2058
   * - ``pe`` x16
     - MAC
     - yes, II=1
     - yes, II=1
   * - ``accu``
     - per-instruction
     - no, iter 261
     - no, iter 261
   * - ``accu``
     - ``mm`` / ``vadd`` / ``vrelu`` / ``mvout``
     - II=1 / 2 / 2 / 1
     - unchanged
   * - ``dma_st``
     - rows
     - yes, II=1
     - yes, **II=4**

**Nothing lost its pipelining.** ``dma_ld``'s instruction loop keeps II=1 because
the burst is a separate loop outside it. ``dma_st`` is the one regression, II 1
-> 4, and it is the bus rather than the schedule: it now writes ``C`` over
``m_axi`` four bytes at a time instead of into a buffer someone else stores back.
M_AXI table, unchanged by any of this: all four ports 8/64-bit, Max Read and
Write Burst Length 16, Num Read and Write Outstanding 16. A 56- or 256-beat
burst is issued as requests of 16, at II=1 each.

Cosim, one build, all five shapes bit-exact:

.. list-table::
   :header-rows: 1

   * - shape
     - dyn. instrs
     - before
     - after
     -
   * - 4x4x4
     - 6
     - 1004
     - **680**
     - 1.48x
   * - 8x8x8
     - 15
     - 1108
     - **831**
     - 1.33x
   * - 12x12x12
     - 28
     - 1294
     - **1066**
     - 1.21x
   * - 16x16x8
     - 25
     - 1344
     - **1139**
     - 1.18x
   * - 16x16x16
     - 45
     - 1586
     - **1457**
     - 1.09x

Least squares against dynamic instruction count: **fixed 907 -> 557, marginal
15.12 -> 20.07 cycles/instruction.** That is a trade, and it is stated as one.
The crossover is at ``350 / 4.95 = 71`` dynamic instructions, past the longest
program this MAXDIM admits (45), so the burst build wins everywhere it can be
run -- but at a larger MAXDIM it would not, without also fixing what the
marginal term buys. The +5 cycles/instruction was mostly ``dma_st``'s strided
four-byte writes.

Two negative results
~~~~~~~~~~~~~~~~~~~~

* **Halving the operand burst bought nothing.** Merging the A and B bursts into
  one loop bounded by ``max(na, nb)`` does halve the burst time, and it changed
  the cosim count at all five shapes by exactly **zero cycles**. It was built,
  measured, and reverted to the version that reads the fewest bytes. (The reason
  it bought nothing turned out to be the hidden ``spad`` memset, not that operand
  traffic does not matter -- see :ref:`tinytpu-history-prefixes`.)
* **Port widening looked blocked.** The ``m_axi`` ports were 8 bits, so even a
  perfect burst moved one byte per cycle, and
  ``config_interface -m_axi_max_widen_bitwidth 512`` appeared to do nothing. This
  was first attributed to ``[HLS 214-307]``; that attribution was later
  corrected (:ref:`tinytpu-history-prefixes`).

The I/O trade, as it was tabulated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``wrap_io`` was for a long time an architectural choice with a crossover. Ratios
are against end-to-end Gemmini:

.. list-table::
   :header-rows: 1

   * - config
     - marginal
     - fixed
     - 4x4x4
     - 16x16x16
   * - ``wrap_io=True``, imem 256
     - 18.1 cyc/instr
     - 1102
     - 2.12x
     - 1.94x
   * - ``wrap_io=False``, strided
     - 39.8
     - 481
     - 1.21x
     - 2.25x
   * - ``wrap_io=True``, imem 56
     - 15.1
     - 907
     - 1.75x
     - 1.61x
   * - ``wrap_io=False``, bursts
     - 20.1
     - 557
     - 1.18x
     - 1.48x
   * - (+ flat accumulator, reverted)
     - 19.3
     - 563
     - 1.18x
     - 1.44x
   * - **+ no memset, + widened AXI**
     - **17.3**
     - **151**
     - **0.44x**
     - **0.93x**
   * - Gemmini (end-to-end)
     - 10.8
     - **483**
     - 1.00x
     - 1.00x

The ``0.44x`` / ``0.93x`` row is the one the withdrawn "faster than Gemmini"
claim was read from; against Gemmini's like-for-like window the same build is
1.55-1.8x slower (:doc:`gemmini_comparison`).


.. _tinytpu-history-lastloops:

The last two unpipelined loops, and which one paid
--------------------------------------------------

After the burst DMA the only loops left with ``Pipelined = no`` were the array's
per-``mm`` loop and ``accu``'s per-instruction loop. Both were attacked; neither
landed.

``accu``: flat at II=1 with a write-behind rotation -- REVERTED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Not in the design.** It was built (``ca978b97``), it was bit-exact, it was 2.3%
faster, and it was reverted for area (``644d8cdc``, 2026-09-18): the rotation
costs ``accu`` 13.7x its flip-flops (1,270 -> 17,450), the trade gets worse as T
grows, and one line of ``#pragma HLS dependence`` would have bought the same II
for nothing. The measurement is kept because it is what prices
:ref:`limitation-21` -- the number that says what the missing pragma is worth in
silicon. ``microarch_isa.py`` is back at the nested form, identical to the
burst-DMA build (``9c6609e2``) except for the docstring that records this
attempt.

The obstacle was one dependence: read ``ar[f1 + r]``, add, write it back, with
``r`` a carried register. In BRAM that is

.. code-block:: text

   Unable to enforce a carried dependence constraint (II = 1, distance = 1)
   between 'store' on array 'ar' and 'load' ('rv') on array 'ar'

``Final II = 3``; completely partitioned into registers it is ``Final II = 2``,
no further. Vitis will take ``#pragma HLS dependence variable=ar inter false``
for exactly the first of those, and **Allo has no primitive that emits one**. So
the proof had to be made unnecessary instead of waived: ``accu`` kept the last
two computed rows in registers, wrote ``ar`` two iterations late, and answered a
read that landed inside that window from the registers. The loop-carried path
became

.. code-block:: text

   adder -> rotation register -> bypass mux -> adder

with the memory off it. ``Final II = 1, Depth = 6``, one iteration per row for all
four opcodes. ``ar`` stays completely partitioned. This is the shape Gemmini's
output-stationary PE uses for the same reason (two accumulators ``c1``/``c2``),
applied to a register file. ``vadd``'s second ``ar`` read, which forced its own
loop to II=2 under a dual-port BRAM, came free from the same partition.

.. list-table::
   :header-rows: 1

   * - shape
     - dyn. instrs
     - nested (**shipped**)
     - rotated (reverted)
     -
   * - 4x4x4
     - 6
     - **680**
     - 676
     - -4
   * - 8x8x8
     - 15
     - **831**
     - 827
     - -4
   * - 12x12x12
     - 28
     - **1066**
     - 1062
     - -4
   * - 16x16x8
     - 25
     - **1139**
     - 1125
     - -14
   * - 16x16x16
     - 45
     - **1457**
     - 1423
     - -34

Fixed cost 557 -> 563, marginal **20.07 -> 19.32** cycles/instruction. Area:
``accu`` FF 1270 -> 17450 and LUT 2465 -> 6430, top-level FF 11524 -> 27377, LUT
19235 -> 23239, BRAM 16 -> 12. All five shapes bit-exact, ``vadd``/``vrelu``
included.

**The verdict: 2.3% for 13.7x the flip-flops in that unit, and 2.4x at the top
level.** At T=4 that is already a bad trade, and ``ar`` is ``T``-by-``MAXDIM``,
so the register cost grows with the array while the 2.3% does not. What this
priced is not the rotation -- it is the pragma Allo cannot emit.

The per-instruction boundary was real but mostly hidden. It is about 8 cycles --
a decode, the call into the sub-function Vitis extracts each inner loop into,
and that loop's fill and drain -- so 20 accu instructions at 16x16x16 predicted
~160 cycles and delivered 34. ``vru`` (464 words) and ``dma_st`` (II=4 on 64
rows) are in front of it.

The array: flattened to II=1, and measured SLOWER
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**First, the report was being misread.** The PE outer loop's 2058 iteration
latency is not the prologue -- the MAC sub-loop alone is 2051 (trip 2047 at II=1,
the worst case the 12-bit row count admits). The prologue is 4-7 cycles, and the
sub-function call plus the MAC loop's own fill add ~5 more, so the per-``mm``
cost really at stake was ~12 cycles, ~190 at 16x16x16.

A PE flattens the way ``vru`` does: charge it one iteration per word it receives,
``nw + 1`` where ``nw`` is ``T - i`` down column 0 and 1 from the west, so the body
is straight-line and the MAC path can stay at II=1. Three successive II=2 results
had to be cleared:

.. list-table::
   :header-rows: 1

   * - build
     - ``Final II``
     - why
   * - a ``get`` in the header arm and another in the weight arm
     - 2
     - a read port per arm on one FIFO
   * - phase counter rearmed at the bottom of the compute arm
     - 2
     - carried dependence, ``select`` (``pro``) -> ``fifo read``
   * - row counter armed on the header word
     - 2
     - the FIFO output gates the next FIFO read
   * - latch on the header, arm on the LAST prologue word
     - **1**
     - ``nw`` is a constant >= 1, so the two are different iterations and what
       arms the counter is a register

All 16 PEs then pipeline at II=1 and no ``Pipelined = no`` is left in the array.
Cosim:

.. list-table::
   :header-rows: 1

   * - shape
     - baseline
     - flat PE, II=2
     - flat PE, II=1
   * - 4x4x4
     - 680
     - 686
     - **679**
   * - 8x8x8
     - 831
     - 850
     - 844
   * - 12x12x12
     - 1066
     - 1133
     - 1076
   * - 16x16x8
     - 1139
     - 1254
     - 1149
   * - 16x16x16
     - 1457
     - 1664
     - 1467

**A wash at 4x4x4 and about +10 everywhere else, so it was not landed.** The II=2
column costs +207 at 16x16x16 for +256 array cycles, i.e. **the array's
throughput is worth ~0.8 cycles of runtime per cycle of MAC**. The II=1 column
removes ~190 cycles of per-``mm`` overhead and buys *nothing*, because ``vru``
upstream spends the same ``T + 1`` words per ``mm`` pushing the header and the
weights at II=1 whatever the PE does -- the PE was never the thing waiting. What
is left is the flat body's deeper pipeline, 5-6 stages against the MAC loop's 4,
which costs about 10 cycles of extra latency through the T-deep chain. So:
**array throughput matters and array per-instruction overhead does not.** The
flat PE is written up in ``pe``'s docstring in full, including the three II=2
traps.


.. _tinytpu-history-prefixes:

The two hidden prefixes, and the silent refusal to widen
--------------------------------------------------------

The largest single step in the design's history (``b4be2b10``, "1457 -> 919: two
beliefs in this repo were wrong, and both were load-bearing"), and it came from
correcting two recorded beliefs rather than from a new idea.

Belief 1: "the fixed cost is essentially closed"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It was not closed, it was **hidden**. Two serial prefixes of nearly equal length
ran beside each other, so removing either alone measured as nearly worthless:

* ``spm`` opened with a **514-cycle zero-fill of** ``spad``. Allo lowers
  ``spad: UInt(VW)[SPAD_ROWS] = 0`` through ``linalg.fill``
  (``allo/ir/builder.py``), and that becomes a real memset loop in the RTL. A
  bare annotation with no ``= 0`` emits none.
* ``dma_ld``'s operand burst ran **~512 cycles** at II=4 alongside it.

This is the mechanism behind the earlier "merging the A and B bursts changed the
cycle count by exactly zero". That measurement was true and the conclusion drawn
from it -- that operand traffic did not matter -- was false. The burst was hidden
behind the memset, so halving it changed nothing.

Removing the ``= 0`` from six arrays (``ib``, ``rbA``, ``rbB``, ``spad``, ``vr``,
``ar``) is worth **-168 cycles alone**. Paired with widening it is worth
**-538**, which is strongly super-additive and is the fingerprint of two hidden
prefixes: you have to remove both before either shows up.

**This is an ISA semantic change, not a six-character optimisation.** ``ar`` is
the accumulator. Zero-filled, the hardware guaranteed a clean accumulator;
un-filled, **the program must write before it reads**. Every program here does,
and all five shapes are bit-exact -- but that is evidence about these programs,
not a proof about all of them. Gemmini has the same property (``mvin`` to the
accumulator carries an overwrite/accumulate bit, so the program owns the
initial state), so this moves toward its semantics rather than away, but it is
a contract change and belongs in the ISA documentation.

Belief 2: "HLS 214-307 blocks widening"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**It does not reproduce on the real design.** ``config_interface
-m_axi_max_widen_bitwidth 512`` is accepted, csynth completes, and there are
*zero* 214-307 messages. The ports just stay at bit width 8 with no diagnostic
-- which is worse than an error, because nothing tells you. 214-307 was a
standalone probe's behaviour, quoted in three tracked files as a whole-design
fact (the probe results are :ref:`limitation-23`).

The diagnosis under it was right: no alignment attribute, so Vitis assumes one
byte and declines to widen. With ``align_value`` emitted
(``configs={"align_value": 64}``, :doc:`/backends/vitis`), the same setting gives
gmem0 **bit width 512**, gmem1/2 32, and takes ``dma_ld``'s burst loop and
``dma_st`` from II=4 to II=1. Worth **-177 cycles alone**. Verified in the
build's ``csynth.log``: one port at 512, two at 32, zero 214-307.

Measured, one build, bit-exact at every shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - shape
     - before
     - after
   * - 4x4x4
     - 680
     - **252**
   * - 8x8x8
     - 831
     - **383**
   * - 12x12x12
     - 1066
     - **591**
   * - 16x16x8
     - 1139
     - **667**
   * - 16x16x16
     - 1457
     - **919**

Mismatches 0/16, 0/64, 0/144, 0/128, 0/256. Fixed cost **557 -> 151**; marginal
**20.07 -> 17.28** cycles per dynamic instruction. Cosim ran with
``-m_axi_latency 0``; the latency sensitivity is on :doc:`gemmini_comparison`.

**These are not faster than Gemmini.** An earlier revision set this table against
Gemmini's 574 / 615 / 740 / 784 / 986 and called it a lead at all five shapes,
with a 3.2x fixed-cost win. Both claims are withdrawn (:doc:`gemmini_comparison`).

What ``dma_st``'s II=4 actually was
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Answered by a unit-level probe matrix, and it was none of the usual suspects:

.. list-table::
   :header-rows: 1

   * - variant
     - II
   * - strided int8
     - 4
   * - strided + widen pragma
     - 4
   * - **contiguous int8**
     - **4** -- so the stride was never the cause
   * - strided + ``align_value(64)`` only
     - 4
   * - strided + one 32-bit store per iteration
     - **1**
   * - strided int8 + align + widen
     - **1**

It was **element width**: four scalar byte accesses per iteration through a port
serving one per cycle. Subsumed by the change above; no separate work. A design
that wants the same effect without an Allo change can declare the operands
``UInt(32)`` instead, which reaches II=1 with no alignment and no widen setting
at all.


Superseded readings and plans
-----------------------------

These are the conclusions and to-do lists as they stood at the burst-DMA build
(680 ... 1457), before the memset/widening pass and before the Gemmini window
audit. They are kept because later text refers to them; each is annotated with
what happened.

"Honest reading", at the burst-DMA build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    On one fixed build, matched in data type and array size, the design is
    **1.18x behind Gemmini at 4x4x4 and 1.48x at 16x16x16** across five
    shapes. The ratio now *rises* with problem size, where it used to fall: the
    fixed charge that dominated the small shapes has been removed, and what is
    left is a per-work gap. Utilization tracks at roughly two thirds of
    Gemmini's at the largest shape.

    Two terms make up the gap: **marginal, 20.1 vs 10.8 cycles/instruction
    (1.9x), and it went UP** -- ``dma_st``'s four-byte writes to ``C``,
    ``accu``'s read-modify-write accumulate, which does not flatten below II=2,
    and the array's per-``mm`` prologue; and **fixed, 557 vs 483 cycles
    (1.15x)** -- the region's own start-up and drain plus the 56-word
    instruction prefetch. The priority has inverted: the fixed term is 1.15x and
    the marginal term is 1.9x, so the next pass belongs on the marginal side,
    starting with the write path.

    Neither term is about the array, the data type, or the dataflow -- all three
    are correct and RTL-verified, and ``microarch_ws.py`` reached exactly 100% of
    roofline with the same PE structure. The compute is not the limit. Gemmini's
    26% at 16x16x16 is the number to chase, and at 17.6% we are now two thirds of
    the way there.

*Status:* the Gemmini figures in this reading are end-to-end
``tiled_matmul_auto``. The fixed-term reading (557 vs 483) was superseded twice:
the memset/widening pass took ours to 151, and the window audit showed Gemmini's
483 is mostly driver software. The like-for-like result is 1.55-1.8x slower.

"What would close the rest"
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Pipeline the per-instruction loop** -- *done*, measured at 1.08x rather than
   the 1.7x estimated (:ref:`tinytpu-history-rowflat`).
2. **A program-controlled burst DMA** -- *done*, 1.48x at 4x4x4 falling to 1.09x at
   16x16x16 (fixed 907 -> 557). Allo did not need extending
   (:ref:`tinytpu-history-burst`).
3. **The write path.** ``dma_st`` still writes ``C`` four bytes at a time; both
   ways of bursting it (clobber the unnamed columns, or defer the write-back to
   the end, where it serializes behind the last ``mvout``) cost something real,
   so it had to be measured rather than assumed. *Status:* the probe matrix
   above showed the cause was element width, and the widening pass took
   ``dma_st`` to II=1.
4. **A wider bus.** The ``m_axi`` ports are 8 bits. ``m_axi_max_widen_bitwidth``
   is blocked on Allo emitting an alignment attribute on the argument pointers.
   *Status:* done -- Allo now emits ``align_value`` (opt-in).
5. **Then multiple instructions in flight.** A credit/scoreboard scheme over the
   in-order units would approach Gemmini's 48-entry reservation station without
   its complexity. *Status:* open.

Splitting ``accu`` (giving the vector ALU its own issue slot) was addressed
differently: the GEMM inner loop no longer uses the vector ALU at all.

``RESULTS_ISA.md`` "Next" list, same period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. The write path (see item 3 above).
2. An alignment attribute on the emitted argument pointers -- an Allo codegen
   change, not a design one. *Status:* done.
3. **Re-measure T=16.** The last T=16 number (1176 at 16x16x16) predates both
   row-flattening and the burst DMA, and at T=16 a DRAM row is one packed word,
   so the operand burst is the whole matrix with no column overfetch at all.
   *Status:* open.
4. Then multiple instructions in flight. *Status:* open.
