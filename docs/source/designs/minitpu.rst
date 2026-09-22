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

#########################################
MiniTPU: Reference Architecture
#########################################

MiniTPU is the reference architecture the TinyTPU-isa work benchmarked
against (see :doc:`/designs/gemmini_comparison`). Originally titled "MiniTPU --
reference architecture we benchmarked against".

**Not our design, and not ours to modify.** MiniTPU is another engineer's
machine; its source is ``~/core/minitpu`` (live) and ``~/core/npu`` (read by an
earlier pass, now empty), and neither travels to zhang-21. This records what we
*found* about a reference architecture we benchmarked against, because the
findings outlive the tree. Everything was read-only. Distilled from two rescued
scratchpad reports; *(verified)* means re-checked against ``~/core/minitpu`` on
2026-09-18.

1. Shipped-bitstream QoR (xczu7ev-ffvc1156-2-e, Vivado 2023.2)
--------------------------------------------------------------

Every figure here is a **board measurement of a named bitstream**, reported by
MiniTPU's owner. Two builds are kept, because §2's noise-floor argument needs
two to compare; the bitstream id is part of the figure.

.. list-table:: Current: bitstream ``0xB1FF21F4``, 187.5 MHz, verified current against MiniTPU's RTL at its HEAD
   :header-rows: 1
   :widths: 24 38 38

   * -
     - value
     - note
   * - CLB LUT
     - 194,459 (84.4 %)
     - LUT is the scarce resource, not DSP
   * - CLB registers
     - 123,294
     -
   * - DSP
     - 838
     -
   * - WNS
     - +0.053 ns at 200 MHz
     -
   * - Power
     - 7.880 W on-chip (vectorless)
     -
   * - GPT-2 body throughput
     - **160.60 tok/s at 32 rows, 158.04 at 256** — medians of five runs,
       2026-09-22. 28.4 GFLOP/s, **29.6 % of the 96 GFLOP/s array peak**
       (was 24.2 %).
     - **Confirmed on a second board**: .189 measured 160.43 and 158.40 on the
       same commit, agreeing within 0.4 %. Supersedes the earlier
       79.16 / 129.70 pair — **the hardware did not change**; see "The
       concession that became the win" below.
   * - Kernel checks
     - 17/17 GPT-2, 6/6 Qwen
     - batch invariance bit-identical for ``B = 1..16`` — read with the
       row-wise qualifier in §4

.. list-table:: Earlier: bitstream ``0xB182E69D``, read 2026-09-18 under ``board_package/minitpu/qor/``
   :header-rows: 1
   :widths: 24 38 38

   * -
     - value
     - source
   * - CLB LUT
     - 191,231 (83.00 %)
     - ``utilization_report.txt``
   * - CLB registers
     - 118,684 (25.76 %)
     - same
   * - DSP
     - 838 (48.50 %)
     - same
   * - BRAM tile
     - 147 (47.12 %)
     - same
   * - URAM
     - 15 (15.63 %)
     - same
   * - WNS
     - +0.114 ns on ``clk_pl_0``, 5.000 ns constraint
     - ``timing_summary.txt``
   * - Power
     - 8.255 W on-chip (vectorless)
     - ``power_report.txt``

BRAM and URAM were not re-reported for ``0xB1FF21F4``; the earlier build's
figures are the only ones on record for them, and are not restated as current.

**Every board number was taken at 187.5 MHz, not the 200 MHz the bitstream is
constrained to.** PL0 is set by hand (``scripts/set_pl_clock.sh``,
``PL0_REF_CTRL = 0xFF5E00C0``); without it the PL runs at 99.999 MHz and every
check still passes, and ``clk_summary`` misreports it.
``tools/dma_fence_report.py`` hardcodes ``CLOCK_HZ = 187.5e6``. Any cycles→seconds
or GFLOP/s comparison must use 187.5 MHz; 200 MHz overstates it by 6.7 %.

A stale README power figure we reported, and the value it was corrected to,
are in `Earlier measurements and corrections`_.

2. The QoR noise floor — the most reusable finding here
-------------------------------------------------------

**Two builds of an identical netlist differed by −1407 LUT and −0.046 ns.**

So in this flow **no QoR delta below roughly 1.5k LUT (~0.6 % of the device) or
50 ps is evidence of anything** — place-and-route seeding alone covers it. That
is a fact about the tool flow, not about MiniTPU, and it applies to us:

- It is the standard behind commit ``644d8cdc``: the accumulator rotation bought
  2.3 % end-to-end for 13.7× the flip-flops in one unit — cost far outside the
  noise band, benefit barely outside it. Reverted.
- Every future Catapult PPA claim needs a delta above this band, or repeated
  builds with different seeds. A single-build A/B below it is a coin flip
  reported as a result.

Provenance: the ±1407 pair came from rescued build logs that did not survive —
the nine ``~/core/minitpu*`` worktrees as of 2026-09-18 all reported 191,231 /
+0.114 ns (the ``0xB182E69D`` build in §1), so it cannot be re-derived. The
magnitude is the load-bearing part.

3. Architecture
---------------

The shape of the machine, read out of ``~/core/minitpu/src/`` at the same HEAD
as the *(verified)* rows below. Parameter names are the RTL's; the values are
the shipped configuration (``NUM_LANES = 16``, ``DATA_WIDTH = 16``,
``NUM_VREGS = 32``):

.. code-block:: text

   host (AXI-Lite) -> command unit -> IRAM loader -> IRAM
                                                       |
                                                       v
   +--------------------------------------------------------------+
   | sequencer   fetch; STACK_DEPTH = 4 loop stack; LB_CAP = 24   |
   |             bundle loop buffer; stalls only on DELAY,        |
   |             matrix-busy and halt-drain -- no interlocks      |
   +--+--------------+--------------+-------------+--------+------+
      | V            | M            | MEM         | S      | C
      v              v              v
   +----------+  +------------+  +-----------+
   | VPU      |  | MXU        |  | vld / vst |
   | ALU, SFU |  | DIM x DIM  |  | vmemld /  |
   | xlu      |  |  = 16 x 16 |  | vmemst    |
   +-----+----+  | BF16 PEs,  |  +-----+-----+
         |       | MXU_ACC_W  |        |
         |       |  = 24 bit  |        |
         |       +-----+------+        |
         |             | per-lane      |
         |             | output FIFO   |
         |             v               |
         |      +---------------+      |
         +----->| VREG file     |<-----+
                | 3R1W, 32 regs |
                | ONE write     |  <-- the measured bottleneck,
                | port          |      not the array
                +-------+-------+
                        |
                        v
                +---------------------+       +-----------+
                | VMEM, one flat word |       | DMA,      |
                | array; compute port |<----->| 2 async   |<--> DRAM
                | and DMA port, whole |       | channels  |
                | words only          |       +-----------+
                +---------------------+

One VMEM word is one VREG: ``NUM_SUBLANES * NUM_LANES * DATA_WIDTH`` = 128 B,
which the DMA names as ``NUM_SUBLANES`` = 4 beats of 32 B. That is the only
role ``NUM_SUBLANES`` plays in addressing -- see the VMEM correction below.

+---------------+----------------------------------------------------------------------------------+
|               |                                                                                  |
+===============+==================================================================================+
| Array         | 16×16 weight-stationary BF16 PEs; ``psum_in`` zero only at row 0, so it          |
|               | accumulates sixteen **terms** deep in hardware — deeper is a BF16 ``vadd`` in    |
|               | the VPU. **Not an exactness guarantee**: per MiniTPU's owner, only the first     |
|               | term is exact (via a zero bypass) and every later add rounds.                    |
|               | Their arithmetic semantics are documented by them at                             |
|               | ``core/minitpu/docs/ARITHMETIC.md``, which is the source of record.              |
+---------------+----------------------------------------------------------------------------------+
| Accumulator   | **24-bit float** per PE (sign + 8 exp + 15 frac, ``MXU_ACC_W``), not fixed point |
+---------------+----------------------------------------------------------------------------------+
| Bundle        | **128-bit VLIW**: slots V / M / MEM / S / C, shared immediate, **7-bit DELAY**   |
|               | (``DELAY_W = 7``, ``sequencer_pkg.sv:23``) *(verified)*. MEM's two kinds, LDST   |
|               | (``vld``/``vst``) and DESC (``vmemld``/``vmemst``), cannot share a bundle        |
+---------------+----------------------------------------------------------------------------------+
| Hazards       | **none in hardware.** Fetch stalls only for DELAY, matrix-busy, halt-drain; VREG |
|               | RAW, write-port collisions, WAW, SREG RAW, VMEM ownership are the assembler's    |
|               | job (``asm.py schedule()``)                                                      |
+---------------+----------------------------------------------------------------------------------+
| Loops         | **4-deep hardware stack**, one ``{body_start, iv, hi, step}`` frame per level    |
|               | (``STACK_DEPTH = 4``, ``sequencer_loop_ctrl.sv``) *(verified)*                   |
+---------------+----------------------------------------------------------------------------------+
| Replay        | **24-bundle** loop buffer (``LB_CAP = 24``), innermost loop only; a longer body  |
|               | refetches from IRAM each iteration — time, not correctness                       |
+---------------+----------------------------------------------------------------------------------+
| AGU           | **1-term, shift-only**: ``agu_valid``, 2-bit ``agu_level``, 3-bit ``agu_shift``  |
|               | — one loop-index term, shifted, no multiply. DMA bypasses it, walking its own    |
|               | base/stride                                                                      |
+---------------+----------------------------------------------------------------------------------+
| Fences        | ``wait.channel(mask)`` over two async DMA channels; ``halt`` waits for DMA idle, |
|               | **not** for VREG writes in flight                                                |
+---------------+----------------------------------------------------------------------------------+
| IRAM          | self-fetching: host loads bundles over AXI-Lite via a command unit and IRAM      |
|               | loader, then the core fetches for itself                                         |
+---------------+----------------------------------------------------------------------------------+
| Silent limits | MXU output FIFO 32/lane — **overflow drops results, no replay**; loop buffer 24; |
|               | DELAY width. Checked by ``gen_isa_doc.py --check`` and ``tb_isa_conformance.sv`` |
|               | against RTL localparams                                                          |
+---------------+----------------------------------------------------------------------------------+

.. note::

   **One row above no longer matches the tree.** "MXU output FIFO 32/lane" is
   the 2026-09-18 reading, and §7 records both source trees agreeing on 32.
   ``vpu_pkg.sv`` now defaults ``MXU_OUTPUT_FIFO_DEPTH`` to **64**
   (``MINITPU_MXU_OUTPUT_FIFO_DEPTH``), and the parameter is the thing that
   must match ``docs/isa_latency.json`` or the board overflows it silently.
   Stated rather than restated, because this is another team's design and
   their tree is the record: the depth to quote is whatever
   ``MINITPU_MXU_OUTPUT_FIFO_DEPTH`` is at the commit being discussed.

The bundle's fixed bit layout, from ``sequencer_pkg.sv``'s
``encoded_bundle_t``, which is declared MSB first so that the struct *is* the
layout (cells not to scale; ``BUNDLE_WIDTH = 128``, of which 113 bits carry a
slot):

.. code-block:: text

   +-------+-------+-------+-------+-------+-------+-------+-------+
   |   V   |   M   |  MEM  |   S   |   C   |  IMM  | DELAY | rsrvd |
   |127:108|107:100| 99:66 | 65:53 | 52:46 | 45:22 | 21:15 | 14:0  |
   +-------+-------+-------+-------+-------+-------+-------+-------+
   bits:  20      8      34      13       7      24       7      15

``IMM_W = 24`` is the one shared constant -- S's immediate, ``loop.begin``'s
payload, or a descriptor's displacement -- and **exactly one slot may claim it
per bundle**; ``encode_bundle`` fails the build if two do, because the losers
would silently read the winner's constant. ``DELAY_W = 7`` holds issue 0..127
cycles, sized so one field covers the MXU's 82-cycle result latency.

Why BF16: accumulator width was chosen first from the LUT/DSP budget, and an
8-bit-exponent format is what fits it — "there is no FP32 state anywhere in this
design… fp32's extra 8 bits could only ever hold zeros there"
(``minitpu_config_pkg.sv``). Why no interlocks: "a legal schedule is a correctness
argument, and the assembler carries it" (README) — paid for with two silicon bugs
nothing caught at runtime.

VMEM is one flat array, not four banks per lane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rescued scaling report describes VMEM as **4 banks per lane**
(``bank = row mod NUM_SUBLANES``, conflict cost ``gcd(stride,4)``, an assembly-time
conditional latency) and stakes its whole staged plan on it. **Wrong for the
live design.** ``src/core/vpu/vpu_word_array.sv`` is a **single flat dual-port
URAM array** — one compute port, one DMA port, whole words only — with **no
arbiter and no interlock**; its own comment: "same-word collisions across the
two ports are undefined, as on the FPGA" *(verified)*. No banking, no conflict
cost model, nothing to model as a conditional latency. Do not build against the
banked model.

4. Measured performance, and the bound that is not a measurement
----------------------------------------------------------------

Cycle model — a **simulator** number from MiniTPU's own
``tools/gemm_cycle_model.py``, not a board figure; ``n`` output column tiles,
``B`` contraction blocks of 32. The current fit, per MiniTPU's owner,
re-measured over 20 points with **maximum residual 0**:

.. code-block:: text

   cycles = 46 + 281·n + 64.5·B + 111·n·B

The fit this page carried before the ``vmatpop`` occupancy correction
(``72c8b78``) is kept in `Earlier measurements and corrections`_.

The **matrix step is 52 cycles**, and the bottleneck is the **write port of the
3R1W vector-register file** — port C is shared by store and matrix — **not the
array** (from MiniTPU's owner).

- **19.0 % of peak is the MEASURED figure** at the margin.
- **36.4 % of peak is a BOUND**, set by the matrix controller not overlapping a
  push with a pop.
- **Never present the bound as a measurement.** That error was made once already
  and corrected.

A 16×16×16 matmul takes **168 cycles**, capped by the serialized matrix-command
FSM (push and pop share one controller). (*Originally from the rescued
comparison work and not stated in either surviving source doc; MiniTPU's owner
now reports the same 168 for the 16×16 GEMM Verilator workload, so it is
corroborated.*)

The rest of the gap is result latency — 82 cycles from ``vmatpush`` to first
result — and it **cannot be hidden by reordering**: loading the next weights
early corrupts the tile in flight, because weight-commit reaches every PE on one
cycle (``mxu_pe.sv``). Recovering it needs hardware (hold commit until the array
drains, or a second weight bank); splitting the matrix controller alone was
measured and gains nothing.

**Board-validated** means exactly three bring-up kernels: ``halt`` 4 cy,
``vreg`` 12 cy, ``scalar`` 8 cy. **Nothing else on this page is a board cycle
count.**

The workload figures below are **Verilator** counts, not board figures. Per
MiniTPU's owner, after its commit ``72c8b78`` corrected ``vmatpop`` occupancy
from 7 to 4, 16×16 GEMM is **168** cy and 16×16 GEMM→GELU is **155** cy. The
remaining Verilator targets are unchanged on record: 32×64 BF16 add 119 cy,
32×64 multiply-add 155 cy, 32×32 DMA-streamed multiply-add 929 cy, 768-element
LayerNorm 941 core cy / 218 bundles.

Accuracy: GPT-2 124M block 0.68 % vs fp32, Qwen2.5-0.5B decoder layer 3.81 %.
``B = 1..16`` is bit-identical **row-wise only**.

What this page said about all three before it was corrected is in
`Earlier measurements and corrections`_.

5. What this says about our own cost model
------------------------------------------

``docs/isa_latency.json`` (in the MiniTPU tree) is the measured-from-silicon version of what
``tpu.latency(unit, ii=, depth=)`` asserts pre-synthesis, and carries two latency
shapes our ``UnitLatency`` (``allo/exp/dsa/core.py:438``, on ``chia-codesign``; see :doc:`/extensions/act`) does not express:
**conditional latency** (latency depending on a statically known operand
property — in the npu tree ``vld``'s ``L = W + conflict degree``; per the correction
above the live design has no such case), and **occupancy ≠ result latency**
(``vmatpush`` occupies the controller 5 cycles, first result at +82 — queue
semantics, not an ``(ii, depth)`` pair).

Writeback offsets for scale: ALU 5, SFU 7, cross-lane reduction 15 (it crosses
all 64 elements — a reduction tree, not a per-lane pipe), ``vld`` 6, ``vmatpop``
3..6. Occupancies: ``vmatload`` 18, ``vmatpush`` 5, ``vmatpop`` **4** (was 7
here; corrected by MiniTPU's commit ``72c8b78``).

Also worth carrying: **the binding resource is the instruction encoding, not the
silicon.** The compute bundle has 1 spare bit — doubling VMEM rows per slot fits
exactly it; doubling VREG count needs 5 bits that do not exist.

6. Defects reported outward to that design's owner
--------------------------------------------------

- The stale **7.255 W** README figure (shipped: 8.255 W) — since fixed.
- ``src/core/mxu/mxu_adapter.sv`` **is dead code**: listed in ``core.f`` and ``vpu.f``,
  instantiated nowhere in ``src/`` or ``tb/`` *(verified, still present)*.
  **CORRECTION**: the rescued report treats the MXU adapter path as live.
- A stale ``matrix_busy_o`` doc comment.

7. Where the two source trees disagree
--------------------------------------

The rescued report was written against ``~/core/npu``; ``~/core/minitpu`` is live
and has moved. Do not mix vocabularies:

=========== =============================== ====================================
\           rescued report (``~/core/npu``) live ``~/core/minitpu`` *(verified)*
=========== =============================== ====================================
VLIW slots  E / M / X / S / D / F           V / M / MEM / S / C
DELAY field 6 bits                          **7 bits**
VMEM        4 banks/lane + conflict model   **flat dual-port URAM, no arbiter**
=========== =============================== ====================================

Both agree on: 16×16 BF16 array, 24-bit accumulator, 128-bit bundle, no
interlocks, 24-bundle loop buffer, 32-entry MXU output FIFO, 82-cycle
push→result latency.

Consequence: **the rescued report's 8-stage TinyTPU scaling plan rests on the
banked-VMEM model and a live MXU adapter, so it is invalidated in those parts.**
Not reproduced here; what survives of it is §5.

The concession that became the win
----------------------------------

The rule first, because it is the part that transfers: **an overhead you
exclude from a window is an overhead you cannot see, so put the excluded
quantity beside someone else's** -- the comparison is what makes an absurd
value look absurd. What follows is the episode that established it, and it is
a method finding rather than an architectural one.

While establishing that a fair comparison must **count every machine's host or
none of them** (:doc:`/designs/gemmini_comparison`), MiniTPU's owner volunteered
a figure against their own interest: their per-launch host work is about **132
microseconds with a 44-microsecond register-access floor**, roughly **24,750
cycles at 187.5 MHz**, none of which their simulator numbers include. It was
offered so that a comparison counting Gemmini's ~390 cycles of driver would not
quietly omit theirs.

Stating it next to Gemmini's ~390 is what made it **absurd rather than
normal** — two orders of magnitude, for the same job. Within hours, two commits
on their master attacked exactly that cost, reading a configuration file once
instead of once per launch and spinning past the slowest launch, and took the
GPT-2 body from 79.16 to a claimed **160.38 tok/s on the same bitstream**.
Neither cost was visible to their simulator, which is why both survived as long
as they did.

So the fairness adjustment was a **diagnostic**. The honest accounting did not
merely make the comparison defensible; it located the largest measured win of
that night, in a quantity nobody had been optimising because nothing in the
measurement setup showed it.

The transferable form, and the reason this is recorded on our side too: **an
overhead you exclude from a window is an overhead you cannot see.** Our cosim
figures contain no host at all — not a fast host, none — so we have no
equivalent number to be shocked by yet, and a real deployment eventually will.
The rule that follows is not only "say what your window excludes" but "put the
excluded quantity beside someone else's, because the comparison is what makes an
absurd value look absurd."

What aligning to MiniTPU's semantics costs
------------------------------------------

An experiment, now stopped, asked whether TinyTPU-isa could be made
semantically comparable to MiniTPU so the two machines' cycle counts would mean
the same thing. Six increments were built and independently re-measured; the
answer is that the alignment is expensive enough to stop pursuing, and that
result is the finding.

The variant lives on the branch ``tinytpu-align`` (``10ee6882``), which is
**not merged into main** and is kept as the record: the repros, the verification
logs, the csynth report and the per-unit table are all in that commit. The
shipped design (:doc:`/designs/tinytpu_isa`) is unaffected and remains the
baseline for every performance claim.

Measured, by cosim, at the five benchmark shapes: **216 / 408 / 809 / 933 /
1521** cycles with zero mismatches, against the shipped design's 172 / 262 /
418 / 484 / 686 as it then stood (that row is 171 / 261 / 417 / 483 / 685
since 2026-09-22, which does not move the ratio) — **1.26x to 2.22x the
cycles**. At ``T=8`` the variant measures
302 / 474 / 699, also exact. Functional gates pass throughout: bench ``ALL
EXACT``, stress 487/487, 34 crafted bad programs rejected against 390 generated
programs accepted, and 42 mutants caught (41 by the functional gates, one only
by RTL cosim).

The cycles are not the whole cost, and this is what stopped the track.
Synthesis of the same build (xcu280, 3.33 ns target, ``T=4``, ``MAXDIM=16``)
estimates **3.782 ns with -1.35 ns of top-level slack** — the variant does not
close at the shipped design's frequency, where the shipped design estimates
2.431 ns with margin. The critical path is in ``vpu_0`` and ``vmu_0``, the two
units the MiniTPU-semantics increments created and the only two carrying
``style=flp`` loops; every other unit still reports 2.431 ns. So the honest
comparison multiplies a 1.26-2.22x cycle cost by a roughly 1.56x longer clock,
and the variant's resources are BRAM 41 / DSP 21 / FF 17,918 / LUT 24,244
against the shipped design's BRAM 42 / DSP 14 / FF 17,481 / LUT 26,583.

Where the cycles went, per increment, at 4x4x4 and 16x16x16:

============================= ================= ==================================
Increment                     Cycles            Step
============================= ================= ==================================
1 (``b3793f85``)              176 / 750         +4 / +64
2 (``8f113313``)              186 / 1216        **+10 / +466**
3 (``7b087c44``)              197 / 1904        **+11 / +688**
3b (``c565ebea``)             199 / 1514        +2 / -390
4 and 5a (branch HEAD)        216 / 1521        +17 / +7
============================= ================= ==================================

The two large steps are increments 2 and 3, and they are real: each was
re-measured independently at its own commit with zero mismatches. Increment 3b
recovered 390 cycles of increment 3's 688.

Three claims the building agent made were corrected by the re-measurement, and
are recorded on the branch: there are 42 mutants rather than 41; ``frp``
**passes** the pipeline-style repro rather than deadlocking as the register
said; and a 72-bit stream does not "run clean" — 65, 72 and 96 bits all corrupt
the simulator heap, and what varies is only whether the overrun lands somewhere
fatal (see :doc:`/developer/limitations`). Two documentation defects also stand
on that branch: ``DELAY`` is described as an honoured minimum issue spacing but
appears nowhere in the implementation, and a cited ``isa_dsl.eltwise_program``
does not exist.

One piece of the experiment was judged worth keeping independently of it:
``s.pipeline(style=)`` (``0038833c``), a ~20-line schedule primitive modelled on
``rewind`` and covered by ``tests/test_vhls.py::test_pipeline_style``. Any
cyclic dataflow design needs it. Note that only ``EmitVivadoHLS.cpp`` reads
``pipeline_style``; the Intel, Catapult, Tapa and XLS emitters ignore it
silently.

Provenance
----------

The "rescued scaling report" referred to above was kept in the repository as
``notes/archive/minitpu_scaling.md`` until the notes were moved into this
documentation. It holds the 8-stage TinyTPU scaling plan, dropped here as plan
rather than evidence. **Read it with care:** it describes ``~/core/npu``, a
*different machine* from the live ``~/core/minitpu`` (see the table in the
previous section), and its scaling stages rest on the banked VMEM and on
``mxu_adapter.sv``, which is dead code. It is recoverable from git history:

.. code-block:: bash

   git show a2d92cd8:notes/archive/minitpu_scaling.md

Neither source tree travels (``~/core/npu`` is now empty), so nothing in it can
be re-derived from source on another host.

Earlier measurements and corrections
------------------------------------

What this page said before, kept so the corrections are checkable. None of it
is current.

The README's stale power figure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The design's README once stated **7.255 W**, which was stale for the
``0xB182E69D`` build. Fixed after we reported it; it then read 8.255 W. The
report itself is §6.

The cycle-model fit before the ``vmatpop`` correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fit this page previously carried, exact over 20 simulated points before
the ``vmatpop`` occupancy correction (``72c8b78``), was
``46 + 194·n + 64.5·B + 303·n·B``. Kept as the earlier reading, not as
current. The fixed term and the ``B`` coefficient did not move; the ``n`` and
``n·B`` terms did.

Readings of another team's design that this page got wrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* An earlier revision said the array accumulates **"exactly 16 deep"**, which
  read as an exactness guarantee. It is not one: only the first term is exact,
  via a zero bypass, and every later add rounds.
* An earlier revision of §4 implied that figures other than the three bring-up
  kernels were board cycle counts. They are not.
* The Verilator workload figures were stale here as well: 16×16 GEMM was
  published on this page as **283** cy and 16×16 GEMM→GELU as **253**, against
  the corrected 168 and 155.
* An earlier claim that ``B = 1..16`` is bit-identical **omitted the row-wise
  qualifier, and was wrong for attention**.
