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

#################################################
The Co-design Loop: Mapping and Hardware Together
#################################################

An agent proposes a machine. A **frozen mapper** enumerates the whole loop-nest
mapspace against that machine, asks it which nests it can encode, and the best
one it can is what goes under the RTL. Cycles come from Vitis C/RTL cosim and
resources from ``csynth`` of the same build, reported as a pair.

This is the loop :doc:`/extensions/act` argued for, built. It reuses ACT's
*interface* -- a mapping as a tuple of ``Loop(rank, factor, level, spatial)`` --
and none of its code, for the reasons that page measures. ACT is Kai Shao's
work (https://github.com/kkkaishao/allo, branch ``act``); see ``ATTRIBUTION.md``
at tag ``chia-codesign-final``.

The design-level loop it is built on -- the tool surface, the path allowlist,
the fresh evaluation tree from ``git show``, byte-identity on frozen files,
import-time policing, ``bwrap``, nonce-vouched gates, the spend caps -- is
:doc:`/extensions/chia`. Only what the co-design loop adds is described here.


Why the loop has to be a co-design loop
=======================================

Because the mapping search alone finds almost nothing, and that is measured.

At 16x16x16 the mapper enumerates 1,226 nests and the shipped hardware can
encode **3**. On the shipped hardware the nest the design already ships is the
best of those 3 at 16x16x16, so the only thing a pure mapping search buys is
**3 cycles at 4x4x4** -- the trip-count-1 ``loop``/``endloop`` pair the
enumerator omits and the hand-written program keeps (see *The control*, below).

The thing that collapsed the mapspace is a **hardware/ISA parameter**, not a
compiler bug:

.. list-table:: refusals at 16x16x16, shipped hardware
   :header-rows: 1
   :widths: 10 20 70

   * - nests
     - cause
     - what it is
   * - 1,150
     - ``acc-peel``
     - ``acc`` is a **static instruction field with no predicate on an
       induction variable**, so the k=0 tile has to be a peelable prefix. That
       pins K innermost and unsplit and kills every permutation that moves it.
   * - 54
     - ``emitter``
     - a limitation of ``isa_dsl.gemm_from_nest`` itself, not of the machine:
       it re-stages A per m-tile and cannot interleave that with an n loop.
   * - 17
     - ``agu-terms``
     - ``AGU_TERMS=3``, one instruction word's address-term budget. It does not
       merely forbid nests, it **chooses the reuse strategy**: an m-tiled nest
       is encodable only if A is re-staged per m-tile, because keeping A
       resident across m needs a fourth term on the accumulating ``mm``.
   * - 2
     - ``accumulator-raw-distance``
     - ``AR_RAW_DIST``, refused by ``check_program``.

``LOOP_DEPTH=4`` refuses nothing at all here. So the agent's lever is the
machine, and the number of encodable nests is the first thing a change moves.


What is editable, and what is frozen
====================================

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * -
     - files
     - how it is enforced
   * - **editable**
     - ``microarch_isa.py`` (the hardware, the ISA, the encoding, the program
       validator: ``acc``'s predicability, ``AGU_TERMS`` and the AGU word's
       packing, ``LOOP_DEPTH``, ``AR_RAW_DIST``, ``IMEM_SIZE`` via
       ``_MAX_STATIC``, the intrinsic tile) and ``isa_dsl.py`` (the encoder,
       including ``gemm_from_nest``)
     - the agent edits a private copy under ``<run>/<worker>/spec/``, never the
       repository, and only through three MCP tools that refuse any path but
       those two bare names
   * - **frozen**
     - ``chia_agent/mapspace.py`` -- **the mapper: its enumerator, its
       selection rule and its objective** -- plus ``codesign_gate.py``,
       ``codesign_cosim.py``, and everything the design-level loop already
       freezes: ``cosim.py``, ``bench_isa.py``, ``stress_isa.py``,
       ``isa_ref.py``, ``kpn_model.py``, ``param_check.py``,
       ``gate_runner.py``, ``evaluate.py``, ``spec_policy.py``
     - ``evaluate.py`` composes a fresh evaluation tree per candidate, with
       every frozen file read from ``git show`` and never from disk; the design
       evaluator is additionally checked byte-identical to main @ ``MAIN_BASE``;
       the two spec files are the only thing taken from the spec directory and
       anything else there is **ignored**; every candidate process runs under
       ``bwrap`` and the tree is re-hashed after each stage

The agent may **read** the mapper (``read_reference("mapspace.py")``). It should
-- the rule is what "better nest" means. It just cannot change it. The frozen
half is what makes "the best mapping for this hardware" a claim rather than a
sample.


The inner loop is exhaustive, not agentic
=========================================

For each hardware candidate, ``mapspace.search`` enumerates the **whole**
mapspace and asks the candidate's encoder about every nest. It does not search
cleverly and does not stop early.

That is the design decision the loop turns on. A non-exhaustive inner search
would turn "which hardware runs the better program" into "whose search got
luckier". Keeping the mapspace enumerable is therefore a requirement, not a
convenience -- at 16x16x16 it is 1,226 nests and about one second of pure
python.

The selection rule, stated once and frozen: among the nests the candidate can
encode at a shape, take the minimum of ``(dynamic instruction issues, static
instruction words, nest string)``. Both counts are *static* properties of the
program, not cycle estimates; the nest string only breaks ties so the choice is
deterministic. A nest must be encodable at **both** relu settings, and is
ranked by the ``relu=False`` program, which is the one the scored testbench
runs.

A nest counts as encodable only if the encoder produced a program **and**
``assemble`` took it -- which is where ``IMEM_SIZE`` is enforced. A program the
instruction memory cannot hold is refused, never truncated.


Feedback is measured, not modelled
==================================

- **Cycles**: Vitis HLS 2023.2 ``csynth`` + ``xsim`` C/RTL cosim, through main's
  frozen ``cosim.py``, at 4x4x4 and 16x16x16 in the search and at all five
  ``SHAPES`` in ``accept.py``. Each testbench bit-exact against numpy, one row
  per shape, ``mismatches = 0`` in the summary *and* in that shape's own log,
  a PASS, and the cycle count cross-checked against the log's simulated time.
- **Resources**: FF, LUT, BRAM18K, DSP, URAM and the estimated clock from the
  ``csynth.xml`` of the same build. The clock must meet 3.33 ns, or cycles at a
  clock the design cannot meet are not comparable.

``codesign_cosim.py`` is how the mapper's program reaches the RTL: ``cosim.py``
stays byte-identical to main, and this frozen driver binds ``cosim.gemm_program``
to ``gemm_from_nest(mapspace.select(...))`` before calling ``cosim.main()``.
Same testbench generator, same operands, same golden reference, same TCL.

**No modelled cycle count is ever reported.** The only proxy in the loop is the
static ranking above, it lives inside ``mapspace.py``, and it never leaves it.
The failure mode being avoided is named in :doc:`/extensions/act`: ACT's
``Priced.cost`` is ``(makespan, emits)`` and models no instruction memory, while
``assemble`` asserts ``len(words) <= IMEM_SIZE``, so a loop scored on it would
reward a program the machine cannot hold. The five-point regression
``74.5 + 21.70 x dynamic`` invites the same mistake more cheaply.


The objective is a pair, and is never collapsed
===============================================

``classify()`` in ``loop.py``, frozen:

- **win** -- a Pareto improvement: no shape's cycle count higher, at least one
  lower, and no resource term grown.
- **trade** -- the total cycles fall, but something is paid for it: a resource
  term grew, or another shape regressed.
- **regression** -- the total does not fall.

A candidate that trades 2.3x the block RAM for 9% of the cycles is a trade. That
exact trade has already happened on this design once: run 1's ``dma_ld``
widening took BRAM18K from 42 to 98 for -59 cycles at the largest shape. The
loop still tracks its best by total cycles so that the search can move, but the
pair is what gets reported.


Correctness is absolute
=======================

A candidate must pass, in order, before any cycle is reported:

1. ``bench_isa.py`` -- ALL EXACT, and it requires ``isa_dsl.gemm_program`` to
   equal ``microarch_isa.gemm_program_handwritten`` word for word.
2. ``stress_isa.py`` -- 492/492 runs exact: full-range/corner/boundary int8 at
   all 64 shapes, ``C`` prefilled and compared in full, vector and 200 random
   programs against ``isa_ref.py``, and the program validator's controls.
3. ``param_check.py`` -- the design rebuilt at MAXDIM 8 and 12 and exact there.
4. **the mapspace gate** (new): the seam -- the canonical nest must re-emit the
   candidate's own ``gemm_program`` word for word -- then the exhaustive
   enumeration, then **every** encodable nest (up to 48 per shape, best first,
   always including the one that gets cosimmed) proved against ``isa_ref.run``
   at full-range int8 with ``C`` prefilled and the whole of ``C`` compared.
5. **the cosim** -- bit-exact at every shape reported.

Point 4 is what makes "we widened the encoder" checkable. Deleting a legality
check to make 1,150 nests encodable is the cheap way to a big number, and the
gate rejects it immediately: the nests compute the wrong thing.

Each check runs under ``gate_runner.py``, which reads a per-run nonce on stdin
before the candidate is imported, takes the check's **return value** rather than
its printed output, freezes every loaded ``numpy*``/``allo*`` module against
rebinding, and only then prints the line the evaluator requires.


The control
===========

``accept.py``'s ``BASELINES`` table is keyed on the git blob of the two spec
files, so prose-only edits invalidate it. The co-design loop therefore
**measures its control in the same run**: ``loop.py`` iteration 0, and
``test_codesign.py`` case k1.

Measured 2026-09-22, unmodified design, same environment and tool state:

.. list-table::
   :header-rows: 1
   :widths: 16 16 16 52

   * - shape
     - co-design
     - published
     - why
   * - 4x4x4
     - **169**
     - 172
     - N/T = K/T = 1, so the enumerator offers a nest with no emitted loops
       while the hand-written program keeps a trip-count-1 ``loop``/``endloop``
       pair around the output body: 24 words against 28, the same 4 dynamic
       issues, 3 fewer cycles. Bit-exact.
   * - 16x16x16
     - **686**
     - 686
     - the mapper's pick **is** the canonical nest, so the program under the RTL
       is ``gemm_program`` word for word.

The published column is main @ ``476a70d8``: 172 / 262 / 418 / 484 / 686. It
reproduces exactly wherever the program is the same one, which is the check that
matters -- the 4x4x4 difference is the mapper finding a better program, not the
tools moving.

Resources at that build, ``csynth``: BRAM18K 42, DSP 14, FF 17,481, LUT 26,583,
URAM 0, estimated clock 2.431 ns against the 3.33 ns target.


Running it
==========

``$0``, no model, before anything is spent::

    conda activate allo
    export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
    examples/tinytpu/reproduce_codesign.sh mapspace   # ~2 s
    examples/tinytpu/reproduce_codesign.sh control    # ~3 min
    examples/tinytpu/chia_agent/test_codesign.py      # ~7 min

A search::

    conda activate chia_env
    set -a; source chia.env; set +a          # CHIA_TOTAL_CAP_USD per allocation.json
    ray start --head --resources='{"opencode_creds": 2}' --include-dashboard=false
    cd examples/tinytpu/chia_agent
    python preflight.py --budget-usd 15                       # the gate alone, $0
    python swarm.py --codesign --workers 2 --iterations 3 --budget-usd 15
    python accept.py --codesign --diff <run>/<worker>/best.diff \
                     --out <run>/accept-<worker>
    ray stop

The Vitis project goes wherever ``TPU_PRJ`` says and is wiped before every
evaluation, so a stale report can never be read as a result. Disk on this host
is tight; ``reproduce_codesign.sh`` deletes each project when it is done.


The LLVM-free test suite
========================

``chia_agent/test_codesign.py``, ``$0``, no model. ``test_harness.py`` still
covers the design-level loop; this covers only what the co-design loop adds.

.. list-table::
   :header-rows: 1
   :widths: 8 42 50

   * - case
     - what it does
     - expected
   * - k1
     - a no-op (re-save the two files)
     - reproduces the control exactly: the cycles, and the **whole** pinned
       refusal histogram and chosen nest at both shapes
   * - k2
     - a real hardware widening: ``AGU_TERMS`` 3 -> 4 with the AGU word
       repacked from three 19-bit terms to four 16-bit ones, in all five places
       the 19 was a literal (encoder, sequencer kernel, ``expand``,
       ``check_program``)
     - the gate passes and the **encodable count rises**, 3 -> 7 at 16x16x16,
       with ``agu-terms`` going to 0 and the next constraint appearing. A
       fixture for the harness, not a recommendation
   * - k3
     - the encoder widened by **deleting** ``gemm_from_nest``'s ``acc-peel``
       position check
     - REJECTED at ``gate:mapspace``: the newly "encodable" nests are WRONG
       against ``isa_ref``
   * - k4
     - the seam broken -- ``gemm_from_nest``'s prologue order changed so
       ``gemm_program`` still matches the hand-written reference but the
       canonical nest no longer re-emits it
     - REJECTED at ``gate:mapspace``, naming the differing instruction
   * - k5
     - the PE partial sum narrowed from int32 to int16
     - REJECTED at ``gate:stress``, and **no mapspace report is produced** --
       the co-design stages never run on a design that is not correct
   * - k6
     - ``_MAX_STATIC`` 24 -> 12, so ``IMEM_SIZE`` is 32 words and the chosen
       16x16x16 program needs 34
     - REJECTED, the refusal naming ``IMEM_SIZE``; ``assemble`` **raises**
       rather than truncating, and the mapper reports 0 encodable with the
       refusal attributed to ``imem``
   * - k7
     - a sabotaged ``mapspace.py``/``codesign_gate.py``/``cosim.py`` dropped
       into the spec directory, and ``accept.py`` handed diffs that touch the
       mapper, ``cosim.py`` and ``stress_isa.py``
     - the sabotage is **ignored** (the evaluator composes those from git) and
       the verdict is the control's; each diff is refused by path
