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

###############################################################
Co-design Record: The Mapspace Refusal Histogram, Per Machine
###############################################################

.. note::

   **Dated measurement record.** TinyTPU-isa at ``codesign-loop``, T=4,
   MAXDIM=16; the frozen mapper ``chia_agent/mapspace.py``; dated **2026-09-22**.
   Re-derive with ``python chia_agent/histogram.py`` (``$0``, no model, no Vitis,
   ~6 s for all five variants); ``--second-cause`` and ``--interaction`` produce
   the other two tables on this page. Cosim figures are Vitis HLS 2023.2 + xsim,
   xcu280-fsvh2892-2L-e, 3.33 ns target.

   **No number on this page is a cycle count except where it says cosim.** The
   histogram is a count of loop nests. See :doc:`/extensions/codesign` for the
   loop, :doc:`/extensions/act` for why the mapper looks like this.

What is being counted
=====================

For one GEMM shape, the mapper enumerates the whole loop-nest mapspace --
factorisations x permutations over the emitted slots, with the innermost level
pinned to the array's intrinsic tile -- and asks the machine's encoder whether it
can *say* each nest. At 16x16x16 that is **1,226 nests**. A nest counts as
encodable only if ``isa_dsl.gemm_from_nest`` produced a program **and**
``microarch_isa.assemble`` took it, which is where ``IMEM_SIZE`` is enforced.

Each refusal is attributed to the constraint that caused it. That attribution is
the co-design signal: it says which part of the machine is standing between the
compiler and a better program.

The variants
============

Each is a hardware/ISA edit to ``microarch_isa.py`` and nothing else, applied to
the design as it ships. They are the same edits ``test_codesign.py`` uses, so the
table and the test cannot drift apart.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - variant
     - what changes
   * - ``shipped``
     - the design as it ships: ``AGU_TERMS=3``, ``LOOP_DEPTH=4``,
       ``IMEM_SIZE=56``, ``AR_RAW_DIST=4``
   * - ``agu4``
     - ``AGU_TERMS=4``. The AGU word is 64 bits and three terms use 19 each
       (target 4 bits, level 3, stride 12), so a fourth has to come out of the
       field widths: four terms of 16 (target 4, level 3, stride 9, 8 usable;
       programs at MAXDIM=16 use strides of at most MAXDIM). Five exact edits,
       one per place the 19 was a literal -- ``enc_agu``, the sequencer's Allo
       kernel, ``expand``, ``check_program``
   * - ``agu4+depth6``
     - and the loop stack from 4 frames to 6 (one constant; every array it
       sizes follows, and the 3-bit level field already reaches 7)
   * - ``agu4+depth6+imem``
     - and ``_MAX_STATIC`` 24 -> 48, so ``IMEM_SIZE`` 56 -> 104
   * - ``depth6``
     - **the control**: the 6-frame loop stack *alone*

Measured: 16x16x16
==================

1,226 nests enumerated for every row.

.. list-table::
   :header-rows: 1
   :widths: 20 12 20 10 10 10 10 10

   * - variant
     - encodable
     - chosen nest
     - ``acc-peel``
     - ``emitter``
     - ``agu-terms``
     - ``loop-depth``
     - ``ar-raw-dist``
   * - ``shipped``
     - **3**
     - ``N4>K4 rows=16``
     - 1150
     - 54
     - **17**
     - --
     - 2
   * - ``agu4``
     - **7**
     - ``N4>K4 rows=16``
     - 1150
     - 54
     - --
     - **6**
     - 9
   * - ``agu4+depth6``
     - **8**
     - ``N4>K4 rows=16``
     - 1150
     - 54
     - --
     - --
     - **14**
   * - ``agu4+depth6+imem``
     - 8
     - ``N4>K4 rows=16``
     - 1150
     - 54
     - --
     - --
     - 14
   * - ``depth6``
     - 3
     - ``N4>K4 rows=16``
     - 1150
     - 54
     - 17
     - --
     - 2

Measured: 4x4x4
===============

4 nests enumerated. N/T = K/T = 1, so the space is almost empty.

.. list-table::
   :header-rows: 1
   :widths: 24 14 24 19 19

   * - variant
     - encodable
     - chosen nest
     - ``agu-terms``
     - ``ar-raw-dist``
   * - ``shipped``
     - 1
     - ``- rows=4``
     - 1
     - 2
   * - ``agu4`` (and the two supersets)
     - 1
     - ``- rows=4``
     - --
     - 3
   * - ``depth6``
     - 1
     - ``- rows=4``
     - 1
     - 2

What the table says
===================

**1. The binding refusal migrates, twice.** Widening one instruction-word field
moves the bottleneck rather than removing it:

.. code-block:: text

    shipped        agu-terms 17  ->  loop-depth  0   ar-raw  2     3 encodable
    agu4           agu-terms  0  ->  loop-depth  6   ar-raw  9     7 encodable
    agu4+depth6    agu-terms  0  ->  loop-depth  0   ar-raw 14     8 encodable

This is what co-design evidence looks like. A hardware field change expands the
software's expressible space, and the next constraint becomes visible for the
first time. A change that fixed everything at once would not be believable.

**2. The new refusals are created by the change, not pre-existing.** That is
what the ``depth6`` control is for. On the shipped design ``LOOP_DEPTH=4``
refuses **nothing**, and a 6-frame loop stack on its own changes not one number
in the table. The 6 ``loop-depth`` refusals in ``agu4`` exist *because* the
4-term AGU let nests through that then ran out of loop frames. Likewise
``ar-raw-dist`` rises 2 -> 9 -> 14 as more nests get far enough to be judged by
``check_program``.

**3. Instruction memory never binds here.** ``IMEM_SIZE`` 56 -> 104 changes
nothing: the chosen program is 34 words and the widest survivor 38. That matters
because instruction-memory capacity is precisely what a cost model like ACT's
``(makespan, emits)`` does not have (:doc:`/extensions/act`), and it is worth
recording that on *this* machine it is not the limit -- the limit is the
encoding.

**4. None of it changes the program that gets run.** The chosen nest is
``N4>K4 rows=16`` in every variant, including the widest. The mapper's pick is
unchanged, so the RTL runs the same instruction stream, so **the cosim cycle
count does not move** -- while the area of a 4-term AGU and a 6-frame loop stack
does. Read on its own, that is a cost with no benefit.

It is not, and the next section is why.

**5. The 54 ``emitter`` refusals are ours, not the machine's.** They are
``gemm_from_nest`` declining to interleave its per-m-tile A staging with an n
loop. They belong in the table because an honest histogram distinguishes "the
hardware cannot say this" from "our generator has not learned to".

**6. The first-cause histogram overstates the prize**, and it must not be read
as "1,150 nests a fixed ``acc`` would free". The encoder raises at the first
constraint a nest violates. Removing the ``acc-peel`` *position* check and
re-censusing (``histogram.py --second-cause``; counting only, since the
programs it admits overwrite the accumulator on every k-tile, which is test k3)
gives:

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * - census at 16x16x16
     - expressible
     - ``acc-peel``
     - ``emitter``
     - ``agu-terms``
     - ``ar-raw``
   * - first cause, shipped
     - 3
     - 1150
     - 54
     - 17
     - 2
   * - position check dropped
     - 12
     - **897**
     - **274**
     - 35
     - 8
   * - dropped, and ``agu4``
     - 17
     - 897
     - 274
     - --
     - 20

So 897 of the 1,150 are refused by the *other* ``acc-peel`` branch -- K split
across two emitted loops, where ``acc`` would have to follow two induction
variables at once -- and 274 by the encoder's own m/n limitation. Only 12 become
expressible.

.. note::

   **Reconciled with an independently built enumerator** (the ACT rebuild),
   2026-09-22, and the disagreement was a definition rather than a bug in
   either:

   - its **930** counts nests that violate the RAW-distance predicate asked of
     every nest *independently of every other constraint*. That is a different
     quantity from anything a first-cause histogram reports, and both are
     legitimate.
   - splitting ``acc-peel`` into its two branches, its ``acc-split`` count is
     **897 -- exactly the figure measured here**, by a separately written
     enumerator over the same mapspace.
   - its encodable count of **5** was withdrawn by its author as a *hardware*
     claim: the two extra row-tiled nests pass the reference model, the
     validator, the cycle model, the dataflow simulator and Vitis csim, and
     then **cosim does not complete** -- stuck at
     ``Inter-Transaction Progress 0/1`` for over half an hour, against about
     two minutes for the shipped mapping. Encodable is not runnable, and a
     hardware claim takes the smaller number. **3** stands.

   The corollary is worth carrying, because it governs how any of these counts
   may be compared: **a first-cause histogram is a property of check order and
   of emitter limits, not of the hardware.** Only an independent census (each
   constraint asked of every nest) and a relieve-and-re-census are comparable
   across implementations.

The interaction: two changes that are complements, not alternatives
===================================================================

This is the headline, and it is the one result here that a one-knob-at-a-time
search cannot reach.

``acc`` is field ``f2``, and ``f2`` **is** an AGU target. So the no-peel form --
``acc`` carried by one additive AGU term, no peeled k=0 tile -- is a thing the
ISA can nearly say already. Its accumulating ``mm`` names A (``f0``, one term),
``acc`` (``f2``, one term) and its weights at ``B_SP + nb*MAXDIM + kb*T``
(``f3``, **two** terms): four in all, as soon as the program has an n loop at
all.

Measured, ``histogram.py --interaction``, for the no-peel ``mm`` at each
(Kt, Nt):

.. list-table::
   :header-rows: 1
   :widths: 16 28 28 28

   * -
     - Nt=1
     - Nt=2
     - Nt=4
   * - **AGU_TERMS=3**, Kt=2
     - no: AGU budget
     - no: AGU budget
     - no: AGU budget
   * - **AGU_TERMS=3**, Kt=3
     - no: AGU budget
     - no: AGU budget
     - no: AGU budget
   * - **AGU_TERMS=3**, Kt=4
     - no: AGU budget
     - no: AGU budget
     - no: AGU budget
   * - **AGU_TERMS=4**, Kt=2
     - **expressible, exact**
     - **expressible, exact**
     - **expressible, exact**
   * - **AGU_TERMS=4**, Kt=3
     - no: f2 out of range
     - no: f2 out of range
     - no: f2 out of range
   * - **AGU_TERMS=4**, Kt=4
     - no: f2 out of range
     - no: f2 out of range
     - no: f2 out of range

The two constraints are **in series, and the first masks the second**:

- At ``AGU_TERMS=3`` every cell fails on the address-term budget. The
  monotonicity limit is never reached, so on the shipped machine it **cannot be
  observed at all** in any program with an n loop.
- Widen the AGU and the budget clears -- and now monotonicity becomes the
  binding constraint, visibly: Kt=2 works and is numerically exact against
  ``isa_ref``, Kt>=3 dies because an additive term's third value is 2 and
  ``check_program`` requires ``f2`` in {0, 1}.

Hence: ``agu4`` alone unlocks nests and changes no pick, so on its own it looks
like a cost with no benefit. A step mechanism for ``acc`` alone cannot even be
exercised, because the budget refuses the instruction before ``f2``'s value is
ever in question. **Neither alone shows anything; only together does either
have anything to bite on.** That is what makes the pair a co-design result
rather than a knob: a search that varies one parameter at a time sees nothing in
either direction and concludes, wrongly, that the mapspace is not the
constraint.

What it does **not** say is that ``agu4`` plus an additive term is a fix. Kt=2
is K <= 8, and the scored shape is 16x16x16 with Kt=4. A working mechanism needs
the step as well -- saturation, or a predicate on ``iv_now[level] == 0`` -- and
that is what the search is for.


Where a fix has to go, and why
==============================

A rule, not a prohibition, and it falls out of one line:
``isa_ref.run`` iterates ``expand(prog)``.

``expand`` yields the AGU-**resolved** fields. So:

- A step resolved **in the AGU** -- the sequencer's kernel, with ``expand`` kept
  in lockstep -- delivers ``f2`` in {0, 1} to the unit. The instruction's
  architectural meaning is unchanged and **the frozen reference model does not
  need unfreezing**.
- The same step resolved **in a unit's decode** makes ``f2 = 2`` mean
  "accumulate". That *is* a change to what the instruction means, and
  ``isa_ref`` -- which is frozen, and which ``stress_isa.py`` checks every
  program against -- rightly rejects it.

This converts "do not touch the reference model" from something a candidate can
only trip over into something that tells it where to put its change.

It is worth recording that the first agent run at this problem found the
mechanism and put it in the forbidden place. Its one edit, unscored -- the run
was stopped mid-iteration because its prompt carried the wrong diagnosis -- was
``f2 = 1 if w0[30:42] != 0 else 0`` in a unit's decode: saturation, reached for
unprompted, at the site the freeze boundary forbids.


What the mapping search alone buys
==================================

Three cycles, at one shape, measured.

.. list-table::
   :header-rows: 1
   :widths: 14 16 16 54

   * - shape
     - co-design
     - published
     - why they differ
   * - 4x4x4
     - **169**
     - 172
     - the enumerator offers a nest with no emitted loops, while the
       hand-written program keeps a trip-count-1 ``loop``/``endloop`` pair
       around the output body: 24 words against 28, the **same 4 dynamic
       issues**, 3 fewer cycles. Bit-exact
   * - 16x16x16
     - **686**
     - 686
     - the mapper's pick *is* the canonical nest, so the program under the RTL
       is ``gemm_program`` word for word

Published is main @ ``476a70d8``: 172 / 262 / 418 / 484 / 686. It reproduces
exactly wherever the program is the same one, which is the check that matters.

The full five-shape co-design control, measured 2026-09-22 in one run
(``evaluate.py --codesign``, all five shapes, 272 s of cosim, every testbench
bit-exact, csynth 2.431 ns):

.. list-table::
   :header-rows: 1
   :widths: 14 14 14 12 46

   * - shape
     - co-design
     - published
     - delta
     - the nest the mapper chose
   * - 4x4x4
     - **169**
     - 172
     - **-3**
     - ``- rows=4`` -- no emitted loops at all
   * - 8x8x8
     - 262
     - 262
     - 0
     - ``N2>K2 rows=8``
   * - 12x12x12
     - 418
     - 418
     - 0
     - ``N3>K3 rows=12``
   * - 16x16x8
     - 484
     - 484
     - 0
     - ``N2>K4 rows=16``
   * - 16x16x16
     - **686**
     - 686
     - 0
     - ``N4>K4 rows=16``

This was **predicted before it was measured**, and the prediction is recorded
because it is the stronger claim: the middle three picks are bit-identical to
the canonical nest, so their programs are ``gemm_program`` word for word and
their cycles had to be the published ones; 4x4x4 is the only shape whose space
is degenerate enough for the picks to differ. Four of five exact, and the fifth
differing by the amount and for the reason stated.

Those 3 cycles are **real, not sampling**: this cosim is deterministic and the
five shapes reproduce run after run. They are also the *whole* of what a mapping
search buys on this hardware, which confirms rather than contradicts what the
ACT investigation predicted -- the shipped nest was already the best of the
three the machine could encode.

.. warning::

   Do not compare a 3-cycle result of ours against a Gemmini figure without
   carrying Gemmini's spread. Our cosim is deterministic; Gemmini's column was
   measured to have a trial-to-trial spread reaching 20 cycles at a shape
   totalling 161.

Pre-registration: what the search is being asked, and what I expect
===================================================================

Written **2026-09-22, before the corrected search was run**, and reported
against afterwards either way. A pre-registered expectation that then happens is
evidence; the same sentence written afterwards is an excuse.

**What the run measures.** Not "can an agent find a design point". The
interaction above changes the question to **can an agent find a two-part change
where neither part shows anything on its own.** A candidate needs both a wider
address-term budget *and* a mechanism that gives ``acc`` its step, and it has to
get both right in one iteration for the cosim to show anything at all -- because
with only the first the mapper's pick does not move (measured), and with only
the second the instruction does not fit (measured). That is materially harder
than either half, and it is the case a one-knob-at-a-time search cannot reach.

**What I expect.** The most likely single outcome is that **the agent finds a
mechanism but only reaches Kt=2, so the pick does not move and the cycle count
is flat.** Kt=2 is K <= 8; the scored shape is 16x16x16 with Kt=4. Second most
likely is a candidate that widens the AGU and stops there, which is already
measured to change nothing. A cycle improvement at 16x16x16 requires the step to
work at Kt=4, and I do not predict it.

**A flat result is a result, given that this was said first.** It would be a
measurement of the difficulty of a two-part co-design change rather than a
failed search -- provided the difficulty was stated in advance, which is what
this section is for.

**What was seeded**, because the seeding is part of the experimental setup and a
reader should know how much guidance any outcome came with. Everything seeded is
a measurement from this page:

- the first-cause histogram at 16x16x16 (3 of 1,226; 1150 / 54 / 17 / 2);
- the second-cause census (897 / 274 / 35 / 8, 12 expressible), so no iteration
  is spent believing a fixed ``acc`` frees 1,150 nests;
- that ``agu4`` alone raises encodable nests to 7, zeroes the ``agu-terms``
  refusals and **changes no pick**, so its cycles are flat while its area is
  not; and that ``depth6`` alone and ``IMEM_SIZE`` 56 -> 104 each change nothing;
- the masking grid: ``acc`` is ``f2``, ``f2`` is an AGU target, an AGU term is
  additive and monotone, a term on ``f2`` costs one of the three, and at three
  terms every cell is refused on the budget while at four Kt=2 is expressible
  and exact and Kt>=3 is out of range;
- that the mapping search alone is worth 3 cycles at 4x4x4, 24 words against 28;
- the freeze boundary as a **rule of the environment**: ``isa_ref.run`` iterates
  ``expand(prog)``, so a change resolved in the AGU leaves the instruction's
  meaning and the frozen reference model untouched, while the same change
  resolved in a unit's decode alters what a field value means and is rejected.

**What was deliberately not seeded:** any mechanism. Saturation is not named,
predicating on an induction variable is not named, and the workers are not told
where to put a fix -- only what each location costs them. The stopped pilot
shows that constraint is not redundant: it reached for a mechanism unprompted
and put it at the forbidden site.


The result: run 2, against the pre-registration
===============================================

Run ``codesign-run2-20260922-143234``, harvested from its ``variants.jsonl``,
``worker.log`` and opencode's database after the fact: two workers x 5
iterations, $68 budget, ``gemini-3.1-pro-preview``, at ``codesign-loop``
``d7546a0d``. Baseline, measured by the run's iteration 0 in each worker:
**169 / 686** cosim cycles (4x4x4 / 16x16x16), 3 of 1,226 nests encodable at
16x16x16 (1,150 ``acc-peel``, 54 ``emitter``, 17 ``agu-terms``, 2
``accumulator-raw-distance``), chosen nest ``N4>K4 rows=16``, BRAM18K 42 / DSP
14 / FF 17,481 / LUT 26,583, 2.431 ns.

**It was stopped after iteration 2 of 5 by a harness defect, not by its
budget.** Both workers logged ``not starting iter3: spent $52.60 + next call
~$23.26 would pass the $64.50 cap``. The $52.60 was every message on the
account in the run's window; the run's own six sessions cost **$47.65**. The
$23.26 projection was the account-wide spend during an iteration-1 call; the
run's largest single session cost **$11.06**. Counted by the run's own
sessions, $47.65 + $11.06 = $58.71 is under the cap, and iteration 3 would have
started. The per-run cap now sums only the run's own sessions
(:doc:`/extensions/chia`).

**Every model call in the run timed out.** All six calls (iteration 1, its
debug session, and iteration 2, in each worker) ran into the 2,400 s limit and
returned no session id; the loop recorded each as $0.00 and scored what was on
disk when the call was killed. Nothing below is an agent's finished answer.

The two arms were given different angles and **the same system message**,
which states the diagnosis: that ``acc`` is ``f2``, that ``f2`` is an AGU
target, that an AGU term is additive and monotone, that it costs one of the
three terms, and that a step resolved in the AGU leaves ``isa_ref`` untouched.
The ``open`` arm's angle also states the ``agu4`` measurement: AGU_TERMS=4
"raises encodable nests from 3 to 7 and drives the ``agu-terms`` refusals to
zero -- and the mapper's CHOSEN nest does not change". Neither arm was told a
mechanism.

``acc-follows-k`` -- the directed arm. Not a search result.
-----------------------------------------------------------

Its angle was the corrected diagnosis, the interaction grid and the
AGU-resolution rule.

- **Iteration 1: rejected at** ``gate:bench_isa``. It broke the seam:
  ``gemm 4x4x4: instruction 1 AGU word differs: generated
  0x0000000040200085, hand-written 0x0000000040200083``.
- **Iteration 2: passed every gate.** Encodable **3 -> 8** at 16x16x16; cycles
  **169 / 686, unchanged**; chosen nest ``N4>K4 rows=16``, unchanged; FF
  **+113**, LUT **+389**, BRAM and DSP unchanged, 2.431 ns; classified
  ``regression`` (no cycle moved, area rose). The diff sets AGU_TERMS=4 and
  LOOP_DEPTH=8, and makes the ``mm`` accumulate field saturate (``f2 | 1``) in
  the AGU resolution, in both the sequencer and ``expand``.

  **The 1,150 ``acc-peel`` refusals were relabelled, not relieved.** The
  histogram at iteration 2 reads 8 encodable, 54 ``emitter``, 14
  ``accumulator-raw-distance`` and **1,150** under a new cause, ``other:
  Ref.at() takes the induction variable a `with k.loop(...)` y...``. The diff
  renamed the encoder's two ``acc-peel:`` refusal messages to ``disable is``
  and ``ignore:`` and routed those two exceptions to a new fallback emitter,
  which refuses the same 1,150 nests with an error of its own. The +5
  encodable nests are the ``agu4`` fixture's +4 (below) and one of the six
  ``loop-depth`` refusals that LOOP_DEPTH=8 relieves.

``open`` -- the unguided arm
----------------------------

Its angle: the first-cause histogram, the ``agu4`` measurement quoted above,
and "decide for yourself what to attack".

- **Iteration 1: rejected at** ``gate:mapspace``. It made **17** nests
  encodable at 16x16x16 (897 ``acc-peel``, 274 ``emitter``, 20
  ``accumulator-raw-distance``, 18 ``loop-depth``), and **10 of the 17 compute
  the wrong answer against** ``isa_ref`` -- 20 of the 34 programs checked,
  both ``relu`` settings of each -- including the nest the mapper would have
  chosen, ``K4>N4 rows=16``. The reference-model sweep caught every one; the
  seam check passed.
- **Iteration 2: passed every gate.** Encodable **3 -> 7**; cycles **169 /
  686, unchanged**; chosen nest unchanged; FF **+186**, LUT **+703**, BRAM and
  DSP unchanged, 2.431 ns; ``regression``. Refusals: 1,150 ``acc-peel``, 54
  ``emitter``, 9 ``accumulator-raw-distance``, and **6** ``loop-depth``, a
  cause the baseline has none of. The diff sets AGU_TERMS=4 (four
  16-bit terms) and makes ``f2`` saturate to 0/1 for ``mm`` in the AGU
  resolution; the encoder is not taught to use the step, so ``acc-peel`` does
  not move. It also leaves a stray fragment, ``"); raise Exception(str(acc));
  pass #``, inside a docstring line of ``isa_dsl.py``.

  This histogram is **identical** to ``test_codesign.py``'s k2 fixture,
  ``AGU4_MAPSPACE`` (7 encodable; 1,150 / 54 / 9 / 6). The arm reproduced
  the one change its prompt had measured for it. Its saturating step -- a
  mechanism, which nothing seeded -- is real and sits at the permitted site,
  but it is not connected to the encoder, so it unlocks nothing.

Against the pre-registration
----------------------------

The prediction was: most likely a mechanism that reaches only Kt=2, so the
pick and the cycles do not move; second, a candidate that widens the AGU and
stops there. **What happened is the second, in both arms**, with a mechanism
attached that nothing uses: both widened the AGU to four terms, both wrote a
saturating ``acc`` step in the AGU resolution, neither got the encoder to emit
it (the directed arm's fallback emitter tries, and fails on every nest),
neither moved the chosen nest, the cycles did not move, and area rose.

What the run does **not** show: that the loop found the idea. The ``open``
arm's encodability gain is the measurement its own prompt quoted, to the
refusal; the directed arm's is the same plus LOOP_DEPTH, and its apparent
removal of ``acc-peel`` is a renamed error message. The one thing neither
prompt contained is the saturating step, and both arms reached for it at the
site the system message described; that is the finding to test next, with the
``agu4`` fact withheld from the unguided arm.

What it shows about the harness: the refusal histogram's *categories* come
from exception text in ``isa_dsl.py``, which the candidate edits, so a
candidate can move nests between categories without relieving them. The
encodable count cannot be moved that way -- every counted nest, up to
``CHECK_MAX`` = 48, is checked against ``isa_ref`` -- but a histogram quoted by category needs the count of
``other:`` causes beside it.


Resources, for the pair
=======================

``csynth`` of the ``shipped`` build, measured in the same run as the cycles
above: BRAM18K **42**, DSP **14**, FF **17,481**, LUT **26,583**, URAM 0,
estimated clock **2.431 ns** against the 3.33 ns target.

The objective is this pair and is never collapsed. For scale, run 1 of the
design-level search widened ``dma_ld``'s operand bursts for -59 cycles at the
largest shape and took BRAM18K from 42 to **98**: a trade, not a win.

A note on where this check belongs
==================================

The 1,150 ``acc-peel`` refusals are **not** an Allo limitation. At the Allo
level the predicate is ``scf.if`` on ``cmpi eq, index_cast(k), 0``, and Allo
builds it; nothing refuses it and no primitive removes it. What refuses the nest
is *our instruction encoding* -- an address-term budget, and an address term
that grows additively where a step is wanted -- met by a compiler that had no
way to check an encoding.

An ``Encoding`` abstraction with ``s.encodable_on`` has since landed on ``main``,
making a target's instruction-word budget a checkable schedule property
re-verified after every primitive. That is the same question this table asks from
the hardware side, in two vocabularies: ``mapspace.py``'s ``agu-terms`` /
``loop-depth`` / ``imem`` refusals and ``Encoding``'s legality rules should be
made to meet, and the one that lives in the compiler is the one to prefer. This
record is the measured target for that work, not a competing implementation of
it.


A note on numbers and their derivations
=======================================

Three figures on this page were corrected on the day it was written, and in each
case the correction came from someone being asked for a derivation rather than
from anyone being more careful:

- the explanation of the 1,150 refusals was wrong in three documents and two
  worker prompts ("``acc`` is a static field"). Checking it took four minutes
  and a thirty-line probe, which found that ``acc`` is AGU-reachable and that
  the obstacle is monotonicity;
- a census of 930 nests violating the RAW-distance contract, and an encodable
  count of 5, were relayed from another measurement and are **not reproduced
  here**. This page's counts carry ``histogram.py`` as their derivation and the
  disagreement is recorded as open rather than settled by preference;
- the selection rule under-charged control flow, because ``expand`` yields
  nothing for ``LOOP``/``ENDLOOP``. Charging them changes no pick at any of the
  five shapes, so nothing on this page moved -- but the rule was right for the
  wrong reason until it was checked.

The practical point is that **a derivation is usually much smaller than the work
that produced the number.** Refusing the 930 cost nothing, because
``--second-cause`` already existed; verifying the ``acc`` claim cost four minutes
and overturned an explanation several documents rested on. That is why "a number
travels with its derivation, or it does not travel" is a workable rule rather
than a pious one.
