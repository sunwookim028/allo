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

   An independent census from the ACT rebuild reports 930 nests also violating
   the RAW-distance contract, and an encodable count of 5 rather than 3. Neither
   is reproduced here: this page's numbers are measured with **this** encoder by
   ``histogram.py``, and by first cause the RAW distance accounts for 2 (and for
   8 after the position check is dropped). The two censuses may be counting
   different quantities -- every constraint evaluated independently, versus
   re-censusing after relieving one -- and that is being reconciled. Until it
   is, these are the numbers with a derivation attached.

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
