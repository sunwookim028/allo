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

######################################
ACT: Results, Findings and Corrections
######################################

Measurements, findings and withdrawn claims for the two ACT pages -- the mapper
on :doc:`act` and the workload specs and judge on :doc:`act_specs`. Nothing here
is reference material; the current state of either tool is on its own page. Each
figure is dated where it matters, and the corrections at the foot are kept so
that they stay checkable.

Why the core was rebuilt rather than ported
===========================================

**Rebuild the core against this fork's abstractions, using ACT's algorithms as
the reference; cite ACT, copy nothing.** That is what ``allo/act/`` is. The decision
was forced rather than preferred, and the reasoning is worth keeping because it
is the argument any future port has to answer:

1. **A port could not have been validated.** ACT's suite does not run on this
   host -- see :ref:`act-cannot-run` -- so porting would have meant carrying
   code we could neither execute before the move nor compare against after it.
   Every bug would have been indistinguishable from a porting mistake.
2. **The parts we want are not separable.** ``mapspace`` imports ``mapping``
   imports ``search`` imports ``allo._mlir.ir``; with ``allo/__init__.py``
   bypassed, the only module in ``allo/exp/dsa/`` that imports cleanly is
   ``errors.py``. Extracting the mapper means rewriting its imports anyway,
   which is most of the work of rewriting it.
3. **Its upper half has never run.** ``tests/dsa/`` covers the frontend,
   matcher, solvers, planner and oracle, and covers **none** of ``mapping.py``,
   ``mapspace.py``, ``epoch.py`` or ``check.py``. No ``Binding`` is constructed
   anywhere in the tree. A port would import untested code into a tested
   design.
4. **The target is a library of composable IPs, not this one design.** A port
   inherits ACT's flat step list, in which "time" is a position in one Python
   list (``search.py:2357``), an operand's address is a single scalar offset
   (``mapping.py:966``) and the nest is fully unrolled (``mapping.py:904``).
   None of that shape survives a second architecture.

What is given up by not porting, stated plainly: ACT's pattern matcher (TOSA DAG
to instruction, a memoized tree DP that is globally optimal because multi-use
values are forced cut points), its residence and layout solvers, its Belady
spiller, and its MLIR emission. Those are the parts of ACT that are genuinely
hard and genuinely good, and ``allo/act/`` has none of them. ``allo/act/`` is a mapper, a
cost model and a lowering, and it stops where ACT's interesting work begins.
When the bindings for the chia lineage can be built, the matcher is the piece to
revisit first -- not the mapper.

What is kept from ACT, deliberately: the interface. ``allo.act.nest.Loop`` carries
ACT's four field names in ACT's order, so a real ``mapping.Mapping.loops`` can
be handed to ``allo.act.search`` unchanged, and ``Priced.cost`` is ACT's
``(makespan, emits)`` lexicographic pair.


ACT's mapper, as the code has it
================================

``mapping.Loop(rank, factor, level, spatial)`` is a frozen 4-field dataclass
(``mapping.py:122-131``) and ``Mapping(ranks, loops, bypass)`` is frozen with the
nest as a persistent field (``mapping.py:147-149``). So the chosen nest **is**
readable back out: ``search(binding, op).best.mapping.loops``. That much of our
standing plan is real.

Everything else about it is worse than we thought.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - claim
     - what the code says
   * - "the mapper is a loop-nest mapper"
     - True, but the **enumeration is entirely in** ``mapspace.py:102-174``.
       ``mapping.py`` contains no enumeration: it parses one given nest
       (``from_timeloop``, 39 lines) and lowers it.
   * - "``mapping._lower`` discards the loops"
     - Understated. ``mapping.py:904`` is
       ``for point in itertools.product(*(range(loop.factor) for loop in nest))``
       -- the nest is **fully unrolled**, temporal and spatial alike, so program
       size is the product of the factors. Below it an operand's address is
       already a single scalar element offset (``mapping.py:966-970``), so the
       multi-dimensional index structure an AGU needs does not exist there at
       all.
   * - "``mapspace.py`` is dead code"
     - True, and it is not alone. ``git grep mapspace`` outside the file itself:
       zero hits, here and on ``kai/act``. One commit ever ("checkpoint",
       2026-08-19). Of ``mapping.py``'s 1,292 lines the only live import is
       ``lower_site`` at ``search.py:2690``, reachable only when ``mapping_for``
       is passed to ``compile_program`` -- which nothing does. **And no code
       anywhere constructs a** ``Binding``, which every entry point requires.
       ``from_timeloop``/``to_timeloop`` have no callers.
   * - "factorisations x permutations x spatial splits"
     - Only with ``instances > 1``. At the default ``instances=1`` there are
       **no spatial slots at all** (``mapspace.py:139-141``), and ``instances``
       is the one number the ISA cannot declare.
   * - "``Priced.cost`` is ``(makespan, emits)``"
     - Confirmed, ``mapspace.py:189-196``. ``emits`` is ``len(program.emits)``
       and is only a lexicographic tie-break. Nothing in ``allo/exp/dsa/``
       models instruction-memory capacity. Pricing a candidate calls
       ``assemble`` -- ACT's **whole** codegen and allocator -- so its ranking
       is unavailable without the machinery we wanted to bypass.

Also: ``_derive`` supports exactly two operations, ``tosa.matmul`` and
``tosa.conv2d`` (``mapping.py:526``), and refuses conv2d with padding or a
non-zero bias.

**Dependencies.** MLIR python bindings are unavoidable for in-place use, three
independent ways: ``core.py:28``, ``search.py:48-49``, and ``search.py:83``
pulling in ``oracle.py``, which imports ``ExecutionEngine`` and ``PassManager``.
``sympy`` and ``ml_dtypes`` come along transitively although the mapper never
uses them. A **port** of the mapper alone -- ``Loop``, ``Mapping``,
``Dataspace``, a re-modelled ``Binding``, ``mapspace``'s enumerator, the
coverage/intrinsic half of ``Mapping.check``, and the pure geometry helpers
``_split_body``/``_weights``/``_span``/``_tile``/``_project``/``_packing``/
``_run_offsets``/``_reduction_ranks`` (~105 lines of the last group) -- is on
the order of **250-350 lines** and would be free of MLIR, sympy, ml_dtypes and
the JIT. That is the honest size of "use ACT as a mapper only".


Measured, 2026-09-22
====================

Run in a detached worktree at ``chia-codesign-final``, with the ``allo`` env
(python 3.12.13, pytest 9.1.1):

.. code-block:: text

   $ PYTHONPATH=$WT python -m pytest tests/dsa/ -q -p no:cacheprovider
   collected 9 items / 27 errors
   !!!!!!!!!!!!!!! Interrupted: 27 errors during collection !!!!!!!!!!!!!!!!
   $ ...  --continue-on-collection-errors
   9 passed, 27 errors in 0.99s

**One file of 28 runs.** ``tests/dsa/test_tinytpu_agent_policy.py`` (4 test
functions, 9 parametrised items) passes; it imports no ``allo`` and tests an
agent sandbox policy, not an ACT algorithm. The other 27 files hold 226 of the
230 test functions.

**The failures are one cause wearing twenty-seven hats, and it is not a bug in
ACT.** Every collection error is the same meta-cause -- no build of that
lineage's ``mlir/`` -- which presents as four faces in sequence:

.. list-table::
   :header-rows: 1
   :widths: 8 92

   * - files
     - exception, at its originating line
   * - 27
     - ``ModuleNotFoundError: No module named 'allo._mlir.schedule'`` at
       ``allo/schedule/keys.py:10``
   * - 27
     - after shimming that: ``ImportError: cannot import name 'allo' from
       'allo._mlir.dialects.transform'`` at ``allo/schedule/script.py:20``
   * - 27
     - after shimming that: ``ImportError: cannot import name 'ir_ext' from
       'allo._mlir._mlir_libs._allo'`` at ``allo/schedule/core.py:49``
   * - 1
     - bypassing the package root: ``ImportError: cannot import name
       'register_passes' from 'allo._mlir.dialects.allo'`` at
       ``allo/exp/dsa/oracle.py:34``

There is **no assertion failure anywhere in the run**. Shimming stops at the
third face, which is a C++ submodule of that lineage's own ``.so``. One subtlety
is worth recording: ``allo._mlir`` *does* resolve, but to a foreign lineage --
the editable install maps ``allo`` to another worktree, so ``allo`` came from
the checkout under test and ``allo._mlir`` from elsewhere. None of the eight
build trees on this host has ``allo/_mlir/schedule*``, and no ``_allo*.so`` on
the host contains the string ``ir_ext``.

**On 238 / 40 / 0.** That figure exists **only in the message of commit**
``cee57bc0`` ("Skip the torch-backed dsa tests instead of failing them"; 2 files,
+3 -6, replacing bare ``import torch`` with ``pytest.importorskip``). Nothing
committed records it -- no log, JSON, ``.rst`` or ``.md``. The suite and the code
under test are byte-identical between that commit and the tag, so the claim is
about exactly this code; it is neither confirmed nor refuted here, only
unreproducible on this host. **243 / 0 / 35** appears nowhere in git at all and
remains uncitable.

Census, which needs no build: **28 files, 230 test functions, 32
``pytest.importorskip`` sites** (23 on ``torch``, 7 on ``torch_mlir.fx``, 2 on
``ml_dtypes``), and no ``pytest.mark.skip`` anywhere. ``grep -rn makespan
tests/`` is still zero hits.


The scheduler's guarantee, and the claim that was not it
========================================================

``Priced.cost = (makespan, emits)`` is confirmed, but it lives in
``mapspace.py:189-196`` -- the dead file -- not in the scheduler. The scheduler
is ``epoch.schedule`` (``epoch.py:432``), and it is **one forward
order-preserving greedy ASAP pass**: no search, no backtracking, no priority
heuristic. Its own docstring says it derives the sigma the emitted program
*has*, and does not search for a better one.

A claim this project once made for ``epoch.schedule()`` -- that it reproduces
its makespan exactly, 100 == 100, over 200,000 random topological orders --
**is withdrawn**: it describes a property ACT never claims and could not have.
The argument is in `Earlier measurements and corrections`_.

What is true and testable is pointwise minimality. Fix the epochs, the unit
assignment and each epoch's issue and depth; then the derived start times are
the **pointwise minimum** over every assignment satisfying (a) non-negativity,
(b) ``start_dst >= finish_src`` for each derived dependence, and (c)
``start_later >= start_earlier + issue_earlier`` for consecutive epochs on one
unit. That is the proposition ``tests/act/test_schedule.py`` holds
``allo.act.schedule.run`` to, against two solvers that do not share its algorithm: a
least-fixpoint iteration over the constraint system, and brute force over every
start vector for the smallest cases.

Also worth recording for a future port: ``epoch.py`` is 480 lines with **zero**
``ir.`` uses, so it is the most portable piece in the package, and ``check.py``
sits directly on it.


Reconciling the encodable count: 5 against 3
============================================

This flow reports **5** encodable nests at 16x16x16 where the earlier
``act_nest.py`` prototype and an independently built enumerator report **3**.
The five are ``N4>K4``, ``M2>N4>K4``, ``M4>N4>K4``, ``N4>M2>K4`` and
``N4>M4>K4``; the three are the first three. So the difference is exactly the two
nests whose **column loop is outside the row loop**, and it is not a
disagreement about the machine: ``act_nest.py`` refused them explicitly, in a
message that called itself *"A limit of this file, not of the machine"*, because
it staged the activations only outside the whole nest. This flow stages them
immediately inside the innermost row loop wherever that loop sits, which costs
redundant re-staging when a column loop encloses it -- correct, and ranked worse
by the cost model, which is the right outcome for a mapper rather than a
refusal.

Both extra nests pass ``isa_ref.run``, and ``--gate`` verifies every encodable
mapping of every registered workload at every shape. **But that is not enough
to claim them, and the check that says so is below.** Read with the next
section, the reconciliation is: 5 encodable by the encoder and the reference
model, **3 confirmed on the RTL**, and the other tree's 3 is the better-grounded
number for any claim about the hardware.

The ranking, checked against cosim
==================================

``act_cosim.py`` measures the mappings the search ranks, and
``--baseline`` measures ``isa_dsl.gemm_program`` for the same shape beside them:

.. code-block:: bash

   TPU_PRJ=/tmp/act-cosim.prj python act_cosim.py gemm 4x4x4 --top 1 --baseline

Re-measured 2026-09-25 on this host at ``TPU_T=4 TPU_MAXDIM=16``, one
synthesis per project, default testbench, every row ``exact``:

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14

   * - program
     - static
     - dynamic
     - model
     - **cosim**
   * - ``gemm`` 4x4x4, ``isa_dsl.gemm_program``
     - 10
     - 4
     - 50
     - **175**
   * - ``gemm`` 4x4x4, the search's choice
     - 8
     - 4
     - 40
     - **172**
   * - ``gemm`` 8x8x8
     - 13
     - 10
     - 115
     - **265**
   * - ``gemm`` 12x12x12
     - 13
     - 18
     - 227
     - **421**
   * - ``gemm`` 16x16x8
     - 13
     - 16
     - 261
     - **482**
   * - ``gemm`` 16x16x16
     - 13
     - 28
     - 453
     - **674**
   * - ``gemm.relu`` 16x16x16
     - 14
     - 32
     - 517
     - **738**

The ``gemm`` rows reproduce **175 / 265 / 421 / 482 / 674** exactly, all five,
so the harness is the one those figures came from. (The table was first taken on
2026-09-22 against the then-published **172 / 262 / 418 / 484 / 686**, and it
reproduced that row exactly too. It moved to **171 / 261 / 417 / 483 / 685**
when the memory sizing became derived and to the present row on 2026-09-24 when
``QD=16`` became the default channel depth; the whole table was re-measured on
2026-09-25 rather than edited, which is why the ``model`` column is unchanged and
every ``cosim`` number moved.) The
``gemm.relu`` row settles what they are: **plain** ``gemm``, because
``cosim.py``'s ``testbench(M, K, N)`` leaves ``relu`` at its default.
``gemm.relu`` at 16x16x16 is 738, and 738 - 674 = **64**, exactly the four
``vrelu`` instructions' 64 ``accu`` rows --- the same 64 that 750 - 686 gave
before ``QD=16``. The delta is accounted for to the cycle and the accounting
survived a change that moved every other number on this page, which is the best
evidence available that the units' work counts are the right model of this
machine.

The error bar a pick has to carry
=================================

Over those six points the model fits ``cosim = 135 + 1.20 x makespan`` with a
worst residual of **34 cycles** (34.21, at ``gemm`` 16x16x8) -- computed from the
table by ``act_machine.fit``, not typed in, and re-checked by a test that fails
if the model drifts from the stored points. The re-measurement barely moved it:
it was ``128 + 1.24 x makespan``, worst 33.74, also at 16x16x8. A
two-parameter fit over makespan absorbs most of what ``QD=16`` did, which is the
opposite of what happened to ``act.cycles.estimate``'s one-variable fit over
critical work; makespan already carries the sequencer term, so it is closer to
monotone in the thing ``QD`` changed. So the model's *absolute* level is
predictable across shapes to a few percent at the large shapes once the
intercept is allowed --- and to 20 % at the smallest, which is the number a
small-shape claim has to carry.

That is the wrong statistic for ranking, and saying so matters. Ranking compares
two mappings of **one** shape, where the intercept is common and only the
difference matters, and there is exactly **one** such pair measured: at 4x4x4 the
model put the two programs **1.25x** apart and the machine put them **1.017x**
apart, same order. The model therefore overstated the gap by roughly an order of
magnitude while getting the direction right. Both cycle counts moved by +3 under
``QD=16`` (172 / 169 became 175 / 172), so the machine's margin is essentially
unchanged and this conclusion is not an artefact of the row it was first taken
on.

``act_compile.py`` prints this wherever it reports a pick, rather than leaving
it on this page:

.. code-block:: text

   chosen: N4>K4  cost (517, 32)
   margin over the runner-up 1.24x. That margin sizes nothing: cosim ~ 135 +
   1.20 x makespan over 6 measured points, worst residual 34 cycles, but the
   model's ORDER is validated on 1 same-shape pair -- gemm 4x4x4: model 1.25x
   -> cosim 1.02x, same order -- so it overstated the gap by an order of
   magnitude while getting the order right. Treat the pick as a ranking
   hypothesis and measure it with act_cosim.py.

Until more same-shape pairs are measured, **"the best mapping for this hardware"
is not a claim this flow can make** -- "the mapping this cost model ranks first,
whose order has been checked once" is. That distinction is the reason the
sequencer correction mattered: charging ``LOOP``/``ENDLOOP`` moved a pair from
509-vs-512 to 517-vs-642 and reversed it, so a term the model omits can flip an
order, and only a measurement closes it.

Where the search beats the hand-written generator
=================================================

At four of the five shapes the search's choice **is** ``isa_dsl.gemm_program``,
word for word. At 4x4x4 it is not: every tile count is 1 there, so the mapspace
offers a single nest with no emitted loops, and the program it lowers drops a
trip-1 hardware loop the hand-written generator keeps -- 8 static instructions
against 10, the same 4 dynamic issues, and a model cost of 40 against 50.

Cosim, in the table above: **172 cycles against 175**, both exact. The
hand-written program reproduces the published 175 for that shape to the cycle,
and the search's program is 3 cycles faster. The model predicted a 10-cycle
saving and the machine gave 3, which is the expected direction of error for a
model that charges a fixed ``II`` per fetch and no overlap. It was 169 against
172 before ``QD=16``: **the mapping-side saving is still exactly 3 cycles**, so
what ``QD`` changed is the machine's fixed cost and not what dropping a trip-1
loop is worth.

That is the whole shape of what a mapper buys on this machine today: it
reproduces a carefully hand-tuned choice where that choice is right, and it
removes overhead the generator could not see, because the generator has no cost
model at all. Three cycles in 172 is a small win, and it is the honest size of
the win -- what matters is that it is a *measured* win, chosen by a cost model
and confirmed by RTL, with the functional check passing in both cases.

The co-design loop this argues for
==================================

.. note::

   **It is built.** See :doc:`/extensions/codesign` for the loop as it exists,
   its frozen/editable split, its measured control (169 / 686) and its
   ``$0`` test suite. What follows is the argument that produced it; every
   point below is now enforced mechanically rather than stated.

The finding above is the shape of a publishable loop, because the thing that
collapsed the mapspace was a **hardware/ISA parameter**, not a compiler bug.

- **The agent edits the ISA and the microarchitecture**: the ``acc`` field's
  predicability, ``AGU_TERMS``, ``LOOP_DEPTH``, ``T``, ``MAXDIM``,
  ``IMEM_SIZE``, the accumulator RAW distance.
- **Frozen**: the mapper's objective and the program generator. The agent may
  not edit the thing that scores it. (This is the guard CHIA already needs and
  the reason its runs are auditable.)
- **The inner loop is exhaustive, not agentic**: for each hardware candidate,
  enumerate the mapspace and take the best encodable nest. The agent proposes
  hardware; enumeration answers what that hardware can run.
- **Feedback is a measurement**: cycles from RTL cosim (``cosim.py``) and
  resource estimates from HLS ``csynth``. PD is deferred. The pure-python proxy
  is for ranking inside one candidate only, and must never be the reported
  number.
- **Publishable rather than anecdotal** needs: the frozen-vs-edited split stated
  up front; every reported cycle count from cosim, with the exhaustive inner
  search making "the best mapping for this hardware" a claim rather than a
  sample; and a cost/benefit axis, since ``AGU_TERMS=4`` and a predicable
  ``acc`` both cost area that ``csynth`` can price.

.. _act-specs-kpn-verdict:

Is ``kpn_model.py`` good enough to be the cheap gate?
=====================================================

**No, and it does not claim to be.** As shipped it reports no cycle number at
all: it is a deadlock and protocol model with bounded FIFOs, it answers
"minimum channel depth" and "which process is blocked on which channel", and
its own docstring says data values are not modelled. Run over every shipped
program it prints ``minimum depth 1`` and ``KPN OK`` and no cycles
(reproduced in this session).

It can be *made* to produce a cycle number, and ``cycles.kpn_rounds`` does it
without touching the model: ``kpn_model.build`` hands back the process
network, and stepping it barrier-synchronously -- one channel action per
process per round -- counts the cycles a KPN with II=1 links and no memory
latency would take. That number is worse than free and worse than the header
counts. Single-feature least squares over the five published points, all
measured in this session:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - predictor
     - cost to compute
     - max \|error\| in sample, 5 points
   * - ``kpn_rounds`` (barrier-synchronous KPN)
     - 0.6-16 ms
     - **12.6%**
   * - dynamic instruction count
     - 0.1-0.3 ms
     - 12.9%
   * - ``accu`` header count
     - 0.1-0.3 ms
     - 13.1%
   * - ``dma_ld`` header count
     - 0.1-0.3 ms
     - 8.0%
   * - ``vru`` header count
     - 0.1-0.3 ms
     - 8.7%
   * - ``max`` over the five unit counts
     - 0.1-0.3 ms
     - **9.2%**

(Timed in this session: 116-314 us for ``cycles.estimate``, 0.6-15.5 ms for
``cycles.kpn_rounds``, on the same programs.)

So the KPN round count is *less* accurate than a single number the assembler
already computes, and fifty times more expensive: at ``gemm_16x16x16`` it
predicts 1484 rounds against 686 measured cycles, and the 1484 is dominated by
the 4x4 array's 32 weight-loader and PE processes each taking a round to pass a
token along a chain that the RTL walks in one cycle. A barrier-synchronous
scheduler charges a round per *process* per hop; the hardware charges a cycle
per *link*. Fixing that means modelling the array's chains as pipelines rather
than as processes, which is a different model, not a calibration of this one.

It is not a complete deadlock oracle either. On ``relu_16x16`` it reports
``minimum channel depth 1`` and no deadlock, and the RTL does not complete
(:ref:`act-specs-rtl-hang`). It models the channel *counts* -- who is promised
how much work -- and not the latencies and pipeline depths a real hang can
come from, so it catches assembler/microarchitecture mismatches and nothing
else. That is still worth having, and it is worth knowing its edge.

``kpn_model``'s real value is the layer it already occupies in the judge: it
is the ``protocol`` check in ``judge.py legal``, where a promised-work
mismatch is named with the blocked process instead of hanging the simulator.
It should stay there and stay out of the cycle gate.

.. _act-specs-gate-measured:

What the cheap gate is worth
============================

Measured with ``act/calibrate.py``, which runs one ``csynth_design`` and then
one ``cosim_design`` per program on that same RTL
(``dev/records/tinytpu/logs/cosim_act_corpus_sweep.log`` and
``dev/records/tinytpu/logs/cosim_act_relu_hang.log``).

.. warning::

   **This whole section is a measurement of the design at ``e24e433b``, at the
   then-default ``TPU_QD=8``, and it was NOT re-measured in the 2026-09-25
   refit.** Its ``cosim`` column is that design's (the ``gemm`` rows are
   172 / 262 / 418 / 484 / 686) and its ``estimate`` column is the
   ``173.2 + 1.621`` fit of that time. The shipped fit is now
   ``179.0 + 1.573`` against 175 / 265 / 421 / 482 / 674.

   The estimate column was deliberately **left alone** rather than recomputed.
   Recomputing it under the new fit while leaving the old ``cosim`` column in
   place would make every number in the ``error`` column a comparison between
   two different designs, which is worse than being out of date: the in-sample
   9.2 %, the out-of-sample 13.9 % and the 6.0 % mean below are all statistics
   of that pairing and only of it.

   **What it would take to re-derive them:** ``act/calibrate.py specs`` and
   ``act/calibrate.py variants`` re-run on the current design at
   ``TPU_T=4 TPU_MAXDIM=16`` --- one ``csynth`` plus a cosim per program over
   the twelve specs, of which one (``relu_16x16``) is expected not to complete
   (:ref:`item 24 <limitation-24>`). Until that runs, read the error column as
   a property of the model in the ``QD=8`` regime. It is tracked in
   ``chia_agent/control.NEEDS_REFIT``.

.. code-block:: bash

   TPU_PRJ=/somewhere/with/room python act/calibrate.py specs
   TPU_PRJ=/somewhere/with/room python act/calibrate.py variants

.. list-table::
   :header-rows: 1
   :widths: 26 10 8 10 9 9 8

   * - spec
     - critical
     - work
     - estimate
     - cosim
     - error
     - in fit
   * - ``gemm_4x4x4``
     - ``spm``
     - 9
     - 188
     - **172**
     - +9.2%
     - yes
   * - ``gemm_8x8x8``
     - ``vru``
     - 48
     - 251
     - **262**
     - -4.2%
     - yes
   * - ``gemm_12x12x12``
     - ``vru``
     - 144
     - 407
     - **418**
     - -2.7%
     - yes
   * - ``gemm_16x16x8``
     - ``vru``
     - 192
     - 484
     - **484**
     - +0.1%
     - yes
   * - ``gemm_16x16x16``
     - ``vru``
     - 320
     - 692
     - **686**
     - +0.9%
     - yes
   * - ``gemm_relu_16x16x16``
     - ``accu``
     - 384
     - 796
     - 750
     - +6.1%
     - no
   * - ``gemm_reuse_m_16x16x4``
     - ``vru``
     - 128
     - 381
     - 383
     - -0.6%
     - no
   * - ``gemm_reuse_n_4x4x16``
     - ``spm``
     - 36
     - 232
     - 269
     - -13.9%
     - no
   * - ``gemm_reuse_k_4x16x4``
     - ``spm``
     - 36
     - 232
     - 256
     - -9.5%
     - no
   * - ``batched_matmul_2x4x4x4``
     - ``spm``
     - 18
     - 202
     - 209
     - -3.2%
     - no
   * - ``row_reduce_16x16``
     - ``vru``
     - 128
     - 381
     - 371
     - +2.6%
     - no
   * - ``relu_16x16``
     - ``accu``
     - 192
     - 484
     - *never*
     - --
     - no

Two things fall out of the left column before the model is even discussed.
**The five published cycle counts reproduce exactly** -- 172 / 262 / 418 /
484 / 686, the published row *as it then stood* -- through a testbench that
compares all 256 bytes of ``C`` against
``isa_ref.run`` rather than only the ``M x N`` region, which is a stricter
check than the one the published numbers come from. And two of the corpus's
three non-GEMM einsums run on real RTL: ``batched_matmul_2x4x4x4``
(``bmk,bkn->bmn``) at 209 cycles and ``row_reduce_16x16`` (``mk->m``) at 371,
both with 0 of 256 bytes wrong. The third, ``relu_16x16`` (``mn->mn``), is
:ref:`the one that does not <act-specs-rtl-hang>`.

On the model: **in sample, max error 9.2% over the five points it is fitted
to; out of sample, max error 13.9% and mean absolute error 6.0% over the six
held-out specs that produced a cycle number.** The two worst cases are both
ones where ``spm`` is
the critical unit at a small work count, where what the design is actually
doing is filling the array's weight and activation chains -- a term the model
does not have, because ``max`` over the unit counts cannot see a pipeline it
does not model.

A 6% mean is good enough to sort candidates that differ by more than that and
useless for candidates that differ by less. The gate is set at 10% of the
reference submission's estimate for that reason: it is a filter on the
obviously-worse, not a ranking.

.. _act-specs-rtl-hang:

What the expensive judge found that nothing cheaper did
=======================================================

**Two of the programs the judge was given pass every cheap check and then do
not complete in RTL**: ``relu_16x16``'s submission, and the ``row_blocked``
variant of ``gemm_8x8x8``. For both, ``check_program`` accepts, ``kpn_model``
reports minimum channel depth 1 and no deadlock, the Allo simulator is
bit-exact against ``isa_ref``, and Vitis csim reports 0 of 256 bytes wrong --
and then ``cosim_design`` never finishes the transaction.

No deadlock is reported by Vitis' own detector, so this page does not call it
one, and nothing here locates the stall. What a hanging run's log looks like
beside a completing one is in `Earlier measurements and corrections`_.

The full characterisation, the bisection, and the two hypotheses it rules out
are :ref:`item 24 <limitation-24>` of the limitations register, with the repro
in ``tests/limits/item24_cosim_hang.py`` and the family in
``act/rtl_hang.py``. Two things from it are worth carrying here, because they
are about how a compiler should be judged rather than about this design:

- **The minimal case is sixteen instructions, fully unrolled, with no
  ``vrelu`` and no epilogue** -- plain int8 GEMM tiling. Nothing exotic has to
  be generated to reach it.
- **The hypothesis that an in-nest transfer is the trigger is measurably
  false.** :doc:`act` leaves open whether "a transfer inside the emitted nest
  rather than hoisted into a prologue" is what the failing mappings share. It
  is not: both non-completing programs here stage every transfer in a
  prologue, and both ``weights_reloaded`` mappings, which issue a ``dma_ld``
  between ``mvout``\ s inside the output nest, complete (256 cycles at 8x8x8
  and 638 at 16x16x16, 0 of 256 bytes wrong). A staging column would pass both
  failures and flag two mappings that run, so it is not the guard to use.

What it does to the judge is a tier above "is it fast", and that is
reference rather than a finding: :ref:`act-specs-tiers`.

.. _act-earlier-corrections:

Earlier measurements and corrections
====================================

Superseded numbers and claims these pages have withdrawn. None of it is the
current state; each entry is kept so the correction is checkable.

**The makespan claim that was withdrawn.** "``epoch.schedule()`` reproduces
its makespan exactly, 100 == 100, over 200,000 random topological orders" is
not merely absent from git -- **it describes a property ACT never claims and
could not have.** ``depends`` takes an edge's source to be the earlier stream
index (``epoch.py:372-386``), so permuting the stream changes which
dependences exist, which changes the program's meaning rather than its
schedule. What replaced it is pointwise minimality; see
`The scheduler's guarantee, and the claim that was not it`_.

**Earlier numbers on this page:**

- "A minimal ACT pilot was run, using a toy ISA." **No artifact exists.** There
  are no ``pilot`` hits in either repo's notes or git log. If it ran, it ran in
  a scratch checkout deleted 2026-09-07 (the CHIA checkpoint on
  ``chia-codesign``). Nothing in ``chia_runs/`` is it:
  ``20260905-060830/variants.jsonl`` is a single baseline line, and
  ``swarm-20260905-063857/`` is a 6-worker **CHIA LLM-agent** search (best
  126,432 -> 31,056 cycles on the ``dram`` hypothesis), which is not a mapspace
  search.
- "ACT has a mapspace search" -- it has a mapspace *file*, ``mapspace.py``, and
  nothing reaches it. The enumerator this page describes is the rebuilt one, not
  ACT's.
- "Our 9 opcodes" listed ``DMA_ST`` as expressible. The machine **refuses** it
  (``check_program`` rejects ``OP_DMA_ST``) and ``isa_dsl`` has no emitter for
  it. There are 10 encodings, 9 executable: 7 data opcodes plus
  ``LOOP``/``ENDLOOP``.
- "``gemm.relu`` at 16x16x16 is 49 instructions = 106 words flat, 17
  instructions = 42 words looped, headroom 14 words." Measured today: **32
  instructions / 72 words flat, 14 instructions / 36 words looped**, headroom
  **20** of ``IMEM_SIZE=56``.
- "The re-roll is the missing inverse of an identity we already test." It is
  not an inverse of anything: ``gemm_program`` exposes no tiling parameter, so
  the adapter is a new emitter, which is what ``act_target.py`` is.
- "Of 1,226 nests, 54 are refused by a limit of the prototype emitter." Fixed
  rather than corrected: ``act_target`` stages the activations at the innermost
  row loop wherever it sits, so those nests are now encodable or refused by a
  named hardware constraint, and five nests survive instead of three.
- "Three survivors, and the shipped ``N4>K4`` is the cheapest by the proxy."
  The proxy still says so. The ``(makespan, emits)`` model does **not**: it puts
  ``M2>N4>K4`` first by 0.6%, which is inside any honest error bar for a model
  that charges no row-level overlap. That disagreement is what ``act_cosim.py``
  is for; see :ref:`act-cost-model-honesty`.


What a hanging cosim run looks like in the log
----------------------------------------------

What that looks like, and what it does not: a completing run of the same
design prints two progress lines and a ``$finish`` --
``0 / 1 @ "109000"``, then ``1 / 1 @ "777000"``, then
``$finish called at time : 796590 ps`` for a 198-cycle program. A hanging run
prints the **first** line and never the second. ``109000`` is picoseconds and
is simply where Vitis makes its first periodic report, so it is the same
number in every log, passing or hanging; it is not where the design stops, and
nothing here locates the stall.
