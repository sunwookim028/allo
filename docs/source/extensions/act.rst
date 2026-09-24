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

##############################
ACT and the TinyTPU-isa Mapper
##############################

``allo/act/`` is a loop-nest mapper for :doc:`/designs/tinytpu_isa`. Given a workload
spec and a shape it enumerates the mapspace, lowers each nest to the machine's
instruction words, prices what it lowered, and prints the ranked mappings with a
census of why the rest were refused. It is target-independent pure python whose
own imports are numpy only --- though it moved from ``act/`` at the repository
root to ``allo/act/`` on 2026-09-24, so importing it now runs
``allo/__init__.py`` and **does** need this checkout's compiled MLIR bindings,
which it did not before. It is a rebuild of the core of **ACT**
-- Kai Shao's accelerator-compilation work -- against this fork's abstractions,
using ACT's algorithms as the reference and copying none of its code. The corpus
it compiles and the judge that grades the result are on :doc:`act_specs`; the
audit that produced the rebuild decision, the measurements and every withdrawn
claim are on :doc:`act_results`.

.. note::

   **ACT is Kai Shao's work**, developed at https://github.com/kkkaishao/allo
   (the ``kai`` remote), branch ``act``. This fork's ``chia-codesign`` lineage
   imports it at commit ``3c1ad38d`` as merge ``29cb1d99``; ``ATTRIBUTION.md``
   at tag ``chia-codesign-final`` (``629c2767``) records the terms. These pages
   audit that work **from the outside** to decide whether to connect it to
   :doc:`/designs/tinytpu_isa`. It cites ACT; it does not republish it. ``main``
   has no ACT sources.

Audited 2026-09-22 against ``chia-codesign-final``: ``allo/exp/dsa/`` is 12
files, 8,347 lines.

Quick start
===========

The fork's usual environment: the ``allo`` env with ``LLVM_BUILD_DIR``
exported (``CLAUDE.md``).

.. code-block:: bash

   cd examples/tinytpu
   python act_compile.py gemm.relu 16x16x16   # map one workload at one shape
   python act_compile.py --gate               # verify every encodable mapping
   python act_compile.py --list               # the registered workloads

The first prints the chosen nest with its ``(makespan, emits)`` cost, the margin
over the runner-up together with the error bar that margin has to carry, and the
refusal census below -- 1,226 nests in 1.3 s for ``gemm.relu`` at 16x16x16.
``--gate`` compiles 12 problems and verifies **every** encodable mapping of each
against numpy in 1.3 s, printing one grep-able verdict line.

Mappings this flow reports are *encodable*, not *confirmed*: the RTL measurement
is a separate command, and the difference is not cosmetic -- see
:ref:`act-encodable-tiers`.

.. code-block:: bash

   TPU_PRJ=/tmp/act-cosim.prj python act_cosim.py gemm 4x4x4 --top 1 --baseline

How it works
============

What ACT Is
-----------

An accelerator ISA is declared with decorators, and a TOSA program is compiled
onto it.

- ``@isa.instruction(src, dst)`` on ``def <mnemonic>(I)``. Inside,
  ``@I.access`` (a strided / contiguous / view / layout pattern) and
  ``@I.compute`` (a TOSA DAG) are **both required** -- omitting either raises
  ``AcceleratorDescriptionError`` (``core.py:967-970``). ``@I.schedule``
  (a finite domain plus a legality predicate) and ``@I.expand`` are optional.
- ``ISA.buffer``/``global_``/``scalar``/``vector``/``tile``/``hbm`` declare the
  memory hierarchy; ``ISA.bind``, ``latency``, ``network``, ``configures`` tie
  instructions to ``@unit`` kernels.
- ``ISA.compile_program(source, mapping_for=None)`` emits an MLIR module whose
  ``func @main`` is a **straight-line** sequence of ``allo.emit`` ops
  (``codegen.py:543-576``; ``@inspect`` points add ``call`` anchors, and the
  inspected buffers are the function's memref arguments). No control flow.
- ``@isa.oracle`` is a functional simulator, JIT-executed through the MLIR
  execution engine.

Two things we previously said about ACT that are **wrong**:

- *"A minimal ACT pilot was run."* No artifact exists in either repo.
- *"ACT has a mapspace search."* It has a mapspace *file*. See below.

The one machine ACT has ever targeted, chia's TinyTPU
(:doc:`/extensions/chia`), is **not a loop machine**: a flat PC walk, no loop
stack, fp32, 4-5 ``i32`` instruction words, 11 ACT-declared instructions. Ours
(:doc:`/designs/tinytpu_isa`) is int8/int32, two packed ``UInt(64)`` words, a
PC plus a 4-deep loop stack and a 3-term AGU. They share no code -- the
branches' merge base ``76130c63`` predates both, and the two ``microarch``
files share only their SPDX header. So ACT emitting flat streams is not an
oversight; ours would be the first loop machine it met.


The rebuilt core
----------------

Target-independent, pure python, and deliberately **not** under ``allo/``,
because ``allo/__init__.py`` imports ``allo._mlir`` unconditionally and the core
has to stay testable without a build of the bindings:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - module
     - what it is
   * - ``allo/act/workload.py``
     - an einsum over named ranks plus a pointwise epilogue. It evaluates itself
       with ``np.einsum`` over subscripts derived from the rank tuples, so **a
       spec is its own gold** -- one object drives the mapspace, the lowering
       and the correctness check.
   * - ``allo/act/workloads.py``
     - the registry. Adding a workload is adding one entry.
   * - ``allo/act/nest.py``
     - ACT's four-field ``Loop``, exact coverage, the intrinsic peel, and
       ``Refused(cause, detail, also)``. A refusal carries **every** cause the
       nest violates, so the census does not depend on the order the checks
       happen to run in.
   * - ``allo/act/mapspace.py``
     - factorings x permutations over any rank set, with the innermost band
       supplied by the target. ACT's ``mapspace.py`` is dead code, so this is
       written fresh rather than revived.
   * - ``allo/act/machine.py``
     - a machine as data: spaces, units, and one opcode row per instruction
       kind, giving the work each unit does per issue and the regions it reads
       and writes.
   * - ``allo/act/schedule.py``
     - concurrent units, each sequential and in program order. ``Priced.cost``
       is ACT's ``(makespan, emits)``.
   * - ``allo/act/search.py``
     - enumerate, lower, price, rank, and count the refusals by cause.
   * - ``allo/act/target.py``
     - the five questions a machine must answer: its intrinsics, how to lower a
       nest, its steps, its emit count, and how to verify a program.

The machine model is declarative on purpose, and that is the part aimed at the
library goal rather than at this one design. TinyTPU-isa's declaration is 70
lines of data in ``act_machine.py``, and every number in it is read off
``assemble``'s header -- the per-unit work counts the hardware is actually
promised -- plus the sequencer's ``II=5``. A second architecture is a second
declaration, not a second generator.

The TinyTPU-isa target
~~~~~~~~~~~~~~~~~~~~~~

``examples/tinytpu/act_target.py`` derives what the earlier
``act_nest.py`` prototype hardcoded:

- the **row / reduce / column roles** from the spec's rank structure, not from
  the names ``M``, ``K``, ``N``;
- **placement** -- DRAM rows and column blocks, scratchpad rows, operand vregs,
  accumulator regions -- from the operand list, with a named ``capacity``
  refusal when it does not fit;
- the **staging point** from where the row loops sit, which drops the
  prototype's "m loops must be outermost" restriction. Those 54 nests are now
  either encodable or refused by a named hardware constraint.

``act_nest.py`` is removed rather than kept beside it: two emitters for one
machine is the usability defect this work exists to fix. Everything it proved is
now a committed test.

What ``acc`` actually is
------------------------

"``acc`` is a static field" is not quite right, and the sharper statement is the
useful one. ``acc`` is ``mm``'s ``f2``, and ``f2`` **is a legal AGU target**
(``AGU_F2 = 3``, and the sequencer's resolve adds into it without checking the
opcode). Driving it from the reduce loop therefore assembles -- for exactly two
k-tiles. Probed directly against ``check_program``:

.. code-block:: text

   Kt=1: ACCEPTED  f2 stream = [0]
   Kt=2: ACCEPTED  f2 stream = [0, 1]
   Kt=3: rejected -- instruction 5 (mm, loop ivs [2]): f2=2, must be 0
         (overwrite) or 1 (accumulate)

So the obstacle is not staticness but **additive monotonicity**: the AGU
computes ``base + iv*stride`` and the predicate a reduction needs is
``k != 0``, which is affine only where the loop has two trips. Everything else
needs the peel.

That names the cheapest hardware change that would return 1,150 nests to the
mapspace: make the ``acc`` term saturate (``min(f2, 1)``), or add a
predicated-``acc`` encoding that reads "accumulate unless ``iv_now[level]``
is 0". The bit already exists; only its source has to become
induction-variable-dependent, and ``check_program``'s ``f2 in (0, 1)`` bound
relaxes to accept the encoding. An explicit accumulator-zeroing opcode is the
alternative, and costs a pass over ``ar`` plus ``AR_RAW_DIST`` slack before the
first ``mm``.

Two constraints on where such a fix belongs, both found on the co-design track
and both corroborated here:

**A term on ``f2`` spends one of the three.** The probe above fits in
``AGU_TERMS=3`` only because it has a single column tile and so needs no term
for one: its terms are ``f2``, ``f0`` and ``f3``, already the whole budget. Add a
column loop and the accumulating ``mm`` wants a fourth and is refused. So
widening ``AGU_TERMS`` to 4 -- which on its own raises the encodable count while
changing no cycles, and therefore looks like area for nothing -- is better read
as the **prerequisite** for any AGU-resolved ``acc`` fix to work beyond one
column tile. The two changes are complements, and neither alone shows anything.

**The reference model does not need unfreezing, if the fix is resolved in the
AGU.** ``isa_ref.run`` consumes ``expand(prog)``, which is the AGU-*resolved*
field stream, so a clamp applied during resolution hands the unit
``f2 in {0, 1}`` and leaves the instruction's architectural meaning untouched.
Resolve it instead inside a unit's decode and ``f2=2`` acquires a new meaning,
which ``isa_ref`` rightly rejects. That is a real constraint on the design space,
not a detail of the harness.

The two ``acc`` rows are counted separately on purpose. Splitting the reduce
rank and moving it off the innermost slot are different asks of the hardware --
one needs a per-iteration predicate, the other needs a whole body duplicated --
and lumping them as one ``acc-peel`` bucket of 1,150 is what made this census
incomparable with an independently built one. Together they are still 1,150.

What this flow should emit next
-------------------------------

Today ``allo/act/`` chooses a mapping for a **fixed** design and emits instructions
for it. The stated end state is the other direction: a library of parametrized,
modular IPs that compose into different architectures. The useful fact, from the
same branch, is that **the composable IR already exists** -- ``df.customize``
emits each unit as a ``func.func`` with explicit ``!allo.stream<i32,4>``
arguments and an ``stypes`` attribute for port direction, with the region
constructing the streams and wiring by ``call``, and parametrized units
instantiated at two sizes in one program already work and are covered by the
dataflow test suite. What is missing is only the *source syntax*: ``Stream`` has
no ``__class_getitem__``, and stream ops are keyed to construct sites rather
than values.

So the emission target for a generated design is that IR form, not the
closure-nested source form this design uses -- where all seven units are
``@df.kernel`` closures inside one ``@df.region`` sharing about fifteen
region-scope streams by capture, so no unit is separable, importable or testable
on its own. ``allo/act/machine.py`` is deliberately the same information a generator
for that form would need: units, the work each does per issue, and the spaces
they read and write. Turning that declaration into ``func.func`` units is the
next piece of work, and nothing here forecloses it.

Reference
=========

The refusal census
------------------

Measured in this worktree, ``gemm.relu`` at 16x16x16, 1,226 nests in 1.3 s:

.. list-table::
   :header-rows: 1
   :widths: 10 10 80

   * - first
     - also
     - cause
   * - 897
     - 897
     - ``acc-split`` -- the reduce rank is split across two emitted loops, so
       the first partial sum is not a contiguous prefix at all
   * - 253
     - 253
     - ``acc-position`` -- the reduce rank is not innermost among the emitted
       loops, so peeling its first iteration would duplicate the whole body
   * - 55
     - 930
     - ``ar-distance`` -- rows below ``AR_RAW_DIST=4`` make the accumulating
       ``mm`` re-read its own row inside the pipeline window
   * - 13
     - 13
     - ``AGU_TERMS=3``, one instruction word's address-term budget
   * - 3
     - 3
     - ``LOOP_DEPTH=4``

Two census definitions, and which number is which
-------------------------------------------------

The two columns above are different quantities, and the difference has already
caused one figure to be relayed as if it were the other.

**First cause** (column one) is what a sequential enumerator naturally reports:
each nest attributed to the check that refused it first. Under this flow's check
order that is 897 / 253 / 55 / 13 / 3, summing with the 5 encodable nests to
1,226.

**Independent** (column two) asks each predicate of *every* nest on its own,
regardless of what else refuses it. That is where **930** comes from: 930 of the
1,226 nests have an intrinsic row tile below ``AR_RAW_DIST=4``. It is re-derivable
in one command, and it is the only sense in which that number is true:

.. code-block:: bash

   python act_compile.py gemm.relu 16x16x16 --census
   #    930  refuse   ar-distance
   #    897  express  acc-split
   #    253  express  acc-position

A **third** quantity exists and is the one a co-design track actually wants:
*what is left once you relieve a constraint*. Remove the ``acc-position`` check
and re-census, and the largest remaining obstacle is ``acc-split`` -- measured
independently in another tree as **897**, which is exactly this flow's
``acc-split`` count. Two independently built enumerators agreeing to the nest on
that number is the strongest cross-check either of them has.

Where the same two trees disagree is ``ar-distance`` as a *first* cause: 55 here
against 8 there. That is not a contradiction either. The other tree's emitter
refuses 274 nests for a restriction of its own (it re-stages the activations per
row tile and cannot interleave that with a column loop), and it refuses them
*before* reaching the RAW-distance check; this flow removed that restriction, so
those nests survive to be judged on rows. The lesson worth keeping: a first-cause
histogram is a property of the check order and of the emitter's limits, not of
the hardware, and only the independent census and the relieve-and-re-census
numbers are comparable across implementations.

Two kinds of gap: cannot express, cannot refuse
-----------------------------------------------

A failing build hides two opposite problems, and the census now sorts every
refusal into one of them (``act_target.CAUSE_KIND``). This vocabulary is from
the abstractions work on branch ``act-abstractions`` (``65ae98c6``), which found
that four of the five gaps it examined were of the second kind.

``express``
   The instruction word has no field, or no room, for what the nest asks. The
   design is fine and the encoding is not, and the repair is a wider word or a
   new field. ``acc-peel``, ``AGU_TERMS``, ``LOOP_DEPTH``, ``capacity``.

``refuse``
   The machine would accept the program and produce a wrong answer or hang.
   Only a hand-written check says no, and the repair is a checker. This is
   ``ar-distance`` -- and it is why ``check_program`` has to exist outside
   ``allo/`` at all, since ``s.dependence`` makes a claim that no Allo pass
   verifies.

At ``gemm.relu`` 16x16x16 that is **1,166 express against 55 refuse** by first
cause, and 1,166 against 930 counting every cause a nest violates.

The distinction cuts the other way too, and it is worth stating because it is
easy to misread ``acc-peel`` as a limit of Allo. It is not: expressed at the
Allo level, the predicate a reduction needs is ``scf.if`` on
``cmpi eq, index_cast(k), 0``, and **Allo builds that happily**. Nothing in
Allo refuses it and no primitive removes it. The refusal is our ISA's encoding
meeting a compiler that cannot check encodings -- which is exactly the hole
``allo/encoding.py`` on that branch fills. ``act_machine.ENCODING`` states
TinyTPU-isa's budgets in that record's field names, so the two sides state one
budget rather than two once the branches meet.

Limits and known failures
=========================

.. _act-cannot-run:

It cannot be run here
---------------------

``scripts/act-test-recipe.sh`` probes every precondition and prints a verdict.
As of 2026-09-22 four are missing:

- ``allo/_mlir`` must be built from *this* tree: ACT reaches
  ``allo._mlir.schedule`` (``mlir/python/allo/schedule.py``) and the ``allo``
  dialect's ``AlloISAOps``/``AlloISATypes``, none of which exist in ``main``'s
  ``mlir/``. Pointing the symlink at ``main``'s build fails at
  ``ModuleNotFoundError: No module named 'allo._mlir.schedule'``.
- That build reads ``$LLVM_BASE_DIR`` (not the ``LLVM_BUILD_DIR`` the dataflow
  simulator wants) and needs ``llvm-project`` at ``040a6419``. The 11 GB build
  on this host is ``6b09f739``, which is ``main``'s pin.
- ``mlir/CMakeLists.txt:17-23,35`` FATAL_ERRORs without CIRCT, and
  ``MLIRAlloRegisterEverything`` -- which the bindings link -- lists
  ``CIRCTHW``/``CIRCTComb``/``CIRCTSeq`` plus ``MLIRAlloMicroarch`` and
  ``MLIRAlloScheduling``, so it is not separable. ``externals/circt`` is an
  empty submodule directory and no ``CIRCTConfig.cmake`` exists anywhere on the
  filesystem.
- ``mlir/CMakeLists.txt:27-33`` FATAL_ERRORs without an OR-Tools **cmake
  package**. ``$HOME/chia-ortools`` is not one: it ships runtime ``.so`` files
  to keep an already-built tree importable.

Two corrections to the environment as it was described to us: the conda env is
**not** ``chia_env`` (python 3.10.19, ray + google-genai + mcp, and no numpy or
pytest -- it is the CHIA agent's env); and no ``cpython-314`` extension module
survives on this host, the tree that had them having been deleted. The
``allo`` env (python 3.12.13) does have all four pure-python deps --
numpy, sympy, ml_dtypes, pytest -- so the deps are not the problem, the bindings
are.

.. _act-encodable-tiers:

"Encodable" has three tiers of evidence
---------------------------------------

Attempting to cosim the two extra nests turned up something worth more than the
count. For ``gemm.relu`` at 16x16x16:

.. list-table::
   :header-rows: 1
   :widths: 42 29 29

   * - checker
     - shipped ``N4>K4``
     - ``N4>M2>K4`` (row-tiled)
   * - ``isa_ref.run`` (numpy, the ISA's meaning)
     - correct
     - correct
   * - ``check_program`` / ``assemble``
     - accepts
     - accepts
   * - ``kpn_model.run`` (channel protocol, bounded FIFOs)
     - runs, minimum depth 1
     - runs, minimum depth 1
   * - ``df.build(target="simulator")``
     - completes, correct
     - completes, correct
   * - Vitis **csim**
     - ``mismatches = 0``
     - ``mismatches = 0``
   * - Vitis **cosim** (RTL)
     - **750 cycles**
     - **did not complete**

Two row-tiled mappings were tried and behaved the same way: no completion, with
the simulator holding a full core, one of them for over half an hour, where the
shipped mapping's whole run -- synthesis, csim and cosim -- takes about two
minutes. No deadlock is *reported*, so this page does not call it one; what is
measured is that the run does not finish.

.. warning::

   An earlier version of this page said the failing runs "sit at
   ``Inter-Transaction Progress: 0 / 1``" as though that were a signature. **It
   is not.** ``109000`` is picoseconds and that line is simply Vitis's first
   periodic report, printed in every log including passing ones, which then
   print ``1 / 1`` and finish. The observation is only that the second line
   never comes; nothing measured locates the stall. See
   :ref:`item 24 <limitation-24>`, where the failure was reduced.

The lesson is the one this design's own history already taught once, when the
dataflow simulator passed a bug that only cosim caught: **four checkers agreeing
is not evidence about the RTL.** Two of those four are derived from
``assemble``'s header, so they cannot see an error in the header formula itself;
``isa_ref`` is a model of the ISA, not of the machine; and csim compiles the
units' C without their handshakes. Only cosim exercises the streams.

So a mapper on this machine can report two different things, and should say
which: *encodable* -- the encoder accepts it and the reference model agrees --
and *confirmed* -- the RTL ran it. ``act_compile.py`` reports the first, and its
``check`` column says so. Anything published as a property of the hardware needs
``act_cosim.py``.

One reassurance, and it is a test rather than a hope: at every shape and every
registered workload the mapping the flow **picks** is the one whose program
cosim has actually measured -- ``tests/act/test_tinytpu.py`` asserts it. The
unconfirmed mappings are ranked, reported and never chosen.

This is also the sharpest ``cannot refuse`` gap found in this work, and it is
not in the ISA. It is filed as :ref:`item 24 <limitation-24>`, reduced there to
a sixteen-instruction program by the workload-specs track, with a bisection.

**The suspicion this page raised is dead, killed twice.** It was that the
trigger is a data transfer *inside* the emitted nest rather than hoisted into a
prologue -- the one feature the failing mappings shared and the shipped one
lacked. From the small end, four programs built to have exactly that feature all
complete (169 / 189 / 171 / 186 cycles,
``tests/limits/item24_cosim_small_programs_complete.py``). From the failing end,
**both** non-completing programs stage every transfer in a prologue, and both
mappings that do stage inside the nest complete. In-nest staging is neither
necessary nor sufficient, which is why this flow ships no staging column: a
predicate that passes both failures and flags two programs that run is worse
than none. The bisection also rules out ``vrelu``, the hardware loop, and
monotonicity in size.

What survives is the consequence, not the cause. The mapspace this flow
enumerates cannot be trusted past the mappings cosim has actually run, which is
the real cost of the item and the reason it outranks widening any hardware
parameter.

**Five nests are encodable** (the prototype found three) and all five compute
the spec against ``isa_ref.run`` -- but only three are confirmed on the RTL; see
:ref:`act-encodable-tiers`. The shipped ``N4>K4`` mapping is one
of them, and ``act_target`` re-emits the shipped ``gemm``/``gemm.relu`` program
**word for word** at all five ``bench_isa.SHAPES`` -- the regression anchor,
committed as ``tests/act/test_tinytpu.py``.

``--gate`` compiles 12 problems and verifies **every** encodable mapping of each
against numpy in 1.3 s, printing one grep-able verdict line.

.. _act-cost-model-honesty:

Where the cost model is honest and where it is not
--------------------------------------------------

``makespan`` is a model, not a measurement. Every work count in it is read off
``assemble``'s header -- the counts the units are actually promised -- plus the
sequencer's ``II=5``, and the sequencer is charged for its ``LOOP``/``ENDLOOP``
fetches as well as the data issues, which ``expand`` drops. ``act_machine``
recovers those from the program by mirroring ``_trace``'s control flow and
nothing else; a test pins the data fetches to ``expand``'s own stream.

Charging the loop stack is the difference between a model that ranks and a model
that misleads. Without it the top two mappings at ``gemm.relu`` 16x16x16 came out
509 against 512 -- the model preferring a 60-emit mapping over the shipped
32-emit one by 0.6%. With it they are 642 against 517, the shipped mapping
first, and the whole ranking becomes monotone in the emit count. The correction
is grounded rather than fitted: the ``II=5`` the sequencer closes at *is* the
loop-stack recurrence, so a model that does not charge the loop instructions is
undercharging exactly the nests that use them most.

What it still does not model: dependence edges are finish-to-start at
instruction granularity, so no row-level overlap is charged. At ``gemm.relu``
16x16x16 the model says 517 where cosim measures 750, about 31% low. It ranks;
it does not measure.

Failure modes, named
--------------------

- **Scoring on ACT's cost model optimises a model.** ``Priced.cost`` is
  ``(makespan, emits)``; ``makespan`` comes from ``epoch.Sigma``, whose
  agreement with our RTL has never been tested, and nothing models instruction
  memory -- while ``assemble`` asserts ``len(words) <= IMEM_SIZE`` and ACT's own
  backend unrolls, so its program size is the product of the loop factors.
  Any loop scored on ACT's objective would reward a program our machine cannot
  hold.
- **The proxy invites the same mistake more cheaply.** It is a five-point
  regression; a loop that optimises it will find its residual.
- **An agent given both the hardware and the scorer will move the scorer.**
- **An inner search that is not exhaustive turns a hardware comparison into a
  search-quality comparison.** Keeping the mapspace small enough to enumerate is
  a requirement, not a convenience.

Results and history
===================

:doc:`act_results` carries the measurements, the findings and the corrections:
why the core was rebuilt rather than ported, what ACT's own mapper does when it
is read, the 2026-09-22 run of ACT's test suite, the ranking checked against
cosim, the error bar a pick has to carry, the co-design loop this work argued
for, and every claim these pages have withdrawn.
