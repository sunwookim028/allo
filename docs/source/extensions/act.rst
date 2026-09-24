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

#################################
ACT, and Why We Cite It
#################################

.. note::

   **ACT is Kai Shao's work**, developed at https://github.com/kkkaishao/allo
   (the ``kai`` remote), branch ``act``. This fork's ``chia-codesign`` lineage
   imports it at commit ``3c1ad38d`` as merge ``29cb1d99``; ``ATTRIBUTION.md``
   at tag ``chia-codesign-final`` (``629c2767``) records the terms. This page
   audits that work **from the outside** to decide whether to connect it to
   :doc:`/designs/tinytpu_isa`. It cites ACT; it does not republish it. ``main``
   has no ACT sources.

Audited 2026-09-22 against ``chia-codesign-final``: ``allo/exp/dsa/`` is 12
files, 8,347 lines.

The other side of that connection -- the workload specs a compiler would be
given and the judge that decides whether what comes out is right, legal and
fast -- is on :doc:`act_specs`, and contains no part of ACT.


Verdict
=======

**Rebuild the core against this fork's abstractions, using ACT's algorithms as
the reference; cite ACT, copy nothing.** That is what ``act/`` is. The decision
was forced rather than preferred, and the reasoning is worth keeping because it
is the argument any future port has to answer:

1. **A port could not have been validated.** ACT's suite does not run on this
   host -- see `It Cannot Be Run Here`_ -- so porting would have meant carrying
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
hard and genuinely good, and ``act/`` has none of them. ``act/`` is a mapper, a
cost model and a lowering, and it stops where ACT's interesting work begins.
When the bindings for the chia lineage can be built, the matcher is the piece to
revisit first -- not the mapper.

What is kept from ACT, deliberately: the interface. ``act.nest.Loop`` carries
ACT's four field names in ACT's order, so a real ``mapping.Mapping.loops`` can
be handed to ``act.search`` unchanged, and ``Priced.cost`` is ACT's
``(makespan, emits)`` lexicographic pair.


What ACT Is
===========

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


The Mapper
==========

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


It Cannot Be Run Here
=====================

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

Measured, 2026-09-22
--------------------

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
--------------------------------------------------------

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
``act.schedule.run`` to, against two solvers that do not share its algorithm: a
least-fixpoint iteration over the constraint system, and brute force over every
start vector for the smallest cases.

Also worth recording for a future port: ``epoch.py`` is 480 lines with **zero**
``ir.`` uses, so it is the most portable piece in the package, and ``check.py``
sits directly on it.


The Rebuilt Core
================

Target-independent, pure python, and deliberately **not** under ``allo/``,
because ``allo/__init__.py`` imports ``allo._mlir`` unconditionally and the core
has to stay testable without a build of the bindings:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - module
     - what it is
   * - ``act/workload.py``
     - an einsum over named ranks plus a pointwise epilogue. It evaluates itself
       with ``np.einsum`` over subscripts derived from the rank tuples, so **a
       spec is its own gold** -- one object drives the mapspace, the lowering
       and the correctness check.
   * - ``act/workloads.py``
     - the registry. Adding a workload is adding one entry.
   * - ``act/nest.py``
     - ACT's four-field ``Loop``, exact coverage, the intrinsic peel, and
       ``Refused(cause, detail, also)``. A refusal carries **every** cause the
       nest violates, so the census does not depend on the order the checks
       happen to run in.
   * - ``act/mapspace.py``
     - factorings x permutations over any rank set, with the innermost band
       supplied by the target. ACT's ``mapspace.py`` is dead code, so this is
       written fresh rather than revived.
   * - ``act/machine.py``
     - a machine as data: spaces, units, and one opcode row per instruction
       kind, giving the work each unit does per issue and the regions it reads
       and writes.
   * - ``act/schedule.py``
     - concurrent units, each sequential and in program order. ``Priced.cost``
       is ACT's ``(makespan, emits)``.
   * - ``act/search.py``
     - enumerate, lower, price, rank, and count the refusals by cause.
   * - ``act/target.py``
     - the five questions a machine must answer: its intrinsics, how to lower a
       nest, its steps, its emit count, and how to verify a program.

The machine model is declarative on purpose, and that is the part aimed at the
library goal rather than at this one design. TinyTPU-isa's declaration is 70
lines of data in ``act_machine.py``, and every number in it is read off
``assemble``'s header -- the per-unit work counts the hardware is actually
promised -- plus the sequencer's ``II=5``. A second architecture is a second
declaration, not a second generator.

The TinyTPU-isa target
----------------------

``examples/accelerator/tinytpu_vitis/act_target.py`` derives what the earlier
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

One command
-----------

.. code-block:: bash

   python examples/accelerator/tinytpu_vitis/act_compile.py gemm.relu 16x16x16
   python examples/accelerator/tinytpu_vitis/act_compile.py --gate
   python examples/accelerator/tinytpu_vitis/act_compile.py --list

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

Reconciling the encodable count: 5 against 3
--------------------------------------------

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

What this flow should emit next
-------------------------------

Today ``act/`` chooses a mapping for a **fixed** design and emits instructions
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
on its own. ``act/machine.py`` is deliberately the same information a generator
for that form would need: units, the work each does per issue, and the spaces
they read and write. Turning that declaration into ``func.func`` units is the
next piece of work, and nothing here forecloses it.

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

The ranking, checked against cosim
----------------------------------

``act_cosim.py`` measures the mappings the search ranks, and
``--baseline`` measures ``isa_dsl.gemm_program`` for the same shape beside them:

.. code-block:: bash

   TPU_PRJ=/tmp/act-cosim.prj python act_cosim.py gemm 4x4x4 --top 1 --baseline

Measured 2026-09-22 on this host, one synthesis per project, default testbench:

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
     - **172**
   * - ``gemm`` 4x4x4, the search's choice
     - 8
     - 4
     - 40
     - **169**
   * - ``gemm`` 8x8x8
     - 13
     - 10
     - 115
     - **262**
   * - ``gemm`` 12x12x12
     - 13
     - 18
     - 227
     - **418**
   * - ``gemm`` 16x16x8
     - 13
     - 16
     - 261
     - **484**
   * - ``gemm`` 16x16x16
     - 13
     - 28
     - 453
     - **686**
   * - ``gemm.relu`` 16x16x16
     - 14
     - 32
     - 517
     - **750**

The ``gemm`` rows reproduce **172 / 262 / 418 / 484 / 686** exactly, all five,
so the harness is the one those figures came from. (That was the published row
when this was measured. It moved to **171 / 261 / 417 / 483 / 685** on
2026-09-22 when the memory sizing became derived, and to
**175 / 265 / 421 / 482 / 674** on 2026-09-24 when ``QD=16`` became the
default channel depth; the reproduction above is of the design as it then
stood, and is not a disagreement.) The
``gemm.relu`` row settles what they are: **plain** ``gemm``, because
``cosim.py``'s ``testbench(M, K, N)`` leaves ``relu`` at its default.
``gemm.relu`` at 16x16x16 is 750, and 750 - 686 = **64**, exactly the four
``vrelu`` instructions' 64 ``accu`` rows. The delta is accounted for to the
cycle, which is the best evidence available that the units' work counts are the
right model of this machine.

The error bar a pick has to carry
---------------------------------

Over those six points the model fits ``cosim = 128 + 1.24 x makespan`` with a
worst residual of **34 cycles** -- computed from the table by
``act_machine.fit``, not typed in, and re-checked by a test that fails if the
model drifts from the stored points. So the model's *absolute* level is
predictable across shapes to a few percent once the intercept is allowed.

That is the wrong statistic for ranking, and saying so matters. Ranking compares
two mappings of **one** shape, where the intercept is common and only the
difference matters, and there is exactly **one** such pair measured: at 4x4x4 the
model put the two programs **1.25x** apart and the machine put them **1.02x**
apart, same order. The model therefore overstated the gap by roughly an order of
magnitude while getting the direction right.

``act_compile.py`` prints this wherever it reports a pick, rather than leaving
it on this page:

.. code-block:: text

   chosen: N4>K4  cost (517, 32)
   margin over the runner-up 1.24x. That margin sizes nothing: cosim ~ 128 +
   1.24 x makespan over 6 measured points, worst residual 34 cycles, but the
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
-------------------------------------------------

At four of the five shapes the search's choice **is** ``isa_dsl.gemm_program``,
word for word. At 4x4x4 it is not: every tile count is 1 there, so the mapspace
offers a single nest with no emitted loops, and the program it lowers drops a
trip-1 hardware loop the hand-written generator keeps -- 8 static instructions
against 10, the same 4 dynamic issues, and a model cost of 40 against 50.

Cosim, in the table above: **169 cycles against 172**, both exact. The
hand-written program reproduces the published 172 for that shape to the cycle,
and the search's program is 3 cycles faster. The model predicted a 10-cycle
saving and the machine gave 3, which is the expected direction of error for a
model that charges a fixed ``II`` per fetch and no overlap.

That is the whole shape of what a mapper buys on this machine today: it
reproduces a carefully hand-tuned choice where that choice is right, and it
removes overhead the generator could not see, because the generator has no cost
model at all. Three cycles in 172 is a small win, and it is the honest size of
the win -- what matters is that it is a *measured* win, chosen by a cost model
and confirmed by RTL, with the functional check passing in both cases.


The Co-design Loop This Argues For
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



.. _act-earlier-corrections:

Earlier measurements and corrections
====================================

Superseded numbers and claims this page has withdrawn. None of it is the
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
  is for; see `Where the cost model is honest and where it is not`_.
