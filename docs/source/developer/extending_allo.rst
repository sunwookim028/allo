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

.. _extending-allo:

##############################
Extending Allo's Abstractions
##############################

This page is for whoever -- person or agent -- is about to add something to
Allo. It is not a catalogue of features. It is a way of getting from a
*symptom* to the right kind of extension, and a standard the extension has to
meet before it is worth writing.

The standard comes from CAKE (arXiv 2608.12629): at an equal budget, generating
into a **typed IR** reached 1.144x of a reference while generating raw CUDA/PTX
reached 0.928x. The lesson this fork draws from that is not "use an IR". It is
three specific things, and they are the whole of this page:

1. a primitive arrives with its **analyses** and its **legality rule**, or it
   is not ready;
2. **construction should type-check** -- the mistake should be refused at the
   call that made it, not at the build twenty minutes later;
3. **cheap static gates run before expensive ones.**

.. important::

   **If you cannot finish the sentence "X is legal if and only if ...", stop.**
   Write the gap down on this page instead of writing the extension. An
   extension that cannot state its own legality rule will be *used* by an
   agent that cannot tell when it is lying, and the cost of that is measured
   below (see :ref:`extending-allo-dependence`).


Start from the symptom
======================

.. list-table::
   :header-rows: 1
   :widths: 40 24 36

   * - What you noticed
     - What kind of gap it is
     - Where the extension belongs
   * - You hand-coded a hardware limit (a field width, a stack depth, a term
       budget) in Python, outside ``allo/``, to decide whether a schedule was
       buildable.
     - Allo has no **target**. There is nowhere to write down what
       distinguishes this architecture from another.
     - A **type** (a capability record) plus a **pass** that checks it. Done
       once: see :ref:`extending-allo-encoding`.
   * - You copied a unit instead of instantiating it; or the same design exists
       several times, differing only in topology.
     - Allo has no **unit with an interface**. Composition is by lexical
       capture, so a unit cannot be named, imported or wired.
     - A **type** (a stream that can be a port) plus **frontend** support for
       calls that pass it. Open: :ref:`extending-allo-ports`.
   * - You wanted one iteration of a loop to do something different from the
       rest, and wrote two loop nests by hand.
     - Allo has no **index-set splitting**. ``split`` cuts a loop into a nest;
       nothing cuts an iteration space into pieces with different bodies.
     - A **schedule primitive** (dialect op + ``LoopTransformations.cpp``).
       Open: :ref:`extending-allo-peel`.
   * - A long backend run failed, or silently produced the wrong RTL, for a
       reason that was already visible in the MLIR.
     - A **gate** is missing, not a feature. The information was there and
       nothing looked at it.
     - A **pass** that refuses and never transforms. Add a rule to
       :ref:`extending-allo-encoding`.
   * - A primitive accepted something it should not have, and the damage
       appeared later somewhere else.
     - A primitive shipped **without its legality rule**.
     - Add the rule to the existing primitive. This is repair, not extension,
       and it outranks new features.
   * - Allo cannot express the thing at all, and you are not sure what the
       thing *is* yet.
     - Not a gap yet. It is a question.
     - Write it on this page with the evidence. An unnamed gap with a repro
       beats a named abstraction without one.

The distinction that matters most in that table is between **cannot express**
and **cannot refuse**. They look the same from inside a failing build and they
need opposite extensions. Establish which one you have before proposing
anything:

.. code-block:: python

   # Build the thing and look at the IR. If Allo emitted it, Allo can express
   # it, and what is missing is a refusal -- not a feature.
   s = allo.customize(kernel)
   print(s.module)

Four of the five gaps recorded below turned out to be *cannot refuse*. The IR
was already right. Only the frontend, or a checker, was missing.


.. _extending-allo-encoding:

The abstraction added here: ``Encoding``
========================================

``allo/encoding.py`` plus the ``s.encodable_on`` primitive in
``allo/customize.py``.

What it is for
--------------

An architecture differs from another architecture mostly in **what one
instruction word can carry**. A machine with a 3-term address generator and a
4-deep hardware loop stack refuses schedules that a machine with an
8-term generator accepts, and neither refusal is a property of the *kernel* --
it is a property of the pairing of kernel and target. Before this, Allo had
nowhere to write that down. ``configs={"device": "u280"}`` names a part number
for a vendor tool; it says nothing an Allo pass can check.

``Encoding`` is that record, and ``s.encodable_on(encoding)`` makes it a
standing obligation on a schedule:

.. code-block:: python

   from allo.encoding import Encoding

   LPU_LIKE = Encoding(
       name="deterministic-vliw",
       has_predicated_fields=False,       # instruction fields fixed at assembly
       requires_static_trip_counts=True,  # the sequencer's trip count is a field
       requires_affine_addressing=True,
   )

   s = allo.customize(kernel)
   s.encodable_on(LPU_LIKE)
   s.split("i", 3)        # raises EncodingError here, not at s.build()

The declaration is re-checked after **every** subsequent primitive, and the
error names the primitive that broke it. That is point 2 of the standard,
implemented as one call in ``wrapped_apply``.

The invariants it must preserve
-------------------------------

These are not style preferences. Breaking any one of them turns a gate into a
source of miscompiles, which is strictly worse than having no gate.

1. **A rule refuses; it never transforms.** ``allo/encoding.py`` imports no
   builder and constructs no op. A gate that also rewrites cannot be trusted as
   a gate, because a failure to refuse becomes a silent rewrite.
2. **An unset field constrains nothing.** ``Encoding()`` must be a no-op on
   every kernel in the tree, and there is a test that says so
   (``test_unconstrained_encoding_refuses_nothing``). Every existing target
   keeps working because every existing target declares nothing.
3. **A rule is decidable from the IR in front of it.** No rule may need a
   measurement, a vendor run, or a promise from the user. If a property is not
   decidable, it is not a rule -- see the obligation/rule split under
   :ref:`extending-allo-dependence`.
4. **A violation carries its repair.** ``Violation.repair`` is not
   documentation, it is the payload: the reader is usually an agent that has to
   act on the refusal without a catalogue. A refusal that does not say what to
   do instead is a dead end.
5. **The rule set is open and the rules are independent.** A rule is a
   generator over ``Site`` records added to ``RULES``; no rule may depend on
   another having run.

The rules that ship with it
---------------------------

Each is stated as the legality sentence that the standard demands.

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Rule
     - Legal if and only if
     - Caught at construction
   * - ``predicated-field``
     - every ``scf.if`` / ``affine.if`` inside an affine band has a condition
       that is **invariant** in all enclosing induction variables.
     - A reduction whose accumulator is guarded by its own reduction index
       (``if k == 0``).
   * - ``address-terms``
     - every ``affine.load`` / ``affine.store`` depends on at most ``N``
       distinct enclosing induction variables. (Distinct *variables*, not
       subscripts: ``C[i, i]`` linearises to one strided term, and MLIR has
       already dropped unused dims from the map, so the count is exact.)
     - A nest one level deeper than the address generator can stride.
   * - ``loop-depth``
     - no ``affine.for`` is nested more than ``N`` deep.
     - A tiling that outgrows the hardware loop stack.
   * - ``static-trip-count``
     - every ``affine.for`` has single-result lower and upper bound maps with
       no dims and no symbols.
     - ``s.split(axis, f)`` where ``f`` does not divide the extent -- which
       Allo otherwise accepts, emitting ``to min affine_map<(d0) -> (3, d0 *
       -3 + 10)>``.
   * - ``affine-addressing``
     - no ``memref.load`` / ``memref.store`` appears inside an affine band.
     - A subscript that fell off Allo's affine path. ``build_indices``
       (``allo/ir/builder.py:997``) takes that fallback **silently** on the
       first non-affine expression, so this is otherwise invisible.

How to add a rule
-----------------

Write the legality sentence first. Then:

1. Add the field to ``Encoding`` with a default that constrains nothing.
2. Write the analysis as a free function over a ``Site``, and reuse
   ``varying_induction_variables`` if the question is "does this vary per
   iteration" -- two of the five rules are the same analysis asked twice.
3. Write the rule as a generator yielding ``Violation``, add it to ``RULES``.
4. Add a **pair** of tests: one kernel refused, one kernel accepted for the
   same rule. A gate with only negative tests will happily refuse everything.

.. code-block:: bash

   python -m pytest tests/test_encoding.py -q     # 12 tests, ~2 s

What it does not do
-------------------

It does not model instruction *semantics*, so it cannot tell you whether a
band computes what an instruction computes; that is
:ref:`extending-allo-tensorize`, which is not ready. It does not model timing
or capacity. And it is a necessary condition only: passing every rule does not
make a schedule encodable, it only means these five reasons for refusing it are
absent.


The precedents in this tree
===========================

Read these before proposing anything. Two are good templates; one is a warning.

.. _extending-allo-dependence:

``s.dependence`` -- right mechanism, absent legality rule
---------------------------------------------------------

``allo/customize.py:834``, fork commit ``bbea2af0``, register item
:ref:`21 <limitation-21>`. This is the template for **how** to land a
primitive on this fork: about 112 lines of Python that attach a ``dependence``
attribute to a loop, 46 lines of C++ in ``EmitVivadoHLS.cpp`` to emit the
pragma, three tests in ``tests/test_vhls.py``. It closed a real defect and it
is the reason TinyTPU-isa's accumulator reaches II=1.

It is also the fork's own counter-example to point 1 of the standard, and its
docstring says so: ``dependent=False`` is *the programmer's claim*, and

    if the claim is false the RTL computes a wrong answer while every software
    simulation, which ignores the pragma, still passes.

The price of that is on the record: the claim is true only for programs that
never read an accumulator row within ``AR_RAW_DIST`` iterations of writing it,
and the only thing on this host that can see a violation is a ``TPU_TB=stress``
cosimulation. Nothing in Allo can.

The lesson is **not** "don't ship primitives with unprovable claims" -- an
escape hatch for what the scheduler cannot prove is exactly what the pragma is
for. The lesson is that such a primitive has two halves and must ship both:

* the **rule**, which is decidable and must be enforced: if a dependence *is*
  provable at a distance the claim denies, the claim is a lie and must be
  refused. The analysis already exists in C++
  (``analyzeDependency``, ``checkDependence``, ``mlir/include/allo/Support/Utils.h:111``)
  and is not wired to the primitive.
* the **obligation**, which is not decidable and must be *declared* as an
  obligation, so that whoever relies on it knows they are the one discharging
  it.

``s.dependence`` today ships neither half explicitly. Closing that is repair
work and it outranks a new feature.

``align_value`` -- an opt-in that admits it is not sufficient
-------------------------------------------------------------

``configs={"align_value": N}`` (``allo/backend/hls.py:356``, register item
:ref:`23 <limitation-23>`). The template for a *promise* to a downstream tool:
opt-in, off by default, no effect on any existing flow. Its register entry is
recorded as a **negative result** -- necessary and not sufficient for port
widening -- which is the right way to leave an extension that did not reach
its goal. Do that rather than quietly dropping it.

``check_perfect_affine_kernel`` -- the in-tree shape of a gate
--------------------------------------------------------------

``allo/autoscheduler/util.py:30``, with ``check_call_graph_acyclic``,
``check_all_functions_inlined``, ``check_single_producer_single_consumer``.
These are whole-program admissibility predicates that run before an expensive
MILP -- point 3 of the standard, already in the tree. Two limitations follow
them, and ``Encoding`` exists to fix both: they are ``assert``\ s (so they name
no site and offer no repair), and the constraints are **hard-coded** rather
than a property of a declared target, so nothing else can reuse them.

Refusal as a feature
--------------------

Fork commit ``c5a45338`` makes the emitter refuse a ``Stateful`` shared by more
than one function instead of emitting silent per-kernel copies. Turning a
silent miscompile into an error is a legitimate, landable contribution on its
own. It needs no new syntax.


.. _extending-allo-gaps:

The open gaps, ranked
=====================

Ranked by whether the extension makes **a second architecture expressible**,
per unit of implementation risk. The end state this serves is a library of
parametrized, modular units that compose into different architectural choices
-- a deterministic-dataflow pipeline, a tiled NPU, a column of AIE-style tiles
-- with any one accelerator being something the library *instantiates*.
An extension that removes a wart from the design we already have ranks below
one that lets a second design share a representation with it.

.. _extending-allo-ports:

1. A stream that can be a port, so a unit can have an interface
---------------------------------------------------------------

**The gap.** A ``@df.kernel`` cannot name its stream ports. Streams are
declared as annotations in a ``@df.region`` body and reached from kernel bodies
by lexical name, so a unit's interface is not a property of the unit -- it is
derived, by whole-region analysis (``move_stream_to_interface``), from the
region it is pasted into. Therefore no unit can be lifted out, imported,
instantiated twice, or tested alone, and a topology cannot be written down
separately from the units it wires.

**The evidence.**

* ``examples/accelerator/tinytpu_vitis/microarch_isa.py``: one
  ``@df.region()`` taking no arguments, 8 ``@df.kernel``\ s all nested inside
  it as closures, 0 defined at module level. 16 stream declarations, **29
  capture edges**, and every stream has exactly one producer kernel and one
  consumer kernel. The closure surface is *exclusively* streams: 0 shared
  arrays, 0 captured Python values, 0 captured region parameters. The blocker
  for lifting a kernel is 2 captured streams at best (``dma_st``) and 5 at
  worst (``sequencer``, ``pe``). Nothing else stands in the way.
* That design has since been decomposed as far as the current syntax allows
  (:doc:`/designs/tinytpu_library`): all 8 units are module-level functions
  with no closure, and the parameters and ISA constants are passed rather than
  captured. It took **composing the region's source** to do it, because a
  kernel is reached only as a nested ``ast.FunctionDef``. The 29 capture edges
  did not go away -- they became 29 *declarations*, checked against each
  body's free names, and they are listed per unit at
  :ref:`tinytpu-library-capture-census`. That table is the exact scope of what
  this extension would replace, and the three things still impossible without
  it: two architectures must agree on a channel's *name*, a unit cannot be
  instantiated twice against different channels, and a unit cannot be built or
  tested outside a composed region.
* Upstream's own suite, independently: ``tests/dataflow/test_1D_systolic.py``,
  ``test_systolic.py``, ``test_tiled_systolic.py``, ``test_packed_systolic.py``
  and ``test_weight_stationary_gemm.py`` are five separate files, each one
  region with **one** kernel, whose topology lives in subscript arithmetic on a
  stream array inside the replicated body. 1-D and 2-D share no unit.
* **The IR is already right.** ``df.customize`` on a three-kernel region emits
  exactly the composable form -- each unit a standalone ``func.func`` with
  explicit ``!allo.stream<i32, 4>`` arguments and an ``stypes`` attribute
  giving each port's direction (``"_o"``, ``"io"``, ``"_i"``), and the region a
  ``func.func`` that constructs the streams and wires them by ``call``. Allo's
  IR is a netlist over ported units. Only the frontend cannot write one.
* **Parametrized units already exist too.** ``tests/dataflow/test_hierachical.py``
  defines ``def inner[P0, P1](A, B, C)`` and instantiates it twice in one
  program as ``inner[2, 2](...)`` and ``inner[4, 4](...)``. Parameters work;
  they just cannot be wired by streams.

**The measured blocker.** Two hops, both narrow:

1. ``Stream`` has no ``__class_getitem__`` (``allo/ir/types.py:306``), so
   ``Stream[int32, 4]`` raises ``TypeError: type 'Stream' is not subscriptable``
   whenever it is *evaluated* -- which is what happens in a signature, and
   never happens for a body annotation. ``Stream`` is a syntax, not a type.
2. With that patched, a module-level kernel with stream ports defines and the
   next failure is one line, repeated six times -- once per stream method
   (``put``, ``get``, ``try_put``, ``try_get``, ``empty``, ``full``) -- at
   ``allo/ir/builder.py`` lines 2610, 2632, 2655, 2686, 2717 and 2745::

       stream = ctx.get_symbol(new_name).clone(ip=ctx.get_stream_construct_ip())

   A stream operation is keyed to a **construct site**, not to a value: it
   re-clones the ``allo.stream_construct`` op that the declaration created. A
   parameter has no construct site, so a port fails with ``AttributeError:
   'MockArg' object has no attribute 'clone'``. Meanwhile the *op* already
   takes a block argument -- the lowered dump above is
   ``allo.stream_put(%arg1, [], %0)`` where ``%arg1`` is a block argument. The
   fix is to resolve the stream as a value.

**Legality rules that must ship with it.** These are the point of the
extension: constraints that are whole-program analyses today become local
checks on a signature or on one wiring edge.

* *Direction is single-valued.* Within a unit, each stream port is used only
  for ``get``/``empty`` (in) or only for ``put``/``full`` (out). Mixed use is
  refused. The analysis exists -- ``allo/dataflow.py:127`` already raises
  ``Stream is not used correctly`` -- but it runs over a whole region; here it
  runs once per unit and becomes part of the unit's declared type.
* *Wiring type-checks.* A stream wired to a port must agree on element type,
  shape, and depth, and the arity must match. Pure construction-time checking.
* *Single producer, single consumer.* Each stream has exactly one writer
  instance and one reader instance -- a degree constraint on the wiring graph,
  checkable without opening any unit. There is already an implementation to
  reuse (``check_single_producer_single_consumer``,
  ``allo/autoscheduler/util.py:164``), and TinyTPU-isa's 16 streams already
  satisfy it, so the rule type-checks the shipped design on day one.
* *No dangling port and no unconnected stream.* Every port of every instance is
  wired; every declared stream has in-degree and out-degree one.
* *Cycles are legal; zero-capacity cycles are not.* Feedback is normal in
  dataflow, so acyclicity must **not** be required. The decidable necessary
  condition is that every cycle in the wiring graph has positive total stream
  depth. **Deadlock-freedom itself is not decidable from the netlist** -- it
  needs per-unit rates -- so that half is an obligation, declared as one, not a
  rule. State this limit in the primitive's docstring; an extension that
  overstates its own guarantee is the failure mode
  :ref:`s.dependence <extending-allo-dependence>` already paid for.

**Risk.** Moderate, and lower than it looks: the destination IR is already
emitted and already exercised by the whole ``tests/dataflow`` suite, so this is
a frontend path to an existing IR shape rather than a new IR. The two
blockers above are located to the line. The risk is concentrated in
``allo/ir/builder.py`` and ``allo/ir/infer.py``, which the golden tests cover
densely -- run them at every step.

2. ``Encoding`` and ``s.encodable_on``
--------------------------------------

Landed; see :ref:`extending-allo-encoding`. It ranks second on the same
criterion for a reason worth stating: gap 1 is the *unit* half of "a library of
units instantiated on a target", and this is the **target** half. A unit is
only reusable if the compiler can check it against the machine it is being
instantiated for, and before this there was nowhere to say what that machine
is. The evidence that this was being done outside Allo instead: a loop-nest
mapper written against TinyTPU-isa hand-codes its four architectural
constraints in Python over a hand-written assembler, none of it reachable from
an Allo schedule, and ``check_perfect_affine_kernel`` hard-codes the
autoscheduler's own constraints with no target parameter.

.. _extending-allo-peel:

3. ``s.peel(axis, count=1)`` -- index-set splitting
---------------------------------------------------

**The gap.** Allo's whole mapping vocabulary is ``split`` and ``reorder``.
There is no way to give one iteration of a loop a different body from the
rest -- no ``peel``, no index-set split, no ``tensorize``, nothing. So a
reduction whose first tile must behave differently from the rest can only be
expressed by writing two loop nests by hand, and once they are two nests no
primitive can tile or reorder them together.

**The evidence.** Measured on a two-level loop-nest mapspace for TinyTPU-isa
(branch ``act-investigation``,
``examples/accelerator/tinytpu_vitis/act_nest.py``; reproduced here):
**1,226 nests enumerated, 3 encodable, and 1,150 of them refused at one
instruction field, ``acc``.** The ``k = 0`` tile has to be a peelable prefix,
which pins the reduction innermost and unsplit and kills every permutation that
moves it. The next constraint down, the 3-term address generator, refuses 17.
The 4-deep loop stack refuses none.

**Two corrections to that evidence, both found after this page was first
written, and both worth reading before relying on the headline.** First, the
mechanism: ``acc`` is *not* a static field that cannot be driven from an
induction variable. It is ``mm``'s ``f2``, ``f2`` is a legal AGU target, and
driving it from the reduce loop assembles for exactly two k-tiles before
rejecting at Kt≥3 — the obstacle is **additive monotonicity in the address
term**. The need for ``s.peel`` is unaffected, since a peelable prefix is still
the only way to express the first tile differently; what changes is that the
hardware fix is cheaper than "make the field dynamic" implies. Second, the
count: a first-cause census hides overlap, and **930 of those 1,150 nests also
violate the accumulator RAW-distance contract**, so the primitive does not
recover 94 % of the mapspace on its own. Sorted by the distinction this page
argues for, it is **1,166 express against 55 refuse**.

A third fact, which is the one that should temper any promise made for this
primitive: widening the address generator from 3 terms to 4 raises the
encodable count and **does not change the chosen nest**, so the emitted program
is identical, the cycles do not move, and the area rises. Enlarging a legal set
is not the same as improving a design, and this mapspace has already
demonstrated the difference.

**Legality rule.** ``s.peel(axis, count)`` is legal if and only if ``axis``
names a loop with constant bounds whose trip count exceeds ``count``. Peeling
is an index-set split, not a reordering, so it is unconditionally
semantics-preserving for a sequential ``affine.for`` -- which is what makes
this a rule and not an obligation. The real content of the rule is the handle
algebra: after peeling, the band has two loops where it had one, so any
``LoopWrapper`` previously captured for ``axis``, and any replayed
``primitive_sequences`` entry naming it, is invalid and must be refused rather
than silently resolved to one of the two.

**Construction-time checking** catches a non-constant bound, ``count`` at or
above the trip count, and a stale handle.

**Risk.** Higher than it sounds. Every structural primitive in Allo is a
dialect op consumed by the C++ ``loop-opt`` pass, so this needs ``allo.peel``
in ``AlloOps.td``, an implementation in ``LoopTransformations.cpp``, and a
rebuild of ``mlir/build``. It is the largest unlock on this list and the one
most likely to go wrong; do it after gap 1, not before, and never by pointing
one worktree's build at another worktree's tree.

.. _extending-allo-tensorize:

4. ``s.tensorize(band, intrinsic)`` -- NOT READY
------------------------------------------------

**The gap.** Nothing in Allo says "this innermost band *is* one instruction of
shape T x T". The mapper cited above hand-codes it: its ``split_body`` asserts
that the innermost level carries all three ranks, that two of them equal the
array dimension, and that the third is within ``MAXROWS``. Every accelerator
design repeats that by hand.

**Why it is not ready.** It cannot state its own legality rule. "This band
implements this instruction" requires a model of instruction semantics to check
against, and Allo has none -- there is no instruction op, no ISA type, nothing
in the dialect between ``affine.for`` and the HLS emitter. Without that, a
``tensorize`` primitive would be a promise with no discharger, i.e. exactly
:ref:`s.dependence <extending-allo-dependence>` again, on a much larger
surface: a wrong ``dependence`` claim costs a wrong pipeline, a wrong
``tensorize`` claim costs a wrong answer.

**What would make it ready.** A checkable instruction semantics. Note the
narrow path: a *structural* rule (the band's shape, rank set and access
pattern match the intrinsic's declaration) is decidable and would catch most
mistakes, while equivalence of the computation is not. If you take that path,
ship the structural rule as the rule, and declare the semantic half an
obligation -- do not let the structural check imply the semantic one.

5. Repair: give ``s.dependence`` its rule
-----------------------------------------

Not a new abstraction, which is why it is below the four, but it is the item
with the best ratio of risk to harm removed. See
:ref:`extending-allo-dependence`.


Before you claim an extension is legal
======================================

.. code-block:: text

   [ ] The legality sentence is written down, in the form
       "X is legal if and only if ...", in the docstring and on this page.
   [ ] What the rule cannot decide is named as an obligation, not implied to
       be checked.
   [ ] There is a refused example AND an accepted example, as tests.
   [ ] The default constrains nothing, and a test proves the no-op case.
   [ ] The mistake is refused at the call that made it, not at build time.
   [ ] The diagnostic names the site and the repair.
   [ ] No rule transforms anything.
   [ ] The golden tests pass, and you say that you ran them:
         python tests/dataflow/test_df_unit.py
         python tests/dataflow/test_region_stateful.py
   [ ] The fork-vs-upstream feature map is updated:
         https://github.com/sunwookim028/allo/issues/13

The last two are not formalities. Everything on this page was found by running
things, and every number on it was measured in this tree.
