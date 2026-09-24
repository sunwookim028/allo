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

.. _actions:

################################################
Actions: an Instruction as Composed Unit Effects
################################################

``allo/actions.py``. An instruction is not a primitive: it is an ordered list
of effects, each one belonging to a named unit and spending a named port of
that unit. Everything a consumer usually restates -- which units an opcode
reaches, how many steps each spends on it, which rows of which memory it
reads before it writes, what value comes out -- is a query over that one
list. The model is machine-independent; :doc:`/designs/tinytpu_isa` builds
one out of ``isa_spec.json`` and ``examples/tinytpu/ip/reduce.py`` builds a
second out of the adder tree's parameters.

This page is about the question the two models raised the moment they
existed side by side: **``allo/compose.py`` and ``allo/actions.py`` were two
descriptions of one machine with no type in common, so the Action layer was a
second declaration of every unit** -- which is the failure class it exists to
remove. What follows is what was measured when they were joined, what joined
cleanly, and what did not.

.. contents::
   :local:
   :depth: 1

The two models, and the one question
====================================

:doc:`/designs/tinytpu_library` describes ``allo/compose.py``: a ``Unit``
declares the ``instances`` it is replicated into, the region arguments it
addresses, and every channel it reads or writes; an ``Architecture`` binds
them and emits the region's *source*, because Allo reaches a ``@df.kernel``
only as a nested ``ast.FunctionDef``. It is a **structural** model: what is
wired to what.

``allo/actions.py`` is a **behavioural** model: a ``Unit`` has ``ports`` and
an ``ii``, an ``Action`` spends a port over declared ``State``, and an
``Instruction`` is a composition of them. Its rule is one-sided in the shape
``allo.dependence`` uses: it refuses what it can disprove and confirms
nothing, and what an accepted composition still rests on comes back as an
``Obligation``.

The question, in one sentence: **can the behavioural model's ports be
derived from the structural model's channels and memories, so that nothing
is declared twice?**

What derives, and what does not
===============================

``allo.actions.structure(architecture)`` answers it by construction. It
builds the units, ports, states and channels a composed ``Architecture``
already implies: one port per channel endpoint the unit declares, and one
``<memory>.read`` / ``<memory>.write`` port per array its body addresses --
read off the same AST that ``Unit.check`` already parses to hold the
declaration to the body, so no new declaration is introduced to derive from.
``allo.actions.projection(architecture, machine)`` diffs that against a
hand-written machine, and ``gen_isa.py --check`` runs the diff as a gate.

On TinyTPU, whose 7 Action units declare 28 ports between them:

.. list-table::
   :header-rows: 1
   :widths: 10 46 44

   * - Ports
     - What they are
     - Does it derive?
   * - **20**
     - a channel the unit is declared to touch, or a memory its body
       addresses -- ``spad.read``, ``wcol``, ``ar.write``, ``ac2sp`` ...
     - **Yes.** ``spm`` and ``vru`` derive completely: every port they have
       is the composition's own, and nothing about them is stated twice
   * - **4**
     - an ALIAS: one declared port standing for composed ones --
       ``dram.read`` for ``A.read`` + ``B.read``, ``dispatch`` for the five
       control channels, ``fetch`` for ``imem.read``
     - **Partly.** What it stands for is checked; the aggregation itself is
       a capacity claim about the AXI path, which is a hardware fact
   * - **3**
     - ARITHMETIC: ``dma_ld``'s operand select, the array's MAC, the
       accumulator's ALU
     - **No, and never.** A composed region declares what a unit is wired
       to, never what it computes. This is the Action layer's reason to
       exist
   * - **1**
     - ``array.instructions``, a counting port that carries nothing: the
       header word ``mm_count`` is ``items(array, 'instructions')``
     - **No.** It is an artefact of counting through ports

Going the other way, the composition implies **15 ports the Action model
does not model**, each of which is now listed with its reason in
``gen_isa.UNMODELLED`` and checked in both directions, and **three on-chip
memories the model never had**: the sequencer's prefetch window ``program``
and the two burst landing buffers ``a_onchip``/``b_onchip``. The derivation
found them; reading the RTL had not.

Two of those reasons are results rather than bookkeeping.

**The control path cannot be written as an action.** Every data unit reads
its dispatch queue when its row counter runs out, *inside* the step it is
already spending. An action is per-row or per-instruction: a per-row receive
would say every row fetches, and a per-instruction one adds a head step to
the unit's work count -- which is the number the instruction-memory header
carries, so the sequencer would promise ``accu`` one iteration more than it
performs. ``test_the_control_path_cannot_be_modelled_as_an_action`` writes
the per-instruction version and measures the damage: 6 steps become 7.

**The two models disagree about what a unit is.** A compose unit is a
KERNEL, replicated by ``instances`` and wired to its copies by stream
arrays. An Action unit is a DISPATCH DOMAIN: one work counter, fed one row
count by the sequencer. They are the same object for every unit instantiated
once, and a different object for every unit that is not. TinyTPU has exactly
one such region -- ``wld`` and ``pe``, ``T x T`` of each, wired by five
chains -- and the ISA sees it as one ``array``. This is not a naming
difference and no convention closes it: ``items(array, 'mac')`` is the
header's ``mm_rows``, and at the composed grain the same quantity is a
per-instance count over ``T*T`` kernels, of which the Action model has no
word.

On ``DotTree``, the second architecture, every unit is instantiated once, so
the grain agrees and the whole structure derives. ``ip/reduce.py``'s
``machine()`` declares exactly two ports of its own -- ``mul`` and ``adder``
-- and takes its three units, their channel and memory ports, its six states
and the packed word's lane count from ``structure(architecture())``.

A latency is not an initiation interval
=======================================

The Action layer's headline defect, found by the second machine and not
findable by the first: ``Machine.work`` returned **80** steps for
``reduce_tree`` where the hardware does **16**. ``_book`` placed each action
at ``max(ready of its args) + at`` and took the per-row cost to be the SPAN
of that placement, so a measured ``at=2`` was charged as OCCUPANCY -- paid
once per row instead of once per stream. ``Unit.ii`` could not stand in for
it: ``ii`` is the interval between a unit's own steps, so ``ii=5`` would have
bought the right number by declaring something false. TinyTPU is described
throughout without ``at=``, which is why one machine could never have shown
it.

``_book`` now returns both numbers and they answer different questions:

* the **span** is the latency, the cycles from a row's first effect to its
  last, which a declared ``at=`` lengthens;
* the **initiation** is the rate, the cycles before the same actions can be
  placed again, which is set by the busiest resource they book -- read off
  the ports the unit already declares.

Nothing new is declared. The calendar is unchanged (the fold still lands two
cycles after the word arrives, both emits two after that), ``Unit.ii`` is
untouched, and ``reduce_tree`` is now 16 steps of work with a latency of 20
cycles for 16 rows. Every TinyTPU work count is unchanged, because a unit
with no declared latency has span == initiation, which is why ``gen_isa.py
--check`` is still ``ISA OK`` and the published cosim row is unmoved.

The distinction also repairs a refusal: ``Unit(elastic=False)`` used to be
refused when a row's *latency* exceeded its ``ii``, which is exactly what a
pipelined unit is. It is now refused when its *rate* does.

**And it cost a catch, which is the more interesting half.**
``mutate_actions.py`` declares ``accu``'s ALU one lane operation a step
instead of two -- the single hardware fact ``vaddrelu`` rests on -- and
``gen_isa.py --check`` used to catch it: a single-issue ALU pushed the
rectify into a third cycle and the work count went to 3 a row against the
sequencer's 2. That catch was an **artefact of the defect**. A pipelined unit
retires a row every ``max(resource)`` cycles, and ``accu`` reads ``ar`` twice
a row through one port, so the read port binds at 2 a row either way: the
rectify of row *r* and the add of row *r+1* take alternate cycles and fit a
single-issue ALU exactly. Correcting the cost model therefore **lost** a
static catch and gained nothing back, and the mutant table now says so in its
own row rather than quietly dropping to fourteen. A checker that is
conservative in the wrong place catches more than it is entitled to, and some
of what it catches it has no right to.

A lane count belongs to the channel
===================================

A reduction's leaf order is checkable only against a WIDTH, and the model
read that width off addressed state alone. MiniTPU's fold reads a vector
register file, so its width is a property of a memory; ours arrives on a
FIFO, and the packed word had to be declared a one-row ``State`` -- a memory
nothing addresses -- to get the composition accepted at all.

``compose.Channel`` now carries ``lanes`` and ``lane_bits``, and **derives**
its ``dtype`` from the pair, so the count is written once and the width
cannot disagree with it:

.. code-block:: python

   Channel("red_in", depth="QD", lanes="RED_LANES", lane_bits="RED_IN_BITS",
           carries="one packed word of RED_LANES lanes, per output")

``actions.Channel`` carries the same two fields, and an action that folds a
value takes its width from the channel the value arrived on -- found by
following the value flow the rule already checks, so no field is added to
``Action``. The shadow state is gone. The finding underneath is small and
exact: what the model was missing was **a lane count, not addressed state**.
A channel has no rows, no bank map and no collision rule, and a fold needs
none of them.

What is still true
==================

A consolidation that quietly drops a negative result is worse than no
consolidation. These survived intact.

**One fold read at two taps still cannot be said.** ``reduce_tree`` computes
one tree and reads it at the root and one level down; the model has no way
to say "one fold, two taps off it", so the pair is written as two ``compute``
actions and the single reassociation is reported as **two** obligations. The
direction is right -- discharged under ``EXACT``, open under ``ROUNDING`` --
and the multiplicity is the model's. The repair would be an ``Action`` with
more than one ``into``, each with its own ``at``; it has not been made.

**Six wrong declarations out of fifteen are invisible to every static
check.** ``mutate_actions.py`` breaks one action of one instruction in a
sandbox copy of the spec and asks which level notices: 5 are refused by the
rule when the machine is built, 3 move a work count and are caught by
``gen_isa.py --check``, **6 are caught only by running a program on the
hardware** -- a source declared as the other source, a subtract for an add, a
fused instruction without its rectify, an accumulate read that loses its
predicate, a load that ignores its column block, a store to the wrong DRAM
row -- and **one is caught by nothing static at all** (the ALU's width,
above). The model verifies composition, not arithmetic, and the port
projection added here does not change that split: none of the six touches a
port. The split was 5 / 4 / 6 before the cost model was corrected.

**A derived port is one read and one write port.** ``structure()`` gives
every array a single read and a single write port, and a fully partitioned
array -- ``reduce_tree``'s ``node``, which is wires -- is not that. The unit
says so in Vitis terms, through its ``directives``, and nothing structural
says it. An action over such an array would be costed wrongly, which is why
the tree's actions do not name it.

What one file would have to keep
================================

The end state this was aimed at is *a single file, architectures composable,
an ISA modelled as composed Actions*. The parts that did not reach it, with
the reason each:

* **The structural model may not import the behavioural one, or the
  reverse.** ``allo/actions.py`` imports nothing from Allo -- ``structure()``
  reads an architecture by duck typing -- and that independence is what lets
  the ISA spec's machine be built and checked without the front end. Merging
  the two files would make the Action model depend on ``allo/dataflow.py``
  and on MLIR bindings, for no capability.
* **The ISA spec's unit table may not simply become the derivation.** The
  discipline ``gen_isa.py`` enforces is that the spec is the source of truth
  and the design is held to it *independently*; a spec that reads its ports
  out of the design cannot check the design's ports. Turning the second
  declaration into a **checked projection** is what is available, and it is
  what landed: where the two models overlap they can no longer drift, and
  where they cannot overlap the reason is written down and checked in both
  directions.
* **The grain.** Until an Action unit can be a replicated kernel, or a
  compose unit can declare that it is one dispatch domain with its copies,
  one machine in eight will not correspond.

