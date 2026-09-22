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

###############################################
The Unit Library: Composing a Region from Units
###############################################

TinyTPU-isa is not written as a design any more. It is written as eight units
in ``examples/accelerator/tinytpu_vitis/ip/units/``, one per module, and one
*architecture* -- ``ip/tinytpu.py`` -- that names the channels wiring them, the
parameters sizing them and the order they are declared in.
``microarch_isa.py`` is what is left of the design file: the parameter set read
from the environment, and the module-level names the harness imports.

This page is about the decomposition. :doc:`tinytpu_isa` remains the page about
the architecture and the ISA, and :doc:`tinytpu_history` about how the numbers
were reached.

Why the region is composed rather than written
==============================================

Allo's dataflow front end does not let a unit be written outside the region
that contains it. Two mechanisms, both in ``allo/ir/infer.py`` and
``allo/ir/builder.py``:

* a ``@df.kernel`` is recognised only as a **nested** ``ast.FunctionDef``
  inside the ``@df.region()``'s own source (``TypeInferer.visit_FunctionDef``
  descends into the region's body and looks for the decorator);
* the names in a kernel body -- including every stream it touches -- are
  resolved against the **enclosing region's scope** (``ctx.scopes`` is shared
  with the parent context), so a stream is reached by lexical name and never
  as a parameter.

A ``@df.kernel`` therefore has no interface. Its ports are whatever the region
it was pasted into happens to have declared, under whatever names. This is the
mechanism behind "the design is a monolith": it is not that the units were
tangled, it is that a unit had nothing to declare.

``ip/compose.py`` works with that rather than against it: it composes the
region's **source**. A unit is an ordinary module-level function; an
``Architecture`` emits a ``@df.region()`` that nests all of them, declares the
channels above them, and binds every free name in their bodies from its own
parameter namespace. The text is compiled with ``exec`` and registered in
``linecache`` under ``<composed tinytpu_isa>``, because Allo reads a region
back with ``inspect.getsourcelines``. ``ALLO_DUMP_COMPOSED=<dir>`` writes the
composed text out to read or diff.

What this buys, and what it does not
------------------------------------

It buys everything that does not need the front end to change:

* a unit is a module of its own, importable and readable without the region;
* a unit closes over nothing -- ``T``, ``VW``, ``SPAD_ROWS``, the opcodes are
  supplied by the architecture, so the same unit text composes at ``T=4`` and
  at ``T=8`` in the same process;
* a unit's Vitis directives travel with it, so the pragma that holds a unit at
  II=1 is next to the loop it applies to rather than in a ``schedule()`` that
  has to know every unit's internals;
* the interface is *declared and checked*, below.

It does not buy positional binding. A unit still names its channels
(``c_spm``, ``wcol``) rather than receiving them, so three things remain
impossible:

* two architectures must agree on a channel's *name*, not just its type and
  direction;
* one unit cannot be instantiated twice in one region against different
  channels (a second ``dma_ld`` on a second pair of operand ports, say);
* a unit cannot be built on its own. ``df.build`` takes a region, and a unit
  only exists once it has been composed into one, so a unit test is a test of
  a small architecture rather than of a unit.

.. _tinytpu-library-capture-census:

The census of what is still bound by name
-----------------------------------------

Every one of the eight units lifted to module level: **no closure remains**.
What was measured as 29 closure capture edges over 16 stream declarations is
now 29 *declarations*, each checked against the body it describes. The
capture surface was, and is, exclusively streams -- no shared array, no
captured Python value, no captured region parameter -- so this is the whole of
what the missing port syntax would replace:

.. list-table::
   :header-rows: 1
   :widths: 14 10 76

   * - Unit
     - Edges
     - Channels bound by name
   * - ``sequencer``
     - 5
     - out: ``c_dld``, ``c_spm``, ``c_vru``, ``c_acc``, ``c_dst``
   * - ``dma_ld``
     - 3
     - in: ``c_dld``; out: ``dma2sp``, ``dma2vr``
   * - ``spm``
     - 4
     - in: ``c_spm``, ``dma2sp``; out: ``sp2vr``, ``wcol``
   * - ``vru``
     - 4
     - in: ``c_vru``, ``sp2vr``, ``dma2vr``; out: ``acol``
   * - ``wld``
     - 3
     - chain: ``wcol``, ``wrow``; out: ``wq``
   * - ``pe``
     - 5
     - in: ``wq``; chain: ``acol``, ``a_fwd``, ``p_fwd``, ``cw``
   * - ``accu``
     - 3
     - in: ``c_acc``, ``cw``; out: ``ac2sp``
   * - ``dma_st``
     - 2
     - in: ``c_dst``, ``ac2sp``

``dma_st`` is the cheapest unit to give ports to (2) and ``sequencer`` and
``pe`` the dearest (5). Nothing else stands in the way: the parameters and the
ISA constants are already passed rather than captured, and the memories are
already ``args=[...]``.

That is the front-end gap, it is `fork issue #13
<https://github.com/sunwookim028/allo/issues/13>`_ territory, and it is being
worked on separately: ``Stream`` has no ``__class_getitem__``, so
``Stream[int32, 4]`` in a signature raises ``TypeError``, and behind that
``ctx.get_symbol(new_name).clone(...)`` at six sites in ``allo/ir/builder.py``
keys a stream op to a construct site rather than to a value. The composable
form already exists one level down: ``df.customize`` emits each unit as a
``func.func`` taking ``!allo.stream<...>`` arguments with an ``stypes``
attribute giving port direction, and the region as a ``func.func`` that
constructs the streams and wires them by ``call``. What is missing is the
source syntax to write that directly.

The interface is checked, not just written
==========================================

Each unit carries its interface in the ``@unit(...)`` decorator: the channels
it ``reads`` and ``writes``, the ``memories`` it addresses, the ``parameters``
it is sized by, the ``isa`` names it decodes. ``Unit.check`` recomputes the
body's free names from its AST and requires the two sets to be **equal**, so a
declaration cannot drift from the body in either direction. Three rules that
used to be prose are now enforced at composition time:

* **The array decodes nothing.** ``pe`` and ``wld`` declare ``isa=()``. That
  was a paragraph of the old module docstring; it is now a claim that fails
  loudly if anyone puts an opcode test in a PE.
* **One reader and one writer per stream.** ``Architecture._check`` holds every
  scalar channel to one reading unit and one writing unit, and requires each to
  have both. The chains (the stream *arrays*) are exempt: their owner is per
  element -- stage *i* reads ``ch[i]`` and writes ``ch[i + 1]`` -- and the index
  is a runtime value, so nothing static can check them.
* **One owner per off-chip memory.** ``imem`` is the sequencer's, ``A``/``B``
  are ``dma_ld``'s, ``C`` is ``dma_st``'s, and a second unit naming one is a
  composition error rather than a Vitis error later.

What the split made worse: the instruction layout
=================================================

One honest cost, and it is the only place where this decomposition is worse
than the monolith. An instruction is taken apart in three places -- the
encoder, the assembler's decoder and the sequencer's bit slices -- and in one
file those shifts sat within a screen of each other. In three files they can
drift, and **nothing here would catch it**: ``Unit.check`` answers *which
names a unit may decode*, not *what the bits mean*, and the byte-identical
MLIR check proves the region is unchanged, not that two hand-written decoders
agree. A shift off by one in the assembler and not in the sequencer passes
every gate on this page.

Two of the three sites now read one definition. ``ip/isa.py`` names the layout
-- ``OP_LO``/``OP_HI``, ``F0_LO``..``F3_HI``, ``NR_LO``/``NR_HI``,
``AGU_TERM_BITS``, ``AGU_TARGET_BITS``, ``AGU_LEVEL_BITS`` and the masks
derived from them -- ``enc``/``enc_agu`` build from it, and the assembler
decodes through ``opcode_of``, ``rows_of``, ``fields_of`` and ``agu_term``
rather than through ``& 0x3F``, ``>> 54`` and ``19 * term``. The re-derived
encoder was checked against the literal one over 6000 random instructions and
address words: zero differences.

.. _tinytpu-library-symbolic-slice:

The third site cannot, and that is a front-end limitation
---------------------------------------------------------

``sequencer`` still spells its slices ``control_word[0:6]``,
``control_word[54:62]``. Writing them as ``control_word[OP_LO:OP_HI]``
parses, builds and gives the right answers, but Allo cannot infer the width of
a slice whose bounds are names:

.. code-block:: text

   allo/ir/infer.py:582: UserWarning: Cannot infer the bitwidth of the slice,
   use UInt(32) as default

and the extract widens from ``i6`` to ``i32``. Measured 2026-09-22: the
normalized MLIR goes from identical to **796 diff lines** -- every field
extract in the sequencer becomes a 32-bit one, which is a different circuit,
not a different spelling. So the bit layout of an instruction can be named
once for software and must be written twice for hardware, until the front end
can fold a constant into a slice bound.

This is what the ``isa-spec`` track fixes properly: it generates the layout
from a single ``isa_spec.json`` and checks the encoder, ``expand``, the header
and the *emitted HLS slices* against it, which is the guarantee naming alone
cannot give -- one definition stops the sites drifting, it does not prove any
of them matches an ISA written down independently. Deliberately not
hand-rolled here. On the CHIA axis that generator belongs in
``FROZEN_DESIGN``: it judges a candidate rather than being one, and a
candidate editing ``ip/isa.py`` is then checked against the ISA, which the
loop has no way to do today.

Where the boundaries are, and why
=================================

The eight units the design already had, unchanged. They are the boundaries the
memories impose: ``spad`` lives in ``spm``, ``vr`` in ``vru``, ``ar`` in
``accu``, and Allo enforces one owner per memory, so those three units keep
every opcode arm they had -- a one-process-per-opcode split is not expressible
and was not attempted (:doc:`tinytpu_isa`, "One owner per memory"). The
library adds no hierarchy that a second architecture would not use.

The split across modules, though, is by *role* rather than by kernel, and that
is where the parametrization lands:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Module
     - What it owns
   * - ``ip/params.py``
     - ``TpuParams``: ``T``, ``MAXDIM``, ``SPAD_ROWS``, ``NVR``, ``NAR``,
       ``QD``, ``IMEM_SIZE``, and the derived ``VW``, ``AW``, ``WPR``. Frozen,
       validated in ``__post_init__``, and the only thing that decides a unit's
       size. Reads no environment variable.
   * - ``ip/isa.py``
     - The instruction encoding: opcodes, field layout, ``enc``, ``enc_agu``.
       ``ISA_NAMESPACE`` is what a unit's ``isa=`` declaration draws from.
   * - ``ip/compose.py``
     - ``Channel``, ``Memory``, ``Unit``, ``Architecture``: the composition
       itself, with no knowledge of TinyTPU.
   * - ``ip/units/*.py``
     - One unit each, closing over nothing.
   * - ``ip/assembler.py``
     - ``Assembler``, bound to a parameter set: ``expand``, ``trace``,
       ``check``, ``assemble``, and ``AR_RAW_DIST``.
   * - ``ip/programs.py``
     - ``MemoryMap`` and ``GemmPrograms``: the program ABI, which no unit
       knows.
   * - ``ip/tinytpu.py``
     - The TinyTPU architecture: channels, memories, unit order, and
       ``TinyTPU``, which ties a parameter set to its region, its directives,
       its assembler and its reference programs.
   * - ``ip/reduce.py``
     - The **second** architecture, ``DotTree``: an adder-tree reduction
       instead of the systolic chain, out of the same ``Unit``, ``Channel``,
       ``Memory`` and ``Architecture``. :ref:`ip-gaps-compose` is what it
       tests -- ``compose.py`` needed no change to admit a different reduction
       topology, because the topology is the pair (which units, which
       channels).
   * - ``ip/placeholders.py``
     - **Nothing here is implemented.** One declaration per unfilled row of
       :ref:`ip-gaps-table`, each raising ``NotBuilt`` with what would make it
       real.

Instantiating it
================

.. code-block:: python

   from examples.accelerator.tinytpu_vitis.ip import TinyTPU, TpuParams

   wide = TinyTPU(TpuParams(T=8, MAXDIM=32, IMEM_SIZE=56), name="tinytpu_t8")
   module = df.build(wide.region, target="simulator")
   words = wide.assembler.assemble(wide.programs.looped(16, 16, 16))

Two ``TinyTPU`` objects at different parameter sets coexist in one process,
holding the **same** ``Unit`` objects -- the reuse property, demonstrated
(these figures are from before the merge, when the default was MAXDIM=16):

.. code-block:: text

   tpu_t4: T=4 MAXDIM=16 VW=32 WPR=4  457 source lines, region=tpu_t4
      header of a 16x16x16 gemm: [13, 128, 144, 320, 16777232, 320, 64, 1048592]
   tpu_t8: T=8 MAXDIM=32 VW=64 WPR=4  457 source lines, region=tpu_t8
      header of a 16x16x16 gemm: [13, 64, 68, 96, 4194308, 96, 32, 1048592]
   same pe unit object in both: True

Nothing is read from a module global, so the two regions differ only in what
the architecture bound. ``microarch_isa.py`` is one such instantiation, with
its parameters from ``TPU_T`` / ``TPU_MAXDIM`` / ..., so the sweeps and the
CHIA parametricity gate keep working unchanged;
``chia_agent/param_check.py`` at ``T=8, MAXDIM=32`` reports
``PARAM OK: 408/408 runs exact``.

The parameter set is also where the coupling main found lives: the memory
sizes are derived from T and MAXDIM rather than typed in, because three
independent literals were big enough at MAXDIM=16 and silently were not above
it -- the GEMM's operand layout names ``MAXDIM*MAXDIM/T`` rows, which at
MAXDIM=64 is 1024 against a 256-entry vreg file. ``TpuParams`` takes ``None``
to mean "derive" and an explicit value as an override, and its
``__post_init__`` carries the two encoding ceilings.

The refactor moved no number, and that is checked
=================================================

The decomposition was done in two steps so that each could be verified
differently.

**Step 1, the move.** The composed region's AST was compared against the
region on the previous commit with ``ast.dump``: the only difference was the
eight kernel docstrings, which were removed. The MLIR ``df.customize`` emits
was then dumped before and after -- 4314 lines -- and is **byte-identical**,
both with the docstrings present and with them removed. A byte-identical
module means identical HLS C++, so no cycle count can have moved.

**Step 2, the naming.** Renaming locals does change the MLIR: Allo attaches
the source name to loads, stores, loops and buffers. Normalizing the
``name`` / ``from`` / ``to`` / ``loop_name`` / ``variable`` attributes and the
SSA value names makes the two dumps **identical again**, so the emitted C++
differs only in identifier spelling.

**Step 3, the merge.** ``main`` changed the design while the branch was held
-- MAXDIM to 64, the memory sizes derived rather than typed in, two
encoding-ceiling assertions, a parametric operand-burst width, a ``because=``
obligation on the dependence claim -- and the published row moved with it, to
171 / 261 / 417 / 483 / 685 (it has moved once more since, to
175 / 265 / 421 / 482 / 674, when ``QD=16`` became the default on
2026-09-24). The branch was merged and the design
re-decomposed rather than the old shape reapplied, and the comparison that
matters is now against ``main``: the MLIR the composed region emits is
**identical to the MLIR main's monolithic design emits**, 4325 lines, under
the same normalization. The obligation string is quoted verbatim from main so
that even the attribute text matches.

Where main's three changes landed is the test of whether the boundaries were
drawn in the right place: the derived sizing and the ceilings are the
parameter set (``ip/params.py``), the burst width is one unit's concern
(``ip/units/dma_load.py``, with ``DMA_WORDS`` a parameter that unit declares),
and the obligation is the accumulator's own claim
(``accumulator_directives``). Nothing reached ``compose.py``, ``tinytpu.py``,
the other seven units, the assembler or the ISA. A change to how the operand
burst works reached exactly the unit that bursts.

The measured gates, on this branch, after all three steps:

.. code-block:: text

   bench_isa      generated == hand-written word-for-word at all 5 shapes
                  ALL EXACT
   stress_isa     STRESS OK: 640/640 runs exact (96 shapes)
   act_compile    ACT GATE OK: 12/12 problems, every encodable mapping verified
   mutate.py      MUTATE OK: all 33 mutants run were caught (--no-rtl),
                  and ar_claim_false caught by cosim
   param_check    PARAM OK: 408/408 runs exact at T=8 MAXDIM=32

``mutate.py`` shadows the design tree
-------------------------------------

The mutation harness used to copy one file. It now writes its copy of every
design file (``microarch_isa.py`` plus the ``ip`` tree, ``DESIGN``) into
``.mutants/<name>/`` and puts that directory in front of the real one on the
package's ``__path__``, so every design import resolves to the mutated copy
while the harness scripts still come from the checkout. A mutant names no
file: its anchor must occur **exactly once across the whole design**, which is
a stronger uniqueness claim than before and is what locates the mutation.

What the prose deletion was
===========================

The design file is 1691 lines on ``main``, 904 of them comment or docstring
(53%), including a 264-line module docstring and a 30-to-60-line docstring on
each unit. The library is 1787 lines with 449 of prose (25%): **455 lines of
prose deleted**, and the hardware bodies themselves are the same size as
before.

Almost none of it was lost. The module docstring was a second copy of
:doc:`tinytpu_isa` and :doc:`tinytpu_history` -- chains rather than fan-out,
hazards from being in-order, one owner per memory, the data type, the
row-flattening story, the ``wrap_io`` measurement table, ``align_value``, the
write-behind rotation that was reverted -- and in one place it had gone stale,
which :doc:`tinytpu_isa` already flagged ("the module docstring still states
the requirement this way"). What replaced the rest is naming:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Unit
     - Was
     - Is
   * - ``sequencer``
     - ``ib``, ``lp_iv``, ``sp``, ``w0``/``w1``, ``tw``/``lw``/``sw``/``d``,
       ``rw``, ``ws``, ``wv``
     - ``program``, ``loop_iter``, ``loop_sp``, ``control_word``/``agu_word``,
       ``target``/``level``/``stride``/``offset``, ``resolved``,
       ``spm_copy``, ``accu_copy``
   * - ``dma_ld``
     - ``f0``, ``f1``, ``f2``, ``rbA``/``rbB``, ``pw``
     - ``route``, ``dram_row0``, ``col_block``, ``a_onchip``/``b_onchip``,
       ``packed``
   * - ``accu``
     - ``r``, ``rr``, ``ph``, ``ra``/``wa``, ``rv``/``z``, ``dw``, ``xr``,
       ``e``/``e2``/``e3``/``e4``
     - ``step``, ``row``, ``phase``, ``read_row``/``write_row``,
       ``read_word``/``write_word``, ``do_write``, ``vadd_first``,
       ``lane``/``add_lane``/``relu_lane``/``clip_lane``
   * - ``pe``
     - ``a``, ``aw``, ``p``, ``av``/``wv``, ``o``, ``cv``
     - ``activation``, ``activation_word``, ``psum_north``,
       ``activation16``/``weight16``, ``psum``, ``result_word``

``accu``'s pair is the clearest case: a paragraph explaining that ``vadd``
takes "two iterations per row, first source on the even one" is replaced by
calling the loop counter ``step`` and the derived index ``row``.

Two things were deliberately **not** renamed. The unit names
(``spm``, ``vru``, ``accu``, ``wld``) stay as they are, because three long
documentation pages and the Gemmini comparison refer to them; the spelling-out
happens in the module names (``scratchpad.py``, ``vector_regs.py``,
``accumulator.py``, ``weight_loader.py``). And the ISA field names ``f0``
through ``f3`` stay wherever a unit really is working on the raw encoding --
in ``sequencer``, which resolves fields without knowing the opcode, and in the
arms of ``spm``, ``vru`` and ``accu`` where one field means different things
to different opcodes. Where a field has one meaning in a unit, it is named
after that meaning, and the per-opcode meanings are one-line comments on the
declaration.

What this breaks: the CHIA harness
==================================

``chia_agent/`` is built on the premise that the design is two files: both
``allo_tool.EDITABLE`` and ``evaluate.EDITABLE`` are
``("microarch_isa.py", "isa_dsl.py")``, ``accept.py`` refuses a candidate diff
touching anything else, the spec directory is flat, and the agent's own
instructions in ``loop.py`` name those two files. After this refactor a CHIA
agent can still edit ``microarch_isa.py``, but the hardware is not in it, so it
cannot change the design at all.

This is **not repaired here**, deliberately: the fix is a change to the spec
interface of a harness that spends money, and it needs its own change with its
own review (see :doc:`/extensions/chia`). The adaptation is written --
``design-modular``'s commit ``4682046a`` adds ``chia_agent/design.py`` as the
one definition of ``EDITABLE`` and ``FROZEN_DESIGN`` -- and it was landed with
the design *without* its ``chia_agent/`` half, which is the harness owner's to
merge onto its own rewrite of ``accept.py``. It is gated rather than silent --
``chia_agent/test_harness.py`` is the documented $0 preflight for any paid run
and it fails, because all four of its mutant specs and its ``ANCHOR`` point at
text that has moved. What has to change:

1. ``EDITABLE`` in ``allo_tool.py`` and ``evaluate.py``, to the design tree
   rather than two names;
2. ``AlloSpecTool``'s flat ``self.sources`` mapping, ``evaluate.compose`` and
   ``loop.seed_spec``, to carry nested paths;
3. the allowed-file set in ``accept.py``, and its pinned control blobs;
4. ``test_harness.py``'s ``MUTANTS`` and ``ANCHOR``;
5. the file list in the agent prompt in ``loop.py``.

``chia_agent/param_check.py`` and ``gate_runner.py`` need no change: they work
through ``microarch_isa``'s module attributes, which are unchanged.

``impact/make_variants.py`` was textually coupled the same way -- it built the
historical ablation variants by patching ``microarch_isa.py`` -- and went stale
the same way, so it has been deleted. Its measurements are recorded on
:doc:`tinytpu_history`; regenerating them would want the variants expressed as
alternative *architectures* over the same units, which is the first thing the
library makes possible.
