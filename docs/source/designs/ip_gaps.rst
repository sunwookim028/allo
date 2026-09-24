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

##############################################
The IP Library's Gaps, and the Adder Tree
##############################################

What a real accelerator needs that ``examples/accelerator/tinytpu_vitis/ip/``
does not have. The reference is :doc:`minitpu`, another engineer's machine;
the destinations are Groq's **LPU** and OpenAI's **Jalapeño**
(`zartbot's analysis <https://zartbot.github.io/blog/arch/jalapeno/en.html>`_).

The library is in :doc:`tinytpu_library`; this page is the register of what it
cannot do, with the repro for each, and the one gap that has since been
filled.

.. contents::
   :local:
   :depth: 1

Where the two projects actually stand
=====================================

The first thing the comparison produced was not a gap. It was a correction to
the framing.

MiniTPU's units are **not composable in our sense, and they fail by
construction rather than by oversight**: geometry is ``localparam`` in a
package plus ``define`` macros, so you change the machine's shape by
recompiling the package; ``vpu.sv``'s entire control interface is one ~29-field
``vpu_ctrl_t`` struct, so the VPU and the sequencer are one unit wearing two
file names; six files in the whole tree have no package reference at all. The
one block with an interface we would recognise as declared is the MXU -- which
is also the only major block their released course lab does *not* ship, which
its author thinks is not a coincidence: *"the discipline came from having to
hand it to students."*

Their own summary, which is the honest shape of this page:

    *"Your library is ahead of our structure; what we have that you want is a
    longer list of units that survived contact with real workloads."*

So: **we are ahead on structure, they are ahead on list length and on contact
with real workloads.** Nothing below proposes restructuring anything of ours
toward their shape. On our three tests they pass *synthesisable* (on an FPGA,
with two storage units that know it), fail *composable* outright, and meet
*instruction-definable* only for the contract, not for the hardware.

Cannot express, cannot refuse
=============================

:ref:`extending-allo-gaps` makes this distinction for the Allo front end and
it is the same one here, because this project keeps finding the second
disguised as the first:

* **Cannot express** -- there is no way to write the thing down. The repair is
  a new unit, a new parameter or a new field.
* **Cannot refuse** -- the thing can be written down, and something wrong can
  be written down beside it and is accepted. The repair is a checker. This is
  the cheaper and the more dangerous kind, because the failure arrives as a
  wrong number rather than as an error.

Four of the eight rows below are *cannot refuse*, and one row (row 2) is a
false refusal hiding a real contract, which is the third and worst case.

.. _ip-gaps-table:

The gap table
=============

Every "evidence" cell names a test in ``tests/ip/test_ip_gaps.py`` or a place
in the source. Every unfilled row has a declaration in
``examples/accelerator/tinytpu_vitis/ip/placeholders.py`` carrying its
interface, what would make it real, and a ``NotBuilt`` exception that says so
if anything imports it expecting an implementation.

.. list-table::
   :header-rows: 1
   :widths: 4 20 14 20 28 14

   * - #
     - What is missing
     - Which target needs it
     - What would have to change
     - Evidence
     - Kind
   * - 1
     - **A reduction topology that can be chosen.** The PEs forward partial
       sums south through ``p_fwd``, so the reduction is ``T`` units deep and
       its shape *is* the unit graph.
     - Jalapeño: adder-tree reduction, output-stationary, 6-9 cycles, *not*
       a systolic chain
     - a unit. Not ``compose.py`` -- see `What compose.py did not need`_
     - ``ip/units/pe.py`` reduces by ``p_fwd[i-1, j].get()`` /
       ``p_fwd[i, j].put(psum)``; changing it means editing the PE body, and
       ``compose.py`` contains no word for topology
     - cannot express -- **FILLED**, `The adder tree`_
   * - 2
     - **M down to 1.** Two separate things wearing one costume.
     - Jalapeño: *"can go as fine as 1 in the row direction"*, against
       Nvidia's M<64 cliff
     - the accumulator's dependence contract, or an output-stationary unit
       that does not have one
     - ``GemmPrograms._tiles`` asserts ``M % T == 0`` with no message --
       **a false refusal**: bypass it and M=3 at T=4 is bit-exact. The real
       bound is ``AR_RAW_DIST``: at M=1, K=8, N=8 every dependent ``ar`` pair
       has to be padded apart and the program runs **22 dynamic instructions
       against 10 at M=4** -- 2.2x the instructions for a quarter of the work
     - both -- **FILLED**, see below
   * - 3
     - **One unit instantiated N times, at N parameter sets, against
       different channels** -- N slices with their own local memories
     - Jalapeño: per-slice local memory, no unified L2; LPU
     - ``compose.py`` emits ``@df.unit`` instances instead of nesting kernel
       source, **and** ``@df.unit`` accepts parameters at the instantiation
       site
     - ``test_second_instance_of_a_unit`` -- the same ``Unit`` twice fails
       ``Architecture._check`` with *"channel 'b' writes in both copy_unit and
       copy_unit"*. The front end no longer has this limit
       (:doc:`/developer/stream_ports`); this library has not adopted it, and
       instantiation-site parameters are named there as not attempted
     - cannot express
   * - 4
     - **Explicit compile-time placement.** Nothing says where an instance or
       the memory it addresses lives.
     - Jalapeño: ``TensorInfo`` carries logical layout *and* physical
       placement; the compiler decides which HBM slice every weight shard, KV
       and expert belongs to
     - a placement expression on ``Unit``, an owning slice on ``Memory``, and
       a reachability rule in ``Architecture._check``
     - ``test_no_placement_concept``: ``Unit``'s fields are ``body``,
       ``instances``, ``memories``, ``reads``, ``writes``, ``parameters``,
       ``isa``, ``directives``, ``legality``. ``compose.py`` contains none of
       the words place, slice, location, affinity, topology or bank
     - cannot express
   * - 5
     - **A channel whose topology is declared** -- broadcast, all-reduce,
       all-gather -- rather than hand-built as a chain
     - Jalapeño: an independent 8x8 two-stage collective network *beside* a
       general mesh; LPU
     - a kind on ``Channel``, and ``compose.py`` emitting the arrangement it
       names
     - ``test_no_collective_topology``; and ``ip/tinytpu.py``'s own docstring:
       *"every distribution is a CHAIN carrying packed words, never a T-way
       fan-out"* -- the topology is the sixteen ``Channel`` declarations
     - cannot express
   * - 6
     - **A unit cannot declare its latency, and nothing checks one from both
       ends.**
     - both targets, and MiniTPU's own strongest recommendation to us
     - a latency contract beside ``ip/isa.py``, generated-from and
       checked-against like the encoding is, plus a **two-sided** probe
     - ``test_unit_cannot_declare_latency``. ``reduce_tree``'s two outputs sit
       ``RED_DEPTH - RED_TAP_LEVEL`` adder levels apart and nothing says what
       that is in cycles
     - cannot refuse -- **flagged to ``unit-actions``, deliberately not built
       here**
   * - 7
     - **A unit does not declare its arithmetic.**
     - all three: MXFP4/FP8 on Jalapeño, BF16 on MiniTPU, int8 here
     - ``int8``/``int16``/``int32`` leave ``FRONTEND_NAMES`` and become
       declarable like any other parameter
     - ``test_units_do_not_declare_arithmetic``: ``pe`` computes an
       ``int8 x int8`` product into ``int32`` and declares
       ``parameters=("T", "VW", "AW")``. ``reduce_tree`` declares ``RED_IN``
       and ``RED_ACC`` and is the shape the other eight would take
     - cannot refuse
   * - 8
     - **A memory with two writers is not refused** -- the structure an FPGA
       gives away and standard cells reject
     - the ASIC flow itself, which is a standing constraint on every unit here
     - a write-site count in ``Unit.check``, plus the RTL audit
     - ``test_two_write_sites_not_refused``: a unit whose local buffer is
       written in two loops composes *and builds* with nothing remarked.
       :doc:`/designs/benchmarks` and ``asic_synthesis/README.md`` record DC
       refusing exactly that shape with ``ELAB-366`` after Vitis emitted a
       true dual-write-port RAM still named ``_1R1W``
     - cannot refuse -- **partly filled**: ``reduce_csynth.py --audit``
   * - 9
     - **A stream array is not held to one writer per element, nor to its
       declared shape**
     - none -- our own soundness, not a target's requirement
     - an abstract interpretation of the index expressions, or a declared
       per-element ownership rule to check bodies against
     - ``test_two_writers_on_a_chain_not_refused`` (the same pair on a
       *scalar* channel is refused) and
       ``test_chain_index_outside_shape_not_refused`` -- an index 1000 past
       the shape composes, and ``df.build`` then reports *"Variable
       narrow_1000 not defined in current scope"*
     - cannot refuse

Two front-end limits found on the way, which belong to
:doc:`/developer/limitations` rather than to this library:

* **A datatype-parametric unit costs every bit extract 32 bits.** A unit *can*
  take its element type from the architecture -- ``reduce_tree`` does, and an
  architecture binding ``float32`` builds and runs -- but a slice whose bounds
  are expressions over a *named* width (``packed[W*k : W*(k+1)]``) cannot be
  width-inferred: ``allo/ir/infer.py:583`` warns and widens the extract to
  ``UInt(32)``. This is the same limit as
  :ref:`tinytpu-library-symbolic-slice`, reached from the other side: it is
  why the eight shipped units spell their lane widths as literals, and it is
  the thing standing between this library and a genuinely datatype-parametric
  unit set. ``test_datatype_parametric_slices_widen_to_32_bits``.
* **A full-width bit slice does not build.** ``w[0:32]`` on a ``UInt(32)``
  lowers to an ``arith.trunci`` from ``i32`` to ``i32`` and the pass pipeline
  rejects it. ``test_full_width_bit_slice_does_not_build``; it is why
  ``reduce_tree`` refuses ``RED_GROUPS=1`` explicitly rather than emitting a
  tap that happens to be the root.

Row 2, which is the one worth reading twice
===========================================

"M must be a multiple of T" is asserted in ``GemmPrograms._tiles`` as a bare
``assert`` with no message, and it is **not true of the hardware**. Bypassing
it and running the shipped ``flat`` program shape at T=4, MAXDIM=16 on the
simulator:

.. code-block:: text

   M=4 K=4 N=4          4 dynamic instructions   wrong=0/16
   M=3 K=4 N=4          REFUSED by assemble(): mvout reads ar row 0 three
                        accu steps after writing it
   M=3 K=4 N=4 padded   5 dynamic               wrong=0/12
   M=1 K=4 N=4 padded   7 dynamic               wrong=0/4
   M=4 K=8 N=8         10 dynamic               wrong=0/32
   M=2 K=8 N=8 padded  14 dynamic               wrong=0/16
   M=1 K=8 N=8 padded  22 dynamic               wrong=0/8

So there are three distinct claims wearing one assertion:

#. **M need not be a multiple of T.** The array streams M wavefront rows; only
   K and N are tiled by the packed word. The generator's assertion refuses
   shapes the machine computes exactly. That is a *false refusal*.
#. **M < AR_RAW_DIST is a real contract violation**, and ``assemble()`` is
   right to reject it. ``accu``'s II=1 rests on
   ``#pragma HLS dependence ... inter false``, true only when an ``ar`` row is
   read at least ``AR_RAW_DIST=4`` accu steps after it is written, and at
   small M every dependent pair is closer than that.
#. **Padding makes it legal and costs almost everything.** 22 dynamic
   instructions for 8 output elements against 10 for 32: 8.8x the work per
   useful row.

Jalapeño's M-down-to-1 requirement is therefore not a tiling question at all
on this machine. It is a consequence of reducing through a *chain with an
architectural accumulator*, and the fix is the row above it: an
output-stationary tree finishes one output element in one pass, so M is a trip
count and M=1 costs 1/M of M's work. ``test_m_of_one_costs_one_row``.

.. _ip-gaps-tree:

The adder tree
==============

``ip/units/reduction_tree.py``, composed by ``ip/reduce.py`` into ``DotTree``:
a three-unit region computing an ``M x RED_LANES x N`` GEMM with the
contraction done by the tree. It is not bolted beside the systolic chain; it
is a different unit an architecture instantiates instead, and ``DotTree`` is
the second architecture this library composes.

What is declared
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Parameter
     - What it fixes
   * - ``RED_LANES``
     - the width. A power of two. ``RED_DEPTH = log2(RED_LANES)`` is
       **derived**, never declared, so the two cannot disagree
   * - ``RED_IN`` / ``RED_IN_BITS``
     - the lane type and its width. The body needs both -- a type to annotate
       with, an integer to slice with -- so ``legality`` holds the pair to
       each other
   * - ``RED_ACC`` / ``RED_ACC_BITS``
     - the type **every node** of the tree carries. The same width at every
       level: there is no widening down the tree
   * - ``RED_GROUPS``
     - the **leaf mapping**. Input lane ``g + s*RED_GROUPS`` lands at leaf
       ``g*RED_GROUP_SIZE + s``, so each group owns one contiguous subtree
   * - ``RED_TAP_LEVEL`` / ``RED_TAP_BASE``
     - where those subtrees' roots sit. Derived from the two above

``reduction_tree_legality`` refuses a parameter set at **composition** time --
not at build, not in cosim, not in a testbench. The condition that matters is

.. code-block:: text

   RED_ACC_BITS >= RED_IN_BITS + RED_DEPTH

with the message *"Narrower is not a smaller tree, it is a DIFFERENT function:
every level rounds, so the leaf mapping becomes observable and two correct
trees disagree."* At ``RED_LANES=8, RED_IN_BITS=16``, an ``Int(19)``
accumulator is accepted and an ``Int(18)`` is refused, and the test pins that
boundary.

Three things MiniTPU's owner told us to declare, and why
--------------------------------------------------------

**Leaf mapping is interface, not implementation.** Their leaves are
sublane-major, ``sublane*NUM_LANES + lane``, which reassociates ``vredsum``
relative to the lane order a reader assumes:

    *"a tree's topology is not just width and depth, it is which operand lands
    at which leaf -- and if your unit interface cannot say that, two correct
    trees will disagree in the last bits and nobody will know why."*

They **priced** the order rather than assuming it: a separate 4x16 lane-reduce
network synthesised at 19,198 LUT against the tree's own 20,216, so the
sublane-major order plus a four-wire tap beat a second network on cost. That
is why the tap exists here too.

**The relation between the tap and the tree's depth lives with the
parameters.** On MiniTPU exactly one assertion ties the tap to the booked
writeback and it is in a testbench; change the lane count, the sublane count
or the adder's stage count and the tap silently moves, and a sequencer
broadcasts a stale mid-tree value with nothing faulting. In a *parametrized*
unit that is the single most likely way to ship a wrong design, so
``RED_TAP_LEVEL == log2(RED_GROUP_SIZE)`` and
``RED_TAP_BASE == 2*RED_LANES - 2*RED_GROUPS`` are legality conditions on the
unit, not assertions in a test.

**A tap needs something that uses it.** For a whole commit MiniTPU's measured
lane-tap latency was a number nothing used -- the probe was right about the
hardware, the scheduler booked every reduce at the root latency, and the two
halves had never been checked against each other. Wiring them together took a
span-64 softmax from 150 bundles to 142. Here ``red_group`` is written by the
tree, read by ``dot_sink``, landed in a region argument and compared against an
independently built numpy reference from the first commit, and
``test_the_tap_is_not_the_root`` fails a unit that emits the root twice.

.. _ip-gaps-cheap-tree:

The tree we did not build
-------------------------

MiniTPU's tree instantiates the same pipelined BF16 adder 63 times, so a
64-element sum **rounds six times, not once**. Their commit ``875f3bd`` costed
the alternative -- align once to a shared exponent, sum as integers, normalise
once -- at roughly **-6,400 LUT and L ~ 8 instead of 16**, and rejected it only
because it changes ``vredsum``'s numbers against host references they had
already matched. Their words: *"If your reference is not frozen yet, that is
probably the design to take. We are worse than you here by inheritance, not by
argument."*

Ours is not frozen, so that is the form ``reduce_tree`` takes, and the
architectural point is what it does to the unit's *shape*: **rounding stops
being a per-level property of the unit and becomes a property of its two
ends.** ``RED_ACC_BITS`` is then a single legality condition rather than a
per-level error model, and the leaf mapping stops being numerically
observable. The per-level-rounding tree is the variant to keep for comparison,
not the default.

Two habits of theirs this unit deliberately does not inherit: their datapath is
unreset (a 1024-sink reset net avoided, safe only because the valid and op
pipelines *are* reset -- an invariant written down nowhere), and it therefore
toggles every cycle regardless of ``valid``. On the ASIC flow that is power,
and the ASIC flow is our substrate.

Synthesis
---------

``python examples/accelerator/tinytpu_vitis/reduce_csynth.py 8:2 16:4 32:8
64:8`` -- one ``csynth_design`` each, ``xcu280``, 3.33 ns target, project
deleted as soon as it is parsed. The per-instance row for ``reduce_tree_0_U0``
is the unit; the rig's feeder, sink and Vitis's five ``m_axi`` shims are not.

.. list-table:: ``reduce_tree`` alone, post-csynth estimate
   :header-rows: 1
   :widths: 14 12 12 10 10 22 20

   * - lanes : groups
     - LUT
     - FF
     - DSP
     - BRAM
     - tree loop
     - est. clock
   * - 8 : 2 (depth 3)
     - 354
     - 148
     - 0
     - 0
     - 4 cycles, II=1
     - 2.431 ns
   * - 16 : 4 (depth 4)
     - 553
     - 272
     - 0
     - 0
     - 4 cycles, II=1
     - 2.431 ns
   * - 32 : 8 (depth 5)
     - 940
     - 500
     - 0
     - 0
     - 4 cycles, II=1
     - 2.431 ns
   * - 64 : 8 (depth 6)
     - **1,643**
     - 1,036
     - 0
     - 0
     - **5 cycles, II=1**
     - **2.431 ns**

Read as: about 26 LUT per lane, linear in the width; **zero DSP and zero
BRAM** at every width; one reduction accepted per cycle at every width; and
the estimated clock unmoved at 2.431 ns against a 3.33 ns target, so the tree
is not the critical path of the rig it is in. ``Memory: N/A`` in every report
-- ``node`` is wires, which is what the ``Partition.Complete`` directive on it
is for.

**Jalapeño asks for 6-9 cycles of reduction depth.** A 64-lane tree here is 5
cycles at II=1, inside that band, so this is the first unit in the library that
can express the requirement rather than approximate it.

**ASIC legality is checked, not assumed.** ``reduce_csynth.py`` scans every
generated Verilog module for a declared memory written from more than one
clocked ``always`` block -- the ``ELAB-366`` shape -- and reports it as a
failure. Across all four widths: *no memory written from more than one clocked
block*, with the five findings in Vitis's own ``m_axi``/``s_axi`` interface
shims listed separately and not scored, because those modules are emitted
identically into every design this flow generates, including the shipped
TinyTPU that Design Compiler has already accepted.

.. warning::

   **Do not compare 1,643 LUT with MiniTPU's 16,259.** Different datatype
   (int32 nodes against BF16), different tool, different device, and theirs is
   a hierarchical post-synthesis figure from a commit that is not an ancestor
   of their master, while ours is a Vitis HLS *estimate* before any Vivado
   run. Their three figures -- 16,259 LUT / 8.2 % in-design, 20,216 LUT out of
   context, -1,062 LUT for +126 DSP at -1.08 ns -- are not interchangeable
   either. What *does* transfer is the shape: the reduction is adder-shaped,
   not multiplier-shaped, and wants fabric. Both designs land at **zero DSP**,
   independently.

The tree as Actions, and the two things the Action layer could not say
----------------------------------------------------------------------

``tests/ip/test_reduce_actions.py`` describes this unit as a ``Machine`` in
the Action layer (``allo/actions.py``, branch ``unit-actions``), built from
``ReduceParams`` rather than from an example's invented widths, with the two
output depths set to what ``reduce_latency_probe.py`` measured. It is the
second machine that layer asked for, and it was asked for precisely because
only one -- TinyTPU -- had ever exercised it.

The answer is a **qualified yes**. Everything structural survives the trip:
which units the opcode reaches, the leaf order and the refusal of a mapping
that is not a permutation, both output depths as measured, and the
reassociation obligation together with its dependence on the arithmetic
(discharged under exact integer addition, left open under rounding). Three
things do not survive, and none of them is a property of this unit:

#. **Latency is charged as occupancy.** ``Machine.work`` is documented as the
   per-unit work count an instruction-memory header carries. For
   ``reduce_tree`` it returns 80 where the hardware does 16: the model places
   each Action at ``max(ready of its args) + at`` and takes the unit's per-row
   cost to be the *span* of that placement, so the measured ``at=2`` is paid
   once per row instead of once per stream. There is no way to state what the
   probe measured -- latency 2, one word retired per cycle -- because the
   model draws no distinction between a **latency** and an **initiation
   interval** on an Action. ``Unit.ii`` cannot stand in for it: ``ii`` is the
   interval between the unit's own steps, so buying the right work count with
   ``ii=5`` would mean declaring something false. ``dot_feed`` and
   ``dot_sink``, whose Actions carry no ``at=``, model correctly -- which is
   why TinyTPU, described throughout without ``at=``, never surfaced this.

#. **One fold read at two taps cannot be said.** The root and the group tap
   are one fold with two taps off it; the model has only ``compute`` Actions,
   so the pair is written twice and the single reassociation is reported as
   two obligations. The direction is right and the multiplicity is the
   model's.

#. **A lane width has to hang on addressed state.** A lane map is checked
   against a lane *count*, which the model reads off the action's ``state``.
   MiniTPU's fold reads a vector register file, so its width is a property of
   addressed state; this operand arrives on a **channel**, and a channel is
   not a state. The composition is refused -- *"lane map without a width ...
   None declares no lane count"* -- until the packed word is declared as a
   one-row, ``RED_LANES``-wide state. That is a fair reading of a FIFO word
   and it is still the model meeting a dataflow unit half way; the repair the
   error message suggests names a thing this unit does not have.

The first of the three is the consequential one, and it is the reason the
Action hypothesis should not yet be promoted to the default: a work count
that is wrong by 5x on the first unit whose latencies were actually measured
is not a number a header can carry.

.. _ip-gaps-compose:

What ``compose.py`` did not need
================================

The brief's sharpest question was whether admitting a choice of reduction
topology would force a change to ``compose.py`` -- the test of whether it is
architecture-independent or only claims to be.

**It did not, and that is the finding.** Topology is not a property
``compose.py`` should hold; it is the pair (which units, which channels), which
is exactly what an ``Architecture`` *is*. ``DotTree`` composes a tree reduction
out of the same ``Unit``, ``Channel``, ``Memory`` and ``Architecture`` classes
TinyTPU uses, with no new mechanism and no TinyTPU-shaped assumption anywhere
in the way. The claim in :doc:`tinytpu_library` that ``compose.py`` "knows
nothing about TinyTPU" now has a second architecture behind it instead of only
an assertion.

One thing it did need, and it is a *cannot refuse* repair rather than a new
capability: ``Unit`` gained a ``legality`` callable, run by
``Architecture._check`` against the parameter set. ``Unit.check`` answered
*which names a unit may use*; nothing answered *which values it works at*, and
a tree instantiated past what its accumulator is exact for composed, built and
returned plausible wrong numbers. Six lines, and it is where every derived
relation in the tree is pinned.

What is not filled, and why
===========================

.. list-table::
   :header-rows: 1
   :widths: 6 30 64

   * - #
     - Row
     - Why not
   * - 3, 4, 5
     - second instance, placement, collectives
     - All three need the same thing first: ``compose.py`` emitting
       ``@df.unit`` instances, and ``@df.unit`` taking parameters at the
       instantiation site. Building placement or a collective kind on top of
       source-level name binding would be building them twice. Row 3 is
       therefore the highest-value structural change available to this
       library, and it is a front-end-adjacent change, not an IP one
   * - 6
     - declared latency, checked from both ends
     - Belongs to ``unit-actions``. A latency is a property of an *action*,
       not of a unit -- MiniTPU's own ``vmatpush`` occupies its controller 5
       cycles and produces its first result at 82 -- and building it here
       would duplicate that branch. Flagged, with the two-sided probe
       described in ``placeholders.UNIT_LATENCY``
   * - 7
     - declared arithmetic
     - Retrofitting the eight shipped units changes every declaration in the
       library and buys no capability today, against a design whose measured
       behaviour must not move. ``reduce_tree`` demonstrates the shape; the
       retrofit waits for a second architecture that needs it
   * - 9
     - chain ownership
     - The exemption is *honest*: a chain's owner is per element and the
       element is a runtime index, so nothing static decides it today. A
       guess here would refuse the shipped design. It needs a declared chain
       rule to check against, which is a design question and not a missing
       check
   * - 8
     - two-writer memories
     - Half filled. The audit exists and runs; what is missing is
       ``Unit.check`` counting a body's array write sites, which is the
       **cheapest unfilled row on this page** -- the AST is already parsed for
       ``free_names``

What running the gaps looks like
================================

.. code-block:: bash

   pytest tests/ip/                    # 23 pass, 1 xfail -- the xfail IS row 3
   python examples/accelerator/tinytpu_vitis/reduce_csynth.py 64:8

``tests/ip/test_ip_gaps.py`` is one test per unfilled row. A row that gets
filled makes its test fail -- the second-instance test is
``xfail(strict=True)`` precisely so that closing the gap forces the row to be
closed too. ``tests/ip/test_reduction_tree.py`` builds and runs the tree at
five configurations and every leaf mapping, against full-range ``int8``
operands, comparing both outputs with numpy.

TinyTPU's own gates are unaffected and were re-run: ``bench_isa`` ``ALL
EXACT``, ``stress_isa`` ``STRESS OK: 640/640``. Nothing on this page
instantiates a unit into ``tinytpu_isa``.

Still open from their side
==========================

Carried because it is reduction-adjacent and was volunteered rather than
hidden: MiniTPU has an **unclosed defect** where a context past 64 tokens
fails *in simulation* with infinite relative error -- a row's exponentials
summing to zero -- while their board runs 448-key attention correctly. Whether
it is the tree, the softmax scaling or the schedule is unknown. Relevant to
anyone reusing that softmax structure.
