..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

###########################################################################
U3 track E: composition design -- engines, instantiation, schedules, options
###########################################################################

.. note::

   **Design study with prototypes, 2026-10-04.** zhang-21, branch
   ``u3-compose`` from ``u1-pilot`` at ``dab9e7b4`` (which carries
   ``core-scoping``: C1/C2 fixed). The owner is away; every decision below
   is a **draft for the owner** under D-9, with its reason, and changes
   nothing under ``allo/`` or ``mlir/``. Prototypes live in
   ``examples/minitpu/template/`` and run on the Allo simulator and, where
   noted, the SystemC csim stand-in (Catapult 2024.2 libraries). MiniTPU at
   ``b3ba0a4d``, read only. Inputs: README (Design target, D-1, D-4,
   D-9..D-14, U3-U5), ``u3_plan_2026-10-04.rst`` (track E, Q1-Q5, H10/H11/H15,
   P-6/P-7), ``u1_alu_2026-10-02.rst`` (C1-C11), ``u3_phase0_2026-10-04.rst``,
   ``allo/compose.py``, ``tinytpu_library.rst``, ``ip_gaps.rst``,
   ``stream_ports.rst``, ``extending_allo.rst``, MiniTPU ``UNITS.md`` §2-§6
   and ``ARITHMETIC.md`` §7-§9.

   **[V]** verified by a run recorded here or by reading cited source;
   **[I]** inferred, to test.

Reproduce (``source examples/minitpu/harness/env-zhang21.sh`` first)::

   PYTHONPATH=$PWD $ALLO_PYTHON -m examples.minitpu.template.run_u3e          # 60 s: 27 OK, 1 finding
   PYTHONPATH=$PWD $ALLO_PYTHON u3_composition_design_2026-10-04/sc_probe.py <prj>   # SystemC csim
   ALLO_DUMP_COMPOSED=<dir> ...                                                # the composed regions

Outputs beside this file: ``gate.log``, the composed sources
``composed_pe_two.py`` / ``composed_mxu_tree_bf16_d4.py`` /
``composed_vpu_lane_sfu.py``, the H15 probes, the SystemC probes.

Summary
=======

The Design target asks the template to compose at three levels: parameters,
optional modules, swappable engines. ``allo/compose.py`` already composes a
region from units with declared interfaces and checks the netlist, but it
binds every unit from ONE flat namespace, names a kernel after its def name,
and has no notion of an engine, an option or a derived parameter. The five
questions below are what the template needs that it does not have; each gets
a draft D-entry and a prototype that works today, outside ``allo/``, with
the mechanism ``compose`` would adopt.

.. list-table::
   :header-rows: 1
   :widths: 4 22 30 20 24

   * - #
     - question
     - recommendation (draft)
     - prototype
     - gate (``run_u3e``)
   * - 1
     - engine interface
     - an ``Engine`` declares types, latencies, **accumulate order**, bodies,
       numpy references and directives; a matrix engine declares its order
       and a latency MODEL; a contract reference takes the order (D-15)
     - ``engines.py``, ``matrix_engine.py``
     - bf16 and int8 PE bit-exact; systolic and tree bit-exact vs the
       reference *with their order*, DIM 4 and 16; H11 74/8192 differ
   * - 2
     - parameters at instantiation
     - a unit is instantiated under a name with a binding of its free
       names; ``@df.unit`` gets ``unit[P0, P1]`` type parameters (D-16)
     - ``instantiate.py``: AST rename of a ``compose.Unit``
     - H10: two PEs, two engines, one region, 512/512 + 512/512 (Sim and SC)
   * - 3
     - schedules that travel
     - a directive belongs to the function or engine that needs it and is
       applied by every unit that binds it (D-17)
     - ``Engine.directives`` applied via ``Architecture.directives``
     - ``s.unroll("leading_zeros19:offset")`` found its band
   * - 4
     - optional modules
     - an ``Option`` is units + channels + **rebinds** + ISA slots; the
       netlist rules decide, never a conditional (D-18)
     - ``optional.py``, a VPU lane with/without SFU
     - both match; ``vgelu`` refused without the SFU; both conditional-form
       hazards refused by compose. H15: plain regions accept silently (bug)
   * - 5
     - derived-parameter legality
     - a geometry's derived numbers are properties, each relation a
       ``legality`` on the unit, timing derived from the engine's declared
       latencies (D-19)
     - ``legality.py``: ``MxuGeometry``, ``TreeGeometry``
     - 12 derived numbers == Phase 0; 5 wrong declarations refused

Tool findings made on the way (section 7): the SystemC emitter fails both
matrix rigs (an ``ac_int`` ``set_slc`` out of bounds; a one-line identity
function emitted with the wrong arity), the simulator fails to lower the int8
engine at DIM=4 only, a same-width signedness conversion lowers to
``trunci i32 -> i32``, and a plain ``@df.kernel`` region builds and runs a
Stream with one or no endpoint without a diagnostic.

1. The engine interface
=======================

The problem, with the evidence
------------------------------

* **The Design target names two swappable engines** (the MAC plug-in and
  the matrix engine) and ``compose.py`` has no word for either: a ``Unit``'s
  fields are ``body, instances, memories, reads, writes, parameters, isa,
  directives, legality`` (``compose.py:95-125``) [V]. TinyTPU's ``pe``
  computes ``int8 x int8`` in its body (``ip/units/pe.py``), so swapping the
  MAC means editing the PE.
* **Two matrix engines are two functions.** MiniTPU's array accumulates
  sequentially in acc24 in ascending contraction index and rounds once
  (``ARITHMETIC.md`` §8); a balanced tree rounds once per level. Measured
  here with U1's bit-exact references, DIM=16, 512 random bf16 rows:
  **74 of 8,192** results differ in the last bit between the two orders;
  on exact-sum data (small integers) **0** differ; at int8 (no rounding) 0
  differ [V]. So the order is observable at the engine's output and a swap
  that changes it has no single reference.
* **The timing constant changes with the engine.** Systolic push->valid is
  ``2 + DIM*(PE+1)`` (82 at DIM=16, Phase 0 [V]); a tree over the same
  contraction is ``O(log DIM)`` adder levels (the model says 15). The 85 the
  assembler consumes must therefore come from a manifest per engine (D-10),
  never from a constant.
* **U1 fixed the mechanism.** C1 (an engine passed as a value does not link)
  and C2 (identity is the def name) are fixed on ``core-scoping``: a
  module-level ``eng = twice`` called in a kernel builds and runs here
  (``u3e_sanity``, [V]). What remains is the *declaration*: what an engine
  must say for a unit to bind it safely.

Options
-------

A. **Engine as a bare function** (what C1's fix allows). Bind ``MAC_MUL`` as
   a parameter, nothing else. Cheap; but the unit must guess the types
   (``psum: ?``), the latency is nowhere, and the order is implicit, so a
   tree engine would pass every check and disagree in the last bits.
B. **Engine as a record** (``Engine``: types + widths, bodies, latencies,
   order, numpy references, directives). The unit binds the record's
   namespace as parameters (``MAC_IN``, ``MAC_ACC``, ``MAC_MUL`` ...) exactly
   as ``reduce_tree`` binds ``RED_IN``/``RED_ACC``. The order and types are
   checked at composition; the reference is evaluated with the engine's
   own arithmetic. **Chosen.**
C. **Engine as a unit with stream ports** (the ALU's ``netlist`` variant:
   ``adder(src=, dst=)``). Right for an engine with its own state machine
   (a pipelined adder as a unit, T2); wrong grain for a MAC inside a PE,
   where a stream per operand costs a FIFO per PE per operand
   (``u3_fifo_composed``: 3 cycles push->pop in Catapult).

Recommendation
--------------

B for the MAC plug-in; B's matrix-engine form (``MatrixEngine``: the unit,
its ``order``, a ``latency_model``) for the matrix engine; C where the engine
is itself a pipelined unit (T2's adders). A swap that keeps the order is
checked against the same reference; a swap that changes it is a different
function and is checked against the reference evaluated with that order
(P-7 confirmed). The template exposes MiniTPU's sequential order as the only
order *for the MiniTPU instance* and admits the tree as a declared different
function for DotTree and Jalapeño (answers O3 provisionally).

Draft README entry
------------------

.. code-block:: text

   **D-15 (2026-10-04, DRAFT for the owner). An engine is declared, and a
   swap that changes the accumulate order is a different function.**
   - A swappable engine is a record, not a function: operand, accumulator
     and output types with their widths; `mul`/`add`/`pack` bodies; declared
     `mul_latency`/`add_latency`; the accumulate `order` it is exact for;
     the same arithmetic in numpy; and the directives it needs of a
     schedule. A unit binds the record's names as parameters (`MAC_IN`,
     `MAC_ADD`, ...), never a bare function.
   - A matrix engine declares its `order` (`sequential` | `tree`) and a
     latency MODEL beside it. The model is what a composition books; what a
     backend built is `latency.json` (D-10). Neither is a constant in a body.
   - The contract reference of a composite takes the order as an argument.
     An engine swap that keeps the order is verified against the same
     reference; one that changes it is verified against the reference
     evaluated with the new order, and the change is recorded as a
     different function, as `ip_gaps.rst` does for a narrower node type.
   - MiniTPU's instance admits one order, `sequential` acc24 with one
     `pack_bf16` (P-5); DotTree and Jalapeño instances declare `tree`.
   - Evidence: `dev/records/minitpu/u3_composition_design_2026-10-04.rst` §1.

What would change in ``allo/``
------------------------------

* ``compose.py``: an ``Engine`` dataclass (as ``template/engines.py``) and
  an ``engines=`` field on ``Unit`` naming the engine slots the body binds,
  so ``Unit.check`` holds the body's engine names to the slot and
  ``Architecture._check`` holds the bound record's types to the channels
  that carry them (``Channel.lane_bits == engine.IN_BITS``).
* ``Architecture`` applies ``engine.directives`` for every unit binding the
  engine (section 3).
* ``allo/actions.py``: a compute port's arithmetic is the engine's
  ``ref_*``; today "arithmetic that no composition can carry"
  (``actions.rst``) becomes carried by the engine record.

Prototype and measurements [V]
------------------------------

``template/engines.py``: ``BF16_ACC24`` (U1's ``mul_acc24`` and
``acc24_add_pipe`` ``bits`` kernels lifted to functions,
``template/bf16_engine.py``) and ``INT8_INT32``. ``template/mac_pe.py``:
one ``mac_pe`` body, nothing in it names a type. ``template/matrix_engine.py``:
``systolic_engine`` and ``tree_engine`` share one declaration (reads
``me_lhs``, addresses ``W``, writes ``me_out``, binds the MAC names);
``matrix_rows(A, W, mac, order)`` is the contract reference with the order
as an argument (``order="sequential"`` at bf16 is ``ref_mxu.mxu_row``).

.. list-table::
   :header-rows: 1
   :widths: 40 14 14 32

   * - check
     - Sim
     - SC
     - note
   * - ``mac_pe`` at bf16->acc24, 4,096 random (specials included)
     - 4096/4096
     - 256/256
     - vs ``ref.mxu_bf16_mul_acc24`` + ``ref.mxu_acc24_add``
   * - ``mac_pe`` at int8->int32, 1,024
     - 1024/1024
     - 256/256
     - vs numpy
   * - systolic engine, bf16, DIM 4 / 16
     - 256/256, 512/512
     - **fails**
     - vs reference ``sequential``; SC: ``ac_int`` set_slc out of bounds (§7)
   * - tree engine, bf16, DIM 4 / 16
     - 256/256, 512/512
     - **fails**
     - vs reference ``tree``; same SC abort
   * - systolic / tree, int8, DIM 16
     - 256/256 each
     - **fails**
     - SC: ``pack_int32`` emitted with the wrong arity (§7)
   * - systolic, int8, DIM 4
     - **fails to lower**
     - --
     - ``trunci i32 -> i32`` after the simulator's passes (§7)
   * - tree engine at DIM=6
     - refused
     - --
     - the engine's own ``legality``: a balanced tree needs a power of two

Latency models (bookings, not measurements): systolic ``2 + DIM*(1 +
add_latency + 1)`` = 22 / 82 at DIM 4 / 16 (equal to Phase 0's measured
push->valid); tree ``1 + mul + add_latency*log2(DIM) + 1`` = 9 / 15 [I].

2. Unit parameters at instantiation
===================================

The problem, with the evidence
------------------------------

* **``@df.unit`` sizes freeze at decoration** (C9 [V]): ``UnitSpec`` snapshots
  ``get_global_vars(func)``; resizing a module global afterwards gives a
  type mismatch for an array port and a **deadlock** for a stream-only unit.
  ``stream_ports.rst``: "a unit cannot be parametrized at the instantiation
  site"; ``tests/dataflow/test_hierachical.py`` has the syntax one level
  down (``inner[P0, P1]``) but not for units. The ALU's four
  ``_unit_*(n)`` factories exist only for this (C11).
* **``compose`` has the same gap one level up** [V]: ``Architecture.region``
  binds every unit's free names from ``self.parameters``, one namespace, and
  ``Unit.name`` is ``body.__name__``, so one ``Unit`` composes at most once
  per region and every instance would see the same parameters
  (``ip_gaps.rst`` row 3: "the same Unit twice fails
  ``Architecture._check``"). The template's first need -- two PEs, two
  engines, one region (Q1) -- is exactly a second instance with a different
  binding.
* The Design target's "parameters: shapes, widths, depths" is *per
  instance* in every target named: Jalapeño's slices differ in memory, the
  LPU's units in M; MiniTPU's own tree is elaborated at N=64 for the XLU
  and N=16 for ``MINITPU_NUM_LANES=4`` builds (Phase 0).

Options
-------

A. **Factories** (C11, ``@df.unit`` inside ``def make(n)``): works today,
   distorts the source, one factory per (unit, engine), and the unit's
   identity is lost (two instances are two unrelated units to every check).
B. **A textual instance of a ``compose.Unit``**: re-emit the body under a
   new def name with its free names renamed (``MAC_MUL -> MAC_MUL__a``,
   ``lhs -> lhs_a``), bind the renamed names in the architecture namespace.
   Every ``compose`` check runs unchanged on the result. **Prototyped**; no
   change to ``allo/``; a stop-gap, because the rename is the binding the
   front end should do.
C. **Instantiation-site binding in ``compose``**: ``Architecture`` takes
   ``(unit, name, bind)`` triples; ``Unit`` stays one object. B's mechanism
   moved inside ``compose.py`` -- six lines of ``ast`` plus a check that a
   ``bind`` covers only free names. **Recommended now.**
D. **``@df.unit`` type parameters**: ``unit[P0, P1](...)`` as
   ``test_hierachical`` does for kernels, so a ``@df.unit`` is sized at the
   instantiation and the netlist rules type-check the wiring. The proper
   fix for C9; a front-end change (``allo/dataflow.py`` ``UnitSpec``,
   ``allo/ir/infer.py``), to be a D-n before code and to land with its
   legality rule (``wiring-type`` extended to parameters).

Recommendation
--------------

C now (the template's units are ``compose.unit``s, P-6), D as the front-end
work the ``stream_ports`` page already names as "the obvious next step".
Factories (A) only for ``@df.unit`` leaves until D lands, each recorded.

Draft README entry
------------------

.. code-block:: text

   **D-16 (2026-10-04, DRAFT for the owner). A unit is instantiated, and
   the instantiation binds its parameters, channels and engines.**
   - `compose.Architecture` instantiates a `Unit` under an instance name
     with a binding of the unit's free names: parameters (`DIM`, `N`),
     channels, engines (D-15). One `Unit` object composes any number of
     times in one region; two instances at two bindings are the reuse
     property the Design target's "parameters" level means.
   - A binding names only free names of the body, and the checks `compose`
     already makes (declaration == body, one owner per channel, `legality`
     on the bound parameter set) run per instance.
   - `@df.unit` gets the same at the front end: `unit[P0, P1](...)` type
     parameters as `test_hierachical`'s kernels have, resolved at the
     instantiation and type-checked by the `wiring-type` rule. Until then a
     `@df.unit` sized by a module global is frozen at decoration (C9), and
     a factory (C11) is the recorded workaround for leaves only.
   - A parameter that appears in a slice bound (`word[0:MAC_IN_BITS]`) is
     refused today (no inferable width); a bound unit converts by typed
     assignment instead until the front end folds constants into bounds.
   - Evidence: `u3_composition_design_2026-10-04.rst` §2; `u1_alu` C9/C11.

What would change in ``allo/``
------------------------------

``compose.py``: ``Architecture.units`` accepts ``Instance(unit, name,
bind)`` beside bare ``Unit``s; ``Unit.source()`` renames through ``bind``
(the ``_Rename`` transformer of ``template/instantiate.py``); ``_check``
verifies ``bind.keys() <= unit.free_names()``. ``allo/dataflow.py``:
``UnitSpec`` keeps the function and resolves its globals at the
instantiation, with ``unit[...]`` parameters merged in; ``allo/netlist.py``
``wiring-type`` compares the bound port types.

Prototype and measurements [V]
------------------------------

``template/instantiate.py``: ``instance(unit, name, bind, **overrides)``
returns an ``Instance(Unit)`` whose ``source()`` is the body re-parsed,
renamed and unparsed; ``bound(namespace, suffix)`` makes an engine's names
per instance. ``mac_pe.pe_rig_two``: ``pe_feed_a/mac_pe_a/pe_sink_a`` at
bf16 and ``..._b`` at int8 in one region, every channel and memory
suffixed (``composed_pe_two.py``). Simulator 512/512 + 512/512; SystemC
csim 256/256 + 256/256. Found on the way: the one place a type parameter
cannot go is a slice bound (``a_word[0:MAC_IN_BITS]`` -> "Cannot infer the
bitwidth of the slice" -> ``trunci i32 -> i32`` fails to build);
inside ``meta_for`` the same expression folds and works.

3. Schedules that travel
========================

The problem, with the evidence
------------------------------

* C10 [V]: Catapult needs ``leading_zeros17``'s loop unrolled for II=1, and
  the ALU's build had to know the adder's internals to say
  ``s.unroll("leading_zeros17:offset")``; ``"add_bits:offset"`` is refused.
  A reused *function* carries no schedule; a ``compose`` unit carries its
  own ``directives`` (``tinytpu_library.rst``); a ``@df.unit`` carries none.
* T2 will instantiate ``N - 1`` adder units; each needs the same unroll.
  Writing it ``N - 1`` times in a region's ``schedule()`` is the shape
  ``compose`` exists to remove.

Options
-------

A. **Directives on the unit only** (today): a unit that binds an engine
   repeats the engine's directives by hand -- the knowledge leaks up one
   level, and a second engine with different needs breaks every unit.
B. **Directives on the engine, applied by the binding unit**: the
   ``Engine.directives(s, ctx)`` is run by ``Architecture.directives`` for
   every unit instance that binds the engine. Needs the band to be nameable
   from outside the unit; measured: it is (``leading_zeros19:offset`` is a
   ``func.func`` of its own, so one unroll covers every caller). **Chosen.**
C. **A schedule attribute on the function** (``@allo.schedule(unroll=...)``
   on ``leading_zeros19`` itself): the cleanest, a front-end change; the
   directive would be applied when the function is built, before any
   region. To be a D-n with the Catapult track's evidence of which
   directives SystemC can carry (today "schedule directives are dropped",
   README Status).

Recommendation
--------------

B now, through ``compose``; C as the front-end form, gated on the SystemC
emitter translating or refusing the directive rather than dropping it (D-1).

Draft README entry
------------------

.. code-block:: text

   **D-17 (2026-10-04, DRAFT for the owner). A schedule belongs to the
   function that needs it and travels with it.**
   - A function or engine that needs a directive to meet its declared
     latency carries that directive (`Engine.directives`); every unit that
     binds it applies it, through `Architecture.directives`, without naming
     the function's internals. A region's `schedule()` names only what the
     region adds.
   - A function the front end builds as its own `func.func` has nameable
     bands (`leading_zeros19:offset`), so one directive covers every caller;
     an inlined function does not, and a directive on it is refused, not
     dropped (D-1).
   - A backend that cannot honour a carried directive refuses it naming the
     function; the SystemC emitter drops schedule directives today and must
     refuse or translate before this entry applies to Catapult.
   - Evidence: `u3_composition_design_2026-10-04.rst` §3; `u1_alu` C10.

What would change in ``allo/``
------------------------------

``compose.Architecture.directives``: after each unit's own directives, run
the directives of every engine the unit binds (one call per distinct
engine; the band is shared). Longer term, ``allo/customize.py``: a
``schedule`` attribute on a function applied at ``build_Call``.

Prototype and measurements [V]
------------------------------

``BF16_ACC24.directives = lambda s, ctx: s.unroll("leading_zeros19:offset")``;
``pe_rig`` passes it as the ``mac_pe`` instance's ``directives``.
``df.customize(arch.region())`` then ``arch.directives(s)``: applied; the
MLIR holds one ``loop_name = "offset"`` and the functions ``pe_feed_0,
mul_acc24_bits, leading_zeros19, acc24_add_bits, mac_pe_0, pe_sink_0``.
Not measured: the effect on Catapult (track C; needs the emitter to carry
the unroll).

4. Optional modules
===================

The problem, with the evidence
------------------------------

* The Design target lists four optional modules (accumulator file, SFU,
  transpose, scalar unit). Today's mechanism is a Python conditional over
  ``Architecture.units`` and ``channels`` (Q3). A conditional over the unit
  list alone leaves the module's channel declared with one endpoint; one
  over the channel list too leaves the neighbour reading a channel that no
  longer exists; and nothing stops the assembler from emitting the module's
  instruction.
* MiniTPU's own form: the SFU is a fixed stage in ``vpu.sv`` and ``vgelu``
  a fixed opcode; the ISA is one table for one machine. An instance without
  an SFU is not expressible there at all (``UNITS.md`` §2.1: geometry by
  recompiling the package).
* H15 asked whether the netlist refuses the dangling channel. Measured [V]:
  ``compose`` refuses both forms (``nothing writes 'sfu_out'``;
  ``channel 'alu_out' reads in both writeback and sfu``); the ``@df.unit``
  netlist refuses an unwired stream and a nested kernel writing an unread
  one (``NetlistError``); a **plain ``@df.kernel`` region builds and runs**
  a declared-untouched Stream and a written-never-read Stream with no
  diagnostic on the simulator, and the SystemC emitter emits the latter
  (``h15_frontend_probe.py``). The netlist rules run only when a region
  contains unit instances.

Options
-------

A. **Conditional composition** (today): both hazards above are the user's.
B. **An ``Option`` record**: the units and channels a module brings, the
   **rebinds** of its neighbours' ports when present (``writeback``'s input
   is ``sfu_out`` with the SFU, ``alu_out`` without), and the ISA slots
   that exist only with it. ``assemble(base, *options)`` composes the
   architecture and the instance's ISA; ``compose``'s netlist check decides
   legality. An assembler refuses a slot not in the instance's ISA, naming
   the module. **Chosen.**
C. **A bypass unit per optional module** (a pass-through ``sfu_bypass`` when
   absent): keeps the wiring fixed, costs a FIFO stage per absent module on
   every path -- a Catapult stream link is 3 cycles -- and the ISA still
   needs B's slot rule.

Recommendation
--------------

B. The rebind is the piece a conditional cannot express and the netlist
rules then do the rest. The front end should run ``unconnected-stream``
and ``single-producer-single-consumer`` on *every* region, nested kernels
counted as endpoints (the H15 bug), so that a composed region is held to
the same rules as a unit netlist by the front end and not only by
``compose``.

Draft README entry
------------------

.. code-block:: text

   **D-18 (2026-10-04, DRAFT for the owner). An optional module is a
   declared delta, and an instance's ISA is the slots its modules bring.**
   - An optional module is an `Option`: the units and channels it adds,
     the rebinding of its neighbours' ports when it is present, and the ISA
     slots that exist only with it. An architecture is a base plus options;
     no unit list is built by a Python conditional.
   - Composition is legal iff the netlist rules pass on the result: a module
     left out whose channel stays declared, or added without its rebind, is
     refused at composition, naming the channel.
   - The assembler of an instance refuses an instruction whose module is
     not composed in, naming the module; `gen_isa --check` holds the
     instance's spec to its options.
   - The front end runs the netlist rules on every region, nested kernels
     counted as endpoints. Today a plain `@df.kernel` region builds and
     runs a Stream with one or no endpoint silently (bug, filed).
   - Evidence: `u3_composition_design_2026-10-04.rst` §4 (H15).

What would change in ``allo/``
------------------------------

``compose.py``: ``Option`` and ``Architecture.with_options(base, *options)``
(``template/optional.py`` ``assemble``), the rebind implemented by D-16's
instantiation binding. ``allo/dataflow.py``/``allo/netlist.py``: run the
region rules whenever a region declares a Stream, not only when it has unit
instances; a nested ``@df.kernel`` is already an endpoint for the
unit-netlist path (rule (e) measured). ``examples/tinytpu/gen_isa.py``:
an instance's spec lists its options.

Prototype and measurements [V]
------------------------------

``template/optional.py``: a VPU lane ``issue -> alu -> writeback`` with
``SFU_OPTION`` (unit ``sfu``, channel ``sfu_out``, rebind
``writeback: alu_out -> sfu_out``, isa ``vgelu``). Simulator: 64/64 with
and without; ``["vadd", "vgelu"]`` refused without the SFU, assembled with
it; ``SFU_OPTION_NO_REBIND`` refused (two readers); the SFU removed with its
channel left declared refused (``nothing writes``). The ALU and SFU bodies
are stubs; the real ones are track A's (S1).

5. Derived parameters and legality
==================================

The problem, with the evidence
------------------------------

* MiniTPU declares ``N``, ``LEVELS``, the tap, the latencies and the span in
  separate places and holds them together with generate-scope ``$error``s,
  a ``$clog2`` in ``xlu.sv``, and one assertion in a testbench (``UNITS.md``
  §5, §8.3); the adder's 2-cycle latency "is assumed in two independent
  places and nothing checks it". Phase 0 measured every derived number
  (``u3_phase0``): push->valid 12/22/82, span 5/15/75, tree 13/9 and 9/5.
* ``ip_gaps.rst`` already states the pattern (``reduction_tree_legality``:
  "derived, never declared"), and ``compose.Unit.legality`` runs at
  composition. What is missing is the *timing* half: a derived latency that
  follows from an engine's declared latency, so that a 3-cycle adder built
  as 5 cycles moves ``push_to_valid`` through one relation instead of
  leaving a stale 85 in the assembler.

Options
-------

A. **Assertions in testbenches** (MiniTPU): found at simulation, never at
   build; the tap relation lived only there.
B. **Derived numbers as properties of a frozen geometry record, each
   relation a ``legality`` on the unit** (``ReduceParams``/``TpuParams``
   pattern), with the timing relations taking the ENGINE's declared
   latencies as inputs. **Chosen.**
C. **Infer the timing from the built RTL only** (D-10's manifest): necessary
   but not sufficient -- the manifest says what was built, the relation says
   what the contract requires; the harness compares the two.

Recommendation
--------------

B, and the comparison of C against B in the harness: a derived number that
disagrees with the manifest is a verdict, not a surprise.

Draft README entry
------------------

.. code-block:: text

   **D-19 (2026-10-04, DRAFT for the owner). A derived parameter is a
   property, and every relation it rests on is a legality condition.**
   - A geometry is a frozen record whose derived numbers (`LEVELS`, the
     tap level, the switch span, push->valid, a composite's latency) are
     properties computed from the declared ones and the bound engines'
     declared latencies; none is a field, so none can be typed in beside
     the number it must equal.
   - Each relation a unit's correctness rests on is a `legality` on that
     unit, run at composition, naming the parameter and the consequence;
     never an assertion in a testbench.
   - A derived latency is a booking (D-10): the harness compares it with
     the manifest's measured value per (unit, backend, clock) and a
     difference is a recorded verdict. The assembler reads the booking
     from the composed instance, never a constant.
   - Evidence: `u3_composition_design_2026-10-04.rst` §5; the 12 derived
     numbers equal Phase 0's measurements, and five wrong declarations are
     refused.

What would change in ``allo/``
------------------------------

Nothing in ``compose.py`` -- ``Unit.legality`` suffices -- but
``Architecture.parameters`` should accept a geometry record (its
``namespace()``) and the manifest comparison belongs in the harness
(``examples/minitpu/harness/latency.py``) reading ``latency.json`` against
``MxuGeometry.push_to_valid`` and ``TreeGeometry.latency``.

Prototype and measurements [V]
------------------------------

``template/legality.py``: ``MxuGeometry(DIM, NUM_SUBLANES, INPUT_DEPTH,
OUTPUT_ROWS, mac)`` with ``PE_LATENCY = 1 + mac.add_latency``,
``push_to_valid``, ``switch_span``, ``result_latency``;
``TreeGeometry(N, NUM_LANES, ADDER_LATENCY)`` with ``LEVELS``,
``LANE_LEVELS``, ``latency``, ``tap_latency``, ``max_path_delay``;
``mxu_legality`` and ``tree_legality``. ``check_against_phase0``: 12
relations equal (push->valid 12/22/82; span 5/15/75; PE 4; ``vmatpush``
85; tree 13/9 at N=64, 9/5 at N=16). ``refusals``: a stale
``PUSH_TO_VALID`` after an adder change, ``OUTPUT_ROWS`` not whole groups,
a tap one level off, ``LEVELS`` declared beside ``N``, a hand-written max
path delay for a changed adder -- all refused with the parameter named.

6. Hypotheses H10, H11, H15
===========================

Tracks A (``u3-xlu-sfu``, T2) and B (``u3-mxu``, P2) had not pushed when
this record was written (polled through 06:30); what each hypothesis needs
from them is marked **pending**.

.. list-table::
   :header-rows: 1
   :widths: 6 10 10 10 64

   * - #
     - Sim
     - SC
     - Cat
     - result
   * - H10
     - **Y**
     - **Y**
     - pending
     - One PE source, two MACs, one region: works through ``compose``
       (``instantiate.instance``), bit-exact on Sim (512/512 + 512/512) and
       SC csim (256/256 + 256/256). The ``@df.unit`` half -- "C9 forces a
       factory" -- stands from U1's ALU ``netlist`` variant (C11) and is not
       re-measured; the real PE (track B's P2) replacing ``mac_pe`` is
       pending.
   * - H11
     - **Y**
     - finding
     - --
     - Tree differs from systolic in the last bits on random data (bf16,
       DIM=16: 74/8192; DIM=4: 3/2048) and is equal on exact-sum data (0)
       -- in the reference, and each Allo engine is bit-exact against the
       reference with its own order on Sim. SC csim fails both matrix rigs
       (emitter, §7). The DotTree-shaped MXU out of track A's T2 adders is
       pending; here the tree is one unit over the MAC engine's ``add``.
   * - H15
     - **bug** / refused
     - **bug**
     - --
     - Removing the SFU and leaving its channel: refused by ``compose``
       and by the ``@df.unit`` netlist (``unconnected-stream``); a plain
       ``@df.kernel`` region -- the form ``compose`` emits -- builds and
       runs it silently on the simulator, and the SystemC emitter emits it.
       The real SFU (track A's S1) in place of the stub is pending; the
       mechanism does not depend on it.

7. Tool findings (for the U3 matrix and issues)
===============================================

E1. **Plain region, Stream with one or no endpoint: silently accepted** --
   **bug** (front end). ``h15_frontend_probe.py``: declared-untouched builds
   and runs; written-never-read builds (would block once full) and emits to
   SystemC. The ``@df.unit`` path refuses both (``NetlistError``). Repair:
   run the region rules on every region that declares a Stream.
E2. **SystemC emitter: ``ac_int`` ``set_slc`` out of bounds on the matrix
   rigs** -- **bug** (emitter). ``mxu_rig(SYSTOLIC|TREE, BF16_ACC24, 4,
   32)``: csim aborts ``Assert in ac_int.h:2912 Out of bounds set_slc``;
   the two-engine PE rig (no packed lane word) passes. Not yet isolated to a
   statement; the candidates are the lane set-slices into
   ``UInt(DIM * MAC_OUT_BITS)`` / ``UInt(DIM * MAC_IN_BITS)`` under
   ``meta_for`` (``sc_probe.py``; project ``u3e_sc/mxu_tree_d4``).
E3. **SystemC emitter: a one-statement identity function is declared with
   one parameter and called with two** -- **bug** (emitter).
   ``pack_int32(v: int32) -> int32: return v`` becomes
   ``void pack_int32(int32_t)`` at the definition and
   ``pack_int32(v, &out)`` at every call: g++ "too many arguments"
   (``sc_probe2.py``, ``systolic_int8_int32_d16``). Candidate: the callee's
   result buffer (``c8bc7bdd`` gave it the callee's sign) is not created
   when the body is a bare ``return <parameter>``.
E4. **Simulator: int8 matrix engine at DIM=4 fails to lower** --
   **bug**, not isolated. ``run_rig(SYSTOLIC, INT8_INT32, 4, 8)``:
   ``'arith.trunci' op operand type 'i32' and result type 'i32'`` after
   the simulator's pass pipeline (absent from ``df.customize``'s module).
   DIM=16 at int8 and DIM=4/16 at bf16 lower; eight isolated probes of the
   32-bit packed word (set/get 8-bit slices, streams, ``meta_for``, int32
   into 32-bit slices) all build (``u3e_t9``/``u3e_t11`` in the session
   scratch; to be reduced in the U3 matrix).
E5. **Same-width signedness conversion lowers to ``trunci i32 -> i32``** --
   **bug** (builder). ``lane: int32 = <UInt(32) slice>`` fails to build;
   keeping the word unsigned works (``me_sink``). Related to B1/B2.
E6. **A type parameter cannot appear in a slice bound** -- **missing
   abstraction** (known: ``tinytpu_library.rst`` "symbolic slice"), met
   again at ``a_word[0:MAC_IN_BITS]``; inside ``meta_for`` the bound folds
   and works. Workaround recorded in D-16's draft.

8. Provisional decisions taken here (D-9)
=========================================

* **E-P1.** The template's units are ``compose.unit``s bound to ``Engine``
  records (P-6 confirmed); engines are swapped by binding, never by def
  name (C11 retired for compose units).
* **E-P2.** The matrix engine's order is part of its declaration and the
  contract reference takes it (P-7 confirmed, O3 answered provisionally:
  MiniTPU's instance is ``sequential`` only; DotTree declares ``tree``).
* **E-P3.** Per-instance binding is done textually in
  ``examples/minitpu/template/instantiate.py`` until D-16 lands in
  ``compose``; the composed source is dumped beside each record.
* **E-P4.** Wide packed words are kept on the Allo side (``UInt(DIM*16)``)
  and unpacked at the region boundary (P-8), which is what exposed E2/E4.

Open for the owner: O3 (above), and whether D-16's front-end half
(``unit[P0, P1]``) is U3 work or waits for the TinyTPU-instance track.
