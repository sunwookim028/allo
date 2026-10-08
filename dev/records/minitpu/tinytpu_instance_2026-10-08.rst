..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

###########################################################################
TinyTPU-isa as an instance of the template (README D-4, D-15, D-17, D-19,
D-20; the "TinyTPU as an instance" track)
###########################################################################

.. note::

   zhang-21, 2026-10-08, branch ``tinytpu-instance`` from ``origin/u1-pilot``
   at ``ebbb7b7a``, worktree ``scratch/wt-ttinst`` with ``wt-u1``'s bindings
   copied in (Python-only change). Env: ``conda activate allo``,
   ``OMP_NUM_THREADS=8``; Vitis HLS 2023.2. Scope, from the owner mid-track
   (README D-22 on ``u1-pilot``): this track is a PROBE of the template's
   mechanisms -- the int8 ``Engine``, the accumulator ``Option``,
   ``Instance`` binding, the geometry record -- not a requirement to
   reproduce TinyTPU-isa; one cosim at one shape, findings first. The
   frozen design ``examples/tinytpu/`` was not modified (D-4); it is
   imported.

Summary
=======

TinyTPU-isa composes from the template as a base plus one option, unit for
unit and channel for channel the frozen ``ip/tinytpu.py`` wiring, and passes
TinyTPU's functional gates unedited: ``bench_isa`` ALL EXACT, ``stress_isa``
492/492, ``gen_isa --conform`` ISA OK with a new ``isa_slots`` arm (11 of 11
opcodes, by module), on the simulator at ``TPU_MAXDIM=16``. The emitted
Vitis text of the FROZEN design is unchanged through every change here
(sha256 ``6bc774bc...b2ef95``).

Three things the template could not express about TinyTPU, each now a
mechanism or a finding: a base that an option COMPLETES (the sequencer
dispatches to the accumulator's queues, so the base alone is not a legal
machine) -- ``Architecture(draft=True)``, F4; a parameter name in a slice
bound at a PE's pid (``activation_word[MAC_IN_BITS * i : ...]``) -- the
front end does not fold it, F1/F2, worked around by a constant shift; and
an engine body that returns its own argument (``pack_int32``) -- a core
pass crashes under ``s.partition``, F3. And the headline on expressiveness:
swapping the MAC engine in the PE is NOT swapping the machine's arithmetic.
Every reused unit is int8/int32 through literals (``wld``'s ``[0:8]``,
``accu``'s ``32 * lane``, the ``int8`` DRAM boundary), so the instance is
engine-generic in exactly one unit (F8). Cycles: 175 at 4x4x4 on the
instance, the frozen design and the published table alike (§5), once the
host's linker was worked around (F13).

1. What is reused, what is re-expressed
=======================================

``examples/minitpu/template/instances/tinytpu/`` (provisional location,
§8).

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - part
     - status
     - how, and what it cost
   * - ``sequencer``, ``dma_ld``, ``spm``, ``vru``, ``wld``, ``accu``,
       ``dma_st``
     - reused by import
     - ``examples.tinytpu.ip.units.*``, unchanged. The template has no
       counterpart yet (U4/U5 are the sequencer and DMA; the vector units
       are MiniTPU-shaped). Each is int8/int32 through literals (F8).
   * - the PE
     - re-expressed, ``pe.py``
     - ``ip/units/pe.py`` line for line with the MAC taken out of the body:
       ``MAC_ADD(MAC_MUL(activation, weight), psum_north)`` on slot ``MAC``
       (D-15); lane widths ``MAC_IN_BITS``/``MAC_OUT_BITS``; types
       ``MAC_IN``/``MAC_ACC``; instanced ``("T", "T")`` (D-17). The
       activation lane is a constant shift plus a typed assignment (F1).
   * - the parameter set
     - re-expressed, ``geometry.py``
     - ``TinyTpuGeometry``: ``T``, ``MAXDIM``, ``QD``, ``IMEM_SIZE``,
       ``DMA_WORDS``, ``AR_RAW_DIST``, ``mac`` declared; ``VW = T *
       mac.IN_BITS``, ``AW = T * mac.ACC_BITS``, the memory sizes and
       ``WPR``/``OPERAND_ROWS`` as properties (D-20). ``AR_RAW_DIST <= T``
       -- the relation the frozen assembler asserts and nothing checked at
       composition -- is a legality on the record (F9). Builds the frozen
       ``TpuParams`` for the assembler and programs.
   * - the accumulator file
     - re-expressed, ``accumulator.py``
     - ``Option("accumulator")``: units ``accu`` + ``dma_st``, channels
       ``c_acc``/``c_dst``/``ac2sp``, memory ``C``, parameter
       ``AR_RAW_DIST``, slots ``mm``/``vadd``/``vrelu``/``vaddrelu``/
       ``mvout`` (D-19). ``dma_st`` is in the option because without the
       accumulator nothing produces ``ac2sp``: the base has no result path.
   * - the wiring
     - re-expressed, ``instance.py``
     - ``base(geometry)`` (six units, 13 channels, draft) +
       ``with_options(..., ACCUMULATOR)`` = the frozen eight units in the
       frozen declaration order (the option's units append last, and
       ``accu``/``dma_st`` ARE last in ``ip/tinytpu.py``; F10). ``order=
       "sequential"`` declared. ``TinyTpuInstance`` has ``TinyTPU``'s
       interface; ``compare_isa`` is §3.
   * - the MAC engine
     - reused, ``template/engines.py`` ``INT8_INT32``
     - the int8 -> int32 engine track E already had; one edit (W2).
   * - ISA layout, assembler, memory map, programs
     - reused by import
     - ``ip/isa.py``, ``ip/assembler.py``, ``ip/programs.py``. The ISA
       names ride in the geometry's ``namespace()`` because an
       ``Architecture`` takes one parameter set (G5).
   * - the harness
     - copied glue, ``glue/microarch_isa.py`` + ``run_gates.py``
     - the frozen ``microarch_isa.py`` with one switch, ``TPU_INSTANCE=
       template|frozen``, mounted in front of ``examples.tinytpu`` the way
       ``mutate.py`` mounts a mutant tree; the frozen gates run unedited.
       ``gen_isa``'s bare ``import microarch_isa`` is pre-bound to the
       mounted module. ``run_mutate.py`` is the mutation driver (§6).

Not reused: ``ip/units/reduction_tree.py`` (TinyTPU does not compose it;
DotTree does). Track B's ``mxu_pe_unit``/``mxu_unit`` are MiniTPU's
cycle-trace PE with the bf16 MAC inline (impl record P5) and the template's
``matrix_engine`` is a per-row unit, not a T x T array of chained processes;
neither is a TinyTPU PE, so the PE is re-expressed from TinyTPU's own.

2. The composition, and the one ``compose`` change
==================================================

Base + option. The base is six units and thirteen channels. It is not a
legal machine: the frozen sequencer decodes ``OP_MM``..``OP_MVOUT`` and
``put``\ s on ``c_acc``/``c_dst`` whatever is composed, so the base alone is
refused -- ``unit sequencer writes undeclared channel 'c_acc'`` -- which is
D-19's refusal naming the channel. But ``Architecture.with_options`` took a
composed base, and ``Architecture`` checks itself at construction, so the
base could not even be built (F4).

``Architecture(draft=True)`` (``allo/compose.py``, +12 lines): a draft
binds its geometry and skips the netlist check; ``with_options`` composes
it and checks the RESULT, as D-19's text says ("legal iff the netlist rules
pass on the result"); ``source``/``region``/``build``/``plan``/
``directives``/``machine`` of a draft are refused naming it. ``with_options``
also keeps the base's geometry record on the result (it rebuilt from the
namespace and lost it: a D-19 x D-20 seam). Tests:
``tests/test_tinytpu_instance.py`` (6) and the existing
``tests/test_compose*.py`` (28 passed, 3 skipped); the frozen emission is
identical.

Geometry. ``Architecture(parameters=TinyTpuGeometry())`` binds the record;
``_bind_geometry`` holds every property to its value. Refused at
construction: ``AR_RAW_DIST=5`` at ``T=4``, ``T=2`` (``AR_RAW_DIST=4 > T``),
``MAXDIM=96`` (the 11-bit address ceiling, from ``TpuParams``). At ``T=8,
MAXDIM=32`` the derived ``VW``/``AW`` move to 64/256 through the engine's
widths. The option's ``AR_RAW_DIST`` must agree with the geometry's
(``option accumulator sets AR_RAW_DIST=4, which the architecture already
sets to 3`` -- the contract parameter cannot be two numbers).

Engine and channels. ``acol`` and ``cw`` are declared as ``lanes="T"`` of
``MAC_IN_BITS``/``MAC_OUT_BITS``, ``a_fwd``/``p_fwd`` as ``MAC_IN``/
``MAC_ACC``, where the frozen design wrote ``UInt(VW)``/``UInt(AW)``/
``int8``/``int32``; ``compose``'s D-15 check holds the PE's ``get``/``put``
annotations to them. The emitted region declares ``acol: Stream[UInt(T *
MAC_IN_BITS), QD][T]`` and the PE body calls ``mul_int8``/``add_int32``/
``pack_int32`` as ``func.func``\ s (``func.call`` in the MLIR; Vitis emits
them as functions and inlines).

3. ``isa_slots`` against ``isa_spec.json``
==========================================

``compose.isa_slots(arch)`` of the composed instance, against the spec's
eleven opcodes. The spec has no module field; the module an opcode belongs
to is DERIVED from its ``actions`` -- an opcode with an action at ``accu``
or ``dma_st`` is the accumulator's (``instance.spec_modules``).

.. list-table::
   :header-rows: 1
   :widths: 16 20 20 44

   * - opcode
     - instance (``isa_slots``)
     - spec (derived from ``actions``)
     - note
   * - ``nop``, ``loop``, ``endloop``
     - base
     - base
     - no actions; the sequencer's own
   * - ``dma_ld``, ``vld``
     - base
     - base
     - actions at ``dma_ld``/``spm``/``vru``
   * - ``dma_st``
     - base
     - base
     - opcode 2, retired: the sequencer dispatches nothing for it (a ``nop``);
       kept a base slot because the spec numbers it
   * - ``mm``
     - accumulator
     - accumulator
     - its sink is the accumulator file (``accu`` ``ar.write``); the base
       has the array and no place for its result
   * - ``vadd``, ``vrelu``, ``vaddrelu``
     - accumulator
     - accumulator
     - the accumulator's ALU
   * - ``mvout``
     - accumulator
     - accumulator
     - ``accu`` reads ``ar``, ``dma_st`` writes ``C``

Missing in the instance: none. Extra: none. Module differs: none.
``run_gates.py gen_isa`` runs this as an arm beside ``gen_isa.py``'s
(``isa slots: 11 of the composed instance == the spec's 11 opcodes``).

Gaps for "ISA spec and ``gen_isa`` generalised to instances":

G1. **No module attribute in the spec.** Membership is derived from
    ``actions``; a spec ``"module": "accumulator"`` per opcode, held to
    ``isa_slots(arch)`` by a ``gen_isa`` arm, is the generalisation D-19's
    third bullet names ("the ISA table is derived from the composition
    rather than written beside it"). The arm exists in ``run_gates.isa_arm``;
    the field does not.
G2. **``gen_isa`` takes the design as PATHS, not as an object.** Of its ten
    ``--check`` arms, four read files of ``examples/tinytpu/``:
    ``check_design_parameters`` parses ``microarch_isa.py``'s source for the
    ``os.environ`` reads; ``check_design_slices`` scans ``ip/units/*.py``
    for 64-bit-word slices (the instance's ``pe.py`` is not scanned -- it
    slices no control word, so nothing is missed today);
    ``check_parameter_agreement`` re-imports ``microarch_isa`` in fresh
    interpreters (the FROZEN module: it is not run here);
    ``check_actions`` compares the spec's dispatch with
    ``ip/units/sequencer.py``'s text and each unit's ``isa=`` with
    ``ip.tinytpu.units()``; ``check_reference`` reads ``isa_ref.py``. They
    pass on the instance because the instance reuses those files. The
    generalisation is one signature: ``check(spec, design)`` with
    ``design`` an ``Architecture`` plus its glue module, the slices read off
    ``Unit.source()`` and the dispatch off the composed sequencer.
G3. **The sequencer's dispatch is not an option delta** (F5): the spec's
    ``dispatch`` and each opcode's ``actions`` already say which queues an
    opcode feeds; a sequencer generated from them per instance is what
    makes the base a runnable machine and the ISA truly derived.
G4. **The spec names ``array = wld + pe``** as one unit with a ``mac`` port
    and ``matmul`` compute; the engine (``numerics.active = int8``) is a
    sibling section. An instance's engine is what ``numerics`` should name:
    ``"engine": "int8_int32"`` with the widths derived from it, as the
    geometry derives ``VW``/``AW``.
G5. **One namespace per ``Architecture``.** The ISA constants (``OP_MM``,
    the field bounds) are bound through the geometry's ``namespace()`` so
    the units can read them; a geometry record is not where an opcode
    number belongs. ``Architecture(parameters=, constants=)`` or the
    option carrying its own opcode numbers would separate them.

4. Gates on the instance
========================

All at ``TPU_MAXDIM=16`` (where the reference numbers are measured), T=4,
QD=16; ``TPU_INSTANCE=frozen`` through the same mount is the control.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - check
     - instance
     - control / reference
   * - ``run_gates.py bench`` (``bench_isa.py``)
     - ALL EXACT (5 shapes x {gemm, gemm.relu} x {flat, loop}, vadd+relu)
     - frozen through the mount: ALL EXACT
   * - ``run_gates.py stress`` (``stress_isa.py``)
     - STRESS OK 492/492 (18 crafted bad programs rejected, 390 generated
       accepted; 64 GEMM shapes, vector and random programs)
     - frozen: 640/640 at MAXDIM=64 (impl record); 492 is the MAXDIM=16 count
   * - ``run_gates.py gen_isa --conform``
     - ISA OK: constants, layout, parameters, slices, behaviour (2700
       encoder words, 73 programs), actions (45 over 7 units), derived
       properties (5), isa slots (11/11), emitted HLS (6/6 fields)
     - frozen ``gen_isa.py --conform``: ISA OK (impl record)
   * - ``tests/test_tinytpu_instance.py``
     - 6 passed
     - --
   * - ``tests/test_compose*.py``
     - 28 passed, 3 skipped (csim needs Catapult)
     - 28 + 3 before
   * - ``run_u3e --quick``
     - GATE OK, 1 finding (int8 DIM=4: F1)
     - same before
   * - frozen Vitis emission, sha256
     - ``6bc774bc...b2ef95``
     - identical (before compose.py's change, after it, after engines.py's)
   * - composition vs ``ip/tinytpu.py``
     - same 8 units in order, same 16 channels, 4 memories, parameter set
     - --
   * - ``run_gates.py cosim`` at 4x4x4 (§5)
     - 175 cycles, 0 mismatches
     - frozen through the mount: 175; published 175
   * - ``run_mutate.py`` (§6)
     - 34 caught, 0 survived, 4 not applicable, 1 RTL-only
     - frozen ``mutate.py``: cannot locate ``vadd_holds_stale_x`` (F11b)

The functional gates needed two workarounds to run (W1, W2 in §7); with
them nothing in the gates was edited or relaxed.

5. Cycles against the frozen reference
======================================

One shape, per the owner's scope change; the instance and the frozen
design through the SAME path on the same host, ``TPU_MAXDIM=16``, default
testbench, one csynth each.

.. list-table::
   :header-rows: 1
   :widths: 16 20 20 20 24

   * - shape
     - instance
     - frozen, same host
     - published reference
     - mismatches
   * - 4x4x4
     - **175**
     - 175
     - 175
     - 0 / 16 (both)

COSIM5_PENDING

The instance's RTL is a different build from the frozen one -- its PE calls
``mul_int8``/``add_int32``/``pack_int32`` (``func.call``, emitted as C++
functions Vitis inlines) and takes the activation lane by a shift -- and
lands on the same cycle count, so the engine form cost nothing at this
shape and the two workarounds (W1, W2) are wiring in the schedule.

**Getting the number took a host fix, not a design fix (F13).** On
zhang-21 the cosim's testbench link fails for the frozen design exactly
as for the instance: ``/usr/bin/ld`` is binutils 2.30 and cannot read the
compressed ``.debug_info`` Vitis's gcc emits (``unable to initialize
decompress status for section .debug_info``, then ``file format not
recognized``). ``cosim.py``'s ``-B/usr/bin`` is the fix for ace-01's 2.42
(``dev/toolchains.rst``). The ``allo`` env carries binutils 2.44
(``x86_64-conda-linux-gnu-ld``); ``run_gates.py cosim`` links it into a
scratch directory and points ``cosim_design``'s ``-B`` there (``TPU_LD_DIR``
overrides). This means ``reproduce.sh``'s cosim stage has not been passing
on zhang-21 as the tree stands; ``dev/toolchains.rst`` should say so and
``cosim.py``'s ``LDFLAGS`` should pick the linker per host -- the frozen
file was not edited here (D-4).

6. Mutation
===========

``run_mutate.py``: ``mutate.py``'s MUTANTS table and levels (bench, stress;
no cosim level) on the instance's tree -- the reused ``ip/`` units, this
package's files, ``engines.py`` and the glue -- written under
``$TPU_MUTANTS`` (scratch), never under the frozen tree. A frozen mutant
whose anchor is in the frozen PE is NOT APPLICABLE (the PE is
re-expressed); five instance mutants replace them: three on the instance's
PE (activation lane ``j`` not ``i``; shadow weight not swapped; partial sum
not forwarded) and two on the ENGINE (product truncated to int8;
accumulate subtracts), which is where a MAC mutant lives once the MAC is a
plug-in.

``TPU_MUTANTS=scratch/ttinst_mutants``, ``--jobs 6``, 40 mutants named,
``TPU_MAXDIM=16``:

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - outcome
     - count
     - which
   * - caught at ``bench_isa``
     - 21
     - ``pe_weight_lane_swapped``, ``wld_rows_from_weight`` (hang),
       ``spm_weight_off_by_one``, ``vru_act_off_by_one``,
       ``vru_dma_ignores_f3``, ``dma_ld_src_swapped``, ``prefetch_lane7_dup``
       (hang), ``agu_stride_halved``, ``agu_f3_dropped``, ``loop_extra_trip``
       (hang), ``vrelu_rows_short`` (hang), ``mm_acc_dropped``,
       ``mm_always_acc``, ``mm_dst_base_ignored``, ``relu_off_by_one``,
       ``vadd_subtracts``, ``assembler_span_short``,
       ``inst_pe_act_lane_swapped``, ``inst_pe_shadow_not_swapped``,
       ``inst_pe_psum_not_forwarded``, ``engine_add_int32_sub``
   * - passed ``bench_isa``, caught at ``stress_isa``
     - 13
     - ``spm_vld_off_by_one``, ``dma_ld_row_ignored``, ``mvout_row_ignored``,
       ``dma_st_accumulates_C``, ``vrelu_dst_is_src``, ``vadd_dst_is_src1``,
       ``vadd_src2_is_src1``, ``vrelu_src_base_ignored``,
       ``mvout_src_base_ignored``, ``clip_hi_off_by_one``,
       ``clip_lo_off_by_one``, ``ar_contract_unenforced``,
       ``engine_mul_int8_truncated``
   * - survived
     - 0
     - --
   * - not applicable
     - 4
     - ``pe_psum_int16``, ``pe_act_lane_swapped``, ``pe_shadow_not_swapped``
       (anchors in the frozen PE's text; replaced by the three ``inst_pe_*``
       above -- ``pe_psum_int16`` has no counterpart: the partial sum's
       width is the ENGINE's ``ACC`` now, and narrowing it is
       ``engine_mul_int8_truncated``'s kind of mutant); ``vadd_holds_stale_x``
       (see below)
   * - RTL-only, not run
     - 1
     - ``ar_claim_false`` (needs the cosim level; passes both functional
       levels by design)

``MUTATE OK: 34 caught, 0 survived``. The control (``none``) passed both
levels through the mount. The engine mutant ``engine_mul_int8_truncated``
passes ``bench_isa`` (operands in [-4, 4]: no product leaves int8) and is
caught by ``stress_isa``, which is the performance-vs-correctness split
``tinytpu_isa.rst`` documents, now observed on a plug-in.

**``vadd_holds_stale_x`` cannot be located in the FROZEN suite either**
(F11b): its anchor ``vadd_first = read_word`` occurs twice in
``ip/units/accumulator.py`` since the ``vaddrelu`` arm was added, and
``mutate.py``'s ``locate`` requires exactly one occurrence across the
design, so ``reproduce.sh --with-mutants`` fails loudly on it ("a refactor
that moves or renames anchored code makes this script fail loudly"). Found
by running the table on the instance; fix in the frozen table (anchor on
``elif op == OP_VADD:``), for the fork's issues.

7. Findings
===========

Classified: **bug** (a tool does the wrong thing), **missing abstraction**
(nothing can say it), **workaround** (what was done instead), **semantic
mismatch** (two layers mean different things). Core bugs are proposals.

F1. **Bug, front end (core, proposal).** ``allo/ir/infer.py``
    ``TypeInferer.visit_symbol`` makes EVERY ``ast.Name`` a sympy symbol and
    resolves no global, so a slice ``w[MAC_IN_BITS * i : MAC_IN_BITS * (i +
    1)]`` has width ``MAC_IN_BITS`` (symbolic), is not an Integer, and
    falls to "Cannot infer the bitwidth of the slice, use UInt(32)". With
    literals, ``8 * (i + 1) - 8 * i`` is 8. At T=4 int8 the activation word
    is 32 bits, so the default IS the word: ``allo.get_slice(... i32) ->
    i32`` then ``arith.trunci i32 -> i32``, "cast incompatible", and no
    build. This is the root of the template's recorded finding "systolic
    engine at int8, DIM=4 (32-bit packed word): lowering fails"; at DIM=16
    the same unit WORKS only because a 32-bit extract then a truncation to
    the lane type yields the right bits -- a different circuit from the
    one written (D-17 said as much). *Proposal:* in ``visit_symbol``, an
    ``ast.Name`` bound in ``ctx.global_vars`` to an ``int`` returns
    ``sympy.Integer(value)``; then a parameter in a slice bound folds, and
    D-17's "until the front end folds constants into bounds" closes.
    *Workaround W1* (``pe.py``): ``lane_shifted: UInt(T * MAC_IN_BITS) =
    activation_word >> (MAC_IN_BITS * i)`` and ``activation = lane_shifted``
    (the typed assignment keeps the low ``MAC_IN_BITS`` bits); a constant
    shift is wiring. The result-word STORE ``result_word[MAC_OUT_BITS * j :
    ...] = MAC_PACK(psum)`` also warns, but ``allo.set_slice`` takes its
    width from the value (``%26 : i32`` at ``[31:0]``), so it is correct --
    a silent default that happens to be right.
F2. **Bug, compose.** ``Unit._check_slice_bounds`` refuses a bound whose
    names are ONLY free names and accepts one that also names a bound
    index, on the premise that "inside ``meta_for`` the expression folds".
    It does not: ``MAC_IN_BITS * (r + 1) - MAC_IN_BITS * r`` is as symbolic
    as the free form; a pid ``i`` is the same. The check lets F1 through.
    *Proposal:* refuse any slice bound naming a parameter until F1 lands,
    then accept all of them.
F3. **Bug, core pass (proposal).** ``allo/passes.py analyze_use_def``: for
    a ``func.return``, ``ret.owner.attributes`` assumes the returned value
    is an op's result; ``pack_int32(v) -> return v`` returns its own
    parameter, the owner is a ``Block``, and ``s.partition`` (any
    ``_get_equivalent_buffers``) raises ``AttributeError: Block has no
    attribute 'attributes'``. Seen at ``schedule(s)`` in ``cosim.py``;
    invisible on the simulator and in every compose test (none partitions).
    *Proposal:* skip a ``BlockArgument`` owner (``isinstance(ret,
    BlockArgument)``) in that loop. *Workaround W2* (``engines.py``):
    ``r: int32 = v; return r``.
F4. **Missing abstraction, compose (closed).** A base an option completes
    could not be built: ``Architecture`` checks itself at construction and
    ``with_options`` wanted a built base. ``Architecture(draft=True)`` (§2).
    Also: ``with_options`` dropped ``geometry``; kept now.
F5. **Missing abstraction, D-19.** The sequencer's DISPATCH is not an
    option delta. ``Option`` adds units, channels, memories, parameters,
    engines, rebinds and slots, but the decoder that feeds the option's
    queues is written into the base's sequencer (``if op == OP_MM:
    c_acc.put(...)``). So the base is a draft, never a machine, and
    ``check_program`` refusing ``mm`` without the accumulator is a software
    refusal with no hardware behind it (the sequencer would dispatch it to
    a queue nobody reads and hang). The spec already carries each opcode's
    dispatch (``actions``, ``dispatch``); a sequencer GENERATED per instance
    from the composed slots (G3) is the mechanism. MiniTPU's v2 decoder
    refusal (D-19 last bullet) is the RTL-side half of the same thing.
F6. **Semantic mismatch, ``mm``.** On TinyTPU ``mm`` is an accumulator
    instruction: its result has nowhere to go without the accumulator
    file, so the option brings ``mm`` although the array is base. On
    MiniTPU ``vmatpush``'s result pops to the vector register file and the
    accumulation is ``vadd``. The ``Option`` form states this correctly
    (``isa_slots`` puts ``mm`` on the accumulator), where a unit list with a
    conditional would have put ``mm`` with the array.
F7. **Missing abstraction, D-20 x D-17.** ``Unit.legality`` receives the
    parameter namespace, not the engine namespace, so a width relation
    (``VW == T * MAC_IN_BITS``) cannot be a unit legality; here it holds by
    DERIVATION (the geometry computes ``VW`` from ``mac``) and by
    ``compose``'s ``lane_bits`` check on the PE's channels. And a reused
    frozen unit takes no legality from outside (impl record P7, by design),
    so ``AR_RAW_DIST <= T`` sits on the geometry record, not on ``accu``.
F8. **Missing abstraction (the headline).** The engine swap reaches one
    unit. ``wld`` packs the weight lane at ``pe_word[0:8]`` and the row
    count at ``[8:20]``; ``spm``/``vru``/``dma_ld`` move ``UInt(VW)`` words
    of ``8 * lane`` slices; ``accu`` slices ``32 * lane`` and clips to
    ``int8``; ``dma_st`` writes ``int8``; the DRAM boundary is ``int8[...]``.
    A ``TinyTpuGeometry(mac=BF16_ACC24)`` composes (the widths follow) and
    would compute garbage: the reused units are int8/int32 through literals
    that no check holds to the engine, because they bind no engine slot
    (D-15's check fires only in units that do). What the template CAN say
    is the PE; what it cannot yet say is that a machine's arithmetic is one
    declaration every unit binds. *Proposal:* a unit that slices a packed
    word by a literal lane width declares the width it assumes
    (``lanes=``/``lane_bits=`` on the unit, held to the channel), or binds
    the engine's ``*_BITS`` as slots -- which is F1's fix made necessary.
F9. **Workaround, D-20.** ``AR_RAW_DIST <= T`` as a geometry legality
    (see F7). The frozen ``Assembler.__init__`` still asserts it; two
    checks of one relation until the assembler reads the composed
    instance (D-20 third bullet).
F10. **Potential semantic mismatch, ``with_options`` ordering.** An
    option's units are appended LAST. Here that reproduces the frozen
    declaration order because ``accu`` and ``dma_st`` are last in
    ``ip/tinytpu.py``; an option whose unit sits mid-stream (the SFU
    between ``alu`` and ``writeback``) lands after its consumer, which is
    the order Vitis csim runs processes in (limitations item 15). Not
    measured here (no csim level); ``Option(after="alu")`` or a dataflow
    sort of ``units`` is the fix.
F11. **Bug, harness scope.** ``mutate.py`` writes under
    ``examples/tinytpu/.mutants`` and locates anchors in the frozen files
    only; a design in another location needs its own driver
    (``run_mutate.py``). Three frozen PE mutants are NOT APPLICABLE to the
    instance (their anchors are the frozen PE's text), and the MAC mutants
    belong on the engine now (§6).
F13. **Bug, host toolchain (zhang-21).** See §5: the Vitis cosim testbench
    does not link with ``/usr/bin/ld`` 2.30 on this host, for the frozen
    design as for the instance; ``-B`` at the ``allo`` env's binutils 2.44
    links. For ``dev/toolchains.rst`` and ``cosim.py``'s ``LDFLAGS``.
F12. **Observation, D-18.** ``INT8_INT32`` carries no directives, and the
    four reused units' own directives (``sequencer`` partition, ``dma_ld``
    partitions, ``accu``'s ``s.dependence``, ``dma_st`` partition) apply
    through ``Architecture.directives`` with ``ctx.instance("accu")``
    resolving to the same names as before; ``accu``'s obligation quotes
    ``ctx.parameters["AR_RAW_DIST"]``, which the option supplies. The
    engine's ``latency={"mul": 1, "add": 1}`` is booked by nothing yet: no
    TinyTPU number derives from it (the harness compares bookings with
    manifests only for MiniTPU units).

8. The location question (for the owner)
========================================

The instance is at ``examples/minitpu/template/instances/tinytpu/``: the
template is MiniTPU's home (D-3) and its instances sit beside its
mechanisms. Alternatives: ``examples/tinytpu/instance/`` (beside the frozen
design it will retire, D-4 -- but inside the frozen tree, which this track
must not edit), or a top-level ``examples/instances/``. The harness glue
(``glue/``, ``run_gates.py``, ``run_mutate.py``) is a copy by the track's
rule; when the frozen design retires, the switch collapses and the gates
can import the instance directly. The README's milestone row for this
track is the owner's to update; nothing here edits it.

9. Not done
===========

- The cycle table beyond what §5 shows (owner's scope change: one shape).
- ``TPU_TB=stress`` cosim and ``mutate --cosim`` (RTL-only mutant
  ``ar_claim_false``): not run; the accumulator's dependence obligation is
  discharged on the frozen design only.
- A second geometry through RTL (``T=8``): composes and is checked
  (``test_geometry_relations_are_legality``), not built.
- Docs: ``actions.rst`` has the ``draft`` paragraph; ``tinytpu_library.rst``
  does not point here yet (after the owner settles §8).
