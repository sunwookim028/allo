..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

###################################
The ASIC manifest
###################################

``allo/backend/asic`` (see its ``PROVENANCE.md``) has two entry points, and
they sit on opposite sides of one question.

The **flat flow** consumes a directory of Verilog and an ``sv2v_manifest.f``
file list, entered at the design collector. It knows nothing about Allo, and
every ASIC number the fork has published came through it — which is why those
runs were fed by *committed RTL exports* rather than by the compiler.

**AAAH**, the ``allo-*`` nodes, consumes Allo-level architecture, and **not
through files**. Its compilation node loads an Allo design and *calls*
``build(project, target, mode, configs)`` with
``configs["asic_manifest"] = {...}``: Allo is asked to emit, as part of
compilation, the architecture it just compiled. Downstream planners then place
macros on a grid from PID row/column deltas, weight the wires between them by
**pre-HLS stream bit width**, and decide which RTL modules are one macro by
*semantic identity* rather than by name. None of that survives in emitted
Verilog, so nothing later in the flow can recover it or check it.

Until 2026-09-24 this fork did not emit that manifest, which is why the TinyTPU
runs were routed around the compilation node. :mod:`allo.backend.asic_manifest`
is the emitter.

.. important::

   Emitting the manifest is not the same as closing the loop. Allo now emits
   what its own ASIC backend consumes, verified against the consumer and
   against TinyTPU's known post-HLS structure; **an end-to-end PD run from the
   specification has not been performed**, and cannot be on a host without a
   Synopsys DC licence. What still has to be proved is at the bottom of this
   page.


What is emitted, and when
=========================

Two files, matching what ``nodes/allo-asic-compilation/validate_build.py``
checks for, each with a ``.tcl`` sibling because the flow's nodes stage both:

``asic-manifest.json`` — ``stage: "pre_hls"``
    The architecture. Written inside ``allo.dataflow.build()``, from the
    schedule that is about to be handed to the backend, **before any HLS tool
    runs** — so it exists whether or not a licence does.

``asic-manifest-final.json`` — ``stage: "post_hls_enriched"``
    The same records joined to the RTL the backend produced. Written from
    :class:`~allo.backend.hls.HLSModule` once ``csyn`` has succeeded, because
    only then do the module names exist to join to.

``asic-debug/`` holds flat TSV views of the same records, for reading by eye.

Nothing happens unless ``configs["asic_manifest"]["enabled"]`` is set, so the
ordinary Vitis and Catapult paths are unchanged.

.. code-block:: python

   import allo.dataflow as df

   df.build(
       design,
       target="vitis_hls",
       mode="csyn",
       project="top.prj",
       configs={
           "frequency": 300.0,
           "device": "u280",
           "asic_manifest": {
               "enabled": True,
               "path": "asic-manifest.json",
               "debug_artifacts": True,
               "debug_dir": "asic-debug",
           },
       },
   )

That ``configs`` dict is not an invention of this page: it is verbatim what
``nodes/allo-asic-compilation/backend.py`` builds and passes in.


The schema, as the consumers define it
======================================

The schema is not documented anywhere in the flow; it is whatever its readers
read. The fields below are the ones actually consumed, with the consumer that
demands each.

Envelope
--------

``stage``
    ``"pre_hls"`` / ``"post_hls_enriched"``. ``validate_build`` rejects
    anything else.
``top``
    The RTL top module name. ``plan_assembly`` requires it to be both the
    node's ``top_module`` parameter and a module present in ``design.v``.
``backend``
    ``vitis`` / ``catapult`` / ``systemc``; ``plan_macros`` and
    ``plan_assembly`` refuse a mismatch with their own parameter.
``summary``
    ``unmatched_or_ambiguous`` and ``unjoined_post_hls_records`` **must both
    be 0**, checked twice (``validate_build``, ``plan_macros``).
``top_arguments[]``
    Order is load-bearing: it must equal the frozen workload's
    ``call_signature``. Each entry carries ``ordinal``, ``name``, ``shape``,
    ``type``, ``direction``.

``pe_instances[]``
------------------

``semantic_id``
    ``<top>/<kernel>/pid=<i>[,<j>]``. The format is fixed by
    ``plan_physical_intent``: it takes the first two ``/``-segments as the
    kernel key and regex-matches ``pid=([0-9,-]+)``. Must be unique --
    several consumers key a dict on it with a bare ``[]`` lookup.
``kernel``
    The bare kernel name (*not* ``top/kernel``).
``pid``
    List of ints. ``plan_macros`` turns a +-1 delta on the last axis into
    ``E``/``W`` and on the second-to-last into ``S``/``N``; the row index
    increases **southward**.
``ports[]``
    The stream interface, in a dense 0-based ``ordinal`` order, each with
    ``channel_id``, ``stream``, ``direction`` (``in``/``out``) and ``type``.
    The flow zips this positionally against the RTL's handshake bundles, so an
    ``m_axi`` memory argument is **not** one of them.
``post_hls_records[]``
    Post-HLS only: ``rtl_root_module``, ``rtl_modules[].name``,
    ``rtl_equivalence_hash``.

``channels[]``
--------------

``channel_id`` and ``stream``
    Provenance; ``ports[].channel_id`` joins to it, and a ``null`` there means
    a SystemC-internal channel.
``type``
    The MLIR stream type. ``plan_physical_intent`` takes the **first**
    ``i<N>``/``u<N>`` in it as the placement edge weight, so
    ``!allo.stream<i32, 16>`` weighs 32 and the depth is ignored.
``endpoints[]``
    ``pe`` (a ``semantic_id``), ``role`` (``producer``/``consumer`` -- read by
    physical intent), ``direction`` (``in``/``out`` -- read by macro planning),
    and ``accesses[].port_ordinal``. Both vocabularies are emitted, because the
    two nodes read different ones.

``macro_groups[]`` (post-HLS)
-----------------------------

``macro_class_id``, ``representative`` (must be in ``pe_instances``),
``member_count``, ``members[]`` (``semantic_id``, ``rtl_module``,
``orientation``), and ``proof.status`` — which must be exactly ``"proven"`` or
the class is silently dropped.


What the emitter derives it from
================================

Everything comes out of the **realized dataflow IR**, never from a table
someone typed. A hand-written manifest would be a second editing surface for
the architecture, which is precisely what having one Allo specification exists
to remove.

After ``allo.dataflow.customize``, the module is one ``func.func`` per kernel
instance plus a top function whose body constructs every stream and then calls
each instance with the streams it is wired to. **That body is the netlist.**

* **PE instances** — every ``func.func`` carrying ``df.kernel``. Its name is
  ``<kernel>_<i>_<j>``; the region's own ``mappings`` dict gives the split
  exactly (a kernel whose name ends in ``_<digits>`` is otherwise
  indistinguishable from one grid axis).
* **Ports** — the ``stypes`` attribute, stamped per argument by
  ``move_stream_to_interface``: ``i`` read, ``o`` written, ``_`` not a link.
* **Channels and their endpoints** — the SSA value each ``allo.stream_construct``
  defines, matched against the operands of each ``func.call``.
* **Top arguments** — ``allo.dataflow._build_manifest_top_arguments`` pairs the
  *realized* argument order with ``analyze_arg_load_store``'s directions.
  A region's arguments are not necessarily emitted in declaration order
  (:doc:`/developer/pitfalls`), and direction is a fact about the body.
* **Memories** — ``postprocess_hls_code`` pragmas argument *i* onto
  ``bundle=gmem{i}``, so the manifest names the ``<top>_gmem<i>_m_axi`` adapter
  Vitis will generate for it.

The emitter refuses what the consumers assume and never state: a duplicate
``semantic_id``, a non-dense port ordinal, an endpoint naming an unknown PE or
an ordinal that PE does not have, and a channel with two writers or two
readers. Each of those is otherwise a ``KeyError``, a ``StopIteration`` or a
silently misplaced macro several nodes downstream.

Post-HLS, the join walks Vitis's naming: a kernel becomes ``<top>_<instance>``
(sometimes with a numeric disambiguator — ``dma_ld_0`` becomes
``tinytpu_isa_dma_ld_0_1``), with ``<root>_...`` children for each pipelined
loop it split out. Which modules belong to no kernel is decided by *this
design's* kernel names rather than by a list of vendor module-name patterns:
an unclaimed module named after one of the design's kernels is the error, and
everything else is plumbing.


Validated against TinyTPU
=========================

At the shipped configuration (``TPU_T=4``, ``TPU_MAXDIM=64``) the emitter
produces, entirely from the design:

* 8 kernels and **38 PE instances** — ``pe`` and ``wld`` at 4×4, the other six
  singletons — matching ``TPU.architecture``'s ``instances=``;
* **85 channels**, one per element of the 16 declared ``Channel``\ s including
  the array ones, of which 12 (``3 * T``) are the terminal elements of the
  three chains and carry no endpoints — the last column forwards nothing east
  and the bottom row sends ``cw`` rather than a partial sum south;
* 4 memories on ``gmem0``–``gmem3``.

The post-HLS join, run against the committed 145-file RTL export, reproduces
the module names in the shipped ``T4_MAXDIM64_shipped`` hierarchy report
exactly: ``accu_0``, ``spm_0``, ``vru_0``, ``dma_ld_0_1``, ``sequencer_0_1``
and ``gmem0..3_m_axi``, with 0 unmatched PEs and 0 unjoined records.

The sharpest check is that the flow's *own* planner reads the array back out.
``plan_macros.graph_pin_sides`` is given PIDs and channel endpoints and nothing
about systolic arrays; on ``pe(1,1)`` it returns ``a_fwd`` in from the **W**,
out to the **E**, and ``p_fwd`` in from the **N**, out to the **S**. 90 of the
146 stream pins resolve as ``same_kernel_neighbor``.

``tests/dataflow/test_asic_manifest.py`` runs all of this, computing its
expectations from ``TPU.architecture`` rather than hard-coding them, and loads
``plan_macros`` and ``plan_physical_intent`` from
``allo/backend/asic/nodes/`` so that "the consumer accepts it" is executed
rather than asserted.


What is *not* emitted
=====================

**Per-argument Catapult RTL protocol capture.** ``allo-testbench-generation``
wants ``catapult_argument``, ``data_ports``/``triosy_ports``, ``packing`` and
``interface.roles`` on the ``catapult``/``systemc`` path. Those are a parse of
Catapult's generated RTL, not an architectural fact, and they belong to a
separate post-HLS extractor. The Vitis path — the one every TinyTPU ASIC number
came from — is complete; the post-HLS join is hooked only to Vitis ``csyn``
for the same reason, because emitting a wrong join would be worse than
emitting none.

**Arithmetic.** A composed region declares what a unit is wired to and never
what it computes, so the manifest carries no compute ports. That is the Action
layer's to add (:doc:`/developer/actions`).

**Macro reuse, on the Vitis path.** Each of TinyTPU's 38 PEs hashes to its own
macro class, so ``member_count`` is 1 everywhere. This is not a defect in the
hashing: Vitis specializes every instance, and two interior PEs that are the
same hardware are emitted with *different port orders*, which
``plan_macros.validate_and_map_ports`` correctly refuses to call equivalent.
Macro planning at the default ``min_macro_reuse=2`` would therefore find no
class to harden. Getting reuse back is a question about how the design is
emitted, not about the manifest.


What a licensed host would still have to prove
==============================================

Success here is that the consumer accepts the manifest. It is **not** a PD run.
On a host with Vitis HLS *and* a Synopsys DC licence, the path from
specification to PPA would have to show, in order:

#. ``run_design.py`` → ``validate_build.py`` passing on a real ``csyn``: the
   four manifest files present, both stages correct, both summary counters 0,
   and the frozen workload's interface matching ``top_arguments``. Everything
   but the ``csyn`` is exercised offline today.
#. ``sv2v-rtl-allo`` and ``allo-asic-macro-plan`` running on that manifest.
   This is where the reuse ceiling above bites first, and where the failure
   would be ``no proven macro classes meet reuse threshold`` rather than
   anything about the manifest's shape.
#. ``allo-asic-assembly-plan`` → ``allo-asic-physical-intent`` →
   ``allo-asic-rtl-assembly``, which additionally require every
   ``stable_instance_name`` to survive into the synthesized netlist, and the
   ``macro-link.rpt`` count to match ``elaborated_macro_instance_count``.
#. A DC run that does **not** dissolve the boundaries. At the flattening effort
   the published runs use, synthesis flattens the hierarchy again unless told
   otherwise — and changing that makes new numbers non-comparable with the four
   already published. The manifest buys planning and attribution; it does not
   by itself buy a hierarchy-preserving netlist.

What could still be wrong at that point, honestly: the Catapult/SystemC
per-argument capture is absent, so those two backends cannot reach the
testbench node; the post-HLS join is validated against one design's naming and
a second design could exercise a Vitis renaming this walk does not cover (the
counters would say so rather than the join guessing); and ``top_interface`` on
the non-Vitis paths is a documented default rather than an observation.
