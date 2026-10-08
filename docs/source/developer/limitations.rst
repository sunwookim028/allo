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

#########################
Allo Limitations Register
#########################

This page lists what a user of this fork must know about Allo's limitations:
concrete obstacles met while building accelerators on Allo, each of which
required either a workaround in user code or a patch to the Allo library.
**Live status is in the fork's GitHub issues** (README, decision D-2); the
tables below are the 2026-09-19 re-verification. The full register, with every
root-cause analysis, dated correction and measurement, is archived verbatim at
`dev/records/limitations/register_2026-10-01.rst <https://github.com/sunwookim028/allo/blob/main/dev/records/limitations/register_2026-10-01.rst>`__. It was consolidated from two
passes:

* the first pass (items 1-10), from adding a VPU, a hardware loop and an
  on-device transpose to an L2 TPU running FlashAttention
  (``levels/L2/tpu.py``);
* the second pass (items 11-23), from building the instruction-programmable
  TinyTPU-isa (:doc:`/designs/tinytpu_isa`) on ``main`` and taking it through
  the Vitis dataflow path to RTL co-simulation;
* the third pass (item :ref:`24 <limitation-24>`), from compiling *for* that
  design rather than building it -- generated programs that every cheap check
  accepts and the RTL does not run.


Gaps in Allo's *abstractions* -- things that are missing a type, a primitive
or a pass rather than a bug fix -- are ranked with their legality rules in
:doc:`/developer/extending_allo`, which is also the standard a new primitive
has to meet before it lands.

Related feature-gap tracking lives as fork issues and is not restated here:
combinational wires (fork issue #9), HLS dependence pragma (fork issue #10),
shared mutable memory across kernels (fork issue #11; relates to items 1-2),
streams as top-level inputs (fork issue #12), and the nested sub-region Stream
compile-time-constant shape constraint (fork issue #4; item :ref:`H <limitation-h>`). The
living fork-vs-upstream feature map is the pinned fork issue
https://github.com/sunwookim028/allo/issues/13.


Register
--------

Every item was re-verified on ``main`` at ``7a24c21e`` on 2026-09-19. Each
older item has a standalone repro under ``tests/limits/`` that prints one
``[item N] <STATUS>`` line; the repros were run again against ``main``'s
working tree (after the upstream #612 merge ``dc6b8fa6``) with the same
verdicts. Statuses:

* **REPRODUCES**: the limitation is present.
* **FIXED by <commit or PR>**: a commit removed it. Fixes marked
  *fork-only* exist only on this fork and are **upstreaming candidates**.
* **CANNOT-REPRODUCE**: the claimed failure does not occur, and no fix
  explains it (it may never have been broken).
* **NOT-A-LIMITATION**: the behaviour is not Allo's.

Layers: *frontend* (``allo/ir/infer.py``, ``allo/ir/builder.py``,
``allo/customize.py``), *simulator* (``allo/backend/simulator.py``), *HLS
driver* (``allo/backend/hls.py``, ``allo/backend/vitis.py``), *emitter*
(``mlir/lib/Translation/Emit*.cpp``), *SystemC fork* (the
``choonsik1/allo:SystemC-emitter`` emitter). File:line references are to
``7a24c21e`` unless a fork commit is named. Fix sizes are the root-cause
agent's estimates unless marked *verified* (a prototype was built and
passed).

To run a repro from a checkout of this repository:

.. code-block:: bash

   source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
   cd tests/limits && python item04_math_exp.py      # -> [item 4] REPRODUCES: ...

``tests/limits/_worktree.py`` makes every repro test the tree it is committed
in: from the primary checkout it does nothing, and from a separate worktree
(which needs its own built ``allo/_mlir``) it strips the conda env's editable
install. ``item15`` needs ``vitis_hls`` on ``PATH`` and takes about 4 minutes;
every other repro is codegen or simulator only. The files are not named
``test_*.py``, so ``pytest`` does not collect them.

.. _limitations-open:

Open
~~~~

.. list-table::
   :header-rows: 1
   :widths: 6 12 9 26 20 10 17

   * - Item
     - Status
     - Layer
     - Root cause
     - Impact on TinyTPU-isa
     - Fix size
     - Repro
   * - :ref:`4 <limitation-4>`
     - REPRODUCES
     - frontend
     - ``infer.py:1218`` treats only ``allo``-module functions as library ops;
       ``math.exp`` falls through to the user-function lookup at
       ``infer.py:1316`` -> ``KeyError``. The documented workaround,
       ``allo.exp``, does not run on the simulator (item A).
     - none (no transcendental ops)
     - ~10 lines
     - `item04_math_exp.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item04_math_exp.py>`__
   * - :ref:`5 <limitation-5>`
     - REPRODUCES (extended by E)
     - frontend
     - Kernel contexts share the region's scope list (``infer.py:781``); the
       annotated-assign check at ``infer.py:682-690`` searches all scopes
       (``visitor.py:244``).
     - forces the ``l_imem`` / ``lA`` / ``lB`` / ``lC`` naming at
       ``microarch_isa.py:495,630,1095``
     - ~15 lines
     - `item05_region_param_shadowing.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item05_region_param_shadowing.py>`__
   * - :ref:`8 <limitation-8>`
     - REPRODUCES -- but **loudly**: a symbol-redefinition error, not a silent
       break
     - frontend
     - ``builder.py:2828-2894`` rebuilds the sub-region per call under the same
       symbol names (``redefinition of symbol named 'feed_0__0_fixed'``).
     - none (one region, no sub-region calls)
     - 8 lines (prototyped; simulator-verified, HLS untested)
     - `item08_subregion_two_callsites.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item08_subregion_two_callsites.py>`__
   * - :ref:`10 <limitation-10>`
     - (a) REPRODUCES; (b) CANNOT-REPRODUCE
     - simulator, HLS driver, frontend
     - (a) ``simulator.py:1641`` re-parses ``str(mod)``, dropping locations (also
       ``hls.py:254``, ``llvm.py:51,60``); ``ASTContext.copy()``
       (``visitor.py:126``) drops ``file_name``; ``customize.py:1352`` records
       ``allo/dataflow.py`` as the source file. (b) three regions build and run
       in one process.
     - diagnosis time only
     - 3-line prototype yields ``file.py:12:19``
     - `item10_error_source_location.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item10_error_source_location.py>`__
   * - :ref:`14 <limitation-14>`
     - REPRODUCES, plus a false positive
     - emitter
     - ``EmitVivadoHLS.cpp:3019-3044`` rejects any nested function with a 2-D
       argument -- including a 2-D *local* that never touches a top-level
       pointer.
     - forces flat ``A`` / ``B`` / ``C`` addressing at
       ``microarch_isa.py:402-409,454-457``
     - not sized
     - `item14_wrapio_false_multidim.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item14_wrapio_false_multidim.py>`__
   * - :ref:`15 <limitation-15>`
     - REPRODUCES -- via ``df.build(mode="csim")`` it **hangs silently**
       instead of printing the documented error
     - frontend
     - Process calls are emitted in ``node.body`` order
       (``builder.py:2234-2290``).
     - forces ``dma_st`` to be declared last (``microarch_isa.py:1098-1104``)
     - ~30 lines (topological sort)
     - `item15_csim_declaration_order.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item15_csim_declaration_order.py>`__
   * - :ref:`16 <limitation-16>`
     - REPRODUCES
     - HLS driver
     - ``hls.py:331-338`` rejects ``mode="cosim"``; ``vitis.py:410`` emits
       ``m_axi`` with no ``depth=``.
     - ``tinytpu/cosim.py`` (216 lines, ``patch_axi_depths`` at ``:109``)
       exists because of it
     - ~150 lines
     - `item16_cosim_not_wired.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item16_cosim_not_wired.py>`__
   * - :ref:`17 <limitation-17>`
     - (a), (c) REPRODUCE; (b) CANNOT-REPRODUCE; (d), (e) are capabilities and
       work
     - frontend
     - (a) ``infer.py:1051``, message at ``visitor.py:458``. (c) the ``meta_if``
       body gets its own scope (``builder.py:3905``, ``infer.py:1599``).
     - (c) forces declare-before-assign at
       ``microarch_isa.py:879,891,903,916,924,938``
     - (c) 4-line prototype passes 12 dataflow tests (the AIE non-unroll path
       must keep the scope)
     - `item17_frontend_constraints.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item17_frontend_constraints.py>`__
   * - :ref:`18 <limitation-18>`
     - REPRODUCES (fork-only code)
     - emitter
     - ``EmitCatapultHLS.cpp:321-389`` emits blocking ``read`` / ``write`` with
       ``success = true``; introduced by fork-only ``73cbba0c``.
     - none (Vitis target)
     - ~10 lines to reject instead
     - `item18_catapult_try_ops_blocking.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item18_catapult_try_ops_blocking.py>`__
   * - :ref:`19 <limitation-19>`
     - REPRODUCES
     - emitter, HLS driver
     - Only construct/get/put are dispatched (``EmitTapaHLS.cpp:425-429``); the
       base methods are empty (``EmitBaseHLS.h:85-88``); the message at
       ``hls.py:311-314`` blames ``wrap_io``.
     - none (Vitis target)
     - not sized
     - `item19_tapa_try_ops_message.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item19_tapa_try_ops_message.py>`__
   * - :ref:`22 <limitation-22>`
     - REPRODUCES
     - SystemC fork
     - ``emitWireGet`` / ``emitWirePut`` (fork ``EmitSystemC.cpp:1828-1848``) are
       bare ``sc_signal`` accesses; nothing ties the reader's loop to the
       writer's.
     - none (Vitis target); blocks a MiniTPU-class delay line on the SystemC
       path
     - the ``SC_METHOD`` comb mode scoped in ``c7402f9f`` (five phases)
     - ``tests/systemc/rtlsim/REPRO.sh``
   * - :ref:`23 <limitation-23>`
     - REPRODUCES (on probes)
     - HLS driver, emitter
     - ``m_axi`` pragma is a regex rewrite of emitted text
       (``vitis.py:381,410``); ``emitFunctionDirectives`` interface body is
       dead-commented.
     - none on the shipped design: the opt-in ``align_value`` widens gmem0 to
       512 bits (``b4be2b10``)
     - see item text
     - probes (not committed)
   * - :ref:`shared memory <limitation-shared-memory>`
     - REPRODUCES (Allo's rule, not Vitis's)
     - emitter; SystemC fork
     - Allo refuses a region-scope Stateful shared by two kernels
       (``EmitVivadoHLS.cpp:3083-3122``) and never emits ``#pragma HLS stream
       type=unsync``. SystemC fork: memory instances keyed per (call, operand)
       (fork ``EmitSystemC.cpp:2539-2545``), so each client gets a replica.
     - **0 cycles, measured**: every restructure the design needed was
       Allo-legal
     - SystemC fork: ~50-100 lines (bind one writer and one reader to one
       instance's two pin sets)
     - `impact/probe_shared/ <https://github.com/sunwookim028/allo/tree/main/examples/tinytpu/impact/probe_shared>`__
   * - :ref:`A <limitation-a>`
     - FIXED (``6675130a``)
     - simulator
     - The simulator pipeline has no math-to-LLVM pass
       (``simulator.py:1674-1686``), so ``allo.exp`` / ``allo.log`` fail there.
       This also breaks item 4's documented workaround.
     - none (no transcendental ops)
     - 1 line (verified)
     - `new_sim_math_lowering.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/new_sim_math_lowering.py>`__
   * - :ref:`B <limitation-b>`
     - REPRODUCES
     - frontend
     - A scalar ``Stream.get()`` result is never cast to the destination type
       (``builder.py:1054``).
     - not assessed
     - 3 lines (verified)
     - `new_stream_get_no_cast.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/new_stream_get_no_cast.py>`__
   * - :ref:`C <limitation-c>`
     - REPRODUCES
     - frontend
     - ``customize()`` calls ``sys.exit(1)`` on frontend errors
       (``customize.py:1382,1408``); uncatchable by ``except Exception``.
     - harness only: a sweep or test that expects to catch a build failure is
       terminated
     - not sized
     - `new_customize_sys_exit.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/new_customize_sys_exit.py>`__
   * - :ref:`D <limitation-d>`
     - REPRODUCES (fork-only code)
     - emitter
     - Catapult ``full()`` always returns ``false``
       (``EmitCatapultHLS.cpp:402-414``); same class as item 18.
     - none (Vitis target)
     - not sized
     - reported by `item18_catapult_try_ops_blocking.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item18_catapult_try_ops_blocking.py>`__
   * - :ref:`E <limitation-e>`
     - REPRODUCES
     - frontend
     - Item 5 extended: same-name, **same-type** shadowing of a region
       parameter gives an MLIR region-isolation error. Same root cause as
       item 5.
     - as item 5
     - with item 5
     - variant (b) of `item05_region_param_shadowing.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item05_region_param_shadowing.py>`__
   * - :ref:`F <limitation-f>`
     - REPRODUCES
     - frontend (AIE)
     - The AIE ``cpp-style`` typing rules (``typing_rule.py:855-860``) have no
       bitwise ops, so every bitwise op is rejected there.
     - none (Vitis target)
     - not sized
     - informational line of `item07_bitwise_and.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/item07_bitwise_and.py>`__
   * - :ref:`G <limitation-g>`
     - REPRODUCES
     - frontend, HLS driver
     - ``= 0`` on an array lowers to a runtime zero-fill loop
       (``builder.py:1063-1075`` -> ``linalg.fill`` -> ``hls.py:288``
       ``convert-linalg-to-affine-loops``).
     - **+409 / +361 cycles** at 4x4x4 / 16x16x16 when restored on ``spad``
       (measured, ``v_memset``); the shipped design avoids it
     - not sized (warn; or reset-time init; or elide -- see item)
     - ``v_memset`` in `impact/ <https://github.com/sunwookim028/allo/tree/main/examples/tinytpu/impact>`__
   * - :ref:`H <limitation-h>`
     - REPRODUCES
     - frontend
     - A called sub-region is type-checked with the *caller's* globals:
       ``ASTContext(global_vars=ctx.global_vars.copy(), ...)`` at
       ``builder.py:2838`` (re-parsed via ``inspect.getsource`` at ``:2825``).
       A ``Stream[T, d][N]`` whose ``N`` (or ``Stream`` itself) is defined only
       in the sub-region's module fails ``infer.py:90`` ("stream array shape
       should be a compile time constant") or "Unsupported type ``Stream``".
       Fork issue #4.
     - none on TinyTPU-isa (one region); blocks composing regions across
       modules. Workaround: import the sub-region's globals into the caller
     - ~5 lines (merge the callee's own module globals over the caller's);
       not prototyped
     - `new_subregion_foreign_globals.py <https://github.com/sunwookim028/allo/blob/main/tests/limits/new_subregion_foreign_globals.py>`__

   * - :ref:`I <limitation-i>`
     - PARTLY FIXED (fork-only) -- bf16 emits and is bit-exact on Vitis; three
       residual divergences below
     - emitter
     - ``bf16`` had no branch in any ``getTypeName``; every emitter fell through
       to ``assert(1 == 0 && "Got unsupported type.")`` -- a SIGABRT, not an
       exception. Vitis HLS 2023.2 has no bf16 type to point at, so the emitter
       now ships an ``allo_bfloat16`` shim; Catapult and SystemC use
       ``ac::bfloat16``.
     - unblocks the BF16 MiniTPU model, which could not be emitted at all.
       On Vitis a bf16 MAC still costs a **full fp32** multiplier (3 DSP) and
       adder -- the win is storage and bandwidth, not arithmetic
     - landed (emitters + ``infer.py`` bitcast); Catapult synthesis untested
       (no licence on this host)
     - ``tests/dataflow/test_bf16_hls.py``,
       ``tests/test_vhls.py::test_unsupported_type_is_a_diagnostic_not_an_abort``

.. _limitations-closed:

Fixed or closed
~~~~~~~~~~~~~~~

Closed items are tracked as closed GitHub issues, not as rows in this file:
the issue body carries the status, root cause, fix commit/PR and repro that
used to live here. This keeps live state in git/GitHub rather than in a
checked-in snapshot.

.. list-table::
   :header-rows: 1
   :widths: 10 60 15

   * - Item
     - What it was
     - Issue
   * - :ref:`1 <limitation-1>` (+ :ref:`6 <limitation-6>`)
     - Region-scope ``@ Stateful`` lowering incomplete
     - `#36 <https://github.com/sunwookim028/allo/issues/36>`__
   * - :ref:`2 <limitation-2>`
     - ``@ Stateful`` could not be declared inside ``@df.kernel`` bodies
     - `#37 <https://github.com/sunwookim028/allo/issues/37>`__
   * - :ref:`3 <limitation-3>`
     - Simulator dropped nested-call stream lowering
     - `#38 <https://github.com/sunwookim028/allo/issues/38>`__
   * - :ref:`7 <limitation-7>`
     - No bitwise ``&`` operator support
     - `#39 <https://github.com/sunwookim028/allo/issues/39>`__
   * - :ref:`9 <limitation-9>`
     - Sim cache invalidation misses imported helpers
     - `#40 <https://github.com/sunwookim028/allo/issues/40>`__
   * - :ref:`11 <limitation-11>`
     - Simulator deadlocked when processes outnumbered OMP threads
     - `#41 <https://github.com/sunwookim028/allo/issues/41>`__
   * - :ref:`12 <limitation-12>`
     - Bit-slices lowered to signed ``ap_int<N>``, silently
     - `#42 <https://github.com/sunwookim028/allo/issues/42>`__
   * - :ref:`20 <limitation-20>`
     - Emitter could generate a local colliding with a parameter name
     - `#43 <https://github.com/sunwookim028/allo/issues/43>`__
   * - :ref:`21 <limitation-21>`
     - No ``#pragma HLS dependence`` primitive
     - `#10 <https://github.com/sunwookim028/allo/issues/10>`__ (predates this
       migration; not reused for anything else). Its section below is kept in
       full rather than trimmed: the legality rule that landed after the fix
       (``allo/dependence.py``) and the ``s.split`` defect it found are live,
       not closed.

   * - :ref:`K <limitation-k>`
     - ``@ Stateful(reset=False)`` refused by SystemC in any kernel that is not
       Wire-only (U4 track C, C2)
     - none yet (closed on ``core-fixes-4``; draft upstream text in
       ``dev/records/limitations/core_fixes_4_2026-10-08.rst``)

   * - :ref:`L <limitation-l>`
     - A D-12 read port of latency ``L`` delivered at ``t + L - 1`` on Stream
       links, ``t + L`` on Wire links (U4 track A T-2; track C C8)
     - none yet (closed on ``core-fixes-4``)

   * - :ref:`M <limitation-m>`
     - No stream could be flushed (U4 track A T-4/H6; T-5 the epoch
       workaround's failure)
     - none yet (closed on ``core-fixes-4``: README D-25)

:ref:`Item 13 <limitation-13>` is **not** in this table: it was largely
retracted, but a real convenience gap (no ``allo.dma`` intrinsic) remains
under the same item number, so its status is not unambiguous enough to close
as a GitHub issue outright -- see its own section below.

Two sub-items sit inside rows of the open table: 10(b) and 17(b) are
CANNOT-REPRODUCE.


.. _limitations-index:

Item index
----------

One entry per item, so that links from other pages land here. Each names its
issue; the archived register holds the analysis.

.. _limitation-1:

1. Region-scope ``@ Stateful`` lowering was incomplete (+ item 6) -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#36 <https://github.com/sunwookim028/allo/issues/36>`__. Analysis: archived register, under this heading.

.. _limitation-2:

2. ``@ Stateful`` could not be declared inside ``@df.kernel`` bodies -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#37 <https://github.com/sunwookim028/allo/issues/37>`__. Analysis: archived register, under this heading.

.. _limitation-3:

3. Simulator dropped nested-call stream lowering -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#38 <https://github.com/sunwookim028/allo/issues/38>`__. Analysis: archived register, under this heading.

.. _limitation-4:

4. ``math.exp`` / ``math.log`` are not recognized by the AST builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#15 <https://github.com/sunwookim028/allo/issues/15>`__. Analysis: archived register, under this heading.

.. _limitation-5:

5. Variable shadowing between region params and kernel-local names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#16 <https://github.com/sunwookim028/allo/issues/16>`__. Analysis: archived register, under this heading.

.. _limitation-6:

6. Local ``int32`` decls inside ``elif`` branches don't dominate uses -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Merged into item 1 (wrong diagnosis; see the corrections in the archive). Analysis: archived register, under this heading.

.. _limitation-7:

7. No bitwise ``&`` operator support in Allo expression DSL -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#39 <https://github.com/sunwookim028/allo/issues/39>`__. Analysis: archived register, under this heading.

.. _limitation-8:

8. Single-MXU-call rule (Allo region instantiation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#17 <https://github.com/sunwookim028/allo/issues/17>`__. Analysis: archived register, under this heading.

.. _limitation-9:

9. Sim cache invalidation misses imported helpers -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#40 <https://github.com/sunwookim028/allo/issues/40>`__. Analysis: archived register, under this heading.

.. _limitation-10:

10. Error messages point at lowered MLIR, not source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#18 <https://github.com/sunwookim028/allo/issues/18>`__. Analysis: archived register, under this heading.

.. _limitation-11:

11. The dataflow simulator deadlocked when processes outnumbered OMP threads -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#41 <https://github.com/sunwookim028/allo/issues/41>`__. Analysis: archived register, under this heading.

.. _limitation-12:

12. Bit-slices lowered to *signed* ``ap_int<N>``, silently, and the simulator disagreed -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#42 <https://github.com/sunwookim028/allo/issues/42>`__. Analysis: archived register, under this heading.

.. _limitation-13:

13. "No program-controlled DMA" (struck) -- **largely RETRACTED**; the gap is convenience, not capability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No issue yet. Analysis: archived register, under this heading.

.. _limitation-14:

14. ``wrap_io=False`` rejects multi-dimensional arguments to nested kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#19 <https://github.com/sunwookim028/allo/issues/19>`__. Analysis: archived register, under this heading.

.. _limitation-15:

15. Vitis ``csim`` executes dataflow processes in declaration order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#20 <https://github.com/sunwookim028/allo/issues/20>`__. Analysis: archived register, under this heading.

.. _limitation-16:

16. ``cosim`` is not wired into ``df.build``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#21 <https://github.com/sunwookim028/allo/issues/21>`__. Analysis: archived register, under this heading.

.. _limitation-17:

17. Frontend constraints worth documenting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#22 <https://github.com/sunwookim028/allo/issues/22>`__. Analysis: archived register, under this heading.

.. _limitation-18:

18. Catapult lowers ``try_get``/``try_put`` to *blocking* reads with ``success`` hard-coded true
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#23 <https://github.com/sunwookim028/allo/issues/23>`__. Analysis: archived register, under this heading.

.. _limitation-19:

19. ``try_get``/``try_put`` on the TAPA target fail to emit, with a generic message
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#24 <https://github.com/sunwookim028/allo/issues/24>`__. Analysis: archived register, under this heading.

.. _limitation-unused-capability:

A failure mode worth naming: an unused capability measures as a worthless one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A methodology note from the second pass, not a defect. Analysis: archived register, under this heading.

.. _limitation-disproved-theories:

Theories tested and disproved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A methodology note from the second pass, not a defect. Analysis: archived register, under this heading.

.. _limitation-20:

20. The emitter could generate a local whose name collided with a parameter -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#43 <https://github.com/sunwookim028/allo/issues/43>`__. Analysis: archived register, under this heading.

.. _limitation-21:

21. No ``#pragma HLS dependence`` primitive, so a false dependence cannot be asserted away
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No issue yet. Analysis: archived register, under this heading.

.. _limitation-22:

22. The SystemC fork's ``Wire`` is semantically incomplete -- and wrong in RTL, not just in simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#25 <https://github.com/sunwookim028/allo/issues/25>`__. Analysis: archived register, under this heading.

.. _limitation-23:

23. ``m_axi`` port widening is not reachable from user code -- ``align_value`` is necessary and **not** sufficient (a NEGATIVE result)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#26 <https://github.com/sunwookim028/allo/issues/26>`__. Analysis: archived register, under this heading.

.. _limitation-24:

24. Five checks pass a TinyTPU-isa program whose Vitis cosim never completes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No issue yet. Analysis: archived register, under this heading.

.. _limitation-24-qd:

The diagnosis: it is a channel-depth threshold, and ``TPU_QD=16`` clears it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Part of item 24. Analysis: archived register, under this heading.

.. _limitation-24-price:

What ``QD=16`` costs the shipped design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Part of item 24. Analysis: archived register, under this heading.

.. _limitation-25:

25. ``bfloat16`` runs in the simulator and aborts the process in every HLS emitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Superseded by item I: the Vitis, Catapult and SystemC emitters now accept ``bf16``. No issue yet. Analysis: archived register, under this heading.

.. _limitation-shared-memory:

Shared on-chip memory: the one-owner rule is Allo's, not Vitis's
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#27 <https://github.com/sunwookim028/allo/issues/27>`__. Analysis: archived register, under this heading.

.. _limitation-a:

A. The dataflow simulator cannot run ``allo.exp`` / ``allo.log``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#28 <https://github.com/sunwookim028/allo/issues/28>`__. Analysis: archived register, under this heading.

.. _limitation-b:

B. ``Stream.get()`` is never cast to the destination type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#29 <https://github.com/sunwookim028/allo/issues/29>`__. Analysis: archived register, under this heading.

.. _limitation-c:

C. ``customize()`` exits the interpreter on a frontend error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#30 <https://github.com/sunwookim028/allo/issues/30>`__. Analysis: archived register, under this heading.

.. _limitation-d:

D. Catapult ``full()`` always returns ``false``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#31 <https://github.com/sunwookim028/allo/issues/31>`__. Analysis: archived register, under this heading.

.. _limitation-e:

E. Item 5 extended: same-type shadowing of a region parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#32 <https://github.com/sunwookim028/allo/issues/32>`__. Analysis: archived register, under this heading.

.. _limitation-f:

F. The AIE ``cpp-style`` typing rules reject every bitwise op
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#33 <https://github.com/sunwookim028/allo/issues/33>`__. Analysis: archived register, under this heading.

.. _limitation-g:

G. ``= 0`` on an array is a runtime zero-fill loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#34 <https://github.com/sunwookim028/allo/issues/34>`__. Analysis: archived register, under this heading.

.. _limitation-h:

H. A sub-region from another module is type-checked against the caller's globals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: `#35 <https://github.com/sunwookim028/allo/issues/35>`__. Analysis: archived register, under this heading.

.. _limitation-i:

I. ``bf16`` ran in the simulator and aborted every HLS emitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No issue yet. Analysis: archived register, under this heading.


.. _limitation-j:

J. A ``Stream`` element wider than 128 bits corrupts the simulator's heap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No issue yet. The data arrive intact; the process then aborts, segfaults or
hangs in ``malloc`` (0 of 13 runs at 128 bits, 6 of 14 at 129-160). Repro
``tests/limits/new_sim_wide_stream_heap.py``; analysis and workaround (split
the token into streams of at most 128 bits):
``dev/records/minitpu/u4_track_a_2026-10-08.rst`` (T-1).

.. _limitation-k:

K. ``@ Stateful(reset=False)`` was refused by SystemC in any kernel that is not Wire-only -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Found by U4 track C (C2): the DMA's landing payload (``dma`` ``bits``,
``streams``; ``dma_vmem`` ``d12``) had no csim or Catapult path in its
declared form. Closed by README D-14's extended lowering: a kernel with a
``Stream``, channel or array port holds unreset storage as a plain module
member its thread writes with no reset action, and ``run.tcl`` scopes
``-RESET_CLEARS_ALL_REGS no`` to that thread (:doc:`/backends/systemc`).
Still refused, naming the storage: a ``Wire[T, comb]`` output reading it in
such a kernel (CIN-233). Tests: ``tests/dataflow/test_systemc_unreset.py``
(``test_unreset_stream_kernel_*``); record
``dev/records/limitations/core_fixes_4_2026-10-08.rst``.


.. _limitation-l:

L. A D-12 read port's latency depended on the link kind -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Found by U4 track A (T-2: ``fetch_d12`` kept the IRAM's read register in its
body) and track C (C8: ``dma_vmem``'s group held each read token one
iteration). The ``registers`` server put its pipe's last stage after the
shift, which a registered Wire link turns into ``L`` iterations and a Stream
link into ``L - 1``. Closed by README D-12 amended: ``L`` is counted in the
owner's iterations on every link kind (:doc:`/developer/dataflow_semantics`,
"Memory ports"). The compensating registers are gone from
``examples/minitpu/units/fetch_d12.py`` and ``dma_unit.py``. Test
``tests/dataflow/test_compose_port_latency.py``; record
``dev/records/limitations/core_fixes_4_2026-10-08.rst``.

.. _limitation-m:

M. No stream could be flushed -- closed (README D-25)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Found by U4 track A (T-4/H6): a sequencer's fetch queue is flushed on every
taken branch; the Allo forms without a flush either drained several tokens in
one iteration (not buildable) or dropped stale tokens by count or a 1-bit
epoch, which lost a bundle under a full queue and, with the tag alone, issued
stale bundles after two close flushes and hung the long trace (T-5). Closed by
``Stream[T, D, flush]`` and ``s.flush()`` (:doc:`/developer/stream_ports`,
:doc:`/developer/dataflow_semantics`). Not built: D-25's epoch for a
self-timed producer (a follow-up). Tests ``tests/dataflow/test_stream_flush.py``;
record ``dev/records/limitations/core_fixes_4_2026-10-08.rst``.
