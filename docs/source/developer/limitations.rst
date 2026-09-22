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

This page is the fork's register of Allo limitations: concrete obstacles met
while building accelerators on Allo, each of which required either a workaround
in user code or a patch to the Allo library. It was consolidated from two
passes:

* the first pass (items 1-10), from adding a VPU, a hardware loop and an
  on-device transpose to an L2 TPU running FlashAttention
  (``levels/L2/tpu.py``);
* the second pass (items 11-23), from building the instruction-programmable
  TinyTPU-isa (:doc:`/designs/tinytpu_isa`) on ``main`` and taking it through
  the Vitis dataflow path to RTL co-simulation.

Each entry keeps its dated corrections and retractions in place rather than
rewriting them away.

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
     - ``tinytpu_vitis/cosim.py`` (216 lines, ``patch_axi_depths`` at ``:109``)
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
     - ``examples/systemc_rtlsim/REPRO.sh``
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
     - `impact/probe_shared/ <https://github.com/sunwookim028/allo/tree/main/examples/accelerator/tinytpu_vitis/impact/probe_shared>`__
   * - :ref:`A <limitation-a>`
     - REPRODUCES
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
     - ``v_memset`` in `impact/ <https://github.com/sunwookim028/allo/tree/main/examples/accelerator/tinytpu_vitis/impact>`__
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
       migration; not reused for anything else)

:ref:`Item 13 <limitation-13>` is **not** in this table: it was largely
retracted, but a real convenience gap (no ``allo.dma`` intrinsic) remains
under the same item number, so its status is not unambiguous enough to close
as a GitHub issue outright -- see its own section below.

Two sub-items sit inside rows of the open table: 10(b) and 17(b) are
CANNOT-REPRODUCE.


.. _limitations-corrections:

Corrections from the 2026-09-19 re-verification and impact analysis
-------------------------------------------------------------------

These replace text that earlier revisions of this page (and of the notes it
came from) stated. The per-item sections below have been corrected in place.

* **Item 6's diagnosis was wrong.** Plain locals declared inside ``elif`` arms
  always worked (repro variants (a) and (b) pass; (c), reading a name declared
  in a *different* arm, is rejected by the frontend). The failure was item 1's
  cause and was fixed with it by ``5c4d1b53``; item 6 is merged into item 1.
* **Item 8 does not break silently.** Two callsites now fail loudly with a
  symbol-redefinition error at build time.
* **Item 7 cannot be reproduced**; bitwise ops have worked since ``12f898d7``
  (2023). Only the AIE ``cpp-style`` rules lack them (item F).
* **Item 9 is not an Allo limitation**: the cache belonged to another
  project's Makefile.
* **Item 14 has a false positive**: a local 2-D array passed to a nested
  function is rejected too.
* **Item 15 hangs silently** via ``df.build(target="vitis_hls", mode="csim")``
  rather than printing ``an hls::stream is read while empty``.
* **The one-owner-per-array rule is Allo's, not Vitis's**
  (:ref:`limitation-shared-memory`). Earlier text, here and on the TinyTPU-isa
  pages, attributed it to Vitis (``HLS 200-779`` / ``200-979``). A Vitis probe
  shows ``#pragma HLS stream variable=buf type=unsync`` shares an on-chip array
  between two processes, one per BRAM port (``HLS 200-824`` / ``200-755`` /
  ``200-634``); ``200-779`` applies only to *synchronized* arrays. Vitis does
  separately forbid one ``m_axi`` bundle read by two processes (``HLS 200-1013``
  / ``200-984``). The earlier claim that removing ``vru``'s double handling
  needs a second producer on a shared memory **was wrong**: measured impact on
  the design is 0 cycles, because every restructure it needed was Allo-legal.
* **Item 21 is priced** (it was "the 2.3% itself, forgone"): 35 cycles on the
  shipped design at 16x16x16, 95 once the design fixes are in; the pragma form
  costs 1,744 FF in ``accu`` against 17,438 for the reverted rotation; and it
  can be injected today by patching ``kernel.cpp`` between
  ``s.build(mode="csyn")`` and running Vitis. See
  :doc:`/designs/gemmini_comparison` for the attribution.
* **Item 22's root cause is located**: ``emitWireGet`` / ``emitWirePut`` at
  fork ``EmitSystemC.cpp:1828-1848`` are bare ``sc_signal`` accesses with
  nothing tying the reader's loop to the writer's. The fix is the
  ``SC_METHOD`` comb mode scoped in ``c7402f9f``.
* **AlloMemPins does not give two clients one memory**: memory instances are
  keyed per (call, operand) at fork ``EmitSystemC.cpp:2539-2545``, so each
  client gets a replica. The fix is ~50-100 lines binding one writer and one
  reader to one instance's two pin sets. :doc:`/extensions/catapult_systemc`
  is corrected accordingly.


Surfaced while building the L2 TPU (FlashAttention)
---------------------------------------------------


Notes from the Track B step 2 effort to add VPU + hardware loop + on-device
transpose to ``levels/L2/tpu.py``. Each item is a concrete obstacle that
required either a workaround in user code or a patch to the Allo library.

Priority annotations below were folded in from the former root ``STATE.md``
(now removed; project state is judged from git/GitHub, and the living
fork-vs-upstream feature map is the pinned fork issue
https://github.com/sunwookim028/allo/issues/13). Related feature-gap tracking
lives as fork issues and is
not restated here: combinational wires (fork issue #9), HLS dependence pragma
(fork issue #10), shared mutable memory across kernels (fork issue #11; relates
to items 1-2 below), streams as top-level inputs (fork issue #12), and the
nested sub-region Stream compile-time-constant shape constraint (fork issue #4;
item :ref:`H <limitation-h>`, not item 3 as an earlier revision said).

.. _limitation-1:

1. Region-scope ``@ Stateful`` lowering was incomplete (+ item 6) -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIXED by ``5c4d1b53`` (fork-only; upstreaming candidate). Closed as
`issue #36 <https://github.com/sunwookim028/allo/issues/36>`__, which carries
the root cause, the fix and the repro.

.. _limitation-2:

2. ``@ Stateful`` could not be declared inside ``@df.kernel`` bodies -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIXED by upstream PR #577. Closed as
`issue #37 <https://github.com/sunwookim028/allo/issues/37>`__, which carries
the root cause and the repro.

.. _limitation-3:

3. Simulator dropped nested-call stream lowering -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIXED by ``5bc104c8`` (fork-only; upstreaming candidate). Closed as
`issue #38 <https://github.com/sunwookim028/allo/issues/38>`__, which carries
the root cause, the fix and the repro.

.. _limitation-4:

4. ``math.exp`` / ``math.log`` are not recognized by the AST builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES. Note that the workaround below, ``allo.exp``, does not run on the dataflow simulator: see :ref:`limitation-a`.

- Inside ``@df.kernel`` bodies, ``math.exp(x)`` raises ``KeyError: 'exp'``.
- Must use ``allo.exp(x)`` (and friends) instead.
- Not documented as a constraint; the failure mode (KeyError on a
  Python-builtin-ish name) does not point at the workaround.
- **Priority: Low** — ``allo.exp`` works.

.. _limitation-5:

5. Variable shadowing between region params and kernel-local names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES. Extended by :ref:`limitation-e` (same-type shadowing).

- Declaring a local ``d_addr: int32 = cmd[3]`` inside ``@df.kernel def compute_driver`` raises ``AssertionError: Invalid assignment to d_addr, type mismatch`` because the enclosing ``@df.region def tpu(..., d_addr: int32[1], ...)`` parameter leaks into the kernel
  scope. The compiler treats the local ``int32`` write as an attempted
  rebinding of the region parameter (a ``int32[1]``).
- Workaround: rename every kernel-local that happens to share a name
  with a region parameter (``d_addr → cmd_d``, etc.).
- The error message names the variable but not the shadowing, so this
  takes a while to diagnose.
- **Priority: Medium** — silent/misdirected error.

.. _limitation-6:

6. Local ``int32`` decls inside ``elif`` branches don't dominate uses -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Merged into :ref:`limitation-1` -- the diagnosis for this item was wrong:
plain locals in ``elif`` arms always worked, and the failure was item 1's
cause. See `issue #36 <https://github.com/sunwookim028/allo/issues/36>`__.

.. _limitation-7:

7. No bitwise ``&`` operator support in Allo expression DSL -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CANNOT-REPRODUCE: bitwise ops work in the default typing rules since
``12f898d7`` (2023). Only the AIE ``cpp-style`` rules lack them:
:ref:`limitation-f`. Closed as
`issue #39 <https://github.com/sunwookim028/allo/issues/39>`__.

.. _limitation-8:

8. Single-MXU-call rule (Allo region instantiation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES, but **loudly**: two callsites now fail at build time with a symbol-redefinition error (``redefinition of symbol named 'feed_0__0_fixed'``), not silently as stated below.

- ``mxu(...)`` cannot appear in two different ``if``/``elif`` branches even
  if they are mutually exclusive at runtime. Allo instantiates
  sub-regions at build time regardless of conditions, so two branch
  callsites become two independent instances. *(Corrected 2026-09-19: the
  build now fails loudly with a symbol-redefinition error; an earlier revision
  said the design silently breaks.)*
- Each level keeps ``mxu(...)`` in exactly one combined branch (``OP_MM | OP_MMT`` for L1; ``COMPUTE_PRELOADED | COMPUTE_ACCUMULATED`` for L2).
- This forces unnatural code structure — the natural reading is "if
  preloaded, do mxu with these args; if accumulated, do mxu and add" —
  but the compiler needs us to flatten them.
- **Priority: Medium** — forces unnatural code structure.

.. _limitation-9:

9. Sim cache invalidation misses imported helpers -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NOT-A-LIMITATION: the ``.cache/llvm_sim/`` cache belonged to another
project's Makefile; Allo keeps no simulator cache. Closed as
`issue #40 <https://github.com/sunwookim028/allo/issues/40>`__.

.. _limitation-10:

10. Error messages point at lowered MLIR, not source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   (a) REPRODUCES; (b) CANNOT-REPRODUCE -- three regions build and run in one process.

- Most failure modes surface as MLIR / LLVM errors at line numbers in a
  generated module that the user never sees. Examples:

  - ``loc("-":1892:3): error: cannot be converted to LLVM IR ...``
  - ``Assertion 'value' failed``
  - ``Failure while creating the ExecutionEngine``

- Mapping these back to the offending Python construct requires
  dumping ``s.module`` and counting lines — there is no source-position
  attribution back to the original ``tpu.py``.
- The MLIR Context cannot be re-instantiated in the same Python
  process without crashing
  (``LLVM ERROR: Option 'fast' already exists!``),
  so debugging via "build twice and diff" doesn't work.
- **Priority: Low.**

Surfaced while building an instruction-programmable TPU (2026-09)
-----------------------------------------------------------------

A second pass, from building ``examples/accelerator/tinytpu_vitis/`` on ``main``:
an int8 instruction-programmable tiled-GEMM accelerator taken through the
Vitis dataflow path to **RTL co-simulation**, and compared against a data-type-
and mesh-matched Gemmini (:doc:`/designs/gemmini_comparison`).

The findings below were previously scattered across ``examples/accelerator/*/``
markdown and **none of them had reached** ``notes/`` (the fork's former notes
directory, now this documentation), which is why they are consolidated here. They are ordered by what they would cost an Allo user, not
by when they were found.

.. _limitation-11:

11. The dataflow simulator deadlocked when processes outnumbered OMP threads -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIXED 2026-09-17 by ``f193c057`` (fork-only; upstreaming candidate, upstream
draft PR #611): the OpenMP team is now sized to the section count instead of
defaulting to the core count. A tier-0 diagnostic watchdog (``7bc6d413``) was
added at the same time; the tier-1 per-channel report remains unbuilt --
track as a fresh enhancement rather than here. Closed as
`issue #41 <https://github.com/sunwookim028/allo/issues/41>`__.

.. _limitation-12:

12. Bit-slices lowered to *signed* ``ap_int<N>``, silently, and the simulator disagreed -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIXED by upstream PR #612 together with fork ``3de74846``, merged into
``main`` in ``dc6b8fa6``. Closed as
`issue #42 <https://github.com/sunwookim028/allo/issues/42>`__.

.. _limitation-13:

13. "No program-controlled DMA" (struck) -- **largely RETRACTED**; the gap is convenience, not capability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   Retracted 2026-09-18 by the entry itself; not part of the 2026-09-19 pass.

``wrap_io=True`` copies each argument into a local buffer before the region runs,
sized to the **declared** array rather than to what the program touches.
``wrap_io=False`` drops the copy but every access then pays bus latency.

Measured, same design, one build each (cycles, cosim):

================= ============== ======= ========= ========
config            marginal       fixed   4x4x4     16x16x16
================= ============== ======= ========= ========
``wrap_io=True``  18.1 cyc/instr 1102    2.12x     1.94x
``wrap_io=False`` **39.8**       **481** **1.21x** 2.25x
Gemmini           **10.8**       **483** 1.00x     1.00x
================= ============== ======= ========= ========

The retraction (2026-09-18)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conclusion drawn from that table -- "Gemmini has a third option and Allo
does not expose it" -- **was wrong**, and the table itself is confounded.

``wrap_io=False`` was measured with a **strided** access pattern and no local
buffer. Probing which patterns Vitis actually bursts, with ``wrap_io=False`` and
flat arguments:

.. code-block:: text

   lA[(off + r) * MAXDIM + e], e over meta_for(T)      <- what our dma_ld does
     [HLS 214-115] Multiple burst reads of length 4 and bit width 8

   for i in range(n): buf[i] = lA[off + i]   (n RUNTIME)
     [HLS 214-115] Multiple burst reads of VARIABLE LENGTH and bit width 8

**A contiguous copy with a runtime length from a runtime offset infers a real
variable-length AXI burst, at II=1.** That *is* ``mvin``, and Allo expresses it
today. The 39.8 cycles/instruction in the table above is the cost of 4-byte
bursts, not of ``m_axi``; it condemns the access pattern, not the configuration.

What remains true, and what is actually left:

- ``wrap_io=True`` genuinely is not a DMA. ``wrap_data_movement``
  (``allo/ir/transform.py:450``) takes its extent from
  ``shape = MemRefType(arg.type).shape`` -- the **static type** -- with no offset
  and no length anywhere in the generated function. It is a whole-argument
  hoist run once at region entry, and every declared word is a startup cycle
  whether the program touches it or not.
- So the two options are "hoist everything" or "issue your own bursts", and the
  second one works. What Allo lacks is only the *convenience* of an
  ``allo.dma(buf, ptr, offset, length)`` intrinsic that makes the burst idiom
  obvious rather than something you discover by reading HLS burst messages.
- **Priority: Medium** (an ergonomics and documentation item), down from High.
  The performance work it was blocking is ours, not Allo's.

The general lesson
^^^^^^^^^^^^^^^^^^

The measurement that produced the wrong conclusion was real and repeatable; it
was the *attribution* that was wrong. Two configurations were compared while a
third variable -- the access pattern -- differed between them, and the result
was charged to the configuration. Before charging a cost to the toolchain,
check that the thing being measured is the thing named.

.. _limitation-14:

14. ``wrap_io=False`` rejects multi-dimensional arguments to nested kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES, with a false positive: a 2-D *local* array passed to a nested function is rejected too.

   Top-level multi-dimensional arrays are linearized to 1D pointers ... which
   cannot be passed to nested functions expecting multi-dimensional arrays

- The message is good and names the fix. Flat ``int8[M*N]`` arguments with manual
  ``row * stride + col`` addressing work, and are arguably the honest shape for
  DRAM anyway.
- **Priority: Low** (documentation), but it interacts with #13: taking the
  low-fixed-cost option forces flat arguments.

.. _limitation-15:

15. Vitis ``csim`` executes dataflow processes in declaration order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES. Through ``df.build(target="vitis_hls", mode="csim")`` the consumer-first order **hangs silently** instead of printing the error below.

A consumer declared before its producer reads an empty stream:

.. code-block:: text

   ERROR [HLS SIM]: an hls::stream is read while empty

- Kernel declaration order in a ``@df.region()`` is therefore **load-bearing** for
  ``csim`` (not for RTL, where processes are concurrent). Nothing documents this.
- Cost here: ``dma_st`` was declared third and consumed what ``accu``, declared
  last, produced. Reordering fixed it with no hardware change.
- **Priority: Medium.** A note in the dataflow docs would be enough.

.. _limitation-16:

16. ``cosim`` is not wired into ``df.build``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES.

``df.build(target="vitis_hls", mode=...)`` handles ``csim`` and ``csyn``; every other
mode routes to the ``XDEVICE`` Makefile flow, and the emitted ``host.cpp`` is an
OpenCL/XRT host, which is not what ``cosim_design`` wants.

- For an *instruction-programmable* design this matters more than it looks: the
  loop trip counts are data, so ``csynth`` can only report a worst-case bound
  derived from the ISA's field widths. Measured, same design, two builds:

  ============================================= ============= ========= =============
  build                                         csynth        cosim     ratio
  ============================================= ============= ========= =============
  16x16 array, 262 instances                    **2.259e+08** **1,176** **~192,000x**
  4x4 array, before narrowing a row-count field 91,407        4,133     22x
  ============================================= ============= ========= =============

  The five-order-of-magnitude case is the headline; the 22x is the one that was
  *fixed*, by giving the row count its own 7-bit field instead of a 12-bit one
  (no datapath change). Cosim is the only number comparable to a real
  accelerator's cycle count.

- **The general rule, which cost two projects time independently:** a model and
  a measurement that disagree are usually answering different questions, and the
  question the model is answering is often about the *design space* rather than
  the *program*. csynth was not broken either time -- it was correctly bounding
  the machine the encoding permitted. The mirror-image case is a model that
  predicts a *schedule* being read as a prediction about the emitter you
  actually shipped.

- ``examples/accelerator/tinytpu_vitis/cosim.py`` is a working driver: it
  generates a plain C++ testbench from the same program and reference the
  simulator uses, patches ``m_axi`` depths (cosim requires them; Allo emits none),
  and drives ``vitis_hls``. It is ~180 lines and could be folded into the backend.

- **Priority: Medium.**

.. _limitation-17:

17. Frontend constraints worth documenting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   (a) and (c) REPRODUCE; (b) CANNOT-REPRODUCE; (d) and (e) are capabilities and work.

Each cost real time; none is a bug exactly, but none is discoverable:

- **Stream-array subscripts must be compile-time.** A runtime index fails with
  "Fail to resolve the expression as symbolic expression" -- correct, but it
  does not mention streams. Use ``allo.meta_for``.
- **Nested** ``meta_for`` **over a 2-D stream array fails** where a single ``meta_for``
  over a 1-D one is fine. Forces flat ``[T*T]`` stream arrays indexed ``i*T + j``.
- **Names bound inside** ``meta_if`` **are not visible after it** ("Unsupported Name
  ``a``"). Declare before, assign inside.
- **Runtime loop bounds do work** in a ``df.kernel``, including with stream ops
  in the body. This is the capability that makes a workload-independent design
  possible at all -- one build, shape as data -- and it is undocumented.
- ``df.build`` **is** ``customize(func)`` **+** ``s.build(...)``, so the schedule
  primitives (``s.partition``, ``MockBuffer``) are reachable on the Vitis path.
  Also undocumented, and load-bearing: partitioning the feeders and the
  accumulator took the top-level interval from 168 to 74 cycles, measured on
  the since-removed weight-stationary design
  (``git show e2451b81:examples/accelerator/tinytpu_vitis/RESULTS_WS.md``, section 3).
- **Priority: Low individually, Medium as a "dataflow gotchas" page.**

.. _limitation-18:

18. Catapult lowers ``try_get``/``try_put`` to *blocking* reads with ``success`` hard-coded true
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES. ``full()`` has the same problem: :ref:`limitation-d`.

``mlir/lib/Translation/EmitCatapultHLS.cpp`` emits ``ch.read(v)`` for
``StreamTryGetOp`` and ``ch.write(v)`` for ``StreamTryPutOp``, then emits
``bool success = true;`` unconditionally. The comment says why: ``nb_read`` /
``nb_write`` inside a spin-while loop segfaults Catapult's go compile (LOOP-19),
and a blocking op "always succeeds → spin-while exits in 1 iteration →
bounded".

The workaround is defensible; what is not is that it is silent, and that the
comment understates it. "Scheduling semantics differ only at runtime" is true
for the ``while not S.try_put(x): pass`` idiom, which is what the bring-up
designs used. It is false for the reason non-blocking ops exist:

- Any design that **branches on failure** -- try this channel, else try the
  next; poll a control channel and do other work if empty; arbitrate between
  requesters -- has its else-branch turned into dead code, because ``success`` is
  a compile-time ``true``. The RTL is silently a different design from the one
  written: a decoupled, backpressured graph becomes lock-step blocking.
- It is not diagnosed. No warning, no ``#error``, nothing in the emitted C++
  marking the substitution. The first sign is a Catapult schedule that does not
  match the intended architecture, or a deadlock in a design the simulator runs
  fine.
- The three backends now disagree on the same frontend op, which is the deeper
  problem: **Vivado** emits honest ``.read_nb(`` / ``.write_nb(``
  (``EmitVivadoHLS.cpp``); **Catapult** emits blocking + ``true``; **TAPA** emits
  nothing and hard-fails the build (see #19 below, and
  ``tests/dataflow/test_stream_ops_hls.py::test_tapa_stream_nb``). A frontend
  primitive whose meaning changes per target is a correctness trap, not a
  portability inconvenience.
- Related Catapult deviation, same file: ``empty()`` is emitted as
  ``!ch.available(1)`` because ``ac_channel`` has no ``.empty()`` in the
  synthesizable subset (EDG CIN-59). That one is a faithful translation.
- Documented today only in :doc:`/extensions/catapult_systemc` (as a backend note,
  not as a correctness risk) and :doc:`/backends/nonblocking_streams`.
- **Priority: Medium** as it stands, **High** for anyone building arbitration
  on the Catapult path. The cheap fix is to refuse: raise on
  ``StreamTryGetOp``/``StreamTryPutOp`` for ``target="catapult"`` unless an explicit
  opt-in attribute says the blocking substitution is acceptable. Failing to
  build beats silently building the wrong circuit.

.. _limitation-19:

19. ``try_get``/``try_put`` on the TAPA target fail to emit, with a generic message
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES.

``EmitTapaHLS.cpp``'s visitor dispatches only ``StreamConstructOp`` /
``StreamGetOp`` / ``StreamPutOp``. The base hooks ``emitStreamTryGet``,
``emitStreamTryPut``, ``emitStreamEmpty``, ``emitStreamFull`` in
``mlir/include/allo/Translation/EmitBaseHLS.h`` are empty bodies and are never
reached: the op falls through to ``visitUnhandledOp``, and ``emitBlock`` reports
``"can't be correctly emitted"``, which surfaces as
``RuntimeError: Failed to emit HLS code. ... Common issues: nested functions with multi-dimensional arrays when wrap_io=False.``

Failing is the right call -- this is strictly better than #18. But the message
names a cause that has nothing to do with the actual one, so the user is sent
looking at ``wrap_io`` instead of at an unsupported op. TAPA has ``try_read`` /
``try_write``, so the gap is implementable, not fundamental.

- ``tests/dataflow/test_stream_ops_hls.py::test_tapa_stream_nb`` asserted
  ``.try_read(`` / ``.try_write(`` and so had been failing outright. Marked
  ``xfail(strict=True, raises=RuntimeError)`` with the mechanism named, so the
  gap is recorded and the test turns red the moment the codegen lands.
- History worth noting: ``3723e817`` deleted this test as dead, and merge
  ``cdac5e68`` resurrected it. A test can come back from the dead in a merge
  without anyone noticing it is red.
- **Priority: Low** for the codegen, **Medium** for the error message --
  ``emitError`` should name the op it could not emit.


.. _limitation-unused-capability:

A failure mode worth naming: an unused capability measures as a worthless one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two independent instances, one from this project and one from the MiniTPU
project, and the trap is sharper than "profile before optimising":

- MiniTPU split a controller FSM to remove a serialisation bound and measured
  **719 cycles before, 719 after** -- because their emitter issues eight pushes
  then eight pops, so the two halves never hold work at the same time.
- We scaled the array from 4x4 to 16x16 and measured **1.47x for 16x the PEs**
  -- because the design is fixed-cost bound and the data path cannot feed it.

In both cases the measurement is correct and the obvious reading of it is
wrong. "Splitting the controller does not help" and "a bigger array does not
pay" are what the numbers literally say, and both conclusions are false: the
capability was *unused*, not *worthless*. Nothing in the measurement
distinguishes those two, which is what makes it dangerous -- a null result
normally retires a hypothesis, and here it silently retires the wrong one.

Practical rule: before changing a mechanism, confirm the workload can present
the mechanism with work it could exploit. If it cannot, fix the schedule or the
feed first, or the experiment will tell you the mechanism is useless.

.. _limitation-disproved-theories:

Theories tested and disproved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recorded so nobody re-runs them. Each looked plausible and each cost a cycle of
investigation:

+--------------------------------+--------------------------------+--------------------------------+
| theory                         | test                           | result                         |
+================================+================================+================================+
| PE ``put`` order closes a      | swap the two puts              | minimum depth unchanged        |
| cycle through the drainer's    |                                |                                |
| read order                     |                                |                                |
+--------------------------------+--------------------------------+--------------------------------+
| a cycle in the process graph   | two repros, one with real      | **both ran** -- cyclic regions |
| deadlocks                      | traffic on every edge          | are fine                       |
+--------------------------------+--------------------------------+--------------------------------+
| the ``vld`` burst length is    | chunk it                       | still hung                     |
| the cause                      |                                |                                |
+--------------------------------+--------------------------------+--------------------------------+
| the sequencer's control        | rewrite as a forwarding chain  | exact, depth unchanged         |
| broadcast is the cause         |                                |                                |
+--------------------------------+--------------------------------+--------------------------------+

The actual cause was #11, in the simulator, not in any of these.

.. _limitation-20:

20. The emitter could generate a local whose name collided with a parameter -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIXED by ``aece11c9`` (fork-only; upstreaming candidate). Closed as
`issue #43 <https://github.com/sunwookim028/allo/issues/43>`__, which carries
the root cause, the fix and the repro.

.. _limitation-21:

21. No ``#pragma HLS dependence`` primitive, so a false dependence could not be asserted away -- closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIXED by ``bbea2af0`` (fork-only; upstreaming candidate): the ``s.dependence``
schedule primitive, next to ``s.partition`` in ``allo/customize.py``. Tracked
as the pre-existing `issue #10
<https://github.com/sunwookim028/allo/issues/10>`__ (not reused for anything
else by this migration). The price TinyTPU-isa paid for this claim on its
accumulator, and what the primitive does and does not check, are design
rationale rather than issue material and live in
:doc:`/designs/tinytpu_isa` ("The accumulator's dependence claim").

.. _limitation-22:

22. The SystemC fork's ``Wire`` is semantically incomplete -- and wrong in RTL, not just in simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES; root cause located (below).

   **Tested 2026-09-18: it is not the free-running rewrite; the conclusion
   below stands.** The hypothesis was that ``isSteadyStateLoop`` (which turns a
   kernel's outermost loop into ``while (1)`` under ``__SYNTHESIS__`` when its
   induction variable is unused) made ``acc`` run unthrottled, and that
   ``72c70dcb``'s guard missed it because ``acc`` has no memory port. The guard was
   extended to any loop body containing a ``WireGetOp`` (``guard_experiment/`` under
   ``examples/systemc_rtlsim/``), ``pe_split.py`` was re-emitted from
   ``choonsik1/SystemC-emitter`` with and without it, synthesised with Catapult
   2024.2, and simulated under Xcelium 24.03:

   - The emitted code changed exactly as intended. ``acc_0`` lost its
     free-running loop and its done-on-entry. ``mul_0`` (Stream reads) kept its
     free-running loop. ``pe_stream`` and ``pe_channel`` emitted **byte-identical**
     code.
   - ``pe_wire`` still fails **8/8 at all 18 pacings, with and without the guard,
     in identical cycle counts** (20 cycles at unit pacing; values
     ``0 8 40 112 240 368 496 624``: ``acc`` samples every other product and then
     holds the last one).
   - The current emitter's ``pe_stream`` and ``pe_channel`` pass all 36 pacings,
     and ``BREAK_DATA`` turns both red.

   Removing ``while (1)`` bounds ``acc``'s trip count but gives it nothing to wait
   on. Its loop still advances on its consumer's ``Push``, never on the
   producer. The missing piece is synchronisation, as stated below. The
   August-netlist matrix further down also reproduces case-for-case under
   Xcelium (``RUNNER=run_mulacc_xrun.sh ./REPRO.sh``).

Recorded here because it closes a question this project spent real time on: whether
the SystemC path (``choonsik1/allo:SystemC-emitter``) offers the **non-handshaked
fixed-latency edge** that a no-interlock machine needs and Allo's ``Stream`` cannot
express. It does not.

An earlier investigation reported ``pe_wire`` as *"synthesises clean and fails
csim"* and concluded the SystemC thread model could not represent a wire --
i.e. that the design was good and the simulator was lying. **That is backwards.**
Simulating Catapult's own ``pe_wire`` netlist under xsim:

.. code-block:: text

   Stream[int32,2]  boundary   PASS   (all 18 producer/consumer pacings)
   Channel[vld_rdy] boundary   PASS   (all 18)
   Wire[int32]      boundary   FAIL   8/8 wrong, at all 18

Measured ``0 0 0 2 2 10 10 28`` against golden ``2 10 28 60 110 182 280 408``. **The
csim failure was a true positive.**

Mechanism, from the netlist rather than the model: ``acc_0`` has no input
handshake at all -- its only input is bare data -- and its loop counter advances
on its *consumer's* ready. ``mul_0`` latches the wire on its *producers'* valid.
Nothing couples the two counters, ``mul`` takes ~3 cycles per product and ``acc``
one per step, so ``acc`` runs the entire loop before ``mul`` produces anything.

- **Positive control:** holding ``acc_0`` in reset 3-4 cycles longer and stepping
  it once per product makes the *identical* wire RTL produce the exact golden
  result. The wiring and arithmetic are right; only the lockstep is missing, and
  the correct window is **2 cycles wide** (delays 2 and 5 both fail).
- The construct names an edge without specifying when either end samples it. Its
  correctness depends on a global cycle discipline that nothing in the emitter
  establishes, checks, or documents. The one design previously cited as evidence
  that wires work was already flagged as luck; this makes luck the general case.
- **A wire design CAN be verified** -- xsim on the Catapult netlist against a
  numpy golden, with four injected faults all going red, including a one-cycle
  latency change. But it needs hand-built lockstep per design and is not
  checkable by the type system.
- **Root cause in the emitter (located 2026-09-19):** ``emitWireGet`` /
  ``emitWirePut`` at fork ``EmitSystemC.cpp:1828-1848`` are bare ``sc_signal``
  accesses, and nothing ties the reader's loop to the writer's. The fix is the
  ``SC_METHOD`` comb mode described next (``c7402f9f``).
- Making wires sound needs the emitter to give cycle-locked kernels a shared
  advance -- one enable driving every stage's counter, which is what a VLIW
  delay line is. That is scoped, in five phases, as an ``SC_METHOD`` "comb"
  emission mode, and its own top risk is whether such a model simulates as well
  as it synthesises. The scoping document is **not on** ``main`` -- it exists only
  in commit ``c7402f9f``, on ``choonsik1/SystemC-emitter``:
  ``git show c7402f9f:notes/archive/SYSTEMC_COMB_MODE.md``. Deliberately left
  there rather than restored into ``notes/archive/`` (now retired into this documentation): it plans work on a branch
  that is not in this checkout, and its repro anchors are ``/tmp`` paths that no
  longer exist, so the two sentences above are what survives of it.

**Consequence for the roadmap:** the SystemC path does not currently supply a
usable non-handshaked edge, so the MiniTPU-class direction is blocked further
back than "Allo won't hand out two memory ports". It needs a scheduling model
first.

Two incidental findings
^^^^^^^^^^^^^^^^^^^^^^^

- **A valid/ready protocol violation in the emitted memory port.** ``feed_0``
  asserts ``v0_req_vld`` from its reset state but books the acknowledgement two
  FSM states later, while ``AlloMem`` consumes the request one cycle after reset
  -- so the full ``pe_wire`` top deadlocks from reset at any reset length. It
  affects ``pe_stream`` and ``pe_channel`` equally, **including the design the
  branch reports as Xcelium-cosim bit-exact**, which suggests these netlists
  were never simulated standalone.
- **No SystemC library or MatchLib on this host** (``ace-01``): no ``libsystemc*``,
  ``systemc.h``, ``connections.h`` or ``mc_connections.h`` anywhere. csim cannot run
  here for any design. SystemC 2.3.x + NVlabs MatchLib would be a few hours and
  no licence.

.. _limitation-23:

23. ``m_axi`` port widening is not reachable from user code -- ``align_value`` is necessary and **not** sufficient (a NEGATIVE result)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Status (re-verified 2026-09-19)

   REPRODUCES on the probes; the real design widens with the opt-in ``align_value`` (``b4be2b10``).

The goal was one line of Tcl: ``config_interface -m_axi_max_widen_bitwidth 512``,
so that a long ``int8`` burst moves 64 bytes a beat instead of one and the operand
traffic becomes nearly free. Vitis refuses:

.. code-block:: text

   [HLS 214-307] Could not widen since type i8 size is greater than or equal to
   alignment 1(bytes)

**Scope, and it is the point of this item: every result below was measured on
PROBES** -- small standalone kernels written to provoke that message -- **not on
the real design** (``examples/accelerator/tinytpu_vitis/microarch_isa.py``).
Whether the real design reproduces ``HLS 214-307`` at all is an **open question**,
under separate investigation as of 2026-09-18. Three files already assert the
widening block as a whole-design fact --
``examples/accelerator/tinytpu_vitis/microarch_isa.py:247``,
``RESULTS_ISA.md:467``, ``COMPARISON.md:280`` -- and the probes do **not** establish
that. Nothing here upgrades them; a probe result is not a design result.

What the probes did settle:

- ``__attribute__((aligned(N)))`` **on the element type is the wrong lever.** It
  is the obvious first reach and it does not produce the alignment Vitis is
  testing: the test is on the pointer *parameter*, not on the type behind it.
- ``align_value`` **on the parameter is necessary.** It is what puts ``align 64`` on
  the argument in the IR; nothing else tried did.
- **And it is not sufficient.** No widening was observed in **six** probe
  combinations. "Add ``align_value`` and the port widens" is false, and retiring
  that is what this item is for -- it is worth six synthesis runs to whoever
  reads "port widening is blocked in Allo" and sets out to unblock it.
- **There is nowhere in the emitter to hang the attribute.** The ``m_axi`` pragma
  that actually ships is a **regex rewrite of already-emitted C++**:
  ``postprocess_hls_code`` (``allo/backend/vitis.py:381``) re-splits each parameter
  line of the top function's *text*, rewrites arrays to ``T *name``, and emits the
  interface line at ``allo/backend/vitis.py:410``. At that point there is no
  MemRef, no memory space and no port in scope -- only a string -- so there is
  nothing to derive an alignment from. The emitter's own interface path, which
  would have had all three, is **dead-commented**:
  ``VhlsModuleEmitter::emitFunctionDirectives``
  (``mlir/lib/Translation/EmitVivadoHLS.cpp:2777``) has its entire
  ``m_axi`` / ``s_axilite`` body commented out at **2779-2837**; the live code from
  2838 emits only ``dataflow``, ``inline``, and the per-array directives.

Pragma inventory (grep over ``mlir/lib``, ``mlir/include``, ``allo/``, 2026-09-18)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Live on the Vivado/Vitis path: ``stream`` (``EmitVivadoHLS.cpp:1775,1792,2621``),
``pipeline`` (2581), ``unroll`` (2595,2597), ``bind_storage`` (2696),
``array_partition`` (2719), ``dataflow`` (2838), ``inline`` (2843), and the ``m_axi``
line rewritten in ``vitis.py:410``. That is the whole list.

Not emitted by **any** emitter -- zero hits, not "hard to reach":

- ``ap_none`` / ``ap_stable`` / ``ap_fifo`` interface modes.
- ``#pragma HLS latency`` (the only ``latency`` hits in ``allo/`` are report
  *parsing*, ``allo/backend/report.py``, ``catapult.py:286``).
- ``#pragma HLS protocol``.
- ``#pragma HLS dependence`` -- see :ref:`item 21 <limitation-21>`, where the cost of its absence is
  measured; not restated here.
- ``#pragma HLS resource`` survives only as dead comment
  (``EmitVivadoHLS.cpp:2756``).

**Correction to :ref:`item 21 <limitation-21>`, in the open.** It states that the only pragmas Allo
generates are "the ``m_axi`` / ``s_axilite`` interface lines in
``allo/backend/vitis.py:410``". The ``m_axi`` half is right. ``s_axilite`` is **not**
emitted on the Vitis path at all: the only live ``s_axilite`` in the tree is
``allo/backend/pynq.py:173,176``, a different backend, and the emitter's copies
are inside the dead comment above. Item 21's argument is unaffected -- it is
strengthened, since the interface pragma set is one line narrower than claimed.

- **Priority: Low as an action, Medium as a warning.** Nothing here asks for
  work: the next honest step is the open real-design question, not more probes.
  If the real design does reproduce ``HLS 214-307``, the fix is in the emitter --
  restore ``emitFunctionDirectives`` so the interface pragma is derived from the
  MemRef instead of reconstructed from text -- and it becomes the same shape of
  gap as :ref:`item 21 <limitation-21>`: a one-line HLS assertion that Allo has no way to reach.

.. note::

   Update, recorded after this item was written (commit ``b4be2b10``,
   2026-09-18): the open real-design question has been answered on the real
   design. ``HLS 214-307`` does **not** reproduce there -- the widen setting is
   accepted, csynth completes, and the bursts simply stay at bit width 8 with
   no diagnostic. With Allo now emitting ``align_value`` on ``m_axi`` pointers
   (opt-in, ``configs={"align_value": 64}``), the same setting takes gmem0 to
   bit width 512. See :doc:`/designs/gemmini_comparison` and
   :doc:`/backends/vitis`. The status of this item is left as last recorded
   pending re-verification.


Surfaced by the 2026-09-19 re-verification and impact analysis
---------------------------------------------------------------

Found while re-verifying the items above (A-F, with repros under
``tests/limits/``) and while pricing the TinyTPU-isa deficit to Gemmini (G and
the shared-memory item; the variants are on ``main`` under
``examples/accelerator/tinytpu_vitis/impact/``, folded in from branch
``impact-limits`` -- commits ``f98c0dac`` and ``55405e00`` -- since deleted).

.. _limitation-shared-memory:

Shared on-chip memory: the one-owner rule is Allo's, not Vitis's
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Earlier text (on this page's sources and on the TinyTPU-isa pages) said Allo
enforces single reader / single writer *and Vitis rejects the violation
outright* (``HLS 200-779`` / ``200-979``). The second half is wrong. A Vitis
2023.2 probe (``impact/probe_shared/``):

.. list-table::
   :header-rows: 1

   * - mode
     - construct
     - Vitis 2023.2
   * - 0
     - 1 writer + 2 readers, no pragma
     - ``HLS 200-779`` single reader / single writer
   * - 1
     - + ``#pragma HLS stable``
     - ``HLS 200-779``
   * - 3
     - + ``#pragma HLS stream type=shared``
     - ``HLS 200-1014``
   * - 4
     - + ``#pragma HLS stream type=unsync``
     - ``HLS 200-780`` 3 processes, only 2 ports
   * - 5
     - 2 processes (one writes and reads, one reads) + ``type=unsync``
     - **accepted**: ``HLS 200-824`` shared without synchronization; port 0 to
       one process, port 1 to the other (``HLS 200-755``)
   * - 6
     - the same 2 processes, no pragma
     - accepted as a synchronized PIPO (ping-pong handoff)
   * - 2
     - one ``m_axi`` bundle read by 2 processes
     - ``HLS 200-1013`` / ``HLS 200-984``

So ``#pragma HLS stream variable=buf type=unsync`` shares an on-chip array
between two processes, one per BRAM port (``HLS 200-824`` / ``200-755`` /
``200-634``), and ``200-779`` applies only to synchronized arrays. Vitis does
separately forbid one ``m_axi`` bundle read by two processes (``200-1013`` /
``200-984``).

- **Allo's side:** Allo refuses a region-scope Stateful shared by two kernels
  (``EmitVivadoHLS.cpp:3083-3122``) and never emits ``stream type=unsync``.
- **SystemC fork's side:** ``AlloMemPins`` instances are keyed per (call,
  operand) at fork ``EmitSystemC.cpp:2539-2545``, so each client gets a
  replica rather than a port of one memory. Fix: ~50-100 lines binding one
  writer and one reader to one instance's two pin sets.
- **Measured impact on TinyTPU-isa: 0 cycles.** Every restructure the design
  needed was Allo-legal. The earlier claim that removing ``vru``'s double
  handling needs a second producer on a shared memory **was wrong**.

.. _limitation-a:

A. The dataflow simulator cannot run ``allo.exp`` / ``allo.log``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simulator's own pipeline in ``LLVMOMPModule.__init__``
(``simulator.py:1674-1686``) has no math-to-LLVM pass, so a kernel using
``allo.exp`` fails with ``Failure while creating the ExecutionEngine``
(``cannot be converted to LLVM IR: ... for op: math.exp``). The plain LLVM
backend lowers the same op (``populateMathToLLVMConversionPatterns`` in
``lower_allo_to_llvm``). This also breaks item 4's documented workaround on the
simulator. Fix: 1 line, verified. Repro: ``tests/limits/new_sim_math_lowering.py``.

.. _limitation-b:

B. ``Stream.get()`` is never cast to the destination type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``x: int8 = s.get()`` or ``b[i] = s.get()`` with an ``int32`` stream and an
``int8`` destination fails IR verification (``'affine.store' op value to store
must have the same type as memref element type``), while the same narrowing
from an ``int32`` *array* element works. Root cause ``builder.py:1054``.
Workaround: land the value in a temporary of the stream's own type first.
Fix: 3 lines, verified. Repro: ``tests/limits/new_stream_get_no_cast.py``.

.. _limitation-c:

C. ``customize()`` exits the interpreter on a frontend error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any frontend error inside ``allo.customize`` (type inference or IR building)
prints a traceback and calls ``sys.exit(1)`` (``customize.py:1382,1408``). A
``SystemExit`` is not caught by ``except Exception``, so a sweep, a notebook or
a test harness that expects to catch a build failure and move on is terminated
instead. Repro: ``tests/limits/new_customize_sys_exit.py``.

.. _limitation-d:

D. Catapult ``full()`` always returns ``false``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``EmitCatapultHLS.cpp:402-414`` emits ``full()`` as a hard-coded ``false``
(``ac_channel`` has no ``.full()``). Same class as item 18: a design that
branches on it has the branch compiled away, silently. Reported by
``tests/limits/item18_catapult_try_ops_blocking.py``.

.. _limitation-e:

E. Item 5 extended: same-type shadowing of a region parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A kernel local with the same name **and the same type** as an enclosing region
parameter (e.g. ``buf: int32[4] = 0`` against a ``buf: int32[4]`` parameter)
does not hit item 5's type-mismatch assertion; it gives an MLIR
region-isolation error instead. Same root cause as item 5 (shared scope list).
Repro: variant (b) of ``tests/limits/item05_region_param_shadowing.py``.

.. _limitation-f:

F. The AIE ``cpp-style`` typing rules reject every bitwise op
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What survives of item 7: the ``cpp-style`` typing rule set used by the AIE
target (``typing_rule.py:855-860``) has no bitwise ops, so ``&``, ``|``, ``^``
and shifts are rejected there while the default rule set accepts them. Shown by
the informational line of ``tests/limits/item07_bitwise_and.py``.

.. _limitation-g:

G. ``= 0`` on an array is a runtime zero-fill loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An array declared ``= 0`` lowers to a runtime zero-fill loop: ``builder.py:1063-1075``
emits ``linalg.fill``, and ``hls.py:288`` (``convert-linalg-to-affine-loops``)
turns it into a loop that runs every invocation. Restoring ``spad = 0`` on
TinyTPU-isa (``v_memset``) measured **+409 / +361 cycles** at 4x4x4 /
16x16x16 (661 / 1280 against 252 / 919, the design shipped until
``e24e433b``; the same with all six arrays ``= 0``, ``v_memset6``). This is the
514-cycle zero-fill the shipped design removed.

The right semantics are one of: warn; initialise at reset through the existing
Stateful ``memref.global`` path (correct only for the first invocation); or
elide the fill when every element is provably written before it is read.


.. _limitation-h:

H. A sub-region from another module is type-checked against the caller's globals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Found 2026-09-19 while re-checking fork issue #4; it had no row until then, and
the register previously cross-referenced #4 to item 3, which is a different,
already-fixed simulator bug.

When a region calls a sub-region, ``builder.py`` builds the callee with
``ASTContext(global_vars=ctx.global_vars.copy(), ...)`` (``builder.py:2838``) --
the **calling** region's globals, not those of the module that defines the
sub-region. So a sub-region declaring ``fifo: Stream[int32, 4][N_SUB]``, with
``N_SUB`` and ``Stream`` imported only in its own module, fails when called from
another module: ``infer.py:90`` reports "stream array shape should be a compile
time constant", or "Unsupported type ``Stream``" if the caller does not import
``Stream`` either. The same sub-region builds fine on its own. Upstream has the
same code.

Workaround: import the sub-region's shape constants and ``Stream`` into the
calling module; composition then runs and gives the right result.

The repro, `new_subregion_foreign_globals.py
<https://github.com/sunwookim028/allo/blob/main/tests/limits/new_subregion_foreign_globals.py>`__,
runs both variants and prints ``REPRODUCES`` while the caller-lacks-globals
variant fails. That variant surfaces as ``SystemExit: 1`` rather than an
exception because ``customize()`` exits on frontend errors (item
:ref:`C <limitation-c>`); the real message is on stderr. The sub-region and the
working variant are written as generated modules, because a nested definition
trips the separate re-parse ``IndentationError`` of upstream issue #588.

Fix: merge the callee's own module globals over the caller's when building a
sub-region -- about five lines, not yet prototyped.
