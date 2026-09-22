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

####################################
Hierarchical Regions: Design Record
####################################

.. note::

   **Historical design record**, drafted 2026-05-12 (status update
   2026-07-15), retired 2026-09-17. Items 1-2 below landed upstream via PR #577
   (merged 2026-05-13). The item-4 file references (``_build_top`` bare-scalar
   emission, ``s_axilite`` in ``postprocess_hls_code``) describe code that has
   since been **reverted** -- on ``main`` today, bare scalars in ``args=[...]``
   are rejected and ``s_axilite`` is not emitted on the Vitis path (see
   ``docs/source/developer/pitfalls.rst`` and :ref:`limitation-23`). Live tracking of the
   open questions is fork issue #7.

This page preserves the parts of the former
``notes/archive/HIERARCHY_DESIGN.md`` ("Allo hierarchical-region pitch:
consolidate or fragment?") that fork issue #7 does not carry: the
``IsolatedFromAbove`` diagnosis (§5) and the three architectural alternatives
with the recommendation between them (§8). Fork issue #7 carries the four
faces, §5's root cause and Alt A, but **not** Alt B, Alt C or the
recommendation. §4 is kept as the context needed to read them. It is still the
fullest statement of the region-as-module argument.

Status update (2026-07-15)
--------------------------

Items 1 and 2 below (simulator nested-call streams; region-scope Stateful
crash) have landed upstream via PR #577 (merged 2026-05-13). The analysis
below is kept as-is as the historical design record dated 2026-05-12; it is
not being rewritten to match. Live tracking of the remaining open design
questions (items 3-4, and the section 7-8 open questions / architecture
alternatives) is fork issue #7.

A research artifact for Sunwoo Kim, drafted 2026-05-12 before a future
"pitch the maintainers" session. The four items in scope are:

1. ``fix/simulator-nested-call-streams`` — simulator's ``_process_function_streams`` does not deep-scan sub-region calls in control flow. (Landed upstream via PR #577; see status update above.)
2. ``feature/region-scope-stateful`` — region-body ``@Stateful`` shared across inner kernels (crashes on upstream ``main``). (Crash fix landed upstream via PR #577; see status update above.)
3. Per-kernel ``static`` emission of ``@Stateful`` globals in ``EmitVivadoHLS.cpp`` (no branch yet).
4. Bare-scalar auto-capture ``s_axilite`` (no branch; PR #577 explicitly rejects scalar in ``args=[]``).

This file decides whether they go upstream as **four separate bugs/features** or as **one structural design discussion**.

4. The four items, re-examined
------------------------------

+---+------------------------------+-------------------------------+-------------------------------+
| # | Item                         | Local cause                   | Real root                     |
+===+==============================+===============================+===============================+
| 1 | Sim nested-call stream       | ``_process_function_streams`` | Sub-region calls inside       |
|   | lowering                     | doesn't recurse into control  | ``affine.for`` only happen    |
|   |                              | flow                          | with the decoder pattern; the |
|   |                              |                               | original walker was correct   |
|   |                              |                               | for the                       |
|   |                              |                               | single-kernel-flat-call-graph |
|   |                              |                               | world                         |
+---+------------------------------+-------------------------------+-------------------------------+
| 2 | Region-scope ``@Stateful``   | ``ASTContext.copy()`` doesn't | Stateful was designed         |
|   |                              | propagate stateful state      | kernel-local before hierarchy |
|   |                              |                               | existed; region-scope is a    |
|   |                              |                               | composition that the          |
|   |                              |                               | type-annotation builder       |
|   |                              |                               | (#509) doesn't handle         |
+---+------------------------------+-------------------------------+-------------------------------+
| 3 | Per-kernel ``static``        | ``emitFunction`` emits        | The HLS emitter treats each   |
|   | emission                     | stateful globals inside each  | function independently;       |
|   |                              | function                      | module-level Stateful symbols |
|   |                              |                               | have no module-level emission |
|   |                              |                               | site                          |
+---+------------------------------+-------------------------------+-------------------------------+
| 4 | Bare-scalar auto-capture     | Auto-captured scalars in      | Auto-capture lowering was     |
|   | ``s_axilite``                | vitis_hls don't get           | designed for arrays (memref)  |
|   |                              | ``s_axilite`` pragma          | and never extended to bare    |
|   |                              |                               | scalars; the AXI-Lite mapping |
|   |                              |                               | convention only exists in     |
|   |                              |                               | ``postprocess_hls_code`` for  |
|   |                              |                               | explicit ``args=[]`` scalars  |
+---+------------------------------+-------------------------------+-------------------------------+

(Status update 2026-07-15: items 1 and 2 above landed upstream via PR #577;
see status update near the top of this file. Items 3 and 4 remain open.)

**Connections between items:**

- Item 1 ↔ Item 2: If a region-scope Stateful triggers a sub-region call inside ``affine.for`` (the decoder driver pattern), item 1 is what makes item 2 lowerable. They cannot be tested separately — fixing one without the other means the test case still fails for the other reason.
- Item 2 ↔ Item 3: Item 2 is the **front-end** of region-scope Stateful (Python → MLIR); item 3 is its **back-end** (MLIR → C++). They are literally the two halves of the same feature. Filing them as separate items is misleading; a user landing item 2 alone gets MLIR that the HLS emitter mis-compiles, and landing item 3 alone gets an HLS emitter that nothing produces valid input for.
- Item 4 ↔ PR #577: The maintainer's "reject scalar in ``args=[]``" decision *creates* item 4 as a maintainer-acknowledged future-work item. PR #577's review correspondence is the explicit handoff to item 4.
- Item 4 ↔ Items 1/2/3: The ``tpu(ctrl, d_addr, n, dma_buf)`` signature is *the* trigger for all four items. Without item 4, every user falls back to ``int32[1]`` workaround, which masks the actual problem the maintainers should be solving.

The four items share four common ancestors: ``dataflow.region``, ``dataflow.kernel``, ``@Stateful``, and the vhls postprocess pipeline. None can be cleanly fixed without touching at least one of the others' files.

5. Structural model assessment
------------------------------

**Single root assumption (named):** *A* ``@df.kernel`` *is an* ``IsolatedFromAbove`` *MLIR function, and all sub-region calls happen at the top of the top kernel's body.*

This assumption was correct for the original Allo (2023–2025), where a region was a thin grouping of kernels each of which was effectively a standalone HLS pipeline. Once ``Stateful`` (Jan 2026, #487/#509) and hierarchical regions (#518/#520/#522) landed within two weeks of each other, the assumption became inconsistent with the new feature surface:

- **Sub-region calls in control flow** (item 1) violate "all calls at top of body" → simulator walker breaks.
- **Region-scope state** (items 2, 3) violates "kernel is isolated from above" → AST builder cache breaks, HLS emitter per-function statics break.
- **Scalar AXI-Lite control plane** (item 4) violates "kernel arg lowering = memref boundary" → auto-capture path has no ``s_axilite`` channel.

All four are *the same incompatibility* between (a) the kernel-is-an-island model and (b) the host-driven mesh-connected accelerator architecture. Industry HLS handles (b) by making the top function the shared-state owner; Allo's hierarchy was designed before that pattern was in scope.

**The structural diagnosis: Allo's** ``@df.region`` **is a function with kernels as bullets, when what mesh accelerators need is a** ``@df.region`` **that is a hierarchical module with kernels as concurrently-running processes and the region body as the shared compute substrate.** The four items are the symptoms of asking the first model to do the second model's job.

8. Suggested architectural alternatives to weigh
------------------------------------------------

Three sketches, not full designs. The future session should pick one as the pitch's centerpiece.

Alt A: Region-as-Module (industry-aligned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``@df.region`` becomes a SystemC-style hierarchical module:

- Region body owns Streams *and* Stateful arrays as module-scope state.
- Each ``@df.kernel`` is a concurrent process; access to region state is by name (no ``args=`` for state, only for top-level interface ports).
- Top-level interface ports get explicit binding attributes: ``@s_axilite("ctrl")``, ``@m_axi(bundle="gmem0")``. No more auto-capture-vs-args asymmetry.
- HLS emits one C++ function per kernel, one ``static T X[N];`` per Stateful at file scope, one top wrapper with the right pragmas.

Pros: matches Vitis HLS/Catapult/SystemC, all four items dissolve into a single redesign. Cons: bigger PR, breaks existing ``args=[A,B,C]`` ergonomics for new users.

Alt B: Capture-by-name without isolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep ``@df.kernel`` as a function, but drop ``IsolatedFromAbove`` for region-scope state symbols. Region-scope Stateful and scalars become *captured* automatically; kernels reference them by name. The MLIR side becomes a pre-pass that lowers captures into per-kernel arg lists *just before* the kernel functions are emitted.

Pros: minimal user-facing change; preserves the current kernel-as-function vocabulary. Cons: keeps the structural mismatch — every backend has to know how to lower captures, items 1/3 still need their own fixes (the items become smaller but don't dissolve).

Alt C: Single-decl multiple-use Stateful + ``df.scalar``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Smallest viable change: keep the current model; add (i) a module-level pass that ensures each ``__stateful_*`` symbol is emitted at file scope exactly once in vhls, (ii) a ``df.scalar(int32)`` type that explicitly binds to ``s_axilite``. Items 1 and 2 remain as separate bugfixes; items 3 and 4 dissolve.

Pros: easiest to land. Cons: doesn't resolve the structural mismatch — the *next* hierarchical-region project hits the same class of bugs in a different corner. Buys 6 months at most.

Recommended path: pitch Alt A as the long-horizon design, propose Alt C as the bridge-the-gap interim if maintainers want to land things in two weeks instead of two months. Alt B is the worst of both worlds and probably not worth pitching.

Recovery
--------

The dropped sections (§0-3 framing and evidence, §6 project implications, §7
open questions, §9 scope, and the file-reference appendix) are recoverable
with:

.. code-block:: bash

   git show a2d92cd8:notes/archive/HIERARCHY_DESIGN.md
