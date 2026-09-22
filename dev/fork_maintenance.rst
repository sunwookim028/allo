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

#####################################
Fork Maintenance and Upstream Merging
#####################################

Durable procedure for reconciling ``main`` with ``upstream/main``. Project state
(open PRs, branch dependencies) is judged from git/GitHub, not checked-in ``.md``
snapshots; the living fork-vs-upstream feature map is the pinned fork issue
https://github.com/sunwookim028/allo/issues/13.

When a PR merges upstream
-------------------------

1. ``git fetch upstream`` to refresh ``upstream/main``.
2. Delete the merged branch locally and on ``origin`` (the fork).
3. Reconcile ``main`` with the merged commit. ``main`` is not a fast-forward of
   ``upstream/main``, so this goes through the main<->upstream reconciliation
   (fork issue #5) and MUST preserve the fork-local features inventoried in
   fork issue #5 (nb-stream primitives, Catapult nb-stream support):
   https://github.com/sunwookim028/allo/issues/5#issuecomment-4977128476.
4. Update the affected fork issues (``gh issue list -R sunwookim028/allo``) and
   any relevant documentation pages (``docs/source/``).
5. Rebase any live feature/fix/wip branches that carried the merged branch as
   a dependency onto the refreshed ``upstream/main`` to drop the merged-in
   commits; check ``git branch -a`` for the current set (branches come and go,
   so do not hardcode names here).

Branch layout (as of 2026-09-19)
--------------------------------

+----------------------------+--------------------------+------------------------------------------+
| Branch                     | Lineage                  | Role                                     |
+============================+==========================+==========================================+
| ``main``                   | cornell-zhang            | Fork integration branch: upstream plus   |
|                            |                          | fork-local features. **Not** a mirror of |
|                            |                          | upstream — 0 behind ``upstream/main``    |
|                            |                          | (``094ab413``, upstream #612) since the  |
|                            |                          | 2026-09-19 reconciliation merge          |
|                            |                          | ``dc6b8fa6``. Holds the one design,      |
|                            |                          | TinyTPU-isa, and the docs source.        |
+----------------------------+--------------------------+------------------------------------------+
| ``upstream``               | cornell-zhang            | Mirror of ``upstream/main``, tracking    |
|                            |                          | the ``upstream`` remote. Refresh it to   |
|                            |                          | see what has landed; diff ``main``       |
|                            |                          | against it to see what the fork carries. |
|                            |                          | **Rebasing main onto it is never**       |
|                            |                          | **automatic — it is an explicit call.**  |
+----------------------------+--------------------------+------------------------------------------+
| ``chia-isa``               | ``main``                 | **Landed on main 2026-09-19** (the CHIA  |
|                            |                          | loop, ``tinytpu_vitis/chia_agent/``) and |
|                            |                          | deleted. History and raw run output:     |
|                            |                          | tag ``chia-isa-run1-evidence``.          |
+----------------------------+--------------------------+------------------------------------------+
| ``chia-codesign``          | ``kkkaishao/allo`` (ACT) | **Retired 2026-09-19**, tag              |
|                            |                          | ``chia-codesign-final`` (``629c2767``).  |
|                            |                          | Kept on origin read-only: the superseded |
|                            |                          | fp32 TinyTPU, Kai Shao's imported work   |
|                            |                          | (its ``ATTRIBUTION.md``) and the CHIA    |
|                            |                          | run history. No worktree. See below.     |
+----------------------------+--------------------------+------------------------------------------+
| ``upstream-omp-team-size`` | cornell-zhang            | One commit on ``upstream/main``:         |
|                            |                          | upstream draft PR #611 (simulator OpenMP |
|                            |                          | team sized to the section count). Delete |
|                            |                          | after it merges, per the procedure       |
|                            |                          | above.                                   |
+----------------------------+--------------------------+------------------------------------------+

``impact-limits`` (the gap attribution's measured design variants) was folded
into ``main`` on 2026-09-19 (``96c3aef6``, under
``examples/accelerator/tinytpu_vitis/impact/``), after its best stack became the
shipped design (``e24e433b``), and deleted.

``main`` is the one long-lived working branch; ``upstream`` is just the mirror,
and ``gh-pages`` holds the published site. ``chia-codesign`` is retired. Exploration branches come and go under the rule in
"Where work lands" below.
``fix/vhls-mlir-percent-alloc-csim`` is gone: upstream PR #554 merged and is now
the tip of ``upstream/main``.

Read-only lineages on the ``kai`` remote, for reference rather than merging:
``kai/main`` (has ``dataflow.py``, plus ``frontend/ harness/ primitives/``),
``kai/allov2`` (``compiler/ lang/ operators/ schedule``, no ``dataflow.py``, no ACT),
and ``kai/act`` (the allov2 lineage plus ``exp/dsa`` — the ACT compiler flow that
``chia-codesign`` descends from).

The ``choonsik1`` remote (``https://github.com/choonsik1/allo.git``) carries the
SystemC/Catapult emitter, which is to be integrated rather than only read.
``SystemC-emitter`` is its most complete branch. ``systemc-ip-integration`` lacks
the RAM-pin memory interface (``b92077f5``) and the free-running-loop fix
(``72c70dcb``), so an integration must start from ``SystemC-emitter``. The
``pe_wire``/``pe_stream``/``pe_channel`` netlists used by
``examples/systemc_rtlsim/`` survive only in its history, at
``0eff4888:agents/noc/rtl/<design>/rtl.v``.

Tags, in place of branches that were retired because their history is reachable
elsewhere: ``tinytpu-rtlgen-base`` (``882f7dd6``, the retired branch tip, now an
ancestor of ``chia-codesign``; the actual fork point between ``main`` and
``chia-codesign`` is ``76130c63``) and ``wip-u280-rescue`` (``1bd3c5a6``, an unreviewed
u280 / nb-stream / fp16 rescue point from 2026-07-06).

Where work lands
~~~~~~~~~~~~~~~~

- ``main`` holds every solid frontier and must reproduce the headline results
  on its own. A result is solid once it has been re-derived independently
  (e.g. a second cosim run, or a replay into a clean tree); an agent's
  unverified number does not qualify.
- Exploration, whether tooling or design, goes on its own branch when it is
  expected to take more than about 6 meaningful commits. Smaller changes go
  straight to ``main``.
- An exploration branch lands on ``main`` once its result is solid, together
  with whatever reproduces it. A dead end is deleted, and its conclusion is
  recorded in the documentation (``docs/source/``) on ``main``.
- Name the branch after the question it answers, and list live ones in the
  table above while they exist. For example, ``sc-wire-guard`` asked whether
  extending the free-running-loop guard fixes ``pe_wire``. The answer was no;
  it is recorded in ``examples/systemc_rtlsim/guard_experiment/`` and in
  ``docs/source/extensions/catapult_systemc.rst``, and the branch is deleted.

``chia-codesign`` is a different codebase, not a feature branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It shares only a March 2026 ancestor (``76130c63``, upstream #555) with ``main``.
Kai Shao's ACT fork (https://github.com/kkkaishao/allo) re-architected the
package (see ``docs/source/extensions/act.rst`` and ``docs/source/extensions/chia.rst``):

+-----------+------------------------------------------+------------------------------------------+
|           | ``main``                                 | ``chia-codesign``                        |
+===========+==========================================+==========================================+
| ``allo/`` | 98 files: ``dataflow.py``, ``ir/``,      | 117 files: ``compiler/``, ``lang/``,     |
|           | ``passes.py``, ``customize.py``,         | ``operators/``, ``schedule/``, ``exp/``  |
|           | ``_mlir/``, ``autoscheduler/``           |                                          |
+-----------+------------------------------------------+------------------------------------------+

None of the first column's files exist in the second. A trial merge
(``git merge-tree main chia-codesign``) reports 26 conflicts, 24 of them "core
file deleted in chia-codesign and modified in main". **The two lineages are
maintained separately and are not expected to converge**; the upstream-merge
procedure above applies to ``main`` only. ``chia-codesign`` carries its own
frontmatter (``CODESIGN.md``) and checkpoint (``notes/CHIA_CHECKPOINT.md``).
**Neither file exists on** ``main`` -- do not go looking for them in this tree.
They live on the retired ``chia-codesign`` branch, which has no worktree any
more: read them with ``git show chia-codesign-final:notes/CHIA_CHECKPOINT.md``.

Project state
-------------

Live state is judged from git and GitHub, not from a checked-in status file:
``git branch -vv``, ``gh pr list -R cornell-zhang/allo``,
``gh issue list -R sunwookim028/allo``, and this documentation.

- Living feature map (fork vs upstream): the pinned fork issue
  https://github.com/sunwookim028/allo/issues/13. It is the single living
  picture of what the fork carries vs upstream.
- Fork-local file inventory: fork issue #5
  (https://github.com/sunwookim028/allo/issues/5#issuecomment-4977128476).
- The LLVM builds and worktree layout that the procedure above has to respect
  are described in ``dev/toolchains.rst``.
