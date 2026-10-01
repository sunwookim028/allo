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
|                            |                          | loop, ``tinytpu/chia_agent/``) and       |
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
``examples/tinytpu/impact/``), after its best stack became the
shipped design (``e24e433b``), and deleted.

``main`` is the one long-lived working branch; ``upstream`` is just the mirror,
and ``gh-pages`` holds the published site. ``chia-codesign`` is retired. Exploration branches come and go under the rule in
"Where work lands" below.
``fix/vhls-mlir-percent-alloc-csim`` is gone: upstream PR #554 merged and is now
the tip of ``upstream/main``.

Checkouts on this host
----------------------

**One checkout, ``~/allo``, on ``main``.** It holds the repository (every
worktree's ``.git`` points into it) and its own ``mlir/build``, rebuilt with
``ninja -C mlir/build`` after a pull that touches ``mlir/``. Other long-lived
paths: ``~/llvm-allo-6b09f739`` (``LLVM_BUILD_DIR``) and ``~/chia-ortools``
(``scripts/act-test-recipe.sh``). ``chia.env`` lives only in ``~/allo``
(gitignored, mode 600). Keep a backup of it outside the repository, never
under a name ``.gitignore`` does not match.

Feature work goes in a worktree, ideally in the session's scratch directory,
and the worktree is removed when its branch merges. Before removing one:

- ``git -C <wt> status`` is clean, and ``git rev-list HEAD --not --remotes``
  is empty. Push the branch if it is not.
- Anything the work produced in **gitignored** paths (``chia_runs/``,
  ``.scratch/``, a hand-made measurement directory) is harvested into
  ``dev/records/`` first. ``git worktree remove`` deletes ignored files
  without asking.
- Nothing is running from it: a CHIA run leaves a Ray head node behind
  (``ray status``; ``ray stop`` from the ``chia_env`` env), which holds tens of
  processes and gigabytes of RAM after its jobs finish.

The 2026-09-29 cleanup retired ``allo-coord``, ``allo-minitpu``,
``allo-smoke``, ``wt/`` and ``work/``, eleven merged branches, and a Ray node
four days idle. Before that it harvested the only copies of CHIA run 4's
evidence (``dev/records/tinytpu/chia-evidence/isa-run4-20260925/``) and of the
fast-feedback measurements (on ``chia-fast-feedback``). ``~/allo``'s old
detached commit is tag ``archive/dev-tree-31a8e9ce``; ``main`` supersedes it.

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
``tests/systemc/rtlsim/`` survive only in its history, at
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
  it is recorded in ``tests/systemc/rtlsim/guard_experiment/`` and in
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
  are described in ``docs/source/developer/toolchains.rst``.


Standing hazards
----------------

Moved from ``dev/roadmap.md`` on 2026-10-01, when that file was retired.

Recorded because each cost real work today.

- **Never run a tree-wide git operation in a checkout another agent is writing
  to.** This destroyed an agent's work once and nearly a second time.
- **Removing a worktree can orphan a daemon that keeps accepting connections.**
  Third instance of "removing a worktree is not free". A Ray head started
  2026-09-22 had its raylet's cwd inside a worktree removed later that day.
  It kept listening, kept accepting connections, and **could not spawn a single
  worker** -- every one died in ``setup_worker.py`` at ``os.getcwd()``. A driver
  using ``address="auto"`` blocked in ``ray.get`` with no error, because
  ``/tmp/ray/ray_current_cluster`` still named it. It cost an agent 25 minutes to
  diagnose, and it is the same fails-open shape: the service was up, the
  connection succeeded, and nothing worked.
  **Third occurrence, 2026-09-25**, and the one that produced a check: a clean
  worktree found ``/tmp/ray/ray_current_cluster`` naming ``128.84.48.164:6399``
  with no ``gcs_server`` and no ``raylet`` behind it, and the first ``smoke.py`` sat
  for ten minutes printing nothing but ``Failed to connect to GCS``.
  **This is now checked.** ``preflight.py`` -- which ``swarm.py``, ``loop.py`` and
  ``smoke.py`` run before any worker and any model call, and which
  ``checkout_setup.sh`` runs as its step 4 -- resolves the address a driver
  would really dial (``RAY_ADDRESS``, else
  ``<RAY_TMPDIR or /tmp/ray>/ray_current_cluster``), probes it with a 3 s
  timeout, and refuses with the address named and the fix spelled out. It also
  refuses a head that *does* accept the connection but whose raylet's working
  directory is deleted, which is the 2026-09-22 shape above. Run it alone with
  ``python examples/tinytpu/chia_agent/preflight.py --check-ray`` ($0).
  *Do not fix this with ``ray stop``* -- it matches by process name across the
  whole host and would kill every other track's raylet. Kill the orphaned
  session's PIDs, owner confirmed. To avoid it: start a head with a private
  ``--temp-dir`` and port and export ``RAY_ADDRESS``, which overrides even an
  explicit ``address="auto"``, so no code change is needed and
  ``ray_current_cluster`` is left alone for everyone else. The temp-dir path
  must be **short** -- ``/tmp/ray-<track>``, not one under a scratch directory:
  Ray puts a Unix socket under it and AF_UNIX caps the path at 107 bytes.
  Related: the CHIA tool servers bind fixed ports from 8000 upward, so two
  ``chia_agent`` processes collide. Serialise them.
- **Remove a worktree when its agent finishes, not when the disk fills.** Each
  costs 0.6–1.9 GB; five live agents accrue ~2 GB.
- **``/tmp`` is 15 GB and shared.** When it fills, the harness cannot create its
  output-capture file, so *every* command fails including the escape hatch, and
  the only way out is a terminal outside the tool.
- **The configuration trap**: the worktree default is not ``MAXDIM=16``, and
  ``TPU_QD`` now defaults to 16. State the variables beside every number. This
  produced two published mistakes and two near-misses in two days.
- **The ``allo`` conda env's editable install pointed at a removed worktree.**
  Repointed to ``/home/sk3463/allo`` on 2026-09-24. It broke silently: `import
  allo` still worked from inside a checkout and failed everywhere else, so it
  surfaced only when an agent ran a docs build from its own worktree. Removing
  a worktree is not free if anything outside git references it.
- **A test whose subprocess does not pin ``PYTHONPATH`` is not testing your
  checkout.** Sharper than the editable-install entry above, and worse: there
  ``import allo`` *failed* and you noticed. Here it *succeeds against the wrong
  tree*. ``tests/dataflow/test_bf16_dataflow.py`` ran its emitter probes through
  ``subprocess.run([sys.executable, script])`` with no ``env=``, so they resolved
  ``allo`` through the editable install -- ``/home/sk3463/allo``, a checkout at
  ``31a8e9ce`` predating every fix of 2026-09-25. Those arms had been going red
  and green for reasons unrelated to the code under test, and the red was read
  as "bf16 still aborts" when bf16 had emitted for hours. Fixed by pinning the
  root via the ``pyproject.toml`` marker. **If a test shells out, pass ``env`` with
  the checkout root on ``PYTHONPATH``.**
- **A pipeline hides the exit code of everything but its last command.**
  ``pytest ... | tail -3`` returns *tail's* status, which is always 0, so a
  chained ``&& git push`` runs on a failing test suite. I did exactly that on
  2026-09-25 and pushed before knowing the result; it happened to be clean, and
  the 18 failures I had seen were a stale-bindings artefact. Redirect to a file
  and check ``$?``, or use ``PIPESTATUS``. Read the output *and* the code -- the
  house rule "read the output, not the exit code" guards against a different
  failure and does not cover this one.
- **Test a check's failure path, not only its pass path.** A check that has
  only ever said ok proves nothing. The ASIC preflight's library-checksum
  branch was confirmed by copying a real ``stdcells.db``, appending one null
  byte, and pointing the preflight at it: it reported the mismatch, named both
  digests, said the areas were not comparable with the committed set, and
  **refused to print the run sequence**. That is discrimination; a passing run
  alone would not have shown it. The same tool has one branch still untested --
  what it does when committed snapshots disagree about the library -- for the
  good reason that there is no second library to test it with, and that is
  recorded rather than glossed.
- **Some provenance cannot be recovered after the fact — capture it at run
  time or not at all.** A content digest over an export's files identifies
  which RTL a synthesis run actually read, where a file-list checksum does not
  (two of our exports hash identically, because the manifest lists the same
  filenames). But it can only be computed while the files the run read still
  exist at the path it read them from. An attempt to recover it by matching
  directory basenames would have stamped a 2026-09-22 area with RTL
  re-emitted two days later — a plausible lie, written by a tool. The fallback
  was deleted; the tool now reports the directory unreadable and computes
  nothing.
- **An instrument can fail by eating the evidence, not only by missing it.**
  The first attempt at that re-capture **overwrote four good snapshots** with
  "unresolved", because their directories had moved. Same shape as the other
  fails-open findings, one step worse: the check did not merely fail to see
  something, it destroyed what was there. Nothing was committed in that state.
  A re-capture now **keeps** the earlier record, marks it ``recaptured``, and
  says so. Treat any tool that rewrites a record in place as a tool that can
  lose one.
- **A refactor can silently remove an agentic loop's reach, and nothing fails.**
  The CHIA loop's editable set was ``("microarch_isa.py", "isa_dsl.py")``. The
  decomposition that made the design a composable library moved the hardware
  into eight ``ip/units/`` modules, so **the editable set no longer contained the
  machine** — the loop could still run, still build, still be graded, and could
  no longer change the thing it was searching over. Both of run 1's wins landed
  in files that are now ``dma_load.py`` and ``sequencer.py``, so that run is not
  reproducible against today's tree. Nothing in the harness noticed, because
  every gate still passed on a candidate that had edited nothing that mattered.
  The fix is one definition of the editable paths (``chia_agent/design.py``)
  rather than a literal list that a refactor can orphan. **Generalise: when a
  harness names the files it may touch, a refactor must move that list too, and
  the list should be derived rather than written.**
- **Subagents do not reliably have ``ListAgents``.** An agent told to announce
  its path claim to its peers could not see them, and guessing at names
  returned "no agent reachable". Pass peer agent IDs explicitly in a brief
  rather than instructing an agent to look them up — and when an agent cannot
  reach a peer it should say so and let the dispatcher relay, not guess.
- **The session scratchpad is shared between concurrent agents.** Two agents
  writing ``reproduce.sh`` output to the same scratchpad path truncated one
  another's log mid-cosim; the verdict survived only because the summary block
  happened to be contiguous at its own offset. An agent that reads a truncated
  log sees a run that did not finish, or worse a run that appears to have
  finished differently. Give every agent a distinct path, and prefer a
  worktree-local file to a shared scratch directory.
- **Never resolve a repository root by counting levels.** Search upward for a
  marker, or take the path as an argument. Hit three times in two days: a
  construct script counting three ``dirname``s landed on ``examples/`` rather than
  the root and could not find its node library from a clean checkout (and was
  *announced fixed without being run* -- a different check was run and taken as
  coverage); the TinyTPU rename found ~30 ``ROOT``/``REPO`` values derived by
  counting, **three already wrong** and made correct only by accident of the
  move; and the first fix was itself a re-count that happened to suit the new
  layout. A relative path that encodes tree shape is a latent break in any
  repository being reorganised, and this one is mid-reorganisation.
- **A licence-free check cannot report a licensed branch as passing.** The
  ASIC preflight's ADK-checksum branch reports as *correctly missing* without a
  licence, which is not the same as passing. Say which branches a run could not
  reach rather than reporting the run as green.
- **Read a gate's output, never its exit code**, and confirm what ran is what is
  being claimed. Four instruments were caught reporting success without having
  run; three further checks ran against the wrong object and passed.

Known-failing tests
-------------------

*Known-failing tests, so nobody chases them.* ``pytest tests/`` does not collect
on this host: 25 errors, all ``tests/dataflow/aie/*``, no ``aie`` module. With that
directory ignored: **800 passed, 64 skipped, 2 xfailed, 7 failed** — and those
same seven fail on a clean ``main``, so they are pre-existing and unrelated to
anything recent: ``test_hierachical_mesh::test_2x2``, three in
``ip_integration/test_external.py``, ``test_builder::test_minmax_cast``, and two in
``test_stateful.py``.
