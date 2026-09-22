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

.. _reorganisation:

###################
Reorganisation Plan
###################

Four rules drive this plan.

a. Issue tracking is separate from the design and the tool implementations.
b. Dev notes are separate from design and tooling documentation.
c. Natural-language notes and docs are minimal.
d. Code and its organisation are self-explanatory **without multi-line
   comments**; modularity and naming carry the explanation.

The long-term goal is a system maintainable by LLMs, in particular Allo's
programming model, MLIR abstractions and compiler passes.

This page is the plan and its sequencing. It is itself scheduled for deletion
once the plan is executed.


Where we are
============

Measured on ``214b20fb``.

.. list-table:: Prose against code, worst offenders
   :header-rows: 1
   :widths: 44 9 9 9 10 19

   * - File
     - Lines
     - Prose
     - Code
     - Prose %
     - Verdict
   * - ``examples/accelerator/tinytpu_vitis/microarch_isa.py``
     - 1579
     - 821
     - 596
     - 52.0
     - 138 prose lines per 100 code lines; one 264-line module docstring
   * - ``examples/accelerator/tinytpu_vitis/isa_dsl.py``
     - 425
     - 191
     - 168
     - 44.9
     - 114 prose lines per 100 code lines
   * - ``examples/accelerator/tinytpu_vitis/chia_agent/param_check.py``
     - 117
     - 45
     - 58
     - 38.5
     - 78 per 100
   * - ``examples/accelerator/tinytpu_vitis/isa_ref.py``
     - 70
     - 25
     - 34
     - 35.7
     - 74 per 100
   * - ``examples/accelerator/tinytpu_vitis/bench_isa.py``
     - 135
     - 41
     - 77
     - 30.4
     - 53 per 100
   * - ``allo/`` and ``mlir/`` additions
     - 1664 added
     - ~190
     - --
     - ~11
     - already within rule (d); ``customize.py`` is 2.8 %
   * - ``tests/limits/`` (23 files)
     - 1701
     - 269
     - 1115
     - 15.8
     - already within rule (d)

The library changes are not the problem. The problem is concentrated in the
design and its harness, and in ``docs/source/``: 9508 of the 14033 ``.rst``
lines are fork-authored, and ``developer/limitations.rst`` alone is 1623 of
them. By class: 24 user-documentation pages (5367 lines), 2 design docs
(1289), 9 experiment records (4861), 3 dev notes (518), 1 issue tracker
(1623). **About 47 % of the published site is not user documentation.**

Four factual defects found while measuring, each a one-line fix, none of them
part of the reorganisation:

* ``designs/tinytpu_isa.rst:44`` says Gemmini is "1.07-1.24x **faster** at all
  five shapes"; ``:913`` of the same file, ``designs/gemmini_comparison.rst:30``
  and commit ``476a70d8`` all say we are 1.07-1.24x **slower**. ``:44`` is
  wrong.
* ``backends/simulator.rst:82-113`` still tells users to raise
  ``OMP_NUM_THREADS`` to 64 or 128 to avoid a hang. Register item 11 records
  that requirement as removed on 2026-09-17 (the simulator now sizes the
  OpenMP team to the section count). The correction is on four other pages and
  not on the one a site reader hits.
* ``backends/nonblocking_streams.rst:243-247`` gives FF 248 / 260;
  ``records/vitis_nb_streams.rst:88-89`` measures 1325 / 1369. Both pages
  disclose the conflict and name the record authoritative. 5.3x apart is worth
  resolving, not annotating.
* ``microarch_isa.py:216`` carries a historical cycle table
  (``4x4x4 | 1004 | 680``) inside the shipped design's docstring with no
  superseded marker; the shipped number is 172.


1. Issues move out of the documentation
=======================================

``docs/source/developer/limitations.rst`` is an issue tracker inside the
documentation: 21 open rows and 10 fixed-or-closed rows, each with a status, a
layer, a root cause at ``file:line``, an impact, a fix size and a repro link.
That is an issue body.

Done
----

Every **open** item now has an issue on ``sunwookim028/allo``, labelled
``limitation`` plus one ``layer:*`` label per affected layer plus one
``priority:*`` label:

.. list-table::
   :header-rows: 1
   :widths: 10 12 46 32

   * - Item
     - Issue
     - Title
     - Labels beyond ``limitation``
   * - 4
     - `#15 <https://github.com/sunwookim028/allo/issues/15>`__
     - ``math.exp`` / ``math.log`` not recognized by the AST builder
     - frontend, medium
   * - 5
     - `#16 <https://github.com/sunwookim028/allo/issues/16>`__
     - Region-param / kernel-local shadowing
     - frontend, high
   * - 8
     - `#17 <https://github.com/sunwookim028/allo/issues/17>`__
     - Sub-region rebuilt under the same symbol names per call site
     - frontend, medium
   * - 10
     - `#18 <https://github.com/sunwookim028/allo/issues/18>`__
     - Errors point at lowered MLIR, not source
     - frontend, simulator, hls-driver, medium
   * - 14
     - `#19 <https://github.com/sunwookim028/allo/issues/19>`__
     - ``wrap_io=False`` rejects multi-dimensional arguments
     - emitter, high
   * - 15
     - `#20 <https://github.com/sunwookim028/allo/issues/20>`__
     - csim runs processes in declaration order and hangs silently
     - frontend, high
   * - 16
     - `#21 <https://github.com/sunwookim028/allo/issues/21>`__
     - ``cosim`` not wired into ``df.build``; ``m_axi`` has no ``depth=``
     - hls-driver, high
   * - 17
     - `#22 <https://github.com/sunwookim028/allo/issues/22>`__
     - Frontend constraints and ``meta_if`` scoping
     - frontend, high
   * - 18
     - `#23 <https://github.com/sunwookim028/allo/issues/23>`__
     - Catapult ``try_get``/``try_put`` lower to blocking reads
     - emitter, low
   * - 19
     - `#24 <https://github.com/sunwookim028/allo/issues/24>`__
     - TAPA ``try_*`` fail to emit with a misleading message
     - hls-driver, emitter, low
   * - 22
     - `#25 <https://github.com/sunwookim028/allo/issues/25>`__
     - SystemC ``Wire`` is semantically incomplete
     - systemc-fork, low
   * - 23
     - `#26 <https://github.com/sunwookim028/allo/issues/26>`__
     - ``m_axi`` widening unreachable from user code
     - hls-driver, emitter, medium
   * - shared memory
     - `#27 <https://github.com/sunwookim028/allo/issues/27>`__
     - The one-owner rule is Allo's, not Vitis's
     - emitter, systemc-fork, medium
   * - A
     - `#28 <https://github.com/sunwookim028/allo/issues/28>`__
     - Simulator cannot run ``allo.exp`` / ``allo.log``
     - simulator, low
   * - B
     - `#29 <https://github.com/sunwookim028/allo/issues/29>`__
     - Scalar ``Stream.get()`` is never cast
     - frontend, medium
   * - C
     - `#30 <https://github.com/sunwookim028/allo/issues/30>`__
     - ``customize()`` calls ``sys.exit(1)``
     - frontend, high
   * - D
     - `#31 <https://github.com/sunwookim028/allo/issues/31>`__
     - Catapult ``full()`` always returns ``false``
     - emitter, low
   * - E
     - `#32 <https://github.com/sunwookim028/allo/issues/32>`__
     - Same-type shadowing of a region parameter
     - frontend, medium
   * - F
     - `#33 <https://github.com/sunwookim028/allo/issues/33>`__
     - AIE ``cpp-style`` rules reject bitwise ops
     - frontend, low
   * - G
     - `#34 <https://github.com/sunwookim028/allo/issues/34>`__
     - ``= 0`` on an array is a runtime zero-fill loop
     - frontend, hls-driver, high
   * - H
     - `#35 <https://github.com/sunwookim028/allo/issues/35>`__
     - Sub-region type-checked against the caller's globals
     - frontend, high

Each body carries the status, the layer, the root cause at ``file:line``, the
impact, the fix size, the exact repro command, a link to the
``tests/limits/`` repro and a link back to the register anchor. Items that
duplicate an older issue cross-reference it in the body rather than editing
it: ``#27`` references ``#11``, ``#35`` references ``#4``, ``#25`` references
``#9``.

Remaining
---------

1. **Close the fixed items.** Items 1 (+6), 2, 3, 7, 9, 11, 12, 13, 20 and 21
   are FIXED, CANNOT-REPRODUCE, NOT-A-LIMITATION or retracted. Open and
   immediately close one issue per item, with the fixing commit or PR in the
   body, so the tracker records the history the register currently carries.
   Do **not** reuse or edit ``#10``, already closed for item 21.
2. **Reduce the register to a pointer.** ``developer/limitations.rst`` becomes
   at most 40 lines: what the ``limitation`` label means, the three
   ``layer:*`` / ``priority:*`` vocabularies, how to run a repro, and a link
   to the label query. Everything else --- 1623 lines --- is either already in
   an issue body or is *narrative* that belongs on a design page.
3. **Route the narrative before deleting it.** Three blocks in the register
   are not issue material and must land somewhere first:

   * ``limitations.rst:1079-1103`` "an unused capability measures as a
     worthless one" --- a methodology finding. Target:
     ``designs/design_space.rst``.
   * ``limitations.rst:1104-1130`` "Theories tested and disproved" --- target:
     ``designs/tinytpu_history.rst``.
   * ``limitations.rst:399-448`` the dated corrections and retractions ---
     target: ``designs/tinytpu_history.rst``, which already holds dated
     history.

4. **Keep the repros where they are.** ``tests/limits/`` is the right place:
   they are not unit tests of Allo's intended behaviour, they are executable
   claims about Allo's *current* behaviour, one file per claim, each printing
   one ``[item N] <STATUS>`` line, deliberately not named ``test_*.py`` so
   ``pytest`` does not collect them. Rename each file to its issue number
   (``tests/limits/issue15_math_exp.py``) so the tracker and the tree agree,
   and add ``tests/limits/run_all.py`` printing one line per repro, so the
   whole register can be re-verified in one command. Two open items have no
   repro (``23``, ``G``); writing them is part of closing ``#26`` and ``#34``.

The documentation guard
-----------------------

``chia_agent/spec_policy.py:376`` sets ``DOC_LOSS_MAX = 15`` and
``doc_violations`` (``:402``) rejects any candidate whose net loss of
comment-plus-docstring lines in a spec file exceeds it, with the message
``Edit or add documentation; do not delete it``. It is live at three call
sites: ``allo_tool.py:169``, ``accept.py:168``, ``evaluate.py:186``. It exists
because the first paid run's accepted diff deleted the 260-line design
docstring of ``microarch_isa.py``.

What it does to a prose move, precisely:

* The agent **cannot** do the move at all. Its writable paths are
  ``microarch_isa.py`` and ``isa_dsl.py`` only (``allo_tool.py:40``);
  ``docs/`` is read-only through ``read_reference`` and ``open`` /
  ``write_text`` are in ``DENIED_NAMES`` (``spec_policy.py:57-145``). Any
  attempt produces a pure deletion.
* The deletion is then refused three times. ``doc_violations`` counts only the
  source file; a matching ``+300`` lines in
  ``designs/tinytpu_isa.rst`` does not offset it, because the guard never
  looks there.
* The count is **net and per-file**, so it is also gameable in the wrong
  direction: delete the 264-line docstring and add 250 lines of filler in the
  same file and it passes at a net 14.

Under rule (d) that guard forbids the very refactor this plan requires, and it
is measuring the wrong thing. It should be re-scoped from *prose volume* to
*information conservation*:

* Replace ``doc_violations`` with a **name-and-structure guard**: reject a
  candidate that deletes or renames a public name (a module-level function,
  a ``@df.kernel`` name, an ISA mnemonic, a design parameter) without the
  rename appearing on both sides of the diff. Prose is then free; the
  explanation that rule (d) relies on --- the names --- is what is protected.
* Keep one volume check, inverted: reject a candidate that *adds* more than
  ``DOC_ADD_MAX`` net prose lines to a spec file. Rule (c) makes growth the
  hazard, not shrinkage.
* Add a **docs-reference guard**: a candidate that removes a prose block
  larger than N lines must, in the same diff, touch a file under
  ``docs/source/designs/``. This is the mechanical expression of "prose moves
  out, it does not evaporate".
* The refactor itself must not be attempted by the CHIA loop. It is a human-
  and-reviewer change; the guard change lands first, the refactor lands
  second, and the loop is only re-enabled against the refactored tree.

Three other frozen checks bite a *human* refactor and are the real
constraint:

* ``evaluate.py:169-171`` refuses to evaluate at all if any of
  ``DESIGN_EVALUATOR`` --- ``cosim.py``, ``bench_isa.py``, ``stress_isa.py``,
  ``isa_ref.py``, ``kpn_model.py`` (``evaluate.py:103-104``) --- differs from
  ``MAIN_BASE = "476a70d8"`` (``:102``). Every one of those five is a file
  this plan edits. ``MAIN_BASE`` has to be bumped in the same reviewed commit,
  each time.
* ``accept.py:59-77,290-294`` keys ``BASELINES`` on the *blob ids* of
  ``microarch_isa.py`` and ``isa_dsl.py``. Any prose move changes both, and
  acceptance then reports ``claim: "no-baseline"`` until a new control run is
  recorded.
* ``loop.py:55-62,236-239`` requires the working-tree copies of those five
  files plus all of ``chia_agent/`` to equal ``HEAD`` before a search starts.

Also, ``evaluate.compose`` calls ``policy_violations`` (hence ``ast.parse``,
``spec_policy.py:228``) with no ``SyntaxError`` guard, so an unparseable
candidate crashes the evaluator with a traceback instead of producing a
``Reject``. ``allo_tool._check:161-164`` does guard it. Fix while the guard is
being re-scoped.


2. Dev notes leave the published site
=====================================

``conf.py:74`` is ``exclude_patterns = []``: every page in
``docs/source/`` is published at https://sunwookim028.github.io/allo/,
including the ones that describe this host.

Proposal: a ``dev/`` tree at the repository root, **not** under
``docs/source/``, in Markdown, added to ``conf.py``'s ``exclude_patterns``
only as belt-and-braces. Not the GitHub wiki: the wiki is a separate git
repository, so a dev note cannot be changed in the same commit as the code it
describes, which is the one property these notes need.

.. list-table:: Pages that move
   :header-rows: 1
   :widths: 40 9 51

   * - Page
     - Lines
     - Destination and why
   * - ``developer/toolchains.rst``
     - 211
     - ``dev/toolchains.md``. Absolute paths on one machine
       (``/opt/xilinx/…``, ``/home/sk3463/llvm-allo-…``). Useless and
       misleading to a site reader; load-bearing for whoever hacks here.
   * - ``developer/fork_maintenance.rst``
     - 176
     - ``dev/fork_maintenance.md``. Branch layout, worktrees, the
       upstream-merge procedure. Repository process, not Allo documentation.
   * - ``developer/pitfalls.rst``
     - 131
     - Split. Each pitfall that is an Allo defect becomes (or references) a
       ``limitation`` issue; each pitfall that is a *correct-usage* rule moves
       into ``dive/dataflow.rst``, where a user looks. Nothing is left, so the
       page is deleted.
   * - ``developer/limitations.rst``
     - 1623
     - Reduced to a pointer, as above.
   * - ``records/*.rst``
     - 1001
     - ``dev/records/``. Dated measurement records are evidence, not
       documentation; they are read once, by us.
   * - ``examples/accelerator/tinytpu_vitis/logs/``
     - 579 KB
     - ``dev/records/tinytpu/``. 23 committed run logs sitting in an
       ``examples/`` tree.
   * - ``backends/catapult.rst:209-508``
     - 300
     - ``dev/toolchains.md``. Not the whole page --- this block only: the
       zhang-21 hostname, ``/opt/siemens/catapult/2024.2``, a dated
       licence-server verification and a hostname-guarded ``~/.bashrc``
       snippet, inside an otherwise good backend user doc. Same class of
       content ``developer/toolchains.rst`` exists to hold.

.. list-table:: Pages that stay published
   :header-rows: 1
   :widths: 40 60

   * - Page
     - Why
   * - ``developer/index.rst``
     - Upstream's contributor guide, unmodified on the fork.
   * - ``developer/dataflow_semantics.rst``
     - Genuine tooling documentation: what the simulator, csim and cosim each
       do and do not model. A user needs it. Rename to
       ``backends/execution_models.rst``, beside the backends it compares.
   * - ``backends/*``, ``dive/*``, ``api/*``, ``setup/*``
     - User documentation.
   * - ``designs/*``, ``extensions/*``
     - The case study. It is the argument the fork exists to make.

``designs/minitpu.rst`` (254 lines) is a third case: it reads out QoR from
*another engineer's* tree at ``~/core/minitpu`` and says so. It is neither our
design nor a general record. It belongs in ``dev/records/``.

After the move ``docs/source/developer/`` holds only upstream's
``index.rst`` (which has no toctree of its own --- it is a leaf listed by the
root), and the "Developer Guide" toctree in ``index.rst:73-84`` shrinks to
four entries. The "Measurement Records" toctree (``index.rst:87``) disappears
entirely.

Two housekeeping items while the pages move:

* ``developer/fork_maintenance.rst:47`` has a branch table "as of
  2026-09-19" that already disagrees with the repository: ``chia-isa`` is
  listed as deleted but still has a worktree, ``chia-codesign`` is listed as
  kept on ``origin`` but ``git ls-remote`` shows no such head, and
  ``tinytpu-align``, ``measure-headline`` and ``audit-reorg`` are missing.
  The page itself says at ``:44`` "branches come and go, so do not hardcode
  names here", and then hardcodes them. Delete the table; ``git branch -vv``
  is the answer.
* ``conf.py:74`` gains the ``dev/`` exclusion as belt-and-braces even though
  the tree will sit outside ``docs/source/``.


3. The design and harness satisfy rule (d)
==========================================

``microarch_isa.py`` today
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 10 10 46

   * - Unit
     - Lines
     - Code
     - Prose
     - Content
   * - module docstring
     - 4-267
     - 0
     - 264
     - the design essay
   * - constants, ``enc_agu``, ``enc``
     - 269-504
     - ~70
     - ~120
     - ISA encoding, with three prose blocks of 34, 44 and 16 lines
   * - ``tinytpu_isa`` (the region)
     - 506-1130
     - 412
     - 193
     - eight ``@df.kernel`` PEs: ``sequencer`` (543-686), ``dma_ld``
       (688-763), ``spm`` (765-830), ``vru`` (834-877), ``wld`` (879-928),
       ``pe`` (930-984), ``accu`` (986-1098), ``dma_st`` (1100-1130)
   * - program generators
     - 1132-1264
     - 68
     - 59
     - ``gemm_program_handwritten``, ``gemm_program_flat``,
       ``vadd_program``, ``expand``
   * - validator and assembler
     - 1267-1555
     - ~165
     - ~86
     - ``_trace``, ``ProgramError``, ``check_program``, ``assemble``
   * - ``schedule``
     - 1558-1579
     - 8
     - 14
     - the schedule primitives

Why the region cannot be split: the mechanism
---------------------------------------------

Kernel discovery is a **purely syntactic scan of the region function's own
source text**. Nothing reads ``__closure__``, ``__code__`` or a registry.

* ``allo/ir/utils.py:148-151`` --- ``inspect.getsourcelines(fn)``,
  ``textwrap.dedent``, ``ast.parse``. For a sub-region,
  ``allo/ir/builder.py:2825-2826`` does the same with ``inspect.getsource``.
* ``allo/ir/infer.py:730-739`` and ``allo/ir/builder.py:2016-2033`` find a
  kernel by matching ``decorator.func.attr == "kernel"`` on a nested
  ``FunctionDef``. Only the attribute *name* is matched --- ``df`` is never
  resolved --- and ``mapping`` must be the **first** keyword, because
  ``builder.py:2030`` reads ``decorator.keywords[0].value``.
* ``allo/ir/builder.py:2227-2236`` iterates ``node.body`` **directly, with no
  ``ast.walk``**. A kernel must therefore be a *direct child* of the region
  function's body.

``@df.region()`` does execute the function once (``allo/dataflow.py:601``,
inside a bare ``except Exception: pass``) but with **zero arguments**, so a
parameterised region like ``tinytpu_isa`` raises ``TypeError`` before its body
runs. The resulting ``func.mappings`` is ``{}`` and is never read --- only
``hasattr(obj, "mappings")`` at ``builder.py:2821`` uses it as a marker. The
eager execution is vestigial and is **not** the discovery mechanism, which is
why no factory or registration trick can work.

Declaration order is consumed in four places, not one:
``builder.py:2224`` (build order fixes MLIR op order), ``ir/utils.py:171-181``
(``get_all_df_kernels``, no sort and no dependence analysis),
``dataflow.py:482,533-549`` (``_build_top`` emits the ``CallOp``\ s in that
order and tags only the last with a ``last`` attribute) and
``backend/hls.py:308,317`` (rendered into ``kernel.cpp`` in op order, where
csim executes them textually). Issue ``#20``.

.. list-table:: The six candidate splits
   :header-rows: 1
   :widths: 28 14 58

   * - Move
     - Verdict
     - Mechanism
   * - ISA encoding and constants to ``isa.py``
     - **safe**
     - ``OP_*``, ``T``, ``MAXDIM``, ``AGU_*`` are plain ints and reach the
       region through the module's ``__globals__``
       (``allo/ir/utils.py:48``) whether defined there or imported.
       ``mapping=[T, T]`` still ``eval``\ s against ``global_vars``
       (``builder.py:2030-2033``), which an imported ``T`` satisfies.
   * - Reference model
     - **already done**
     - ``isa_ref.py``, 70 lines of numpy, imports from ``microarch_isa``.
   * - PE bodies to module-level functions the kernel calls
     - **impossible**
     - Allo does follow a module-level helper cross-module
       (``infer.py:1316,1323`` → ``builder.py:3381``) but builds it as a
       separate ``func.func`` plus ``CallOp``, **not a kernel**: it gets no
       ``df.kernel`` attribute, so ``get_all_df_kernels`` never makes it a
       process. Worse, its context is ``ctx.copy()``
       (``builder.py:3376``), which does **not** propagate ``ctx.scopes``
       (``visitor.py:126-151``) --- contrast the kernel path's explicit
       ``ctx.scopes = old_ctx.scopes`` at ``builder.py:2024,2057``. The moved
       body could not see ``wcol``, ``acol``, ``p_fwd`` or ``wq``, and
       ``move_stream_to_interface`` (``dataflow.py:61,286-292``) would not
       lift its stream ops. Separately, ``df.get_pid()`` is pattern-matched
       on an ``ast.Assign`` (``infer.py:375-385``) and
       ``allo.meta_if`` / ``meta_for`` are matched as ``ast.With`` on
       ``context_expr.func.attr`` (``infer.py:1488-1563``,
       ``builder.py:3818-3894``), with ``meta_for`` a source-level unroll.
       ``wld`` and ``pe`` are saturated with both.
   * - PEs into a factory invoked inside the region
     - **impossible**
     - Nothing converts a *returned* function object into a kernel. The
       ``@df.kernel`` runtime wrapper (``dataflow.py:562-574``) calls
       ``build(funcs=[func])``, and ``funcs=`` is not a parameter of
       ``build`` (``dataflow.py:636-650``), so that path raises
       ``TypeError``. ``pe = make_pe(...)`` in the region body is an
       ``Assign``; the ``@df.kernel`` inside the factory's source is never
       reached by ``builder.py:2228``.
   * - ``schedule()`` to ``schedule.py``
     - **safe**
     - Takes ``s`` and uses *string* handles (``"sequencer_0:ib"``,
       ``"accu_0:x"``). Safe to move the function; **not** safe to rename a
       kernel, change a ``mapping``, or move a loop into a subfunction,
       because the handles are those names.
   * - Host / bench entry
     - **already done**
     - No ``__main__`` in the file.

Two hazards that are easy to trip
---------------------------------

**The ``Stream`` / ``UInt`` globals hazard.** ``get_global_vars``
(``allo/ir/utils.py:31-98``) seeds from ``_func.__globals__`` at ``:48``, but
the object handed to ``df.build`` is the wrapper defined at
``allo/dataflow.py:608``, so the seed is **``allo/dataflow.py``'s** module
dict, which contains neither ``Stream`` nor ``UInt``. The frame walk at
``:55-67`` admits only ``int``, ``float``, ``AlloType`` and functions, so
ints (``T``, ``OP_*``) and ``AlloType`` instances (``int8``, ``int32``)
survive --- but **class-valued globals like ``Stream`` and ``UInt`` do not**.
They arrive only through the worklist at ``:95-96``, i.e. only because some
plain function *defined in* ``microarch_isa`` is reachable from the caller's
namespace, which today it is (``bench_isa.py:28-31``, ``cosim.py:65-67``,
``stress_isa.py:52-58`` all import ``assemble`` / ``expand`` / ``enc`` /
``schedule``). A split that stops doing that makes ``Stream[...]`` and
``UInt(64)`` annotations stop resolving. Keep at least one plain function
from the region's module in every caller's namespace, and add a test that
asserts it.

**``mutate.py`` mutates this file as text, and this is the real blocker.**
``mutate.py:50`` pins ``SRC = microarch_isa.py`` and ``:52``
``MOD = "...microarch_isa"``; ``:191`` asserts ``src.count(anchor) == 1``;
``:215`` writes a single mutated ``microarch_isa.py`` into
``.mutants/<name>/`` and ``:157-167`` shims it in as ``sys.modules[MOD]``.
All **34 anchors** resolve inside this one file: 31 in the region body
(``:589-1130``), plus ``check_program`` (``:1423``), ``assemble``
(``:1521``) and the constant ``AR_RAW_DIST`` at ``:501``. Move any anchored
line to a sibling module and the ``count(anchor) == 1`` assert fires; fix
that and the mutant directory still contains only ``microarch_isa.py``, so
the real sibling is imported and **the mutation silently does not apply** ---
a mutant that passes for the wrong reason, which is the worst possible
failure of a verification harness.

``mutate.py`` therefore needs a per-file anchor map **before** any split,
including the "safe" extraction of constants, because ``AR_RAW_DIST`` is one
of the anchors.

Target layout
-------------

``examples/accelerator/tinytpu_vitis/tinytpu/``

* ``isa.py`` --- opcodes, field widths, ``enc``, ``enc_agu``, the spare-bit
  rule as a table and an assertion rather than a paragraph.
* ``programs.py`` --- ``gemm_program_handwritten``, ``gemm_program_flat``,
  ``vadd_program``, ``expand``.
* ``assembler.py`` --- ``assemble``, ``check_program``, ``ProgramError``,
  ``_trace`` renamed ``trace``.
* ``region.py`` --- the ``@df.region()`` function and its eight kernels,
  unchanged in order and in body, all still lexically inside it. The module
  docstring is one line naming the design page. Each kernel keeps a
  **one-line** docstring naming its role. ``mutate.py``'s anchors follow this
  file.
* ``schedule.py`` --- ``schedule``.
* ``reference.py`` --- absorbs ``isa_ref.py``.
* ``__init__.py`` --- re-exports the names the harness imports today, so
  ``bench_isa.py``, ``stress_isa.py``, ``mutate.py``, ``cosim.py``,
  ``kpn_model.py`` and every ``chia_agent`` module keep working unchanged in
  step 1.

Where the prose goes
--------------------

The 264-line module docstring is really 16 blocks and they do not all go to
the same place. Every one of the 33 prose blocks over 10 lines was checked
against ``docs/source/``:

.. list-table::
   :header-rows: 1
   :widths: 24 10 66

   * - Class
     - Blocks
     - Destination
   * - DESIGN RATIONALE
     - 12
     - ``designs/tinytpu_isa.rst``, which already has *Architecture*
       (``:52``), *Chains rather than fan-out* (``:119``), *One owner per
       memory* (``:196``), *Data type* (``:231``), *The ISA* (``:242``),
       *Instruction format* (``:245``), *Hardware parameters* (``:482``),
       *Verifying a change* (``:658``) and *The accumulator's dependence
       claim* (``:778``).
   * - EXPERIMENT HISTORY
     - 11
     - ``designs/tinytpu_history.rst``, which already holds the row-flattening
       pass (``:399-505``), the ``wrap_io`` cost (``:572-591``), the strided
       vs contiguous burst table (``:594-615``), the 5-shape table
       (``:717-793``) and the accumulator work (``:828-916``).
   * - ALLO LIMITATION
     - 5
     - a one-line comment naming the issue, e.g.
       ``# issue #20: dma_st declared last``. One line, not an explanation ---
       the only in-code prose rule (d) should permit.
   * - TOOLCHAIN / BUILD lore
     - 2
     - ``backends/vitis.rst:338-396`` (already canonical for
       ``align_value``) and ``dev/toolchains.md``.
   * - ISA SPEC / encoding table
     - 3
     - stays, as **data**: the bit layout becomes a table of ``(name, lo,
       hi)`` tuples in ``isa.py`` with an assertion that the fields tile the
       word and do not overlap, and the ``imem[0..7]`` header table becomes a
       named constant list in ``assembler.py``. A table that the code reads
       cannot go stale the way a comment describing it can.
   * - REDUNDANT
     - 9 one-liners
     - deleted: the repeated ``# advanced at the TOP: see the II note``
       (``:744,801,855,948,1029,1117``), ``:748``, ``:831-832``,
       ``:1121``.

**Prose with no home elsewhere: three blocks, and they must be lifted rather
than deleted.**

1. ``:1239-1263`` (``expand``'s docstring) --- the assembler-to-sequencer
   contract: the header must carry *dynamic* work counts, and a unit promised
   more work than it receives hangs. ``tinytpu_isa.rst:350-388`` covers it
   less precisely. This is the one genuine cross-unit invariant in the design.
2. ``:1482-1510`` (``assemble``'s docstring) --- the ``imem[0..7]`` header
   layout, more precise than the page.
3. ``:1319-1354`` (``check_program``'s docstring) --- the per-dynamic-issue
   enumeration of what the checker proves. ``tinytpu_isa.rst:658-704`` is
   thinner.

A fourth is a provenance problem rather than a content one:
``isa_ref.py:10-11`` records that the reference model was written "from the
ISA comments and the unit docstrings, not from the unit bodies". Deleting
those comments removes the artefact the independent golden model was derived
from. The differential value of ``isa_ref.py`` depends on its source being
*not* the implementation, so the ISA semantics must survive somewhere
independent of ``region.py`` --- which is what moving them to
``designs/tinytpu_isa.rst`` and to ``isa.py``'s field table accomplishes,
provided it happens before the deletion and is recorded in
``isa_ref.py``'s one-line docstring.

Four contradictions inside the file to settle while moving prose:

* ``:229-234`` says ``dma_st`` is "still strided at II=4 … the next thing to
  measure"; ``:255-258``, 22 lines later, says ``align_value`` took it from
  II=4 to II=1. ``tinytpu_history.rst:1088-1115,1228-1233`` settles it and
  marks the item done. ``:229-234`` keeps a retired plan alive.
* ``:212-227`` presents a superseded 5-shape table (680 / 831 / 1066 / 1139 /
  1457) with no marker; the shipped design is 172 / 262 / 418 / 484 / 686.
* ``:279`` and ``:471`` call ``nr`` 8 bits; ``:302-303`` calls it 7 bits.
  Both describe one 8-bit field with 7 usable bits. The field table fixes
  this by construction.
* ``_KB`` (``:478``) and ``KB_MAX`` (``:494``) are the same expression as
  ``WPR`` (``:496``). Three names, one value, two of them dead.

And one duplication that stays, deliberately: ``kpn_model.py:42-79``
re-implements the sequencer's dispatch rules, including the ``T + 1`` rewrite
for ``spm``'s ``mm`` and the ``2 * nr`` rewrite for ``accu``'s ``vadd``. That
is the point of a differential model. It must be *stated* that the rules live
in two places and change together --- one line in each, pointing at the
other.

Verification, per step
----------------------

The design must stay bit-exact. Each step is verified by the same gate, run
before and after, with identical output:

.. code-block:: bash

   source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
   cd examples/accelerator/tinytpu_vitis
   ./reproduce.sh              # bench, stress, validator, KPN, 5-shape cosim
   python mutate.py --no-rtl   # the mutants: NOT in reproduce.sh today

The validator and the KPN model are covered transitively ---
``stress_isa.py:411`` runs ``validator_controls`` and ``:423`` runs
``kpn_model.run`` per program --- and the 5-shape cosim counts are pinned at
``reproduce.sh:28``. The mutants are not covered at all, which is why adding
them is a prerequisite rather than a nicety.

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Step
     - Risk
     - Verification
   * - 0. ``mutate.py`` gains a per-file anchor map
     - **high**
     - No design change at all, but it is the step that makes every later
       step verifiable. Verify by running the current 34 mutants through the
       new map and requiring the same 34 verdicts, then by deliberately
       breaking one anchor and requiring a **loud** failure rather than a
       pass.
   * - 1. Move ``isa_ref.py`` into the package
     - low
     - ``reproduce.sh`` bit-identical
   * - 2. Extract ``isa.py`` (constants, ``enc``, ``enc_agu``, the field
       table)
     - **medium**
     - encoding is pure Python, but ``AR_RAW_DIST`` is a ``mutate.py``
       anchor and the field table replaces three prose statements: run the
       mutants, then ``reproduce.sh``
   * - 3. Extract ``programs.py``, ``assembler.py``
     - **medium**
     - ``check_program`` and ``assemble`` each hold a ``mutate.py`` anchor
       (``:1423``, ``:1521``). The validator's own rejections must be
       unchanged, mutant for mutant.
   * - 4. Extract ``schedule.py``
     - **medium**
     - schedule primitives change the emitted pragmas, and the handles are
       kernel *names*: diff the generated ``kernel.cpp`` byte for byte, then
       the 5-shape cosim
   * - 5. Move the region to ``region.py``
     - **high**
     - the frontend re-parses this file and 31 mutant anchors live in it;
       add an explicit test that ``Stream`` and ``UInt`` still resolve
       (the globals hazard above), diff the generated ``kernel.cpp`` byte for
       byte, then bench, stress, mutants and the 5-shape cosim
   * - 6. Delete the prose
     - low, once steps 0-5 pass
     - no behaviour change; the docs-reference guard is what checks it

Step 5 is the only one that can change generated code, and it must be landed
alone. Step 0 is not optional: without it, steps 2, 3 and 5 can produce
mutants that pass because the mutation never applied.

The harness
-----------

Twenty duplication clusters were found. The rule is one definition, one place,
imported everywhere else. The ones that matter:

.. list-table::
   :header-rows: 1
   :widths: 26 10 34 30

   * - What
     - Copies
     - Where the copies are
     - Canonical
   * - GEMM golden model
     - 6
     - ``isa_ref.py:39-70``, ``stress_isa.py:115-121``,
       ``bench_isa.py:75-78,94-95``, ``cosim.py:106-109``,
       ``impact/bench_variant.py:35-47``
     - ``stress_isa.gemm_gold`` for the formula, ``isa_ref.run`` for ISA
       semantics. The two are already asserted equal per case at
       ``stress_isa.py:378`` and ``cosim.py:165``; the other four are
       unchecked.
   * - Operand generator
     - 4
     - ``bench_isa.py:43-49``, ``cosim.py:99-102``, ``stress_isa.py:85-87``,
       ``impact/bench_variant.py:16-18``
     - ``stress_isa.operands``. This distribution is the Gemmini comparison's
       premise; four copies is four chances to break the comparison silently.
   * - ``imem`` packing
     - 6
     - ``bench_isa.py:52-56``, ``stress_isa.py:124-130``,
       ``cosim.py:103-105,186-188``, ``impact/bench_variant.py:21-25``,
       ``impact/ar_distance_probe.py:31``
     - one ``imem_of`` beside ``assemble``
   * - The 5-shape list
     - 7
     - ``bench_isa.py:40``, ``stress_isa.py:64``, ``cosim.py:72``,
       ``chia_agent/evaluate.py:122``, ``test_harness.py:86``,
       ``accept.py:63-75``, ``impact/bench_variant.py:14``
     - ``bench_isa.SHAPES`` --- which ``kpn_model.py:253`` and
       ``isa_dsl.py:418`` already import correctly
   * - The published cycle counts
     - 5
     - ``reproduce.sh:28``, ``accept.py:75``, ``test_harness.py:85``,
       ``swarm.py:75,84``, ``chia_agent/README.md``
     - one JSON file, or ``accept.BASELINES`` --- the only copy keyed to the
       blobs it was measured on
   * - Cosim cycle parsing
     - 5
     - ``cosim.py:259-268``, ``evaluate.py:383``, ``accept.py:243,271``,
       ``reproduce.sh:75``
     - make ``cosim.main`` emit one machine-readable line or a
       ``results.json``; ``reproduce.sh``'s positional ``awk`` is the most
       fragile consumer
   * - ``subprocess`` + ``killpg`` runner
     - 2
     - ``evaluate.py:248-264``, ``accept.py:86-100``
     - ``evaluate.run``
   * - bubblewrap wrapper
     - 2
     - ``evaluate.py:237-245``, ``accept.py:116-124``
     - ``evaluate.sandboxed``
   * - Nonce-vouched gate call
     - 2
     - ``evaluate.py:282-295``, ``accept.py:103-113``
     - one of them. **This is the security-critical primitive and it exists
       twice.**
   * - ``LLVM_BUILD_DIR`` / ``OMP_NUM_THREADS`` setup
     - 8, in 4 styles
     - ``reproduce.sh:34``, ``impact/env.sh:3``, ``evaluate.py:136``,
       ``accept.py:52``, ``loop.py:49``, ``test_harness.py:65``,
       ``evaluate.py:224`` (``env_for``), ``accept.py:180``
     - ``evaluate.env_for`` for children, ``reproduce.sh`` for shells.
       ``mutate.py:188`` and ``impact/pyrun.py`` set **neither** and fail
       unless the caller exported them.
   * - ``git show HEAD:<file>``
     - 5
     - ``evaluate.py:148``, ``accept.py:163,168,291``, ``allo_tool.py:165,205``,
       ``test_harness.py:162``, ``loop.py:227``
     - ``evaluate.git_show``
   * - Clock target 3.33 ns
     - 5
     - ``cosim.py:235`` (authoritative, the TCL), ``evaluate.py:124``,
       ``accept.py:284``, ``loop.py:76``, ``allo_tool.py:317``
     - ``evaluate.TARGET_NS``, cross-checked against the TCL as
       ``evaluate.parse_synth:344`` already does and ``accept.py`` does not

Dead or frozen, to delete or to mark:

* ``impact/tb_shape.py`` --- entirely unreferenced. Its docstring claims
  ``profile.sh`` uses it; ``profile.sh:21-22`` runs only ``analyze_df.py`` and
  ``rle_df.py``. Delete.
* ``microarch_isa.py:478`` ``_KB`` and ``:494`` ``KB_MAX`` --- defined, never
  read. Delete.
* ``impact/make_variants.py`` (774 lines) --- regenerates from
  ``git show 7a24c21e:microarch_isa.py`` and emits programs against ``A_SP``,
  a symbol ``main``'s design no longer defines. Keep as evidence of how the
  attribution table was produced, move to ``dev/records/`` and mark frozen at
  ``7a24c21e``.
* ``impact/ar_distance_probe.py`` --- superseded in-tree by
  ``cosim.py TPU_TB=stress`` and ``mutate.py``'s ``ar_claim_false``.
* ``impact/analyze_df.py:13`` and ``impact/rle_df.py:6`` share an identical
  monitor regex and status-CSV parse; keep one.

Contradictions in the harness to settle while de-duplicating:

* **Gate timeouts disagree three ways.** ``evaluate.py:128`` gives
  ``stress_isa`` 240 s; ``mutate.py:154`` gives it 480 s; ``accept.py:83``
  uses 600 s for gates and 7200 s for cosim. A candidate the evaluator kills
  as a deadlock still passes under ``mutate``.
* **Run counts.** 60, 486 and 492 stress runs all appear in agent-visible
  prose (``evaluate.py:27``, ``allo_tool.py:189,305``,
  ``chia_agent/README.md:105,299,364``). The code is right ---
  ``evaluate.py:306`` deliberately pins no number --- only the prose is
  stale.
* **Parametricity is stated but not tested.** ``spec_policy.py:313`` requires
  ``T`` to stay parametric; ``evaluate.py:115`` concedes "main's design
  supports T=4 alone"; ``PARAM_CONFIGS`` (``:117``) sets only
  ``TPU_MAXDIM``, so ``param_check.py:58``'s ``TPU_T`` read never fires.
* ``bench_isa.py:11`` claims it sweeps "every shape up to MAXDIM"; it runs
  five fixed shapes (``:40``). The 64-shape sweep is
  ``stress_isa.py:65``.

And ``reproduce.sh`` does **not** run the mutants
(``grep mutate reproduce.sh`` is empty), even though ``bench_isa.py:17`` and
``stress_isa.py:9`` both cite ``mutate.py`` as the evidence their gate
catches a broken design. It also does not run ``cosim.py TPU_TB=stress`` or
``param_check.py``. Since the mutant set is the verification this plan leans
on, ``reproduce.sh`` gains a step 3b (``python mutate.py --no-rtl``, about
five minutes) **before** the refactor starts, not after.


3b. Test and verification structure
===================================

Four categories exist and only three of them are named today.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Category
     - Where it lives
     - Definition
   * - Unit test
     - ``tests/*.py``, ``tests/dataflow/``
     - Asserts Allo's *intended* behaviour. Collected by ``pytest``. Fast,
       hermetic, no EDA tool.
   * - Limitation repro
     - ``tests/limits/``
     - Asserts Allo's *current* behaviour, one file per claim, printing one
       ``[item N] <STATUS>`` line. Deliberately **not** named ``test_*.py``
       so ``pytest`` does not collect it, because a repro that starts
       passing is news, not a failure. Right place; wrong names (should be
       issue numbers) and no runner.
   * - Integration gate
     - ``examples/accelerator/tinytpu_vitis/reproduce.sh``,
       ``mutate.py``, ``chia_agent/evaluate.py``
     - Asserts a *design* is unchanged. Slow, needs Vitis. Not
       ``pytest``-collectable and should not be.
   * - Evidence
     - ``impact/results/``, ``logs/``, ``chia_agent/evidence/``,
       ``docs/source/records/``
     - A record of what a run printed on a date. Never re-run, never
       asserted. Moves to ``dev/records/``.

Two defects in the current arrangement:

* ``tests/test_backend_utils.py`` (161 lines, fork-only) duplicates
  upstream's ``tests/utils/test_backend_utils.py`` (166 lines). They share
  two of six test names and diverge on the rest. The fork's four extra cases
  merge into upstream's file and the fork's file is deleted --- otherwise the
  next upstream merge conflicts on a file upstream does not know exists.
* **``align_value`` has no test anywhere.** ``grep -rn align_value tests/``
  is empty. It is a promise to Vitis whose violation produces wrong RTL while
  every software simulation passes --- exactly the class that needs a test
  most. It belongs in ``tests/utils/test_backend_utils.py`` beside
  ``postprocess_hls_code``, which is the function that emits it.

Our Allo changes are otherwise correctly placed: ``s.dependence`` is tested
in ``tests/test_vhls.py`` (emission ``:799``, rejection ``:852``,
dataflow-region ``:870``) and nothing depends on an ``examples/`` file to
prove a library change works.

One more, bordering on part 1: **every ``microarch_isa.py`` line reference in
``limitations.rst`` is stale** --- items 5, 14, 15 and 17 all cite lines that
have since moved, some of them into docstrings. The migrated issues carry a
correction. The lesson for the new tracker is that an issue should anchor on
a *symbol*, not a line.


4. A compiler extension pattern
===============================

Two extensions exist. One follows a pattern; the other does not, and the
difference is instructive.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``s.dependence`` (``bbea2af0``)
     - ``align_value`` (``b4be2b10``)
   * - Surface
     - a ``Schedule`` method, ``customize.py:834``
     - a key of the ``configs`` dict, read at ``hls.py:362``
   * - Validation
     - five explicit ``AlloValueError`` raises, ``customize.py:880-899``
     - none
   * - IR carrier
     - a ``dependence`` ``ArrayAttr`` of ``DictAttr`` on the loop,
       ``customize.py:943``
     - none: a regex rewrite of emitted text, ``vitis.py:381,418``
   * - Emitter
     - ``EmitVivadoHLS.cpp:2603-2647``, one branch in
       ``emitLoopDirectives``
     - ``postprocess_hls_code``, ``vitis.py:381``
   * - Test
     - ``tests/test_vhls.py:799,852,870`` --- emission, rejection,
       dataflow-region case
     - **none**
   * - Docs
     - autodoc'd onto ``api/index.rst`` from the docstring, and nowhere
       else --- ``dive/dataflow.rst:452-462``, the hand-written
       "Customization (Schedule Primitives)" section, shows only
       ``s.partition``
     - prose on ``backends/vitis.rst:347-435``; invisible on the API page,
       because it is a ``configs`` key rather than a ``Schedule`` member

``s.pipeline(style=)`` on ``tinytpu-align`` (``0038833c``) repeats the
``s.dependence`` shape exactly, in 44 lines across three files with one test.
That is the pattern; it just is not written down.

The pattern
-----------

A new schedule primitive is **one commit touching exactly five places**:

1. **The primitive** --- a ``@wrapped_apply`` method on
   ``allo/customize.py:Schedule``. Its signature is the whole user surface.
   Every argument is validated against a closed set, raising
   ``AlloValueError`` with the argument's name and value in the message,
   *before* any IR is touched. One canonical spelling; no alternative form.
2. **The IR attribute** --- a named attribute on the MLIR op the primitive
   targets, set in the same method. A loop directive goes on the loop, beside
   ``pipeline_ii`` and ``rewind``. A structured claim is a ``DictAttr``; a
   repeatable claim is an ``ArrayAttr`` of them, appended to rather than
   overwritten.
3. **The emitter branch** --- one ``if (auto x = getLoopDirective(op, "..."))``
   in the target emitter's directive function, which ``emitError``s rather
   than emitting nonsense when the attribute is malformed or names something
   undeclared. Other emitters that cannot honour it must reject it, not ignore
   it (issues ``#23``, ``#24``, ``#31`` are all "ignored instead of
   rejected").
4. **The test** --- in ``tests/test_vhls.py`` for a Vitis directive,
   ``tests/test_catapult_hls.py`` for Catapult, beside the primitive's
   siblings, never in ``examples/``. Three cases, as ``s.dependence`` has:
   the pragma appears with the right text; every invalid argument raises;
   the primitive works inside a ``@df.region``.
5. **The docs** --- the docstring, because ``api/index.rst`` is
   ``autoclass:: allo.customize.Schedule :members:`` and publishes it. The
   docstring states the emitted pragma, the closed set of each argument, and
   **what is a promise rather than a fact** --- ``s.dependence`` and
   ``align_value`` are both claims that, if false, produce wrong RTL while
   every software simulation passes. A backend page is where the *flow*
   around it is explained, not where the primitive is defined.

A backend configuration key (``align_value``, ``hbm_mapping``,
``sub_funcs``) follows the same five steps with (1) becoming a validated key
in one place rather than scattered ``configs.get`` calls, and (5) becoming a
table of keys on the backend's page **plus** the API page. ``align_value``
currently satisfies neither; ``tests/test_backend_utils.py`` is the file its
test belongs in.

What CAKE says about this
-------------------------

*CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution* (Ye et al.,
NVIDIA + CMU, `arXiv:2608.12629 <https://arxiv.org/abs/2608.12629>`__, August
2026) is the closest published work. Its finding: given the same agent and
the same token budget, agents authoring a typed, hardware-explicit schedule
IR reached **1.144x** a tuned expert baseline, while agents authoring raw
CUDA/PTX reached **0.928x** --- worse than the baseline. A constrained typed
surface beat an unconstrained one.

Four of its design principles are directly actionable here, and three are
already what the pattern above says:

* **Canonical form.** "Prefer one canonical form for each operation over
  equivalent alternative spellings." Two spellings double the agent's search
  space and halve each analysis's coverage.
* **Statically type-checked at construction.** "Use typing rules to constrain
  operation lowering and reject ill-typed programs during construction" ---
  in Allo that means the Python builder and ``customize.py``, not the MLIR
  verifier after lowering. Step 1 above.
* **Analysis-consistent.** "Accompany changes to the IR data model with
  corresponding analysis updates", and, from the paper's body, "a primitive
  and its analyses must evolve together: syntax without effects and legality
  rules makes the IR less analyzable." This is the strongest argument for
  making steps 1-5 one commit rather than five.
* **Hardware-grounded.** "Document the intended hardware behavior of each
  operation" --- an obligation on the primitive, which is why step 5 lives in
  the docstring.

CAKE also separates its checks into three dispositions --- **gate** (cheap,
static, pre-compile), **execution gate** (numerical comparison against an
authoritative external reference, over several shapes and input
distributions) and **non-blocking report or hint** (cost model, profiling)
--- and filters candidates through the static gates *before* spending
hardware time. Our harness already has this shape: ``param_check``/validator
as the static gate, ``isa_ref`` and the 5-shape cosim as the execution gate,
the KPN model as the ranking report. Naming the three tiers explicitly in
``chia_agent`` would make it legible.

Its loop also **routes each failure to a layer**: a runtime crash becomes a
verifier rule, a recurring illegal lowering becomes a static check, a cost
mispredict becomes a calibration target. That is the same move as this plan's
part 1 --- a limitation becomes a labelled issue plus an executable repro ---
and it argues for keeping ``tests/limits/`` as a growing corpus that every
Allo change re-runs.

CAKE says **nothing** about file layout, one-primitive-per-file conventions,
or how an IR schema is put in front of a model. Those are not attributable to
it. Neither is the claim that Allo should adopt a typed IR distinct from the
one it has.

Also relevant and verifiable: *Magellan* (`arXiv:2601.21096
<https://arxiv.org/abs/2601.21096>`__), which synthesises executable C++
decision logic for a real LLVM inlining pass, is the closest work to
LLM-*generated* compiler components as opposed to LLM-*authored* schedules.


5. Sequencing
=============

Serial spine
------------

0. Add ``python mutate.py --no-rtl`` to ``reproduce.sh``, give ``mutate.py``
   a per-file anchor map, and make ``cosim.main`` emit a machine-readable
   cycles line that ``reproduce.sh``, ``evaluate.py`` and ``accept.py`` all
   read. The refactor's verification must exist, and must survive a file
   split, before the refactor does.
1. Re-scope the CHIA documentation guard (part 1), and in the same change
   guard ``evaluate.compose``'s ``ast.parse``. **Nothing else can start until
   this lands**: every subsequent step deletes prose, and the guard rejects
   that today.
2. Add the missing docs content the register and the docstrings will stop
   carrying (part 1 step 3, part 3 "where the prose goes"). Prose must arrive
   at its destination before it leaves its source.
3. Reduce ``limitations.rst`` to a pointer.
4. ``microarch_isa.py`` steps 1-4 (encoding, programs, assembler, schedule).
5. ``microarch_isa.py`` step 5 (the region), alone, with a byte-for-byte
   ``kernel.cpp`` diff and the full gate.
6. Delete the prose that is now duplicated.

Every one of steps 3-6 touches a file in ``DESIGN_EVALUATOR`` or changes a
``microarch_isa.py`` / ``isa_dsl.py`` blob, so each one also bumps
``evaluate.MAIN_BASE`` and records a fresh ``accept.py`` control run. That is
the per-step cost of the frozen-reference design, and it is the reason these
steps are serial rather than parallel.

In parallel with the spine
--------------------------

* Closing issues for the fixed register items (part 1 remaining step 1).
  Independent of everything.
* The ``dev/`` tree and the page moves (part 2). Touches no code.
* Writing the two missing repros (``#26``, ``#34``) and
  ``tests/limits/run_all.py``.
* The ``align_value`` test in ``tests/test_backend_utils.py`` and its entry
  on ``api/index.rst`` (part 4).
* Deleting ``tests/test_backend_utils.py`` in favour of upstream's
  ``tests/utils/test_backend_utils.py``, into which our four extra cases
  merge.
* Harness de-duplication, once the canonical copies are chosen.
* The four factual fixes listed under *Where we are*.
* Deleting ``impact/tb_shape.py``, ``microarch_isa._KB`` and
  ``microarch_isa.KB_MAX``.
* Settling the three gate-timeout values on one set.

Blocked on Allo fixes
---------------------

Splitting the region body across modules --- the thing that would most help
an LLM maintain this design --- is blocked on issue ``#20`` (declaration
order) and issue ``#35`` (a sub-region type-checked against the caller's
globals). Those two are the highest-leverage entries in the whole register
for the LLM-maintainability goal, and they are both estimated at tens of
lines. Issue ``#30`` (``customize()`` calls ``sys.exit(1)``) is third: no
agent loop can catch a build failure while it stands.


Open questions for the owner
============================

Each of these removes or relocates something whose destination is a judgement
call, and none should be executed without a decision.

1. **Where do dev notes live?** This plan proposes a ``dev/`` tree in this
   repository rather than the GitHub wiki, on the grounds that a dev note
   must be changeable in the same commit as the code it describes. The wiki
   is a separate repository and cannot do that. If the preference is the
   wiki, the ``dev/`` tree becomes a stub of links and the atomicity is lost.
2. **Do the closed register items become closed issues, or are they
   deleted?** Ten entries are FIXED, CANNOT-REPRODUCE, NOT-A-LIMITATION or
   retracted, and six of the fixes are fork-only upstreaming candidates. As
   closed issues they stay searchable and carry the upstreaming list; deleted,
   the upstreaming candidates need another home. This plan assumes closed
   issues.
3. **Three prose blocks have no home yet** and will lose information if
   deleted before one is written: the assembler-to-sequencer dynamic-count
   contract (``microarch_isa.py:1239-1263``), the ``imem[0..7]`` header
   layout (``:1482-1510``), and what ``check_program`` proves
   (``:1319-1354``). They need to be written into
   ``designs/tinytpu_isa.rst`` at greater precision than it currently has.
   That is new writing, not moving, and it is the only part of this plan that
   is.
4. **``isa_ref.py``'s independence.** The reference model was derived from
   the ISA comments, not the unit bodies (``isa_ref.py:10-11``). Deleting
   those comments removes its provenance. Is it acceptable for the ISA
   semantics to live in ``designs/tinytpu_isa.rst`` plus ``isa.py``'s field
   table, or should a machine-readable ISA specification become the single
   source that both ``region.py`` and ``isa_ref.py`` are checked against?
   The second is more work and is the better answer for rule (e).
5. **How much may ``examples/`` shrink?** Moving ``logs/`` (579 KB, 23 run
   logs), ``impact/results/`` (41 KB) and ``chia_agent/evidence/`` (202 KB)
   out of ``examples/`` and into ``dev/records/`` is a 822 KB relocation of
   committed evidence. It is the right classification, but it changes every
   docs link that cites a log.
6. **``impact/make_variants.py``** (774 lines) is frozen at ``7a24c21e`` and
   emits programs against a symbol the design no longer defines. Keep it as
   evidence of how the gap-attribution table was produced, or delete it and
   rely on ``designs/gemmini_comparison.rst``? This plan assumes keep-and-mark.
7. **Does the CHIA loop keep the frozen-reference design?**
   ``evaluate.MAIN_BASE`` and ``accept.BASELINES``-by-blob-id make every
   design-file edit a two-commit operation. That is exactly the property that
   makes the loop's results trustworthy, and exactly the property that makes
   this refactor expensive. Keeping it is defensible; it should be a decision
   rather than an accident.
8. **Does ``s.dependence`` belong on ``dive/dataflow.rst``?** It is published
   only through ``autoclass`` on ``api/index.rst``. The hand-written
   primitives section at ``dive/dataflow.rst:452-462`` lists only
   ``s.partition``. Either that section becomes a complete list, or it should
   say it is not one.
9. **The four factual defects** listed under *Where we are* are corrections
   to published claims. Fixing ``designs/tinytpu_isa.rst:44`` changes a
   headline sentence, so it wants an author's eye rather than a mechanical
   edit.
