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

#################################
ACT, and Why We Cite It
#################################

.. note::

   **ACT is Kai Shao's work**, developed at https://github.com/kkkaishao/allo
   (the ``kai`` remote), branch ``act``. This fork's ``chia-codesign`` lineage
   imports it at commit ``3c1ad38d`` as merge ``29cb1d99``; ``ATTRIBUTION.md``
   at tag ``chia-codesign-final`` (``629c2767``) records the terms. This page
   audits that work **from the outside** to decide whether to connect it to
   :doc:`/designs/tinytpu_isa`. It cites ACT; it does not republish it. ``main``
   has no ACT sources.

Audited 2026-09-22 against ``chia-codesign-final``: ``allo/exp/dsa/`` is 12
files, 8,347 lines.


Verdict
=======

**Cite ACT as related work. Do not integrate it.** Reuse its *interface* --
a loop nest as a tuple of ``Loop(rank, factor, level, spatial)`` -- and nothing
below it. Three reasons, each measured rather than argued:

1. **The mapper is dead code, and so is the layer under it.** Not just
   ``mapspace.py``: no code anywhere in the tree ever constructs the
   ``Binding`` that every mapper entry point requires.
2. **It cannot be run here.** Its bindings need a CIRCT build and an LLVM commit
   that do not exist on this host, so no test figure about it is citable.
3. **Its cost model is not a measurement.** ``Priced.cost`` is
   ``(makespan, emits)`` and nothing models instruction memory. A loop scored on
   it would optimise a model.

What is worth taking is small and already prototyped:
``examples/accelerator/tinytpu_vitis/act_nest.py``.


What ACT Is
===========

An accelerator ISA is declared with decorators, and a TOSA program is compiled
onto it.

- ``@isa.instruction(src, dst)`` on ``def <mnemonic>(I)``. Inside,
  ``@I.access`` (a strided / contiguous / view / layout pattern) and
  ``@I.compute`` (a TOSA DAG) are **both required** -- omitting either raises
  ``AcceleratorDescriptionError`` (``core.py:967-970``). ``@I.schedule``
  (a finite domain plus a legality predicate) and ``@I.expand`` are optional.
- ``ISA.buffer``/``global_``/``scalar``/``vector``/``tile``/``hbm`` declare the
  memory hierarchy; ``ISA.bind``, ``latency``, ``network``, ``configures`` tie
  instructions to ``@unit`` kernels.
- ``ISA.compile_program(source, mapping_for=None)`` emits an MLIR module whose
  ``func @main`` is a **straight-line** sequence of ``allo.emit`` ops
  (``codegen.py:543-576``; ``@inspect`` points add ``call`` anchors, and the
  inspected buffers are the function's memref arguments). No control flow.
- ``@isa.oracle`` is a functional simulator, JIT-executed through the MLIR
  execution engine.

Two things we previously said about ACT that are **wrong**:

- *"A minimal ACT pilot was run."* No artifact exists in either repo.
- *"ACT has a mapspace search."* It has a mapspace *file*. See below.

The one machine ACT has ever targeted, chia's TinyTPU
(:doc:`/extensions/chia`), is **not a loop machine**: a flat PC walk, no loop
stack, fp32, 4-5 ``i32`` instruction words, 11 ACT-declared instructions. Ours
(:doc:`/designs/tinytpu_isa`) is int8/int32, two packed ``UInt(64)`` words, a
PC plus a 4-deep loop stack and a 3-term AGU. They share no code -- the
branches' merge base ``76130c63`` predates both, and the two ``microarch``
files share only their SPDX header. So ACT emitting flat streams is not an
oversight; ours would be the first loop machine it met.


The Mapper
==========

``mapping.Loop(rank, factor, level, spatial)`` is a frozen 4-field dataclass
(``mapping.py:122-131``) and ``Mapping(ranks, loops, bypass)`` is frozen with the
nest as a persistent field (``mapping.py:147-149``). So the chosen nest **is**
readable back out: ``search(binding, op).best.mapping.loops``. That much of our
standing plan is real.

Everything else about it is worse than we thought.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - claim
     - what the code says
   * - "the mapper is a loop-nest mapper"
     - True, but the **enumeration is entirely in** ``mapspace.py:102-174``.
       ``mapping.py`` contains no enumeration: it parses one given nest
       (``from_timeloop``, 39 lines) and lowers it.
   * - "``mapping._lower`` discards the loops"
     - Understated. ``mapping.py:904`` is
       ``for point in itertools.product(*(range(loop.factor) for loop in nest))``
       -- the nest is **fully unrolled**, temporal and spatial alike, so program
       size is the product of the factors. Below it an operand's address is
       already a single scalar element offset (``mapping.py:966-970``), so the
       multi-dimensional index structure an AGU needs does not exist there at
       all.
   * - "``mapspace.py`` is dead code"
     - True, and it is not alone. ``git grep mapspace`` outside the file itself:
       zero hits, here and on ``kai/act``. One commit ever ("checkpoint",
       2026-08-19). Of ``mapping.py``'s 1,292 lines the only live import is
       ``lower_site`` at ``search.py:2690``, reachable only when ``mapping_for``
       is passed to ``compile_program`` -- which nothing does. **And no code
       anywhere constructs a** ``Binding``, which every entry point requires.
       ``from_timeloop``/``to_timeloop`` have no callers.
   * - "factorisations x permutations x spatial splits"
     - Only with ``instances > 1``. At the default ``instances=1`` there are
       **no spatial slots at all** (``mapspace.py:139-141``), and ``instances``
       is the one number the ISA cannot declare.
   * - "``Priced.cost`` is ``(makespan, emits)``"
     - Confirmed, ``mapspace.py:189-196``. ``emits`` is ``len(program.emits)``
       and is only a lexicographic tie-break. Nothing in ``allo/exp/dsa/``
       models instruction-memory capacity. Pricing a candidate calls
       ``assemble`` -- ACT's **whole** codegen and allocator -- so its ranking
       is unavailable without the machinery we wanted to bypass.

Also: ``_derive`` supports exactly two operations, ``tosa.matmul`` and
``tosa.conv2d`` (``mapping.py:526``), and refuses conv2d with padding or a
non-zero bias.

**Dependencies.** MLIR python bindings are unavoidable for in-place use, three
independent ways: ``core.py:28``, ``search.py:48-49``, and ``search.py:83``
pulling in ``oracle.py``, which imports ``ExecutionEngine`` and ``PassManager``.
``sympy`` and ``ml_dtypes`` come along transitively although the mapper never
uses them. A **port** of the mapper alone -- ``Loop``, ``Mapping``,
``Dataspace``, a re-modelled ``Binding``, ``mapspace``'s enumerator, the
coverage/intrinsic half of ``Mapping.check``, and the pure geometry helpers
``_split_body``/``_weights``/``_span``/``_tile``/``_project``/``_packing``/
``_run_offsets``/``_reduction_ranks`` (~105 lines of the last group) -- is on
the order of **250-350 lines** and would be free of MLIR, sympy, ml_dtypes and
the JIT. That is the honest size of "use ACT as a mapper only".


It Cannot Be Run Here
=====================

``scripts/act-test-recipe.sh`` probes every precondition and prints a verdict.
As of 2026-09-22 four are missing:

- ``allo/_mlir`` must be built from *this* tree: ACT reaches
  ``allo._mlir.schedule`` (``mlir/python/allo/schedule.py``) and the ``allo``
  dialect's ``AlloISAOps``/``AlloISATypes``, none of which exist in ``main``'s
  ``mlir/``. Pointing the symlink at ``main``'s build fails at
  ``ModuleNotFoundError: No module named 'allo._mlir.schedule'``.
- That build reads ``$LLVM_BASE_DIR`` (not the ``LLVM_BUILD_DIR`` the dataflow
  simulator wants) and needs ``llvm-project`` at ``040a6419``. The 11 GB build
  on this host is ``6b09f739``, which is ``main``'s pin.
- ``mlir/CMakeLists.txt:17-23,35`` FATAL_ERRORs without CIRCT, and
  ``MLIRAlloRegisterEverything`` -- which the bindings link -- lists
  ``CIRCTHW``/``CIRCTComb``/``CIRCTSeq`` plus ``MLIRAlloMicroarch`` and
  ``MLIRAlloScheduling``, so it is not separable. ``externals/circt`` is an
  empty submodule directory and no ``CIRCTConfig.cmake`` exists anywhere on the
  filesystem.
- ``mlir/CMakeLists.txt:27-33`` FATAL_ERRORs without an OR-Tools **cmake
  package**. ``$HOME/chia-ortools`` is not one: it ships runtime ``.so`` files
  to keep an already-built tree importable.

Two corrections to the environment as it was described to us: the conda env is
**not** ``chia_env`` (python 3.10.19, ray + google-genai + mcp, and no numpy or
pytest -- it is the CHIA agent's env); and no ``cpython-314`` extension module
survives on this host, the tree that had them having been deleted. The
``allo`` env (python 3.12.13) does have all four pure-python deps --
numpy, sympy, ml_dtypes, pytest -- so the deps are not the problem, the bindings
are.

Test counts
-----------

- **238 passed / 40 skipped / 0 failed** at commit ``cee57bc0`` (2026-09-07) is
  the only figure with committed provenance.
- **243 / 0 / 35** appears nowhere in git -- only in an untracked
  ``$HOME/chia-ortools/README.txt`` -- so it is **not citable**, and we did not
  reproduce it.
- Measurable without a build: **230 test functions in 28 files**, and 30
  ``pytest.importorskip("torch*")`` gates. ``torch_mlir`` is absent from every
  env on this host, so those tests skip and assert nothing, including every
  numeric cost-model assertion.
- ``grep -rn makespan tests/`` is **zero hits**. No test asserts a makespan
  anywhere; ``mapspace``, ``Mapping``, ``Loop``, ``Priced`` and ``Binding`` have
  zero hits too. Twenty of the 28 tests call ``ISA.compile_program`` without
  ``mapping_for``, so the mapping layer is never entered.

For the same reason, the "``epoch.schedule()`` reproduces its makespan exactly,
100 == 100, over 200,000 random topological orders" result is **also not in
git**, here or on ``chia-codesign``. It is in the same category as 243/0/35.


The Seam, Prototyped
====================

``examples/accelerator/tinytpu_vitis/act_nest.py`` (pure python + numpy) takes a
nest -- anything with ``.rank``/``.factor``/``.level``/``.spatial``, so a real
``Mapping.loops`` plugs straight in -- and emits a TinyTPU-isa program through
``isa_dsl.Program``. It copies no ACT code; the enumerator in it is a
deliberately obvious stand-in, because ACT's own cannot be executed here.

Run it and three things come out.

**The seam is real.** The nest describing the tiling ``isa_dsl.gemm_program``
hardcodes re-emits that program *word for word*, at all five
``bench_isa.SHAPES`` and both relu settings. A mapper's answer can drive our
generator.

**The seam is narrow, and the ISA says where.** Of 1,226 enumerated nests at
16x16x16, **3 are encodable**:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - nests
     - refused because
   * - 1,150
     - ``acc`` is a **static instruction field with no predicate on an
       induction variable**, so the k=0 tile must be a peelable prefix, which
       pins K innermost and unsplit and kills every permutation that moves it.
   * - 54
     - a limit of the prototype emitter, not of the machine (it re-stages A per
       m-tile and cannot interleave that with an n loop).
   * - 17
     - ``AGU_TERMS=3``, one instruction word's address-term budget.
   * - 2
     - the accumulator RAW distance, refused by ``check_program``.

``LOOP_DEPTH=4`` binds nothing. The 3-term AGU does not merely forbid nests, it
**chooses the data-reuse strategy**: an m-tiled nest is encodable only if A is
re-staged per m-tile, because keeping it resident needs a fourth term on the
accumulating ``mm``.

**The search confirms the shipped choice.** All three survivors are correct
against ``isa_ref.run``, and the shipped ``N4>K4`` nest is the cheapest by the
proxy: 14 instructions / 36 words / 32 dynamic issues, against 16/40/60 and
16/40/116 for the m-tiled ones. A negative search result, and an honest one.

That proxy is ``74.5 + 21.70 x dynamic instructions``, the regression over five
cosim points in :doc:`/designs/tinytpu_isa`. It ranks; it does not measure.


Corrections to This Page's Earlier Numbers
==========================================

- "Our 9 opcodes" listed ``DMA_ST`` as expressible. The machine **refuses** it
  (``check_program`` rejects ``OP_DMA_ST``) and ``isa_dsl`` has no emitter for
  it. There are 10 encodings, 9 executable: 7 data opcodes plus
  ``LOOP``/``ENDLOOP``.
- "``gemm.relu`` at 16x16x16 is 49 instructions = 106 words flat, 17
  instructions = 42 words looped, headroom 14 words." Measured today: **32
  instructions / 72 words flat, 14 instructions / 36 words looped**, headroom
  **20** of ``IMEM_SIZE=56``.
- "The re-roll is the missing inverse of an identity we already test." It is
  not an inverse of anything: ``gemm_program`` exposes no tiling parameter, so
  the adapter is a new emitter, which is what ``act_nest.py`` is.


The Co-design Loop This Argues For
==================================

.. note::

   **It is built.** See :doc:`/extensions/codesign` for the loop as it exists,
   its frozen/editable split, its measured control (169 / 686) and its
   ``$0`` test suite. What follows is the argument that produced it; every
   point below is now enforced mechanically rather than stated.

The finding above is the shape of a publishable loop, because the thing that
collapsed the mapspace was a **hardware/ISA parameter**, not a compiler bug.

- **The agent edits the ISA and the microarchitecture**: the ``acc`` field's
  predicability, ``AGU_TERMS``, ``LOOP_DEPTH``, ``T``, ``MAXDIM``,
  ``IMEM_SIZE``, the accumulator RAW distance.
- **Frozen**: the mapper's objective and the program generator. The agent may
  not edit the thing that scores it. (This is the guard CHIA already needs and
  the reason its runs are auditable.)
- **The inner loop is exhaustive, not agentic**: for each hardware candidate,
  enumerate the mapspace and take the best encodable nest. The agent proposes
  hardware; enumeration answers what that hardware can run.
- **Feedback is a measurement**: cycles from RTL cosim (``cosim.py``) and
  resource estimates from HLS ``csynth``. PD is deferred. The pure-python proxy
  is for ranking inside one candidate only, and must never be the reported
  number.
- **Publishable rather than anecdotal** needs: the frozen-vs-edited split stated
  up front; every reported cycle count from cosim, with the exhaustive inner
  search making "the best mapping for this hardware" a claim rather than a
  sample; and a cost/benefit axis, since ``AGU_TERMS=4`` and a predicable
  ``acc`` both cost area that ``csynth`` can price.

Failure modes, named
--------------------

- **Scoring on ACT's cost model optimises a model.** ``Priced.cost`` is
  ``(makespan, emits)``; ``makespan`` comes from ``epoch.Sigma``, whose
  agreement with our RTL has never been tested, and nothing models instruction
  memory -- while ``assemble`` asserts ``len(words) <= IMEM_SIZE`` and ACT's own
  backend unrolls, so its program size is the product of the loop factors.
  Any loop scored on ACT's objective would reward a program our machine cannot
  hold.
- **The proxy invites the same mistake more cheaply.** It is a five-point
  regression; a loop that optimises it will find its residual.
- **An agent given both the hardware and the scorer will move the scorer.**
- **An inner search that is not exhaustive turns a hardware comparison into a
  search-quality comparison.** Keeping the mapspace small enough to enumerate is
  a requirement, not a convenience.
