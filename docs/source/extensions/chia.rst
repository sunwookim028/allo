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
CHIA Agentic Co-Design
####################################

An LLM-agent loop that edits a TPU's instruction set and microarchitecture and
scores every proposal by running the real tools: Vitis HLS C-synthesis and RTL
co-simulation, against a frozen reference model. It produces, per candidate, a
diff plus a cycle count, an area figure and an estimated clock, and a
machine-checked verdict of ``win`` / ``not-better`` / ``rejected``. It needs
the ``allo`` conda environment with this checkout's ``mlir/build``, Vitis HLS
2023.2, a ``chia_env`` Python 3.10 environment, the ``opencode`` CLI, and --
for anything but the $0 harness test -- a Google Cloud project billed to the
CHIA2026 account. The agents themselves are driven by
`CHIA <https://github.com/ucb-bar/chia>`_ (``ucb-bar/chia`` at ``16c35e9``).

Quick start
-----------

Environment (once): the ``allo`` env with this checkout's ``mlir/build``,
``chia_env`` (py3.10, ``requirements.txt``), opencode via
``npm ci --prefix examples/tinytpu/chia_agent``, and
``chia.env`` copied from ``chia.env.example`` at the repository root.
``chia.env`` is gitignored and must never be committed. Then:

.. code-block:: bash

   examples/tinytpu/chia_agent/gcp_setup.sh  # auth, project, billing, APIs, spend report
   conda activate chia_env; set -a; source chia.env; set +a
   # chia.env MUST be sourced in this shell, before the head: Ray workers
   # inherit the raylet's environment (loop.py passes no env_vars), and
   # opencode is on PATH only because chia.env puts OPENCODE_BIN there.
   # A private --temp-dir, short enough for a Unix socket (107 bytes), plus
   # RAY_ADDRESS: never address="auto" on the shared default, never `ray stop`.
   ray start --head --temp-dir=/tmp/ray-chia \
       --resources='{"opencode_creds": 2}' --include-dashboard=false
   export RAY_ADDRESS=<what ray start printed>
   cd examples/tinytpu/chia_agent
   python test_harness.py --phases e,c    # ~2 min, $0; full suite ~30 min, $0
   python preflight.py --budget-usd 30    # the gate alone, $0
   python swarm.py --workers 2 --iterations 3 --budget-usd 30 --run-dir <run> \
       2>&1 | tee <run>/swarm.log         # swarm's own stdout is not persisted
   python swarm.py --status <run>         # any time, from any shell, $0
   python accept.py --out <run>/control                 # the run's control, once
   python accept.py --diff <run>/<worker>/best.diff --out <run>/accept-<worker> \
       --control <run>/control/accept.json              # or omit it and measure again
   python3 spend.py run <run>             # what it cost, from opencode's DB
   pgrep -f /tmp/ray-chia/session_ | xargs -r kill   # this cluster only;
                                          # `ray stop` is host-wide

The full procedure, written for someone who has never run this --- what a run
is and costs, the $0 path, pre-registration, watching a run, reading the
spend, and abandoning an interrupted one --- is the "Quick start, assuming
nothing" section of ``examples/tinytpu/chia_agent/README.md``.

``test_harness.py`` and ``preflight.py`` cost nothing and need no model call,
so run them first: ``--phases e,c`` takes about a minute and the full suite
about thirty. Each candidate's evaluation is 2-3 minutes of cosim, and
``accept.py`` adds one five-shape control measurement of about 4.5 minutes per
run. The one paid run of this shape so far -- 2 workers, at most 3 iterations
-- took 100 minutes of wall time and $28.54
(:doc:`/extensions/chia_results`).

.. warning::

   Paid runs spend real money on GCP. ``chia.env`` is gitignored and must never
   be committed, ``preflight.py`` refuses to start unless the project bills
   ``CHIA_BILLING_ACCOUNT`` and the run's cap fits under
   ``CHIA_TOTAL_CAP_USD``, and ``test_harness.py`` ($0) should pass before any
   paid run. See `Billing`_ below.

How it works
------------

Workers propose diffs through MCP tools; the harness evaluates each one in a
fresh tree of its own and scores it against a control it measured itself:

.. code-block:: text

     workload spec + measured baseline
                   |
                   v
     +------------ swarm.py / loop.py, orchestrated on Ray ------------+
     |  worker "open"            worker "acc-follows-k"          ...    |
     |  opencode + Gemini        opencode + Gemini                      |
     +--------------------------------+---------------------------------+
                                      |  MCP tools only -- loopback, one
                                      |  port per worker; opencode's own
                                      |  file and shell tools are denied
                                      v
          read_spec    replace_text    apply_spec_patch    evaluate
                                      |
        EDITABLE  the design           |   FROZEN  reference model, gates,
                                      |           evaluator, acceptance
                                      v  candidate diff
     +------ evaluate.py: a fresh tree from git for every candidate ----+
     |  policy  -->  sandbox  -->  gates  -->  csynth  -->  RTL cosim   |
     |  static       bwrap         bench, stress, param                  |
     +--------------------------------+---------------------------------+
                                      |  nonce-vouched verdict:
                                      |  cycles + area + clock
                                      v
           accept.py: against a control measured in the same run
                      win  |  not-better  |  rejected
                                      |
                                      v
           evidence/     spend read from opencode's database,
                         capped per run and in total

Two dispositions share this shape. **Using** edits the design with Allo's
abstractions as they stand (``chia_agent/``, on ``main``). **Maintaining**
edits Allo itself -- the compiler -- and adds a rung that exercises a new
primitive through a harness-authored call site (``chia_abstraction/``, on
branch ``chia-abstraction``).

.. note::

   **Attribution.** The agent search is Sunwoo Kim's. Most of the retired
   ``chia-codesign`` branch is **not**: it imports Kai Shao's CIRCT RTL
   generator, ACT compiler flow and ``allov2`` core from
   https://github.com/kkkaishao/allo (503 of its 539 commits not on ``main``;
   see that branch's ``ATTRIBUTION.md`` and :doc:`/extensions/act`). The loop on
   ``main`` carries none of Kai Shao's code. The terms under which it was
   imported are unrecorded, which ``ATTRIBUTION.md`` says must be settled before
   that work is published or landed.


.. _chia-isa-loop:

The design under search
~~~~~~~~~~~~~~~~~~~~~~~

The search runs over the **current** TinyTPU design in
``examples/tinytpu/``: int8/int32, 4x4 weight-stationary
array, PC + 4-deep loop stack + 3-term AGU, GEMM programs generated by
``isa_dsl.py``, shipping at **175 / 265 / 421 / 482 / 674** cosim cycles
(:doc:`/designs/tinytpu_isa`; the run recorded below was scored against the
then-shipping 172 / 262 / 418 / 484 / 686). Workers propose edits, the harness gates and
cosims each one itself, and the best survives. The ``chia-codesign`` loop
searched a different, older design against a cost model; this one is scored
by RTL cosim.

**Where it lives.** ``examples/tinytpu/chia_agent/`` on
``main`` (landed 2026-09-19 from branch ``chia-isa``, now deleted). Its
``README.md`` is the operator's manual; ``dev/records/tinytpu/chia-evidence/`` holds the
trimmed evidence behind every number on :doc:`/extensions/chia_results`. The branch's full history and the
raw run output (worker logs, cosim logs, csynth XMLs) are on the tag
``chia-isa-run1-evidence``, at ``chia_runs/<run>/``.

Editable vs. frozen
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 12 50 38

   * -
     - files
     - how it is enforced
   * - **editable**
     - the seventeen paths ``chia_agent/design.py`` names: the instruction set
       (``isa_spec.json``, the ``isa_encoding.py`` generated from it and the
       ``isa_ref.py`` built on that), ``microarch_isa.py`` (the parameter set,
       and the ``CHIA_CONFIG`` declaration below), ``isa_dsl.py`` (the program
       generator), ``ip/isa.py``, ``ip/tinytpu.py``, ``ip/assembler.py``,
       ``ip/programs.py``, and the eight units under ``ip/units/``
     - The agent edits a private copy in ``<run>/<worker>/spec/``, which
       mirrors the package, never the repository. The hardware left
       ``microarch_isa.py`` for ``ip/units/`` when the design became a unit
       library, so an editable set that stopped at the two old file names
       would no longer contain the machine -- both wins of run 1 landed in
       what is now ``ip/units/dma_load.py`` and ``ip/units/sequencer.py``.
       ``isa_encoding.py`` is generated: the ``regenerate_isa`` tool runs the
       frozen generator on the candidate's own spec, because an agent cannot
       keep a 700-line generated file byte-identical by hand.
   * - **frozen**
     - main's ``cosim.py`` (testbench, ``SHAPES``, golden reference, every
       Vitis/TCL setting), ``bench_isa.py``, ``stress_isa.py``,
       ``kpn_model.py``, ``shapes.py``, ``gen_isa.py``; the workload suite
       (``workloads/``) and ``chia_agent/area_proxy.py`` -- what prices the
       design and what it is priced *on*; the design's own machinery
       (``ip/params.py``, the package ``__init__``\ s, and the reduce IP they
       import); and ``chia_agent/``'s ``evaluate.py``, ``gate_runner.py``,
       ``param_check.py``, ``spec_policy.py``, ``mapspace.py``,
       ``codesign_gate.py``, ``codesign_cosim.py``
     - Read from git, never from disk; main's design evaluator must also be
       byte-identical to the base ``evaluate.main_base()`` **derives** -- the
       merge-base of the frozen ref with main. It was a hand-written
       ``MAIN_BASE`` constant, and it went stale four times in five days,
       each time refusing every candidate at stage ``setup`` with a message
       that blamed the design rather than the pin.

.. _chia-isa-editable:

Why the ISA can move, and what the oracle is
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``isa_spec.json``, ``isa_encoding.py`` and ``isa_ref.py`` were frozen until
2026-09-25, and the reason was sound: ``isa_ref.py`` is what ``stress_isa.py``
checks random programs against, so a candidate that could rewrite its own
reference model could weaken the rule and score strictly better. The cost was
that the loop could not change the instruction set at all, which is half of a
co-design space.

**The referee cannot be the editable surface, so the referee moved outside the
repository.** ``workloads/run.py --verify`` is the one check with something
this project did not write on one side: ``torch.nn.Linear``'s own forward on
the model's real weights, with the machine's epilogue applied in torch. It is
now the FIRST gate, it costs 4 s at MAXDIM=16, and a byte that differs is
``gate:pytorch``. ``allo/actions.py`` is not the answer to this and was not
used for it: it is another model of the same ISA *inside* the repository, with
the same failure mode ``isa_ref`` has.

Two other things carry the weight the freeze used to:

* ``gen_isa.py --conform`` (frozen) is a gate on the candidate's **own** spec:
  ``isa_encoding.py`` byte-identical to what that spec generates, every
  software constant and bit slice of the design at the spec's positions, the
  assembler and generator through the same encoder, the reference model
  structurally forbidden from importing an opcode number or a field position
  from the design, and the bit ranges the *emitted HLS* reads. 36 s.
* **The GEMM goldens were never** ``isa_ref``. ``bench_isa.py``,
  ``stress_isa.py`` and ``cosim.py``'s testbench each compute their own numpy
  gold, and all three are frozen. ``isa_ref`` is the reference for *random
  programs* only.

**What the oracle is worth, stated rather than implied.** 8,912 bytes across
four MLPs at MAXDIM=64, 92 % of it ``mlp_wide``; ``mlp_bias`` contributes none,
because it is the probe that must not map. At the scored MAXDIM=16 only two
models fit, and ``--verify`` draws each eight times (the mapping depends only
on the shapes, so extra draws are nearly free) for 2,688 bytes over six layers.
That is an oracle, not a corpus: it adds a reference nobody here can edit, and
it adds no shape the GEMM sweep does not already cover. Widening it means more
models that fit a small build, and that is the next thing to do to this gate.

Because the spec is editable and ``isa_ref`` *evaluates* the expressions its
actions carry, those expressions are held to an expression language -- a name,
an integer, arithmetic, bitwise and comparison operators -- in two independent
places: ``spec_policy.expression_violations`` at edit time, and
``isa_ref.expression``, an AST walk that replaced ``eval``, at run time. An
``eval`` with emptied builtins still reaches ``().__class__.__bases__[0]`` and
from there the frame holding ``gate_runner.py``'s nonce.

The evaluator
~~~~~~~~~~~~~

- **gate** (functional, Allo simulator, ~110 s): ``run.py --verify`` (the
  PyTorch oracle, first and cheapest -- 4 s), ``gen_isa.py --conform`` (the
  candidate's own spec against its own generated artefacts and the design's bit
  slices -- 36 s), ``bench_isa.py`` (the published
  [-4, 4] setup), main's ``stress_isa.py`` (492 runs: full-range, corner and
  boundary int8 at all 64 shapes, ``C`` prefilled and compared in full, vector
  and random programs against ``isa_ref``, many calls on one build), and
  ``param_check.py`` at three configurations ``evaluate.param_configs``
  derives from the candidate's own (below). Negative controls, all in
  ``test_harness.py``: an int16 partial sum passes ``bench_isa.py`` and every
  cosim testbench and ``stress_isa`` rejects it (251/492 exact); a spec edit
  without ``regenerate_isa`` is refused at ``gate:isa`` for a stale artefact;
  and ``isa_ref``'s ReLU primitive weakened to the identity -- a candidate
  rewriting what used to judge it -- is refused at ``gate:pytorch`` before
  ``bench_isa`` runs at all.
- **score**: a THREE-TERM objective, reported per term and never summed.
  See :ref:`chia-objective` for why each term is there and what it replaced.

  1. ``model`` -- RTL cosim cycles for ``mlp_tiny`` and ``mlp_deep``, the two
     workload-suite models whose every layer fits one build, over one csynth
     shared by all six layers. **The primary**, and measured at
     ``evaluate.SCORED_MODEL_ENV`` --- ``MAXDIM=64``, **not** the GEMM term's
     16, for the measured reason below.
  2. ``gemm`` -- RTL cosim cycles at 4x4x4 and 16x16x16, unchanged. **The
     control**, kept so that a change which helps models and hurts GEMM shapes
     has to be stated rather than hidden.
  3. ``area`` -- a standard-cell area ESTIMATE from the candidate's own
     structural bit census (``chia_agent/area_proxy.py``), in milliseconds.
     csynth's FPGA resource table is still reported beside it and no longer
     decides anything.

  Each testbench bit-exact, the 3.33 ns clock met.
- **acceptance**: ``accept.py`` is the only way a win is claimed. On a clean
  ``git worktree`` with its own ``mlir/`` build it runs ``bench_isa``,
  ``stress_isa``, ``param_check``, cosim at all five shapes, and the
  ``TPU_TB=stress`` RTL testbench (several calls on one RTL instance, whole
  ``C`` compared -- what sees an RTL-only failure such as a dependence pragma
  that is false at a short read-after-write distance). ``claim`` is ``win``
  only against a control ``accept.py`` measured itself, in the same run and
  off the same build (guard 10); the row published when run 1 was scored,
  172 / 262 / 418 / 484 / 686 (``dev/records/tinytpu/chia-evidence/accept-control-476a70d8/``), is what
  that measurement was cross-checked against. Measuring the control per run is
  exactly why the published row having moved twice since -- most recently to
  175 / 265 / 421 / 482 / 674 -- does not invalidate the run.

.. _chia-objective:

The objective, and the two axes it replaced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Until 2026-09-24 the loop scored on **GEMM shapes** against **csynth's FPGA
resource table**, and this project had measured both to be misleading.

**The workload class changes the answer by four to eight times.** The same
optimisation -- widening the operand burst -- is worth **4.3-7.0 %** of runtime
on GEMM shapes and **25-34 %** on multi-layer models
(:doc:`/designs/workload_suite`). A model does not make the problem bigger, it
makes it *longer*, so a fixed per-call cost is repaid at every layer instead of
amortised by one large problem. A loop ranking changes on GEMM shapes ranks
them on the workload class **least** sensitive to the cost they remove. So the
model term is the primary and the GEMM shapes are kept as the control: a
control that is dropped cannot catch a change that buys models with shapes.

**...but only at MAXDIM=64, and that had to be measured.** Every one of those
figures was taken at ``TPU_MAXDIM=64``, and the loop scores at 16, where a DRAM
row is 4 packed words instead of 16 and every operand burst is four times
shorter. Asked directly --- both models, every layer, one csynth per burst
width, at ``T=4 MAXDIM=16 QD=16`` --- the widening is worth **zero cycles**:
861 -> 861 and 1,530 -> 1,530, layer by layer, to the cycle
(``dev/records/tinytpu/model-term-maxdim-20260924.rst``).

Worse than insensitive: at MAXDIM=16 the relationship **inverts**. The GEMM
shapes there *do* see the widening --- CHIA run 1 measured
``0 / 0 / -42 / -59 / -59`` on exactly that configuration --- and the models do
not. A model term at MAXDIM=16 would be the *less* burst-sensitive of the two
terms, which is the opposite of the reason for having one. So the model term is
measured at MAXDIM=64 and the GEMM control stays at 16 because that is the
published row: **two configurations, two csynths per candidate**, and that is
what the term costs.

Measured, that price is **348 s** for the model term (one ``csynth`` and six
layer cosims, returning the published 1 150 and 2 117) on top of a ~230 s
evaluation --- **2.5x per candidate**. Against run 3's whole-run numbers it is
about **1.28x wall**, because the model calls dominate and they have not
changed, and **no change to the spend cap**: the term adds no model calls, so
the $60 cap still buys the six candidates run 3 got for $20.74. What to watch
instead is **Vitis load, 2.5x per candidate**, which across a four-worker swarm
is licence and CPU contention rather than money.

**The FPGA table understates silicon in the components a search most wants to
change.** The burst widening is +43 % flip-flops and +92 % block RAM on FPGA
and **+74.4 % cell area** in 45 nm, of which **99.1 % is two AXI master ports**
that grow 13x when widened. The arithmetic array is **under 4 %** of our
standard-cell logic while the memory-interface adapters are **40.9 %**
(:doc:`/paper`). A Design Compiler run is ~70 minutes on another host with a
licence, so the fix cannot be "run DC in the loop"; it is
``chia_agent/area_proxy.py``, which **counts the bits the design declares** --
every array a unit declares, every channel the architecture wires, at the
candidate's own parameter set -- and prices them, plus the AXI adapters, at
coefficients fitted to the committed DC runs.

It reports itself as an estimate everywhere, and it states its error:

- **0.33 % mean / 0.91 % worst** in-sample over the seven committed TinyTPU DC
  runs, **0.62 % / 1.39 %** leave-one-out;
- its channel census reproduces, **to the bit**, a hand count taken off the
  emitted RTL's instance list (16,640 bits of queue at ``QD=8``, 33,280 at 16);
- ``python examples/tinytpu/chia_agent/area_proxy.py --selfcheck`` re-derives
  every one of those numbers from the committed reports and fails if any moved.

What it is **not**: it is a model of this design and this flow (flip-flop
memories, ``sram_mode='none'``); the four Gemmini runs are not fitted and not
predicted by it; it prices *structure*, not synthesis, so a construct Vitis
renders as a multi-write-port RAM is invisible to it; and the configuration the
loop scores at, T=4 MAXDIM=16, has exactly **one** committed DC run and is the
worst-predicted of the seven. That wants a twelfth DC run, not a better fit.

.. _chia-configuration:

The configuration is a candidate's to propose, inside a stated envelope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Until 2026-09-25 ``evaluate.SCORED`` pinned T=4 MAXDIM=16 and
``check_invariants`` refused any candidate whose build disagreed, so the loop
searched the implementation of eight units at one architecture. A candidate now
declares, at module level in its own ``microarch_isa.py``::

    CHIA_CONFIG = {"T": 8, "MAXDIM": 32}

``evaluate.resolve_config`` reads it with ``ast.literal_eval`` -- no import, no
candidate code -- before the tree is composed, and every stage then runs pinned
to it; ``check_invariants`` still requires the build to *report* what was
declared, so the declaration is a claim with a check on it. A candidate that
declares nothing is scored at T=4 MAXDIM=16 QD=16, the published row, so no
existing candidate, control or published number moves.

Four things refuse a proposal, all of them before anything is built, and all of
them naming what would lift the refusal:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - refusal
     - why
   * - outside the fitted envelope
     - ``area_proxy.envelope_refusals``. The coefficients are fitted to five
       committed DC runs (the two logic-only ones are stubs, not design
       points), which span T in [4, 8], MAXDIM in [16, 64], QD in [8, 16],
       DMA_WORDS in [1, 16]. Outside that span the area term would be an
       extrapolation with no error bar, so ``estimate`` **raises**
       (``OutsideEnvelope``) instead of returning a number. A missing figure a
       search must work around is recoverable; a confident wrong one is not.
       The envelope is derived from ``COMMITTED``, so a twelfth DC run widens
       it by being added to that dict.
   * - past a MAXDIM ceiling
     - ``(MAXDIM // T) * MAXDIM <= 2048`` for the operand layout's addressing
       and ``MAXDIM**3 // T**2 + MAXDIM**2 // T <= 32767`` for a cubic GEMM's
       header count: **88 and 76 at T=4, 128 and 120 at T=8**. Computed from
       the frozen ``isa_encoding.MAXDIM_CEILINGS`` and never typed into the
       evaluator -- this project has already written one of them down wrong in
       three documents ("MAXDIM <= 90" against an answer of 88). The refusal
       names the ceiling and its value.
   * - ``QD`` other than 16
     - depth 8 deadlocks legal programs -- three of the ten tiled programs
       never complete in cosim (:doc:`/developer/limitations` item 24) -- so it
       is refused rather than left to hang for the 1,800 s cosim timeout.
   * - ``T < 4`` or ``MAXDIM % T``
     - ``ip/params.py``'s invariants.

**Inside the envelope is still thin, and the loop should be told so.** The
(T, MAXDIM) box has one interior point -- MAXDIM=16 at T=4, the
worst-predicted of the seven runs. The DC runs that would earn the box, in
order: **T=4 MAXDIM=32** and **T=8 MAXDIM=16** (the corners a co-design loop
reaches for first), then **T=4 at a ceiling (88 or 76)**, then **a clean QD
pair at the scored point** -- two exports of one commit, which is the only way
to replace the +6.8 %-against-+6.6 % agreement of two estimates with a
measurement. The priority list lives in ``area_proxy.py``'s docstring, beside
the model it would correct.

**Cost, measured rather than estimated**
(``dev/records/tinytpu/codesign-space-20260925.rst``). One whole candidate is
**623.9 s** at the published row and **1050.5 s** at ``T=8 MAXDIM=32`` --
**1.68x** -- on the unmodified design, same host, both ``ok: true``. The two
new gates are 37.8 s of the first figure (6.1 %), and ``gen_isa.py --conform``
is 34.9 s of that because it builds the design down the HLS path to read the
bit ranges the emitted C++ takes. A candidate was ~578 s before this work. A
spend ceiling for a search that may propose T=8 should use the T=8 figure, not
an average.

That T=8 run is also the first co-design point this loop could produce:
16x16x16 falls from **674 to 426 cycles** (-36.8 %) and ``mlp_tiny`` from
**1,150 to 783** (-31.9 %) for an area estimate up from **1.20 to 2.23 mm^2**
(+85.6 %), clock met at 2.431 ns. A trade the objective can now state.

**Two things a T=8 candidate cannot measure, and both are frozen machinery
rather than the candidate.** ``cosim.py`` asserts every scored dimension is a
multiple of T, and ``4x4x4`` and ``12x12x12`` are not multiples of 8, so the
GEMM term at T=8 is ``16x16x16`` alone (``shapes_skipped`` says so). The frozen
mapper refuses ``mlp_deep``'s 16x16x12 layer for the same reason, so the model
term is ``mlp_tiny`` alone (``model_skipped``) and the oracle's corpus becomes
``mlp_tiny`` + ``mlp_small``. Either could have been made a refusal; both were
made SCOPE, because refusing a candidate for the shape list's choice of numbers
would leave every T but 4 unreachable and the loop searching one architecture
again. The consequence is real and is reported per run: **a T=8 candidate is
measured on a narrower workload than a T=4 one.** Widening it means shapes and
models legal at more than one T, which is a person's edit to ``shapes.py`` and
``workloads/models.py``.

Whether the new objective is better: the re-score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asserting it would be worthless, so ``chia_agent/rescore.py`` hands both
objectives the four changes this project has already measured and prints how
each ranks them::

    python examples/tinytpu/chia_agent/rescore.py

.. list-table::
   :header-rows: 1
   :widths: 20 38 42

   * - change
     - old objective
     - new objective
   * - burst widening
     - **could not see it** -- never measured at either search shape
     - ``trade``: -25 to -34 % on models, **+73.3 % area (est.)**, 99.6 % of
       it in the adapters
   * - channel depth ``QD`` 8 -> 16
     - ``trade``, **kept** on a -7 total over the two scored shapes
     - ``regression``: the five-shape total is **zero** (+4+4+4-1-11), and
       +6.8 % area
   * - operand mirror removal
     - ``regression`` (+16)
     - ``regression``; no area measurement of it exists on either axis
   * - CHIA run 1's burst win
     - ``trade``, **kept** on -59
     - ``trade``, kept -- and now **priced**: +24.3 % area (est.), 99.6 % of it
       in the AXI master ports

**Nothing is reversed**, which is the first thing to check. What changes is
the classification of the two that matter, and one of them *indicts the old
objective directly*: the channel-depth change sums to **zero** over the five
shapes, so no cycle objective over the five can prefer it, and over the two the
loop actually scored it sums to -7, so the old objective would have kept it for
the wrong reason. What it really buys is three legal programs going from never
completing to completing, and **neither objective has a term for that**. That
gap is open.

Two honest limits on this evidence:

- **Model cycles exist for exactly one of the four changes.** On the other
  three the new objective is running on its GEMM control, and is the old
  objective with a better resource axis. The resource axis is supported
  strongly; the model axis on a single point.
- And that single point is the one the MAXDIM measurement above qualifies: the
  model term earns its place at MAXDIM=64 and demonstrably does not at 16. The
  objective is better *because the term is measured where the effect is*, and
  the honest form of that sentence names the configuration.
- The proxy's attribution of run 1's win -- +24.3 % area, 99.6 % in the AXI
  master ports -- is the same attribution Design Compiler made for the
  MAXDIM=64 widening (99.1 % in two adapters), reached in milliseconds instead
  of 56 minutes. It is a *prediction*, and the way to settle it is a DC run on
  that export.


Reference
---------

Where the code lives
~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Branch
     - Contents
   * - ``main``
     - ``examples/tinytpu/chia_agent/``: the loop for
       TinyTPU-isa, landed from ``chia-isa`` on 2026-09-19, with trimmed
       evidence in ``dev/records/tinytpu/chia-evidence/``. The ``chia-isa`` branch is
       deleted; its history and raw run output are on the tag
       ``chia-isa-run1-evidence``.
   * - ``chia-codesign``
     - A **separate codebase**, not a feature branch. It descends from Kai
       Shao's ACT fork and shares only a March 2026 ancestor (``76130c63``)
       with ``main``; the two are maintained separately and are not expected
       to converge (see ``dev/fork_maintenance.rst``). Carries
       ``CODESIGN.md`` (the claim register), ``notes/CHIA_CHECKPOINT.md``
       (version pins), ``examples/accelerator/tinytpu/`` and ``chia_runs/``.

Neither ``CODESIGN.md`` nor ``CHIA_CHECKPOINT.md`` exists on ``main``; read them
with ``git show origin/chia-codesign:CODESIGN.md``.

Guards
~~~~~~

Two of these are **derived rather than listed**, because the same defect has
now broken the loop twice: a value copied into several files with nothing
comparing the copies to its source. The frozen set is the import closure of
the evaluation's entry points, computed from the ref (``evaluate.import_closure``),
and ``compose`` refuses a tree that is not closed under its own imports. The
published five-shape row is parsed out of ``reproduce.sh``'s ``EXPECTED``
(``control.reproduced``), and the copies that cannot be derived are gated by
``control.check_pins``. Both run in ``test_harness.py``'s guard phases, in
under a minute, rather than after a candidate's evaluation.

What each cost before it was derived: a hand-written ``WORKLOAD_SUITE`` that
omitted its own runner's imports failed **every** candidate at stage ``model``,
so the loop could complete no run; and four literal copies of the pre-``TPU_QD=16``
row in ``swarm.py`` meant every worker a run launched was handed a baseline the
design had not produced for days.

Mechanical enforcement, not instructions:

1. **Tool surface.** opencode's own file and shell tools are denied; the MCP
   edit tools accept only the paths ``design.EDITABLE`` names, and
   ``read_spec()`` with no argument lists them.
2. **Frozen files from git**, a fresh evaluation tree per candidate, anything
   else in the spec directory ignored; ``loop.py`` refuses to start on a dirty
   frozen path.
3. **Static policy** (``spec_policy.py``, executed from git, at edit time and
   again at evaluation): no file I/O or numpy file readers/writers, process
   spawning, ``exec``/``eval``, ``sys.modules``, dunder or frame walking,
   ``SystemExit``/``exit``, or assignment to an attribute of an imported module
   or an alias of one.
4. **Sandbox.** Every process that imports the candidate runs under ``bwrap``:
   read-only filesystem and tree, only the work directory writable, own PID
   namespace. The tree and the checkout's tracked files are compared byte for
   byte after each stage (``tamper``). A ``tamper`` verdict on a run that
   should have been clean is, in practice, usually a **person or another agent
   editing the checkout while a measurement was in flight** -- suspect that
   before the candidate. ``accept.py`` is immune by construction: it measures
   in its own ``git worktree`` at ``--ref``, so an edit elsewhere in the
   checkout cannot reach it.
5. **Vouched verdicts.** A printed ``STRESS OK`` proves nothing when the
   candidate runs in the same process: on the branch, a candidate that printed
   it and raised ``SystemExit(0)`` at import passed the old gate with an int16
   datapath. Each check now runs under ``gate_runner.py``: the evaluator hands it
   a fresh nonce on stdin before the candidate is imported, the verdict is the
   check's *return value*, and ``numpy``/``allo`` are frozen against
   monkeypatching for the duration; only then is ``CHIA-GATE <check> OK
   <nonce>`` printed, and the evaluator requires that line.
6. **Parametricity.** A design specialised to the one configuration it is
   scored at would pass everything else. ``param_check.py`` rebuilds the
   candidate at three other configurations -- at the published row,
   ``TPU_MAXDIM=8``, ``TPU_MAXDIM=12`` and ``TPU_T=8 TPU_MAXDIM=32``, and
   ``evaluate.param_configs`` moves that list onto whatever configuration the
   candidate proposed -- requires the build to report that configuration,
   and requires every GEMM shape of it (full-range, corner and boundary
   operands, ``C`` compared in full) and random programs to be exact
   (``gate:param``). The policy requires ``T`` and ``MAXDIM`` to stay
   ``int(os.environ.get(...))`` parameters.

   ``T`` is varied too. Measured on main,
   ``TPU_T=8 TPU_MAXDIM=32 param_check.py`` gives ``PARAM OK: 408/408 runs
   exact`` and ``TPU_T=8 TPU_MAXDIM=32 bench_isa.py 32 32 32`` gives
   ``ALL EXACT``. What is limited is the **harness**, and the limit is
   ``MAXDIM/T >= 3``, not ``T == 4``: three test-program generators address
   column block 2, which exists only at that ratio, so at T=8 with MAXDIM=16
   (ratio 2) nine of 24 random seeds cannot be generated and ``param_check``
   refuses for want of programs rather than for a wrong answer. An earlier
   revision of this page concluded otherwise; the withdrawal is recorded in
   :doc:`/extensions/chia_results`.
7. **Documentation.** A candidate may not remove, net, more than 15 lines of
   comments and docstrings against the frozen file; rewording is free.
8. **Memory model and premises.** Every ``TPU_*`` variable but the project path
   is scrubbed before cosim (``m_axi_latency`` 0, as Gemmini's harness); the
   build must report the configuration it is scored at (``check_invariants``,
   T, MAXDIM and QD); the 3.33 ns target must be met; and each shape's own
   cosim log and simulated time must agree with the reported cycles.
9. **Loopback.** The MCP tool servers bind 127.0.0.1 by default; a multi-host
   swarm must opt in with ``TINYTPU_TOOL_HOST=node`` and bring its own
   authentication.
10. **A measured control** (``control.py``). A win is "fewer cycles than the
    unmodified design", so both numbers have to come from one machine, one
    Vitis install and one build. ``accept.py`` measures the control itself:
    ``cosim.py`` on the *committed* design, all five shapes, in the same
    ``git worktree`` and off the same ``mlir/`` build, **before** the
    candidate's diff is applied -- the checkout is asserted pristine
    immediately before and after. Nothing a candidate does can reach that
    number: not a byte of it exists on disk when the cycles are taken, its
    processes are sandboxed to their work directory (guard 4) and so cannot
    write a control record either, and the control's cosim verdict is
    nonce-vouched (guard 5) like every other. ``accept.json`` records it as
    ``control`` -- cycles, every editable file's blob id, the ref, the
    estimated clock, the wall time, when it was measured -- so an accepted
    result can be re-derived against the same control.
    ``--control <control-run>/accept.json`` reuses one control for the rest of
    a run's candidates; a record for another design, not vouched, not measured
    on a pristine checkout, short of a shape or missing the clock is refused
    (``claim: no-control``), never silently used. A cosim that flakes at one
    shape lands in the same place -- ``no-control``, before the candidate is
    measured at all, and the run is repeated rather than compared against
    something else. Cost: one five-shape cosim, ~4.5 min, once per run. The search's own baseline (``loop.py``, iteration
    0) was already a self-measurement; it now records the design's blobs and
    cross-checks itself too.

    The control and the candidate are measured by the **same driver** --
    ``accept.py``'s one ``DRIVER``, threaded into both passes, recorded in the
    record, and refused by ``control.unusable`` if a reused record names
    another. A second driver (the co-design mode's mapper-driven
    ``codesign_cosim``) describes the same hardware with a different program,
    so a control measured by one and a candidate by the other shows the
    driver's difference as the candidate's win: at 4x4x4 the mapper's program
    is 24 instruction words against the hand-written 28, same four dynamic
    issues, bit-exact -- a free -3 cycles for every candidate in that mode.

    ``control.RECORDED`` is therefore keyed ``(driver, the two blobs)`` and
    ``control.PUBLISHED`` by driver: a driver with nothing recorded is
    ``UNRECORDED`` and exits 3 with its measured numbers printed, to be
    recorded, rather than quietly checked against the other driver's.

    ``control.RECORDED`` and the published numbers are a **cross-check**, not
    the control. They used to *be* the control, keyed by the blob ids of the
    two editable files -- and a prose-only edit to ``microarch_isa.py``
    (docstring corrections, two dead constants) moved the key, so acceptance
    reported ``no-baseline`` and could accept nothing. Failing closed was
    right; keying a control on a file's bytes was not. As a cross-check the
    same table cannot block and cannot go quiet: a design whose blobs are not
    recorded is compared against the published five-shape numbers instead, and
    a disagreement prints a banner and exits 3, because it means either the
    toolchain moved or the design changed behaviour and no result of that run
    can be trusted until a person says which. A design whose cycles
    deliberately move gets its entry in ``control.RECORDED`` in the same
    commit.

    The comparison is exact -- no tolerance -- because *this* side is
    deterministic: the five shapes reproduce to the cycle, run to run and
    across independent re-measurements, so two measurements of one
    configuration that differ are a finding, never noise to average. (The
    Gemmini column of the comparison is the noisy one, up to 20 cycles of
    trial-to-trial spread; the two sides' tolerances are not the same and this
    check applies only to ours.) And a published number differing from an
    in-run one is not automatically an error in either: the co-design mode's
    control measures 169 at 4x4x4 against the published 172 because its mapper
    emits a 24-word program where the hand-written one has 28, bit-exact and
    fully accounted for. That is the case for measuring the control per run
    rather than publishing one -- a control has to come from the same run as
    the thing it controls.

Every accepted diff is still read by a person.

Billing
~~~~~~~

**Billing.** CHIA runs on the dedicated project ``chia2026-tinytpu``, billed to
the CHIA2026 account; everything is scoped per process, and the host's global
gcloud config and ADC quota project are never changed. opencode's
``google-vertex`` provider takes the project from the provider options
``loop.py`` sets (then ``GOOGLE_VERTEX_PROJECT``, ``GOOGLE_CLOUD_PROJECT``)
and bills the project in the request URL. ``preflight.py`` runs before any
worker or model call in ``swarm.py``, ``loop.py`` and ``smoke.py``, and refuses
unless the project bills ``CHIA_BILLING_ACCOUNT``, Vertex AI is enabled,
``--budget-usd`` is given, and CHIA's cumulative spend so far plus this run's cap
fits **``CHIA_TOTAL_CAP_USD`` ($500 in chia.env, and the committed default in ``preflight.py``)**; it prints the account,
project, spend so far, remaining and the run's cap. Cumulative spend is
opencode's own database, attributed to CHIA2026 by the cutover time recorded
in ``billing.json`` (opencode stores no GCP project with a session). These are
opencode's figures from its price table, not the invoice; the remaining credit
is only in the Cloud Console.

A run in flight, and one that was interrupted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A run is legible from its own directory, with no memory of how it was
started::

   python swarm.py --status <run>

which prints which iteration each worker is on, every candidate with its
verdict, whether the run is still alive, the spend so far from opencode's
database, and the question each worker was asked. It starts nothing and
spends nothing, and it works equally on a live run, a finished one and a
committed evidence directory.

What a run writes, and when, is worth knowing before an interruption rather
than after:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - file
     - written
   * - ``<run>/run.json``
     - once, after the pre-flight gate and before the first worker
   * - ``<run>/<worker>/worker.log``
     - continuously, unbuffered --- the only live signal inside an iteration
   * - ``<run>/<worker>/variants.jsonl``
     - appended once per finished iteration, which can be tens of minutes apart
   * - ``<run>/<worker>/best.diff``, ``calls.json``
     - only on a worker's normal exit
   * - ``<run>/summary.json``, ``<run>/spend.json``
     - only after every worker has exited
   * - ``swarm.log``
     - never, by itself --- pipe ``swarm.py`` through ``tee``, or the
       ``HARD CAP:`` line is lost

**There is no resume.** No flag, no checkpoint: ``loop.py`` never reads
``variants.jsonl`` back. ``swarm.py`` refuses a ``--run-dir`` that already
holds a ``run.json``, because re-using one is worse than starting over: it
overwrites the tag that attributes the old run's spend, truncates
``worker.log``, appends a second run's iterations to ``variants.jsonl``, and
leaves the previous run's edited spec in place --- so the "baseline" the new
run measures is that design rather than ``HEAD``'s. Start again with a new
``--run-dir``.

Abandoning one cleanly means recording its cost **before** anything else,
then killing what a ``SIGTERM`` to the workers does not reach --- opencode is
a grandchild outside the loops' process groups and keeps spending after its
driver dies. The sequence is in
``examples/tinytpu/chia_agent/README.md`` ("If a run was interrupted"). The
money is already counted either way: the cumulative cap sums every session on
the account since the cutover, finished run or not, so an abandoned run still
reduces what the next one may spend.

**Do not edit the checkout while a run or the harness is running.** The
evaluator compares the checkout's tracked files after every gate and returns
``tamper`` if they moved. Measured 2026-09-24: editing three ``chia_agent``
files during a ``test_harness.py`` run turned two ``gate:param`` cases into
``tamper`` failures; both passed on a clean tree.

The $0 harness test
~~~~~~~~~~~~~~~~~~~

``test_harness.py`` runs the whole harness with no LLM (a scripted
OpenAI-compatible model on localhost; every cloud credential variable is
cleared): **control** the control record a claim may rest on and its
cross-check; **e** frozen-file and import-time attacks, forged verdicts,
sandbox, and that a candidate can neither write nor fake a control record;
**c** an int16 partial sum rejected by stress; **g** the
parametricity and documentation guards; **d** a deadlock killed at 240 s;
**i** the co-design space -- every configuration refusal (envelope, both
ceilings, ``QD=8``, the parameter invariants), a spec edit refused at
``gate:isa`` until ``regenerate_isa`` runs, and a weakened ``isa_ref`` refused
at ``gate:pytorch``; **abf** no-op and a slower design scored concurrently;
**loop** the real
``swarm -> loop -> opencode -> MCP`` path; **accept** ``accept.py`` on a
correct-but-slower diff, against a control measured in that same run. 57/57
at landing (``dev/records/tinytpu/chia-evidence/harness-test-20260919-190240/``).

Limits and known failures
-------------------------

- **Every search so far is n=1 per arm.** No rate of discovery is
  demonstrated; see :doc:`/extensions/chia_results`.
- **Run 2 was stopped after 2 of 5 iterations** by a per-run cap that summed
  the whole account rather than its own sessions. Fixed: every session a run
  opens is titled with the run's tag (``--title``), and ``spend.run_spend``
  sums those and the ids the loop saw returned. Nothing else on the account
  is counted against a run's cap.
- **The $0 guard suite passes: 101 cases, 74.5 minutes** (2026-09-25, all nine
  phases, ``dev/records/tinytpu/chia-evidence/harness-test-20260925-033220/results.json``;
  it was 98 cases in 46.9 minutes on 2026-09-24).
  It was 57 at landing and did not pass; the repair that restored it also
  added phase ``s``, which checks the static guards -- the derived main base,
  the pinned scored configuration, the spend attribution and the policy's
  scope -- in twenty seconds and without Ray or a model.
- **A ``tamper`` verdict on a run that should have been clean** is usually a
  person or another agent editing the checkout while a measurement was in
  flight, not the candidate (guard 4). Confirmed the hard way on 2026-09-24:
  an agent edited ``chia_agent/README.md`` -- inside ``CHECKOUT_WATCH`` --
  while a gate was running, and the guard refused the candidate at ``tamper``
  without attributing anything to it. An unplanned violation is a better test
  of that guard than a constructed one.
- **Timed-out opencode sessions are billed but reported as $0** by the
  per-worker figures, which is why every cap reads opencode's database
  (``spend.py``) instead. Measured at run scale on 2026-09-24: all four of run
  3's large iteration calls ran the full 2400 s timeout and returned no
  ``usage``, so **the whole run reads $0.00 on ``usage`` and $20.72 on the
  database**. The known form was one call billed $4.33 and reported as $0.00.
- **A pinned constant nobody notices going stale is the defect, not its
  value.** ``MAIN_BASE`` went stale four times in five days and then stopped
  resolving at all when the design moved to ``examples/tinytpu/``; each time it
  refused **every** candidate at stage ``setup`` with a message that blamed the
  design. It is derived now (``evaluate.main_base``). ``control.PUBLISHED`` had
  gone stale the same way -- still 172 / 262 / 418 / 484 / 686 after the design
  shipped 175 / 265 / 421 / 482 / 674 -- and now reads ``reproduce.sh``'s
  ``EXPECTED``, the one row a gate checks on every run.
- **A Ray head started without ``chia.env`` sourced fails silently.**
  ``loop.py`` connects with ``runtime_env={"working_dir": ...}`` and no
  ``env_vars``, so Ray workers inherit the *raylet's* environment --- and
  ``opencode`` is on ``PATH`` only because ``chia.env`` exports
  ``OPENCODE_BIN``. Start the head from a shell that has sourced it, not just
  the shell that runs ``swarm.py``: otherwise workers cannot find the binary,
  every model call returns empty, and nothing reports an error.
- **``ray.init(address="auto")`` can hang forever on a dead head.** An orphaned
  Ray head whose raylet's working directory has been removed accepts the
  connection and then cannot spawn a single worker: every one dies in
  ``setup_worker.py`` at ``os.getcwd()``, and the driver blocks in ``ray.get``
  with no error. Observed 2026-09-24 against a head left by a deleted
  worktree. ``ray stop`` is not the fix (it is host-wide, see below); start a
  head with a private ``--temp-dir`` and set ``RAY_ADDRESS``, which overrides
  even an explicit ``address="auto"`` and leaves
  ``/tmp/ray/ray_current_cluster`` alone for other tracks. Two things about
  that temp directory, measured here on 2026-09-24: the path must be **short**,
  because Ray puts a Unix socket under it and ``AF_UNIX`` caps the path at 107
  bytes (``/tmp/ray-chia`` is fine, a path under a session scratch directory is
  not); and to stop only that cluster, match ``<temp-dir>/session_`` rather
  than the temp directory alone --- the shorter pattern also matches the shell
  you type it in, and ``pkill`` will kill it. ``kill`` alone left the raylet
  running; follow it with ``kill -9``.
- **The tool servers bind fixed ports from 8000 up**, so two ``chia_agent``
  processes on one host collide. Serialise them.
- **A checkout under ``/tmp`` cannot run the harness at all**, and it fails in a
  way that blames the candidate. ``evaluate.sandboxed`` gives every gate
  ``--tmpfs /tmp`` and then re-binds only the work directory and the composed
  tree, so a worktree living anywhere else under ``/tmp`` --- an agent
  scratchpad, for instance --- is **masked inside the sandbox**. ``allo`` is not
  composed into the tree (``CHECKOUT_WATCH`` guards it instead), so it is looked
  up on ``PYTHONPATH``, which points at the invisible checkout; the import then
  falls through to the ``allo`` env's editable install, which resolves to
  whatever tree last ran ``pip install -e`` --- on this host
  ``/home/sk3463/allo``, whose HEAD had no ``allo/compose.py``. Every candidate
  dies at stage ``import`` with ``ModuleNotFoundError: No module named
  'allo.compose'``, which reads exactly like a broken design. Measured
  2026-09-25 over two full harness runs. **Put the worktree under ``/home``.**
  Copying a prebuilt ``mlir/build`` there is not enough either: the bindings
  carry an RPATH into the old build directory, so inside the sandbox they raise
  ``ImportError: libAlloMLIRAggregateCAPI.so.22.0git``. Build ``mlir/`` in the
  worktree that will run the harness.

Running two tracks on one host: ``ray stop`` is global
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**``ray stop`` matches Ray processes by name across the entire host**, so it
kills raylets belonging to every worktree rather than only the one it is run
from. A track that ends a run or a swarm with it takes down whatever else is
mid-flight.

Observed on 2026-09-22: a paid two-worker search lost both workers 10.8 minutes
in, at $1.69, during their first model call, with ``raylet.out`` showing
``received SIGTERM`` and ``Raylet graceful shutdown triggered ... reason:
EXPECTED_TERMINATION``. Not memory — 757 GB was free. The GCS survived; the
raylet did not.

There is **no per-worktree isolation** for this. Matching is on process name, so
a private temporary directory or a distinct port does not help. Two options, and
the choice is operational rather than technical: serialise Ray use across
tracks, or treat a raylet ``SIGTERM`` as transient and retry, accepting the
wasted spend.

The failure direction is at least safe: workers die rather than producing an
unscored candidate. And a harness that calls ``ray stop`` unconditionally on
teardown makes concurrent tracks impossible on one host — prefer stopping only
your own cluster by address, or simply letting the process exit.


Results and history
-------------------

Every measurement, finding, correction and retraction is on two separate
pages, so that this one stays usable as a manual:

- :doc:`/extensions/chia_results` -- what the loop has found, what is and is
  not demonstrated, the paid runs, the planned experiments, and the
  corrections.
- :doc:`/extensions/chia_codesign` -- the retired ``chia-codesign`` effort,
  which searched a different and older design against a cost model, with its
  claim register and its own setup instructions.
