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
   ray start --head --resources='{"opencode_creds": 2}' --include-dashboard=false
   cd examples/tinytpu/chia_agent
   python test_harness.py --phases e,c    # ~1 min, $0; full suite ~30 min, $0
   python preflight.py --budget-usd 30    # the gate alone, $0
   python swarm.py --workers 2 --iterations 3 --budget-usd 30
   python accept.py --out <run>/control                 # the run's control, once
   python accept.py --diff <run>/<worker>/best.diff --out <run>/accept-<worker> \
       --control <run>/control/accept.json              # or omit it and measure again
   ray stop                               # and remove the Ray session directory

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
     - the fourteen paths ``chia_agent/design.py`` names: ``microarch_isa.py``
       (the parameter set), ``isa_dsl.py`` (the program generator),
       ``ip/isa.py``, ``ip/tinytpu.py``, ``ip/assembler.py``,
       ``ip/programs.py``, and the eight units under ``ip/units/``
     - The agent edits a private copy in ``<run>/<worker>/spec/``, which
       mirrors the package, never the repository. The hardware left
       ``microarch_isa.py`` for ``ip/units/`` when the design became a unit
       library, so an editable set that stopped at the two old file names
       would no longer contain the machine -- both wins of run 1 landed in
       what is now ``ip/units/dma_load.py`` and ``ip/units/sequencer.py``.
   * - **frozen**
     - main's ``cosim.py`` (testbench, ``SHAPES``, golden reference, every
       Vitis/TCL setting), ``bench_isa.py``, ``stress_isa.py``,
       ``isa_ref.py``, ``kpn_model.py``, ``shapes.py``, ``isa_spec.json``,
       ``isa_encoding.py``, ``gen_isa.py``; the design's own machinery
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

Because ``isa_ref.py`` is frozen and ``stress_isa.py`` checks programs against
it, **what each instruction means is part of the contract**: the agent may
change how the hardware executes the ISA and which instructions the generator
emits, not an instruction's semantics.

The evaluator
~~~~~~~~~~~~~

- **gate** (functional, Allo simulator, ~30 s): ``bench_isa.py`` (the published
  [-4, 4] setup), main's ``stress_isa.py`` (492 runs: full-range, corner and
  boundary int8 at all 64 shapes, ``C`` prefilled and compared in full, vector
  and random programs against ``isa_ref``, many calls on one build), and
  ``param_check.py`` at ``TPU_MAXDIM=8``, ``12`` and ``TPU_T=8
  TPU_MAXDIM=32`` (below). Negative control:
  an int16 partial sum passes ``bench_isa.py`` and every cosim testbench, and
  ``stress_isa`` rejects it (251/492 exact).
- **score**: the sum of RTL cosim cycles (Vitis HLS 2023.2 csynth + xsim) at
  4x4x4 and 16x16x16, each testbench bit-exact, the 3.33 ns clock met. About
  2-3 min per candidate. Area is recorded, not scored.
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
6. **Parametricity.** The scored configuration is T=4, MAXDIM=16, so a design
   specialised to it would pass everything. ``param_check.py`` rebuilds the
   candidate at ``TPU_MAXDIM=8``, ``TPU_MAXDIM=12`` and
   ``TPU_T=8 TPU_MAXDIM=32``, requires the build to report that configuration,
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
   is scrubbed before cosim (``m_axi_latency`` 0, as Gemmini's harness), T == 4,
   MAXDIM == 16, the 3.33 ns target, and each shape's own cosim log and
   simulated time must agree with the reported cycles.
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

The $0 harness test
~~~~~~~~~~~~~~~~~~~

``test_harness.py`` runs the whole harness with no LLM (a scripted
OpenAI-compatible model on localhost; every cloud credential variable is
cleared): **control** the control record a claim may rest on and its
cross-check; **e** frozen-file and import-time attacks, forged verdicts,
sandbox, and that a candidate can neither write nor fake a control record;
**c** an int16 partial sum rejected by stress; **g** the
parametricity and documentation guards; **d** a deadlock killed at 240 s;
**abf** no-op and a slower design scored concurrently; **loop** the real
``swarm -> loop -> opencode -> MCP`` path; **accept** ``accept.py`` on a
correct-but-slower diff, against a control measured in that same run. 57/57
at landing (``dev/records/tinytpu/chia-evidence/harness-test-20260919-190240/``).

Limits and known failures
-------------------------

- **Every search so far is n=1 per arm.** No rate of discovery is
  demonstrated; see :doc:`/extensions/chia_results`.
- **Run 2 was stopped after 2 of 5 iterations** by a per-run cap that summed
  the whole account rather than its own sessions.
- **The $0 guard suite, 57 cases at landing, does not currently pass** (one
  stale assertion, one crash in the loop phase).
- **A ``tamper`` verdict on a run that should have been clean** is usually a
  person or another agent editing the checkout while a measurement was in
  flight, not the candidate (guard 4).
- **Timed-out opencode sessions are billed but reported as $0** by the
  per-worker figures, which is why every cap reads opencode's database
  (``spend.py``) instead.

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
