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

An LLM-agent loop that co-designs a TPU's **instruction set and
microarchitecture together**, where a proposal counts only if a real tool --
RTL co-simulation and synthesis -- measures it. Agents are driven by
`CHIA <https://github.com/ucb-bar/chia>`_ (``ucb-bar/chia`` at ``16c35e9``)
through the ``opencode`` CLI.

The principle
-------------

An agent can make an informed co-design decision only after it has measured how
a software function performs -- in **timing and power** -- across **an array of
hardware architectures**. Measured against that, today's loop has the first
half of each:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * -
     - have
     - missing
   * - **timing**
     - RTL cosim cycles, bit-exact, deterministic
     - --
   * - **power**
     - DC estimate, default toggle rates, *indicative only*
     - activity-based power (switching from cosim into synthesis)
   * - **architectures**
     - one design family (TinyTPU-isa, T=4/T=8), two substrates (FPGA,
       45 nm), two references (Gemmini, MiniTPU)
     - the parametrized IP library that would supply a real array

Why the array matters is already measured: a DMA widening that is free on an
FPGA emits a dual-write-port memory that standard cells cannot build, and only
the second substrate said so.

Takeaways
---------

- **It works end to end on real tools.** Every accepted number is RTL cosim
  cycles plus csynth area and clock, bit-exact against a frozen reference
  model. About **$85** spent on the CHIA2026 account so far; the harness is
  also exercised with no model at all.
- **It finds real design changes, not yet novel ones.** Best so far: a DMA
  burst widening worth 55-61 % of the steady-state gap to Gemmini (not
  landed; its banked form synthesises with no cycle loss).
- **Two things are not yet shown:** a *rate* of discovery (every search is
  n=1 per arm) and an agent-authored **architectural** abstraction (zero so
  far -- the abstraction track produced apparatus findings instead).
- **The recurring hazard is instruments that fail open.** Four in one night
  reported success after failing. A negative result counts only if the
  instrument can be shown to have run.

Contributions
-------------

- **A judge agents cannot fool.** Nine mechanical guards -- most added after an
  agent found the hole each closes -- mean a win is accepted only when real
  RTL measures it, which is why a worker's false win claim re-scored as
  exactly baseline.
- **A pre-registered search result.** The unguided ``open`` arm, told only to
  choose its own target from the numbers, independently reached the same
  address-generator widening that directed analysis had found, with the
  refusal bottleneck migrating and cycles flat exactly as predicted in
  advance.
- **Why agent-built compiler extensions are hard to evaluate.** A new
  primitive has no callers, so an agent's abstraction scored neutral *and*
  passed 291 tests while aborting the compiler on first use -- any gate ladder
  that exercises a compiler only through existing designs cannot see a new
  capability in either direction.

How the loop is built
---------------------

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

The rest of this page is the comprehensive record: where the code lives, the
evaluator and every guard with the exploit that bought it, how to run it, each
paid run, and the planned experiments. Superseded figures and withdrawn claims
are at the foot of the page.


Where the Code Lives
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Branch
     - Contents
   * - ``main``
     - ``examples/accelerator/tinytpu_vitis/chia_agent/``: the loop for
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


.. _chia-isa-loop:

The CHIA Loop for TinyTPU-isa
-----------------------------

The search runs over the **current** TinyTPU design in
``examples/accelerator/tinytpu_vitis/``: int8/int32, 4x4 weight-stationary
array, PC + 4-deep loop stack + 3-term AGU, GEMM programs generated by
``isa_dsl.py``, shipping at **171 / 261 / 417 / 483 / 685** cosim cycles
(:doc:`/designs/tinytpu_isa`). Workers propose edits, the harness gates and
cosims each one itself, and the best survives. The ``chia-codesign`` loop
searched a different, older design against a cost model; this one is scored
by RTL cosim.

**Where it lives.** ``examples/accelerator/tinytpu_vitis/chia_agent/`` on
``main`` (landed 2026-09-19 from branch ``chia-isa``, now deleted). Its
``README.md`` is the operator's manual; ``dev/records/tinytpu/chia-evidence/`` holds the
trimmed evidence behind every number below. The branch's full history and the
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
     - ``microarch_isa.py``, ``isa_dsl.py``
     - The agent edits a private copy in ``<run>/<worker>/spec/``, never the
       repository.
   * - **frozen**
     - main's ``cosim.py`` (testbench, ``SHAPES``, golden reference, every
       Vitis/TCL setting), ``bench_isa.py``, ``stress_isa.py``,
       ``isa_ref.py``, ``kpn_model.py``; and ``chia_agent/``'s
       ``evaluate.py``, ``gate_runner.py``, ``param_check.py``,
       ``spec_policy.py``
     - Read from git, never from disk; main's five files must also be
       byte-identical to ``MAIN_BASE`` (``476a70d8``) in ``evaluate.py``.

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
  off the same build (guard 10); the published 172 / 262 / 418 / 484 / 686
  (``dev/records/tinytpu/chia-evidence/accept-control-476a70d8/``) are what that measurement is
  cross-checked against.

Guards
~~~~~~

Mechanical enforcement, not instructions:

1. **Tool surface.** opencode's own file and shell tools are denied; the MCP
   edit tools accept only the two bare file names.
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
   `Earlier measurements and corrections`_.
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
    ``control`` -- cycles, the two editable files' blob ids, the ref, the
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


Running it
~~~~~~~~~~

Environment (once): the ``allo`` env with this checkout's ``mlir/build``,
``chia_env`` (py3.10, ``requirements.txt``), opencode via
``npm ci --prefix examples/accelerator/tinytpu_vitis/chia_agent``, and
``chia.env`` copied from ``chia.env.example`` at the repository root.
``chia.env`` is gitignored and must never be committed. Then:

.. code-block:: bash

   examples/accelerator/tinytpu_vitis/chia_agent/gcp_setup.sh  # auth, project, billing, APIs, spend report
   conda activate chia_env; set -a; source chia.env; set +a
   ray start --head --resources='{"opencode_creds": 2}' --include-dashboard=false
   cd examples/accelerator/tinytpu_vitis/chia_agent
   python test_harness.py --phases e,c    # ~1 min, $0; full suite ~30 min, $0
   python preflight.py --budget-usd 30    # the gate alone, $0
   python swarm.py --workers 2 --iterations 3 --budget-usd 30
   python accept.py --out <run>/control                 # the run's control, once
   python accept.py --diff <run>/<worker>/best.diff --out <run>/accept-<worker> \
       --control <run>/control/accept.json              # or omit it and measure again
   ray stop                               # and remove the Ray session directory

**Billing.** CHIA runs on the dedicated project ``chia2026-tinytpu``, billed to
the CHIA2026 account; everything is scoped per process, and the host's global
gcloud config and ADC quota project are never changed. opencode's
``google-vertex`` provider takes the project from the provider options
``loop.py`` sets (then ``GOOGLE_VERTEX_PROJECT``, ``GOOGLE_CLOUD_PROJECT``)
and bills the project in the request URL. ``preflight.py`` runs before any
worker or model call in ``swarm.py``, ``loop.py`` and ``smoke.py``, and refuses
unless the project bills ``CHIA_BILLING_ACCOUNT``, Vertex AI is enabled,
``--budget-usd`` is given, and CHIA's cumulative spend so far plus this run's cap
fits **``CHIA_TOTAL_CAP_USD`` ($100 in chia.env)**; it prints the account,
project, spend so far, remaining and the run's cap. Cumulative spend is
opencode's own database, attributed to CHIA2026 by the cutover time recorded
in ``billing.json`` (opencode stores no GCP project with a session). These are
opencode's figures from its price table, not the invoice; the remaining credit
is only in the Cloud Console.

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

First paid run, 2026-09-19
~~~~~~~~~~~~~~~~~~~~~~~~~~

``dev/records/tinytpu/chia-evidence/isa-run1-20260919/``: 2 workers x at most 3 iterations, $30 cap,
``gemini-3.1-pro-preview`` on ``chia2026-tinytpu``, seeded with measured facts
from the shipped design's 16x16x16 timeline
(``dev/records/tinytpu/chia-evidence/timeline-476a70d8-16x16x16/``: 211 cycles of ``dma_ld`` and no MAC
before cycle 285; the drain after the last PE). The pre-flight gate passed
before any worker started.

.. list-table::
   :header-rows: 1
   :widths: 13 8 45 22 12

   * - worker
     - iter
     - what it tried
     - verdict
     - cost (DB)
   * - front-end
     - 1
     - widen ``dma_ld``'s operand bursts to 4 rows per iteration
     - 172 / 627, bit-exact, stress 492/492; accepted (-59)
     - $5.17
   * - front-end
     - 2
     - rewrite of the GEMM program generator
     - deadlock, gate timeout at 240 s
     - $8.78
   * - tail
     - 1
     - contiguous ``dma_st`` write-back (session timed out); debug session
       removed the leftovers
     - exactly baseline, 172 / 686 (+0)
     - $4.36 + $1.20
   * - tail
     - 2
     - relay kernels between ``accu`` and ``dma_st``
     - 176 / 686 (+4), rejected
     - $9.03

**$28.54** over five sessions (all counted against CHIA2026's $100 cap), 100
min wall; both workers stopped on the soft cap before iteration 3.

**The finding, verified independently.** ``accept.py`` on front-end iteration
1, on a clean checkout: **172 / 262 / 376 / 425 / 627** (0 / 0 / -42 / -59 /
-59, **-160** over the five shapes), bit-exact, ``stress_isa`` 492/492, RTL
stress 0 mismatches at every shape, 2.431 ns. It costs **2.3x the block RAM**
(BRAM18K 42 -> 98), +20% LUT and +40% FF: -59 cycles (8.6%) at 16x16x16. Read
by hand, the only functional change is ``dma_ld``'s burst loops; rows read past
a program's span stay inside the operand and land in buffer words the program
never names. The diff as written also **deleted the 260-line design
docstring** and hard-coded ``T = 4`` -- both invisible to the gate at the time,
and the reason for guards 6 and 7. Re-expressed parametrically with the
docstring intact (``dev/records/tinytpu/chia-evidence/isa-run1-20260919/param_burst.diff``, no model
call) it gives identical cycles and area, passes the same acceptance, and is
exact at MAXDIM 8 and 12. **It is not landed**: whether the burst widening is
worth its block RAM is a separate decision.

Two observations worth keeping:

- **A worker reported a false improvement.** The tail worker's debug session
  claimed "172 cycles (down from 680)" and "686 (down from 1457)"; its final
  diff was one added ``pass`` and the harness scored it exactly at the
  baseline. The loop never takes an agent's number, and this is why.
- **Timed-out sessions are billed but reported as $0.** Three of the five
  sessions hit opencode's 40-minute timeout. The loop's per-worker figures
  (from opencode's export of each call) record them as $0.00 -- $5.17 and
  $1.20 in the workers' logs -- while opencode's database charges them in full,
  **$22.17** here. Every cap therefore reads the database (``spend.py``), never
  the per-call usage.

A capped smoke run on the earlier design is recorded in
`Earlier measurements and corrections`_.


What Is and Is Not Demonstrated
-------------------------------

The evidence behind the takeaways at the top of this page, one line per claim.

**Demonstrated**

- **The loop runs end to end on real tools** (``chia_agent/`` on ``main``).
  Every accepted figure is RTL cosim cycles plus csynth area and clock,
  bit-exact against a frozen reference model.
- **It catches a false claim.** Run 1 (2026-09-19): a worker reported an
  improvement that re-scored as exactly baseline.
- **It finds a real change.** Run 1: a DMA burst widening, -160 cycles over five
  shapes, bit-exact, stress and RTL-stress clean, at 2.3x block RAM. Not
  landed; the dual-ported form does not synthesise to standard cells and the
  banked form does, with identical cycles (:doc:`/designs/benchmarks`).
- **A pre-registered prediction held.** Run 2 (2026-09-22, two arms): both arms
  passed every gate and raised the encodable count -- 3 to 8 for the directed
  arm, 3 to 7 for the unguided ``open`` arm -- while the chosen nest, and
  therefore the cycles, did not move and area rose. The ``open`` arm's first
  attempt made 16 nests encodable that computed the **wrong answer**, and the
  reference-model sweep caught all 16.
- **The retired ``chia-codesign`` claims C1-C8** replay deterministically,
  including the two agent-found variants (4.07x and 1.98x) at exact numerics.

**Not demonstrated**

- **A rate of discovery.** Every search is n=1 per arm. A rate does not need a
  seed -- it is a property of the sampling process -- so replication (E1 below)
  is the next paid run.
- **An agent-authored architectural abstraction.** None yet. The abstraction
  track's one candidate was an implementation of an existing primitive, it
  leaked its answer through the documentation, and it aborts the compiler on
  first use (:doc:`/extensions/agentic_experiments`).
- **Novelty.** The burst widening is a sensible engineering change, not a
  discovery.

**Known defects, being repaired.** Run 2 was stopped after 2 of 5 iterations by
a per-run cap that summed the whole account rather than its own sessions. The
$0 guard suite, 57 cases at landing, does not currently pass (one stale
assertion, one crash in the loop phase).

The Planned Experiments
-----------------------

Written 2026-09-22, before the runs, so that the design of each experiment can
be read against its result rather than after it. **$300 is authorised for the
overnight runs and about $500 remains for the experiments below.** The split of
the overnight money is recorded in ``chia_agent/allocation.json``: the
abstraction-maintaining track is weighted 2:1 over the design-point search,
because the search's pipeline is proven while the abstraction work is the open
question. Cumulative spend was $28.54 when the split was made.

The gate enforces ``CHIA_TOTAL_CAP_USD`` as a *cumulative* ceiling on
``chia2026_spend()`` and cannot tell two concurrent tracks apart, so a track's
share is honoured by that track setting its own ceiling. Two tracks drawing on
one account means a track that sees spend climbing faster than its own runs
explain is seeing the other track, not an accounting bug.

The one thing every experiment below is designed to fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every claim this page makes about the TinyTPU-isa search is **n=1**. One paid
run found one improvement. That is enough to show the loop works and not enough
to say anything about how well it works, and no amount of further single runs
will change that. So the planned experiments buy *replication and rate* before
they buy anything else, and each one states in advance what result would count
as a negative.

E1. Rate of discovery, replicated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the design-point search from the same baseline **at least three times**
with different seeds, and report the distribution rather than the best run:
how many candidates were proposed, how many passed the static gates, how many
passed acceptance, and what each accepted candidate bought. The deliverable is
an accept rate with an interval around it.

A negative result here is publishable and should be reported as such: if two
of three runs find nothing, the honest claim is that the loop finds an
improvement *sometimes*, and the paper says so.

Prerequisite, and it is not optional: **seed the search**. Outstanding item 2
below has blocked this since 2026-09-07. Without a seed a run is re-runnable
but not repeatable, and a distribution over unrepeatable runs cannot be
attributed to the search rather than to sampling.

E2. Does the refusal bound the search?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Of 1,226 candidate loop nests, **1,150 are refused at ``acc``**, and *why* went
through two wrong diagnoses before the right one, which is itself worth
recording. It is not that ``acc`` is a static instruction field with no
predicate on an induction variable — this page said that, and it is false.
``acc`` is ``mm``'s ``f2``, ``f2`` is a legal AGU target, and driving it from
the reduce loop assembles for exactly two k-tiles before rejecting at Kt≥3. **The
obstacle is additive monotonicity in the address term**, not staticness (see
:doc:`/extensions/act`).

That correction matters because it changes what the fix costs: making a static
field dynamic is a different and more expensive change than saturating a term
or predicating it on ``iv_now[level] == 0``, and a search framed on the wrong
diagnosis would have priced the wrong hardware.

A second correction, to the arithmetic rather than the mechanism: a **first-cause
census hides overlap.** By first cause it is 1,150 ``acc`` / 55 ``ar-distance`` /
13 ``AGU_TERMS`` / 3 ``LOOP_DEPTH``, but **930 of those nests also violate the
accumulator RAW-distance contract**, so relieving ``acc`` alone does not free
1,150 nests. Sorted by the express/refuse distinction it is **1,166 express
against 55 refuse**: Allo can *build* nearly all of these, and our machine
cannot *encode* most of them.

The experiment: run the search unchanged, then against a design that relieves
the monotonicity constraint, and compare what each finds. If the second finds
strictly better design points, the refusal is a real bound and the ISA is the
thing to fix. If it finds nothing better, the reachable nests already contain
the good designs — also a result, and a cheaper one to act on.

There is already evidence for the second outcome, which should be stated before
the experiment rather than after: widening ``AGU_TERMS`` from 3 to 4 raises the
encodable count and **does not change the chosen nest**, so the RTL runs the
same stream, the cycles do not move, and the area rises. A cost with no benefit.
If relieving ``acc`` unlocks 1,150 nests and the pick *still* does not move,
then the mapping space was never the binding constraint on this machine — a
stronger and more surprising claim than a cycle win, and the one this experiment
is now most likely to produce.

E3. Can an agent maintain the abstractions, and at what rate?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The track this project most wants to measure, and the one expected to fail most
often. An agent is asked to extend Allo itself -- a dialect operation, a type, a
schedule primitive or a pass -- rather than to edit a design. Success means the
extension arrives the way ``s.dependence(...)`` did: with its analyses, its
legality rule, tests, and the golden dataflow tests still passing.

Bounded attempts with a hard pass/fail gate, not one long run, because the
result wanted is the *shape of the success rate* and its failure modes, not one
expensive lucky sample. Keep every transcript: when an agent cannot extend a
compiler abstraction, *why* it could not is the evidence.

The stated hypothesis, from the CAKE result (a typed IR reaching 1.144x where
raw generation reached 0.928x at equal budget): an agent given a typed
abstraction with construction-time checking succeeds more often than one given
free rein over the emitter. Testing that needs both arms, so run both.

E4. Co-design, both sides moving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above moves one side at a time. The claim the paper wants is
co-design: a search that changes the instruction set and the microarchitecture
together, where neither change is worth anything alone. The evidence for it is
a design point plus the demonstration that ablating either half loses the gain.
That ablation is the experiment, and it is cheap once a candidate exists --
it is two extra evaluations, no model calls.

What will not be spent on
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Breadth for its own sake.** A fourth hypothesis at n=1 is worth less than a
  second run of an existing one.
- **Re-deriving numbers that are already recorded.** The evidence directories
  hold every accepted diff with its reports; replay is free.
- **The semantics-alignment variant.** Stopped; see
  :doc:`/designs/minitpu`.

E5. Held-out rediscovery
~~~~~~~~~~~~~~~~~~~~~~~~

An extension does not have to be novel to be evidence. If an agent reaches an
abstraction the fork already has, **without being told it exists**, that is a
measurable result and a much cheaper one to grade than novelty, because the
right answer is already in the tree with its tests.

The design: take a fork-local primitive whose history is known -- the worked
example is ``s.dependence(...)``, which exists because a real defect could not
be expressed any other way -- remove it from the agent's view along with the
documentation that names it, and give the agent only the symptom that motivated
it. Grade on whether the agent arrives at an abstraction with the same power,
and on what it proposes instead when it does not.

This is the one experiment in this list with a known correct answer, which makes
it the right place to calibrate how much guidance an agent needs before the
open-ended attempts in E3 are worth paying for.

Why the design driver's end state matters to all of this
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The target is not one accelerator. It is **a library of parametrized, modular
TPU IPs that compose into different architectural choices** -- the named class
being Groq's LPU, OpenAI's Jalapeno, Meta's MTIA and AMD's XDNA -- with TinyTPU
as something such a library *instantiates* rather than something to extend.

That is what "generalizable across design cases" has to mean here, and it is
the standard a proposed extension should be judged against: an abstraction that
makes a second architecture expressible is worth more than one that makes the
current design faster. It also tells E3 and E5 what to reward. An agent that
parametrizes an IP block so it can be composed differently has done the thing
the project wants, even if the immediate design gets no faster.


Earlier measurements and corrections
------------------------------------

A superseded run, a withdrawn claim, and the earlier co-design effort that
this one replaced. The current state is in the sections above.

Earlier: capped smoke run, 2026-09-19, old design, old project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dev/records/tinytpu/chia-evidence/isa-smoke-20260919-035443/``: 2 workers against the design shipped
until ``e24e433b`` (252 / 383 / 591 / 667 / 919), $15 hard cap, billed to the
general project ``test-adrs``. **$15.15**, 56 min, **no candidate completed**:
opencode timed MCP calls out at 60 s while a cosim takes minutes, a hung
evaluator blocked the tool server so the next session saw no tools, a
deadlocked candidate held the gate for 900 s, CHIA silently retried a
40-minute prompt, and unified diffs were the agents' main failure mode. All
five were fixed before run 1 (40-minute MCP timeout, async tools, 240 s gate
timeout with process-group kill, ``retries=1``, ``replace_text``).

Withdrawn: "T is not varied"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An earlier revision of this page said "T is not varied: the shipped design
supports T=4 only", in guard 6 of `The CHIA Loop for TinyTPU-isa`_.
**That was wrong.** The measurement that refutes it is in that guard.
``param_check.py``'s own docstring described the column-block-2 problem
correctly all along; the conclusion drawn from it was the error.

.. _chia-codesign:

The Original Co-Design Effort (``chia-codesign``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"Agentic discovery of a domain-specific computing stack": an AI agent proposes
changes to a TPU-style accelerator's instruction set and microarchitecture at
once, and every proposal is scored by running a real commercial synthesis tool.
One specification generates the simulator, the hardware, and the compiler
backend, so a single agent edit moves all three together. The design searched
here is the fp32 ``examples/accelerator/tinytpu/`` on ``chia-codesign`` -- a
different machine from TinyTPU-isa, sharing no code with it (see
:doc:`/extensions/act`, "What ACT Is").

Headline figures, as stated by ``CODESIGN.md``:

.. list-table::
   :widths: 40 60

   * - Best agent result
     - **4.07x** fewer cycles (126,432 -> 31,056), numerics exact
   * - Second, independent
     - **1.98x** at **zero** extra BRAM
   * - Cost model correction
     - prior hand-written model was **5.7x optimistic**
   * - One evaluation
     - **41 s**, $0 -- synthesis runs locally
   * - One agent candidate
     - ~8-15 min, ~$2-3
   * - Verify both results
     - **~85 s**, two commands

.. code-block:: bash

   ./scripts/claims.sh --fast     # 13s, no synthesis, no API
   ./scripts/claims.sh            # 133s, adds Vitis HLS
   ./scripts/claims.sh --full     # 196s, adds backends + one $0.03 agent round-trip

The five ideas
^^^^^^^^^^^^^^

1. **One specification, three artifacts.** The spec generates simulator,
   microarchitecture, and compiler backend. The agent edits one file and all
   three move -- which is how a single candidate changed the ISA, the compiler's
   lowering, and the datapath together.
2. **The verifier must be the real tool.** The pre-existing hand-written cost
   model was optimistic by 5.7x: ``mxu`` costs 72 cycles, not 36 (it synthesizes
   as two sequential passes, II=2), and each DRAM access pays the real 75-cycle
   ``m_axi`` latency, not an assumed 8. Every decision taken against that model
   was taken against a fiction. Cheap proxies do not merely add noise -- they
   point the wrong way.
3. **A cycles-only objective ranks designs backwards.** The 4.07x variant costs
   **4.7x the BRAM**, so per BRAM it is *worse* than the design it replaced. The
   1.98x variant costs none. Scored on cycles alone, the loop prefers the one
   that consumes the machine.
4. **Breadth beats depth.** Yield across hypotheses spanned ~100x (4.07x vs
   3.2%). Another hypothesis is worth far more than another iteration on the
   incumbent -- but see `Not yet reproducible`_ before treating this as
   measured.
5. **Evaluation is cheap relative to proposal.** 41 s and $0 to evaluate; 8-60
   min and $2-3 to propose. That inverts the usual design-space-exploration cost
   model, where simulation dominates and effort goes into surrogates and caches.
   Here, caching evaluation would be optimizing the 1% side.

Claims: what is demonstrated and what is not
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CODESIGN.md`` is the canonical claim register and splits its claims into two
tiers, because they are not equally strong.

**Deterministic -- demonstrated.** Reproduced by replay and re-synthesis;
identical numbers every run.

.. list-table::
   :header-rows: 1
   :widths: 6 50 34 10

   * - #
     - Claim
     - Command
     - Time
   * - C1
     - One schedule lowers to CPU, Vitis HLS, and CIRCT RTL
     - ``make -C examples/accelerator/tinytpu oracle compiler cpu hls rtl``
     - 29 s
   * - C2
     - Frozen cost model scores 22,160 cycles; synthesis-grounded scores
       126,432 (**5.7x**)
     - ``./scripts/claims.sh --fast``
     - 7 s
   * - C3
     - Synthesis measures ``mxu`` = 72 cyc at II=2; ``dma_load`` depth = 75
     - ``python -m examples.accelerator.tinytpu.synth``
     - 35 s
   * - C4
     - Derived ``(ii, depth)`` reproduces the tool's measured latency exactly
     - ``pytest tests/dsa/test_tinytpu_synth.py``
     - 4 s
   * - C5
     - A self-modifying spec is refused; the shipped spec passes the same
       policy
     - ``pytest tests/dsa/test_tinytpu_agent_policy.py``
     - 3 s
   * - C6
     - The **4.07x** variant replays to 31,056 cycles, numerics exact
     - ``verify_variant.py --run <run> --worker dram``
     - 42 s
   * - C7
     - The **1.98x** variant replays to 63,864 cycles at baseline BRAM
     - ``verify_variant.py --run <run> --worker granularity-retry``
     - 43 s
   * - C8
     - Agent path is live end to end (ADC -> Vertex -> opencode -> CHIA ->
       MCP -> allo)
     - ``python chia_agent/smoke.py``
     - 32 s, $0.03

**Stochastic -- not demonstrated.** An agent search is not deterministic.
Re-running it will not necessarily rediscover the same design. What C6 and C7
establish is that the designs the search *did* find are real and re-derivable
from the recorded trace. What is **not** yet established is the rate at which a
search finds them.

.. list-table::
   :header-rows: 1
   :widths: 6 54 40

   * - #
     - Claim
     - Status
   * - S1
     - A 4-worker x 5-iteration search finds >=2x with probability *p*
     - **n=1.** Needs >=3 seeds per hypothesis.
   * - S2
     - Yield varies ~100x across hypotheses (idea 4)
     - **n=1 per angle.** Suggestive, not measured.
   * - S3
     - Accept rate, and cycles-vs-iteration trajectory
     - **Not yet analysed.** Raw logs in ``chia_runs/``.

**Observations, not claims** (``CHIA_CHECKPOINT.md`` section 2): things
observed once, which nothing reproduces on demand.

.. list-table::
   :header-rows: 1
   :widths: 6 47 47

   * - #
     - Observation
     - Why it is not a claim
   * - O1
     - A live agent wrote self-modifying code into the spec, which the
       evaluator then imported and executed
     - Real, and now guarded -- but the original artifact was reverted by
       ``git checkout`` and never captured. Demonstrating it again would mean
       re-eliciting it.
   * - O2
     - CHIA serializes every LLM call on ``opencode_creds``; one unit caps the
       whole cluster
     - Verified by observation (``4.0/4.0`` after raising it). No test asserts
       it.
   * - O3
     - Vertex preview-model quota, not the harness, is the parallelism ceiling
       -- 3 of 4 workers died on ``RateLimitError`` before proposing anything
     - Seen in worker logs. The quota API does not expose the effective preview
       limit, so it cannot be confirmed independently.

Not yet reproducible
""""""""""""""""""""

- **S1-S3 need replication.** Every hypothesis has n=1. Idea 4 is the project's
  most interesting claim and currently its least evidenced.
- **The search is not seeded.** Re-running produces a different trajectory.
  Only the replay path (C6, C7) is deterministic.
- **No permanent archive.** For an *Artifacts Available* badge this needs a DOI
  (Zenodo), not just a git branch.
- **Agent runs need external services.** Vertex AI and a billing account; C1-C7
  need none of that, which is why the claim tiers are split the way they are.

Setup from zero
^^^^^^^^^^^^^^^

.. code-block:: bash

   # 1. system packages (Ubuntu 24.04)
   sudo apt install -y build-essential cmake ninja-build gcc g++ lld nodejs npm
   mkdir -p ~/.local/allo-bin && ln -sf /usr/lib/llvm-18/bin/ld.lld ~/.local/allo-bin/
   export PATH=~/.local/allo-bin:$PATH        # cmake wants `ld.lld`; Ubuntu ships `ld.lld-18`

   # 2. Vitis HLS 2023.2 at /opt/xilinx/Vitis_HLS/2023.2 (C-synthesis needs no licence)

   # 3. source and build  (~40 min on 144 cores, ~35 GB)
   git clone git@github.com:sunwookim028/allo.git ~/allo
   cd ~/allo && git checkout chia-codesign && git submodule update --init --recursive
   conda create -y -n allo python=3.12 && conda activate allo
   pip install "nanobind>=2.10,<3" "PyYAML<=6.0.1" typing_extensions \
               "scikit-build-core>=0.10" "setuptools_scm>=8" pytest numpy ml_dtypes rich sympy
   export CMAKE_POLICY_VERSION_MINIMUM=3.5    # cmake 4 rejects OR-Tools 9.5's fetched deps
   bash scripts/build-mlir.sh  externals/llvm-project Release gcc g++
   bash scripts/build-circt.sh externals/circt externals/llvm-project/build Release gcc g++
   SKBUILD_CMAKE_DEFINE="CMAKE_C_COMPILER=gcc;CMAKE_CXX_COMPILER=g++" \
   CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$PWD/externals/circt/ext" pip install -v -e .

   # 4. CHIA host (only needed to *run* agents; not needed to verify claims)
   #    Both dependencies are pinned by manifests tracked in the branch.
   conda create -y -n chia_env python=3.10.19
   conda run -n chia_env pip install -r examples/accelerator/tinytpu/chia_agent/requirements.txt
   npm ci --prefix examples/accelerator/tinytpu/chia_agent

   # 5. Google Cloud (only needed to run agents)
   gcloud auth login && gcloud config set project <PROJECT_ID>
   gcloud services enable aiplatform.googleapis.com
   gcloud beta billing projects link <PROJECT_ID> --billing-account=<XXXXXX-XXXXXX-XXXXXX>
   gcloud auth application-default login
   gcloud auth application-default set-quota-project <PROJECT_ID>

   # 6. environment, in one file
   cp scripts/chia.env.example chia.env && $EDITOR chia.env
   #    set GOOGLE_CLOUD_PROJECT, and TINYTPU_ENV if you named the conda env
   #    something other than `allo`

   # 7. check
   ./scripts/claims.sh --fast

``claims.sh`` refuses to run if the conda environment imports ``allo`` from a
different checkout than the one you are standing in. An editable install
resolves through a meta-path finder that outranks ``PYTHONPATH``, so without
that check every claim passes while exercising someone else's tree -- a failure
that looks green. Set ``TINYTPU_ENV`` to the environment you built *this*
checkout into.

The three non-obvious bootstrap fixes (committed in ``1f746005``; these are the
parts that break a naive rebuild):

1. ``lld`` must be reachable as ``ld.lld`` (Ubuntu ships ``ld.lld-18``) -- the
   ``~/.local/allo-bin`` shim above.
2. cmake 4 refuses OR-Tools 9.5's fetched deps; ``build-circt.sh`` exports the
   shim ``CMAKE_POLICY_VERSION_MINIMUM=3.5``.
3. **Build Allo with gcc, not clang.** Under clang, the ``enable_if`` attribute
   on ``llvm::StringLiteral`` makes it non-convertible for ``std::optional``'s
   converting constructor, so the HLS emitters fail to compile.
   ``pyproject.toml`` selects gcc.

Wall time on a 144-core host: LLVM ~12 min, CIRCT ~15 min, OR-Tools ~10 min,
Allo ~2 min.

Budgets
^^^^^^^

.. list-table::
   :widths: 40 60

   * - Build from scratch
     - 11-40 min, mostly LLVM; varies with machine load
   * - Disk, built tree
     - **9.2 GB**
   * - Disk, peak during build
     - ~19 GB (a fresh clone before ``ninja`` prunes intermediates)
   * - Disk, conda envs
     - 1.5 GB (``allo``) + 0.6 GB (``chia_env``); opencode adds 0.7 GB in-tree
   * - Verify all deterministic claims
     - 196 s, $0.03
   * - One agent candidate
     - ~8-15 min, ~$2-3
   * - One 4-worker x 5-iteration search
     - ~3 h, ~$100-190
   * - Rate
     - **$19 / agent-hour** (Vertex, ``gemini-3.1-pro-preview``)

Everything except ``chia_runs`` (85 MB) is reproducible from the repository, so
a machine short on space can delete the build trees and rebuild in ~40 minutes.
``chia_runs`` is the one directory that cannot be regenerated: it holds each
candidate's diff, synthesis report, and cost telemetry, and it is what
``verify_variant.py`` replays.

Preview-model **quota**, not credit, is the concurrency ceiling: an early
4-worker run lost 3 workers to ``RateLimitError`` before any proposed a
candidate.

Repository map
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Path
     - What it is
   * - ``examples/accelerator/tinytpu/isa.py``
     - ISA semantics -- memories, instruction access/compute.
       **Agent-writable.**
   * - ``examples/accelerator/tinytpu/microarch.py``
     - Synthesizable units + fetch-decode-dispatch top + schedules.
       **Agent-writable.**
   * - ``examples/accelerator/tinytpu/synth.py``
     - Runs Vitis C-synthesis; rebuilds the per-unit latency table from the
       report
   * - ``examples/accelerator/tinytpu/ppa.py``
     - The objective: synthesize, re-measure, score cycles
   * - ``examples/accelerator/tinytpu/verify_variant.py``
     - Replays a recorded variant and re-derives its score
   * - ``examples/accelerator/tinytpu/chia_agent/``
     - The search: ``loop.py`` (one search), ``swarm.py`` (many),
       ``spec_policy.py`` (containment), ``smoke.py``
   * - ``chia_runs/``
     - Every candidate ever scored, with diff, synthesis report, and cost
       telemetry
   * - ``scripts/claims.sh``
     - Runs the claims above
   * - ``notes/CHIA_CHECKPOINT.md``
     - Version pins for every component behind these numbers

The agent may edit **only** ``isa.py`` and ``microarch.py``. It cannot reach the
compiler, the runtime, the benchmarks, or the evaluator -- and because those two
files are *imported* by the evaluator, ``spec_policy.py`` refuses edits that
would execute at import. That is a policy check, not a sandbox; real isolation
means a container.

Component inventory (checkpoint of 2026-09-07)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From ``notes/CHIA_CHECKPOINT.md`` on ``chia-codesign``, which pins what version
of everything produced the results above.

.. list-table::
   :header-rows: 1

   * - Component
     - Version / pin
     - Remote
   * - Allo fork (integration HEAD at the time)
     - ``e78bf9b5`` on ``main``
     - ``sunwookim028/allo``
   * - Allo + ACT (Kai Shao)
     - ``0b5fef7`` -- reachable from ``chia-codesign`` and on
       ``origin/chia-tinytpu-rtlgen``
     - ``kkkaishao/allo``
   * - Allo + ACT + RTLGen + CHIA
     - ``chia-codesign``
     - ``sunwookim028/allo``
   * - commit it forks from
     - ``882f7dd6``, tag ``tinytpu-rtlgen-base``
     - ``sunwookim028/allo``
   * - CHIA framework
     - ``16c35e9``, pinned in ``chia_agent/requirements.txt``
     - ``ucb-bar/chia``
   * - opencode CLI
     - ``opencode-ai@1.18.25``, pinned in ``chia_agent/package-lock.json``
     - npm

Submodule pins (``externals/``), all built:

.. list-table::
   :header-rows: 1

   * - Submodule
     - Commit
     - Build artifact
   * - llvm-project
     - ``040a641988f6`` (LLVM 23.0.0)
     - ``build/bin/mlir-opt``
   * - circt
     - ``af5369d7ea19``
     - ``build/bin/circt-opt``
   * - marl
     - ``b8406ab0a825``
     - --
   * - past-python-bindings
     - ``65f989b86750``
     - --
   * - OR-Tools 9.5 (fetched by CIRCT)
     - v9.5 tarball
     - ``circt/ext/lib/cmake/ortools``

Toolchain:

.. list-table::
   :header-rows: 1

   * - Item
     - Version
     - Notes
   * - Vitis HLS / Vitis / Vivado
     - 2023.2
     - ``/opt/xilinx/*/2023.2``; ``settings64.sh`` must be sourced
   * - Target part
     - ``xcu55c-fsvh2892-2L-e``
     - Alveo U55C, 300 MHz target
   * - gcc / clang
     - 13.3.0 / 18.1.3
     - **build with gcc**
   * - cmake / ninja
     - 4.2.1 / 1.11.1
     - cmake 4 needs the OR-Tools policy shim
   * - lld
     - ``ld.lld-18``
     - shimmed to ``ld.lld`` in ``~/.local/allo-bin``
   * - conda env ``allo``
     - Python 3.12.13
     - editable install
   * - conda env ``chia_env``
     - Python 3.10.19, ray 2.54.0
     - CHIA host

Cloud: model ``google-vertex/gemini-3.1-pro-preview``, location ``global``,
authenticated with user Application Default Credentials; spend to date at the
checkpoint was $186.44. User ADC expires, so unattended runs should move to a
service account. Project and billing identifiers are deliberately not recorded
here.

Outstanding work, in priority order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the checkpoint (2026-09-07):

1. **Replicate each hypothesis >=3x.** Every angle is n=1, which makes idea 4 --
   the project's most interesting claim -- its least evidenced. This is the
   single highest-value use of further budget (~$300-600).
2. **Seed the search**, so a run is repeatable rather than merely re-runnable.
   Until then only the replay path is deterministic.
3. ``scripts/bootstrap_chia.sh`` doing the bootstrap end to end, plus
   ``environment-*.yml`` exports so the two conda environments are captured
   rather than tribal.
4. Deposit an archive with a DOI (Zenodo) for *Artifacts Available*.
5. Switch to a service account for unattended runs; user ADC expires.
