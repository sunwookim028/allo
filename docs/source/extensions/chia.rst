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

An LLM-agent search over an accelerator's **instruction set and
microarchitecture at once**: workers propose edits to a TPU-style design, a
harness gates and scores each proposal with a real synthesis or simulation tool
-- not an estimate -- and the best survives. The agents are driven by the
`CHIA <https://github.com/ucb-bar/chia>`_ framework (``ucb-bar/chia``, pinned
at ``16c35e9``) through the ``opencode`` CLI.

Two efforts share this idea, on two different designs and two different
branches. Read the claims of each against its own section:

- :ref:`chia-isa-loop` -- the current loop, on branch ``chia-isa``, searching
  the :doc:`/designs/tinytpu_isa` design and scored by **RTL cosim**. It has
  been run once, capped, and **no candidate completed**.
- :ref:`chia-codesign` -- the original effort, on branch ``chia-codesign``,
  searching an older fp32 TinyTPU and scored by Vitis C-synthesis. Its
  headline results replay deterministically; its claims about the *search*
  itself are n=1.

.. note::

   **Attribution.** The agent search (``chia_agent/``, ``chia_runs/``, the
   claim scripts) is Sunwoo Kim's. Most of the ``chia-codesign`` branch is
   **not**: it imports Kai Shao's CIRCT RTL generator, ACT / DSA compiler flow
   and ``allov2`` core re-architecture from https://github.com/kkkaishao/allo
   (503 of the 539 commits ``chia-codesign`` carries that ``main`` does not;
   see that branch's ``ATTRIBUTION.md`` and :doc:`/extensions/act`). The
   ``chia-isa`` loop carries over only ``chia_agent/`` -- none of the old
   TinyTPU and none of Kai Shao's code. ``ATTRIBUTION.md`` also records an open
   item: nothing in the ``chia-codesign`` tree records the terms under which
   Kai's code was imported, which it states should be settled before that work
   is published, submitted or landed on ``main``.


Where the Code Lives
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Branch
     - Contents
   * - ``chia-isa``
     - ``main`` plus ``examples/accelerator/tinytpu_vitis/chia_agent/``: the
       loop retargeted at TinyTPU-isa (``microarch_isa.py`` + ``isa_dsl.py``),
       and ``chia_runs/isa-smoke-20260919-035443/``. Newest commit at the time
       of writing: ``0fc06999`` (2026-09-19).
   * - ``chia-codesign``
     - A **separate codebase**, not a feature branch. It descends from Kai
       Shao's ACT fork and shares only a March 2026 ancestor (``76130c63``)
       with ``main``; the two are maintained separately and are not expected
       to converge (see :doc:`/developer/fork_maintenance`). Carries
       ``CODESIGN.md`` (the claim register), ``notes/CHIA_CHECKPOINT.md``
       (version pins), ``examples/accelerator/tinytpu/`` and ``chia_runs/``.

Neither ``CODESIGN.md`` nor ``CHIA_CHECKPOINT.md`` exists on ``main``; read them
with ``git show origin/chia-codesign:CODESIGN.md``.


.. _chia-isa-loop:

The CHIA Loop for TinyTPU-isa (``chia-isa``)
--------------------------------------------

The search runs over the **current** TinyTPU design in
``examples/accelerator/tinytpu_vitis/``: int8/int32, 4x4 weight-stationary
array, PC + 4-deep loop stack + 3-term AGU, 9 opcodes, GEMM programs generated
by ``isa_dsl.py``. Workers propose edits, the harness gates and cosims each one,
and the best survives. The ``chia-codesign`` loop searched a different, older
design against a cost model; this one is scored by RTL cosim.

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
     - ``cosim.py`` (testbench generator, ``SHAPES``, numpy golden reference,
       every Vitis/TCL setting), ``bench_isa.py``, ``chia_agent/stress.py``,
       ``chia_agent/evaluate.py``, ``chia_agent/spec_policy.py``
     - See below.

Mechanical enforcement, not instructions:

1. **Tool surface.** opencode's own file and shell tools are denied
   (``{"*": "deny"}``). The MCP tools are all the agent has. ``replace_text`` /
   ``apply_spec_patch`` / ``insert_after`` reject any path except the two bare
   file names. That includes a diff that also touches another file, and ``../``
   paths.
2. **Frozen files come from git, never from disk.** ``evaluate.py`` composes a
   fresh evaluation tree for every candidate. The frozen files come from
   ``git show HEAD:...``, the spec directory supplies only the two editable
   files, and any other file there is ignored. ``cosim.py`` and ``bench_isa.py``
   are also checked byte-identical to main @ ``e2451b81``. ``loop.py`` refuses
   to start if any frozen path is dirty in the working tree.
3. **Import-time code is policed.** The evaluator imports the two editable
   files, so ``spec_policy.py`` (itself executed from git) refuses file I/O,
   process spawning, ``exec``/``eval``, ``sys.modules``, dunder attribute
   access, ``os.environ`` writes, and any ``examples.*`` import other than the
   two spec modules. It runs at edit time and again at evaluation time.
4. **The memory model is not the candidate's.** Every ``TPU_*`` variable is
   scrubbed before ``cosim.py`` runs, so ``-m_axi_latency`` stays at its default
   0, which is the setting that matches Gemmini's harness (see
   :doc:`/designs/gemmini_comparison`). ``-random_stall`` and ``TPU_WRAP`` stay
   off too. The generated ``kernel.cpp``, ``run.tcl`` and logs are checked
   afterwards for interface-latency overrides.
5. **Premises are checked.** T == 4 and MAXDIM == 16. The csynth target stays
   3.33 ns, and the estimated clock must meet it.
6. **The number is cross-checked outside the candidate's process.** Each shape
   must appear exactly once, with ``mismatches = 0 / M*N`` in cosim.py's output
   *and* in that shape's own cosim log, plus a PASS. The cycle count must agree
   with the log's simulated time (+/-12 cycles). The work directory is wiped
   first, so a stale report cannot be read.

This is a policy, not a sandbox. Real isolation would mean a container.

The evaluator (two tiers)
~~~~~~~~~~~~~~~~~~~~~~~~~

- **gate**: ``bench_isa.py`` must print ``ALL EXACT``, and ``stress.py`` must
  pass (functional, Allo simulator, ~10 s). ``stress.py`` is new here. It uses
  full-range int8 operands, three seeds, five shapes beyond the scored five, and
  a sentinel-filled ``C`` that must survive outside the ``M x N`` result.
  Negative control: narrowing the PE partial sum from int32 to int16 passes
  ``bench_isa.py`` and all of cosim's testbenches (their [-4, 4] operands never
  overflow), and stress rejects it (20/60 runs wrong).
- **score**: the sum of **RTL cosim** cycles (Vitis HLS 2023.2 csynth + xsim)
  at 4x4x4 and 16x16x16, each testbench bit-exact. About 2.2 min per candidate.
  Area and clock are recorded but not scored.
- **acceptance**: ``accept.py`` is the only way a win is claimed. It makes a
  clean ``git worktree`` of HEAD, ``git apply``\ s the candidate diff, builds
  that checkout's own ``mlir/`` in-tree (~40 s), then runs ``bench_isa.py``,
  ``stress.py`` and ``cosim.py`` with no ``TPU_*`` set, which covers all five
  shapes. Logs are copied out before the worktree is removed.

What none of this can see is state carried across two invocations of the RTL,
for example an initialisation removed because one call per testbench never
needs it. Every accepted diff still gets read by a person.

Running it
~~~~~~~~~~

Environment (once): the ``allo`` conda env (py3.12) and ``chia_env`` (py3.10,
see ``requirements.txt``). The checkout's own ``mlir/build`` must be built
against ``$LLVM_BUILD_DIR``, so that ``allo/_mlir`` (a tracked relative
symlink) resolves inside the tree:

.. code-block:: bash

   conda activate allo
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build
   cmake -G Ninja -S mlir -B mlir/build -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
     -DPython3_EXECUTABLE=$(which python) -DPython_EXECUTABLE=$(which python) \
     -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=allo
   ninja -C mlir/build                        # ~35 s
   PYTHONPATH=$PWD python -c 'import allo; print(allo.__file__)'   # must be this tree
   npm ci --prefix examples/accelerator/tinytpu_vitis/chia_agent   # opencode, 725 MB, gitignored
   cp examples/accelerator/tinytpu_vitis/chia_agent/chia.env.example chia.env  # fill in; gitignored

``chia.env`` holds credentials and is gitignored; never commit it. Only the
``chia.env.example`` template is tracked. ``.rayignore`` holds
``node_modules/``. Without it Ray's ``working_dir`` package is over 1 GB against
a 512 MB limit, and every worker dies silently.

A search:

.. code-block:: bash

   conda activate chia_env
   set -a; source chia.env; set +a
   ray start --head --resources='{"opencode_creds": 2}' --include-dashboard=false
   cd examples/accelerator/tinytpu_vitis/chia_agent
   python smoke.py                                         # ~30 s, ~$0.05
   python swarm.py --workers 2 --iterations 3 --budget-usd 15
   python accept.py --diff ../../../../chia_runs/<run>/<worker>/best.diff \
                    --out  ../../../../chia_runs/<run>/accept-<worker>
   ray stop

The spend cap is global. It is read from opencode's own session DB
(``spend.py``), which covers every worker at once. Each loop refuses a model
call that would pass its soft cap: the budget minus one projected call per other
worker, projected as the largest call seen, with $3.50 as the floor.
``swarm.py`` kills every worker and this checkout's opencode processes if the
hard cap is reached anyway.

No git worktree is created per worker. Each worker's spec, logs and
``variants.jsonl`` live under the run directory from the start, so no
``git worktree remove --force`` can delete them.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - File
     - Role
   * - ``evaluate.py``
     - frozen two-tier evaluator; one JSON verdict per candidate
   * - ``stress.py``
     - frozen extra semantic gate
   * - ``spec_policy.py``
     - frozen: what an editable file may contain
   * - ``accept.py``
     - clean-checkout, five-shape acceptance of a claimed winner
   * - ``allo_tool.py``
     - the MCP surface: read spec / read frozen reference / replace_text,
       patch, insert / functional check / score
   * - ``llm.py``
     - CHIA's OpenCodeLLM with a 40-minute MCP request timeout
   * - ``loop.py``
     - one search: baseline, propose from best, harness re-scores, keep or
       rewind
   * - ``swarm.py``
     - K searches on different starting angles, global spend cap, report
   * - ``spend.py``
     - USD from opencode's DB since a timestamp
   * - ``smoke.py``
     - the cheapest end-to-end check

Capped smoke run, 2026-09-19: nothing improved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``chia_runs/isa-smoke-20260919-035443/`` (committed as ``0fc06999``): 2 workers
(``operand-path``, ``weight-prologue``), at most 3 iterations each, $15 hard
cap, model ``google-vertex/gemini-3.1-pro-preview``, harness @
``201f9342``..\ ``68dfbc53`` (same frozen files).

.. list-table::
   :widths: 25 75

   * - spend
     - **$15.15** (opencode DB, 234 model messages). The hard cap fired at
       $15.15 and killed both workers
   * - wall
     - 56.1 min
   * - baseline, in-loop
     - cosim **252 / 919** (4x4x4 / 16x16x16), gate + stress pass. Both
       workers reproduced it
   * - candidates completed
     - **0**. Neither worker finished iteration 1. No cycle number was produced
       for any proposal
   * - acceptance control
     - ``accept.py`` on the unmodified design, clean checkout: cosim
       **252 / 383 / 591 / 667 / 919**, all five TBs ``mismatches = 0``,
       ALL EXACT, stress 60/60, est. clock 2.431 ns

What the workers were doing when stopped. Both are unfinished edits, not
candidates. They are in ``<worker>/unscored_leftover.diff``, and neither earned a
cosim number:

- **operand-path** ($7.15 + $0.06): was folding ``vru`` into ``spm`` to remove
  the vector-register tier. ``spm``'s new ``mm`` branch reads ``spad`` but never
  feeds the array, and ``vru`` is stubbed out. Gated post-hoc with the fixed
  harness: **deadlock**, ``gate:bench_isa`` TIMEOUT at 240 s. It also left debug
  junk in the file, such as ``# DUMMY COMMENT FOR TRACEBACK``.
- **weight-prologue** ($6.14 + $1.80): was flattening the PE into one
  state-machine loop. That is not double-buffered weights, and it is the
  flat-PE variant already measured as buying nothing while ``vru`` pays the
  same prologue upstream (:doc:`/designs/tinytpu_history`). It left a duplicate
  ``pe`` and dummy functions. Gated post-hoc: ``gate:bench_isa``, Allo frontend
  error.

Neither leftover is claimed as a result. Most of both sessions went on edit
mechanics: 102 (operand-path) and 69 (weight-prologue, both sessions)
patch/insert calls, most rejected as non-applying or unparseable.

Harness defects the run exposed, all fixed in ``97308da0`` (none of them touch
the objective):

1. opencode timed MCP tool calls out at 60 s, and a cosim score takes 2-4 min,
   so the agent's ``score_cycles`` could never return. The harness's own
   scoring does not go through MCP and was unaffected. ``llm.py`` now sets a
   40-minute MCP timeout.
2. Sync evaluator tools blocked the MCP server's event loop. After a hung
   functional check, the next session saw **no tools at all**, tried
   ``bash``/``python`` (all refused, so containment held), and gave up. The
   tools are now async and run in a thread. Tested: ``read_spec`` answers in
   1 s while a gate runs.
3. A deadlocked candidate held the gate for up to 900 s. The gate timeout is now
   240 s, and the process group is killed.
4. CHIA's ``retries=3`` silently re-ran the 40-minute timed-out prompt, and a
   timed-out call reports $0 usage while still being billed. ``retries=1`` now,
   and the budget projects from the global DB delta.
5. Unified diffs were the agents' main failure mode, so ``replace_text`` (exact,
   unique) is now the preferred edit.

The fixed loop has **not** been exercised against the model: that needs more
than the $15 this run was capped at.


.. _chia-codesign:

The Original Co-Design Effort (``chia-codesign``)
-------------------------------------------------

"Agentic discovery of a domain-specific computing stack": an AI agent proposes
changes to a TPU-style accelerator's instruction set and microarchitecture at
once, and every proposal is scored by running a real commercial synthesis tool.
One specification generates the simulator, the hardware, and the compiler
backend, so a single agent edit moves all three together. The design searched
here is the fp32 ``examples/accelerator/tinytpu/`` on ``chia-codesign`` -- a
different machine from TinyTPU-isa, sharing no code with it (see
:doc:`/extensions/act`, "Two TinyTPUs, Not One").

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
~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
^^^^^^^^^^^^^^^^^^^^

- **S1-S3 need replication.** Every hypothesis has n=1. Idea 4 is the project's
  most interesting claim and currently its least evidenced.
- **The search is not seeded.** Re-running produces a different trajectory.
  Only the replay path (C6, C7) is deterministic.
- **No permanent archive.** For an *Artifacts Available* badge this needs a DOI
  (Zenodo), not just a git branch.
- **Agent runs need external services.** Vertex AI and a billing account; C1-C7
  need none of that, which is why the claim tiers are split the way they are.

Setup from zero
~~~~~~~~~~~~~~~

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
~~~~~~~

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
~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Summary: Demonstrated vs. Not
-----------------------------

**Demonstrated**

- On ``chia-codesign``: C1-C8 -- the deterministic claims, re-derivable by the
  listed commands, including replay of the two agent-found variants (4.07x and
  1.98x) to their recorded cycle counts with exact numerics.
- On ``chia-isa``: the harness reproduces the TinyTPU-isa baseline in-loop
  (cosim 252 / 919) and ``accept.py`` reproduces all five shapes from a clean
  checkout (252 / 383 / 591 / 667 / 919, bit-exact); containment held when a
  worker lost its tools and tried ``bash``/``python``.

**Not demonstrated**

- Any improvement to TinyTPU-isa: the only run produced **zero** completed
  candidates, and the harness fixes from it (``97308da0``) have not been
  exercised against the model.
- The *rate* at which a search finds a good design (S1), the ~100x yield spread
  across hypotheses (S2), and accept rates / trajectories (S3): all n=1 or
  unanalysed, and the search is not seeded.
