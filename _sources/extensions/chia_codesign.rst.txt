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

#####################################################
The Original Co-Design Effort (``chia-codesign``)
#####################################################


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
