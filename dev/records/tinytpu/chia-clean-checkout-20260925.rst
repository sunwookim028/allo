..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

##############################################################
Five blockers between a clean checkout and a CHIA run, 2026-09-25
##############################################################

.. note::

   **Dated record.** Runs 1-3 spent ~$97 and produced graded verdicts, so CHIA
   *was* set up correctly at least three times. What did not exist is a
   **reproducible** setup: every piece of it is per-worktree and ephemeral, and
   nothing in the repository took a clean checkout to a runnable loop. Starting
   from a fresh worktree on 2026-09-25 hit five blockers before a single model
   call could be made. None cost money to find. One fails by hanging.

   A fix was in flight when this was written; if
   ``chia_agent/gcp_setup.sh`` (or its sibling) now does all of this, this page
   is the record of *why* it exists.

.. list-table::
   :header-rows: 1
   :widths: 5 30 65

   * - #
     - blocker
     - what it looked like
   * - 1
     - ``chia.env`` absent, and the only one on the host named project
       ``test-adrs``
     - **``preflight.py`` refused it**, naming the billing account mismatch.
       The guard worked. But nothing creates a correct one, and
       ``chia.env.example`` is already complete -- copying it was the whole fix.
   * - 2
     - ``opencode`` missing
     - A **per-worktree** install: ``npm ci --prefix
       examples/tinytpu/chia_agent``, six seconds, landing in
       ``node_modules/.bin`` where ``OPENCODE_BIN`` points. Nothing tells you
       to run it before you fail.
   * - 3
     - **a stale** ``/tmp/ray/ray_current_cluster`` **naming a dead head**
     - ``128.84.48.164:6399``, with **no** ``raylet`` and no ``gcs_server``
       running. ``ray.init(address="auto")`` reads that file and **blocks
       forever**: ten minutes of ``Failed to connect to GCS`` and nothing else.
       The only one of the five that fails silently, and the **third recorded
       occurrence** of the hazard in ``dev/roadmap.md``.
   * - 4
     - ``.chia_scratch/`` absent
     - ``smoke.py`` died in ``tempfile.mkdtemp``.
   * - 5
     - no ``mlir/build``
     - The frozen gate died at ``import`` on the ``WireConstructOp`` guard.
       ~8 minutes to build; ``examples/tinytpu/reproduce.sh`` does it.

What worked, for reference
==========================

.. code-block:: bash

   git worktree add /home/sk3463/allo-smoke origin/main     # NOT under /tmp
   cd /home/sk3463/allo-smoke
   cp examples/tinytpu/chia_agent/chia.env.example chia.env # already correct
   npm ci --prefix examples/tinytpu/chia_agent
   mkdir -p .chia_scratch
   conda activate allo && bash examples/tinytpu/reproduce.sh --no-cosim

   ray start --head --temp-dir=/tmp/rs --port=6411 \
       --resources='{"opencode_creds": 1}' --disable-usage-stats
   export RAY_ADDRESS=128.84.48.164:6411   # overrides the stale auto-discovery

   conda activate chia_env
   set -a; source chia.env; set +a
   python examples/tinytpu/chia_agent/preflight.py --budget-usd 10   # $0

**The worktree must not be under** ``/tmp``: ``evaluate.sandboxed`` gives each
gate ``--tmpfs /tmp``, so a scratchpad worktree is invisible inside the sandbox
and every candidate dies at ``import`` with ``ModuleNotFoundError:
allo.compose`` -- which reads exactly like a broken design. Copying a prebuilt
``mlir/build`` to ``/home`` does not help either, because of RPATH.

The Ray temp-dir path must be **short**: AF_UNIX caps the socket path at 107
bytes. And **never fix a stale cluster with** ``ray stop`` -- it matches by
process name across the whole host and would kill every other track's raylet.
