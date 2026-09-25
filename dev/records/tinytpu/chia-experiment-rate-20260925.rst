..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

#########################################################
Why the CHIA experiment rate is slow, measured 2026-09-25
#########################################################

.. note::

   **Dated measurement record.** The question was "our experiment rate is
   orders of magnitude slower than it should be -- why". This records what was
   measured that day, including the hypotheses that were *eliminated*, so the
   next person does not re-derive them. Work in flight at the time is named at
   the end.

The time budget, and where it is not
====================================

Run 3 took **167 minutes** of wall time. Of that:

.. list-table::
   :header-rows: 1

   * - what
     - time
     - share
   * - all cosim evaluation, 4 candidates
     - **661 s (11 min)**
     - **7 %**
   * - four large model calls, each the full ``timeout_seconds=2400``
     - 160 min
     - ~96 %

Run 2: **all six** model calls hit the timeout. Run 1: three of five sessions
timed out. So a timeout that fires on essentially every session across three
runs is not a timeout -- **it is the schedule**.

The practical consequence, which is counter-intuitive and worth stating first:
**evaluation speedups are not the lever.** Relaxing PD feedback, skipping the
area proxy or making cosim faster all target the 7 %.

Eliminated: the model endpoint
==============================

Measured directly with ``chia_agent/smoke.py`` on 2026-09-25:

* **12 seconds, 2 turns, $0.02, one session.** No retry, no 429, no queueing.
* Whole smoke end to end 126 s, of which **101 s was the frozen gate running
  locally** and 12 s was Vertex.

So the endpoint answers a scoped question in twelve seconds. Whatever consumes
2400 s is in the agentic loop, not the service.

Eliminated: quota, and we are not on a cheap model
==================================================

``gemini-3.1-pro-preview`` is the **pro** tier; flash is the cheap tier and we
do not use it. In ``chia2026-tinytpu``:

* pro-tier buckets are at **250 rpm**, not the 1 or 5 rpm that some models in
  this project carry (``gemini-1.5-flash-8b`` is 1 rpm);
* there is **no quota bucket for the exact string** ``gemini-3.1-pro-preview``
  -- the nearest is ``gemini-3.1-pro-preview-cider-qcd`` at 250 rpm -- and the
  default bucket is literally ``{}``, no ``effectiveLimit``, no
  ``defaultLimit``. That is ambiguous between "unlimited" and "not enumerated",
  and the smoke call settles it in practice: no throttling was observed.

``preflight.py`` does **not** check quota. It verifies billing and that
``aiplatform`` is enabled, and stops.

The one genuine endpoint concern is **preview, not price**: a preview model has
no deprecation contract and can change under a series, which would silently
invalidate cross-run comparisons. Choosing preview over GA has never been
written down as a decision.

Our own instrumentation is the blind spot
=========================================

**A timed-out call returns no session id and reports ``$0`` on ``usage``.** So
the sessions that most need explaining report nothing at all -- which is why
run 3 reads $0.00 on that field while opencode's database charges $20.72, and
why this took a day to frame rather than an hour. Money is read from the
database for exactly this reason; turn counts are not.

Run 3 recorded **281 model messages across 10 sessions**, so sessions are long
and multi-turn -- but *what those turns did* is unrecorded.

``timeout_seconds=2400``, ``retries=1``, ``RATE_LIMIT_RETRIES=6`` and
``RATE_LIMIT_BACKOFF=45.0`` are all in ``chia_agent/loop.py``. **None has ever
been tuned.** For scale: if all six rate-limit retries fired, the backoff ladder
alone is ~95 minutes, *outside* the per-call timeout -- but that path only runs
on an explicit ``RateLimitError``, and none was observed.

The live hypothesis
===================

Unsettled as of 2026-09-25. The tool the agent calls is
``chia_agent/allo_tool.py``, and **a single tool call may run a gate**: the
smoke's gate took 101 s, with ``gen_isa --conform`` alone at 34.5 s. If so, a
handful of tool calls accounts for the whole 2400 s and model latency is barely
involved. ``fake_model.py`` with ``TINYTPU_OPENCODE_BASE_URL`` drives the entire
loop with **no Vertex call**, so this is testable at $0.

The other candidate is that the task has **no stopping condition the agent can
satisfy**, so it works until killed.
