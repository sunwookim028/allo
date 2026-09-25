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

The time budget: evaluation IS the lever, but not the harness's copy
====================================================================

.. warning::

   **An earlier version of this page said evaluation was 7 % of the run. That
   was wrong**, and the error is instructive: 661 s counts only
   ``cosim_seconds`` from the harness's four verdicts. It misses everything the
   **agent runs itself inside its session**, which is most of it. Reconstructed
   turn by turn from opencode's own database (``part`` table, per-tool-call
   ``state.time.start/end``) across all eight of run 3's paid sessions -- free,
   no new run needed.

Inside a model session (8 sessions, 15,269 s of span):

.. list-table::
   :header-rows: 1

   * - what
     - calls
     - seconds
     - share
   * - ``score_cycles`` (gate + csynth + cosim)
     - 16
     - 5,634
     - **37 %**
   * - ``run_functional_check`` (gate)
     - 31
     - 2,288
     - **15 %**
   * - model latency (~200 turns, ~35 s/turn)
     - --
     - 7,340
     - **48 %**
   * - ``read_spec`` / ``replace_text`` / edits
     - 55+
     - **6**
     - 0.04 %

Whole of run 3 as worker-seconds (2 workers x 167 min ~= 20,040):

.. list-table::
   :header-rows: 1

   * - what
     - worker-s
     - share
   * - in-session ``score_cycles``
     - 5,634
     - 31 %
   * - in-session gate
     - 2,288
     - 13 %
   * - model latency
     - 7,340
     - 41 %
   * - harness re-scoring (8 verdicts + retries)
     - 2,657
     - 15 %
   * - MCP, edits, reads, composition
     - **6**
     - 0.03 %

**Total evaluation is 10,579 worker-s -- 59 % of the run.** MCP and loop
overhead is *six seconds*.

Why a session hits 2400 s
=========================

**Because 2400 s is roughly what the work costs**: ~1,250 s of tool execution
(4-5 cosims at 235 s, 5-7 gates at 60 s) plus ~1,150 s of model latency. The
two sessions that did complete finished at **2,313 s and 2,342 s** -- at the
wall, not comfortably inside it.

Three things are uncapped, and this is the actual defect:

* **``score_cycles`` has no cap.** The agent uses it as a hill-climbing oracle
  2-5 times a session, although the prompt itself says the harness re-scores
  independently. **31 % of the run re-measures what is measured again anyway.**
* **``run_functional_check`` is prompted "until it passes"**, no cap, 4-7 calls.
* **The agent is never told a time or turn budget.** Nothing mentions 2400 s.
  And its one cost hint is wrong: ``run_functional_check``'s docstring says
  **"~15 s"**; measured **52-76 s**, because it now rebuilds at MAXDIM 8, 12 and
  T=8/MAXDIM=32. An agent trying to budget would budget 4x low.

Context growth explains the model half. The smoke call is 12 s because it runs
**zero** evaluator tools and sits at ~3 k context. A real session reaches
~100 k -- ``read_spec`` returns **96 KB of both files whole** and is called
**12-19 times per session** -- and per-turn latency rises from ~4 s to 35-70 s.

Second order, and it costs the series more than the clock: on timeout
``response.result`` is empty, so ``agent_summary`` is ``''`` for **5 of run 3's
6 iterations**. The "Already attempted in this search" block handed to the next
iteration carries cycle counts but **no rationale**. *The search has no
memory.* The edits themselves survive -- 13-112 line diffs were scored -- so a
timeout is not a total loss.

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
