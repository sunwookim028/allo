..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

##############################################################
Designing the next CHIA run series, 2026-09-25
##############################################################

:date: 2026-09-25
:what: Whether fast experiments are now feasible (with arithmetic), whether
       the next runs should be breadth-first or depth-first, and how to stop
       the series rediscovering the operand burst.
:why:  Three of the series' four wins are the same idea. Run 4's result
       section names the cause -- *the burst knob is the cheapest legal win in
       the tree, so a hill-climber with an expensive oracle walks straight to
       it every time* -- and calls it an economics problem rather than a
       capability one. This page is the design that answers it, and the three
       pre-registrations it produced.
:status: **DRAFT.** Nothing here has been run, nothing here is signed off, and
         no paid call has been made against it.

.. note::

   **This page depends on work in flight.** A tiered oracle -- a fast
   functional tier, a slower gate tier, and a capped ``score_cycles`` -- is
   being built by another agent as this is written. Everything below that
   needs it is marked. This page deliberately does **not** predict what that
   work will measure; where the tier's cost appears it appears as a
   parameter, and the arithmetic is given at more than one value of it.

.. contents:: On this page
   :local:
   :depth: 2

What the series has produced, and the problem that is left
==========================================================

Five runs, one row each in
``dev/records/tinytpu/chia-evidence/README.md``:

.. list-table::
   :header-rows: 1
   :widths: 10 16 54 20

   * - run
     - cost
     - result
     - novel design point?
   * - smoke
     - $15.15
     - null, 0 candidates, five harness defects
     - n/a
   * - 1
     - $28.54
     - one win of four: operand bursts, +27 / -291 lines
     - not landable as written
   * - 2
     - $47.65
     - null, the predicted branch; 17 nests encodable, **10 wrong**
     - no
   * - 3
     - $20.72
     - one win: ``TPU_DMA_WIDEN``, **named in the prereg as a rediscovery**
     - no, by prediction
   * - 4
     - $2.76
     - graded verdict; win was ``TPU_DMA_WIDEN`` **again**, guard deleted
     - no, by the definition recorded before the run

The definition the owner recorded *before* run 4 -- *a candidate-authored
change to a unit, the wiring or the instruction set, that beats the run's own
measured control, and is not a knob already in the tree* -- has now been used
once and it excluded the win on its first use. That is the definition working,
not the loop failing.

The diagnosis this page designs against is run 4's, in its own words:

    It is that the burst knob is the cheapest legal win in the tree, so a
    hill-climber with an expensive oracle goes straight to it every time.

Two consequences follow, and they set the whole design:

1. **The search is a hill-climber because the oracle is expensive.** One
   worker, iterative, best-so-far, ~2 cycle scores a session. A search that
   can afford two probes per session cannot do anything but follow the
   steepest local gradient.
2. **The steepest local gradient is the burst.** So the fix is either to
   remove the gradient or to afford more than two probes. This page does both,
   in that order.

Are fast experiments now feasible?
==================================

Yes for wall clock, and the gain is large. **No for dollars** -- and that
distinction is the most useful thing on this page, because it changes what
the series should buy.

Where a session's 2,400 seconds go today
----------------------------------------

Reconstructed turn by turn from opencode's own database across run 3's eight
paid sessions (``dev/records/tinytpu/chia-experiment-rate-20260925.rst``; no
new run, $0):

.. list-table:: one session, ~2,400 s
   :header-rows: 1
   :widths: 40 14 14 32

   * - what
     - per session
     - unit cost
     - note
   * - ``score_cycles``
     - 2 calls, 704 s
     - **352 s**
     - 16 calls / 5,634 s over 8 sessions
   * - ``run_functional_check``
     - ~4 calls, 286 s
     - **74 s**
     - 31 calls / 2,288 s; its docstring says "~15 s"
   * - model latency
     - ~25 turns, ~1,150 s
     - **~35 s/turn**
     - ~4 s/turn at 3 k context, 35-70 s at ~100 k
   * - ``read_spec`` / edits / MCP
     - 55+ calls
     - **6 s total**
     - 0.03 % of the run

So tool execution is ~1,250 s and model latency ~1,150 s, and the two sessions
that finished finished at 2,313 s and 2,342 s -- at the wall, not inside it.

The unit costs the tiering changes
-----------------------------------

Three numbers are measured and are not predictions:

* **A whole candidate end to end is 623.9 s** at the scored ``T=4
  MAXDIM=16``, and **1,050.5 s** at ``T=8 MAXDIM=32``
  (``codesign-space-20260925.rst``).
* **The PyTorch oracle inside it is 2.9 s** (3.2 s at T=8). It is the
  external-oracle correctness check: ``torch.nn.Linear``'s own forward,
  bit-exact, with nothing a candidate writes on either side.
* **``mapspace_report`` is ~30 s and runs no Vitis** -- the gate plus the
  exhaustive nest enumeration.

The fast tier being built is a *legality* tier in that 3 s class. Call its
cost ``c`` and do the arithmetic at more than one value rather than guess it.

The arithmetic
--------------

**Assumptions, marked as assumptions.**

* **A1.** A *variant* -- one edit plus one legality verdict -- costs **2.5
  model turns**. Run 4 spent 56 messages on one candidate and run 3 spent 281
  on six, so 2.5 is an assumption about a *breadth* phase that has never been
  run, not an extrapolation from a depth phase.
* **A2.** Per-turn latency ``L`` is 35 s at run-3 context and ~10-15 s at
  small context. That is a measured *relation* (12-19 ``read_spec`` calls
  returning 96 KB whole is what grows the context) but the breadth phase's
  own context profile is unmeasured.
* **A3.** Dollars track **model messages**, not tool seconds: $0.049/message
  (run 4, 56 messages, $2.76) to $0.074/message (run 3, 281 messages,
  $20.72). Use **$0.06**.
* **A4.** The fast tier's cost ``c`` is a free parameter below. This page
  does not predict it.

.. list-table:: one legality-checked variant = ``2.5L + c`` seconds
   :header-rows: 1
   :widths: 22 14 14 16 16 18

   * - regime
     - ``L``
     - ``c``
     - s/variant
     - variants / 2,400 s session
     - variants / hour / worker

   * - fast tier, run-3 context
     - 35 s
     - 3 s
     - **90 s**
     - **26**
     - **40**
   * - gate tier, run-3 context
     - 35 s
     - 20 s
     - 108 s
     - 22
     - 33
   * - fast tier, small context
     - 15 s
     - 3 s
     - 41 s
     - 59
     - 88
   * - *today, a cycle-scored candidate*
     - 35 s
     - 352 s
     - **562 s**
     - **4**
     - **6**

And in dollars, at $0.06/message:

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - unit
     - $ each
     - per $10
   * - legality-checked variant (2.5 messages)
     - **~$0.15**
     - **~65**
   * - graded candidate, run 3 measured (281 msg / 6)
     - **$3.45**
     - 2.9
   * - graded candidate, run 4 measured (56 msg / 1)
     - **$2.76**
     - 3.6

Reading it, including the part that is not a win
------------------------------------------------

**The fast tier buys wall clock, and it buys experiments per dollar only by
changing the unit.** A cycle score costs $3.45 not because a cosim is
expensive in tokens -- it is free in tokens -- but because run 3 spent 47
model messages per graded candidate. Collapsing 352 s to 3 s does not make a
session cheaper; it makes a session *do more*, at roughly the same
dollars-per-hour. Concretely:

* **Per hour, per worker: ~6 cycle-scored candidates today, ~40 legality-
  checked variants after** -- a **6-7x** rise in experiments per hour, and
  nearer 15x if context stays small.
* **Per dollar: ~$0.15 a variant against ~$3 a graded candidate** -- **~20x**,
  but only because a variant answers a smaller question.
* **Per session the binding constraint moves.** Today a session is
  tool-bound (1,250 s of 2,400 s). After, it is **latency-bound**: 2,400 s at
  35 s/turn is ~68 turns whatever the tools cost. So the ceiling on breadth is
  ``timeout_seconds=2400`` divided by per-turn latency, and the lever on it is
  **context growth**, not the oracle. ``read_spec`` returning both files whole,
  12-19 times a session, is the known driver and is the next thing worth
  measuring after the tier lands.
* **Cost per session rises.** A latency-bound session runs ~68 turns against
  run 3's ~25, so ~$4 a session instead of ~$1.50-2.50. Budget a breadth run
  at **$4-5 per worker-iteration**, not at run 4's $2.76.

**Feasible, then**, with one honest qualification: none of this is measured.
The three ingredients -- 2.5 turns per variant, the tier's ``c``, and whether
context stays small across 60 turns -- are all assumptions, and the first
pre-registration below exists to measure them rather than to search.

Breadth-first or depth-first?
=============================

Our loop is depth-first today by construction, not by choice: one worker,
iterative, best-so-far, with an oracle that permits two probes a session. The
owner has raised BFS and DFS as heuristics, so the question is which suits
**an oracle that is fast on legality and slow on cycles.**

The argument for breadth, from our own evidence
-----------------------------------------------

BFS is affordable exactly when the *filter* is cheap and the *scorer* is
expensive, which is now this loop's shape: 3 s to learn whether a machine
computes the right answer, 352 s to learn how fast it is. Generating twenty
variants and scoring two survivors is 90 s x 20 + 352 s x 2 = ~2,500 s, which
is one session. Generating twenty and *scoring* twenty is 3.1 hours and is
never affordable.

And breadth is where the burst problem is. A hill-climber's next step is its
whole hypothesis; a breadth phase's next step is a *distribution*. Three of
four wins being one idea is a statement about a search that only ever looked
one step ahead.

The argument against breadth, also from our own evidence
---------------------------------------------------------

Run 2 is the warning and it must be quoted exactly. One arm widened the
encodable mapspace from 3 nests to **17**, and **10 of the 17 computed the
wrong answer** against ``isa_ref`` -- *including the nest the frozen mapper
itself picked*. Only the frozen reference sweep caught them.

So: **breadth without a cheap correctness filter does not generate
candidates, it generates wrong machines faster.** Run 2 is what BFS looks like
when the filter costs as much as the score, because then nobody runs the
filter on the whole frontier.

What has changed is precisely that objection. The filter now exists at ~3 s,
and -- this is the part that matters with the ISA editable -- it is an
**external** oracle. ``isa_ref.py`` is editable now; a candidate that rewrites
its own reference model can satisfy any internal check. ``torch.nn.Linear``'s
forward is the one check on the frontier that a candidate cannot move. Breadth
over an editable ISA is defensible *only* because that specific gate exists,
and any breadth phase that skips it is run 2 again.

Recommendation: breadth-first inside an iteration, depth on the survivor
-------------------------------------------------------------------------

Not a choice between them -- a **two-phase iteration**:

**Phase A (breadth, legality only).** N >= 12 distinct variants, each getting
the fast tier and nothing else. ``score_cycles`` is unavailable in this phase.
Each variant is recorded by the per-turn session trace with what it edited,
its legality verdict and its seconds -- so the phase has a deliverable whether
or not anything survives.

**Phase B (depth, cycles).** The worker picks k <= 2 survivors and spends the
capped ``score_cycles``. Depth is where a number comes from; it is just not
where the hypothesis should come from.

Four reasons, each tied to something measured:

1. **The economics fit.** 3 s filter, 352 s scorer. That ratio *is* the case
   for BFS, and it did not exist before this week.
2. **The safety objection is retired by the same change.** Run 2's 10-of-17
   is an argument for a cheap filter, and against breadth only while none
   existed.
3. **A breadth null is informative and a depth null is not.** "24 variants,
   19 illegal, 5 legal, none beat control, here is what each touched" is a map
   of the neighbourhood. Run 2's null was one sentence and cost $47.65.
4. **Depth is the mechanism that produced the rediscoveries.** A best-so-far
   walk with two probes per session cannot express a hypothesis that does not
   pay off in one step, and the burst pays off in one step.

**Where DFS still wins, and must be kept**: the ``T`` / ``MAXDIM`` axis. A
T=8 candidate costs 1,050 s, two of the five GEMM shapes and one of the two
models drop out of its corpus, and the comparison against the published row
becomes ``NOT-COMPARABLE``. Breadth there is unaffordable *and*
uninterpretable. One committed line, three iterations, stating the trade --
674 -> 426 cycles for 1.20 -> 2.23 mm^2 is the measured interior point -- is
the right shape. So the series should run one of each and say which produced
more, which is why run 5 and run 7 below are deliberately a matched pair.

**One constraint breadth needs that depth does not.** Left alone, a breadth
phase will propose twenty variations of the same idea. N distinct variants has
to mean distinct by *what they edit*: the pre-registrations below require a
minimum number of distinct ``EDITABLE`` paths across the phase, counted from
the trace, and that count is reported whether or not it is met.

How to stop rediscovering the burst
===================================

Three options were on the table. Each is stated with its failure mode,
because each has one.

Option 1: a rediscovery register handed to the prompt
------------------------------------------------------

List the known wins in the agent's task statement -- "``TPU_DMA_WIDEN``, the
``ac2sp`` depth and ``QD`` are known; a candidate matching one is not a
finding."

**Failure mode: it teaches the answer.** A register in the prompt hands a
model that is rewarded for wins a *working win*, spelled by file and symbol.
Run 3 named ``TPU_DMA_WIDEN`` in its pre-registration -- where the agent could
not see it -- and the agent found it anyway; naming it where the agent *can*
see it makes it cheaper to find, not dearer. Negative instructions are also
the weakest thing a prompt does. This option risks converting an accidental
rediscovery rate of 3-in-4 into a deliberate one.

Option 2: forbid or absorb the named knob
------------------------------------------

Either freeze ``_WIDEN`` out of ``EDITABLE``, or **land** run 4's three-line
diff so the widest burst is the default and is therefore *in the control*.

**Failure mode: whack-a-mole, plus a refit bill.** Absorbing the burst moves
the published row, so every cross-run comparison in the series needs
re-deriving -- the cost ``control.NEEDS_REFIT`` exists to track, paid once
already in ``refit-20260925.rst``. And it removes one knob: the register's
items 2 and 3, the ``ac2sp`` depth and ``QD``, are the next-cheapest legal
wins and become the new attractor. Freezing instead of landing is worse: it
leaves known headroom in the tree that no run may claim, which makes every
subsequent null ambiguous.

Option 3: ask a question the burst cannot answer
-------------------------------------------------

Change the objective so the burst is not on the gradient. **We have the
measurement that makes this concrete and it is not a guess**
(``docs/source/extensions/chia_results.rst``): the burst widening is worth
**zero cycles on the model suite at the scored configuration** -- 861 to 861,
1,530 to 1,530 -- against -287 and -583 at ``MAXDIM=64``. At ``MAXDIM=16`` a
DRAM row is four packed words rather than sixteen and the prologue hides the
burst for the models. The relationship *inverts* at the scored point: the GEMM
shapes see the widening and the models do not.

So an objective weighted on the **model term** has a measured zero where the
current GEMM-summed objective has the steepest gradient in the tree. That is
not a prohibition the agent can route around; it is an absence of reward.

**Failure mode: a new objective has no control history, and a proxy can be
gamed.** The first run under a re-weighted objective spends part of its budget
establishing a baseline instead of searching, and its result is not comparable
with runs 1-4. And the wrong proxy is actively dangerous here: run 2's arm
optimised *nests encodable*, got 3 -> 17, and 10 were wrong. Any objective
that counts rather than measures repeats that. The model term is safe on this
point only because it is a cosim cycle count under the same bit-exact
testbench, not a count of possibilities.

Recommended: 3, enabled by 2, with 1 demoted to a grading rule
---------------------------------------------------------------

1. **Land the burst** (run 4's diff is three lines, accepted twice, never
   landed) so it is absorbed into the control rather than forbidden, and pay
   the refit. This removes the gradient by *taking* the win, which is also the
   honest thing to do with a result the repository already holds.
2. **Score the model term**, where the same change is measured at exactly
   zero, so the next-cheapest knobs face an objective they were not tuned
   against either.
3. **Keep the register in the pre-registration, never in the prompt.** Its job
   is to *grade* the result after the fact -- which it did, correctly, on its
   first use in run 4 -- not to steer the search.

Both (1) and (2) are changes a person makes to the design and the evaluator,
outside this document and outside the tiering work. Each pre-registration
below names which of them it needs and what it falls back to if they have not
landed.

The series shape
================

Three runs, in order, each gating the next:

.. list-table::
   :header-rows: 1
   :widths: 8 22 34 12 12 12

   * - run
     - shape
     - question
     - workers x iters
     - cap
     - wall
   * - 5
     - **BFS**, shakedown
     - Does the fast tier actually buy breadth, and at what rate?
     - 1 x 2
     - $12
     - ~95 min
   * - 6
     - **BFS**, the editable ISA
     - Can a candidate author a coherent instruction-set change?
     - 2 x 2
     - $25
     - ~95 min
   * - 7
     - **DFS**, the co-design axis
     - With ``T`` / ``MAXDIM`` searchable, does depth state a trade?
     - 1 x 3
     - $18
     - ~135 min

Total ceiling **$55** against $372.98 remaining under ``CHIA_TOTAL_CAP_USD``
after run 4 ($127.02 cumulative).

**Why this order.** Run 5 measures the three assumptions the arithmetic above
rests on, cheaply, before either of the expensive runs is committed to a
shape. If breadth turns out not to exist -- if a worker cannot reach a dozen
variants in a session -- run 6 is re-planned as a depth run and run 7 is
unchanged. Runs 5 and 7 are also the matched BFS/DFS pair: same tree, same
control, opposite heuristics, and the comparison between them is the answer to
the owner's question that no amount of arguing on this page can supply.

**Run 6 is the one that has never been tried.** Seventeen editable paths
include ``isa_spec.json``, ``isa_encoding.py`` and ``isa_ref.py``, and no run
in the series has touched any of them. It is both the surface most likely to
produce something that is not a knob flip and the surface whose failure modes
are entirely unmapped -- which is why its null is worth as much as its win.

What this design depends on
===========================

Stated so that a reader can tell which parts of it die if something does not
land.

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - dependency
     - owner
     - if it does not land

   * - **the tiered oracle** -- a fast functional tier callable freely, a gate
       tier, ``score_cycles`` capped
     - the concurrent tiering work
     - **Runs 5 and 6 do not run as written.** Breadth at 74 s a check is 32
       variants an hour at best and the ``score_cycles`` cap is the thing that
       makes Phase A enforceable rather than advisory. Run 7 is unaffected.
   * - **a per-iteration ``score_cycles`` cap the harness enforces**
     - the tiering work
     - Phase A becomes a request rather than a rule, and a breadth run can
       silently turn into a depth run. Fall back to iterations of 1 with the
       budget as the only cap, and say so.
   * - **the per-turn session trace**
     - landed (``session_trace.py``)
     - the breadth phases have no deliverable on a null, which is most of
       their value.
   * - **landing the burst into the control** (recommendation 2 step 1)
     - a person; three-line diff plus a refit
     - runs 5-7 keep the register as a *grading* rule, the burst stays the
       cheapest win, and a burst rediscovery is reported as a rediscovery
       again. The runs still answer their questions; they just answer them in
       a tree with a known attractor.
   * - **the model term in the scored objective** (recommendation 2 step 2)
     - a person; ``evaluate.SCORED``
     - the runs are scored on the GEMM sum as today and the model term is
       reported beside it for grading only. Named in each pre-registration.
   * - **``T`` / ``MAXDIM`` searchable within the fitted envelope**
     - landed
     - run 7 has no question.

Written 2026-09-25 alongside ``prereg-run5-20260925.md``,
``prereg-run6-20260925.md`` and ``prereg-run7-20260925.md``, all three
**DRAFT, NOT SIGNED OFF**.
