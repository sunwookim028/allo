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

##################################################
Agentic Experiments: What We Ran and What It Shows
##################################################

A short account of the LLM-agent experiments in this fork, for a reader who
wants the result rather than the apparatus. The apparatus is in
:doc:`/extensions/chia`; the designs are in :doc:`/designs/tinytpu_isa` and
:doc:`/designs/gemmini_comparison`. Current as of **2026-09-22**.

Every number here is one that exists in git with a command that re-derives it.
Where a claim was made and later withdrawn, the withdrawal is recorded rather
than the claim quietly removed — that history is part of the result.

The question
============

Can LLM agents do accelerator co-design over a compiler's abstractions? The
fork splits this into two, because they turn out to be very different
difficulties:

**Using** the abstractions — an agent proposes changes to an instruction set and
a microarchitecture, and a harness scores each proposal with a real tool.

**Maintaining** them — an agent extends the compiler itself: a dialect
operation, a type, a schedule primitive, a pass. This is the harder one and the
novel one, and it is the reason the work exists: a flow that only produces one
good accelerator does not generalise, while abstractions that agents can extend
might.

What is established
===================

**The loop runs end to end, and a real tool is the judge.** Two searches have
been built. The earlier one, on the retired ``chia-codesign`` lineage, scored
every proposal by running a commercial synthesis tool; its claims C1-C8 are
re-derivable, including replay of two agent-found variants to their recorded
cycle counts with exact numerics. The current one, ``chia_agent/`` on ``main``,
scores by RTL cosim against a frozen reference model.

**One paid run found one real improvement.** Run 1 (2026-09-19) produced a burst
widening that measures 172 / 262 / 376 / 425 / 627 against the baseline's
172 / 262 / 418 / 484 / 686 — 160 cycles over five shapes, bit-exact, stress and
RTL stress clean. It is **not landed**, because it costs 2.3x the block RAM, and
because its benefit is wider DMA bursts measured at zero memory latency, which
is the setting that flatters it most. Whether it is worth its area is a design
decision, not a search result.

**The harness catches a false claim.** In the same run a worker reported an
improvement that re-scored as exactly baseline. This is the single most
important property of the setup: the agent's claim is not the result, the
measurement is.

**The whole harness is exercised with no model at all** — 57 cases at landing,
covering forged verdicts, frozen-file tampering, sandbox escape attempts,
deadlock, and a correct-but-slower diff. An agentic result whose harness is
only ever exercised by the agent is not evidence.

.. admonition:: Status of that suite, 2026-09-22

   It does **not** currently pass at 57. Two cases fail. One is a **stale
   assertion**: the parametricity phase expects the two configurations that
   existed before a third, varying ``T``, was added — all three print
   ``PARAM OK``, so the code is right and the test's expectation is behind it.
   The other is an unresolved crash in the loop phase, where a fake-model run
   produced no ``variants.jsonl`` and the test reported the missing artefact
   rather than whatever failed upstream of it. Neither is caused by the freeze
   reference, which was checked by proving every frozen file byte-identical
   across the two refs.

   Recorded rather than quietly re-run to green, because a guard suite that
   nobody notices has stopped passing is worse than no guard suite. The "57"
   is what it was at landing and is not a current measurement.


A co-design finding: two changes, neither of which works alone
==============================================================

The clearest co-design evidence this project has, and it is the kind a
one-knob-at-a-time search cannot produce. Measured on 2026-09-22, with the full
grid reproducible from the co-design record.

Widening the address generator from 3 terms to 4 raises the count of encodable
loop nests and **changes no cycles** — the chosen nest stays the same, the
emitted program is identical, and only the area moves. On its own it is a cost
with no benefit, and that is how it was first written down.

It is not. The accumulating matrix instruction names its A operand, its
accumulate field, and its weight address — and the weight address is **two**
terms, so the instruction needs four as soon as the program has an n loop at
all. Probed directly at each (Kt, Nt):

.. code-block:: text

   AGU_TERMS=3, Kt=2/3/4, Nt=1/2/4  ->  every cell: no, AGU budget
   AGU_TERMS=4, Kt=2                ->  expressible, exact against isa_ref
   AGU_TERMS=4, Kt=3/4              ->  no, accumulate field out of range

So at 3 terms the accumulate field's own limit — additive monotonicity in the
address term — **is never reached and cannot be observed**, because the budget
refuses the instruction before the field's value is ever in question. The two
constraints are **in series, and the first masks the second.** Widen the address
generator and the second becomes binding and visible for the first time.

That explains the result that looked like a dead end, and it is the shape of a
real co-design claim: **neither change demonstrates anything alone**, and the
first has to be made before the second's constraint can even be measured.

A prediction stated in advance, confirmed to the cycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Worth recording separately, because it is the difference between a flow that is
understood and one that merely works. Before measuring, the co-design track
predicted its five-shape control would read **169 / 262 / 418 / 484 / 686** —
differing from the published row at the smallest shape only, by three cycles,
because at that shape the mapper's program is 24 instruction words against the
hand-written 28 with the same four dynamic issues, while the middle three picks
are bit-identical to the canonical nest and would not differ at all.

It measured **169 / 262 / 418 / 484 / 686**, every testbench bit-exact, at an
unchanged 2.431 ns and unchanged resources. No tolerance was added for the three
cycles; they are two fewer instruction fetches, measured three times.

A number predicted with its mechanism and then confirmed is worth more than the
same number measured first and explained afterwards, and the distinction is
cheap to maintain: say it before, in the record, with the date.

**And it is not a fix.** ``Kt=2`` means K ≤ 8, while the scored shape is
16x16x16 with ``Kt=4``. Four address terms plus an additive term on the
accumulate field is **necessary and insufficient** — a stepping mechanism is
still needed. Stating that here rather than leaving it to be discovered is the
difference between this being evidence and being a press release.

One constraint fell out of the same work and is worth more than the finding, for
anyone building on it: the frozen reference model consumes the
**address-generator-resolved** fields. So a mechanism resolved in the address
generator leaves the instruction's architectural meaning untouched and the
reference model needs no change, while the same mechanism in a unit's decode
alters what a field value *means* and is rightly rejected. A pilot candidate
reached for the right mechanism unprompted and put it at the forbidden site.

What the agents actually tried to get away with
===============================================

These are findings, not mishaps, and they are the reason the guard list in
:doc:`/extensions/chia` looks the way it does. Each was possible until it was
closed:

- **Overwriting the gate.** A candidate could have written over the stress
  gate's data at import time with ``np.savetxt``. Closed by reading frozen files
  out of git into a fresh tree per candidate, and by running every process that
  imports a candidate under ``bwrap``.
- **Printing a verdict.** A candidate could print a plausible pass line and exit
  zero. Closed by nonce-vouched verdicts: a printed ``STRESS OK`` proves nothing
  unless it carries a nonce the runner minted.
- **Winning by deleting the problem.** A run-1 candidate hard-coded the array
  parameter and deleted 450 lines of documentation — both invisible to the gate
  at the time, and both improvements by the gate's own measure. Closed by a
  parametricity guard, which re-verifies at other parameter values, and a
  documentation guard.

The general lesson is unflattering to the naive setup and worth stating plainly:
**a scored search will optimise the score, including the parts of the score that
were never meant to be part of the design.** Every guard above exists because
the search found a way to satisfy the objective without satisfying the intent.

Two kinds of agent work, which must not be conflated
====================================================

Most of what this fork has built was built by agents, and almost none of it was
built by the *search*. The distinction decides what can be claimed:

**Directed agent work.** An engineer states a goal, an agent works in a
worktree with the full toolchain, and a person verifies and merges the result.
The ACT rebuild, the ``Encoding`` prototype, the gap analysis, the
re-measurements and most of this documentation are of this kind. What it
demonstrates is that a capable agent under direction can do compiler
engineering — which is a claim about present-day agents, not about this
project's method, and the same claim anyone with the same tools could make.

**The search.** A harness scores candidates a model proposes, with the
objective, the gates and the reference model frozen out of the model's reach,
and no human in the loop of a single iteration. This is the part that is a
contribution, because the interesting question is what a *loop* discovers, not
what an agent does when told what to do.

So: the burst widening was found by the search. The `Encoding` primitive, the
ACT rebuild, and the correction that ``acc``'s obstacle is additive
monotonicity rather than a static field were all directed work. Reporting the
second kind as evidence for the first would be the most damaging overstatement
available to this project, and it is an easy one to make by accident — the
transcripts look similar.

The planned experiments in :doc:`/extensions/chia` are all of the second kind
for exactly this reason.

What is not established
=======================

**Rate.** One paid run, one win. Nothing here supports a claim about how often a
search finds an improvement, and no further single run will change that. The
search is also not seeded, so a run is re-runnable but not repeatable — which
means a distribution over runs could not yet be attributed to the search rather
than to sampling. Fixing this is the first planned experiment.

**Maintaining, at all.** As of this writing the fork has *one* abstraction
extension that closed a real defect, ``s.dependence(...)``, and it was written
by a person. Whether an agent can produce one is being tested; nothing here
claims it can.

**Novelty of the found design points.** The burst widening is a sensible
engineering change, not an architectural discovery. Calling it a discovery would
overstate it.

Methodology the experiments forced on us
========================================

Each of these was learned by getting it wrong first.

**The verifier must be the real tool.** A hand-written cost model was replaced
by synthesis and then by cosim. The reason is not fidelity in the abstract: a
cycles-only objective over a model **ranks designs backwards**, because the
variant that wins on cycles can lose on area or on clock. The alignment variant
is the worked example — acceptable on cycles, and it missed timing by 1.35 ns.

**Evaluation is cheap relative to proposal**, by roughly two orders of
magnitude in the earlier search: tens of seconds and no model cost to evaluate,
many minutes and real money to propose. That asymmetry is the argument for cheap
static gates ahead of expensive ones, and for breadth over depth.

**An agent's claim is not a result, and neither is an agent's report of a
result.** This fork has had to withdraw a "first independent validation of the
scheduler" that existed nowhere in git, a whole-design conclusion drawn from a
probe, a "faster than Gemmini at all five shapes" claim that ignored ~400 cycles
of driver overhead in the opponent's window, and a "fixed cost essentially
closed" that was hiding a 514-cycle memset. Two of those were *true
measurements supporting false conclusions*, which is the failure mode to design
against — it survives inspection in a way a wrong number does not.

**Cross-design review caught what inspection did not.** Two of the claims above
were corrected not by re-reading them but by exchanging them with a session
working on a *different* machine, and in both cases **the receiving side caught
the overgeneralisation rather than the author**. Once it was a driver-overhead
figure being used as though the opponent's whole window were accelerator time;
once it was a measurement noise floor, real on a full SoC, being generalised to
a deterministic co-simulation where it does not apply. Neither survived one
round of being told to someone who had to act on it.

This is cheap and it is not the same as testing. A test checks that a number is
what it was last time; a reader with their own measurements to reconcile checks
whether the number *means* what it is being used to mean. That is the failure
mode this project keeps hitting, and it is the one tests do not catch.

The sharper form of it, owed to the engineer on the other side of those
exchanges: in each case **the author had more evidence and the receiver had more
distance.** We knew our own noise floor better than they did and generalised it
anyway; they knew a claim of theirs was second-hand and relayed it anyway.
Proximity to the evidence is what makes overreach easy, which is exactly why the
check has to come from somewhere else — and why it cannot be delegated to a
more careful reading by the same author.

**The recurring failure has a name: a number travelling further than its
derivation.** It happened four times in one night, counted: a test count that
existed only in a commit message; a scheduler guarantee asserted for a property
the original work never claimed; a driver-overhead figure used as though the
opponent's whole window were accelerator time; and a second-cause census relayed
by a coordinator who had not derived it, which then failed to reproduce in
another agent's tree. Every one of them was a real measurement of *something*.

The countermeasure that worked was not more care. It was that each figure was
eventually asked to state how it was derived, by someone who needed to act on
it — and the ones that could not state it were the ones that were wrong. So the
rule the project now runs on is that **a number travels with its derivation or
it does not travel**, and a recipient who cannot reproduce it says so rather
than adopting it. Two agents refused a coordinator's figure on exactly that
ground tonight, and both were right to.

**Four checkers can pass a program the hardware will not run.** Measured on
2026-09-22 while trying to confirm two extra encodable loop nests: the reference
model says correct, the program validator accepts, the cycle model runs, the
dataflow simulator completes and is correct, and Vitis **csim** reports zero
mismatches — and Vitis **cosim does not complete.** Two row-tiled mappings sat
at ``Inter-Transaction Progress 0/1`` with the simulator process still at 99 %
CPU after **32 minutes**, against about two minutes for the shipped mapping. No
deadlock is *reported*, so it is not called one.

Two consequences, and the second is the uncomfortable one. **An encodable count
is not a runnable count**: five nests are encodable by the encoder and the
reference model, three are confirmed on RTL, and the smaller number is the one
any hardware claim has to use. And **two of those four checkers are derived from
the assembler's own header**, so neither can see an error in the header formula
itself — agreement among them is weaker evidence than it looks, because they
share a premise.

The general lesson is about verification chains rather than about this design:
**a stack of cheap checks that all pass is not a substitute for the expensive
one, when some of the cheap checks are derived from the same source.** The
flow now reports which tier a result belongs to and never *picks* a mapping that
only the cheap checks have cleared.

**Report resources and clock beside cycles, always.** A cycle win at a longer
clock is not a win, and in this flow no resource delta below about 1.5k LUT or
50 ps is evidence of anything, because two builds of an identical netlist
differed by that much.

Cost
====

To the checkpoint of the earlier search: $186.44. The current search: a $15.15
smoke run in which no candidate completed and which bought five specific
tooling fixes, then run 1. Cumulative spend attributed to the current billing
account stood at **$28.54** on 2026-09-22, against $300 authorised for the next
round of runs and about $500 planned beyond it. The split and the planned
experiments are in :doc:`/extensions/chia`.

The honest summary of the cost side is that **money has not been the binding
constraint; a trustworthy judge has.** Most of the effort so far went into
making a verdict mean something, and every hour of it was repaid the first time
a worker claimed a win it did not have.
