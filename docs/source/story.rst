..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

###############################################################
Notes from building a small matrix accelerator
###############################################################

We built a small chip design for multiplying matrices, wrote it in a Python
dialect rather than in a hardware language, and measured it two ways: on a
reconfigurable chip (an FPGA) and through the tool flow that would turn it into
a real chip. Then we pointed a language model at the compiler behind it and
asked whether it could invent anything.

This page is what we learned. It is deliberately short, and every number on it
came out of a run that can be repeated.

The design
==========

It is a 4-by-4 grid of multipliers with a small instruction set — load some
numbers, multiply, add, write the answer back. Programs are written for it the
way programs are written for any processor, and the hardware is generated from
a description that is about a hundred lines long, assembled from eight reusable
parts.

It works. The same five test problems take **171, 261, 417, 483 and 685 clock
ticks**, every answer exact, reproduced independently three times.

For comparison we used Gemmini, a well-known open accelerator from Berkeley,
configured to the same grid size and measured over the same window.

Our design is slower, and the gap closes as problems get bigger — from 27%
behind on a small problem to **9% behind on the largest**, where we are running
at 74% of the theoretical maximum. On one problem shape we are 18% faster.
The interesting part is *why*: the loss is almost entirely a fixed setup cost
paid once per call, and the team behind a different accelerator, measuring
independently, found the same shape of loss in their own design. The conclusion
we jointly reached is that Gemmini's advantage here is its **software**, not its
hardware.

Three things that surprised us
==============================

1. The FPGA lies about what things cost
---------------------------------------

An FPGA is a chip you can reprogram, so it is the usual place to measure a
design. It is also, we discovered, systematically misleading about area.

We had an optimisation that made memory transfers wider and bought back a large
part of our deficit. On the FPGA it looked cheap. Run through the flow that
makes a real chip, it costs **74% more silicon**. And 99% of that cost is not
the arithmetic or the memories at all — it is **two adapters that talk to
external memory**, which grow thirteen-fold when widened.

Worse, we then found that the adapter which fetches *instructions* is **60% of
our entire chip**, while the arithmetic grid is a rounding error beside it. It
does not grow when we make the design bigger; it is just always there. None of
this is visible on an FPGA, where that kind of buffering is nearly free.

This is not only our problem. Gemmini's own published measurements say memory
is **67% of their chip** and the multiplier grid is **11%**. Everyone's
intuition about where the silicon goes appears to be wrong in the same
direction.

2. Measuring on real programs changes the answer by four to eight times
-----------------------------------------------------------------------

We had been evaluating on matrix-multiply problems of various sizes, which is
what everyone does. When we built a small suite of actual neural networks and
measured the same optimisation on those, it was worth **25–34% of the running
time** instead of the 4–7% the matrix-multiply table showed.

The reason is simple in hindsight. A real model does not make the problem
*bigger*; it makes it *longer* — many layers, each paying the setup cost again.
A benchmark of one large problem hides a cost that a benchmark of many small
ones exposes.

So an optimisation that looked like a bad trade at +74% silicon for 5% speed
may well be a good one at +74% for 30%. We have not finished that argument, but
we would have reached the wrong answer confidently without the second
measurement.

3. A model can invent a real compiler rule — and get it subtly wrong
---------------------------------------------------------------------

We gave a language model access to the compiler and a set of measurements, and
asked it to find something the language could not express.

It proposed a rule that refuses a design when a memory is written more times
per cycle than it has ports. That is a genuine hardware constraint: on an FPGA
a second write port is free, and in a real chip it does not exist at all — our
synthesis tool rejects such a design outright. The rule correctly refused the
bad design and correctly accepted the good one, and it stated its own
assumptions without being asked.

**It is also wrong in both directions**, which we found by trying it on cases
its author had not considered. It accepts a design where two writes collide
because it counts memory banks without checking *which* bank each write lands
in, and it refuses a design that is perfectly legal because the writes happen
on different cycles and it does not look at time.

Two independent attempts, one of which was not told what to look for, produced
the *same* flawed shape. That convergence is the useful result:

   Counting things against a capacity is the wrong shape for this kind of rule.
   It throws away *which* item is touched *when*, and both mistakes are that
   lost information coming back.

The right shape is a calendar — what happens in which cycle — and the compiler
already knows enough to build one. An engineer on another team, auditing their
own scheduler against our failure, reached the same conclusion from the
opposite direction, and their version is safe precisely because it tracks
cycles instead of counts.

Instructions as small composable pieces
========================================

That led to a question worth its own experiment: can an instruction be
*defined* as a sequence of small per-unit steps — "the register file does this,
the arithmetic unit does that" — rather than hand-written into every piece of
software that needs to know about it?

We built it and added a new instruction both ways to compare.

**It did not save code**: 79 lines the new way against 77 the old way. What it
did change is how many separate places have to agree — from five parts of a
specification plus two separate programs, down to one part plus a single stated
fact about the hardware. Two of the places we replaced turned out to have been
*wrong* already, and a third quietly deadlocked the first time it met an
instruction it had not been told about.

The sharpest result is a test we could not have run before. Two steps, each
perfectly legal on its own, combined into one instruction that is not legal —
two units writing the same port at the same moment. The model refuses it, and
the repair is to move one of them to a different cycle. A rule that merely
counts cannot catch this, because the count is one write per unit and one port
per unit is exactly what the hardware has.

We also measured what it *cannot* catch. Of fifteen deliberately broken
descriptions, five were refused immediately and four more were caught by a cost
check — but **six were invisible to every check short of running the real
hardware**: a description naming the wrong input, or subtraction where addition
was meant. The approach verifies that pieces *fit together*; it does not verify
that they compute the right thing.

Our own conclusion is therefore a qualified one: **worth using where several
programs would otherwise have to repeat the same fact, not yet worth making the
default**, because only one machine has tried it.

What we got wrong
=================

Every item here was published internally and then withdrawn on measurement. We
keep the list because the corrections were more instructive than the claims.

- We said a memory-latency knob told us something about real memory systems. It
  does not — the results are not even ordered consistently, so it is a hint to
  the compiler, not a model of anything.
- We said an optimisation "was not worth it" on the strength of an area number,
  before finding that 99% of the area was in a place that suggested a much
  better question.
- We said our own measurements had noise. They do not; the noise belonged to
  the design we were comparing against.
- We twice claimed a language model had implemented something from a
  specification, when the specification had in fact told it the answer.
- Most often, we checked the wrong thing and it passed: a documentation build
  that skipped the files it was meant to inspect, a hardware simulation run on
  a different configuration than the one being claimed, and a repository check
  that read commit titles rather than their contents. All three came back
  clean.

That last pattern is the most durable lesson here. Over one night, four
independent instruments were each caught reporting success without having
actually run — a server that answered without ever starting, a cost meter
reporting zero for a real charge, a consistent-looking report describing a
truncated file, and an error detector that crashed and was read as "all clear".

The rule we now work by is:

   A clean result from an instrument is only evidence if the instrument can be
   shown to have run — and to have run on the thing you are claiming.

Where it stands
===============

Solid: the design works and is reproducible; the comparison on speed is
measured on both sides over the same window; the silicon-area findings are
measured and attributed to specific components.

Not yet: we have no area number for the comparison design — those runs are
queued. We have no power figures at all, and we say so rather than publishing
an unreliable one: the tool flow we are using would be dominated by effects
larger than the thing we would be trying to measure.

Open: whether the instruction-composition idea holds up on a second, different
machine. That is the next experiment, and it is the one that decides whether
any of this generalises.
