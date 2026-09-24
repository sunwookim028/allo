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
CHIA Agentic Co-Design: Results
####################################

What the CHIA loop (:doc:`/extensions/chia`) has measured: the paid runs, the
claims they support and the claims they do not, the experiments planned
against them, and the figures and statements this page has had to withdraw.
The loop itself -- how to run it, the guards, the evaluator -- is on
:doc:`/extensions/chia`. The retired ``chia-codesign`` effort is on
:doc:`/extensions/chia_codesign`.

Takeaways
---------


- **It works end to end on real tools.** Every accepted number is RTL cosim
  cycles plus csynth area and clock, bit-exact against a frozen reference
  model. **$124.26** spent on the CHIA2026 account so far, of a $500 ceiling
  (``spend.py report``, which recomputes it rather than quoting this page);
  the harness is also exercised with no model at all.
- **It finds real design changes, not yet novel ones.** Best so far: a DMA
  burst widening worth 55-61 % of the steady-state gap to Gemmini (not
  landed; its banked form synthesises with no cycle loss).
- **An agent proposed an architectural legality rule that passed its
  discriminating test and is nonetheless unsound in both directions.**
  ``s.memory_ports`` refused a design needing two write ports and accepted the
  bit-exact banked one; it also accepts a block-partitioned design that needs
  two, and refuses a sequential one that needs one. The property was **seeded**
  (the arm was told where to look, never what to build).
- **A *rate* of discovery is still not shown:** every search is n=1 per arm.
- **The recurring hazard is instruments that fail open.** Four in one night
  reported success after failing. A negative result counts only if the
  instrument can be shown to have run.

Contributions
-------------


- **A judge agents cannot fool.** Nine mechanical guards -- most added after an
  agent found the hole each closes -- mean a win is accepted only when real
  RTL measures it, which is why a worker's false win claim re-scored as
  exactly baseline.
- **A pre-registered search result.** The less directed ``open`` arm, told to
  choose its own target from the numbers, independently reached the same
  address-generator widening that directed analysis had found, with the
  refusal bottleneck migrating and cycles flat exactly as predicted in
  advance.
- **Why agent-built compiler extensions are hard to evaluate.** A new
  primitive has no callers, so an agent-implemented one scored neutral *and*
  passed 291 tests while aborting the compiler on first use -- any gate ladder
  that exercises a compiler only through existing designs cannot see a new
  capability in either direction.
- **A resource predicate is the wrong shape for a port claim.** Counting
  occupants against capacity discards *which* element is touched in *which*
  cycle, and both measured failure directions recover exactly that discarded
  information. The sound form is a calendar -- the index map composed with the
  initiation interval, both of which the IR already carries. MiniTPU's
  assembler (theirs, not ours) does not have the bug because its ``write_port``
  is a **set of cycles** rather than a count; they note their register file has
  one write port and no banking, so they avoided the bug by not yet having the
  structure that creates it, which makes this predictive rather than a
  comparison of care.

What the loop has, measured against the principle
-------------------------------------------------

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

First paid run, 2026-09-19
--------------------------


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

Third paid run, 2026-09-24
--------------------------

``dev/records/tinytpu/chia-evidence/isa-run3-20260924/``: 2 workers x 3
iterations, $60 cap, ``gemini-3.1-pro-preview`` on ``chia2026-tinytpu``,
166.8 minutes, **$20.72** read from opencode's database. The question, what was
expected of it and what would have counted as a negative were committed
**before** the run, in
``dev/records/tinytpu/chia-evidence/prereg-run3-20260924.md``; the result is
appended to that same file.

This is the first run after the harness repair that made the loop run against
the design as it now is: a **package**. The hardware had moved out of
``microarch_isa.py`` into eight modules under ``ip/units/`` while the harness
still named two editable files, so the agent could only edit a 102-line
instantiation with no hardware in it. ``chia_agent/design.py`` now names
fourteen editable paths.

.. list-table::
   :header-rows: 1
   :widths: 12 6 34 20 14 14

   * - worker
     - iter
     - files touched
     - 4x4x4 / 16x16x16
     - verdict
     - $
   * - front-end
     - 1
     - ``ip/units/dma_load.py``
     - 175 / 674
     - not-better
     - 3.33
   * - front-end
     - 2
     - ``ip/assembler.py``, ``ip/units/dma_load.py``, ``ip/units/sequencer.py``
     - 176 / 674
     - not-better
     - 3.84
   * - front-end
     - 3
     - ``ip/units/dma_load.py``
     - 175 / 674
     - not-better
     - 4.26
   * - tail
     - 1
     - ``ip/tinytpu.py``
     - 175 / 674
     - not-better
     - 2.79
   * - tail
     - 2
     - ``ip/assembler.py``, ``ip/tinytpu.py``, ``ip/units/dma_load.py``,
       ``ip/units/sequencer.py``
     - 180 / 683
     - not-better
     - 3.49
   * - **tail**
     - **3**
     - ``microarch_isa.py``, ``ip/tinytpu.py``
     - **175 / 627**
     - **win**
     - 2.84

Baseline, measured by each worker itself: 175 / 674.

**Every candidate reached a graded verdict.** None died at ``setup``,
``policy``, ``import``, ``invariant`` or ``tamper``. That was the run's primary
pre-registered outcome, and it is what "the loop runs end to end" means -- the
$0 harness suite passing does not establish it.

**The reach repair is visible in what the workers edited.** Every candidate but
the winner touched only files under ``ip/``; the winner touched
``microarch_isa.py`` for a single line of parameter default. **Four of the six
could not have been expressed at all** under the editable set the harness had
before the repair. That is behavioural evidence for the decomposition, not an
inference from a file listing.

The win, independently accepted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``accept.py`` on a clean ``git worktree`` with its own ``mlir/`` build, against
a control it measured itself in that run before the candidate's diff existed
(``accept-tail-iter3/``):

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * -
     - 4x4x4
     - 8x8x8
     - 12x12x12
     - 16x16x8
     - 16x16x16
   * - control (this run)
     - 175
     - 265
     - 421
     - 482
     - 674
   * - candidate
     - 175
     - 265
     - **386**
     - **435**
     - **627**
   * - delta
     - 0
     - 0
     - -35
     - -47
     - -47

Total **-129 cycles** over the five shapes; -7.0% at 16x16x16 and -9.8% at
16x16x8. ``bench_isa`` ALL EXACT, ``stress_isa`` 492/492, ``param_check``
exact at ``TPU_MAXDIM=8``, ``TPU_MAXDIM=12`` and ``TPU_T=8 TPU_MAXDIM=32``,
all five cosim testbenches bit-exact, the ``TPU_TB=stress`` RTL testbench clean
over six calls a shape, estimated clock 2.431 ns against the 3.33 ns target,
spec policy clean, and the cross-check agreeing exactly with ``reproduce.sh``'s
published row.

The diff is 22 lines across two files: ``_WIDEN =
TpuParams.widest_burst(T, MAXDIM)`` in ``microarch_isa.py``, which deletes the
``TPU_DMA_WIDEN`` gate and turns the widened operand burst on by default; and
the ``ac2sp`` channel (accumulator to ``dma_st``) deepened from ``QD`` to 32 in
``ip/tinytpu.py``.

**The first half is a rediscovery, and the pre-registration named it before the
run.** ``TPU_DMA_WIDEN`` was already in the tree and off by default. The agent
could not switch it on through the environment -- every ``TPU_*`` variable is
scrubbed before cosim -- so it changed the default. Writing the prediction down
first is what makes calling it a rediscovery a check rather than an argument.
The channel depth was not predicted.

Which half won, measured afterwards at $0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The win is two changes, so ``accept.py`` was run again on the burst-widening
half alone, reusing the control the first acceptance had measured
(``accept-burst-only/``). No model call, so it costs nothing.

.. list-table::
   :header-rows: 1
   :widths: 34 11 11 13 12 13 12

   * -
     - 4x4x4
     - 8x8x8
     - 12x12x12
     - 16x16x8
     - 16x16x16
     - total
   * - control
     - 175
     - 265
     - 421
     - 482
     - 674
     - --
   * - both changes
     - 175
     - 265
     - 386
     - 435
     - 627
     - **-129**
   * - burst widening alone
     - 175
     - 265
     - 386
     - 435
     - 627
     - **-129**

**Identical, shape for shape.** The ``ac2sp`` deepening is worth **exactly zero
cycles** over the five shapes, and the whole win is the half the
pre-registration named before the run.

That agrees, from an independent measurement on a different harness, with what
the objective work found separately: ``channel_depth`` sums to zero over the
five published shapes. That work also established what the depth change *does*
buy -- three legal programs going from hanging to bit-exact -- and that **no
term in either objective scores it**. So the agent proposed a change whose value
the objective cannot see, and kept it because the objective did not penalise it
either. This objective would have accepted that change for no reason and would
equally have discarded it for no reason; neither outcome is a judgement about
the change.

A recorded negative, rediscovered from three lines away
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``front-end``'s first iteration merged ``dma_ld``'s two operand-burst loops
into one bounded by ``max(a_rows, b_rows)``. ``ip/units/dma_load.py:45`` -- in
the file the agent was reading and editing, three lines above the loop it
rewrote -- records that this change "was measured to move nothing", and names
the reason: the bursts are already hidden behind the sequencer's prefetch. It
scored 175 / 674, unchanged, exactly as the note says.

The information was in-band, adjacent and specific, so this is not an artefact
of anything withheld. One hypothesis, worth a designed test rather than a claim
at n=1: a unit-scoped edit is badly placed to weigh a whole-pipeline fact, so
the decomposition that gave the loop its reach back may also have made
locally-plausible global nulls easier to propose.

What this run does not show
~~~~~~~~~~~~~~~~~~~~~~~~~~~

n=1 per arm, like every run before it. No rate of discovery, in either
direction; one win is one win, not a capability.

And the objective it optimised -- summed cosim cycles at 4x4x4 and 16x16x16 --
is now known to be **partial**. Work on another track, completed while this run
was in flight, measured the burst widening as worth **zero** cycles on the
model suite at the scored configuration (861 to 861, 1,530 to 1,530) against
-287 and -583 at ``MAXDIM=64``: at ``MAXDIM=16`` a DRAM row is four packed
words rather than sixteen and the prologue hides the burst for the models. The
relationship inverts at the scored point -- the GEMM shapes see the widening
and the models do not. So a **null** against this objective is weaker evidence
of "no headroom" than it looks, and this **win** is a win on GEMM shapes at one
configuration rather than a win on the workloads. The pre-registration was left
as written and the deviation recorded in its results section, where a reader
can check it.

What Is and Is Not Demonstrated
-------------------------------

The evidence behind the takeaways at the top of this page, one line per claim.

**Demonstrated**

- **An agent proposed an architectural legality rule, and the harness proved it
  discriminates.** Pilot B (2026-09-22, ``chia-abstraction``):
  ``s.memory_ports(target, write_ports)`` refuses when
  ``stores > write_ports x banks``, *before* touching the IR. At a
  harness-owned call site on a frozen design it **refused** the dual-write
  probe (one RAM, two write statements) and **accepted** the cyclically banked
  one (two instances, one write each), which was then bit-exact under csim.
  Every other gate clean: 294/294 tests, ``stress_isa`` 492/492, all design
  cases bit-exact, 20 limits verdicts unchanged, resources flat, +0 cycles. It
  named its own unchecked premise unprompted. **The property was seeded**; the
  abstraction was not. Its diagnosis was half wrong -- it claimed the language
  could not *state* port counts, which is false on its own tree, so the gap was
  enforcement only.
- **The same rule is unsound in both directions**, measured by applying it:
  ``Partition.Block`` factor 2 is **accepted** although both stores land in
  bank 0 and that RTL emits two writes per bank; an unpipelined two-store loop
  is **refused** although sequential stores need one port. Published here in
  the same breath as the success, because a page reporting the primitive and
  not its holes would repeat the failure this project keeps finding in its own
  instruments.
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
  arm, 3 to 7 for the less directed ``open`` arm -- while the chosen nest, and
  therefore the cycles, did not move and area rose. The ``open`` arm's first
  attempt made 16 nests encodable that computed the **wrong answer**, and the
  reference-model sweep caught all 16.
- **The retired ``chia-codesign`` claims C1-C8** replay deterministically,
  including the two agent-found variants (4.07x and 1.98x) at exact numerics.

**Not demonstrated**

- **A rate of discovery.** Every search is n=1 per arm. A rate does not need a
  seed -- it is a property of the sampling process -- so replication (E1 below)
  is the next paid run.
- **That such a rule is sound.** ``s.memory_ports`` is a heuristic and errs
  both ways (above). The correct rule needs the layout map and the initiation
  interval; neither is consulted.
- **That an agent can *find* an architectural gap unaided.** The open arm,
  given measurements only -- and, in the 2026-09-22 run, the shared diagnosis
  as well; see the next entry -- landed in **both** of its runs on ``bind_storage``
  -- a surface Allo already has as ``Memory(resource=, storage_type=)``, which
  both HLS emitters already emit. Its spelling was new; its capability was not.
  It also chose the property its most concrete seeded measurement pointed at,
  which was pre-registered in advance as weak evidence of identification.
- **An unguided condition.** The two arms of the 2026-09-22 run are not
  equally guided, and neither is unguided. This was audited against the
  prompts as actually sent, not from memory, and the result is less flattering
  than the design intended. A **shared system message** carried the corrected
  diagnosis -- monotonicity, ``f2`` as a legal AGU target, Kt=2, that a term on
  ``f2`` costs one of three -- and the freeze rule (``isa_ref`` consumes
  ``expand``) to **both** arms. The ``acc-follows-k`` angle added the full
  (Kt, Nt) masking grid, ``AGU_TERMS=4``, and the second-cause census
  (897 / 274); the ``open`` angle withheld the grid behind a pointer and
  withheld the census entirely. Neither arm was given a mechanism: the strings
  ``saturat`` and ``predicat`` appear in neither worker's system message nor
  either angle. Two consequences bound what may be claimed. First,
  ``acc-follows-k`` is handed the entire intellectual content of the co-design
  finding before it thinks, so a win from it passes a *procedural* test for
  search -- no human inside an iteration -- while being, substantively,
  directed work with an automated typist; it must never be quoted as "the loop
  found this". Second, ``open`` is *less directed*, not undirected: the shared
  message hands it the corrected diagnosis anyway, so **this run contains no
  unguided condition at all.** The strongest attributional claim either arm
  supports is **"the loop found a mechanism, given the diagnosis"** -- not
  "the loop found the idea". The diagnosis was ours, measured and written down
  before the run; a run that tests whether a loop can reach the diagnosis
  itself would have to withhold it from the system message, and is not this
  run. The two arms are therefore reported **separately, always**, and
  labelled *directed* and *less directed*: pooling them would let the directed
  arm's result borrow the other's credibility.
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

Withdrawn: the ``open`` arm called "unguided"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Earlier revisions of this page called the ``open`` arm of the 2026-09-22 run
**unguided**. The prompt audit recorded under
`What Is and Is Not Demonstrated`_ -- taken against the prompts as actually
sent -- shows it was handed the corrected diagnosis through the shared system
message, so it was *less directed*, not undirected. The word is withdrawn.
The arm's pre-registered result -- the refusal bottleneck migrating and the
cycles flat -- is not affected; what changes is only what the arm may be said
to have started from.

Withdrawn: "T is not varied"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An earlier revision of this page said "T is not varied: the shipped design
supports T=4 only", in guard 6 of :ref:`the loop's reference section
<chia-isa-loop>`.
**That was wrong.** The measurement that refutes it is in that guard.
``param_check.py``'s own docstring described the column-block-2 problem
correctly all along; the conclusion drawn from it was the error.


The original co-design effort
-----------------------------

The earlier ``chia-codesign`` search, its claim register and its setup are on
their own page: :doc:`/extensions/chia_codesign`.
