..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

###########################################################################
A Co-Design Compilation Flow as a Harness for Language Models to Improve It
###########################################################################

**Abstract.** We describe a compilation flow that takes an architectural and
microarchitectural specification of an accelerator and produces, from that one
source, both the RTL and the compiler that targets it — so that a proposed
design change yields measured cycles and measured silicon area rather than an
estimate. The flow is built on schedule-level design abstractions over MLIR,
each of which states a machine property and refuses programs that violate it.
We then use the flow as a harness for a language model, and report experiments
in which the model is asked not to tune the design but to **improve the
abstractions themselves**. Two independent search arms proposed the same
architectural legality rule, and the same unsound form of it, which localises
the defect to the shape of the abstraction rather than to either run; we give
the corrected shape. Separately we evaluate modelling instructions as
compositions of per-unit *actions*, and report where it pays and what it
provably cannot check. Measurements from the flow are given throughout as
evidence that its numbers are decision-grade: they reverse two conclusions that
cycle-only, single-substrate evaluation had reached. Against Gemmini under an
identical standard-cell flow, our design is 3.66× larger in logic and **roughly
35× of that difference is the memory interface alone** — 60.2% of our logic
against 6.3% of theirs — a structural difference rather than an efficiency gap,
and one no FPGA measurement had shown.

1. Introduction
===============

Co-design requires a loop, and a loop requires a flow that answers questions
about a *changed* design cheaply enough to ask many of them. The usual
obstruction is that cycles and area come from different descriptions: a
simulator model, an RTL implementation, and a compiler that targets the machine
are written separately and drift apart. A change then costs three edits and a
reconciliation, and a language model driving such a loop spends its effort on
the reconciliation rather than on the design.

We describe a flow in which one specification generates the machine, the
compiler's model of it, and the checks that hold them together, and in which
the design abstractions are *legality rules* rather than annotations — each
refuses a program or a schedule that violates the machine property it states.
This makes the flow usable as a harness: a model may edit the specification, or
the abstractions, and is answered by measurement.

Contributions:

1. **A spec-driven co-design flow** in which architecture, ISA, generated RTL
   and the mapping compiler derive from one source, with the consistency
   checked rather than maintained by hand (§2).
2. **Its use as a language-model harness**, with the guards and the graded
   judge that make an agent's claim checkable rather than plausible (§3).
3. **Experiments on agent-improved abstractions.** Two arms, one unseeded,
   converged on the same architectural legality rule *and the same
   unsoundness*; we give the corrected form — a schedule over cycles, not a
   count over resources (§4).
4. **A measured evaluation of per-unit action composition** for defining
   instructions, including the class of error it cannot detect (§5).
5. **Evidence that the flow's numbers change decisions**: results that reverse
   what cycle-only and FPGA-only evaluation concluded (§6).

2. The flow
===========

**Specification.** A machine is an *architecture* — units, channels and
memories, each unit declaring the names it reads and writes and the parameters
it takes — plus a machine-readable ISA. The architecture is roughly 100 lines
composing eight parametrized units. Declarations are not trusted: unit
signatures are checked by recomputing free names from the AST and requiring
equality with the declaration, and the ISA specification has nine consumers
that a checker holds to it. Two configurations (T=4 and T=8) share the same
unit objects.

**Design abstractions over MLIR.** The units are written in a Python
accelerator design language whose schedule primitives lower to MLIR. The
primitives relevant here each state a machine property and enforce it:

- a **dependence** claim that refuses a provably-false assertion, accepts an
  unprovable one, and records the residue as an obligation the assembler must
  discharge;
- an **encoding** constraint making an instruction word's bit budget a
  checkable schedule property, re-checked after every later primitive so an
  error names the primitive that broke it;
- **memory resource and storage type**, lowered to storage bindings in the
  emitted design;
- **stream ports** on dataflow kernels, with nine legality rules.

**Backend and mapper.** The same specification drives HLS to RTL, from which
cosimulation gives exact cycles, and a standard-cell flow gives area and a
timing estimate. A mapping compiler takes a workload as an einsum with extents,
dtypes and an epilogue, searches the legal mappings onto the fixed machine, and
is graded by a tiered judge: *legal* (the mapping satisfies the machine's
stated rules), *correct* (bit-exact against a reference), *fast* (cycles), with
*confirmed* above *fast* so an estimate can never select a candidate. Workload
specifications are extracted from PyTorch models by tracing and shape
propagation only — the existing PyTorch path synthesises a machine *for* a
model and is the wrong direction for this purpose.

**What closes the loop.** A change to the specification produces a new machine,
a new mapping, new cycles and new area, with the consistency of the parts
checked by construction rather than by review.

3. The flow as a harness
========================

An agent is given the repository, an objective, and a gate ladder; it proposes
a diff. The guards that make the resulting claim worth reading were each bought
by an observed failure: candidate trees are reconstructed from git with the
gate files frozen, so a candidate cannot overwrite the gate that judges it;
verdicts are vouched with a nonce, so a candidate cannot print a passing string
and exit zero; parametricity and documentation guards exist because a candidate
once hard-coded an array dimension and deleted documentation, both of which
improved its score.

Two negative results about harness design generalise beyond this system.

**A gate ladder that exercises a compiler only through existing designs cannot
observe a new capability in either direction.** A newly added primitive has no
callers, so it is simultaneously unrewardable by the objective and undetectably
broken by the gates. This one structural fact explains both halves of a failed
run.

**Selection must compare against the incumbent, not the baseline.** Our keep
rule accepted any candidate whose verdict was in the keep set while scoring
every candidate against the original design, so all accepted candidates scored
alike and a later, worse candidate displaced a genuine improvement (−31% LUT,
latency 206→149). Reported improvement counts overstate accordingly, and we
report the raw counts with the audit rather than the counts alone.

4. Experiment: can a model improve the abstractions?
====================================================

The question is not whether a model can tune a parameter — it can — but whether
it can add an abstraction that lets the *language* state something about the
machine it previously could not.

**Setup.** Two arms over the same compiler and gate ladder, differing in one
respect: one had the target property named in its prompt and a ground-truth
pair of designs seeded in its environment; the other had only the measurements.
A probe rung graded expressiveness by requiring a candidate to **refuse** a
design the standard-cell flow rejects and **accept** a bit-exact rewrite of it.

**Result.** Both arms proposed an architectural legality rule for memory write
ports. Both passed the probe. One stated its unchecked premise unprompted.
Neither is an annotation: both refuse before modifying the IR.

**Both are unsound in the same two directions**, found by applying them to
cases their authors had not designed for. A block-partitioned array is
*accepted* although both stores land in one bank; an unpipelined two-store loop
is *refused* although its stores occupy different cycles. That an unseeded arm
reproduced the defect localises it to the abstraction's shape:

   A predicate over resources is the wrong form for this class of claim.
   Counting occupants against capacity discards *which* element is touched in
   *which* cycle, and both failure directions recover exactly that discarded
   information. The sound form is a **calendar** — an index map composed with
   the initiation interval — and the IR already carries both.

An independently developed scheduler in an unrelated accelerator is immune to
both directions because its port model is a *set of cycles* rather than a
count. Its authors observe that they also lack the banking that makes the error
possible, so the finding is predictive — the defect arrives with banking —
rather than a comparison of care.

**What this does not show.** The property was seeded in one arm, and the other
chose the most salient measurement available to it. Neither demonstrates that a
model can *find* an architectural gap unaided; an earlier arm reached
repeatedly for a surface that already existed, adding a new spelling rather
than a new capability. One arm also arrived at a correct artefact through a
factually wrong diagnosis: the port count was already declarable, and only
enforcement was missing.

5. Experiment: instructions as compositions of per-unit actions
================================================================

If an instruction is a declaration rather than a diff over a compiler, the move
set available to a search changes shape. We tested this directly by defining an
instruction as a sequence of per-unit *actions* and adding one new fused
instruction both ways, on otherwise identical trees, with the shared hardware
change cherry-picked identically into both so only the difference is measured.

**Cost.** +79/−1 lines in one file with actions, against +77/−6 across three
files without. **It does not reduce code.** What it reduces is the number of
independent statements that must agree: five specification regions plus two
consumers, down to one region plus one declared hardware fact. Two of the
replaced restatements were already wrong, and a third consumer deadlocked on
first contact with the new instruction, which is how its staleness surfaced.

**The discriminating test.** Two actions, each legal in an instruction of its
own, compose into an instruction that is not: two units writing one port in one
cycle. The model refuses it, and the repair is a cycle offset, not a port. A
count cannot refuse this — the count is one write per unit, and one port per
unit is what the machine has. This is the predicate/calendar distinction of §4,
reached independently and by a different route.

**What it cannot check.** Of fifteen deliberately incorrect declarations, five
are refused structurally and four are caught by a cost check; **six are
invisible to every static check** — an action naming the wrong operand,
subtraction where addition was meant. The model verifies *composition*, not
arithmetic. A control mutant that changes nothing is correctly not caught,
which is what shows the harness ran the file it was given.

**5.1 Can the behavioural model be derived from the structural one?** The two
descriptions were written independently: a `compose` unit declares what names a
body needs from outside, an `actions` unit declares what effect lands where and
when. If the second can be *derived* from the first, an Action becomes an
annotation on an existing model rather than a second declaration of every unit.
We built the derivation and made the diff a gate.

**Of 28 ports the Action model declares, 24 stop being independent**: 20 are
the architecture's own channels and memories, read off the same AST the
structural checker already parses, and 4 more stand for them. Two units derive
completely. The residue is exactly the interesting part:

- **3 are arithmetic** — a multiplexer, a multiply-accumulate, an ALU. No
  composition can carry these: they are what a unit *does*, not what it is
  connected to.
- **1 counts instructions** and corresponds to nothing physical.
- **The control path cannot be an action at all.** A unit reads its dispatch
  queue inside a step it is already spending; an action is per-row or
  per-instruction, and the per-instruction form adds a head step to the work
  count the instruction header carries. Measured rather than asserted: one
  instruction's cost goes from 6 to 7.
- **The derivation found three on-chip memories the Action model never had** —
  real state, invisible to the ISA because no instruction field names a row of
  them.

**And the load-bearing finding: the two models do not agree on what a unit
is.** A `compose` unit is a *kernel*, replicated by an instance count. An
`actions` unit is a *dispatch domain*, one work counter. They are the same
object for seven of eight units and a different object for the array, where
T×T kernels are one dispatch domain the ISA calls `array`. No naming convention
closes that: a header field counts rows per instance across T×T kernels, and
the behavioural model has no word for the distinction.

**5.2 The two defects the second machine exposed, and what fixing them cost.**
Lane width was previously read off a shadow one-row *state*, because a lane map
seemed to need addressed storage. It does not: giving the structural
``Channel`` a lane count and deriving its element type from it removes the
shadow state entirely, and a fold takes its operand width from the channel the
value arrived on. **The missing thing was a lane count, not addressed state** —
a channel has no rows, no bank map and no collision rule, and a fold needs none
of them.

The work model charged latency as occupancy, reporting 80 steps where the
hardware does 16. Separating an action's **span** (which a declared latency
lengthens) from its **initiation** (the busiest resource a row books, read off
ports already declared) gives 16, leaves the calendar unchanged, and declares
nothing new. Every count on the first machine is unchanged, because a unit with
no declared latency has span equal to initiation — which is why one machine
could never have shown the defect.

**It cost a static catch, and we kept the loss visible.** One deliberately
wrong declaration — an ALU declared one lane operation per step instead of two
— was previously caught by the cost check *because of the defect*: the broken
model pushed the work into a third cycle. Correctly pipelined, the accumulator's
single read port binds at two per row either way, so no work count can see the
ALU's width. The fifteen-mutant split moves from 5 / 4 / 6 to **5 refused
structurally, 3 caught by cost, 6 invisible, 1 caught by nothing**.

**5.3 Consolidation: not one file, and the reason is a result.** The intended
end state was a single file. It should not be, on two independent grounds.
First, the behavioural model imports nothing from the compiler, and that
independence is what lets an ISA be composed and checked without the front end;
merging would make it depend on the dataflow layer and on MLIR bindings for no
new capability. Second, the specification's discipline is that the spec is the
source of truth and the design is held to it — **a spec that reads its ports
out of the design cannot check the design's ports.**

What is available, and what we built, is a **checked projection**: where the
two models overlap they can no longer drift, and where they cannot overlap the
reason is written down and checked in both directions. Two architectures now
compose, and the second one's behavioural model is *derived* rather than
written.

So the end state is reachable in a modified form, and the modification is the
finding: **`Unit` means two different things**, and one file would mean one of
the two meanings silently winning.

**Recommendation.** Worth using where several consumers would otherwise restate
one per-unit fact; **not** a default, because one machine has exercised it.
This is directed work, not a search result, and is reported as such.

6. Do the flow's numbers change decisions?
==========================================

Three results, each of which reverses or reframes a conclusion that a cheaper
evaluation had reached.

**6.1 Speed, against a matched baseline.** Against Gemmini at matched array
dimension, measured over the same window with host overhead treated
symmetrically, the deficit **converges** rather than growing — 1.27× at 16³ to
**1.09× at 64³**, at 74.1% of peak — with the residual localised to a fixed DMA
prologue worth 55–61% of it. At T=8 on one shape we are **1.18× faster**,
clearing the comparison's spread by 4.8×. A second team measuring an unrelated
design independently found the same *shape* of loss, supporting the joint
conclusion that the advantage at these sizes is software, not array. Our
cosimulation is deterministic; the spread in the tables is the comparison's.

**6.2 The substrates disagree, and the FPGA is the misleading one.** An
optimisation that recovers much of the deficit appears inexpensive on FPGA. In
45 nm it costs **+74.4% cell area**, and hierarchical attribution places
**99.1% of the delta in two AXI master ports**, which grow 13× when widened;
the scratchpad, vector registers and accumulator do not move. The FPGA reported
+43% flip-flops and +92% block RAM — the block-RAM axis being exactly the
buffering that dominates in standard cells.

Separately, the **instruction-fetch adapter is 60.0% of the baseline design**,
against 1.1% for the scratchpad and 2.9% for the sequencer, and is constant
across all four synthesised variants — independent evidence that it is
infrastructure rather than part of the machine. This is not idiosyncratic:
Gemmini's published breakdown reports SRAM at 67.1% of accelerator area and the
spatial array at 11.3%. A third instance: a dual-write-port buffer is free on
FPGA block RAM and is **rejected outright** by the standard-cell flow, while a
banked rewrite preserves every cycle.

**6.2c The comparison, measured.** Gemmini DIM=4 and our T=4 MAXDIM=64 are now
synthesised under an identical flow, with every parameter compared and the
standard-cell library matched by checksum rather than assumed.

Logic-only, **Gemmini is 382,026 µm² and we are 3.66× that**; full against
full, 1.88×. But the shape differs more than the size, and the shape is the
result:

.. list-table::
   :header-rows: 1

   * -
     - Gemmini DIM=4
     - ours (T=4)
   * - memory-interface / adapter logic
     - 23,885 µm² (**6.3%**)
     - 840,160 µm² (**60.2%**)
   * - non-combinational share of logic
     - 38.3%
     - 80.1%

**We are 3.66× larger in logic, and roughly 35× of that difference is in the
memory interface alone.** Our four AXI masters are 60.2% of our logic; their
TileLink reader, writer and transaction-tracker path is 6.3% of theirs. This is
a structural difference rather than an efficiency gap — TileLink at this width
does not produce this structure — and it is §6.2's finding arriving from the
comparison rather than from our own hierarchy.

**6.2d The shape difference, at two sizes.** Gemmini DIM=8 logic-only is
**524,402 µm²**, timing met, same flow and same stub rule. Two readings follow,
and the second is the one that does not depend on any argument we could be
accused of choosing.

*Doubling the mesh costs +37.3% of logic* (382,026 → 524,402) for **4× the
PEs** — sub-linear, which is what one expects when control and the memory path
dominate rather than the array. That is the adapter finding of §6.2 arriving
from the opposite direction, on someone else's design.

*And the non-combinational share is stable and opposite on the two designs:*

.. list-table::
   :header-rows: 1

   * -
     - Gemmini DIM=4
     - Gemmini DIM=8
     - ours, every run
   * - non-combinational share of logic
     - 38.3%
     - 37.2%
     - **≈80%**

**Their logic is gate-heavy; ours is flop-heavy, at both sizes.** No node
scaling, no capacity matching and no exclusion rule enters that comparison; it
is a share of each design's own logic.

*Corrected after the full DIM=8 run.* An earlier version of this paragraph
added "and regardless of the memory treatment", which is true of our design and
**false of theirs**: with memories rendered as flip-flops their share moves from
37.2% to **60.7%**, while ours is ≈80% either way. So the substrate distortion
is **larger for their design than for ours** — the opposite of what we
expected — and the gate-versus-flop statement holds on the logic-only pair
only. It is stated that way now. It is therefore a structural statement about **what HLS emits versus
what a Chisel generator emits**, not about either accelerator, and it is the
most transferable result the area track produced.

**A caution that belongs beside the frequency numbers.** Gemmini DIM=8's worst
path group closes at **exactly 0.00 ns** — zero violating paths, zero hold, a
genuine pass, but with nothing spare, against +0.20 to +0.21 ns on every one of
ours. It must be written as *"closes at 3.33 ns"* and never as headroom: a
`0.00` that reads as "met" in a table becomes "both close at 3.33 ns, equally"
in prose. If a frequency comparison is ever attempted, Gemmini's is the number
that moves first.

**6.2e One change, measured two ways.** Gemmini DIM=8 *full* is
**1,138,578 µm²**, against 990,938 at DIM=4 — **+14.9%** for doubling the mesh.
The logic-only pair gives **+37.3%** for the same change. The stubbed
scratchpad and accumulator are 53.9% of the full figure, and that mass is
insensitive to the mesh, so it dilutes the delta.

**The same design change reads as +14.9% or +37.3% depending only on how the
memories were treated.** That is the strongest argument in this work for
quoting the logic-only pair — stronger than the gate-versus-flop share above,
because it is *one change measured two ways* rather than an inference about
which component dominates.

**And that run misses timing**, the only one in the set that does not close:
worst slack −0.01 ns on a 3.33 ns target, 80 violating paths in a
491,512-cell design. Per the methodology fixed before any number existed, this
is reported as **"the comparison design's RTL was not targeted at this
constraint"**, not as "it is slower": 0.3% of the period is what a longer
period, a higher effort or a retarget would absorb, and a synthesis-stage
estimate is a floor rather than an achieved frequency. The miss is stated and
the inference refused.

**The clean pair is still two runs away.** Our 1,396,966 against their 524,402
is 2.66×, but that is our T=4/MAXDIM=64 against their DIM=8 — not a
single-variable comparison. The pair that answers the question is our
T=8/MAXDIM=64 at QD=16 against this run, and both remain queued.

**Two methodological points are now settled on evidence rather than argument.**
First, the exclusion of our instruction-fetch adapter: **no cells inside the
Gemmini boundary match instruction fetch at all.** What exists is command queue
and decode for commands arriving over RoCC, 31,715 µm². The symmetry argument
holds because the structure is genuinely absent, not because we assumed so.
Second, the one arguable cut — the DMA's address-translation block, kept in
because removing it would mean editing their RTL — is **under 1.1% of their
logic** either way it is counted, so the ambiguity was immaterial.

As with the PE array, DC's flattening leaves these as name-prefix sums rather
than hierarchy lines, and the committed reports state that limit.

**6.2a What can be compared before the comparison design is synthesised.** Our
own runs are complete and the comparison design's are not, but two comparisons
are available now from its *published* figures, and they differ sharply in what
they support.

*Structural, and node-independent.* Gemmini's published breakdown (Intel 22 nm,
16×16 int8, 256 KB scratchpad, 64 KB accumulator) gives memory **67.1%** of
accelerator area and the spatial array **11.3%**. Our baseline (FreePDK45,
4×4, flip-flop memories) gives the instruction-fetch adapter **60.0%** and
non-combinational cells **79.7%** of total area. These are fractions of each
design's own total, so no node or capacity scaling is involved, and they agree
on the conclusion that matters:

   On both designs the arithmetic is a small minority of the silicon. The
   majority is state and supply — SRAM capacity in one case, instruction
   supply and register-resident state in the other — and the two designs reach
   that condition by entirely different routes.

That is the substrate-disagreement result of §6.2 restated from an independent
source: the component a designer optimises is not the component that sets the
area.

*Absolute, and only as a sanity check.* Gemmini's accelerator-only total is
**858K µm²** at 22 nm (1,029K less the 171K host-CPU line their table reports
separately). Ours is **1,137K µm²** at 45 nm, or **455K** excluding
instruction supply. A crude standard-cell node scaling of 45 nm to 22 nm is
roughly 3–4×, which places our comparable figure an order of magnitude below
theirs — as it should be, for a 16-PE machine with 2.5 KiB of local memory
against a 256-PE machine with 320 KB. We report this only as confirmation that
nothing is wildly wrong. It is **not** an area-efficiency claim: the
configurations differ by 16× in PE count and 128× in memory capacity, the
memories are flip-flops on our side and SRAM macros on theirs, their figure
includes place-and-route and ours does not, and the node factor is the
dominant uncertainty.

*What is still missing, and why it is the useful one.* A per-PE comparison of
the arithmetic array alone would be free of the memory-technology difference
entirely, since the array is pure logic on both sides — Gemmini's published
figure is **453 µm² per PE** at 22 nm. Our per-instance array area is not in
the committed reports, so this comparison is one hierarchical area report away
and is the first thing to extract when the queued runs execute.

**6.2b Two further instances, from an independent design.** The same pattern
was found in an unrelated FPGA accelerator by its own team, auditing for ASIC
readiness. It has **~63,632 bits held in LUT shift registers**, 15,616 of them
deliberately — one module withholds a payload reset specifically so its delay
line stays a pure shift register. A standard-cell flow has no such primitive,
so every one of those bits becomes a flip-flop, and none of it is visible in a
LUT count. It also holds **512 KiB of lookup tables initialised by
``$readmemh``** across 64 instances, likewise with no standard-cell equivalent.
Their total on-chip storage is ~1.08 MiB against the ~10 KiB in the variant we
measured at 79.7% non-combinational area, so our figure is the one they now use
to predict their own outcome.

This is the same class as the dual-write-port buffer of §6.2 — a primitive that
is free on one substrate and absent on the other — found independently, in a
different design, by a different team, and it raises the count of such
structures to four.

A related figure was corrected twice in one day by its own side, and the
second correction is the methodologically interesting one. That team's
per-launch overhead, published by us as *host software*, is mostly **the device
reloading its own instruction memory** (~81% of the cost removed by reducing
launch count). The remaining ~30 µs register path was then attributed to bus
latency — and, measured against a **no-bus control** (identical driver code
against a plain buffer instead of the mapped device), **the bus is 0.442 µs and
the rest is the interpreter**. The original error was timing single operations
with a timer that costs more than the operation, and the tell was that the
control measured *slower* than the real thing. Each correction made that side's
position worse, and the second one invalidated a fix they had planned, which
could not have recovered more than 0.44 µs. Instruction supply turns out to
dominate both machines in different currencies — 60.0% of our silicon area, and
the majority of their per-launch time.

**6.3 The workload class changes the answer by four to eight times.** On GEMM
shapes the same optimisation is worth **4.3–7.0%** of runtime; on multi-layer
models, **25–34%**. A model does not make the problem larger, it makes it
longer, so a fixed per-call cost is repaid at every layer rather than amortised
by one large problem. An optimisation ranked on GEMM shapes is ranked on the
workload least sensitive to the cost it removes. Three instrument errors
surfaced only under the model suite, including an analytic cycle estimator
26–36% low against the shipped build.

**6.4 A threshold the model could not see.** Three legal tiled programs never
complete at the shipped channel depth and all complete bit-exact at twice that
depth — same tree, same toolchain, one knob — with the published shapes moving
+4/+4/+4/−1/−11, the largest getting *faster* because a deeper queue lets the
sequencer run ahead. The cure is not a characterisation: the obvious counting
rule is refuted by our own shipped program, and our Kahn-process model
pronounces these deadlock-free because it does not model a unit blocking
*mid-instruction* across the queues one instruction fans out to. This is a
stated gap in the flow, not a solved problem.

7. Threats to validity
======================

**The area comparison covers one design point.** Gemmini DIM=4 and our T=4
MAXDIM=64 are synthesised under an identical flow; DIM=8 and the
capacity-matched pair are not run, so no total-area claim is made and the
result is logic-only. DC's auto-ungrouping dissolves instance hierarchy at the
effort these runs use, and no reporting option recovers it — recovering it
requires changing synthesis, which would break comparability with every run
already done. The component figures are therefore **sums over a name-based
selection of flattened leaf cells**.

That method was validated rather than assumed. For a block whose boundary
*did* survive, the hierarchy line reports 705,486.0 and summing the 230,892
flattened leaf cells beneath it gives **705,485.998** — identical to the digit.
So the arithmetic is exact and the only uncertainty is **selection**: whether a
cell named for one unit belongs to it, and whether logic merged across a
boundary was renamed away. On our side the selection is clean (every matched
cell accounted for, and the per-unit distribution reproducing across two
independent designs to the digit); on the comparison design it rests on the
generator's naming surviving flattening, which is the weaker of the two and is
stated as such.

Area results use flip-flop memories rather than SRAM macros, which inflates
memory-resident structures; it does not flatter the adapter findings, which are
the ones we rely on. Power is reported nowhere: without macros and
place-and-route it would be dominated by clock power into flip-flop arrays and
estimated wire capacitance, both larger than the effect under study, so
activity annotation would remove one of three objections and leave two.

§4 is two arms and one seeded property, without unseeded replication. §5 is one
machine and one new instruction, with the demonstrating instruction chosen by
the abstraction's author.

**Cycles and area are joined by declaration, not by proximity.** Cycle counts
come from Vitis cosim and area from Design Compiler. Each flow is internally
consistent, so each flow's own check passes while the *join* between them can
be wrong, and the errors this work made most often lived exactly there. We
therefore treat a cycle count and an area figure as describing one machine only
when one of two conditions holds, and a checker
(``allo/backend/asic/tools/check_pairing.py``) decides which: either both cite
the **same committed RTL export**, verified by the manifest checksum the
synthesis run recorded, in which case their configuration agrees by
construction; or both **declare every configuration parameter** --- ``T``,
``MAXDIM``, ``QD``, ``DMA_WORDS`` --- and the four agree. A parameter that is
absent on either side is treated as *cannot pair*, never as a match, because
the exports predating a parameter record nothing about it and the default at
the time was usually not today's.

Applying that rule to this repository refuses more than it admits, and two of
the refusals are results rather than formalities. **No model-level cycle count
can currently be paired with any committed area figure**: the workload numbers
were measured at ``QD=16`` and every committed TinyTPU export was emitted
before ``QD=16`` became the default and records no ``QD``, so the end-to-end
claim reaches cycles and stops. And the ``T=8`` area is **orphaned**: the
export it was synthesised from was re-emitted two days later, so the RTL a
reader finds at that path is not the RTL that area describes. Neither fact was
visible to any single-flow check; both were produced by the join.

**Instrument failures.** Four instruments were observed reporting success
without having run: a tool server that answered without binding, a cost meter
reporting zero for a billed call, a self-consistent manifest describing a
truncated export, and a leak detector that crashed and was read as clean. Three
further checks ran against the wrong object and passed: an incremental
documentation build that skips the files it was meant to inspect, a simulation
run at a different configuration than the one claimed, and a repository check
reading commit titles rather than contents. We therefore treat a negative
result as evidence only when the instrument can be shown to have run, on the
object being claimed — and we report that rule here because it changed which of
our own results we were willing to keep.

A second rule, from the correction in §6.2b: **attributing a cost requires a
control that shares everything but the mechanism under test.** A measurement
can be accurate and still attribute its result to the wrong cause, and the
error is invisible from inside the measurement. The tell in that case was that
the control came out slower than the thing it controlled for.

8. Conclusion
=============

A co-design flow whose machine, compiler and checks derive from one
specification is usable as a harness: a language model can be pointed at the
abstractions rather than at the parameters, and answered by measurement. Doing
so produced an architectural legality rule twice over, independently, together
with the same unsoundness — which is more informative than either arm alone,
because it identifies the defect as belonging to the abstraction's shape rather
than to a run. The corrected shape is a schedule over cycles, and the same
distinction reappears when instructions are modelled as compositions of
per-unit actions. Whether that modelling generalises is decided by a second
machine, which is the next experiment; and the flow's own numbers are worth
having chiefly because they reversed two conclusions we had already drawn
without them.
