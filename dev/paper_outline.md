# Paper outline — claims, evidence, and what is still missing

A working note, not a draft. One row per intended section: the **claim** it
would make, the **evidence that exists today**, and the **gap** that has to
close before the claim is defensible. Evidence status is one of

- **measured** — a number from a run that can be reproduced by a committed command;
- **argued** — reasoning from measured things, not itself measured;
- **absent** — nothing yet.

Nothing here is a commitment to a venue or a length. The point is to make it
obvious which track is evidence-limited, so effort goes where a claim is thin
rather than where writing is easy. Keep the evidence column honest; a section
whose gap column is empty is the suspicious one.

See `SESSION_REPORT.md` for the underlying ledger of results.

---

## 1. The problem: co-design decisions need timing *and* power, across an array of architectures

**Claim.** Informed co-design requires measuring how a workload performs in
timing *and* power across *several* hardware architectures — and a flow that
produces only cycles, for only one design, cannot support the decisions
co-design is supposed to make.

**Evidence.** *measured*, for the negative half: we have cycles for one design
family (TinyTPU-isa at T=4 and T=8), on two substrates (Vitis FPGA, FreePDK45
ASIC), against two external references (Gemmini, MiniTPU). The dual-write-port
result is the concrete proof that the array matters — an FPGA-only measurement
hid a structural commitment that a second substrate refused outright
(Synopsys DC ELAB-366), and the banked rewrite kept every cycle.

**The strongest single piece of evidence for this section is our own
optimisation, priced.** Banked burst widening buys **55–61% of the steady-state
deficit** against Gemmini and costs **+74.4% cell area** (3,254,024 µm² against
1,865,314, the pair differing in the widening alone, both meeting timing at
+0.21 ns). On cycles alone it was the obvious win; with the second axis it
becomes a question, and the question turned out to be a different one than the
aggregate number suggested.

`report_area -hierarchy` puts **99.1% of the +1,388,710 delta in two AXI master
ports** — `gmem1` and `gmem2` growing **13×** when widened to `gmem0`'s data
width. The scratchpad, vector registers and accumulator **do not move at all**;
the DMA buffers are 0.9%.

| instance | shipped | widened | change |
| --- | --- | --- | --- |
| `gmem1_m_axi_U` | 56,605 | 744,092 | **13.1×** |
| `gmem2_m_axi_U` | 56,580 | 745,280 | **13.2×** |
| `gmem0_m_axi_U` | 703,513 | 703,151 | — |
| `spm_0_U0` / `vru_0_U0` / `accu_0_U0` | 188,053 / 188,017 / 103,857 | 188,299 / 188,046 / 103,837 | — |

Two consequences, and the second is why this must not be written as a verdict.

- **The cost is not an artefact of rendering memories as flip-flops** — the
  operand arrays are not what grew. It prices widening two master ports and the
  adapters' outstanding-transaction buffering scaling with that width, which an
  SRAM design pays too. On the FPGA that buffering is block RAM, which is
  exactly why Vitis reported +92% BRAM against only +43% FF.
- **"The optimisation is not worth it" is not yet supportable.** The live
  question is *do both operand ports need widening, or would one wide port with
  a shared buffer buy the same cycles?* — one emit-and-synthesise cycle away.
  What survives now is narrower and still worth the section: **the aggregate
  +74.4% is real and measured, and the second axis is what made the question
  visible at all.** The premise holds; it points at the AXI adapters rather
  than at the optimisation.

**Gap.** **Power is absent.** DC power exists only as "indicative, default
toggle rates, no activity data", which is not a number this paper can use. And
the architecture array is one family wide. Both gaps are in the *judge*, not in
the design, and the paper should say so rather than quietly evaluating on
cycles alone.

## 2. The substrate: a composable unit library over Allo

**Claim.** An accelerator can be expressed as a library of parametrized units
composed into an architecture, rather than as one monolithic design, without
giving up synthesizability or performance.

**Evidence.** *measured*. `examples/accelerator/tinytpu_vitis/ip/` — eight
units in 672 lines, composed by `compose.py` into one `Architecture`;
`microarch_isa.py` fell to ~100 lines of instantiation. The composed design
reproduces the published cycle row exactly (171 / 261 / 417 / 483 / 685) and
synthesizes through the ASIC flow. `Unit.check` recomputes free names from the
AST and requires equality with the declaration, so a unit cannot silently
acquire a dependency.

**Gap.** The library composes **one architecture at many parameter sets**, not
many architectures. Until a second `Architecture` exists the word "composable"
is carrying more weight than the evidence. The two intended targets are Groq's
LPU and OpenAI's Jalapeño; the sharpest single blocker is **reduction
topology** — TinyTPU bakes reduction into the unit graph through `p_fwd`, and
Jalapeño reduces through an adder tree.

## 3. The Action hypothesis

**Claim.** An instruction can be defined compositionally as a sequence of
**per-unit Actions** — `vadd` as a `vreg` action plus an `alu` action — with
Actions defined over the Allo model; and this modelling measurably helps
co-design.

**Evidence.** *in progress.* The section must contain: the cost of adding a new
instruction measured **both** ways (not one measured and one estimated), in
files and lines, naming which of encoder / assembler / reference model / unit
bodies derive automatically; a **negative control** — what a deliberately
inconsistent Action is caught by, and what it is *not*; and the effect on the
co-design search's move set.

**Gap.** Three, and they are the reviewer's first three questions.

1. **n=1 machine.** The test of a *default* abstraction is a second machine,
   not a third TinyTPU instruction. The adder-tree reduction unit is the
   intended second consumer precisely because TinyTPU cannot express it.
2. **Status of a claim.** Three different things an Action could be, which must
   not be conflated: a *description* (nothing enforces it), a *checked claim*
   (something refuses a violating program, with the residue named as an
   obligation — `s.dependence`'s shape), or a *hardware guarantee* (the
   structure cannot violate it). MiniTPU's own contract is the middle one and
   they say so: their write-port calendar lives inside `ifndef SYNTHESIS`, and
   on the board two units writing the same register port are simply OR-ed
   together — the assembler is the only enforcement.
3. **Effects are not layouts.** An Action that names a unit's effect but not
   *which operand lands where* cannot describe a reduction tree: MiniTPU's leaf
   order is `sublane * NUM_LANES + lane`, deliberately not lane-major, and it
   reassociates the sum. The same gap sank a schedule primitive in §4 below.

**Framing discipline.** This is **directed work**, not a CHIA discovery, and
the paper must not imply a search produced it. What can be argued is the
consequence: if an instruction is a declaration rather than a diff over a
compiler, the search's move set changes shape.

## 4. Abstraction discovery by agents: what a search proposes, and what survives

**Claim.** An agentic loop over a compiler proposes abstractions of
*qualitatively different kinds*, and distinguishing them requires a grader that
can tell "cannot express" from "cannot refuse" from "already expressible".

**Evidence.** *measured*, and the contrast is the result. Two arms, same seeded
measurement:

- **Arm A** proposed a schedule primitive for memory port capacity and
  diagnosed it as *cannot express*. Graded **(4) already expressible** —
  `Memory(resource=, storage_type=)` existed and both emitters already emitted
  `bind_storage` from it. The *spelling* was new; the capability was not.
- **Arm B** proposed `s.memory_ports(target, write_ports)` — a **legality
  rule**, not a declaration: it refuses when `stores > write_ports × banks`,
  before touching the IR, and it named its own unchecked premise unprompted.
  It passed the discriminating probe (refused the dual-port design, accepted
  the bit-exact banked one) with every gate clean.

**The sharpest sub-result is that Arm B's rule is unsound in both directions**,
found by applying it to cases its author did not design it for: block partition
factor 2 is **accepted** although both stores land in bank 0, and an unpipelined
two-store loop is **refused** although sequential stores need one port. Neither
hole was the one its author predicted. The correct rule needs the layout map
composed with each store's affine index, plus the initiation interval — **both
already in the IR, neither consulted.** Counting banks is not checking layout.

**A second observation about Arm A, from its later real run:** it reached for
`bind_storage` *again* — 142 lines across `customize.py` and two emitters,
declaring `s.bind_storage("i", "buf", "ram_s2p", "bram")`. So across both of
its runs it independently chose memory ports **and** the already-existing
declaration surface. The consistency is itself the finding: what an agent
reaches for is stable, and it is a declaration rather than a checker.

**The class-level result**, which is what survives past this design:

> A resource predicate is the wrong **shape** for this class of claim.
> Counting occupants against capacity discards which element is touched in
> which cycle, and both failure directions are recoveries of that discarded
> information. The sound form is a **calendar**: index map composed with
> initiation interval. The IR already carries both.

The structural contrast is MiniTPU's assembler, whose `write_port` is a *set of
cycles* rather than a count, so a bundle advances until none of its writeback
cycles meets one already booked — both failure directions become impossible by
construction. Two honesty constraints travel with it: their safety is **partly
structural luck** (one write port, no banking, hence no layout to compose,
which makes the finding *predictive* — the bug arrives with banking), and the
calendar has its **own exposure one level down**, being only as right as the
declared writeback span it books.

**Gap.** n=2 arms, one seeded measurement. Replication (3+ unseeded runs) is
queued. And Arm B's diagnosis was *factually wrong on its own tree* while its
artefact was right, which needs stating rather than smoothing over.

## 5. Evaluation: against Gemmini, and on silicon

**Claim.** (To be fixed once the baseline is final.) The honest current form is
that the deficit **converges rather than grows** — 1.27× at 16³ to 1.09× at 64³
at 74.1% of peak — with the remainder localised, and that **Gemmini's advantage
at these shapes is its software, not its array**. One shape (T=8, 16×16×8) is a
supportable 1.18× win, clearing the spread by 4.8×.

**Evidence.** *measured*, both sides over the same window, with host-overhead
symmetry and Gemmini's own noise floor stated. MiniTPU independently found the
same *shape* of loss on their own machine, which is what makes the software
conclusion more than a rationalisation.

**Area** has a defensible methodology and **no numbers yet** — the RTL is
exported and DC has not run. What is settled, and worth stating in the paper
because it is where a fair comparison is usually fudged:

- **The cut is computed, not hand-picked**: the transitive closure of the
  module instantiation graph from Gemmini's `Gemmini` module (138 modules at
  DIM=4, 136 at DIM=8, nothing undefined left over). Rocket, the caches and the
  SoC are out because we have no host; `FrontendTLB` stays *in* although we
  have no translation at all, because cutting it would mean editing Gemmini's
  RTL — reported both ways.
- **Blocks kept although unexercised, all in Gemmini's disfavour**: the conv
  pipeline, the output-stationary datapath, and an fp32 scaling pipeline that
  `defaultConfig` instantiates. Removing any of them is tuning the opponent.
- **The logic-only exclusion is semantic and identical on both sides**: the
  operand scratchpad and the accumulator, *and nothing else* — the two arrays a
  real implementation would build from SRAM macros. Our DMA read buffers stay
  because Gemmini's DMA buffering stays as plain registers; our sequencer's
  small queues stay because Gemmini's depth-2 queue RAMs stay. A *structural*
  rule (drop any module declaring an array-of-reg) was considered and rejected
  because it would also take Gemmini's queue RAMs, and no depth threshold
  separates those from ours. **This is the one judgement in the export**, and
  the paper should say so rather than present the cut as mechanical.
- **The logic-only figure excludes the memory *interface* as well as the
  array**, on both sides equally: with the modules out of the file list DC
  infers nothing for them and the ports become dangling nets. It may therefore
  only ever be compared against another logic-only figure.
- **The instruction-fetch adapter is excluded, and the reason is symmetry.**
  `gmem0` — our instruction port — is **60.0% of the published baseline**
  (681,537 of 1,136,598 µm²), against the scratchpad's 1.1%, the DMA's 2.4% and
  the sequencer's 2.9%. **Gemmini has no instruction port at all**: its
  instructions arrive over RoCC from Rocket, which the module cut removes
  because we have no host. Charging us for the mechanism that replaces the
  block deleted from their side is not a small distortion when it is 60% of
  our area. The symmetric pair is *exclude Rocket and our instruction path*;
  including Rocket would measure a CPU.

  | variant | total | minus `gmem0` | share |
  | --- | --- | --- | --- |
  | T=4 MAXDIM=16 baseline | 1,136,598 | **455,061** | 60.0% |
  | T=4 MAXDIM=64 shipped | 1,865,314 | **1,161,801** | 37.7% |
  | T=4 MAXDIM=64 widened | 3,254,024 | **2,550,873** | 21.6% |
  | T=8 MAXDIM=64 | 2,481,926 | **1,783,365** | 28.1% |

  `gmem0` is **essentially constant at ~700k across all four designs** — it
  scales with neither T, nor MAXDIM, nor the widening, because it is a
  fixed-width adapter. That is independent evidence for the reading: it is
  infrastructure, not part of the machine being compared.

- **The correction that runs the other way, stated in the same breath.** Our
  **sequencer stays in** at ~33k, while Gemmini has no decoder of its own
  because Rocket decodes for it — so we keep paying for something they get free
  from an excluded block. Naming both directions is what makes the exclusion
  defensible rather than convenient.

- **Report two figures, not one.** The excluded figure is the **comparable**
  one; the full figure is the **buildable** one; and the difference is *what it
  costs not to have a host*. That is a more interesting sentence than either
  number alone, and it is the honest description of an architectural difference
  rather than an adjustment.

- **Logic-only is the headline, capacity-matched is the second figure.** Stock
  Gemmini's 320 KiB as flip-flops is 2,621,440 registers against 200,561
  sequential cells in our entire T=4 design; that number measures the memory
  treatment, not either design.
- **A total-area comparison is not available at DIM=4** — the capacities differ
  **4.8×** there and no legal Gemmini capacity closes the gap. At DIM=8 the
  memories match to 2% and the total is defensible. The paper must not quote a
  total at DIM=4.

**Frequency and power — two methodology commitments made before the numbers
exist**, so they cannot be bent afterwards by whatever comes back:

- **Frequency is reported as a DC topographical estimate at a 3.33 ns
  constraint, same flow both sides — never as an achieved frequency.** There is
  no routing and no parasitics beyond wire-load estimates. It is fair for
  comparing designs run identically, which is the only use it gets. This also
  closes a real hole: two designs at different achievable frequencies are not
  comparable on cycles at all, and today we have no frequency figure for
  Gemmini whatsoever.
- **Power is absent, and the reason is the flow, not the effort.** The earlier
  rule here — *publish only if both sides are activity-annotated* — was the
  right instinct and the wrong diagnosis, and is superseded. Under
  `sram_mode='none'` and without place-and-route, power is dominated by **clock
  power into flip-flop arrays** and by **guessed wire capacitance**, both
  larger than the effect being measured. Activity annotation removes the
  default-toggle-rate objection and removes neither of the other two, so even a
  fully annotated number from this flow would be **compromised, not merely
  imprecise**. The cheapest credible energy axis is SRAM macros plus P&R —
  which is the same prerequisite that would replace the area methodology
  wholesale. Recorded as **parked, not cancelled**: activity files are being
  produced anyway, which moves the first row of that cost table from "1–2 days
  of bring-up" to "already done".

**Gap.** DC has not run. The channel-depth fix (`QD=16`, which takes three
non-terminating tiled programs to completion) costs **+9.3% FF** and must
appear next to the cycles, not be absorbed; and if the five published shapes
move under `QD=16`, the row and the gate move with them.

**One correction the area work turned up, which the design pages owe
themselves:** the often-quoted "4 KiB scratchpad, 4 KiB vector registers,
2.1 KiB accumulator" describes **`MAXDIM=64`**, not the `MAXDIM=16` baseline
whose 1,136,598 µm² is published — there both arrays are 64 rows of 32 bits,
256 bytes apiece.

## 6. Negative results about the apparatus

**Claim.** Evaluating agent-authored compiler work fails in specific,
repeatable ways, and the failures are about the *instruments*, not the agents.

**Evidence.** *measured*. Two results carry this section.

- **A gate ladder that exercises a compiler only through existing designs
  cannot see a new capability in either direction.** One structural fact — a
  new primitive has no callers — made an agent's work simultaneously
  unrewardable by the objective and undetectably broken by the gates.
- **Four independent instruments were caught failing open in one night**: a
  tool server that never bound and returned anyway; a usage field reporting
  $0.00 for a $4.46 call; a self-consistent manifest describing a truncated
  export; and a leak detector that crashed and was read as clean. Hence the
  standing rule: *a negative result from an instrument is only evidence if the
  instrument can be shown to have run.*
- The billing instrument **recurred with a diagnosed cause**: a call that
  reported $0.00 had in fact cost $4.33, because a **timed-out call is billed
  and reports zero**. The spend caps read the database rather than the usage
  field for exactly this reason. Worth including because it is the one case
  where the failure mode was later *explained* rather than merely observed.

**Gap.** None in the evidence; the risk is tonal. This section is only worth
writing if it generalises past our harness, and the generalisable form is the
gate-ladder result, not the anecdotes.

## 7. Things deliberately not claimed

Worth a short section, because each of these was claimed at some point in
development and then retracted on measurement. They are cheap credibility.

- **The memory-latency sweep does not tell us about real memory systems.**
  `TPU_AXI_LATENCY` is non-monotonic — 16 beats 0 — so it is an HLS
  *scheduling directive*, not a model of a memory system.
- **Our cosim has no noise floor.** It is deterministic (8 identical counts in
  9 runs); Gemmini's ±spread is Gemmini's. The *pipeline* can fail to produce a
  number, which is a different thing from variance.
- **Alignment did not pay.** 1.26–2.22× cycles *and* a worse period (3.782 ns
  against 2.431 ns). Recorded as a finding, not buried.
- **Depth is the cure for the channel deadlock; the characterisation is open.**
  `Kt >= QD` does not cover the whole failing family, and the obvious counting
  rule is refuted by our own shipped GEMM, which sends one unit 272
  instructions at depth 8 and completes.
- **The held-out experiment proved nothing**, twice: once through a docs leak
  (14 leaks, detector failing open), once because the prompt named the pattern
  outright.
