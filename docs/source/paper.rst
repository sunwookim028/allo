..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

#################################################################################
Substrate Disagreement and Agent-Proposed Legality Rules in Accelerator Co-Design
#################################################################################

**Abstract.** We present a programmable GEMM accelerator written in a Python
accelerator design language, decomposed into eight reusable units, and
evaluated on both FPGA and a 45 nm standard-cell flow against Gemmini at
matched array size. Three results follow. First, the two substrates disagree
about cost so strongly that FPGA-only evaluation is misleading: an optimisation
that appears inexpensive on FPGA costs **+74.4% cell area**, of which **99.1%
is two AXI adapters** rather than compute or memory, and the instruction-fetch
adapter alone is **60.0%** of the baseline design. Second, evaluating on
multi-layer models rather than GEMM shapes changes that optimisation's measured
worth by **four to eight times**, because model depth repays a fixed per-call
cost that a single large GEMM amortises away. Third, two independent LLM search
arms — one without the target property in its prompt — proposed the *same*
architectural legality rule for memory write ports, and the same unsound form
of it, which localises the error to the shape of the abstraction rather than to
either run. We give the corrected shape, and a measured evaluation of modelling
instructions as compositions of per-unit actions.

1. Introduction
===============

Co-design decisions require cost on more than one axis. A flow that reports
cycles alone cannot rank a change that trades area for time, and a flow that
reports one substrate cannot see a structure that is free on that substrate and
dominant elsewhere. This work reports measurements from a design carried
through both, and from an agentic loop given write access to the compiler.

Contributions:

1. **A substrate-disagreement result with component-level attribution.** Two
   structures that are near-free on FPGA dominate standard-cell area, and we
   attribute the cost to specific instances rather than reporting a total.
2. **A workload-dependence result.** The same optimisation measured on GEMM
   shapes and on multi-layer models differs by 4–8×, with a stated mechanism.
3. **A convergence result on agent-proposed abstractions.** Two arms produced
   the same architectural legality rule and the same unsoundness; we give the
   corrected form (a schedule over cycles, not a count over resources).
4. **A measured evaluation of per-unit action composition** for defining
   instructions, including what it provably cannot check.

2. Design
=========

A T×T int8 systolic array with int32 accumulation and a 10-opcode instruction
set (load, multiply-accumulate, vector add, rectify, move-out, loop). The
machine is expressed as eight parametrized units composed by an explicit
architecture description of roughly 100 lines; unit declarations are checked
against their bodies by recomputing free names from the AST.

The shipped configuration (T=4, MAXDIM=16) executes five reference shapes in
**171 / 261 / 417 / 483 / 685 cycles**, bit-exact against a numpy reference,
reproduced independently three times and enforced by a single command.

3. Evaluation methodology
=========================

**Against Gemmini.** Matched array dimension, both sides measured over the same
window, with host overhead treated symmetrically and the comparison design's
own run-to-run spread reported. Our cosim is deterministic; the ±spread in the
tables is Gemmini's.

**Area.** The accelerator boundary is the *transitive closure of the module
instantiation graph* from Gemmini's top module — computed, not hand-picked (138
modules at DIM=4). Rocket, caches and SoC are excluded because we have no host.
Blocks kept although unexercised — the convolution pipeline, the
output-stationary datapath, an fp32 scaling pipeline — are kept because
removing them would tune the opponent.

Two figures are reported per design. The **logic-only** figure excludes the
operand scratchpad and accumulator by an identical semantic rule on both sides,
with stubs supplying empty module headers (omission alone is a link error, not
a black box), so the boundary is *everything that drives the memories and
nothing inside them*. The **capacity-matched** figure is secondary: a total is
defensible only at DIM=8, where capacities agree to 2%; at DIM=4 a 4.8×
capacity gap makes a total meaningless in either direction.

**Instruction supply is excluded, symmetrically.** Gemmini has no instruction
port — its instructions arrive over RoCC from the excluded host. Ours arrive
from DRAM. Charging us for the mechanism that replaces the deleted block is not
a small distortion at 60% of area. We report the excluded figure as the
*comparable* one and the full figure as the *buildable* one, and state that the
difference is the cost of not having a host. The correction running the other
way is stated alongside: our sequencer is counted while Gemmini's decode
happens in an excluded block.

**Frequency** is a synthesis-stage estimate at a 3.33 ns constraint, identical
flow on both sides, reported as a floor and never as an achieved frequency.

**Power is absent, with cause.** Without SRAM macros and place-and-route, power
is dominated by clock power into flip-flop arrays and by estimated wire
capacitance, both larger than the effect under study. Activity annotation
removes one of three objections and not the other two, so an annotated figure
would be compromised rather than imprecise. We report no power number.

4. Results
==========

4.1 Speed
---------

The deficit against Gemmini **converges with problem size** rather than
growing: 1.27× at 16³ to **1.09× at 64³**, at 74.1% of peak. At T=8 on one
shape (16×16×8) we are **1.18× faster**, clearing the comparison's spread by
4.8×. The residual is localised to a fixed DMA prologue worth 55–61% of it.
A second team, measuring an unrelated design, independently found the same
*shape* of loss, supporting the joint conclusion that Gemmini's advantage at
these sizes is its software rather than its array.

4.2 Substrate disagreement
--------------------------

Widening the operand DMA recovers much of the deficit and appears inexpensive
on FPGA. In 45 nm it costs **+74.4% cell area** (3,254,024 against 1,865,314
µm², differing in the widening alone, both meeting timing at +0.21 ns).
Hierarchical attribution places **99.1% of the delta in two AXI master ports**,
which grow 13× when widened; the scratchpad, vector registers and accumulator
do not move. The FPGA reported +43% flip-flops and +92% block RAM for the same
change — the block-RAM axis being exactly the buffering that dominates in
standard cells.

The instruction-fetch adapter is **60.0% of the baseline design** (681,537 of
1,136,598 µm²) against 1.1% for the scratchpad and 2.9% for the sequencer, and
is **constant at ≈700k across all four synthesised variants** — independent
evidence that it is infrastructure rather than part of the machine.

This is not idiosyncratic. Gemmini's published breakdown (Intel 22 nm, 16×16
int8) reports **SRAM at 67.1%** of accelerator area and the spatial array at
**11.3%**.

A third instance: a dual-write-port buffer is free on FPGA block RAM and is
**rejected outright** by the standard-cell flow; a banked rewrite preserves
every cycle.

4.3 Workload dependence
-----------------------

On GEMM shapes the DMA widening is worth **4.3–7.0%** of runtime. On a suite of
multi-layer models it is worth **25–34%**. The mechanism: a model does not make
the problem larger, it makes it longer, so a fixed per-call cost is repaid at
every layer rather than amortised by one large problem. An optimisation ranked
on GEMM shapes is therefore ranked on the workload least sensitive to the cost
it removes.

Three instrument errors surfaced only under the model suite: an analytic cycle
estimator 26–36% low against the shipped build, a burst-cost law 40% optimistic
at short layers, and a mapping that does not terminate in cosim at the shipped
channel depth.

4.4 A channel-depth threshold
-----------------------------

Three legal tiled programs never complete at the shipped stream depth and all
complete bit-exact at twice that depth — same tree, same toolchain, one knob.
The published shapes move **+4 / +4 / +4 / −1 / −11**: small shapes pay
pipeline skew, large shapes get *faster*, because a deeper queue lets the
sequencer run ahead of the units it dispatches to.

The cure is not a characterisation. A depth-versus-tile-count predicate does
not cover the failing family, and the obvious counting rule is refuted by our
own shipped GEMM, which issues 272 instructions to one unit at the shallow
depth and completes. The verification gap is identified: our Kahn-process model
runs the protocol at the declared depth and pronounces these deadlock-free,
because it does not model the sequencer blocking *mid-instruction* across the
queues one instruction fans out to.

5. Agent-proposed abstractions
==============================

We gave an LLM loop write access to the compiler, a gate ladder, and a seeded
measurement, and asked for a capability the language lacked.

**Both arms proposed an architectural legality rule for memory write ports**,
one with the property named in its prompt and one without. Both refused the
design the synthesis flow rejects and accepted the bit-exact banked rewrite
with all gates clean; one stated its unchecked premise unprompted.

**Both are unsound in the same two directions**, found by applying them to
cases their authors had not designed for: a block-partitioned array is
*accepted* although both stores land in one bank, and an unpipelined two-store
loop is *refused* although the stores occupy different cycles. Convergence
across an unseeded arm localises the error to the abstraction's shape:

   A predicate over resources is the wrong form for this class of claim.
   Counting occupants against capacity discards *which* element is touched in
   *which* cycle, and both failure directions recover exactly that discarded
   information. The sound form is a **calendar** — index map composed with
   initiation interval — and the IR already carries both.

An independently developed scheduler in another design is safe against both
directions because its port model is a *set of cycles* rather than a count. Its
authors note they also lack the banking that makes the error possible, so the
finding is predictive rather than comparative.

Two negative results about the apparatus. A gate ladder that exercises a
compiler only through existing designs cannot observe a new capability in
either direction, because a new primitive has no callers. And the loop's keep
rule compared each candidate against the *baseline* rather than the incumbent,
so it discarded a genuine improvement (−31% LUT, latency 206→149) in favour of
a later, worse one; reported improvement counts overstate accordingly.

6. Instructions as compositions of per-unit actions
====================================================

We evaluated defining an instruction as a sequence of per-unit *actions* rather
than restating it in each consumer, by adding one fused instruction both ways
on otherwise identical trees.

**It does not reduce code**: +79/−1 lines in one file, against +77/−6 across
three. It reduces *the number of independent statements that must agree* — from
five specification regions plus two consumers, to one region plus one declared
hardware fact. Two of the replaced restatements were already wrong, and a third
consumer deadlocked on first contact with the new instruction, which is how its
staleness was discovered.

**The discriminating test.** Two actions, each legal in an instruction of its
own, compose into an instruction that is not: two units writing one port in one
cycle. The model refuses it, and the repair is a cycle offset rather than a
port. A count cannot refuse this — the count is one write per unit, and one
port per unit is what the machine has. This is the same predicate/calendar
distinction as §5, reached independently.

**What it cannot check.** Of fifteen deliberately incorrect declarations, five
are refused structurally and four caught by a cost check; **six are invisible
to every static check** — a source naming the wrong operand, subtraction for
addition. The model verifies composition, not arithmetic.

We therefore recommend it where several consumers would otherwise restate one
per-unit fact, and **not** as a default, because only one machine has exercised
it.

7. Threats to validity
======================

Single machine and single new instruction in §6, with the demonstrating
instruction chosen by the abstraction's author. Two arms and one seeded
measurement in §5; replication under unseeded conditions is not done. The
agent-proposed rule reached a correct artefact through a factually wrong
diagnosis — the port count was already declarable; only enforcement was
missing. Area results carry flip-flop memories rather than SRAM macros, which
inflates memory-resident structures and understates nothing in the adapters.
The comparison design's area is not yet measured. Published third-party area
figures are a different node, configuration and scope, and are used only as an
order-of-magnitude check.

**Instrument failures.** Four instruments were observed reporting success
without having run: a tool server that answered without binding, a cost meter
reporting zero for a billed call, a self-consistent manifest describing a
truncated export, and a leak detector that crashed and was read as clean. Three
further checks were run against the wrong object and passed: an incremental
documentation build that skips the files it was meant to inspect, a simulation
run at a different configuration than the one claimed, and a repository check
reading commit titles rather than contents. We therefore treat a negative
result as evidence only when the instrument can be shown to have run, on the
object being claimed.

8. Conclusion
=============

Cycles on one substrate and one workload class are insufficient to rank an
architectural change: we show a factor of 4–8 from the workload axis and a
structure at 60% of area that is invisible on the other substrate. An agentic
loop over a compiler can propose a genuine architectural legality rule, and two
arms converging on the same unsound form is more informative than either arm
alone — it identifies the defect as belonging to the abstraction's shape.
Modelling instructions as per-unit actions removes duplicated statements
without removing code, and cannot see arithmetic errors at all. Whether it
generalises is settled by a second machine, which is the next experiment.
