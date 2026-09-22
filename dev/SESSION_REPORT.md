# Session report — 2026-09-21/22

Five sentences, then the figures. Updated at the end of 2026-09-22; where an
earlier version of this report said something now known to be wrong, the
correction is stated rather than the sentence quietly replaced.

**1.** The design does not beat Gemmini on cycles at most shapes, but the deficit was re-measured honestly and **converges rather than grows** — 1.27x at 16³ to **1.09x at 64³ at 74.1 % of peak** — with the remainder localised to one DMA prologue worth 55–61 % of it, and MiniTPU independently found the same *shape* of loss on their own machine, so the joint conclusion is that **Gemmini's advantage at these shapes is its software, not its array**.
**2.** Three abstraction extensions landed and **all three were directed work**: stream ports for `@df.kernel` with nine legality rules, `Encoding`/`s.encodable_on` making an ISA's instruction-word budget a checkable schedule property, and a legality rule for `s.dependence` that refuses a provably-false claim while accepting an unprovable one.
**3.** **This sentence has been corrected.** It read *"the CHIA search produced zero architectural abstractions"*; that was true when written and is no longer. A search arm proposed `s.memory_ports(target, write_ports)` — **a legality rule, not a declaration** — which refused the exact structure the ASIC flow rejects and accepted the bit-exact banked rewrite, with every gate clean and its own unchecked premise named unprompted. **It is also unsound in both directions**, which is published in the same breath (§ *Agentic discovery*).
**4.** The single most useful finding about *apparatus* is that one structural fact — a new primitive has no callers — made an agent's work **simultaneously unrewardable by the objective and undetectably broken by the gates**, which generalises: any gate ladder that exercises a compiler only through existing designs cannot see a new capability in either direction.
**5.** The single most useful finding about *the design* is that **two structures that are nearly free on FPGA dominate the standard-cell design**, and only the second substrate made either visible: the burst widening's +74.4% cell area is 99.1% two AXI master ports rather than memories, and the instruction-fetch adapter is **60.0% of the published baseline** — which is the project's own premise, that cycles alone cannot support a co-design decision, landing twice on its own design in one evening.
---

## Results

### Against Gemmini, matched array, both sides measured over the same window

| shape | ours (T=4) | Gemmini DIM=4 (median of 5) | ratio |
| --- | --- | --- | --- |
| 4x4x4 | 218 | 208 ±25 | inside noise |
| 8x8x8 | 357 | 324 ±36 | inside noise |
| 16x16x16 | 879 | 691 ±44 | 1.27x |
| 32x32x32 | 3 752 | 2 977 ±34 | 1.26x |
| 48x48x48 | 10 289 | 9 100 ±35 | 1.13x |
| **64x64x64** | **22 123** | **20 287 ±34** | **1.09x** |

At T=8 against a matched DIM=8 build: **16x16x8 is 424 against 500 ±16 — a 1.18x win, clearing the spread by 4.8x.** The only shape where this design beats Gemmini on a supportable margin.

### The shipped design, current

| | |
| --- | --- |
| cycles (T=4, MAXDIM=16) | **171 / 261 / 417 / 483 / 685**, bit-exact, verified 3x independently |
| FPGA (xcu280, 3.33 ns) | BRAM 40, DSP 14, FF 17 075, LUT 26 558, **est. 2.431 ns** |
| ASIC (FreePDK45, DC, memories as flops) | **cell area 1 136 598**, 79.7 % non-combinational, **timing MET +0.21 ns**, 0 violating, 0 hold |
| steady state | **74.1 % of peak at 64³**, still climbing |

### The burst-widening decision, now fully priced

| | cycles 48³ / 64³ | `rbA` write ports | FF | LUT | BRAM |
| --- | --- | --- | --- | --- | --- |
| shipped | 10 289 / 22 123 | 1 | 17 488 | 26 554 | 52 |
| widened, dual-ported | 9 569 / 21 163 | **2 — DC rejects** | 24 001 | 31 396 | 116 |
| **widened, banked** | **9 569 / 21 163** | **1** | 25 026 | 33 799 | 100 |

Identical cycles. **The gain was the widening, not the second write port** — and the banked form synthesises. An FPGA block RAM hands you a second write port free; standard cells have no such primitive, so the FPGA resource table was encoding a structural commitment it could not state.

### Standard cells say something the FPGA could not, twice

**The widening, priced.** Banked burst widening costs **+74.4% cell area**
(3,254,024 against 1,865,314 µm², the pair differing in the widening alone,
both meeting timing at +0.21 ns). `report_area -hierarchy` puts **99.1% of the
delta in two AXI master ports** — `gmem1` and `gmem2` growing **13x** when
widened to `gmem0`'s data width — while the scratchpad, vector registers and
accumulator **do not move at all**. So the cost is *not* an artefact of
rendering memories as flip-flops, and it is why Vitis reported +92% BRAM
against only +43% FF: that buffering is block RAM on an FPGA. **"The
optimisation is not worth it" is not yet supportable** — the live question is
whether one wide port with a shared buffer buys the same cycles, which costs
cycles that must be measured rather than assumed.

**The instruction port.** `gmem0` is **60.0% of the published baseline**
(681,537 of 1,136,598 µm²), against the scratchpad's 1.1%, the DMA's 2.4% and
the sequencer's 2.9%, and it is **essentially constant at ~700k across all four
synthesised designs** — scaling with neither T, nor MAXDIM, nor the widening,
because it is a fixed-width adapter. That constancy is what makes
"infrastructure, not part of the machine" a measurement rather than an
assertion.

**And it is a fairness problem, not only a design one.** Gemmini has **no
instruction port**: its instructions arrive over RoCC from Rocket, which the
module cut removes because we have no host. Charging us for the mechanism that
replaces the block deleted from their side is not a small distortion at 60%.
The reporting rule, settled before the numbers exist: **the excluded figure is
the comparable one, the full figure is the buildable one, and the difference is
what it costs not to have a host.** The correction running the other way is
stated in the same breath — our sequencer stays in at ~33k while Gemmini's
decode happens in an excluded block.

| variant | total | minus `gmem0` | share |
| --- | --- | --- | --- |
| T=4 MAXDIM=16 baseline | 1,136,598 | **455,061** | 60.0% |
| T=4 MAXDIM=64 shipped | 1,865,314 | **1,161,801** | 37.7% |
| T=4 MAXDIM=64 widened | 3,254,024 | **2,550,873** | 21.6% |
| T=8 MAXDIM=64 | 2,481,926 | **1,783,365** | 28.1% |

### The channel-depth threshold, diagnosed and decided

Three legal tiled programs **never complete** at the shipped stream depth 8 and
**all ten complete bit-exact at 16** — same tree, same toolchain, one knob. It
explains what read as random for days: *"not monotone in program size" is what
a threshold looks like from either side.* The five published shapes move
**+4 / +4 / +4 / −1 / −11**: the three smallest pay deeper-FIFO pipeline skew,
and the two largest get **faster**, because a deeper queue lets the sequencer
run further ahead of the units it dispatches to. `stress_isa` holds 492/492
including `ar_distance(4)`.

**The cure is not the characterisation.** `Kt >= QD` does not cover the whole
failing family, and the obvious counting rule is refuted by our own shipped
GEMM, which sends one unit 272 instructions at depth 8 and completes. The
verification gap is named: `kpn_model` runs the protocol at depth `QD` and
calls these deadlock-free, so it is blind to the sequencer blocking
*mid-instruction* across the queues one instruction fans out to.

### Does utilisation amortise with M? (joint experiment, shared axis not shared shapes)

| M | joint with MiniTPU? | T=4 % of peak | T=8 % of peak |
| --- | --- | --- | --- |
| 16 | ours alone | 59.4 | 40.3 |
| **32** | **joint** | **68.5** | **50.6** |
| **64** | **joint** | **74.1** | **57.8** |

A straight line in M with **one** fixed intercept — 1 782 cycles at T=4, 1 014 at T=8 — fitted on M=32/64 and predicting M=16 to within 24 and 8 cycles. So our fixed term is paid **once per call**; theirs is paid once per 32-row launch (`ceil(M/32)`), which is why they are flat at 33.6 %. They replaced their acceptance criterion with this test.

### Agentic discovery, stated at its true strength

| | what | grade |
| --- | --- | --- |
| search, run 1 | burst widening, −160 cycles, 2.3x BRAM | real, **not landed**, "a sensible engineering change, not an architectural discovery" |
| search, co-design | `AGU_TERMS` 3→4 raises encodable nests 3→7→8, bottleneck migrates twice, **chosen nest never changes** | cost with no benefit — then found to be a **prerequisite**, not an alternative |
| abstraction, held-out | 162-line `Schedule.dependence(...)`, 5 places | **implementation from a prose specification, near miss** — its system prompt named the pattern, the attribute shape, the emitter, "must reject" and "a false claim produces wrong RTL", so none of those features are the agent's; it also leaked via the docs, and aborts the compiler on first call |
| abstraction, pilot A | a schedule primitive for memory port capacity, reached for in **both** of its runs | **(4) already expressible** — `Memory(resource=, storage_type=)` existed and both emitters already emitted `bind_storage` from it. The *spelling* was new; the capability was not |
| abstraction, pilot B | `s.memory_ports(target, write_ports)` — refuses when `stores > write_ports × banks`, **before** touching the IR | **the first architectural result.** Refused the dual-port design, accepted the bit-exact banked one, every gate clean, unchecked premise named unprompted. **Unsound in both directions** (below). The property was **seeded** |

**Pilot B's rule is unsound, and that is the more useful half.** Applied to
cases its author did not design it for: `Partition.Block` factor 2 is
**accepted** although both stores land in bank 0, and an unpipelined two-store
loop is **refused** although sequential stores need one port. Neither hole was
the one its author predicted, and its diagnosis was factually wrong on its own
tree — it claimed the language could not *state* port capacity, so the gap was
enforcement only.

**The class-level result, which is what survives past this design.** A
*resource predicate* is the wrong **shape** for this claim: counting occupants
against capacity discards which element is touched in which cycle, and both
failure directions are recoveries of exactly that discarded information. The
sound form is a **calendar** — index map composed with initiation interval —
and the IR already carries both. The structural contrast is MiniTPU's
assembler, whose `write_port` is a *set of cycles* rather than a count, so a
bundle advances until none of its writeback cycles meets one already booked;
both failure directions become impossible by construction. Two honesty
constraints travel with it: their safety is **partly structural luck** (one
write port, no banking, hence no layout to compose — which makes the finding
*predictive*, the bug arrives with banking), and a calendar has its **own
exposure one level down**, being only as right as the declared writeback span
it books.

---

## Apparatus (built, and why each exists)

| thing | bought by |
| --- | --- |
| frozen files from git, fresh tree per candidate, `bwrap` | a candidate could overwrite the stress gate at import via `np.savetxt` |
| nonce-vouched verdicts | a candidate could print `STRESS OK` and exit 0 |
| parametricity + documentation guards | run 1 hard-coded `T` and deleted 450 lines of docs — both improvements by the gate's own measure |
| tiered judge (`legal → correct → confirmed`) | five checks pass a program whose cosim never completes; two of the five derive from the assembler's own header |
| `gen_isa.py --check`, ~6 s, 15 named failures | a formula written `MAXDIM²/T` where the layout computes `(MAXDIM // T) * MAXDIM` agrees everywhere except at the point in dispute |
| per-run cap fix | `spent_since` sums the whole account: one run read $15.82 against a $15 cap, of which $4.46 was its own |

**Ratio, abstraction track: 9 239 lines of harness to 162 lines of agent output.**

---

## The gap the owner named

The abstraction track discovered **apparatus** abstractions, not **architectural** ones. The cause is diagnosable: its gap inventory was drawn from the limitations register, which catalogues front-end and tooling defects, so the agent was aimed at tooling; and the objective rewards payoff at existing call sites, which a new architectural primitive by construction does not have.

Architectural/microarchitectural targets that tonight's measurements have made concrete, none of which has an abstraction today:

1. **Address-term algebra.** `acc` is a legal AGU target; the obstacle is *additive monotonicity* — `base + iv·stride`, no predicate, no saturation, no wrap. 1 150 of 1 226 nests die there. An abstraction over address terms with a declared legality frontier is architectural semantics.
2. **Memory port requirements.** A buffer that needs two write ports is free on FPGA and does not exist in standard cells. Nothing in the language says a memory's port count; the FPGA resource number stood in for it and could not. — **This one was subsequently reached by the search**, as `s.memory_ports`, and the diagnosis had to be refined: the port count *was* already declarable through `Memory(resource=, storage_type=)`; what was missing was the **legality rule**. Reaching it is a real result and the property was seeded, so it is evidence that a search can produce an architectural rule when pointed at a concrete measurement, not that it can find the gap unaided.
3. **Row-count fields versus layout.** `K = 128` overflows `nr = 127` **on the shipped layout only**; splitting the operand load removes `K` from the row count. The ISA never forbade it — the generator did. An abstraction separating *what the ISA forbids* from *what this generator emits*.
4. **Dependence distance as a machine contract.** `AR_RAW_DIST = 4` is true only because the assembler enforces it; no simulator can see a violation, only RTL. Today an unchecked obligation.
5. **Rate/token semantics.** Deadlock-freedom is an obligation rather than a rule precisely because no per-unit rate exists. It is also what blocks building a unit standalone.
