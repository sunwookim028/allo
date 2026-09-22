# Session report — 2026-09-21/22

Five sentences, then the figures.

**1.** The design did not beat Gemmini, but the deficit was re-measured honestly and now converges rather than grows — 1.27x at 16³ to **1.09x at 64³ at 74.1 % of peak** — with the remainder localised to one DMA prologue worth 55–61 % of it, and MiniTPU independently found the same *shape* of loss on their own machine, so the joint conclusion is that **Gemmini's advantage at these shapes is its software, not its array**.
**2.** Three abstraction extensions landed and **all three were directed work**: stream ports for `@df.kernel` with nine legality rules, `Encoding`/`s.encodable_on` making an ISA's instruction-word budget a checkable schedule property, and a legality rule for `s.dependence` that refuses a provably-false claim while accepting an unprovable one.
**3.** The CHIA search produced **zero architectural abstractions**; what it produced was one near-miss implementation (leaked, retracted) and a series of findings about the *apparatus* — which is a real result about evaluating agent-authored compiler work, and is **not** the result the goal asks for.
**4.** The single most useful finding is that one structural fact — a new primitive has no callers — made an agent's work **simultaneously unrewardable by the objective and undetectably broken by the gates**, which generalises: any gate ladder that exercises a compiler only through existing designs cannot see a new capability in either direction.
**5.** Four independent instruments were caught **failing open** in one night (a tool server that never bound and returned anyway, a usage field reporting $0.00 for a $4.46 call, a self-consistent manifest describing a truncated export, and a leak detector that crashed and was read as "clean"), which is why the standing rule is now *a negative result from an instrument is only evidence if the instrument can be shown to have run*.

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
2. **Memory port requirements.** A buffer that needs two write ports is free on FPGA and does not exist in standard cells. Nothing in the language says a memory's port count; the FPGA resource number stood in for it and could not.
3. **Row-count fields versus layout.** `K = 128` overflows `nr = 127` **on the shipped layout only**; splitting the operand load removes `K` from the row count. The ISA never forbade it — the generator did. An abstraction separating *what the ISA forbids* from *what this generator emits*.
4. **Dependence distance as a machine contract.** `AR_RAW_DIST = 4` is true only because the assembler enforces it; no simulator can see a violation, only RTL. Today an unchecked obligation.
5. **Rate/token semantics.** Deadlock-freedom is an obligation rather than a rule precisely because no per-unit rate exists. It is also what blocks building a unit standalone.
