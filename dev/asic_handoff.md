# ASIC evaluation handoff — state at the credit limit

Written because a chat channel is not durable and a result that exists only in
a message can be lost. Anyone picking this up — a synthesis session, a later
agent, or me after a reset — should be able to continue from this file alone.

## The decision that changes the runs

**`QD=16` is adopted.** Measured against a `QD=8` control in the same tree that
reproduced the published row exactly:

| shape | QD=8 | QD=16 | delta |
| --- | --- | --- | --- |
| 4x4x4 | 171 | 175 | +4 |
| 8x8x8 | 261 | 265 | +4 |
| 12x12x12 | 417 | 421 | +4 |
| 16x16x8 | 483 | 482 | -1 |
| 16x16x16 | 685 | **674** | **-11** |

Not a uniform cost: the three smallest pay deeper-FIFO pipeline skew, the two
largest get *faster* because a deeper queue lets the sequencer run further
ahead of the units it dispatches to. Three legal tiled programs go from never
completing to completing bit-exact. `stress_isa` holds 492/492 including
`ar_distance(4)`, so the accumulator's dependence contract survives.

**It has not landed on `main`**, deliberately — landing it would silently
invalidate any in-flight synthesis of our designs. When it lands, `reproduce.sh`
EXPECTED moves to `175 / 265 / 421 / 482 / 674` and **our three ASIC runs need
redoing**. The four Gemmini runs are unaffected.

**And the Gemmini parity sweep was measured at `QD=8`.** Since the largest
shape gets faster, the deficit must be re-measured before any comparison is
restated. A narrower deficit claimed on the old sweep would be unearned.

## The six runs

Order: Gemmini DIM=4 logic-only, DIM=8 logic-only, our T4_MAXDIM64 and
T8_MAXDIM64 logic-only, then the DIM=4 and DIM=8 capacity-matched pair.
T4_MAXDIM16 logic-only is deliberately **not** run — our published 1,136,598
um2 is a with-memory figure and a logic-only twin would invite a wrong
comparison.

Reference wall times, same machine and flow: T=4/MAXDIM=16 with flop memories
37 min, T=4/MAXDIM=64 56 min, T=8/MAXDIM=64 72 min. Logic-only should be well
under these because the flop-memory array dominates elaboration.

## The two questions whose answers must not be lost

1. **Does anything inside the `Gemmini` boundary do instruction fetch?** The
   symmetry argument for excluding our `gmem0` rests on the answer being *no* —
   Gemmini's instructions arrive over RoCC from Rocket, which the module cut
   removes because we have no host.
2. **Do `StreamReader` / `StreamWriter` / `XactTracker` carry adapter mass
   comparable to ours?** A zero is a finding and must be written as *"TileLink
   at this width does not produce this structure"*, never as an area advantage.
   If they carry comparable mass, we compare adapter to adapter instead.

## Where results belong

In git, under `examples/accelerator/tinytpu_vitis/asic_synthesis/`: the area
numbers, the per-instance `report_area -hierarchy` breakdowns including the
Gemmini ones, worst slack and violating-path counts, and the settings each run
used. Not in a chat message.

## Next experiments, in order, after the six

1. **Narrow `gmem0` to 32 bits.** Cheapest in the set and it does not touch the
   dataflow. The trade is the ~350 cycles the one-burst program load bought
   against an adapter that is 60.0% of the published baseline.
2. **One wide operand port instead of two.** `gmem1` and `gmem2` together are
   99.1% of the widening's +74.4%. Serialising operand A and B through one
   adapter **costs cycles** — `dma_ld` issues both bursts concurrently today —
   so the cycle cost must be measured, not assumed away.

Both need a variant emitted and its cycles measured before any area number
means anything. Neither was started; the agent dispatched for them died at the
credit limit before its first tool call.

## Methodology already settled (see `dev/paper_outline.md` section 5)

- Logic-only leads; total area is defensible only at DIM=8, where capacities
  match to 2%. At DIM=4 the 4.8x capacity gap makes a total meaningless.
- The exclusion is semantic and identical on both sides: operand scratchpad and
  accumulator, nothing else. A structural array-of-reg rule was rejected
  because it would also take Gemmini's queue RAMs.
- Omitting a module does **not** black-box it — it makes the reference
  unresolvable (LINK-5). Stubs come from `asic_synthesis/tools/make_stubs.py`,
  and with them the boundary is "everything that drives the memories, nothing
  inside them", identically on both sides.
- Frequency is a DC topographical estimate at 3.33 ns, reported as a **floor**,
  never as an achieved Fmax. Our four runs: +0.21 / +0.21 / +0.20 ns, zero
  violating, so >= 320.5 / 320.5 / 319.5 MHz.
- A Gemmini timing miss is reported as a miss, and written as *"Gemmini's RTL
  was not targeted at this constraint"*, not *"Gemmini is slower"*.
- Power is **absent with a reason**: under `sram_mode='none'` and without P&R
  it is dominated by clock power into flip-flop arrays and guessed wire
  capacitance, both larger than the effect. Activity annotation removes one of
  three objections. Our SAIFs exist anyway (`saif_capture.py`), so the first
  row of that cost table is already done if macros and P&R ever arrive.

## Cheapest outstanding request: one `report_area -hierarchy`, no re-synthesis

**This needs no synthesis run and no approval slot.** The mapped databases for
our four designs already exist under `/scratch/users/sk3463/build_*`. What is
missing is one report off a database that is already there — very likely a
report that was already generated, since per-instance numbers for
`T4_MAXDIM64_shipped` and `..._burstwiden` have already been quoted
(`gmem0/1/2`, `dma_ld_0_1_U0`, `spm_0_U0`, `vru_0_U0`, `accu_0_U0`).

**What we need:** the **PE array** instance line, for
`T4_MAXDIM16_shipped_baseline` and `T4_MAXDIM64_shipped`. The full
per-instance table is welcome; the array line is the one that matters.
Committed under `asic_synthesis/reports/<variant>/`, not messaged.

**Why it is the single most valuable number still missing.** The spatial array
is **pure logic on both sides** — no SRAM, no memory-technology difference, no
capacity mismatch. Gemmini's published figure is **116K µm² for 256 PEs at
22 nm = 453 µm²/PE**. With our array area we get a per-PE comparison that
survives every objection currently attached to the total-area comparison, and
it is available *before* their six runs execute.

The remaining node difference (45 nm against 22 nm) is then the only
uncertainty, and it is one clearly-named factor rather than four compounding
ones. Our T=4 array has 16 PEs; a per-PE figure also makes the T=4 and T=8
designs directly comparable to each other, which the totals are not.

Everything else in this file stands unchanged.

## Answers to the two open questions (2026-09-22, late)

**1. Yes — commit the other session's Gemmini DIM=4 results, with the settings
read out of the build directories rather than assumed.** Logic-only 382,026 and
full 990,938 existing only on scratch is exactly how a result gets lost, and
reading the settings out is the same discipline that made the T=8 row usable:
every DC and mflowgen parameter compared, and the standard-cell library matched
by checksum, because two runs can agree on every setting and still resolve a
different `stdcells.db`. Record it as *"run by another session, settings
verified identical"* — provenance, not a caveat, since the check removes the
doubt rather than raising one. If any setting differs, say which and do not
average over it.

With our T=4 MAXDIM=64 logic-only in flight, that completes the approved pair.
DIM=8 and the capacity-matched pair remain unstarted and unapproved.

**2. The 4x per-PE spread is the more interesting finding, and it deserves
promotion from caveat to result.** 617 to 2,533 µm² across sixteen instances of
what the source says is one repeated unit means **the array is not homogeneous
after synthesis**. The mean is the least informative thing about that
distribution.

What would settle it, in order of value:

- **Is the spread positional?** Corner and edge PEs have operands that are
  constant, unconnected or immediately terminated, so constant propagation and
  dead-logic removal should hit them hardest. If the cheap PEs are the corners
  and the expensive ones the interior, that is a clean mechanism and a
  reportable one. If the spread is *not* positional, that is more surprising
  and worth more.
- **Does it scale?** The same measurement at T=8 says whether the effect is a
  fixed boundary cost — in which case it shrinks as a fraction with array size,
  which is a real argument about how systolic arrays amortise — or something
  else.

This matters beyond curiosity: **a per-PE mean is only a fair basis for
comparison against another design's per-PE figure if the underlying
distribution is tight.** At 4x spread ours is not, so the comparison against
Gemmini's 453 µm²/PE at 22 nm should be stated as a range, with the mean
labelled as a mean.

Keep the three caveats already committed — the node scaling is a rule of thumb,
the two PEs may not contain the same functions, and the mean hides the spread.
The second is the one most likely to be decisive and the hardest to check
without their netlist.

**On the PE-array hierarchy line not existing:** reporting that, rather than
producing a plausible number by another route and calling it the same thing,
was the right call. The flattening effort that dissolves the instances is the
same setting that makes these runs comparable to each other, so it should not
be changed to recover a hierarchy line; the name-prefix sum is the correct
substitute and is correctly labelled as a different kind of number.

**And the duplicate-detection point should be a standing practice**: three
near-duplicate runs caught tonight by checking disk before launching. Two
sessions with the same tools and the same repository will converge on the same
work unless one of them looks first.

## `TPU_TILED=32x512x128` does not complete at the shipped depth (2026-09-24)

A cosim of that shape ran for **29 hours with 28 seconds of CPU**, last
reporting `RTL Simulation : 0 / 1` at 5.1 ms of simulated time — simulation
time advancing, nothing retiring. Killed by PID, owner confirmed.

This is **another instance of limitations item 24**, at a much larger shape
than the ten-program family the item was filed for, and it has a consequence
for the merge train: the `big-shapes` branch is held pending a measurement of
`TPU_TILED=32x512x128`, and **that measurement was blocked by the very
deadlock `QD=16` was shown to clear**. Re-run it at `QD=16` before concluding
anything about that branch's mixed-sign `EXPECTED` row.

It also adds a data point the item's open predicate has to survive: the failing
set now spans a family of small tiled programs *and* a single very large one,
while the shipped GEMM issues 272 instructions to one unit at the same depth
and completes. Program size still does not predict membership.

**Operational note.** This was invisible from inside the session that started
it — the shell was blocked on the simulator, so nothing reported. It was found
by another session's fleet-wide process sweep. A run that can deadlock needs a
timeout at the launcher, not only a gate at the end; `ACT_COSIM_TIMEOUT` exists
and this launcher did not use it.
