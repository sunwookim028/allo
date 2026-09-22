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
