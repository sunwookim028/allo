# Symptom: an accumulator loop will not reach II=1

You are working on an instruction-programmable TPU design written in Allo's
dataflow DSL and compiled to RTL through Vitis HLS. One unit is the bottleneck
and the compiler will not schedule it as tightly as the hardware allows.

## What was measured

The unit accumulates into an on-chip array. Reduced to its essentials, the
inner loop is

    ar[f1 + r] = ar[f1 + r] + v

with `r` the loop's induction variable and `f1` a value fixed for the loop.
Each iteration reads one element, adds to it, and writes it back.

Vitis schedules this loop at **Final II = 3** with `ar` in block RAM, where the
synthesised store-to-load distance is 1. Fully partitioning `ar` into registers
brings it to **II = 2**. Neither is II = 1, and the reported reason is a
loop-carried dependence through `ar`.

## Why the dependence is not real

The programs this design runs never read an accumulator row within two
iterations of writing it. The instruction stream is produced by an assembler
that enforces that spacing, so at run time the read in iteration *n* and the
write in iteration *n-1* touch different elements of `ar`. The scheduler cannot
see this: the subscript is a computed expression, and the assembler's guarantee
lives outside the compiler entirely.

So the author knows the two accesses never alias, and the compiler has no way
to be told.

## What was done about it instead

The recurrence was engineered away in hardware. The last two rows were held in
registers, writes to `ar` were deferred by two iterations, and reads landing in
that window were answered from a bypass multiplexer. This takes the memory off
the carried path and does reach **II = 1**.

It was built, it was bit-exact at all five benchmark shapes, and it was
**reverted**. The cost was **13.7x the flip-flops in that unit** — 1,270 rising
to 17,450 — for a **2.3 % end-to-end** gain. Worse, `ar` scales with the array
dimension, so the flip-flop cost grows as the design is parametrised upward
while the 2.3 % does not.

For scale: in this synthesis flow, two builds of an identical netlist differed
by 1,407 LUT and 0.046 ns, so nothing below roughly 1.5k LUT or 50 ps is
evidence of anything. The 2.3 % is barely outside the noise; the area cost is
far outside it.

## The measured size of the prize

The scheduling difference alone — II = 2 against II = 1 in this unit, with no
other change — is worth **35 cycles at 16x16x16** on the design as it stood,
and **95 cycles** once other pending design fixes are in. Against a total of
several hundred cycles at that shape, and against a competitor the design is
currently 1.07x to 1.24x behind, that is a material fraction of the gap.

## Your task

Find the right way for this design to reach II = 1 in that unit **without**
paying the area price, and implement it.

The judgement you are being asked to exercise is about *where* the solution
belongs, not only whether it works. Consider what a second design with the same
problem would have to do, and what a user of this compiler should have to write
to express what the author knows and the scheduler cannot infer.

Constraints that any solution must respect:

- Results must stay bit-exact against the design's reference model. The stress
  and benchmark harnesses check this and are not yours to modify.
- Cycles are measured by RTL co-simulation, not by a cycle model and not by
  C simulation.
- Report resources and the estimated clock period alongside cycles. A cycle win
  at a longer clock is not a win.
- Whatever you build must still work when the design's array dimension
  parameter changes. A solution valid only at the shipped size is not a
  solution.

State plainly what your solution assumes to be true, and how someone would find
out if that assumption were ever violated.
