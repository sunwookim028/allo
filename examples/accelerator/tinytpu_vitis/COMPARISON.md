# TinyTPU-isa vs Gemmini: a matched comparison

Both sides measured, both int8/int32 on a 4x4 array, same shapes, same operand
distribution. This is the comparison the headline claim rests on.

## Making the baseline matched

Neither stock Gemmini config is a fair reference:

| config | mesh | dtype | matched? |
|---|---|---|---|
| `GemminiRocketConfig` | 16x16 | int8/int32 | no -- 16x the array |
| `FPGemminiRocketConfig` | 4x4 | fp32 | no -- FPU, and fp32 on an FPGA is soft-float |
| **`Int8Dim4GemminiRocketConfig`** (added) | **4x4** | **int8/int32** | **yes** |

`gemmini.GemminiCustomConfigs.int8Dim4Config` is `defaultConfig` -- Gemmini's own
`inputType = SInt(8.W)`, `accType = SInt(32.W)` -- with `meshRows/Columns = 4`.
Elaboration confirms it: `GEMMINI DIM=4 elem_t_bytes=1`. `allo_cmp.c` needed no
source change, being written against `elem_t` and filled with values in [-4, 4],
which is also what `bench_isa.py`/`cosim.py` feed our design.

Both numbers are **measured on RTL**: ours by Vitis `cosim` (xsim), Gemmini's by
`rdcycle` under Verilator. Ours is the accelerator alone; Gemmini's includes
RoCC dispatch from Rocket and its own tiling loop, which favours us slightly and
is noted rather than corrected for.

## One hardware build, workload swept as data

**This is the correction that matters most.** Earlier revisions of this file
compared per-workload builds against Gemmini's single elaboration: `M`, `K`,
`N`, the instruction count and every unit's loop bound were compile-time
constants, so 4x4x4 and 16x16x16 were *different accelerators*. That is not a
comparison an instruction-programmable claim can rest on.

The design is now built **once** -- `T=4`, `MAXDIM=16`, fixed scratchpad, vregs
and imem -- and the shape arrives as data:

* the imem header carries the instruction count and a per-unit count, so every
  loop bound in the design is runtime data;
* `A`, `B`, `C` are flat `int8[MAXDIM*MAXDIM]` at the fixed `MAXDIM` stride,
  exactly as `allo_cmp.c` does for Gemmini (`MAXDIM, MAXDIM, 0, MAXDIM`);
* `gemm_program(M, K, N)` assembles the stream; the RTL never changes.

Verified: **11 programs on one build, all bit-exact** -- `gemm` and `gemm.relu`
at five shapes plus a vector-unit program. Two of those shapes (12x12x12,
16x16x8) could not be run at all before, because each would have needed its own
accelerator.

## The numbers (one build, cosim, one csynth)

| shape | ours | Gemmini int8 4x4 | ratio | our util | Gemmini util |
|---|---|---|---|---|---|
| 4x4x4    | **676**  | 574 | **1.18x** |  0.6% |  0.7% |
| 8x8x8    | **827**  | 615 | **1.34x** |  3.9% |  5.2% |
| 12x12x12 | **1062** | 740 | **1.44x** | 10.2% | 14.6% |
| 16x16x8  | **1125** | 784 | **1.43x** | 11.4% | 16.3% |
| 16x16x16 | **1423** | 986 | **1.44x** | 18.0% | 26.0% |

(Our column is after the burst-DMA pass and the flat accumulator -- see the I/O
section below and `RESULTS_ISA.md`. It read 680 / 831 / 1066 / 1139 / 1457
before the accumulator was flattened, and 1004 / 1108 / 1294 / 1344 / 1586 at
1.75x / 1.80x / 1.75x / 1.71x / 1.61x before the burst DMA, which is what
earlier revisions of this table quoted.)

**The shape of the gap has inverted, and that is the result.** It used to be
flat at ~1.7x and *falling* with size -- the signature of a fixed charge being
amortised. With the argument-copying gone the ratio *rises*, 1.18x -> 1.48x:
at 4x4x4 we are now within 18% of Gemmini, and what is left is a per-work gap
rather than a per-run one. That is a smaller total gap and a harder one, and it
is the honest reading: the easy 350 cycles have been taken.

## Marginal cost across the sweep — three different kinds of machine

Fitting only two points hides the shape. Successive marginal efficiency across
all five shapes (MAC/cycle between adjacent shapes, as a fraction of each
machine's own peak):

| step | Gemmini | ours |
|---|---|---|
| 4x4x4 -> 8x8x8 | 10.93 MAC/cyc (**68.3%**) | 2.97 (**18.5%**) |
| 8x8x8 -> 12x12x12 | 9.73 (**60.8%**) | 5.17 (**32.3%**) |
| 12x12x12 -> 16x16x8 | 7.27 (45.5%) | 5.08 (31.7%) |
| 16x16x8 -> 16x16x16 | 10.14 (**63.4%**) | 6.87 (**43.0%**) |
| least squares, all five | 9.71 (**60.7%**), fixed 566 | 5.54 (**34.6%**), fixed 718 |

(Our column is post-burst-DMA and post-flat-accumulator. It read 2.97 / 5.17 /
4.38 / 6.44 and 33.2% with fixed 717 before the accumulator was flattened;
4.31 / 6.54 / 6.40 / 8.46 and 44.1% with fixed 1028 after row-flattening and
before the burst DMA; and 3.50 / 5.43 / 6.67 / 6.92 and 37.0% with fixed 1067
before that. **The burst DMA took the marginal efficiency DOWN, 44.1% ->
33.2%, and the fixed cost down further, 1028 -> 717**, which is the trade
stated in its own terms and which at every shape in this table the second term
wins. The flat accumulator is the first change since that moved the marginal
term back up, 33.2% -> 34.6%, with the fixed term unchanged.)

The cube sweep varies all three dimensions at once and spans only 2.4x in
cycles, which makes each marginal a difference of two similar numbers. The
**tall sweep is the better-conditioned measurement** -- `allo_cmp.c`'s
`gemm_tall` (TALLM=64) holds K=N=16 and sweeps M, so exactly one dimension
moves, and it ran in the same int8 DIM=4 pass:

| M | cycles | d cycles | d MACs | marginal |
|---|---|---|---|---|
| 16 |   986 |  -- |    -- | -- |
| 32 | 1,393 | 407 | 4,096 | 10.06 MAC/cyc (**62.9%**) |
| 64 | 2,219 | 826 | 8,192 |  9.92 MAC/cyc (**62.0%**) |

Closed form: **`cycles = 573 + 25.71 * M`**, maximum error **0.18%** across the
three points. Marginal = 256 MAC/row / 25.71 cyc = **62.2% of peak**.

Two independent sweeps agreeing is the point: the 5-point cube fit gives 60.7%
with fixed 566, the clean 1-D sweep gives 62.2% with fixed 573. **Quote 62.2%**
-- it varies one dimension rather than three.

The residual is structural rather than noise (this simulator is deterministic):
it is `tiled_matmul_auto` re-deciding its tile split as M grows, plus RoCC
dispatch that does not divide evenly. Worth contrasting with the MiniTPU target,
whose marginal is *exact* -- `cycles = 432 + 2023 * column_tiles`, zero deviation
over a 39.7x range -- because nothing in it arbitrates or backpressures, so there
is no mechanism by which two runs could differ. **A machine whose compiler
carries the whole hazard burden is a machine whose performance is a closed form.**
Gemmini has a reservation station and a tiling heuristic and both leave a trace.

The 12x12x12 -> 16x16x8 step dips for both machines because it changes aspect
ratio rather than growing uniformly; it is not a clean sweep point.

**Gemmini's marginal efficiency is roughly flat at ~61%. Ours still rises,
18.5% -> 40.3%, but far less steeply than it did.** The rise is the signature
of a fixed cost being amortised, and shrinking the fixed cost is exactly what
flattens the curve: the intercept fell 1028 -> 717 when the argument copying
was replaced by program-controlled bursts (see the I/O section below), and the
curve came down with it. What is left is closer to a constant factor on the
work, which is the harder kind of gap and the honest description of where the
design now stands. Flattening the per-instruction loops had lifted the whole
curve (21.9% -> 43.2% became 26.9% -> 52.9%) and left the intercept where it
was; the burst DMA did the opposite, and the burst DMA was worth more.

For contrast, the MiniTPU target's marginal efficiency is **flat at 19.0%** and
does not move with size. Measured on its own RTL over a 12x sweep of output
column tiles (`[32,192]@[192,16 / 64 / 192]`: 2,455 / 8,524 / 24,708 cycles for
98,304 / 393,216 / 1,179,648 useful MACs), it is 48.6 MAC/cycle = 19.0% at both
steps, to three digits.

Two distinctions that matter here, and an earlier revision of this file got the
first one wrong:

- **19.0% is the measured marginal; 36.4% is a bound.** One `mxu_matrix_ctrl`
  FSM cannot hold a `vmatpush` and a `vmatpop` at once, which permits 4 output
  rows per 11 cycles = 93.1 MAC/cycle = 36.4% of its 256 peak. The emitter
  delivers 48.6 of those 93.1 -- **52% of the bound** -- so roughly half the
  distance to peak is the FSM and the other half is elsewhere in the schedule
  and is, as of this writing, unattributed. Quoting 36.4% as the machine's
  marginal conflates a bound with a measurement.
- **Flat versus rising is the real difference, not the number.** Our 37.0% and
  its 36.4% looked like the same quantity and are not: ours rises 26.9% ->
  52.9% and would keep climbing, theirs sits at 19.0% and does not move. With
  the corrected figure the two no longer even look alike, which is the more
  honest presentation.

A ceiling and an overhead are different objects. Ours is an overhead: a fixed
charge the sweep is paying off. Theirs is a ceiling: no amount of amortisation
reaches past it without changing the emitter.

**A correction worth recording:** an earlier analysis put Gemmini's marginal
efficiency at "essentially 100% of peak" from the two-point difference
`986 - 574 = 412` cycles. That is wrong: 412 cycles for the 4032 additional
MACs is 9.79 MAC/cycle, i.e. **61.2%**. The overhead-vs-ceiling distinction
survives the correction, but at 1.7x rather than 2.7x.

## T=16: scaling the array 16x bought 1.47x, and that settles the priority

The 16x16 build became possible only after the simulator's OpenMP team was
sized to the section count (`notes/ALLO_SHORTCOMINGS.md` #11); before that fix
a 262-instance region hung silently. It builds and runs in **50 s**, 821
streams, all three programs bit-exact, and cosim measures:

| | cycles @16x16x16 | PEs | roofline | utilization |
|---|---|---|---|---|
| ours T=4 | 1733 | 16 | 256 cyc | 14.8% |
| **ours T=16** | **1176** | **256** | **16 cyc** | **1.36%** |
| MiniTPU 16x16 | 168 | 256 | 16 cyc | 9.5% |
| Gemmini 4x4 | 986 | 16 | 256 cyc | 26.0% |

(Both "ours" rows predate the row-flattening pass below, which took the T=4
number to 1586. T=16 has not been re-measured since; the argument this section
makes is about the fixed term, which flattening did not move.)

**16x the PEs bought 1.47x the speed, and utilization fell 14.8% -> 1.36%.**

That is the cleanest evidence in this whole comparison for where the problem
is, and it is worth more than the ratio it produces:

- Array time fell from 256 cycles to 16. **Overhead went 1477 -> 1160, i.e.
  barely moved.** The overhead is the same absolute quantity at both array
  sizes, because it is data movement over a 16x16 operand set, which does not
  depend on T.
- At T=16 the program is **6 instructions**, not 45, so the marginal term --
  18.1 vs Gemmini's 10.8 cycles/instruction -- is almost entirely absent from
  this measurement. 1176 cycles for 6 instructions is not an issue-rate
  problem.
- Therefore **the fixed term is not one of two roughly equal problems; at a
  useful array size it is essentially the only problem.** An earlier revision
  of this file listed "pipeline the per-instruction loop" and "program-
  controlled burst DMA" as comparable next steps. They are not. At T=4 the
  marginal term is visible because 45 instructions multiply it; at T=16 it
  nearly vanishes and 98.6% of the machine sits idle waiting for operands.

Resources at T=16: 240 DSP, 111,776 FF, 144,329 LUT (11% of the xcu280), 34
BRAM. All 256 PEs instantiated.

A methodological note that repeats a lesson from earlier in this file: csynth
reports a top-level latency of **2.259e+08** for this build, because the loop
trip counts are runtime data and it must bound them by the ISA's field widths.
The real number is cosim's 1176. A bound is not a measurement.

## The I/O trade, measured -- and then removed

`wrap_io` was for a long time an architectural choice with a crossover, and
neither setting was what Gemmini has:

| config | marginal | fixed | 4x4x4 | 16x16x16 |
|---|---|---|---|---|
| `wrap_io=True`, imem 256 | 18.1 cyc/instr | 1102 | 2.12x | 1.94x |
| `wrap_io=False`, strided | 39.8 | 481 | 1.21x | 2.25x |
| `wrap_io=True`, imem 56 | 15.1 | 907 | 1.75x | 1.61x |
| `wrap_io=False`, bursts | 20.1 | 557 | 1.18x | 1.48x |
| **+ flat accumulator** | **19.3** | **563** | **1.18x** | **1.44x** |
| Gemmini | **10.8** | **483** | 1.00x | 1.00x |

* `wrap_io=True` copies each argument into a local buffer before the region
  runs, so the units read BRAM -- cheap per access, but the copy is the
  *declared* length regardless of the shape being run. At MAXDIM=16 that is
  imem 56 + A 256 + B 256 + C 256 = **824 words**, and cosim charges it as a
  fixed 907 cycles: 57% of 16x16x16 and 90% of 4x4x4.
* `wrap_io=False` lets the units read `m_axi` directly, and the row above it is
  the measurement that made this look like a dead end: fixed 481, essentially
  Gemmini's 483, but the marginal cost more than doubled.
* **That row measured an access pattern, not a configuration.** Vitis turns
  `lA[(f1 + r) * MAXDIM + f2 * T + e]` into `[HLS 214-115] Multiple burst reads
  of length 4 and bit width 8` -- a four-beat AXI transaction per row -- and it
  turns `imem[NHDR + pc * IWORDS]` into a **two-word** burst per instruction,
  which alone takes the sequencer's fetch loop from II=5 to II=13. A contiguous
  sweep with a runtime trip count gets `... of variable length`, one real burst,
  at the port's own limit.
* The bottom row is the design built around that: the sequencer pulls the whole
  program on-chip in one burst, and `dma_ld` opens with one variable-length
  burst per operand matrix covering exactly the DRAM rows the program names.
  **Fixed cost 907 -> 557 and faster at all five shapes**, 1.48x at the
  smallest and 1.09x at the largest.

The crossover between the last two rows is at **71 dynamic instructions**,
which is past the longest program this MAXDIM admits (45) -- so the burst build
wins everywhere it can be run, and would stop winning at a larger MAXDIM
without also fixing what the marginal term buys. What that +5 cycles/instruction
is: `dma_st` still writes `C` with the strided pattern
(`[HLS 214-115] Multiple burst writes of length 4 and bit width 8`, II=4). A
contiguous write-back would have to either clobber the columns the program never
named or be deferred to the end of the run, where it would serialize behind the
last `mvout` instead of overlapping the compute it currently overlaps. It is the
next thing to measure.

The `+ flat accumulator` row is the first change to move the marginal term
since the burst DMA, and it moved it by 0.75 cycles/instruction: `accu`'s
per-instruction loop became one flat row loop at II=1, which needed a
write-behind rotation to break the `ar` read-modify-write recurrence
(`RESULTS_ISA.md`). The array's per-`mm` loop was flattened the same way in the
same pass, also reached II=1, and measured about +10 cycles at four of five
shapes, so it was not landed -- the two together say that **array throughput is
worth ~0.8 cycles of runtime per cycle of MAC while array per-instruction
overhead is worth nothing**, because `vru` upstream pushes the same `T + 1`
prologue words per `mm` whatever the PE does.

**Gemmini still has the better version of this**, and the residual marginal gap
is where it now shows: `mvin`/`mvout` transfer exactly the tiles the program
names in both directions, at a bus width that is not 8 bits.
`config_interface -m_axi_max_widen_bitwidth 512` would give us the second half
of that and does nothing today -- `[HLS 214-307] Could not widen since type i8
size is greater than or equal to alignment 1(bytes)`, because Allo emits the
argument pointers with no alignment attribute. That is an Allo codegen gap
rather than a design one, and it is worth roughly the whole operand-traffic
term if it were closed.

## Where the gap was, originally

Fitting cycles against instruction count (6, 20, 76 instructions):

| | fixed cost | marginal cost |
|---|---|---|
| ours | 573 cycles | **24.0 cycles/instruction** |
| Gemmini | 539 cycles | **5.9 cycles/instruction** |

**The fixed costs are the same (573 vs 539). The entire gap is marginal cost,
and it is 4x.** That is a specific, mechanical difference, not a diffuse one --
and it explains the shape of the table: the ratio grows with size precisely
because the gap is per-instruction.

The mechanism is in the synthesis report. Every unit's per-instruction loop:

```
o l_S_c_0_c    iter_latency=2   II=1   trip=80   pipelined=yes    <- sequencer
o l_S_c_0_c1   iter_latency=69  II=-   trip=80   pipelined=no     <- dma_ld
o l_S_c_0_c2   iter_latency=69  II=-   trip=80   pipelined=no     <- spm
o l_S_c_0_c3   iter_latency=73  II=-   trip=80   pipelined=no     <- vru
o l_S_c_0_c4+  iter_latency=74  II=-   trip=80   pipelined=no     <- the PEs
```

Only the sequencer's loop is pipelined. In every other unit the loop over
instructions is **not** pipelined, so a unit finishes instruction *n* before
starting *n+1*: **there is no inter-instruction overlap inside a unit.** The
units overlap with *each other* (that is what `dataflow` buys, and it is why the
measured 24 is far below the report's ~70), but within a unit instructions are
strictly serial.

This is the same root cause found on `chia-codesign`, where a unit was a
`func.call` the compiler would not pipeline across and consecutive instructions
overlapped by exactly zero (`FINDINGS_v2.md` G.4). The structure is much better
here -- persistent processes, so the cost is 24 cycles rather than a full
pipeline fill and drain -- but the property is the same, and it is now
quantified against a matched baseline.

**Gemmini's answer is its reservation station.** `ReservationStation.scala` (48
entries: 8 ld / 16 ex / 4 st) exists precisely to have many instructions in
flight at once, issuing out of order when their operands do not overlap. Its
5.9 cycles/instruction is that hardware working. We are strictly in-order, by
construction: each unit consumes the instruction stream in order and every
channel is point-to-point, which is what makes the hazard logic free, and also
what caps the throughput.

**Since fixed, in part.** Four of the five units now run one flat loop over
rows that pipelines at II=1; `accu` and the array do not, for reasons the
row-flattening section above sets out. Marginal cost went 18.07 -> 15.12
cycles/instruction, worth 1.08x at 16x16x16.

## Row-flattening the per-instruction loop: 1.08x, not the 1.7x it looked like

This was item 1 on the list below, and the estimate attached to it -- "worth
the 1.7x marginal term" -- was wrong in a way worth recording, because the
arithmetic that produced it is the kind that is easy to repeat.

**The change.** Every unit was a loop over *instructions* containing a loop
over *rows*, and Vitis reported `Pipelined = no` on all five outer loops. Four
of them are now a single flat loop over rows (or words), with the instruction
fetched on the iteration that needs it and the opcode test surviving as a mux
inside a pipelined body. The header carries a per-unit dynamic *work* count
instead of a dynamic instruction count; `assemble()` already expanded the
control flow to compute the latter, so this is `nr` summed rather than counted.

**Measured, same flow, same testbenches, all five shapes bit-exact:**

| shape | before | after | speedup |
|---|---|---|---|
| 4x4x4    | 1017 | **1004** | 1.013x |
| 8x8x8    | 1145 | **1108** | 1.033x |
| 12x12x12 | 1369 | **1294** | 1.058x |
| 16x16x8  | 1417 | **1344** | 1.054x |
| 16x16x16 | 1713 | **1586** | **1.080x** |

Fitting against dynamic instruction count (6, 15, 28, 25, 45):

| | marginal | fixed |
|---|---|---|
| before | 18.07 cyc/instr | 902 |
| after | **15.12 cyc/instr** | 907 |
| Gemmini | 10.8 | 483 |

**The marginal term moved 1.20x, not 1.7x, and the fixed term did not move at
all** (5 cycles is fit noise). Two things account for the gap between the
estimate and the result:

1. **1.7x was the ratio of our marginal to Gemmini's, not the headroom in the
   change.** 18.1/10.8 = 1.68 is what closing the whole marginal gap would buy;
   pipelining the loop only removes the part of the marginal cost that is
   per-instruction pipeline fill, and the rest is real per-row work that no
   scheduling change touches.
2. **Amdahl.** At 16x16x16 the marginal term is 811 of 1713 cycles, 47%; at
   4x4x4 it is 108 of 1017, 11%. Even a marginal term driven to zero could not
   have given 1.7x overall at any shape in this sweep, and the measured
   speedups track that share exactly -- 1.3% at the smallest shape rising
   monotonically to 8.0% at the largest.

**Per-unit, before and after** (csynth, the loop over instructions):

| unit | before | after |
|---|---|---|
| `sequencer` | 1 loop, II=5 | unchanged |
| `dma_ld` | iter latency 133, **Pipelined = no** | one loop, **yes, II=1**, iter latency 4 |
| `spm` | iter latency 133, **no** | one loop, **yes, II=1**, iter latency 5 |
| `vru` | iter latency 137, **no** | one loop, **yes, II=1**, iter latency 5 |
| `accu` | iter latency 261, **no** | **unchanged -- see below** |
| `pe` x16 | iter latency 2055-2058, **no** | unchanged |
| `dma_st` | iter latency 132, **no** | one loop, **yes, II=1**, iter latency 3 |

Area is essentially unchanged: BRAM 16, DSP 15, FF 12432 -> 12835, LUT
18416 -> 18968.

### What would not split, and what would not flatten

The plan was per-opcode queues and one process per opcode. **One owner per
memory rules that out for exactly the units that matter.** `spad` is written by
`dma_ld` and read by `vld`; `vr` is written by `vld` and read by `mm`; `ar` is
touched by all four of `accu`'s opcodes. Allo enforces single reader and single
writer and Vitis rejects the violation outright (HLS 200-779 / 200-979). The
two units that own no memory, `dma_ld` and `dma_st`, serve one opcode each, so
there was nothing to split there either -- the splitting half of the plan had
no legal instance anywhere in the design. Flattening turned out to get what the
split was wanted for without moving a memory, which is why it is the change
that shipped.

**`accu` resisted the flattening too, and that is the honest limit of this
pass.** It was built twice and was bit-exact both times, and both times it was
slower than the nested loop it replaced:

* With `ar` in BRAM, `Final II = 3`. Once `r` is a carried register rather than
  the inner loop's induction variable, `ar[f1+r]` stops being affine in the
  loop index, and Vitis can no longer prove that iteration *n*'s store and
  iteration *n+1*'s load touch different rows. Deferring the write by one
  iteration, the textbook fix, only moves the violation onto the enable
  register.
* With `ar` completely partitioned into registers -- no ports to arbitrate, no
  aliasing to prove -- `Final II = 2`, and no further. Read, add, write, and the
  next iteration may read what this one wrote is a genuine recurrence through a
  register file. It also cost 33k FF, 12k LUT and ten minutes of synthesis.

Two cycles a row for 384 rows is worse than 20 instruction boundaries plus 384
rows at one, so `accu` keeps the nested loop and keeps `mm` at II=1. The array
was left alone deliberately: a PE's per-`mm` prologue is a different shape from
its MAC body, and folding them would set the II of the one loop in the design
that is already II=1 and carries the actual arithmetic.

**The general rule, which is the transferable result here:** flattening trades
a fixed per-instruction cost for a permanent per-row II, and only pays where
the II stays at 1. Four units kept it; one did not, and did not.

Two Vitis-specific things were worth a factor of two each and are not obvious
from the source:

* **increment the row counter at the TOP of the body.** With `r += 1` at the
  bottom, Vitis schedules the add in the last stage and the next iteration's
  conditional queue read depends on it -- a distance-1 recurrence that does not
  close in one cycle, `Final II = 2` on every unit. Hoisting it costs nothing.
* **read the memory once, at an address the branch selects.** A read written
  into each arm -- the obvious transcription -- synthesizes to one read port per
  arm on one memory: `vru` came back at II=2 and `accu` at II=3.

## What would close the rest

1. ~~**Pipeline the per-instruction loop**~~ -- done, and measured at 1.08x
   rather than the 1.7x estimated; see the section above. What remains of the
   marginal term (15.1 against Gemmini's 10.8) is in `accu` and the array,
   neither of which flattens for the reasons given there.
2. ~~**A program-controlled burst DMA**~~ -- done, and measured at 1.48x at
   4x4x4 falling to 1.09x at 16x16x16 (fixed 907 -> 557). Allo did not need
   extending after all: the mechanism was already there and the previous
   measurement had condemned the access pattern rather than the configuration.
   See the I/O section above.
3. **The write path.** `dma_st` still writes `C` four bytes at a time, which is
   most of what separates the new 20.1 cyc/instr from the 15.1 the buffered
   build got, and the reason the crossover sits at 71 instructions rather than
   further out. Both ways of bursting it (clobber the unnamed columns, or defer
   the write-back to the end) cost something real, so this one has to be
   measured rather than assumed.
4. **A wider bus.** The `m_axi` ports are 8 bits, so a perfectly bursted
   operand set still costs one cycle per byte. `m_axi_max_widen_bitwidth` is
   blocked on Allo emitting an alignment attribute on the argument pointers.
5. **Then multiple instructions in flight.** A credit/scoreboard scheme over
   the in-order units would approach Gemmini's 48-entry reservation station
   without its complexity.

Item 2 of the original list -- splitting `accu` -- was addressed differently and
better: rather than giving the vector ALU its own issue slot, the GEMM inner
loop no longer uses it at all.

## A bug that only RTL caught

Consolidating the `vld` broke 16x16x16 in cosim -- **251 of 256 outputs wrong**
-- while the Allo dataflow simulator passed it. The cause is in the emitted HLS:
a bit-slice is extracted into a **signed** `ap_int<N>`.

```cpp
ap_int<7> v268;  v268 = w02(60, 54);   // the nr field
int32_t nr = v268;                     // 64 -> 0b1000000 -> -64
```

So `nr = 64` for a 64-row `vld` read back as -64, the loop ran zero times, and
the design silently produced zeros. It appeared exactly at the field's sign-bit
boundary: `nr <= 63` worked, 64 did not. `nr` is now 8 bits, and `enc()` asserts
`0 <= v < 2^(w-1)` for every field.

The lesson for the methodology: **an N-bit ISA field safely carries
0 .. 2^(N-1) - 1**, and the dataflow simulator does not model this, so it is a
genuine simulator/RTL divergence. It is the argument for keeping cosim in the
loop rather than running it once at the end -- this bug was invisible to every
functional check that had been passing all session.

## Honest reading

On one fixed build, matched in data type and array size, the design is **1.18x
behind Gemmini at 4x4x4 and 1.48x at 16x16x16** across five shapes. The ratio
now *rises* with problem size, where it used to fall: the fixed charge that
dominated the small shapes has been removed, and what is left is a per-work
gap. Utilization tracks at roughly two thirds of Gemmini's at the largest shape.

Two terms make up the gap, and both are named:

1. **Marginal, 20.1 vs 10.8 cycles/instruction (1.9x), and it went UP.** It was
   18.1 until four of the five units were row-flattened so their loops pipeline
   at II=1 (worth 1.08x), then 15.1, and the burst DMA traded 5 of those cycles
   back for 350 off the fixed term. Three things hold it there: `dma_st`'s
   four-byte writes to `C`, `accu`'s read-modify-write accumulate, which does
   not flatten below II=2, and the array's per-`mm` prologue.
2. **Fixed, 557 vs 483 cycles (1.15x).** This was 907 vs 483 and was the whole
   story of this comparison for most of the project. What remains is the
   region's own start-up and drain plus the 56-word instruction prefetch, and
   it is close enough to Gemmini's that it is no longer where the work is.

**The priority has genuinely inverted.** For most of this project the argument
was that the fixed term was the only problem -- 16x the PEs bought 1.47x, and
at T=16 the marginal term was almost absent from the measurement. That argument
was correct and it has now been acted on; the fixed term is 1.15x and the
marginal term is 1.9x, so the next pass belongs on the marginal side, starting
with the write path.

Neither term is about the array, the data type, or the dataflow -- all three are
correct and RTL-verified, and `microarch_ws.py` reached exactly 100% of roofline
with the same PE structure. The compute is not the limit.

Both utilizations are low in absolute terms because these are tiny problems on
a 4x4 array where fixed costs dominate; Gemmini's 26% at 16x16x16 is the number
to chase, and at 17.6% we are now two thirds of the way there.

## Reproducing

```bash
# ours -- one build, all shapes
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build
export PYTHONPATH=/home/sk3463/allo OMP_NUM_THREADS=32   # >= 22 processes
python bench_isa.py          # functional sweep, one build
python cosim.py              # one csynth, cosim per shape
TPU_WRAP=1 python cosim.py   # the old hoisted-argument variant, for comparison

# Gemmini, matched
cd ~/chipyard && source env.sh
cd sims/verilator && make CONFIG=Int8Dim4GemminiRocketConfig -j16
./simulator-chipyard.harness-Int8Dim4GemminiRocketConfig +permissive +permissive-off \
  ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/allo_cmp-baremetal
```

Raw Gemmini output in `logs/gemmini_int8_dim4.log`.
