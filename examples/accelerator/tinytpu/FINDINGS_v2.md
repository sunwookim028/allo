# Building a real systolic array + true overlap in Allo: what the compiler did

Everything below was found by measuring the emitted RTL or the scheduler's own
diagnostics, not by reading the source. Line references are to
`examples/accelerator/tinytpu/microarch_v2.py` on `chia-codesign`.

## A. Things that silently cost 4x, and the rule behind them

The array is a `t` loop containing DIM*DIM unrolled PEs. Its II is the whole
ballgame: II=4 means the 16 MACs are folded onto 4 and the array runs at a
quarter rate. Five separate things pushed it off II=1, and **all five are the
same rule**: an access the polyhedral dependence test cannot see as affine is
assumed to alias at distance 1.

| what | II | why |
|---|---|---|
| accumulator read fed into the top of the psum chain | 2 | genuine cycle: store -> row-0 load -> DIM-1 PE hops -> store. Latency exceeds distance at *every* pipeline depth, because each stage buys 1 cycle of distance and 1 of latency. Measured at PHOP=2/3/4 -> II=3/2/2. |
| accumulate as a read-modify-write inside the array loop | 4 | the two accesses cannot be separated; store(t) pairs with load(t+1) |
| bottom-edge store addressed `t - oj - c` (oj = unrolled column) | 4 | iteration t column oj and iteration t+1 column oj+1 name the same accumulator **row**. Different banks, but the test pairs them on the row index alone. |
| index held in an `i32` temporary (`arow: i32 = ...; accbuf[arow, oj]`) | 4 | not raised to affine even though the arithmetic is. Writing the subscript out in full at the access fixes it. |
| conditional load (`if zero_first == 0: prev = accbuf[...]`) | 4 | predicated, so the *access* becomes conditional and stops being affine. Load unconditionally, select the value. |

The fix that actually reached II=1 was structural, not a workaround: the array
writes a store-only output buffer, and a separate `ar` loop adds that into the
accumulator. A pass then costs `rows + SKEW + rows` instead of
`4 * (rows + SKEW)`. Real designs have that buffer at the array's output edge
for exactly this reason.

**Worth upstreaming as a diagnostic.** The scheduler already prints the binding
recurrence and its `total latency N over distance M`. It knows when a link in
that chain is a `memref` op that *would have been* affine but for a temporary or
a predicate. Saying so ("this access is not affine because ...; II is bounded by
a conservative distance of 1") would have turned a day into an hour.

## B. `Complete` partitioning is not banking

`partition(dim=2, kind=Complete)` on `f32[512, 4]` does not produce 4 memories
of 512 rows. It scatters every element into a flip-flop: 2048 registers behind
2048-way address decoders, a 98k-line netlist, and a combinational loop
Verilator rejects (`UNOPTFLAT` on the executor's issue logic).

The compiler states the rule itself when you try to combine the two:

    ERROR[ALLO-E0006]: Array memref<28x4xf32> is bound to storage 'bram' and
    also completely partitioned, which scatters it into registers; the two
    cannot both hold. Drop one of them

So a banked scratchpad has to be written as N separate arrays. That is what
`sp0..sp3` / `ac0..ac3` / `zb0..zb3` are. It works -- `ac*` and `zb*` come out as
real memories -- and it is honest hardware, but it means an N-bank design cannot
be parameterised in N without generating source.

Residual: the *top-level* `sp0..sp3`, which three concurrent processes share,
still emit as 512 flip-flops each. Same-shaped `ac*` inside one process stays a
memory. Shared-across-processes storage looks like the trigger.

## C. Compiler crashes (SIGSEGV in `[PREP]`)

Three distinct ones, all in the if-conversion pass, all on the unrolled PE body:

1. **Nested bank select.** `if j == 0: { if i == 0: av = sp0[..] elif ... }`
   inside the DIM*DIM unrolled body. Reproducible 3/3.
2. **Non-nested selects in the unrolled body.** After hoisting the bank select
   into `west[i]` feed registers, a plain `if j == 0 ... else` plus
   `if i > 0 ... ` still crashed. Reproducible.
3. **Intermittent.** With the conditionals gone the same module compiles, but
   roughly one run in three still dies in an LLVM thread pool
   (`llvm::StdThreadPool::processTasks`), same input, no source change. A retry
   loop gets through. This one smells like a data race in a parallel pass.

(1) and (2) both went away by giving the array's registers one extra edge each
-- `a_reg[i, 0]` is the west feed, `ps_dl[0, j, :]` the north zero -- so a PE's
inputs are plain indexed reads with no conditional at all. That is a better
description of the hardware anyway, but it should not be the difference between
compiling and a segfault.

## D. Two smaller ones

- **`ALLO-N0004`**: a completely-partitioned top-level argument cannot be passed
  to a sub-kernel ("it crosses the top boundary as one port per element"). So
  the instruction memory cannot be made IWIDTH-wide, and the sequencer pays
  IWIDTH cycles per instruction (II=5, measured). Bounded, because the sequencer
  runs concurrently with the units.
- **One call site per unit.** Calling a unit from two places in a dispatch chain
  emits two RTL copies and `compose` reaches only the first; the second comes
  out unscheduled -- for `mmu` that is a single time-multiplexed MAC instead of
  the array. Found by counting multipliers in the Verilog (`mul=15, add=12` in
  one copy and `mul=1, add=1` in the other). The mode has to travel as a value.

## E. What the compiler got right, and loudly

- The feedback-cycle check caught a real deadlock statically, with the fix in
  the message: `[ALLO-E0012] ... has no initial tokens and will deadlock; seed a
  channel on the cycle with an initializer`.
- `ALLO-E0006` above names both halves of the conflict and what to do.
- The `II cannot go below N here: one iteration takes N slots of a resource
  serving 1 per cycle. Banking or replicating what this access reaches is what
  lowers that bound` message is exactly the right shape -- it says which knob.

## F. Results (all cosim, Verilator, fp32, DIM=4)

`v2-serial` is the control: byte-identical compute units, generated from the
same source by `mkserial.py`, differing only in that the top is one
fetch-decode-dispatch loop instead of four concurrent processes. So the two
columns separate the array from the decoupling.

| shape | v1 | v2 serial | v2 NBUF=1 | v2 | array | decouple | dbl-buf | total | Gemmini | v2/Gem |
|---|---|---|---|---|---|---|---|---|---|---|
| 4x4x4 | 205 | 178 | 153 | **153** | 1.15x | 1.16x | 1.00x | 1.34x | 613 | **0.25x** |
| 8x8x8 | 1765 | 683 | 504 | **472** | 2.58x | 1.36x | 1.07x | 3.74x | 720 | **0.66x** |
| 12x12x12 | 6103 | 1564 | 1151 | **1035** | 3.90x | 1.36x | 1.11x | 5.90x | 893 | 1.16x |
| 16x16x8 | 7321 | 1703 | 1264 | **1168** | 4.30x | 1.35x | 1.08x | 6.27x | 941 | 1.24x |
| 16x16x16 | 14641 | 2869 | 2194 | **1936** | 5.10x | 1.31x | 1.13x | 7.56x | 1141 | 1.70x |

ReLU MLP layers (activation fused into the accumulator drain): 4x8x8 = 376,
8x8x8 = 472, 8x16x16 = 1410 cycles, all correct.

Back-pressure: at 16x16x16 with `stall_prob=0.3` (randomized input starvation
and output back-pressure on every port) the design returns **1936 cycles and the
same correct result** as at `stall_prob=0`. The credit protocol holds.

Instruction counts collapse too, because a weight tile is now latched once and a
whole M-row panel streams through it: 16x16x16 is **60 instructions against
v1's 656**.

### Reading the columns

- **array** is the systolic array plus its ISA (weight-stationary `loadw`/`mm`,
  in-array accumulation, a drain that fuses ReLU). It grows with the shape --
  1.15x at 4x4x4, 5.10x at 16x16x16 -- which is the signature of amortising the
  fill/drain skew over a longer stream. At a single 4x4 tile a systolic array is
  *slower* than an unrolled dot product; it only pays once M is large.
- **decouple** is a flat ~1.35x across every shape above the smallest, which is
  what a four-stage decoupled machine should look like: the win is bounded by
  the longest stage, not by the problem size.
- **dbl-buf** (NBUF 1 -> 2) is a further 1.07-1.13x, growing slowly. Deeper
  credits are the obvious next knob and it is one constant.

### Against Gemmini

Same datatype (fp32), same array dimension (4x4), same MAC latency. v2 is
**3.9x faster at 4x4x4** and **1.70x slower at 16x16x16**, against v1's 12.83x
slower. The remaining gap is entirely shape-scaling: over 64x more work Gemmini
grows 1.9x, v2 grows 12.7x (v1: 71.4x). Gemmini amortises ~600 cycles of fixed
RoCC/tiling overhead almost perfectly; v2 still pays per tile.

Array utilization at 16x16x16 is **13%** of the 16-MAC/cycle roofline, up from
v1's 1.7%. So the array is no longer the problem and the issue rate no longer
dominates -- what is left is the DMA. Each weight tile is a separate 16-word
transfer and each `mm` is only `2*rows + SKEW` cycles of work, so the loader is
now the critical stage. A strided/2D DMA descriptor, and streaming several
weight tiles per credit, are where the next factor is.

## F.2 Retested: can the accumulate walk be folded back in? No.

Section A blamed the split output buffer on a dependence test that could not
separate an accumulator read-modify-write. That diagnosis was made when the
accumulator was one 2-D array indexed by the *unrolled* column, so it was worth
retesting after the banking rewrite made `ac0..ac3` four separate 1-D arrays
with the subscript affine in `t`.

It still fails, identically: `II=4`, binding recurrence
`arith.addf -(2,d0)-> memref.store -(0,d1)-> memref.load -(2,d0)->`,
`total latency 4 over distance 1`. The MAC count also *drops* to 12, because
the binder folds units once the loop has that much slack -- so the array stops
being an array.

The ops are `memref.load`/`memref.store`, not `affine.*`, even though
`ac0[APAD + t - SKEW]` is syntactically affine in the loop variable. So this is
not the "distance 1 assumed because non-affine" case of section A: the access
never reaches affine form at all. A store and a load to the *same* address in
one iteration, with an FP add between them, is a recurrence of latency 4 over
distance 1 no matter what the dependence test can prove -- the value written at
`t` is read at `t` and the adder takes longer than a cycle.

The real conclusion is stronger than section A's: **a systolic array whose
accumulation is in-place cannot hold II=1 in this backend at all**, because the
accumulator RMW is a genuine single-iteration recurrence. Gemmini does not have
this problem because its accumulator is a real dual-ported RMW memory with the
add in the memory's own write path (`AccumulatorMem.scala`), not an add between
a load and a store in the datapath. The split buffer plus a second pass is
therefore the correct structure for this backend, and its cost (one extra
cycle per row, 256 of 1072 marginal cycles at 16x16x16) is the price of not
having an accumulating memory primitive. That is a missing *storage
realization*, not a scheduler weakness -- `bind_storage` offers RAM_1P /
RAM_2P / RAM_S2P / RAM_T2P and no accumulating variant.

## G. Where the residual gap to Gemmini actually is (it is not OoO)

The v1-vs-v2 table invites the reading that v2 is structurally worse and falls
further behind as the workload grows. Both halves of that are wrong, and the
measurements that show it are worth more than the ratio itself.

### G.1 The gap does not widen; Gemmini's fixed cost amortizes

Removing each machine's own fixed overhead (its 4x4x4 cost) leaves a **flat**
marginal ratio:

| shape | v2 cyc/MAC | Gemmini cyc/MAC | ratio |
|---|---|---|---|
| 8x8x8 | 0.712 | 0.239 | 2.98x |
| 12x12x12 | 0.530 | 0.168 | 3.15x |
| 16x16x8 | 0.512 | 0.165 | 3.09x |
| 16x16x16 | 0.442 | 0.131 | 3.38x |

A flat ratio is a *throughput* difference. The apparent widening in the raw
table is Gemmini amortizing ~600 cycles of RoCC/tiling overhead that v2 never
pays -- which is also why v2 wins outright at 4x4x4.

### G.2 An upper bound on what any scheduler could recover

Each stage was run alone on the real instruction stream with every dependency
bit cleared, so nothing blocks (`bench_v2.py cosim stages`; timing only, the
isolated streams read uninitialized memory and are not checked). The sum
reproduces the sequential control to 1%, which validates the method.

| | 8x8x8 | 16x16x16 |
|---|---|---|
| loader alone | 214 | 806 |
| **executor alone** | **318** | **1448** |
| storer alone | 84 | 292 |
| sequencer (II=5) | 90 | 300 |
| sum (zero overlap) | 706 | 2846 |
| *serial control, measured* | *683* | *2869* |
| **measured** | **472** | **1936** |
| max (perfect overlap) | 318 | 1448 |
| Gemmini | 720 | 1141 |

So at 16x16x16 the 1.70x splits into **1.34x scheduling** (the most any issue
policy could recover) and **1.27x throughput** (unreachable by any scheduler).
At 8x8x8 perfect overlap would be 0.44x, i.e. 2.3x faster than Gemmini.

### G.3 It is not run-ahead depth either

Sweeping the credits and queue depth: NBUF 2 -> 4 gains 2.4% (1936 -> 1890) and
NBUF 4 -> 8 gains nothing. The loader has slack (806 against the executor's
1448) but is nearly balanced *per weight tile*, so there is no independent work
to hoist -- which is also why out-of-order issue would buy little here.

### G.4 What it is: a per-instruction fixed cost

Streams of K identical instructions, dependency bits cleared, K varied. Exactly
linear at every point:

    mm  = 2.00 cyc/row + 35.0 fixed per instruction     (fit at rows = 4, 8, 16)
    loadw = 15.0 cyc/instruction, of which 4 is work    -> ~11 cyc dispatch

That decomposes the executor's 1448 completely (model 1420, 2% low):

| | cycles | % of executor |
|---|---|---|
| array actually streaming (`t` loop, rows+SKEW) | 352 | 25% |
| **accumulate walk (the second cyc/row)** | **256** | **18%** |
| `mm` pipeline fill/drain | 288 | 20% |
| instruction dispatch (36 x ~11) | 396 | 28% |
| `loadw` + `accst` work | 128 | 9% |

**Only a quarter of the executor is the array doing work.** The 35-cycle fixed
cost per `mm` was being amortized over `MAXROWS = 16` rows, and `MAXROWS` was 16
because the accumulator sits at a compile-time offset -- the thing that bought
the array II=1 in the first place.

### G.5 Confirming it: longer panels, same 60 instructions

Raising `MAXROWS` to 64 and streaming a longer panel, on both machines at
matched shapes:

| M (x16x16) | v2 | Gemmini | ratio | v2 cyc/MAC | Gem cyc/MAC |
|---|---|---|---|---|---|
| 16 | 1936 | 1163 | 1.66x | 0.473 | 0.284 |
| 32 | 3008 | 1710 | 1.76x | 0.367 | 0.209 |
| 64 | 5152 | 2835 | 1.82x | 0.314 | 0.173 |

v2's cyc/MAC improves 1.51x (0.473 -> 0.314, 13% -> 20% of roofline) with the
instruction count unchanged at 60. Both machines amortize, so the *ratio* barely
moves -- but the **marginal** ratio falls from ~3.4x to **1.92x**, and the
marginal cycles are now fully accounted for. Per +16 rows of M, measured delta
1072, model 1072 exactly:

| | cycles | attributable to |
|---|---|---|
| `mm` array streaming, 1 cyc/row | 256 | the design working correctly |
| `mm` accumulate walk, the 2nd cyc/row | 256 | **the compiler** (section A) |
| loader, A panel at 1 word/cycle | 256 | DMA bus width |
| storer, C tile at 1 word/cycle | 256 | DMA bus width |
| `accst` drain | 48 | |
| **total** | **1072** | vs Gemmini's 557 |

### G.6 The projection

Arithmetic on the measured coefficients above -- **not itself measured**:

| | marginal cyc | vs Gemmini |
|---|---|---|
| as built | 1072 | 1.92x |
| without the accumulate walk | 816 | 1.46x |
| ... and with Gemmini's 128-bit DMA bus | 432 | **0.78x** |

Half the remaining marginal gap is one compiler limitation (the dependence test
could not separate an accumulator read-modify-write, so accumulation had to
become a second pass over the data) and half is a bus width. Neither is a
property of generating the accelerator from Allo, and neither is issue policy.

### G.7 On measuring the OoO contribution directly

Restricting the comparison to "programs with no OoO opportunity" is both harder
and weaker than the stage isolation in G.2. Tiled GEMM is dense with hazards --
RAW on every weight load, WAR on every buffer reuse -- so hazard-free programs
barely exist; what differs is *who* resolves them, and Gemmini's ROB is the
mechanism that gives it any overlap at all, so removing the opportunity measures
"Gemmini fully serialized" rather than "Gemmini without OoO". G.2 bounds what an
*ideal* scheduler achieves, which upper-bounds OoO's value without constructing
anything or rebuilding Gemmini.

What static resolution does cost, and it is not cycles: a mis-scheduled program
returns a plausible wrong answer. Two such bugs appeared during this work -- a
dependency-token off-by-one (correct at 4x4x4 and 8x8x8, wrong from 12x12x12)
and a write-after-read on the output staging area across processes (correct at
Nt <= 2, wrong at Nt >= 3). Both were silent, and neither was visible on the CPU
backend, which runs the processes sequentially. Gemmini cannot produce either.
The fix is an assembler-side hazard checker -- the same RAW/WAR/WAW
address-overlap rules as `ReservationStation.scala:308-340`, run offline -- so
that unguarded hazards are *rejected at assembly time* rather than detected in
hardware.

## H. Fusing the weight latch into `mm`: the fix that follows from G.4

G.4 measured that consecutive instructions overlap by **exactly zero** cycles --
K back-to-back `mm`s cost `43.0*K + 4`, and 43 is one instruction's own latency.
The compiler states the mechanism:

    WARN: [PREP] Conditional left as an opaque scheduling unit because
                 'func.call' cannot be predicated; the enclosing loop cannot
                 pipeline across it
    INFO: [SCHED] Detected imperfect nest, decomposing into sub-regions
                  scheduled in program order.   (at loop 'c')

Every TinyTPU instruction is a `func.call` inside the dispatch loop, so the unit
starts empty, fills, streams, drains, and only then does the next instruction
begin. The array is II=1 -- cold-started once per instruction.

Gemmini does not have this: its units are persistent state machines.
`ExecuteController.scala` re-enters `compute` from three arrival sites without
passing through `flush`, and `MeshWithDelays` carries `tags_in_progress`, a
*queue* of in-flight matmul tags, so several matmuls occupy the mesh at once and
`pause` is per-cycle backpressure rather than a per-command drain. Fill/drain is
paid once per sequence there, once per command here.

The available fix without leaving the calling convention is to make each
instruction do more, so fewer calls are made. `loadw` and `mm` were adjacent by
construction -- every weight tile is latched then immediately streamed -- and
`mm`'s `a1` operand was unused, so they fuse into one instruction at no ISA
cost. `loadw` is deleted; `mmu` latches the tile itself.

| shape | before | after | gain | instrs |
|---|---|---|---|---|
| 4x4x4 | 153 | **137** | 1.12x | 6 -> 5 |
| 8x8x8 | 472 | **412** | 1.15x | 18 -> 14 |
| 12x12x12 | 1035 | **903** | 1.15x | 36 -> 27 |
| 16x16x8 | 1168 | **1052** | 1.11x | 32 -> 24 |
| 16x16x16 | 1936 | **1734** | 1.12x | 60 -> 44 |
| 32x16x16 | 3008 | **2806** | 1.07x | 60 -> 44 |
| 64x16x16 | 5152 | **4950** | 1.04x | 60 -> 44 |

ReLU MLP: 376/472/1410 -> **316/412/1198**. Array unchanged at 16 multipliers,
20 adders, `t` loop II=1. At `stall_prob=0.3`, 16x16x16 returns the same 1734
cycles and the same correct result.

Roofline utilization: 15% at 16x16x16, **21%** at 64x16x16 (v1 was 1.7%).
Against Gemmini: **1.49x** at 16x16x16 (from 1.70x), and 0.22x at 4x4x4 --
4.4x *faster* where fixed overhead dominates.

The gain shrinks as the panel lengthens (1.12x at M=16, 1.04x at M=64), which is
the signature of a per-instruction cost being amortized rather than removed: at
M=64 there are the same 44 instructions against 3.4x more streaming. Removing
the rest needs persistent units -- each unit its own `async` process draining a
command queue, so the array is never cold-started -- which is a restructure of
`executor` alone and is the next step, not a limitation of the ISA.
