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
