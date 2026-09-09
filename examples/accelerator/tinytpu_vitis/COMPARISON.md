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

## The numbers (after the fixes; see "What was fixed" below)

| shape | before | ours now | Gemmini int8 4x4 | ratio was | ratio now | instrs |
|---|---|---|---|---|---|---|
| 4x4x4    |  717 |  **760** | 574 | 1.25x | 1.32x |  6 -> 6  |
| 8x8x8    | 1003 |  **945** | 615 | 1.63x | 1.54x | 20 -> 15 |
| 16x16x16 | 2395 | **1721** | 986 | 2.43x | **1.75x** | 76 -> 45 |

**1.39x faster at 16x16x16**, and utilization there went 10.7% -> 14.9%
(Gemmini 26.0%). 4x4x4 got 6% *slower*: with only 6 instructions there is
nothing to save, and the accumulate-select added a multiplexer to the
accumulator's write path. That is the right trade -- it is the one shape where
fixed cost is everything -- but it is a regression and is not hidden here.

Functionally exact at every shape (`mismatches = 0`) for `gemm`, `gemm.relu`,
**and** a vector-unit program, verified in RTL by cosim rather than only in the
dataflow simulator.

## What was fixed, and what was not

Three changes, in the order they were measured:

| change | 16x16x16 | what it did |
|---|---|---|
| selective dispatch | 2395 -> 2163 | the sequencer sends each instruction only to the units that act on it, so a unit's loop counts *its* instructions. The PEs saw 80 and needed 16; `dma_st` saw 80 and needed 4; 76% of all decode work in the design was for instructions the unit would ignore. |
| accumulate in `mm` | 2163 -> 1801 | `mm` takes an accumulate bit, so a Kt-deep contraction is Kt `mm`s and **no `vadd`**. `accu` had been doing 576 row-ops for 256 wavefronts of work -- 2.25x oversubscribed -- because psum collection *and* `vadd` both ran there, serially. It was the critical unit (csynth interval 10772 vs ~6000 elsewhere). This is Gemmini's structure: `AccumulatorMem.scala` puts the add in the memory's write path and the matmul carries an accumulate bit. |
| consolidate loads | 1801 -> 1721 | one `dma_ld` per column block of B instead of one per (nb, kb) tile, and one `vld` for all of A. Pure program changes. |

**What was *not* fixed, and this is the honest headline:** the per-instruction
cost did not improve at all.

| | fixed cost | marginal cost |
|---|---|---|
| before | 506 cycles | 24.9 cycles/instruction |
| **after** | 557 cycles | **25.9 cycles/instruction** |
| Gemmini | 539 cycles | **5.9 cycles/instruction** |

The whole 1.39x came from *executing fewer instructions* (76 -> 45), not from
making an instruction cheaper. The unpipelined per-instruction loop named in the
original analysis is **still unpipelined**: a unit's body branches on the
opcode, so it is one large multiplexed region and Vitis will not pipeline it as
written. Splitting each unit's `if op == ...` arms into separate always-running
inner loops fed by per-opcode queues is still the 4x, and it is still the next
thing to do. What the work above did was remove the *count* the 26 cycles is
multiplied by, which is real but is a different lever.

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

## What would close the rest

1. **Pipeline the per-instruction loop** -- unchanged from the original
   analysis and still worth the remaining ~4x on the marginal term. Split each
   unit's `if op == ...` arms into separate always-running inner loops fed by
   per-opcode queues so instruction *n+1* starts while *n* drains.
2. **Then multiple instructions in flight.** A credit/scoreboard scheme over
   the in-order units would approach Gemmini's 48-entry reservation station
   without its complexity. Doing it before (1) optimizes the wrong term.

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

At 4x4x4 we are within ~32% of a matched Gemmini; at 16x16x16, 1.75x behind,
down from 2.43x. The remaining gap is one named, located property -- an
unpipelined per-instruction loop -- and not anything about the array, the data
type, or the dataflow, all of which are correct and RTL-verified. The array
holds one MAC per PE per cycle, and `microarch_ws.py` reached exactly 100% of
roofline with the same PE structure, so the compute is not the limit.

Worth being precise about the trend: the ratio still *grows* with problem size
(1.32x, 1.54x, 1.75x), which is the signature of a per-instruction cost gap
rather than a fixed-cost one. Reducing the instruction count moved every point
down but did not change that slope. Only pipelining the per-instruction loop
will.

Both utilizations are low in absolute terms because these are tiny problems on
a 4x4 array where fixed costs dominate; Gemmini's 26% at 16x16x16 is the number
to chase, and item (1) above is worth most of the distance.

## Reproducing

```bash
# ours
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build
export PYTHONPATH=/home/sk3463/allo OMP_NUM_THREADS=32
TPU_M=16 TPU_K=16 TPU_N=16 python cosim.py

# Gemmini, matched
cd ~/chipyard && source env.sh
cd sims/verilator && make CONFIG=Int8Dim4GemminiRocketConfig -j16
./simulator-chipyard.harness-Int8Dim4GemminiRocketConfig +permissive +permissive-off \
  ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/allo_cmp-baremetal
```

Raw Gemmini output in `logs/gemmini_int8_dim4.log`.
