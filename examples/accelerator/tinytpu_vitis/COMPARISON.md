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

## The numbers

| shape | ours (cosim) | Gemmini int8 4x4 | ratio | roofline | our util | Gemmini util |
|---|---|---|---|---|---|---|
| 4x4x4    |  **717** | 574 | 1.25x |   4 |  0.6% |  0.7% |
| 8x8x8    | **1003** | 615 | 1.63x |  32 |  3.2% |  5.2% |
| 16x16x16 | **2395** | 986 | 2.43x | 256 | 10.7% | 26.0% |

Our design is functionally exact at every shape (`mismatches = 0`), including
the int8 clip, for both `gemm` and `gemm.relu`.

## Where the gap is, exactly

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

## What would close it

In the order the evidence supports:

1. **Pipeline the per-instruction loop.** The loop body branches on the opcode,
   so a unit's body is a big multiplexed region and Vitis will not pipeline it
   as written. Splitting each unit's `if op == ...` arms into separate
   always-running inner loops fed by per-opcode command queues would let
   instruction *n+1* start while *n* drains. This is the whole 4x, and it needs
   no new hardware -- it is a restructuring of the same units.
2. **Give `accu` its own issue slot.** It is the slowest unit in the report
   (interval 10772 vs ~6000 for the others) because it does psum collection,
   `vadd`, `vrelu` and `mvout`. Splitting the vector ALU from the accumulator
   collection is the single most valuable unit-level split.
3. **Then, and only then, consider multiple instructions in flight.** A
   credit/scoreboard scheme over the existing in-order units would approach
   Gemmini's reservation station without its complexity. Doing this before (1)
   would be optimizing the wrong term.

## Honest reading

At 4x4x4 we are within 25% of a matched Gemmini. At 16x16x16 we are 2.4x
behind, and the reason is one named, located, fixable property -- an unpipelined
per-instruction loop -- rather than anything about the array, the data type, or
the dataflow, all of which are now correct and verified. The array itself holds
one MAC per PE per cycle and `microarch_ws.py` reached exactly 100% of roofline
with the same PE structure, so the compute is not the limit.

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
