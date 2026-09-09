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
| 4x4x4    | 1037 | 574 | 1.81x |  0.4% |  0.7% |
| 8x8x8    | 1165 | 615 | 1.89x |  2.7% |  5.2% |
| 12x12x12 | 1389 | 740 | 1.88x |  7.8% | 14.6% |
| 16x16x8  | 1437 | 784 | 1.83x |  8.9% | 16.3% |
| 16x16x16 | **1733** | 986 | **1.76x** | 14.8% | 26.0% |

**The ratio is flat at ~1.8x and falls slightly with size.** That is a
qualitatively different result from the per-workload measurement, whose ratio
*grew* (1.32x -> 1.54x -> 1.75x). Fixing the hardware exposed that the earlier
growth was partly an artifact of specialization, and the real remaining gap is
close to a constant factor.

## The I/O trade, measured

`wrap_io` is not a default to accept -- it is an architectural choice with a
crossover, and neither setting is what Gemmini has:

| config | marginal | fixed | 4x4x4 | 16x16x16 |
|---|---|---|---|---|
| `wrap_io=True`, imem 256 | 18.1 cyc/instr | 1102 | 2.12x | 1.94x |
| `wrap_io=False` | **39.8** | **481** | **1.21x** | 2.25x |
| `wrap_io=True`, imem 76 | **18.1** | 922 | 1.81x | **1.76x** |
| Gemmini | **10.8** | **483** | 1.00x | 1.00x |

* `wrap_io=True` copies each argument into a local buffer before the region
  runs, so the units read BRAM -- cheap per access, but the copy is the
  *declared* length regardless of the shape being run.
* `wrap_io=False` lets the units read `m_axi` directly. The fixed cost drops to
  481, essentially equal to Gemmini's 483 -- but every access now pays bus
  latency instead of hitting a buffer, and the marginal cost **doubles**. It
  also requires flat arguments ("Top-level multi-dimensional arrays are
  linearized to 1D pointers"), which is why `A`/`B`/`C` are 1-D.
* Crossover is ~29 instructions, so buffering wins across this benchmark set
  once the imem is trimmed to the longest admissible program (256 -> 76 words,
  worth 180 cycles at every shape).

**Gemmini has the third option and Allo does not expose it: a bursted DMA the
program controls, giving low fixed cost *and* low marginal cost.** `mvin`
transfers exactly the tiles the program names. That, not the array and not the
data type, is the largest single remaining item.

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

1. **Pipeline the per-instruction loop** -- worth the 1.7x marginal term. Split each
   unit's `if op == ...` arms into separate always-running inner loops fed by
   per-opcode queues so instruction *n+1* starts while *n* drains.
2. **A program-controlled burst DMA** -- worth the 1.9x fixed term, and the
   one item that needs something Allo does not currently expose (see the I/O
   trade above).
3. **Then multiple instructions in flight.** A credit/scoreboard scheme over
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

On one fixed build, matched in data type and array size, the design is a **flat
~1.8x behind Gemmini** across five shapes, and the ratio does not degrade with
problem size (1.81x at 4x4x4, 1.76x at 16x16x16). Utilization tracks at roughly
half Gemmini's at every point.

Two terms make up the 1.8x, and both are named:

1. **Marginal, 18.1 vs 10.8 cycles/instruction (1.7x).** The per-instruction
   loop in each unit is `pipelined = no`, so a unit finishes instruction *n*
   before starting *n+1*. Gemmini's 48-entry reservation station is what buys
   the difference.
2. **Fixed, 922 vs 483 cycles (1.9x).** Allo's argument handling: bulk-copy the
   declared arrays, or unbuffered `m_axi`, but no program-controlled burst DMA.

Neither term is about the array, the data type, or the dataflow -- all three are
correct and RTL-verified, and `microarch_ws.py` reached exactly 100% of roofline
with the same PE structure. The compute is not the limit.

Both utilizations are low in absolute terms because these are tiny problems on
a 4x4 array where fixed costs dominate; Gemmini's 26% at 16x16x16 is the number
to chase, and item (1) above is worth most of the distance.

## Reproducing

```bash
# ours -- one build, all shapes
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build
export PYTHONPATH=/home/sk3463/allo OMP_NUM_THREADS=32   # >= 22 processes
python bench_isa.py          # functional sweep, one build
python cosim.py              # one csynth, cosim per shape
TPU_WRAP=0 python cosim.py   # the unbuffered m_axi variant

# Gemmini, matched
cd ~/chipyard && source env.sh
cd sims/verilator && make CONFIG=Int8Dim4GemminiRocketConfig -j16
./simulator-chipyard.harness-Int8Dim4GemminiRocketConfig +permissive +permissive-off \
  ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/allo_cmp-baremetal
```

Raw Gemmini output in `logs/gemmini_int8_dim4.log`.
