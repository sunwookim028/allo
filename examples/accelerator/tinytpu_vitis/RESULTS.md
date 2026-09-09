# TinyTPU-vitis: the feeder/drainer restructure, and what Vitis says

## It synthesizes

`vitis_hls -f run.tcl` at 4x4x4, one instruction: **0 errors**, against the
single-grid version's four (`../tinytpu_grid/repro/vitis_csyn_errors.log`). Full
log in `csyn_4x4x4.log`, report in `csynth_4x4x4.rpt`.

```
| Modules & Loops              | Latency | Interval | Pipelined | DSP | FF    | LUT   |
| + tinytpu_vitis*             |      49 |       42 |  dataflow |  80 | 11857 | 13768 |
|  + sequencer_0               |       1 |        1 |        no |   - |     4 |   180 |
|  + loader_0                  |      11 |       11 |        no |   - |   145 |   235 |
|   o l_S_k_0_k                |       9 |        2 |       yes |   - |     - |     - |
|  + pe_0_0                    |      41 |       41 |        no |   5 |   680 |   709 |
|   o l_S_k_0_k1               |      34 |        7 |       yes |   - |     - |     - |
|  ... 15 more identical PEs                                                          |
```

Three things worth reading off that table:

- **`dataflow` at the top.** 19 concurrent persistent processes -- sequencer,
  loader, 16 PEs, drainer -- not 36 instances each holding a copy of every
  array. This is the structure the fan-out errors were about.
- **80 DSPs = 16 PEs x 5.** The array is real: one FP MAC per PE, none folded.
- **The PE inner loop is II=7, not 1.** That is the whole remaining problem.

## The PE recurrence: the same finding, on a third backend

```
WARNING: [HLS 200-880] The II Violation in module 'pe_0_0_Pipeline_l_S_k_0_k1':
  Unable to enforce a carried dependence constraint (II = 1, distance = 1,
  offset = 0) between 'store' operation ... on local variable 'v121' and
  'load' operation ... on local variable 'v121'
```

`v121` is `acc`. Vitis walks II = 1, 2, 3, 4, 5, 6 and settles at 7 -- the fp32
adder's latency. `acc += a * b` reads and writes the same location every
iteration, so the recurrence is one iteration long and the adder cannot fit
inside it.

This is the third backend to say the same thing about in-place accumulation:

| backend | how it shows up |
|---|---|
| chia RTLGen | `addf -> memref.store -> memref.load -> addf`, latency 4 over distance 1; fixed only by a **separate** store-only output buffer plus a second pass (`FINDINGS_v2.md` F.2) |
| Vitis HLS | `HLS 200-880` carried dependence, II 1..6 all rejected, **Final II = 7** |
| SystemC/Catapult | their own pitfall list: "A read-modify-write output array (`C[i] += ...`) forces single-shot execution ... you can't free-run/pipeline it" |

So it is not a quirk of any one scheduler. **An accumulator expressed as a
read-modify-write in the datapath cannot be pipelined at II=1 anywhere**, and
Gemmini's answer is to not express it there -- `AccumulatorMem.scala` puts the
add in the memory's own write path.

## The fix for the PE, and it is standard

Split `acc` into `P` partial sums rotated by `k % P` and combine at the end.
Consecutive iterations then touch different registers, the recurrence distance
becomes `P`, and `II = ceil(adder_latency / P)`. With the measured latency of 7,
`P = 7` reaches II=1 at the cost of 7 registers and a 7-way reduction per
instruction, amortised over `K`. This is the classic modulo-unroll and it needs
no new compiler feature -- unlike the chia route, which needed a whole extra
pass over the data.

`loader_0`'s II=2 is separate and easier:

```
WARNING: [HLS 200-885] Unable to schedule 'load' ... on array 'v765' due to
  limited memory ports (II = 1). Please consider ... partitioning the array
```

One port, two reads per iteration. Cyclic-partition `A` and `B` by `T`.

## Status, honestly

| | state |
|---|---|
| Vitis dataflow legality | **passes**, 0 errors |
| real T x T array in hardware | **yes**, 80 DSPs = 16 x 5 |
| instruction-programmable | **yes**, `mm` / `mm.relu` from `imem`, verified |
| functionally correct | **yes at one tile**, both opcodes, in the KPN simulator |
| multi-tile | **hangs the KPN simulator** -- same signature as the single-grid
  version, so it is *not* caused by the restructure. It has not yet been tried
  under Vitis `csim`/`cosim`, which is a different engine, so it may not be a
  property of the design at all. That is the next run. |
| deeply pipelined | **not yet**: PE II=7, loader II=2 |
| cycle count | not yet -- `cosim` needs the multi-tile question settled first |

The 49-cycle latency above is a **synthesis estimate for one instruction with no
memory system**, so it is not comparable to Gemmini's 605-cycle end-to-end
`rdcycle` at 4x4x4 and should not be quoted against it. What the table does
establish is that the structure is legal, the array is physically there, and the
two things standing between it and a deeply pipelined machine are both named by
the tool with the fix in the message.

## Next, in order

1. `csim`/`cosim` at 8x8x8 -- does Vitis' engine have the KPN simulator's
   multi-tile hang? This decides whether the hang is the design or the simulator.
2. Partial-sum rotation in the PE (`P = 7`) -> II=1.
3. Cyclic-partition `A`/`B` -> loader II=1.
4. `cosim` for a real cycle count, comparable to chia's 137 at 4x4x4 and
   Gemmini's 605.
