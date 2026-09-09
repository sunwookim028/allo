# TinyTPU-ws: weight-stationary, int8, and what the numbers say

`microarch_ws.py`. Two things changed from `microarch.py`, and they turned out to
be independent -- each fixes something the other does not.

## 1. Output-stationary -> weight-stationary fixes II

`microarch.py` is output-stationary: PE(i, j) owns `C[i, j]` and contracts over
k in place. That makes the contraction a **loop-carried dependence of distance
1 with the adder inside it**, so `II = adder latency`. Vitis:

```
[HLS 200-880] Unable to enforce a carried dependence constraint
  (II = 1, distance = 1) between 'store' and 'load' on 'v121'   <- acc
... Final II = 7
```

In the WS design the partial sum *leaves* the PE every cycle, so nothing is
loop-carried at all. Holding the dtype at fp32 -- a genuine single-variable
change -- the same tool now reports:

```
Pipelining result : Target II = NA, Final II = 8, Depth = 24, loop 'l_S_c_0_c3'
```

II = 8 for M = 8 wavefronts is **one MAC per cycle**, and the fp32 adder's 7
cycles have become *depth* (24) instead of initiation interval. That is the
entire point of a systolic array: it is deep so that it can be fast.

Worth recording: Gemmini's `PE.scala` supports **both** dataflows, and in its OS
mode it holds *two* accumulators `c1`/`c2` with `propagate` alternating between
them. That is partial-sum rotation at P=2, in hardware -- Gemmini needs the same
trick for OS, for the same reason. Its tapeout/lean configs all pin
`dataflow = Dataflow.WS`, and `allo_cmp.c` already passes `WS` to
`tiled_matmul_auto`, so the reference numbers were WS all along.

## 2. fp32 -> int8/int32 fixes depth and area

Gemmini's *default* config is `inputType = SInt(8.W)`, `accType = SInt(32.W)`.
The fp32 configs are the off-default ones, and on an FPGA they are soft-float:
an fp32 add is ~7 cycles where an int32 add is 1. Holding fp32 therefore
measures FPGA soft-float latency rather than the quality of the generated
architecture. int8/int32 is both the fair comparison and Gemmini's own default,
so `TPU_DTYPE` now selects it, with `fp32` kept for single-variable runs.

Also relevant to fairness: `GemminiFP32DefaultConfig` derives from
`defaultFPConfig`, which is **`meshRows = meshColumns = 4`**. So the fp32
Gemmini the reference numbers come from is a 4x4 mesh, matching `T = 4`. Array
size was already matched; the dtype was not.

At 8x8x8, everything else held (csynth estimates):

| design | dtype | PE II / MAC | PE depth | top interval | DSP | FF | LUT |
|---|---|---|---|---|---|---|---|
| output-stationary | fp32 | **7** | -- | -- | 80 | 11857 | 13768 |
| weight-stationary | fp32 | **1** | 24 | 88 | 88 | 40865 | 31037 |
| weight-stationary | int8 | **1** | 12-14 | 74 | 96 | 14472 | 25760 |

The dataflow change bought II; the dtype change bought depth (24 -> 14) and 2.8x
the flip-flops back.

## 3. Feeders: partitioning the memory, not the process

The array wants T activations and T weights per cycle; one process reading one
2-port memory supplies two, which was `loader`'s `[HLS 200-885] Final II = 2`,
and it degrades linearly in T. The fix is `s.partition(...)` -- reachable
because `df.build` is just `customize(func)` plus `s.build(...)`, so the
schedule primitives are available on the Vitis path:

```python
s.partition(f"{top}:A", Partition.Cyclic, dim=2, factor=T)   # and B, C
s.partition(MockBuffer("drainer_0", "acc"), Partition.Complete, dim=2)
```

The drainer needed it too, and Vitis named the fix in as many words:

```
[HLS 200-885] Unable to schedule 'store' ... on array 'acc' due to limited
memory ports (II = 31). Please consider ... partitioning the array 'acc'.
```

With that, the drainer went from II=32 to II=8 per instruction and the top-level
interval from 168 to 74. **Every stage now runs at M cycles per instruction**:
sequencer, wloader (T*T weights in T cycles), loader (M*T activations in M
cycles), 16 PEs, drainer. No stage is the bottleneck.

## 4. The array reaches its roofline

Array time is `NI * II = (Nt * Kt) * M` cycles, against a roofline of
`M*N*K / T^2`:

| shape | instructions | II | array cycles | roofline | utilization |
|---|---|---|---|---|---|
| 8x8x8    |  4 |  8 |  32 |  512/16 =  32 | **100%** |
| 16x16x16 | 16 | 16 | 256 | 4096/16 = 256 | **100%** |

For comparison, the `chia-codesign` v2 design measured 15-21% of roofline, and
the reason was named there: a unit is a `func.call`, the compiler will not
pipeline a loop across one, and consecutive instructions overlapped by exactly
zero. Vitis emits a dataflow region as persistent processes, so that cost is
gone rather than paid per instruction.

## 5. Cycle counts, with the caveats stated

| shape | ours int8 (est.) | ours fp32 (est.) | chia v2 fp32 (cosim) | Gemmini fp32 4x4 (rdcycle) |
|---|---|---|---|---|
| 8x8x8    |  316 lat / 74 int | 397 / 88 | 412  |  727 |
| 16x16x16 | 1020 lat / 330 int | -- | 1734 | 1163 |

**These are not yet an apples-to-apples claim, and should not be quoted as one.**
Three reasons, in order of how much they matter:

1. **Ours are csynth *estimates*; Gemmini's are measured `rdcycle`.** `cosim` is
   the number that can be compared, and it has not been run.
2. **Dtype still differs in the headline column.** Our int8 against Gemmini's
   fp32 is not a fair fight in our favour. The fp32 column (397 at 8x8x8 vs
   727) is the honest dtype-matched, mesh-size-matched comparison, and it is an
   estimate against a measurement.
3. **Gemmini's `rdcycle` includes RoCC dispatch** from the CPU and its own
   mvin/mvout; our latency includes the `m_axi` burst copies, which is the same
   shape of measurement but not the same overheads.

What *is* established without caveat: the array is real (DSP scales with T^2 and
with dtype), it holds one MAC per PE per cycle, every stage matches it, and the
steady-state array time is exactly the roofline.

## 6. The deadlock: measured facts, and a wrong guess retracted

The design deadlocks in the KPN simulator unless the stream depth is large. This
is the one unresolved problem, and it is worth separating what is measured from
what is not.

**Measured.** A control run settled where to look: upstream's own 64-tile grid
design (`test_multi_cache_gemm.py`) passes the simulator in **2.8 s**, so the
simulator is fine and the design is not. Then, with per-class depths
(`TPU_QI/QW/QA/QP/QC`):

| experiment | result |
|---|---|
| minimum workable depth, 8x8x8 (M=8, NI=4) | **32** |
| minimum workable depth, 16x16x16 (M=16, NI=16) | between 128 and **256** |
| all channels 64, drop `QI` (sequencer -> units) to 16 | pass |
| all channels 64, drop `QW` (weight shift-in) to 16 | pass |
| all channels 64, drop `QA` (activations) to 16 | **hang** |
| all channels 64, drop `QP` (partial sums) to 16 | **hang** |
| all channels 64, drop `QC` (array -> drainer) to 16 | **hang** |

So the three *data* channels each need the depth and the two *control* channels
do not, and the threshold tracks `M * NI` -- the entire per-channel traffic of
the whole run. A depth that must scale with the problem size means back-pressure
never works at all: the producer has to be able to run to completion.

**Retracted.** I guessed the cause was the order of the two `put`s in the PE
body -- forwarding the activation east before emitting the psum south, which
would let a full `a_fwd` stop a column from producing and deadlock against the
drainer's fixed read order. It is a real-looking cycle and it is **not the
cause**: swapping the two puts leaves the minimum depth at exactly 32 for
8x8x8, unchanged. The code keeps the reordering with a comment saying it makes
no difference, so the next person does not re-run the experiment.

**What the evidence does point at.** Both remaining suspects have the same shape
-- a single process that fans out to, or in from, `T` channels in a *fixed
order*: the `loader` writes `a_in[0..T-1]` in sequence, and the `drainer` reads
`c_out[0..T-1]` in sequence. If any one of those `T` channels fills (or
empties), that process stops servicing the other `T-1`, and the rows it starves
are the ones whose progress would have drained the channel it is stuck on. This
also explains why upstream does not have the problem: **upstream has no such
process.** `offchip_loadA` writes exactly one stream, and the border PEs
daisy-chain it (`L2_A[i] -> L2_A[i+1]`), so every process writes at most a
couple of channels and the chain order matches the consumption order.
`test_tiled_systolic.py` runs on depth-**4** FIFOs for the same reason.

Which makes the fix the same thing as the original question about whether the
feeders are "fully partitioned": replace the `T`-way fan-out with upstream's
daisy chain. Allo also supports non-blocking stream ops
(`tests/dataflow/test_stream_nb_simple.py`), which would let a fan-out process
skip a full channel instead of blocking on it -- a smaller change, and worth
measuring against the chain.

**And an abstraction finding regardless of which suspect it is.** Nothing
catches this. It is not a type error; Vitis synthesizes it happily, since a
bounded FIFO is legal hardware either way; and the simulator's only symptom is a
hang with no message -- no deadlock detection, no indication of which channel is
full or which process is blocked. For a language whose pitch is that a `Stream`
grid is the natural way to express a spatial architecture, the *global* property
"these channel depths suffice" is load-bearing and completely unsurfaced. A
simulator deadlock report naming the blocked processes and the full channels
would have turned a multi-session hunt into a single run, and is much cheaper
than the static analysis.

## Reproducing

```bash
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build   # not set by the env
export PYTHONPATH=/home/sk3463/allo OMP_NUM_THREADS=8

# functional, exact for int8 (maxerr = 0, clip included)
TPU_M=8 TPU_K=8 TPU_N=8 python bench_ws.py simulator

# synthesis; TPU_DTYPE=fp32 for the single-variable dtype comparison
TPU_M=8 TPU_K=8 TPU_N=8 TPU_DTYPE=int8 python bench_ws.py csyn
cd ws_csyn_int8_8x8x8.prj && source /opt/xilinx/Vitis_HLS/2023.2/settings64.sh \
  && vitis_hls -f run.tcl
```

## Next, in order

0. **Bound the stream depths** -- replace the `T`-way fan-out loader/drainer
   with upstream's daisy chain, or use non-blocking stream ops. Until this is
   fixed the design needs depth proportional to the problem, which is not a
   real accelerator. It is also the one item that blocks scaling past 16x16x16.
1. **`cosim`** for a measured cycle count. This is the only thing standing
   between the table above and a real comparison, and `df.build`'s `vitis_hls`
   path handles `csim`/`csyn` but routes other modes to the `XDEVICE` Makefile
   flow, so the `cosim_design` tcl has to be written by hand.
2. **A Gemmini int8 DIM=4 build** so the int8 column has a matched reference.
   `allo_cmp.c` needs no source change -- it is written against `elem_t` and
   fills with values in [-4, 4] -- so this is a config plus a Chipyard rebuild.
3. Narrow the MAC: 8 DSPs per PE for an int8 product is the int16 widening
   being taken literally.
4. Push T to 16 to match Gemmini's int8 default mesh, now that II is
   dtype-independent and the feeders partition by T.
