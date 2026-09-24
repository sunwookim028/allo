# MiniTPU (Allo dataflow simulator)

An Allo model of MiniTPU's compute core: a 16x16 weight-stationary BF16
systolic array with a 24-bit float accumulator, the 32-entry vector register
file with its one shared read port and its one write port, a flat dual-port
VMEM, the per-lane output FIFOs, and the two matrix engines that drive them.
It runs a BF16 GEMM in the Allo dataflow simulator and checks the result
against MiniTPU's own arithmetic -- the acc24 chain, the single rounding to
BF16 at the array edge, and the BF16 `vadd` that sums 16-deep tiles.

MiniTPU is another engineer's machine (`~/core/minitpu`); this is a model read
out of that RTL, not a copy of it and not a reproduction of its cycle counts.
The reference page is [`docs/source/designs/minitpu.rst`](../../docs/source/designs/minitpu.rst).
Building this model turned up nine places where that page disagreed with the
live source; they are corrected there and listed under **Where the page was
wrong** below.

## Quick start

```bash
# the allo env sets neither of these (CLAUDE.md)
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
export PYTHONPATH=$PWD

python examples/minitpu/run.py --quick    # one 16x16x16 shape, ~2 min
python examples/minitpu/run.py            # four shapes, ~7 min
```

It prints one line per shape and then the schedule checks:

```
  16x16x16    28 cmds  build  89.4s  run  11.1s     EXACT vs model  rel err  0.162%  (bf16 tile bound 0.391%)
  ...
  refused: vst inside a matrix burst (VREG port C)
  refused: output FIFO overflow (64 result rows/lane)
  refused: vmatload refills an undrained weight bank
  accepted: the assembled 16x16x16 schedule

PASS
```

`EXACT` means bit-identical to `reference.py`, which implements MiniTPU's
arithmetic independently in numpy -- **our reading of it, not theirs**; see
"The weakest link" below. `rel err` is against float64 and is the measure
MiniTPU itself uses; its acceptance bar is 1 %.

Most of the wall time is elaboration: 256 PE kernels plus the array's edges
come to ~280 concurrent kernels, and the simulator gives each one an OS thread
(`allo/backend/simulator.py:1388`).

## How it works

Five files, no harness sprawl:

| file | what it is |
| --- | --- |
| `microarch.py` | the model: one `@df.region()` whose kernels are named after RTL modules |
| `program.py` | VMEM layout, the tape, and the three schedule rules the hardware does not enforce |
| `reference.py` | MiniTPU's arithmetic in numpy: acc24 chain, one BF16 round at the edge, BF16 tile accumulation |
| `run.py` | the gate |
| `README.md` | this |

Each kernel in `microarch.py` is one piece of the machine:

| kernel | RTL | what it preserves |
| --- | --- | --- |
| `sequencer` | `sequencer/sequencer.sv` | in-order issue; nothing stalls on a hazard |
| `stream_engine` | `mxu/mxu_stream_engine.sv` | **one** engine for `vmatload` and `vmatpush`, so they can never overlap |
| `pop_engine` | `mxu/mxu_pop_engine.sv` | a **second** engine: a pop runs while a load or push streams |
| `port_c` | `vpu.sv:429-431` | one physical VREG read shared by the matrix stream engine and the store path |
| `write_port` | `vpu.sv:289-305` | the single VREG write port every writer contends for |
| `dispatch` | `mxu.sv` | the shared 257-bit input FIFO, demuxed by kind; the weight switch armed by the next activation |
| `pe` (16x16) | `mxu/mxu_pe.sv` | weight-stationary, 2 pending banks, 24-bit accumulator, zero bypass at row 0 |
| `edge` (16) | `mxu.sv:82-94,129-157` | `pack_bf16` -- the one rounding point -- and the per-lane output FIFO |
| `alu` | `vpu/vpu_alu.sv` | BF16; where a contraction deeper than 16 is summed |
| `vmem_compute` / `vmem_dma` | `vpu/vpu_word_array.sv` | the flat word array's two ports, whole words only |

The arithmetic is the part worth reading. `acc24` is float32's exponent field
with eight fewer fraction bits, so it is emulated by a `float32`/`uint32`
bitcast pair and the RTL's own add-half-plus-lsb rounding, inlined in the PE.
A BF16 x BF16 product has a 16-bit significand, which acc24 holds exactly --
that is why the accumulator is 24 bits wide and why BF16 is the format. Row 0
adds into a zero psum through the adder's bypass, so the first term of each
column is exact; rows 1..15 each round; the column result is rounded to BF16
once. Depth beyond 16 is then summed in BF16 by `alu`, and that is where the
error actually comes from.

`program.py` carries the correctness argument the hardware does not. MiniTPU
has no interlocks anywhere -- its own README's line is that "a legal schedule
is a correctness argument, and the assembler carries it" -- so `check()`
refuses the three schedules that fail silently: a `vst` sharing port C with a
matrix stream command, an output FIFO overflow, and a `vmatload` refilling a
weight bank whose tile has not drained.

## Reference

`microarch.build(prog, m_rows, dim=16, sub=4, vmem_words=64)` elaborates the
model for one tape. The geometry constants are at the top of `microarch.py`,
each with its `vpu_pkg.sv` line:

| constant | value | RTL |
| --- | --- | --- |
| `DIM` (`NUM_LANES`) | 16 | `vpu_pkg.sv:25` |
| `SUB` (`NUM_SUBLANES`) | 4 | `vpu_pkg.sv:28` |
| `NVREG` | 32 | `vpu_pkg.sv:34` |
| `WEIGHT_BANKS` | 2 | `vpu_pkg.sv:108` |
| `FIFO_ENTRIES` | 16 groups of 4 = 64 rows | `mxu.sv:116`, `vpu_pkg.sv:39` |
| `MXU_FIFO_DEPTH` | 4 | `vpu_pkg.sv:51` |
| acc24 | 1 + 8 + 15 | `vpu_pkg.sv:100-101` |

`dim`/`sub` are parameters so the model can be exercised at 4x4 in seconds;
16/4 is the machine.

Shapes must satisfy `M % 4 == 0`, `K % 16 == 0`, `N % 16 == 0`, and must fit
the VREG budget: `DIM/SUB` registers for the weight tile plus `M/SUB` each for
the token tile, the pop destination and the accumulator, against a file of 32.
That allows `M` up to 32; `Layout` computes the bases and says what it needs
when a shape does not fit.

**`16x16x16` is the array's own tile, not a product MiniTPU's stack would
build.** One `vmatload`, one `vmatpush` per VREG, one `vmatpop` -- an ISA-level
tile, and a legitimate thing to model, but `minitpu-cc` would never emit it:
its blocks round up to multiples of 16 on M and N and its reduction pads to a
64-deep quantum, so the smallest product their compiler builds is
`[32,64]@[64,16]`. `run.py` runs both -- the ISA tile and their floor shape.

## Limits and known failures

1. **It cannot be emitted.** bf16 aborts every HLS emitter --
   `EmitVivadoHLS.cpp:115`, `EmitCatapultHLS.cpp:102` and `EmitSystemC.cpp`
   all reach `assert(1 == 0 && "Got unsupported type.")`, which is a SIGABRT,
   not an exception. `tests/dataflow/test_bf16_dataflow.py` pins this: float16
   emits on all three, bf16 dumps core on all three. Emission needs a
   `BFloat16Type` case in each `getTypeName`, and the C++ spelling is unknown
   -- `ap_bfloat16` and `hls::bfloat16` are the plausible Vitis names and
   neither is tested here.
2. **No cycle model.** Nothing in the model counts cycles, so it says nothing
   about the 168-cycle 16x16x16, the 52-cycle matrix step or the 85-cycle
   result latency, and it must not be quoted for them.
3. `Layout` rejects an `M` the VREG file cannot hold (32 is the ceiling, and
   it says what it needs) and any `N` or `K` that is not a multiple of 16.
4. **A separate agent owns bf16 emission.** Two findings handed to it rather
   than acted on here: SystemC has no BF16 type at all, so that emitter needs
   a 16-bit container plus conversions -- a design decision, not a type-table
   entry -- and Catapult's `ac_std_float<16, 8>` has BF16's field layout but
   is untested.

## The weakest link: there is no golden vector set yet

`EXACT` above means the model agrees with `reference.py`, and `reference.py` is
**our own reimplementation of MiniTPU's rounding rules**, read out of their RTL
and `docs/ARITHMETIC.md`. Two implementations of one reading agreeing is a
check on the model, not on the reading.

MiniTPU's owner is building a golden vector set -- weight and activation tiles
as BF16 hex words, expected output as BF16 hex words, with provenance (commit,
testbench, Verilator or board). No date promised. It is on their roadmap for
their own reasons: it is the instrument that would hold their functional
emulator to silicon, which they have never had either. **Until it exists,
nothing here is checked against MiniTPU's arithmetic -- only against our
reading of it.** Their `tools/emulate_image.py` is fp64 throughout with one
rounding at VREG writeback and no acc24 at all; their docs call it "useless as
a numerical oracle" in those words, which is why `reference.py` exists rather
than being borrowed.

## Where this model diverges from MiniTPU

Stated, not hidden. In rough order of how much they matter.

1. **No timing, so timing hazards are static rather than observed.** The
   weight-switch contract -- a `vmatload` no earlier than 75 cycles
   (`WEIGHT_SWITCH_SPAN`, `mxu.sv:60`) after the first `vmatpush` since the
   load two before it -- has no cycle to measure here. The model keeps the two
   banks and the bank tag, and `program.check()` enforces the *ordering* rule
   (the bank's previous tile must be drained). A violation of the 75-cycle
   window that satisfies the ordering rule is invisible to this model, exactly
   as it is invisible to MiniTPU's own functional emulator
   (`tools/emulate_image.py`, whose docstring says "Timing is not modelled").
   Only the RTL simulation sees it, through a `` `ifndef SYNTHESIS`` assertion
   (`mxu.sv:246-262`).
2. **The port-C mux serializes instead of corrupting.** In the RTL the matrix
   stream engine wins port C unconditionally and the store reads the matrix's
   register, silently (`vpu.sv:429-431`). Here `port_c` serves one requester
   per beat with matrix priority, so a collision costs order rather than
   producing garbage; `program.check()` refuses the collision instead.
   Likewise the write port, which in the RTL is an OR-mux that produces
   garbage when two writers fire (`vpu.sv:289-305`, `$onehot0` assertion).
3. **The weight switch is skewed east but not south.** The RTL skews commit by
   `row * MXU_PE_LATENCY` and then walks it east one PE per cycle
   (`mxu.sv:230,235`). The commit here rides the activation beat, which
   reproduces the eastward walk and the "armed by the next `vmatpush`" rule
   (`tile_starts`, `mxu.sv:170`) but not the row skew. The 61-deep west-edge
   activation skew line (`LHS_SKEW_DEPTH`, `mxu.sv:50`) is not modelled: it is
   purely a timing device.
4. **acc24 is emulated in float32 -- and for range that is exact, not an
   approximation.** `MXU_ACC_W = 1 + 8 + 15` gives acc24 *the same 8-bit
   exponent field as float32*; only the significand differs, 15 bits against
   23. Subnormal and overflow thresholds are therefore identical, and no
   BF16 x BF16 product can leave the range by construction, since BF16 inputs
   carry the same 8-bit exponent. An earlier revision of this list called this
   an inherited approximation; it was wrong, and MiniTPU's owner corrected it.
   **It holds because the accumulator is float.** A dtype-parametrised variant
   with an int32 accumulator in one configuration and a float one in another
   would break this silently, so it is a fact about this design and not a
   general licence.

   The one real difference: a float32 add followed by an acc24 round can
   double-round where the RTL rounds the exact sum once. Measured over 24
   million random acc24 operand pairs, the two disagree on **6,933 (0.03 %)**,
   always by one ulp of acc24, which usually vanishes at the final BF16
   rounding. `tools/accum_precision.py` takes the same float32 path, so this
   model and MiniTPU's own host reference agree; **neither can be held to
   silicon until the golden vector set exists** (see below).
5. **A NaN whose payload lives entirely in the dropped low 8 fraction bits
   becomes Inf in the RTL, and stays NaN here.** `pack_bf16` adds the
   round-to-nearest-even constant and takes the top 16 bits, which for such a
   NaN carries into the exponent and yields Inf. This model rounds f32 to bf16
   with MLIR's `truncf`, which preserves NaN. **The declared choice is to
   preserve the NaN**, because it is outside the operating range for real
   activations and nothing in either stack depends on NaN payloads -- but it
   is a divergence in behaviour, not a rounding difference, so it is written
   down rather than left to be discovered. No clamp is needed or missing
   otherwise: a carry into the exponent from a large finite value is correct
   IEEE rounding, and overflow inside the chain is classified
   (`SPECIAL_INF`/`SPECIAL_NAN`) rather than wrapped.
6. **Capacities are shrunk, structures are not.** VMEM is 64 words here and
   4096 (512 KiB) on the machine; the structure -- one flat array, two ports,
   whole words only, no arbiter -- is the machine's.
7. **`OP_FENCE` is not a MiniTPU instruction.** MiniTPU has no scoreboard;
   its assembler proves the schedule and the hardware never checks. Without
   cycles a model cannot make a RAW ordering deterministic that way, so the
   tape carries an explicit retirement fence. It is the one construct here
   with no counterpart in the machine.
8. **Out of scope, listed so the omission is not mistaken for a claim:** the
   128-bit VLIW bundle and its slots, the shared immediate, the 7-bit `DELAY`
   field, the 4-deep loop stack, the 24-bundle loop buffer, the AGU, the SFU,
   the cross-lane reduction tree and transpose, the DMA descriptor path and
   `wait.channel`, and the DRAM/AXI side. The model's tape is one command per
   entry and one command per engine.

## Where the page was wrong

`docs/source/designs/minitpu.rst` is our transcription, and the live source had
moved under it. Read from `~/core/minitpu` on 2026-09-24, read-only. **All nine
are now corrected on that page**, each marked `CORRECTED 2026-09-24`; they are
kept here as the list of what a second reading of the RTL changed:

| the page says | the source says |
| --- | --- |
| "a serialized matrix-command FSM -- push and pop share one controller" | `vmatload` and `vmatpush` share `mxu_stream_engine`; `vmatpop` has its own `mxu_pop_engine` and runs concurrently (`mxu_matrix_ctrl.sv:6-7`) |
| "weight-commit reaches every PE on one cycle (`mxu_pe.sv`)" | commit is skewed by `row*PE_LATENCY` then walks east; `WEIGHT_SWITCH_SPAN = 75` (`mxu.sv:60,230,235`) |
| "Recovering it needs hardware (... or a second weight bank)" | `MXU_WEIGHT_BANKS = 2` already (`vpu_pkg.sv:108`) |
| "MXU output FIFO 32/lane" | `MXU_OUTPUT_FIFO_DEPTH = 64` result rows per lane = 16 entries of 64 b (`vpu_pkg.sv:39`, `mxu.sv:116`) |
| "`mxu_adapter.sv` is dead code (verified, still present)" | no such file exists anywhere under `~/core` |
| "the **write port** of the 3R1W register file -- port C shared by store and matrix" | port C is one of the three **read** ports (`raddr_c`, `vpu_regfile.sv:16-27`); the single write port is a separate OR-mux over six sources |
| "82-cycle push->result latency" | 85 (`docs/isa_latency.json`, `docs/ISA_AND_INTERFACES.md:57`); 82 survives only in a stale comment at `sequencer_pkg.sv:32` |
| "`sequencer_pkg.sv:23`" for `DELAY_W` | `src/core/sequencer/sequencer_pkg.sv:33`; there is no `src/pkg/sequencer_pkg.sv` |
| "4 banks per lane ... **Wrong for the live design**" | right about VMEM, which is flat; but `NUM_SUBLANES = 4` is commented "== banks per lane" (`vpu_pkg.sv:45`) and the **VREG file** is four physical 256-bit banks (`vpu_vreg_stripe.sv:73-87`) |
| accumulator "not fixed point" (correct) | and subnormals are **enabled** in acc24; the multipliers flush them, the adders do not (`ARITHMETIC.md` sec. 5) |
