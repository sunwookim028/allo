# Gemmini accelerator, int8/int32, DIM=8, capacity-matched

Chisel-elaborated RTL for the ASIC synthesis-only handoff
(DC, freepdk-45nm `view-standard`, memories as flip-flops, no
P&R) -- the **same flow and the same settings** our own variants
go through, so the areas compare. The top module is `Gemmini`.

## Two things this export needs that ours did not

1. **sv2v IS needed here** -- or a SystemVerilog reader. "Chisel
   emits Verilog" is not true of firtool 1.75.0: `CounterFile`,
   `LoopMatmulStC` and `RRArbiter` use packed multidimensional
   arrays (`wire [7:0][6:0] _GEN_1`) and assignment patterns
   (`'{3'h5, 3'h0, ...}`) **outside any `ifdef`**. Verilator
   rejects all four file lists under `--language 1364-2001` with
   6-7 syntax errors and accepts them as SystemVerilog. So either
   set `normalize_rtl: True` and let sv2v convert, or read with
   `analyze -format sverilog`. The files are named `.sv` for that
   reason; the memory models, which are chipyard's and are plain
   Verilog-2001, are `.v`.
2. **`SYNTHESIS` must be defined** when the files are read. With
   it, `plusarg_reader` collapses to `assign out = DEFAULT` and
   the randomisation scaffolding disappears; without it DC meets
   `$value$plusargs` inside an `initial` block and 173 `logic`
   declarations under `ENABLE_INITIAL_REG_`.

The **top-module check is fine**: `module Gemmini(` carries a
trailing comment, not an attribute, so the collector's
`^\s*module\s+<top>` pattern matches -- the defect that forced
`normalize_rtl: False` on the Vitis export does not apply here.

Verified on this host: all four file lists (both configurations x
full and logic-only) pass `verilator --lint-only -DSYNTHESIS
--top-module Gemmini` with no error.

This is the opponent for **TinyTPU-isa T=8, MAXDIM=64 (`rtl_handoff/T8_MAXDIM64`)**.

## What is inside the boundary

The cut is the transitive closure of the module instantiation
graph from `Gemmini`, computed from the elaborated Verilog -- not a
hand-picked list. 136 modules, 4 of
them memory arrays.

- **mesh** -- `Mesh`, `Tile`, `PE`, `MacUnit`, `MeshWithDelays`, `TransposePreloadUnroller`, `AlwaysOutTransposer` -- the systolic array and its skew registers. Our `pe`/`accu` units.
- **scratchpad** -- `Scratchpad`, `ScratchpadBank`, `mem`/`mem_ext` -- 8 KiB of int8 operand storage. Our `spad` plus vector registers.
- **accumulator** -- `AccumulatorMem`, `TwoPortSyncMem`, `mem_0`/`mem_0_ext`, `AccumulatorScale`, `AccPipe`, `ScalePipe` -- 4 KiB of int32. Our `ar`.
- **controllers** -- `ExecuteController`, `LoadController`, `StoreController`, `ReservationStation`, `LoopMatmul*` -- decode, dependency tracking and the hardware loop. Our `sequencer`.
- **DMA** -- `StreamReader`, `StreamWriter`, `BeatMerger`, `XactTracker`, `DMACommandTracker`, `TLBuffer`, `TLXbar` -- the accelerator's own TileLink master and its buffering. Our `dma_ld`/`dma_st` and their AXI interfaces.
- **queues** -- `Queue*`, `Pipeline*`, `MultiHeadedQueue`, `TagQueue`, `RRArbiter`, `WeightedArbiter`, `ram_*` -- the small flop-backed memories the controllers run on.

## What was cut, and why

Chipyard elaborates an SoC. Our design is an accelerator with no
host, no cache and no core, so everything below is outside the
comparable unit; leaving any of it in would compare a CPU with a
matrix unit.

- **Rocket core and its tile** -- `RocketTile`, `CSRFile`, `FPU`, `Frontend`, the branch predictor. Our design has no host: its program is in DRAM and it starts on `ap_start`. Gemmini has no decoder of its own -- it is a RoCC accelerator and the CPU issues its instructions -- so keeping the core would be measuring a CPU.
- **L1 I/D and the L2** -- `rockettile_dcache_*`, `rockettile_icache_*`, `cc_banks_*`, `cc_dir_*`, `InclusiveCache`. Our operands come straight from DRAM over AXI, uncached.
- **SoC buses and peripherals** -- `SystemBus`, `MemoryBus`, `PeripheryBus`, `TLXbar`s outside the accelerator, CLINT, PLIC, debug, bootrom, `TSIToTileLink`, the serial adapter and clock/reset infrastructure.
- **DRAM model and test harness** -- `SimDRAM`/`TestHarness`. Ours is `-m_axi_latency 0`; neither side synthesises its memory system.

### The one ambiguous cut: Gemmini's own TLB

`FrontendTLB` is **inside** this export, because it is inside the
`Gemmini` module: Gemmini's DMA issues virtual addresses and the
4-entry TLB translates them. Ours takes physical addresses over
AXI and has no translation at all, so a reader may reasonably
want it out. It is not cut here, because cutting it would mean
editing Gemmini's RTL; instead **report it separately**: DC's
`report_area -hierarchy` gives the `FrontendTLB` instance's area
directly, and both the with-TLB and without-TLB figures should be
quoted. The modules are `FrontendTLB`, `DecoupledTLB`, `DTLB_2`, `PMAChecker`, `PMPChecker_s6`, `OptimizationBarrier_TLBEntryData`.

### Kept although unexercised, and why

- **The conv pipeline** (`LoopConv*`, `Im2Col`, `PixelRepeater`,
  `ZeroWriter`) and **the output-stationary datapath**
  (`dataflow = BOTH`) are never used by a GEMM. They are kept
  because removing them is tuning the opponent; both make Gemmini
  look bigger, which is the direction that does not flatter us.
- **The fp32 scaling pipeline** (`MulAddRecFN_e8_s24`,
  `INToRecFN_*`, `RecFNToIN_*`, `RoundAnyRawFNToRecFN_*`) comes
  from `defaultConfig`'s `mvin_scale_args`, which are `Float`.
  Gemmini's default int8 accelerator carries an fp32 multiplier
  per mvin lane and in the accumulator scale; we have none. Same
  reasoning, same direction -- but this one is worth naming in
  any area quote, because it is a real block and not a corner.
- **`TLMonitor_43`..`_47`** are TileLink protocol assertions with
  no outputs. DC optimises them to nothing; they are kept so the
  file list is the closure and not the closure minus a judgement.

## Configuration

- `meshRows / meshColumns` = 8 (DIM = 8)
- `tileRows / tileColumns` = 1
- `inputType` = SInt(8.W)
- `accType` = SInt(32.W)
- `spatialArrayOutputType` = SInt(20.W)
- `dataflow` = BOTH (weight- and output-stationary; only WS is exercised)
- `sp_capacity` = 8 KiB in 4 banks (stock: 256 KiB)
- `acc_capacity` = 4 KiB in 2 banks (stock: 64 KiB)
- `tlb_size` = 4 entries
- `dma_buswidth / dma_maxbytes` = 128 bits / 64 bytes

The delta from Gemmini's own `defaultConfig` is **four fields**:
`meshRows`/`meshColumns` (the array size, which is the point),
`has_training_convs = false`, and `sp_capacity`/`acc_capacity`.
`tileRows`/`tileColumns` also appear in the `copy()` at the values
they already hold. Everything else -- `spad_read_delay`,
`tile_latency`, `mesh_output_delay`, `dataflow`, the reservation
station and queue depths, `max_in_flight_mem_reqs` -- stays at
Gemmini's defaults.

### Why the capacity was matched, and what that gives up

Stock Gemmini has a **256 KiB scratchpad and a 64 KiB
accumulator**. Both sides are synthesised with
`sram_mode='none'`, so a stock Gemmini would put **2,621,440
flip-flops** on the scale where our whole T=4 design has 200,561
sequential cells: the number would measure the memory treatment
and nothing else.

8 KiB of scratchpad and 4 KiB of accumulator -- 98,304 bits --
is what this carries instead. 4 KiB is the smallest accumulator
inside the envelope the cycle-neutrality sweep in
`designs/gemmini_comparison.rst` already covers; 2 KiB is legal
but sits exactly on the binding half-capacity limit for a 16x16
int32 tile, where the tiling search has never been run.

**Against which of our builds that is a match, measured from the
RTL rather than from the prose:**

| our variant | operand + accumulator storage | vs Gemmini's 12 KiB |
| --- | --- | --- |
| `T4_MAXDIM16_shipped_baseline` | 2.5 KiB (0.25 + 0.25 + 2.0) | Gemmini has **4.8x** |
| `T4_MAXDIM64_shipped` | 10.1 KiB (4 + 4 + 2.1) | Gemmini has 1.19x |
| `T8_MAXDIM64` | 12.25 KiB (4 + 4 + 4.25) | **matched to 2 %** |

The 4 KiB / 4 KiB / 2.1 KiB figures this project quotes for its
own memories describe **`MAXDIM=64`**, not the `MAXDIM=16`
baseline whose 1,136,598 is published: at `MAXDIM=16` the
scratchpad and vector registers are 64 rows of 32 bits each,
256 bytes apiece. So the capacity match is near-exact against
`T8_MAXDIM64` and against `T4_MAXDIM64_shipped`, and it is
**4.8x in Gemmini's disfavour** against the MAXDIM=16 baseline.
That is why the logic-only figure, not the full one, is the
headline.

**This is legitimate only because the shrink is cycle-neutral.**
`tiled_matmul_auto`'s own tiling search issues exactly one
`loop_ws` at every published shape from 256/64 KB down to 4/4 KB,
so none of the five Gemmini cycle numbers moves. It stops being
neutral at 32x32x32, so this argument does not extend to larger
shapes without being re-measured.

What it gives up: the 256 KiB scratchpad is a **real capability**
that this export deliberately removes. The area below answers
*what does the same amount of local memory cost in each design*,
not *what does the shipped Gemmini cost*. Anyone quoting it
against a published Gemmini area is quoting the wrong thing.

## The memories in this export

- `mem_0_ext`: 64 x 256 bits x 2 instances = 32,768 bits, ports `mwrite,read`
- `mem_ext`: 256 x 64 bits x 4 instances = 65,536 bits, ports `mrw`
- total memory-array storage: **98,304 bits** (12.0 KiB), which is 98,304 flip-flops under
  `sram_mode='none'` -- against our T=4 design's 200,561
  sequential cells. Stock Gemmini would be 2,621,440.

They arrive as behavioural flop arrays (`reg [7:0] ram [0:N]`
inside `split_*_ext`), which is what `sram_mode='none'` means on
our side too. A second file list is provided so the same export
yields a memory-free figure:

| file list | what DC sees | what the area means |
| --- | --- | --- |
| `sv2v_manifest.f` | everything | accelerator + 12 KiB of flops |
| `sv2v_manifest_nomem.f` | `mem`/`mem_0` as empty black boxes | accelerator LOGIC only |

The logic-only figure is the one that is invariant to both the
memory treatment and the capacity choice, so it is the safer of
the two to quote. Our side needs the same pair to be comparable:
the matching omission is every `*_RAM_*` module in
`rtl_handoff/`.

## Provenance

- chipyard `e0207441`, gemmini `25809f7`
- elaborated by `make verilog CONFIG=Int8Dim8AreaGemminiRocketConfig`
- hardware read back out of the RTL's own memory geometry: `ACC_DIM`=8, `ACC_ROWS`=128, `BANK_NUM`=4, `BANK_ROWS`=256, `DIM`=8 -- **not** from `gemmini_params.h`, which every elaboration
  rewrites in place and which therefore describes whichever
  config was elaborated last
- exported by `export_gemmini_rtl.py` at allo `e4b9d58f`

`sv2v_manifest.f`, `sv2v_manifest_nomem.f` and
`MANIFEST.json` are GENERATED from the module instantiation
graph of these files; do not hand-edit them.
