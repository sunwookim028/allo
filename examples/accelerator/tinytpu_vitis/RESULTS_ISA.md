# TinyTPU-isa: instruction-programmable tiled GEMM in grid Allo

`microarch_isa.py`. This is the machine the requirements name: an ISA, a vector
unit, a SIMD scratchpad, vector registers streaming to the array's ports, and
tiled GEMM as a *program*. Status is honest below -- part works and is exact,
part does not yet scale.

## What it is

```
        imem ─► sequencer ──(decoded 64-bit word)──► every unit
  A,B ──► dma_ld ─► spm[spad] ─► vru[vr] ─► 4x4 WS array ─► accu[ar] ─► dma_st ─► C
```

Seven units, `T*T + 6 = 22` concurrent processes:

| unit | owns | does |
|---|---|---|
| `sequencer` | `imem` | fetch, decode, broadcast; consumes nothing |
| `dma_ld` | `A`, `B` | DRAM -> scratchpad, packing T lanes/cycle |
| `spm` | `spad` | the scratchpad; **pure SIMD access** |
| `vru` | `vr` | operand vregs; drives the array's ports |
| `pe` x16 | its lane | weight-stationary MAC, decodes nothing |
| `accu` | `ar` | accumulator vregs **and the vector ALU** |
| `dma_st` | `C` | accumulator -> DRAM, clipped |

**The ISA** (one 64-bit word, 6-bit opcode + five fields):
`nop`, `dma_ld`, `vld`, `mm`, `vadd`, `vrelu`, `mvout`.

**`vadd` is load-bearing.** `mm` computes the psums of *one* k-tile; summing
across k-tiles is an explicit `vadd`. Delete it and tiled GEMM stops working --
it is not decoration. This is Gemmini's split too: the adds live in
`AccumulatorMem`'s write path, not in the mesh.

**Pure SIMD scratchpad.** A row of `spad` *is* one `UInt(T*8)` packed word of T
int8 lanes; there is no way to address a lane. That is what lets a
single-ported memory feed T lanes per cycle, and it satisfies `HLS 200-779`
(single reader, single writer) without a pragma.

**The PEs decode nothing.** A header word leads every instruction down the same
chain the weights use, carrying `is_mm` and `nrows`. No command fan-out to T*T
PEs, no opcode in the array -- Gemmini is the same, its PEs are dumb and
`ExecuteController` decodes. The instruction set can grow without touching the
array.

**Chains, not fan-out.** Every distribution is a daisy chain carrying packed
words (`wcol[i] -> wcol[i+1]`, the bottom row packing psums as they travel
east), which is upstream's `test_multi_cache_gemm.py` idiom.

**Hazards come from being in-order.** Each unit consumes the instruction stream
in order and every channel is point-to-point, so `vld` before `mm` before `vadd`
before `mvout` is enforced by construction. Gemmini spends a 48-entry
reservation station to get out-of-order issue on top of this; this design is
strictly in-order and does not pretend otherwise.

## Status

| shape | instrs | vadd runs | result |
|---|---|---|---|
| 4x4x4   (Kt=1, Nt=1) |  6 / 7  | no  | **exact**, `wrong=0/16` |
| 8x8x8   (Kt=2, Nt=2) | 20 / 22 | yes | **exact**, `wrong=0/64` |
| 8x16x8  (Kt=4, Nt=2) | 40 / 42 | yes | **exact**, `wrong=0/64` |
| 16x16x16(Kt=4, Nt=4) | 76 / 80 | yes | **exact**, `wrong=0/256` |
| 32x16x16(Kt=4, Nt=4) | 88      | yes | **exact**, `wrong=0/512` |

Both `gemm` and `gemm.relu` are bit-exact against a numpy reference *including
the int8 clip* at every passing shape. Kt>=2 is the interesting case: it is the
first one where `mm` alone cannot finish a tile and `vadd` actually runs.

8x8x8 is the first shape that exercises the whole ISA -- multiple k-tiles, so
`vadd` actually runs, and multiple n-tiles, so the accumulator is reused. Both
`gemm` and `gemm.relu` are bit-exact against a numpy reference including the
int8 clip.

## Synthesis, and why a programmable design's csynth number is not a cycle count

`vitis_hls` at 8x8x8: **0 errors**, `dataflow` at the top, all 16 PEs
instantiated as separate modules. But the first run reported a top-level
latency of **91407 cycles**, against 74 for the fixed-function
`microarch_ws.py`, and the report says why:

```
o VITIS_LOOP_300_1   Trip = 1023   Pipelined = yes
o VITIS_LOOP_474_4   Trip = 2049
```

Nothing is slow. **Vitis bounds a runtime-bounded loop by the range of its
index**, and the row count was arriving in a 12-bit instruction field, so it
assumed up to 4095 rows per instruction. The number is a worst-case bound
derived from the *encoding*, not a property of the machine.

That is a real consequence of programmability, and it is worth stating plainly
as an evaluation finding: for a fixed-function design, csynth's interval *is*
the answer (`microarch_ws.py` hit its roofline exactly and could be checked
statically). For an instruction-programmable design, **trip counts are data**,
so static estimates become bounds and only `cosim` gives a cycle count. Any
comparison against Gemmini's measured `rdcycle` has to be a cosim comparison.

The bound is still worth tightening, because it is the ISA's fault rather than
the tool's: `nr` is now a dedicated **7-bit** row-count field (`MAXROWS = 127`)
used by every instruction, instead of a 12-bit general field, and the header
that carries the row count into the array was narrowed to match. Gemmini does
the same -- its mvin/mvout carry an explicit bounded row count.

**Narrowing one field, with no change to any datapath, moved the bound 22x:**

| | before (12-bit count) | after (7-bit `nr`) |
|---|---|---|
| top-level latency | 91407 | **4133** |
| top-level interval | 90333 | **3059** |
| reported trip counts | 1023, 2049 | **63** |

Re-verified exact at 4x4x4, 8x8x8 and 16x16x16 after the change. Per unit at
8x8x8 (`logs/csynth_isa_8x8x8.rpt`), 0 errors, `dataflow`, 22 processes:

```
| + tinytpu_isa*  | latency 4133 | interval 3059 | dataflow | BRAM 42 | DSP 12 | FF 16247 | LUT 21918 |
|  + sequencer_0  |         24   |               |          |
|  + dma_ld_0     |       1519   |               |          |
|  + spm_0        |       2034   |               |          |
|  + vru_0        |       1866   |               |          |
|  + pe_0_0       |       1629   |  ... 16 PEs, each its own module
|  + accu_0       |       3058   |  <- the critical unit
|  + dma_st_0     |       1497   |               |          |
```

`accu_0` at 3058 is the whole top-level interval, so the vector unit -- not the
array -- is the thing to optimize next. DSP is 12 rather than the 96 of the
int8 `microarch_ws.py` build because int8 multiplies mapped into LUTs here;
that is a mapping difference, not a missing array (all 16 `pe_i_j` modules are
present in the report).

The FIFOs are a constant depth 8, so unlike earlier revisions of this file the
area figure is one a real build would have.

## Cosim: a real cycle count

The flow is wired in `cosim.py` and **passes** at 8x8x8:

```
+----------+--------+------------------+
|   RTL    | Status | Latency (cycles) |
|  Verilog |  Pass  |       1449       |
+----------+--------+------------------+
TB: 8x8x8 gemm mismatches = 0 / 64
C/RTL co-simulation finished: PASS
```

(That 1449 was measured with `imem` declared `[1024]`; sized to the program it
is **1003**. See the fixed-cost note below.) Final measured numbers, all exact:

| shape | cosim cycles |
|---|---|
| 4x4x4    |  717 |
| 8x8x8    | 1003 |
| 16x16x16 | 2395 |

The matched Gemmini comparison and the gap decomposition are in
`COMPARISON.md`. In short: identical fixed cost (573 vs 539 cycles), but 24.0
cycles/instruction against Gemmini's 5.9, and the report names why -- every
unit's per-instruction loop is `pipelined = no`.

**1003 cycles at 8x8x8, int8, on a 4x4 array**, and the RTL is functionally
exact against the same numpy reference the KPN simulator uses. `csim` passes
too, which is the first functional check of the *emitted HLS code* rather than
of the Allo simulator's interpretation of the design.

Three things had to be built or fixed to get here, none of them design changes:

1. **A real testbench.** `df.build(target="vitis_hls", mode=...)` handles `csim`
   and `csyn`; anything else routes to the `XDEVICE` Makefile flow, and the
   emitted `host.cpp` is an OpenCL/XRT host. Cosim needs a plain C++ `main`
   calling the top directly, so `cosim.py` generates one -- from the same
   `gemm_program()` and numpy reference the simulator uses, so the vectors
   cannot drift from the design.
2. **`-B/usr/bin`.** Vitis 2023.2 ships binutils 2.37, which cannot read this
   system's glibc: `unknown type [0x13] section '.relr.dyn'`, then
   `cannot find libm.so.6`. Both the csim and cosim links fail without pointing
   the compiler driver at the system linker (2.42).
3. **Explicit `m_axi` depths.** `A depth specification is required for MAXI
   interface port 'gmem0' for cosimulation` -- cosim has to know how much memory
   to model behind each port, and Allo emits the pragmas without a depth, so
   `cosim.py` patches them in the port order of the top function.

One incidental finding worth acting on: cosim compiles an
`AESL_deadlock_detect_unit` into the RTL testbench, so **Vitis cosim already has
the deadlock reporting the Allo simulator lacks** -- which makes it a useful
oracle for the class of bug the next section is about.

## The FIFO problem, exactly: it was the simulator's thread count

**Resolved, and it was not the design.** Earlier revisions of this file called
this "the design's one unresolved problem" and sized `QD` from the program to
work around it. That was wrong, and here is the actual answer.

**The design's channel graph needs depth 4.** `kpn_model.py` models the exact
channel structure -- every unit as a generator yielding blocking `get`/`put`,
bounded FIFOs, a cooperative scheduler, and a deadlock report naming each
blocked process and the occupancy of the channel it waits on. It is the
instrumentation the Allo simulator does not provide. It completes at **depth 4
for every shape**, 4x4x4 through 16x16x16. So there is no circular wait in the
architecture at all.

**The Allo simulator needs one thread per process.** It appears to give each
`df.kernel` instance an OMP thread and to block that thread on an empty or full
stream. With fewer threads than processes, a blocked process can hold a thread
that its own producer needed, and the region wedges. This design has
`T*T + 6 = 22` processes. At 16x16x16 with `QD=16`:

| `OMP_NUM_THREADS` | result |
|---|---|
| 8  (the value in `CLAUDE.md`) | **hang** |
| 16 | **hang** |
| 24 | pass |
| 32 | pass |

The threshold sits exactly at the process count. And with 32 threads the depth
requirement disappears: **16x16x16 passes at `QD=4`** -- matching the model --
and **32x16x16, which had never passed at any depth, passes at `QD=8`**.

So the "required depth grows with the program" law was an artifact throughout.
Deep FIFOs were masking a thread-starvation deadlock by letting each producer
run to completion before anyone had to block; the more instructions, the more
buffering that took. `QD` is now a constant **8**.

Two consequences worth carrying elsewhere:

- **`CLAUDE.md`'s `OMP_NUM_THREADS=8` is not a safe default.** It is fine for
  the small regions in `tests/dataflow`, but a design with more processes than
  threads can deadlock with no diagnostic. The rule is
  `OMP_NUM_THREADS >= number of kernel instances`.
- **The simulator should say this.** The symptom is a silent hang; there is no
  message, no indication of which process is blocked on which channel. A
  deadlock report -- or simply a warning when a region has more processes than
  threads -- would have saved this entire investigation. `kpn_model.py` shows
  the report is about thirty lines of bookkeeping.

**Four theories I tested and disproved along the way**, kept so they are not
re-tried: the PE's `put` order (swapping changed nothing), a cycle in the
process graph (two repros, one with real traffic on every edge, both ran), the
`vld` burst length (chunking did not help), and the sequencer's control
broadcast (rewritten as a forwarding chain; exact, depth unchanged). The
forwarding chain is kept because control following the data path is the better
structure, not because it fixed anything.

## What did fix a real deadlock, twice

Both fixes were the same shape, and both match Gemmini's structure:

**1. One unit doing DMA in both directions deadlocks.** A single `dma` unit put
`spm` in a two-process cycle (`dma -> dma2sp -> spm` *and*
`spm -> sp2dma -> dma`). The round trip `dma_ld; dma_st` hung with **every**
body variant tried -- constant trip counts on both sides, no `meta_for`, no
store, a constant put. Splitting into one-way `dma_ld` and `dma_st` units fixed
it immediately and exactly. Gemmini splits the same way
(`LoadController.scala` / `StoreController.scala`).

**2. Writing results back into the input scratchpad deadlocks.** The original
`vst` (accumulator -> scratchpad) hung even when its body was reduced to
`ac2sp.put(123)` -- no `ar` read, no packing, no clipping -- and with
compile-time trip counts on both sides, and at `QD=256`. The put side alone
ran; the get side alone ran; together they hung. I could not reduce this to a
minimal repro (see the two disproved guesses above), so the mechanism is still
unexplained.

The fix was architectural: **make the accumulator the output memory.** `mvout`
reads `ar` and writes DRAM directly, the scratchpad becomes input-only, and the
back edge disappears. This is exactly Gemmini: `AccumulatorMem` is a separate
memory from the scratchpad, and `StoreController` reads it directly -- results
never re-enter the input scratchpad. It also collapsed two instructions
(`vst` + `dma_st`) into one.

That an architecture-level property (where the output memory lives) is what
makes a design runnable, while the symptom is an unexplained silent hang, is the
strongest argument yet for a deadlock report in the simulator: naming the
blocked processes and the full channels would have replaced a long bisection
with one run.

## Reproducing

```bash
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build   # not set by the env
export PYTHONPATH=/home/sk3463/allo
export OMP_NUM_THREADS=32     # >= 22 processes; 8 deadlocks, see above

TPU_M=8  TPU_K=8  TPU_N=8  python bench_isa.py simulator   # exact, QD=8
TPU_M=32 TPU_K=16 TPU_N=16 python bench_isa.py simulator   # exact, QD=8
python kpn_model.py                                       # channel-graph model
python cosim.py                                           # csim + csynth + cosim
```

## Row-flattening: every unit is one loop now, except the accumulator

Each unit was a loop over *instructions* containing a loop over *rows*, and
Vitis reported `Pipelined = no` on all five of those outer loops. That is not a
tool defect: modulo scheduling needs a fixed II, an inner loop whose trip count
arrives in an instruction field cannot be unrolled to give one, and so the outer
loop has no II at all.

`dma_ld`, `spm`, `vru` and `dma_st` are now **one flat loop over rows** (words,
for `vru`), with the instruction fetched on the iteration that needs it and the
opcode test surviving as a mux inside a pipelined body. The header carries a
per-unit dynamic *work* count instead of an instruction count; `assemble()`
already expanded the control flow with `expand()`, so this is `nr` summed
rather than counted.

| unit | before | after |
|---|---|---|
| `sequencer` | 1 loop, II=5 | unchanged |
| `dma_ld` | iter latency 133, `Pipelined = no` | one loop, **yes, II=1** |
| `spm` | 133, `no` | one loop, **yes, II=1** |
| `vru` | 137, `no` | one loop, **yes, II=1** |
| `accu` | 261, `no` | **unchanged** |
| `pe` x16 | 2055-2058, `no` | unchanged |
| `dma_st` | 132, `no` | one loop, **yes, II=1** |

Cosim, one build, all shapes bit-exact:

| shape | before | after |
|---|---|---|
| 4x4x4    | 1017 | **1004** |
| 8x8x8    | 1145 | **1108** |
| 12x12x12 | 1369 | **1294** |
| 16x16x8  | 1417 | **1344** |
| 16x16x16 | 1713 | **1586** (1.08x) |

Marginal cost 18.07 -> 15.12 cycles/instruction; fixed cost 902 -> 907, i.e.
unmoved. Area: BRAM 16, DSP 15, FF 12432 -> 12835, LUT 18416 -> 18968.

**The 1.7x this was estimated at was the ratio of our marginal cost to
Gemmini's, not the headroom in the change**, and at 16x16x16 the marginal term
is only 47% of the cycles. `COMPARISON.md` has that arithmetic and the two
Vitis-specific tricks (increment the row counter at the *top* of the body; read
the memory *once*, at an address the branch selects) that were each worth a
factor of two in II.

**What resisted.** The plan was per-opcode queues and one process per opcode,
and one owner per memory rules that out for every unit where it would have
mattered: `spad`, `vr` and `ar` each have arms on both sides. `accu` would not
even flatten -- with `ar` in BRAM the flat loop is `Final II = 3`, because `r`
stops being affine in the loop index and Vitis can no longer prove that
iteration *n*'s store and *n+1*'s load differ; with `ar` completely partitioned
into registers it is II=2, and no further, because read-add-write into a
register file is a real recurrence. Both versions were built and bit-exact and
both were slower than the nested loop, which keeps `mm` at II=1. The array was
left alone on purpose: folding a PE's per-`mm` prologue into its MAC body would
set the II of the one loop in the design that already runs at 1.

## The burst DMA: the 907-cycle fixed charge, and what it actually was

The fixed term had not moved all project: 907 cycles, 57% of 16x16x16 and 90%
of 4x4x4, and row-flattening left it at 907 from 902. It was not a DMA cost at
all -- it was an argument-passing convention. `wrap_io=True` makes Allo hoist
every `m_axi` argument into a local buffer before the region starts, via
`wrap_data_movement` (`allo/ir/transform.py`), whose extent is
`MemRefType(arg.type).shape` -- the STATIC type, with no offset and no length.
The csynth report names all four copies:

```
| m_axi_gmem0 | read  |  56 | 64 | l_S_load_buf0_load_buf0_l_0   |   # imem
| m_axi_gmem1 | read  | 256 |  8 | l_S_load_buf1_load_buf1_l_0   |   # A
| m_axi_gmem2 | read  | 256 |  8 | l_S_load_buf2_load_buf2_l_0   |   # B
| m_axi_gmem3 | write | 256 |  8 | l_S_store_res3_store_res3_l_0 |   # C
```

824 words at II=1, copied whether the program touches them or not.

**The earlier verdict on `wrap_io=False` was wrong, and wrong in an instructive
way.** It measured fixed 481 / marginal 39.8 and concluded that `m_axi` is
inherently slow. It was measured with the strided access pattern, and the two
patterns synthesize to completely different hardware:

| pattern | `[HLS 214-115]` says | loop II |
|---|---|---|
| `lA[(f1 + r) * MAXDIM + f2 * T + e]`, `e` unrolled | `burst reads of length 4 and bit width 8` | 4 |
| `imem[NHDR + pc * IWORDS]`, `pc` a register | `burst reads of length 2 and bit width 64` | 13 (was 5) |
| `for i in range(n): b[i] = lA[i]`, `n` a runtime value | `burst reads of variable length` | port-limited |

So Allo could express a program-controlled burst DMA the whole time. Two
changes, both to access patterns rather than to Allo:

* **`sequencer` prefetches the program.** One `IMEM_SIZE`-word contiguous burst
  at start-up, then every fetch is a BRAM read. The fetch loop goes back to
  II=5. This was most of the old 39.8 -- ~76 dynamic sequencer iterations each
  paying full bus latency for two words.
* **`dma_ld` bursts each operand matrix once**, covering exactly the DRAM rows
  the program will name. The spans come from the assembler: `expand()` now
  resolves the AGU exactly as the sequencer does and `assemble()` takes the
  maximum `f1 + nr` over the `dma_ld`s of each source into `imem[7]`. The bytes
  are packed into `UInt(T*8)` words as the burst sweeps, so the instruction
  loop does one BRAM read per row and no packing at all, back at II=1.

Resolving the AGU in `expand()` also made `bench_isa`'s loop-vs-flat check
strict: the two program forms must now agree on every resolved address field,
not just on the opcode and row-count stream. They do, at all five shapes.

### Measured

Bursts inferred, which is the thing to check -- a change that does not change
the 214-115 message has not worked:

```
| m_axi_gmem0 | read  | 56       | 64 | l_S_i_0_i        |   # imem, one burst
| m_axi_gmem1 | read  | variable |  8 | VITIS_LOOP_596_3 |   # A,    one burst
| m_axi_gmem2 | read  | variable |  8 | VITIS_LOOP_656_4 |   # B,    one burst
| m_axi_gmem3 | write | 4        |  8 |                  |   # C,    unchanged
```

Per-unit `Pipelined` / II, one csynth each, `wrap_io=True` before against
`wrap_io=False` + bursts after:

| unit | loop | before | after |
|---|---|---|---|
| `sequencer` | fetch/dispatch | yes, II=5 | yes, **II=5** |
| `sequencer` | imem prefetch | (hoisted, II=1) | yes, **II=1**, trip 56 |
| `dma_ld` | operand prefetch | -- | yes, **II=4** (8-bit port) |
| `dma_ld` | instruction rows | yes, II=1 | yes, **II=1** |
| `spm` | rows | yes, II=1 | yes, II=1 |
| `vru` | words | yes, II=1 | yes, II=1 |
| `pe` x16 | per-`mm` | no, iter 2058 | no, iter 2058 |
| `pe` x16 | MAC | yes, II=1 | yes, II=1 |
| `accu` | per-instruction | no, iter 261 | no, iter 261 |
| `accu` | `mm` / `vadd` / `vrelu` / `mvout` | II=1 / 2 / 2 / 1 | unchanged |
| `dma_st` | rows | yes, II=1 | yes, **II=4** |

**Nothing lost its pipelining.** `dma_ld`'s instruction loop keeps II=1 because
the burst is a separate loop outside it -- a conditional prefetch inside the
flat body would have put a runtime-bounded inner loop back in and taken the
whole loop out of pipelining, which is the trade row-flattening had bought.
`dma_st` is the one regression, II 1 -> 4, and it is the bus rather than the
schedule: it now writes `C` over `m_axi` four bytes at a time instead of into a
buffer someone else stores back.

M_AXI table, unchanged by any of this: all four ports 8/64-bit, Max Read and
Write Burst Length 16, Num Read and Write Outstanding 16. A 56- or 256-beat
burst is issued as requests of 16, at II=1 each.

Cosim, one build, all five shapes bit-exact:

| shape | dyn. instrs | before | after | |
|---|---|---|---|---|
| 4x4x4    |  6 | 1004 | **680**  | 1.48x |
| 8x8x8    | 15 | 1108 | **831**  | 1.33x |
| 12x12x12 | 28 | 1294 | **1066** | 1.21x |
| 16x16x8  | 25 | 1344 | **1139** | 1.18x |
| 16x16x16 | 45 | 1586 | **1457** | 1.09x |

Least squares against dynamic instruction count over the five shapes:
**fixed 907 -> 557, marginal 15.12 -> 20.07 cycles/instruction.**

That is a trade, and it is stated as one. The crossover is at
`350 / 4.95 = 71` dynamic instructions, past the longest program this MAXDIM
admits (45), so the burst build wins everywhere it can be run -- but at a larger
MAXDIM it would not, without also fixing what the marginal term buys.

### Two negative results worth the space

* **Halving the operand burst bought nothing.** Merging the A and B bursts into
  one loop bounded by `max(na, nb)` does halve the burst time -- they are
  separate bundles, and Vitis attributes a variable-length burst on *each* to
  the one loop -- and it changed the cosim count at all five shapes by exactly
  **zero cycles**. The operand burst is already entirely hidden behind the
  sequencer's dispatch and the units' fill. It was built, measured, and reverted
  to the version that reads the fewest bytes.
* **Port widening is blocked in Allo, not in Vitis.** The `m_axi` ports are 8
  bits, so even a perfect burst moves one byte per cycle;
  `config_interface -m_axi_max_widen_bitwidth 512` is the fix and it does
  nothing: `[HLS 214-307] Could not widen since type i8 size is greater than or
  equal to alignment 1(bytes)`. Allo emits the argument pointers with no
  alignment attribute, so Vitis must assume 1 byte and refuses. Closing that in
  the emitter is worth roughly the whole operand-traffic term.

### What is left, and it is not the fixed term any more

`dma_st` is most of the +5 cycles/instruction. Bursting it contiguously has to
either clobber the columns the program never named -- the accumulator holds one
column block per `mvout`, so a whole-row write-back invents the rest -- or
defer the write-back to the end of the run, where it serializes behind the last
`mvout` instead of overlapping the compute it currently overlaps. Both cost
something real, so it is the next thing to *measure*, not to assume. Fixed 557
against Gemmini's 483 is 1.15x; marginal 20.1 against 10.8 is 1.9x. After a
project spent arguing that the fixed term was the only problem, it is not.

## The loop levels are derived now, not typed (`isa_dsl.py`)

`gemm_program` was hand-emitting its AGU *levels*: a comment reading "level 0
is nb, level 1 is kb" and four `enc_agu((AGU_F1, 0, MAXDIM), ...)` calls that
had to agree with where the `loop`/`endloop` pairs happened to sit. That is the
failure mode MiniTPU's kernels have -- theirs carry
`level = (1 if grouped else 0) + 2` and keep a trip-count-1 loop alive so the
number does not shift. The loop structure is stated twice and nothing checks
that the two statements agree.

`isa_dsl.py` ports MiniTPU's mechanism (`board_package/dsl.py:126-141`, its
`loop()`): **nesting depth IS the AGU level**, and the only handle on a level
is the induction variable the `with` yields.

```python
with k.loop(Nt, "n") as nb:                       # level 0, derived
    k.vld(W_VR, Ref(B_SP).at(nb, MAXDIM), rows=T)
    k.mm(A_VR, AR_C, W_VR, rows=M, acc=False)     # peeled: overwrite
    with k.loop(Kt - 1, "k") as kb:               # level 1, derived
        k.vld(W_VR, Ref(B_SP + T).at(nb, MAXDIM).at(kb, T), rows=T)
```

Generalised to our AGU rather than copied from theirs: theirs is one term on
one field as a power-of-two shift, ours is three `(target, level, stride)`
terms with arbitrary 11-bit strides, any number of which may land on the same
field -- `B_SP + nb*MAXDIM + kb*T` is two terms on `f1` and their encoding
cannot say it. So `Ref` is a base plus an ordered list of `(iv, stride)` terms
and any field may be one.

**It emits the identical instruction stream, and that is the whole test.**
`isa_dsl.assert_matches_handwritten` compares both 64-bit words of every
instruction against `gemm_program_handwritten` (kept, unchanged, as the
reference) at all five shapes and both relu settings; `bench_isa.py` runs it
before it builds anything. Bit-identical means cosim cannot move, so it was not
re-run. Bench: ALL EXACT, unchanged.

**What it bought.** Three errors that were previously expressible are not:
a level that disagrees with the nest, an induction variable used after its loop
closed (the sequencer would resolve it against a stale `iv_now[level]`), and a
`loop` whose `endloop` was forgotten. Two more are now caught at the point of
writing rather than at `enc_agu`: a nest deeper than `LOOP_DEPTH`, named
("the nest is n > k > m > j > i"), and a trip count of 0, which the do-while
sequencer would run **once**.

**What it cost, honestly.** It is ~120 lines of machinery to remove four typed
integers from a 40-line program, and at this size the hand-written form was not
actually hard to keep right -- the trade only pays once there is more than one
program. It also trades one hand-maintained correspondence for another: the
instruction methods (`vld(vr, spad, rows)`) restate the field layout that the
opcode table comments give, where the hand-written form wrote `f0=`/`f1=`
against that table directly. The new correspondence is stated once per opcode
instead of once per instruction and the bit-identity assertion checks it, which
is why it is the better of the two, but it is not free. And the genuinely
subtle part of this program -- the peeled first k-tile, and the `B_SP + T` base
that encodes "kb starts at 1" -- is exactly as subtle as it was; the generator
has no opinion about peeling and could not form one.

## Next

1. **The write path.** `dma_st` is the last strided `m_axi` access and most of
   what separates 20.1 cyc/instr from the 15.1 the buffered build reached. Both
   ways of bursting it cost something; measure rather than assume.
2. **An alignment attribute on the emitted argument pointers**, so
   `m_axi_max_widen_bitwidth` can take the 8-bit ports wider. This is an Allo
   codegen change, not a design one.
3. **Re-measure T=16.** The last T=16 number (1176 at 16x16x16) predates both
   row-flattening and the burst DMA, and the burst DMA is the change that
   argument was asking for -- at T=16 a DRAM row is one packed word, so the
   operand burst is the whole matrix with no column overfetch at all.
4. **Then multiple instructions in flight**, which is the remaining structural
   difference from Gemmini's reservation station.
