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
| 32x16x16(Kt=4, Nt=4) | 88      | -   | **hangs** (queue depth, below) |

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

These remain **estimates with a caveat**: the FIFOs are sized from the program
(below), so the area figure is not one a real build would have.

## The queue-depth problem: unresolved, with four theories disproved

The required stream depth grows with the program -- 16 at 4x4x4, 64 at 8x8x8,
256 at 16x16x16 -- so `QD` is sized from `NPROG` to make the design run. **That
is a workaround, not a fix.** Depth proportional to the program means
back-pressure never fully engages, and it is why no area number below should be
read as final: FIFOs sized to the program are not a real machine.

Four theories, each tested and **disproved**, recorded so they are not re-tried:

| theory | test | result |
|---|---|---|
| the PE's `put` order closes a cycle through the drainer's fixed read order | swap the two puts in `microarch_ws.py` | minimum depth stayed exactly 32 -- **no effect** |
| a cycle in the process graph deadlocks | `repro_cycle.py`, and `repro_cycle2.py` with real traffic on every edge of a `mem -> use -> acc -> mem` loop | **both ran** -- a cyclic region is fine |
| the `vld` burst (`Kt*M` words into one channel) is the cause | chunk it to `VLD_CHUNK=8` rows per instruction | 8x8x8 at QD=16 **still hung** |
| the sequencer's broadcast puts every unit in a cycle with it | rewrite control as a forwarding chain, each unit forwarding before executing | exact at the same depths, but **required depth unchanged** |

The chain is kept anyway -- control following the data path is the better
structure, and it is how a real decode pipeline is built -- but it is recorded
here as *not* the fix.

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
export PYTHONPATH=/home/sk3463/allo OMP_NUM_THREADS=8

TPU_M=4 TPU_K=4 TPU_N=4               python bench_isa.py simulator   # exact
TPU_M=8 TPU_K=8 TPU_N=8 TPU_QD=64     python bench_isa.py simulator   # exact
```

## Next

1. **The queue depth.** Four theories are dead (table above), so the next step
   is instrumentation rather than another guess: a per-channel occupancy trace,
   or a simulator deadlock report naming blocked processes and full channels.
   Everything else is gated on this -- an area or cycle number measured with
   FIFOs sized to the program would not mean anything.
2. `cosim` -- the only thing that yields a real cycle count for a programmable
   design, for the reason in the synthesis section above.
3. A Gemmini int8 DIM=4 build so the comparison is dtype- and mesh-matched;
   `allo_cmp.c` needs no source change since it is written against `elem_t`.
4. Scale T to 16 to match Gemmini's int8 default mesh.
