# Systolic-array design space — sourced survey

What is worth *searching* in a TPU-like design space, and what is not. Distilled
from a rescued survey; citations kept, they are the value. **[P]** primary,
**[S]** secondary (summary of a primary), **[Inf]** inference.

## 1. Real TPU parameters

TPU v1 — Jouppi et al., ISCA 2017, arXiv:1704.04760 [P]:

| | |
|---|---|
| MXU | 256×256 8-bit MAC array = 65,536 ALUs [P] |
| Peak | 92 TOPS int8 (65,536 × 2 × 700 MHz) [S] |
| Unified buffer | 24 MiB dual-banked SRAM [S] |
| Accumulators | 4 MiB = 4096 entries × 256 × 32-bit [S] |
| Weight FIFO | 4 tiles deep, hides weight shift-in behind compute [S] |
| Buses | ~256 B wide; accumulator bus 1024 B [P] |
| Clock / process / TDP | 700 MHz / 28 nm / 75 W nameplate [S] |
| ISA | ~a dozen CISC ops with a repeat field, 5 do the work; CPI 10–20, `MatrixMultiply` being a macro-op (B×256 · 256×256 ≈ B pipelined cycles) [P]/[S] |

**Ridge point ≈ 1350 MACs/byte of weight read** [P], against ≈13 ops/byte for
contemporary Haswell and ≈9 for a K80. With the ridge that far right, most of
Google's production inference sat on the **memory-bound slant** (MLPs and LSTMs
bandwidth-bound, CNNs compute-bound), so the paper concludes that for that mix
more MXU would have helped less than more bandwidth: DDR3→GDDR5 lifts the
roofline for ~10 % die area [S on the 10 %].

Later generations, [P] jax-ml.github.io/scaling-book/tpus/ unless noted:

| | v2 | v3 | v4 |
|---|---|---|---|
| Datatype | bf16 mul / fp32 acc (bf16 debuts) | bf16 | bf16 |
| MXU | 128×128 | 128×128 | 128×128, 2× as many |
| Peak bf16 | ~45 TFLOPS | ~140 TFLOPS | ~275 TFLOPS |
| HBM | — | 32 GB @ 900 GB/s | higher (v5p 2.8 TB/s) |
| ICI | 2D torus, 512-chip pods | 2D torus 32×32, 1024 | **3D torus 16×16×16**, 4096 |
| Cooling | air | liquid (first gen needing it) [S] | liquid |

**Array size is non-monotone in the real world:** v1 256×256 → v2–v5 128×128
(*several* MXUs per core, not one bigger array) → v6e "Trillium" and TPU7x back
to 256×256. Smaller arrays mapped better for four generations, then the tradeoff
reversed [P for the sizes, [Inf] for the why].

## 2. Dataflow taxonomy — Eyeriss, Chen/Emer/Sze, ISCA 2016 [P]

| dataflow | resident in PE | wins | loses |
|---|---|---|---|
| Weight-stationary | one weight, its reuse window | weight-read energy at high reuse | batch-1 / low reuse: PEs reload |
| Output-stationary | one output's partial sum | psum traffic, no accumulator spill | pays NoC broadcast; idles when output tile < array |
| Row-stationary (Eyeriss) | filter row + input row + psums | balances all three reuse types; minimizes *total* DRAM/NoC/RF movement energy | convolution-shaped, awkward for dense matmul |

Key result [P]: dataflow moves measured energy efficiency by **more than an
order of magnitude** across CNN layers, and **no stationary dataflow wins across
all layer shapes**. Dataflow × array size × buffer capacity is one joint,
shape-dependent tradeoff, not three separable axes.

## 3. Gemmini DSE — closest prior work [P] PMC10007457

Swept array size (8/16/32/64 square), dataflow (WS vs OS), hardware im2col, with
scratchpad (4×256 KB) and accumulator (2×64 KB) held **fixed** — so it says
nothing about buffer capacity, bit width or DRAM bandwidth. Metric: perf/area,
defined as speedup over a Rocket scalar core ÷ CLB LUTs.

- **Small shapes invert everything:** on 2×2 GEMMs the plain Rocket scalar core
  beat *every* Gemmini config, 64×64 being up to **6× slower than the CPU**.
  Doubling array dimensions cost **3.3×** area/power for diminishing speedup.
- **Dataflow rank-inverts with size:** WS ≈ **2× faster than OS on small
  matrices**, and the ranking **flips at 128×128**, where the output tile
  completes in-array and weight reloads stop paying.
- **16×16 was the best perf/area point** tested, though 64×64 had the highest
  absolute speedup.
- im2col: 1.1× for +1.01×/+1.06× area/power — rejected by the study; the
  template for a small monotone gain at real cost.

Corroborating: "Design-Space Exploration of Systolic Array for Edge Inferencing"
[P] ACM AI-ML Systems '24 — PE utilization *falls* as the array grows on
irregular shapes: "the execution time does not decrease with an increase in SA
size… an optimal size lies between the two extremes," and that optimum **shifts
larger with more memory bandwidth**. "Scale-out Systolic Arrays" (Yüzügüler et
al., arXiv:2203.11540) [P] makes the many-small-arrays case.

Tooling: **Timeloop** (Parashar et al., NVIDIA, ISPASS 2019) [P], joint
architecture+mapping sweeps; **MAESTRO / GAMMA** (Georgia Tech), analytical cost
model + GA; DOSA / Polaris / CSDSE / DEAP (2023–25) converge on **EDP** and
**perf/area** as the accepted scalarizations.

## 4. Axes worth searching — each has a real crossover

Array dimension (rectangular allowed); dataflow, swept *crossed* with array size
and never alone; scratchpad capacity vs DRAM traffic; precision, reporting
cycles *and* synthesized area, since pack/unpack logic is not free in HLS;
instruction granularity CISC vs RISC, where dispatch overhead is proportionally
larger the smaller the array; vector width vs array width, a matching problem
that wastes in either direction; accumulator depth, whose knee is tied to array
*latency* not throughput (v1: 4096 entries, double-buffered); and scratchpad↔
array bus width independent of capacity — the roofline axis, wasted on small
arrays and binding on large ones.

## 5. Axes that are monotone — do not spend search budget

Clock frequency alone (better until a hard timing failure — a feasibility
cutoff, not a curve; read it out as a side effect of structural change); opcode
count added without removing any and without a front-end bottleneck; DRAM
capacity at fixed benchmark size (neutral until the working set exceeds it, then
binary); scalar front-end pipeline depth alone (Fmax until a hazard wall, a tool
artifact); SIMD lane count in isolation from VRF depth and array width; more
accumulator/scratchpad ports without a contention benchmark; register file size
with no register-pressure benchmark.

## 6. Objective and benchmark shapes

Cycles alone degenerates to "make the array as big as fits." Report a **Pareto
frontier** of (cycles, LUT/DSP/BRAM, power); if one scalar is needed, use
**perf/area** or **EDP**, and log the frontier anyway.

Benchmarks must span shapes that tile evenly *and* shapes with awkward
remainders (17×17, or 100×100 on a 32×32 array); one GEMV/batch-1
(memory-bound) and one large square (compute-bound); one large GEMM *and* a run
of small back-to-back ones (separating dispatch overhead and accumulator depth
from throughput); and every shape at every precision, since the result of
interest is whether the crossover *shifts*.
