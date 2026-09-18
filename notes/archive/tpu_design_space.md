# TPU/Systolic-Array Design Space Research — for TinyTPU Co-Design Search

Sources are marked **[P]** primary (paper/vendor doc), **[S]** secondary (blog/course
write-up summarizing a primary source — used only where the primary source's PDF text
could not be extracted directly), or **[Inference]** my own reasoning, not a sourced claim.
Where a secondary source is used to state a number, I flag it; numbers I could confirm
from Jouppi et al. text extraction are marked [P].

---

## (a) Factual reference table: real TPU parameters across generations

### TPU v1 (Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing
Unit," ISCA 2017; arXiv:1704.04760) [P]

| Parameter | Value | Source |
|---|---|---|
| MXU (systolic array) | 256×256 8-bit MAC array = 65,536 ALUs | [P] arXiv:1704.04760 |
| Datatype | 8-bit integer multiply; the paper also mentions the array supports 8×16 and 16×16 multiply modes; accumulation is wider than the multiply datapath | [P]/[S] alastairreid.github.io summary of the paper |
| Peak throughput | 92 TOPS (8-bit) | [S] widely reported, consistent with 65,536 MACs × 2 ops × 700 MHz ≈ 91.75×10¹² ops/s [Inference: arithmetic check] |
| Unified Buffer (on-chip activation/IO memory) | 24 MiB, dual-banked, SRAM | [S] thechipletter.substack.com/p/googles-first-tpu-architecture (Chip Letter deep-dive), corroborated by multiple summaries |
| Accumulators | 4 MiB total = 4,096 entries × 256-element × 32-bit vectors (i.e., one 32-bit accumulator per MXU column, 4096 deep) | [S] Chip Letter / course summaries of the paper's Table/Fig text |
| Weight FIFO | 4 tiles deep (double-buffers weight loads to hide systolic "weight shift-in" latency behind compute) | [S] Chip Letter |
| Databus width | ~256 bytes wide for most buses; accumulator bus 1024 bytes wide | [P] paper text (confirmed via extraction) |
| Clock | 700 MHz | [S] Chip Letter, cross-confirmed by multiple sources |
| Process / die | 28 nm; die size reported as roughly half the size of a contemporary Haswell die | [P] paper's comparison discussion |
| TDP | 75 W (workload-dependent range 28–40 W typical inference reported elsewhere; 75 W is the commonly cited nameplate) — reported inconsistently across secondary sources, treat as approximate | [S] |
| Instruction set | ~a dozen instructions total (CISC-style, includes a repeat/count field so one instruction drives many cycles); five do most of the work: `Read_Host_Memory`, `Read_Weights`, `MatrixMultiply`/`Convolve`, `Activate`, `Write_Host_Memory` | [P]/[S] paper text + summaries |
| Instruction execution model | 4-stage pipeline, one CISC instruction per stage; CPI (cycles per instruction) typically 10–20 because instructions like `MatrixMultiply` are long-running macro-ops (a B×256 × 256×256 matmul takes ~B pipelined cycles) | [P]/[S] |
| Roofline result | Ridge point (operational intensity where compute-bound begins) ≈ **1350 MACs/byte of weight read** for the TPU, vs. ≈13 ops/byte for contemporary Haswell CPU and ≈9 ops/byte for Nvidia K80 GPU. Because TPU's ridge point is so far right, most of Google's production DNN workloads sat on the **memory-bandwidth-bound slanted part of the roofline**, not the compute-bound flat part. | [P] paper's roofline figure/discussion; [S] alastairreid summary |
| Workload characterization | MLPs and LSTMs were memory-bandwidth-bound (small batch, low arithmetic intensity, dominated by weight reads); CNNs were compute-bound (large weight reuse) | [P]/[S] |
| What this implied | The paper explicitly argues that for TPU v1's *actual* workload mix, adding more MXU compute (a bigger array or higher clock) would have helped less than adding memory bandwidth, because most inference workloads never reached the ridge point. They discuss (as a design counterfactual) that swapping the DDR3 weight memory for GDDR5 would substantially lift the roofline for memory-bound workloads at a modest (~10%) die-area cost, more effectively than scaling compute. | [P] paper's "what if" discussion; [S] corroborating summaries — I could not re-extract the exact wording, so treat the "~10% die area" figure as [S]/secondary-sourced, not verified against primary text directly |

### TPU v2 / v3 / v4 (+ pointer to v5/v6/v7 for trend continuation)

| Parameter | TPU v2 | TPU v3 | TPU v4 | Source |
|---|---|---|---|---|
| Datatype | bfloat16 multiply, fp32 accumulate (bf16 introduced here; becomes an industry standard) | bf16 | bf16 (later gens add int8 support back for inference) | [S]/[P]-adjacent: Google Cloud TPU docs, multiple secondary architecture write-ups agree bf16 debuted at v2 |
| MXU array size | 128×128 per MXU | 128×128 per MXU | 128×128 per MXU (2× more MXUs than v3, plus higher clock) | [P] jax-ml.github.io/scaling-book/tpus/ ("How to Think About TPUs", a widely used technical reference by ML systems engineers, cites 128×128 for v3–v5 generation, 256×256 only returning at v6e/"Trillium" and TPU7x) |
| Peak bf16 FLOPs/chip | ~45 TFLOPS | ~1.4×10¹⁴ (140 TFLOPS) | ~2.75×10¹⁴ (275 TFLOPS) | [P]/[S] scaling-book table; v2/v3 numbers also widely reported in secondary architecture surveys |
| HBM per chip | not confirmed precisely here | 32 GB | not confirmed precisely here (v4 boards commonly cited near this range) | [P] scaling-book (v3: 32 GB @ 9.0×10¹¹ B/s) |
| HBM bandwidth/chip | — | ~900 GB/s | higher (scaling continues into v5p at 2.8 TB/s) | [P] scaling-book |
| Interconnect (ICI) topology | 2D torus, pods up to 512 chips | 2D torus (32×32), pods up to 1024 chips | **3D torus** (16×16×16), pods of 4096 chips — major topology shift at v4 | [P]/[S] scaling-book + multiple secondary sources agree on the 2D→3D torus transition at v4 |
| Cooling | air | liquid (first TPU gen requiring it, driven by power density increase) | liquid | [S] secondary sources (Chip Letter / Tom's Hardware Hot Chips coverage) |
| What drove each change | bf16 (v2): training workloads need wider dynamic range than int8 but full fp32 range isn't needed — bf16 gives fp32-like exponent range at half the bits/area/bandwidth cost. HBM (v2 onward): training is far more memory/bandwidth hungry than v1's inference-only workload, so DDR3→HBM. 3D torus (v4): at pod scale, 2D torus interconnect bisection bandwidth becomes the bottleneck for large all-reduce/all-to-all training traffic; 3D torus improves bisection bandwidth and fault-tolerant routing at similar per-link cost. | [Inference], consistent with what secondary sources report as the stated rationale, but the underlying "why" reasoning here is my synthesis, not a verbatim quote |

**Note on the 256×256→128×128→256×256 oscillation:** v1 used 256×256; TPU v2–v5 generations dropped to 128×128 (with *multiple* MXUs per core instead — e.g., 4 MXUs/core on later chips) rather than one huge array; TPU v6e ("Trillium") and TPU7x returned to 256×256. This is a useful, non-obvious real-world data point: **Google did not monotonically grow array size across generations** — they found multiple smaller arrays (better utilization on the tile/mapping shapes chip compilers actually emit, better mapping flexibility) preferable to one bigger array for several generations, then reversed once process/mapping-software maturity changed the tradeoff. [P] jax-ml.github.io/scaling-book/tpus/; [Inference] on the "why," since I did not find a primary Google statement of this specific rationale.

### Dataflow taxonomy (Chen, Emer, Sze; Eyeriss, ISCA 2016) [P] eems.mit.edu/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf

| Dataflow | What stays put in each PE / local register | What it optimizes | Weak point |
|---|---|---|---|
| **Weight-stationary (WS)** | Filter weight held in PE register for its full reuse window | Minimizes weight-memory reads/energy when weight reuse (batch size, output spatial reuse) is high | Poor when weights don't reuse much (large weight, small activation reuse) — under output/small-batch-1 conditions the PE array spends cycles reloading weights |
| **Output-stationary (OS)** | Partial sum for one output element accumulated locally until finished | Minimizes partial-sum read/write traffic (avoids spilling accumulators to memory) | Weight and input reuse must come from broadcast/multicast on the NoC, which costs energy; can under-use PEs when output tile is smaller than the array |
| **Row-stationary (RS, Eyeriss's contribution)** | A 1-D row of a convolution filter and the corresponding row of input activations, plus partial sums, all stay resident in a PE | Balances *all three* reuse types (weight, input activation, partial sum) simultaneously rather than picking one to optimize — Eyeriss's paper shows this minimizes total data-movement energy across DRAM/NoC/RF levels, not just one buffer level | Design is convolution/CNN-shaped (row/sliding-window structure); less natural fit for pure dense matmul, and mapping complexity is higher |

Key finding from Eyeriss [P]: dataflow choice changes measured energy efficiency by **>1 order of magnitude** across CNN layers, and no single stationary dataflow (weight/output/input) dominates across all layer shapes — the "right" dataflow depends on layer shape (batch size, channel depth, spatial size) relative to array size and buffer capacity. This directly supports the framing that **dataflow × array-size × buffer-capacity is a joint, shape-dependent tradeoff**, not decomposable into independently-optimal axes.

### Gemmini DSE case study (empirical crossover data) [P] "Deep Learning Accelerators' Configuration Space Exploration Effect on Performance and Resource Utilization: A Gemmini Case Study," PMC10007457

This is the single most directly relevant piece of prior DSE work for TinyTPU because Gemmini, like TinyTPU, is a small HLS/RTL-generated systolic accelerator studied via a parameter sweep.

- Swept: array size (8×8, 16×16, 32×32, 64×64), dataflow (WS vs OS), presence/absence of a hardware im2col unit. Scratchpad (4×256KB banks) and accumulator (2×64KB banks) were **held fixed** — i.e., this study did *not* explore buffer capacity, bit-width, or DRAM bandwidth as axes.
- Objective/metrics used: execution cycles, speedup vs. baseline Rocket CPU, **"performance-per-area"** (defined as total speedup ÷ total CLB LUTs consumed), plus raw FPGA resource counts (LUTs, registers, BRAM, DSPs), max frequency, and power.
- **Non-monotonic finding #1 (array size vs. shape):** on 2×2 GEMMs, the plain Rocket scalar core beat *every* Gemmini configuration; the 64×64 array was up to **6× slower** than the CPU baseline on tiny matrices, purely from underutilization/setup overhead dominating. Doubling array dimensions cost **3.3×** area/power for diminishing absolute speedup.
- **Non-monotonic finding #2 (dataflow crossover with size):** WS dataflow gave ~**2× speedup over OS on small matrices**; the ranking **flipped** at 128×128 matrices, where OS became faster because the whole output tile completes in-array without weight-reload overhead. This is a genuine crossover, not a monotone result.
- **Sweet spot:** 16×16 was the best performance-per-area point among all configurations tested, even though 64×64 had the highest absolute speedup — a classic "best absolute performance ≠ best efficiency" DSE result.
- im2col hardware unit gave only 1.1× speedup for +1.01×/+1.06× area/power cost — flagged by the study as *not* worth it, a useful "trivial/negative" DSE result pattern (small monotone gain, real cost — the agent should learn to reject axes like this).

### Systolic-array sizing vs. utilization (general literature)

- "Design-Space Exploration of Systolic Array for Edge Inferencing Applications" [P] (ACM AI-ML Systems '24, dl.acm.org/doi/fullHtml/10.1145/3639856.3639858): confirms average PE-array utilization *decreases* as array size increases, when matrix shapes are irregular/small relative to the array (common in early CNN layers with small channel counts) — because in the classic 2D systolic matmul mapping, one array dimension must be ≥ the operand's contracted dimension or the array is only partially filled. The paper explicitly frames array size as having an **optimum**, not a monotone-better relationship: "the execution time does not decrease with an increase in SA size... an optimal size lies between the two extremes," and that optimum **shifts** with available memory bandwidth (higher bandwidth → optimum array size shifts larger).
- "Scale-out Systolic Arrays" (Yüzügüler et al., arXiv:2203.11540) [P]: studies exactly this large-vs-many-small-array tradeoff for GEMM accelerators, motivating splitting one big array into several smaller independently-fed arrays to recover utilization on irregular/small shapes — directly analogous to Google's own v1(256×256)→v2-v5(multiple 128×128)→v6e(256×256 again) trajectory noted above.

### DSE methodology / tooling references
- **Timeloop** (Parashar et al., NVIDIA, ISPASS 2019) [P]: canonical infrastructure for sweeping accelerator architecture + mapping jointly; models a broad space of (array shape, buffer hierarchy sizes, dataflow) via a declarative spec and reports cycles + energy per configuration.
- **MAESTRO / GAMMA** (Georgia Tech): analytical dataflow-cost-model + genetic-algorithm-based DSE, commonly paired with Timeloop-style models.
- Multi-objective DSE papers (DOSA, Polaris, CSDSE, DEAP — all 2023-2025 arXiv) consistently report **EDP (energy × delay)** and **perf/area** as the two dominant scalarized objectives when a single number is needed, with Pareto-frontier (perf vs. area vs. energy) reporting as the credible/publishable alternative to a single scalar.

---

## (b) Ranked design-space axes for TinyTPU (highest priority first)

For each: parameter, realistic range, why the tradeoff is non-obvious (has a real crossover, not monotone), and the benchmark that exposes it.

### 1. Systolic array dimension (currently fixed 4×4)
- **Range to explore:** 4×4 (baseline) up to 32×32 or 64×64, independently on each axis (rows ≠ cols) — i.e., allow rectangular arrays, not just square, since square-only hides an entire axis of the space.
- **Why non-obvious:** Gemmini's own DSE data (above) shows 64× area for one config was *slower than a scalar core* on 2×2 GEMM, and best perf/area landed at a mid-size (16×16), not the largest tested. TPU history itself shows Google moving *away* from one huge array to several smaller ones for four generations before reversing. The crossover depends on (i) how well the benchmark's matrix shapes divide into array-sized tiles, (ii) how fast the scalar dispatch/load-store path can refill the array between tiles, and (iii) on-chip buffer bandwidth to keep the array fed. This is exactly the kind of "bigger is not free" result an LLM-driven search should be able to *discover*, not assume.
- **Benchmark to expose it:** A GEMM sweep across shapes that are exact multiples of candidate array sizes (e.g., 32×32, 64×64) *and* shapes with awkward remainders (e.g., 17×17, 100×100 tiled with 32×32 arrays leaving a partial tile) — utilization crashes on the latter unless the design also handles boundary tiles well. Include at least one very small GEMM (matches TinyTPU's likely real use case for embedded control code) to surface the "small array wins on small problems" crossover.

### 2. Dataflow: output-stationary vs weight-stationary vs a row/tile-stationary variant
- **Range:** implement ≥2 of {OS, WS, RS-like blocked variant}, selectable per-kernel or statically per design.
- **Why non-obvious:** Eyeriss and the Gemmini study both show *rank inversions* — WS wins at small matrix/tile sizes, OS wins at large ones (Gemmini: crossover exactly at 128×128 in their setup); RS wins on data reuse patterns neither pure OS nor pure WS captures well. Because dataflow interacts multiplicatively with array size and buffer capacity, this is a genuinely joint (not separable) axis — sweeping it alone, or array-size alone, misses the interaction that makes the result interesting.
- **Benchmark:** Same GEMM sweep as #1, cross-producted with dataflow choice, to find the (array size, dataflow) crossover curve directly, not just each axis independently.

### 3. On-chip scratchpad capacity vs. DRAM traffic (tiling/blocking pressure)
- **Range:** 8192 words (baseline) up to ~256K–1M words, in several steps (e.g., 8K/32K/128K/512K).
- **Why non-obvious:** Classic roofline/tiling tradeoff — a bigger scratchpad reduces DRAM round-trips (more reuse per byte fetched) but costs SRAM area/power that could otherwise go to more PEs; and past a certain tile size relative to the *working set* of a given GEMM shape, additional buffer capacity stops reducing DRAM traffic at all (the tile already holds the full operand). This produces a genuine knee in the perf/area curve, and the knee's *location* depends on GEMM shape and array size — not fixed. TPU v1's own roofline story (ridge point at 1350 ops/byte, MLP/LSTM being memory-bound) is direct historical evidence that buffer/bandwidth, not raw compute, was the actual bottleneck for realistic workloads.
- **Benchmark:** GEMM shapes spanning a range of arithmetic intensity (batch-1 GEMV-like shapes = low AI = memory-bound, large square GEMM = high AI = compute-bound) run against each scratchpad size, plotting cycles vs. scratchpad size to find the knee, and separately plotting DRAM-word-traffic vs. scratchpad size.

### 4. Precision (currently fixed fp32)
- **Range:** add int8 and/or bf16-equivalent (e.g., a custom 16-bit float or fixed-point) alongside fp32, as a *selectable* per-kernel mode, not a wholesale replacement.
- **Why non-obvious:** Lower precision is not simply "smaller and faster" once accuracy/rounding and *conversion/packing overhead* are accounted for — a design that must pack/unpack 4× int8 values per fp32-wide datapath lane can spend enough control overhead that a naive int8 mode does not deliver a 4× array-density win in practice, especially at small array sizes where per-PE control/muxing overhead is proportionally larger. Because HLS-synthesized designs pay real area/timing cost for that packing logic (Vitis HLS won't give it to you for free), there is a real crossover: at small arrays, precision-splitting overhead can eat most of the theoretical throughput gain; at large arrays, it doesn't. This is directly analogous to the documented ~20–37% area-per-bit-width scaling data point above.
- **Benchmark:** Same GEMM at fixed problem size, same array size, run at fp32 vs int8 vs bf16-equivalent; report cycles *and* synthesized area/DSP/LUT count, to see whether the throughput-per-area gain from lower precision actually materializes at TinyTPU's array-size scale, or gets eaten by control overhead.

### 5. Instruction granularity: CISC macro-ops vs. finer-grained micro-ops
- **Range:** from TinyTPU's current 10 scalar opcodes up toward TPU v1-style "a dozen" macro-instructions where one `MatrixMultiply`-class instruction drives many array cycles (CPI 10–20), vs. a more RISC-like decomposition where the fetch/decode/dispatch scalar front-end issues once per array cycle.
- **Why non-obvious:** TPU v1's actual choice (CISC-style, high CPI-per-instruction, "keep the matrix unit busy" philosophy) exists specifically because scalar fetch/decode/dispatch overhead is *not free* relative to a small systolic array's cycle time — the smaller/faster the array cycle, the more the front-end's overhead matters *proportionally*. This means the "right" granularity is a function of array size and clock, not a fixed answer — coarse macro-ops that look wasteful (poor flexibility, harder compiler mapping) at large scale can be a net win at TinyTPU's small scale, where dispatch overhead is a larger fraction of total cycles. This is a good candidate axis for TinyTPU specifically because its current design already has a scalar fetch-decode-dispatch front end that the search could plausibly bloat or shrink.
- **Benchmark:** A GEMM with many small tiles (stresses dispatch-loop overhead: many decode events per useful array-cycle) vs. one huge GEMM (dispatch overhead amortizes away) — the granularity choice should show a crossover between these two benchmark regimes.

### 6. Vector unit width / vector register file depth relative to array width
- **Range:** currently 8 lanes / 8×8 VRF; explore decoupling these from the systolic array's own width (e.g., wider or narrower than the array), and varying VRF depth (4–32 slots).
- **Why non-obvious:** If the vector unit's width doesn't match the systolic array's column count, either the array is starved feeding activations, or the vector unit becomes the bottleneck packing/unpacking array outputs (e.g., for `Activate`-style post-processing, matching TPU v1's own architecture where the Activation Pipeline sits directly off the array width). This is a matching problem, not a "wider is better" problem — a mismatch in either direction (too narrow OR too wide relative to array) wastes resources or throughput respectively, so it has a real optimum near array-width-matched configurations rather than a monotone curve.
- **Benchmark:** GEMM followed immediately by an elementwise/activation-heavy epilogue (ReLU, bias-add) at varying vector width relative to fixed array width, measuring whether the epilogue becomes the new bottleneck (Amdahl's-law-style shift) at some (array, vector) width ratio.

### 7. Accumulator width/depth relative to array size and pipeline depth
- **Range:** currently implied by scratchpad-only design (no separate accumulator); explore adding a dedicated accumulator buffer, sized/organized as vectors-of-width-equal-to-array-columns (as TPU v1 does), swept in depth (e.g., 64–4096 entries).
- **Why non-obvious:** TPU v1's actual accumulator sizing (4096 entries, later "rounded up... with double buffering") was chosen to hide **systolic pipeline fill/drain latency and weight-reload stalls**, not chosen for peak throughput per se — undersizing it serializes MatrixMultiply issue with accumulator drain; oversizing it wastes SRAM that buys nothing once it exceeds the double-buffering depth needed to hide the pipeline. This produces a genuine knee tied to array *latency* (pipeline depth = array dimension), not array *throughput* — a subtlety a naive "bigger buffer is always better" search would miss.
- **Benchmark:** Back-to-back independent GEMMs (stresses double-buffering/overlap) vs. a single large GEMM (stresses steady-state throughput only) at varying accumulator depth.

### 8. Interconnect/data-supply bandwidth between scratchpad and array (bus width)
- **Range:** currently one flat 8192-word scratchpad feeding a 4×4 array; as array grows (axis #1), sweep the scratchpad↔array bus width/ports independently of scratchpad *capacity*.
- **Why non-obvious:** This is the axis that TPU v1's own paper flags as more valuable than raw compute scaling for their actual workload mix (roofline ridge point argument) — but only up to the point where the array itself becomes the bottleneck again. So there is a real crossover in (array size, bus width) space: for small arrays, extra bus width is wasted; for large arrays with narrow buses, the array stalls waiting for operands. This is the single most direct, literature-grounded "non-monotone" axis available, and it directly interacts with axis #1 (array size) and axis #3 (buffer capacity), which is exactly the kind of multi-way interaction that makes a DSE result credible rather than trivial (see part (e) of the Gemmini/edge-DSE literature above).
- **Benchmark:** Sweep (array size × bus width) jointly on a fixed large GEMM; plot the achieved-vs-theoretical-peak utilization surface to find the ridge/knee.

---

## (c) Axes to AVOID — monotone, and prove nothing

- **Raw clock frequency (holding everything else fixed).** Higher clock is (almost) always better for cycle count until you hit a synthesis/timing wall Vitis HLS reports as a hard failure — that's a binary feasibility cutoff, not an interesting tradeoff curve. Not worth spending search budget on as a standalone axis; if included at all, it should only appear as a *side-effect readout* of other structural changes (e.g., a bigger array lowering achievable Fmax), not as something the agent "chooses" independently.
- **Number of scalar general-purpose opcodes, added without removing any / without a real front-end bottleneck.** Just adding more opcodes for convenience (without exercising axis #5's dispatch-overhead tradeoff) is monotone: more capability, roughly free area cost at HLS scale, no crossover. Only make instruction *count* interesting by tying it to a genuine granularity tradeoff (axis 5), not "add more distinct ops."
- **DRAM capacity (currently 65536 words) as a standalone axis.** Making DRAM bigger while keeping GEMM benchmark sizes fixed and well within the existing 65536-word capacity changes nothing measurable — it's monotone-neutral until the benchmark's working set actually exceeds capacity, at which point it becomes a binary pass/fail rather than a graded tradeoff. Only interesting jointly with benchmark problem *size* scaling, which is a benchmark-design decision, not an architecture-search axis.
- **Pipeline depth of the scalar front-end alone (without touching dispatch semantics).** Deeper pipeline for the scalar decode/dispatch stage, in isolation, monotonically improves Fmax until a hazard/stall wall — again a synthesis-tool artifact rather than an architectural tradeoff with a real crossover.
- **SIMD lane count in isolation from VRF depth and array width (see axis #6 — combined, it's interesting; alone, "more lanes always help until you run out of matching work" is not a surprising finding).**
- **Adding more accumulator/scratchpad *ports* without a corresponding contention benchmark.** More ports monotonically remove structural hazards; only interesting if the benchmark actually creates contention (multiple concurrent array tiles or vector ops) — otherwise it's "more is free, more is better," a trivial finding.
- **Purely increasing register file size/count with no register-pressure-inducing benchmark.** Same shape of triviality as the point above.

---

## (d) Recommended objective function(s) and benchmark shapes

**Objective function.** Do not scalarize to cycle count alone — the DSE and accelerator-architecture literature consistently treats single-cycle-count optimization as the "trivial" regime, because it degenerates to "make the array as big as fits," which is exactly the monotone failure mode described in the prompt. Recommended, in order of preference:

1. **Report a Pareto frontier** of (cycles, synthesized area/resource-utilization from Vitis HLS reports — LUTs/DSPs/BRAM, and if available power/energy) rather than a single number. This is what the DSE-methodology literature (Timeloop/MAESTRO-based studies, DOSA, Polaris) treats as the credible/publishable form of a DSE result, because it makes crossovers/knees visible rather than hiding them behind one scalar's arbitrary weighting.
2. If a single scalar is required for a search algorithm's reward signal, use **performance-per-area** (cycles⁻¹ ÷ resource-utilization, following the Gemmini DSE study's own metric definition) or **EDP (energy × delay)** if a power estimate is available from HLS/synthesis reports — both are standard, well-precedented scalarizations, and both explicitly punish the "just make the array bigger" degenerate strategy that pure cycle-count reward would reinforce.
3. Whatever scalar is chosen, **always additionally log the Pareto set** so that a "found nothing new" run can be distinguished from "found a scalar-optimal point that hides a worse frontier."

**Benchmark shapes.** To make each axis in (b) actually show a crossover rather than a monotone trend, benchmarks should span, at minimum:
- **Shape regime:** (i) shapes that tile evenly into every candidate array size (e.g., multiples of 64), (ii) shapes with awkward remainders relative to array size (e.g., primes or +1 over a power of two) to expose utilization loss, (iii) at least one GEMV/batch-1-like shape (tall-skinny) to expose the memory-bound/roofline regime, and (iv) at least one large square GEMM to expose the compute-bound regime — together these directly instantiate the TPU v1 roofline lesson (MLP/LSTM = memory-bound, CNN-like large square GEMM = compute-bound).
- **Sequence regime:** both a single large GEMM (steady-state throughput test) and a sequence of many small back-to-back independent GEMMs (front-end dispatch overhead / double-buffering test), to separate axis #5 (instruction granularity) and axis #7 (accumulator depth) effects from raw array throughput.
- **Precision regime:** if precision (axis #4) is explored, every shape above should be run at each precision to see whether the crossover in array/dataflow choice *shifts* with precision — this is exactly the multi-way interaction that made the Eyeriss and Gemmini results credible rather than trivial.

---

## Summary of what's sourced vs. inferred

- Solidly primary-sourced: TPU v1 array size/datatype/databus widths, roofline ridge-point number (1350 ops/byte) and MLP/LSTM-vs-CNN memory/compute-bound characterization, instruction count/CISC/CPI description, Eyeriss's three-way dataflow taxonomy and its energy-efficiency claims, Gemmini DSE study's swept parameters/objective/crossover findings, TPU generation-over-generation array-size/HBM/interconnect numbers from the scaling-book reference.
- Secondary-sourced (blog/course summaries corroborating but not independently re-verified against original PDF text): TPU v1 unified buffer/accumulator exact sizes, clock frequency, TDP, weight-FIFO depth; TPU v2/v3 exact TFLOPS figures; cooling-technology transition at v4.
- My own inference/synthesis (flagged inline as [Inference]): the causal "why" behind Google's generation-to-generation choices (bf16, HBM, 3D torus) where I did not find a primary-source quote of Google's stated rationale; the specific applicability arguments connecting each literature finding to TinyTPU's exact current parameters; the axis-avoidance reasoning in part (c); and the objective-function recommendation in part (d) (grounded in cited DSE literature practice, but the specific recommendation for TinyTPU's search is my synthesis, not a quoted claim).
