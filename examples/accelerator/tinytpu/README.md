# TinyTPU

TinyTPU is a small, runnable accelerator flow. `isa.py` describes the buffers
and instruction semantics; the generic ACT compiler maps TOSA into those
instructions; `microarch.py` is the composed Allo instruction interpreter.
One `top_s` schedule is selectable at the normal Allo backend boundary:
LLVM-JIT CPU, Vitis HLS C++, or Kai Shao's CIRCT RTLGen.

This integration carries `kkkaishao/allo`'s `allo-rtlgen` commit
`9e2d5716d7afdf4371b03f4121570a2891ab903d`. ACT remains responsible for
workload-to-ISA selection and tiling; the chosen Allo backend lowers the
microarchitecture. The ISA description does **not** synthesize a decoder by
itself: `microarch.py` explicitly composes and connects the decoder and units.

## Reproduce

From this directory, with the local project built in the `allo` Conda
environment:

```bash
make oracle       # 8 hand-written assembly examples
make compiler     # direct-TOSA compiler, tiling, and spill checks
make cpu          # scaffold LLVM-JIT lowered MLIR
make hls          # generate and sanity-check Vitis HLS C++
make rtl          # schedule and emit Kai RTLGen SystemVerilog
make synth        # run Vitis HLS C-synthesis, print the measured latency table
make ppa          # the co-design objective: synthesize, re-measure, score cycles
```

`make synth` and `make ppa` invoke Vitis HLS, so source its `settings64.sh`
first (`. /opt/xilinx/Vitis_HLS/2023.2/settings64.sh`). The other targets need
no synthesis tool.

Set `CONDA=/path/to/conda` or `ENV=name` when needed. The compiler check uses
TOSA text directly, so it does not require PyTorch or torch-mlir.

## Verified results

| Check | Result |
| --- | --- |
| Hand-written ISA programs | 8/8 pass against NumPy |
| TOSA vector add | 7 emitted instructions; output matches NumPy |
| TOSA pressure graph | 73 emitted instructions, 4 `vstore`s; output matches NumPy |
| ISA extension: TOSA negate | 5 emitted instructions; output matches NumPy |
| Tiled GEMM 8x8x8 | 216 ISA instructions; output matches NumPy |
| Tiled GEMM 8x16x32 | 1792 ISA instructions; output matches NumPy |
| CPU backend | complete TinyTPU interpreter compiles to LLVM JIT |
| Vitis backend | HLS C++ export succeeds without Vitis installed |
| Kai RTLGen backend | complete TinyTPU emits CIRCT-scheduled SystemVerilog |
| Vitis HLS C-synthesis | 411 MHz Fmax, 8515 LUT / 9111 FF / 20 DSP / 26 BRAM18K on `xcu55c` |
| Synthesis-grounded objective | 126,432 cycles across both GEMMs (41 s per evaluation) |

The pressure graph has 20 vector inputs and 10 partial sums, deliberately
exceeding the eight vector-register slots. It exercises allocation, movement,
and spilling.

## Closing the loop with synthesis

`microarch.py` declares each unit's cycle model as a frozen constant, marked in
the source as replaceable by a synthesis-derived table. `synth.py` does that
replacement: it runs C-synthesis on the same composed schedule and rebuilds
every unit's `(ii, depth)` from the report, reconciling the measured call
latency with the per-loop `ii`/depth/trip count so `depth + ii * trips`
reproduces exactly what the tool measured.

The correction is large, which is the point — the frozen guesses were not close:

| Unit | declared | measured | note |
| --- | --- | --- | --- |
| `vload` | 11 cyc | 9 cyc | |
| `vstore` | 11 cyc | 10 cyc | |
| `vpu` | 13 cyc | 18 cyc | |
| `mxu` | 36 cyc | 72 cyc | two sequential passes, so `ii=2` |
| `dma_load` | depth 8 | depth 75 | the `m_axi` read latency HLS actually builds |

`ppa.py` scores a candidate under that measured table: the objective is
`sum(CompiledProgram.cycles())` over the two GEMMs. Area and Fmax are reported
for every candidate but deliberately not scored. Because the table is
overwritten from synthesis before scoring, editing an `ISA.latency` declaration
cannot move the score — only building different hardware can.

## Map of the flow

- `isa.py`: memories, vector-register shape, and instruction access/compute
  semantics. Add a compiler-visible operation here.
- `microarch.py`: opcode numbers, decode, datapaths, schedules, and HLS export.
  Add the matching hardware behavior here too.
- `oracle.py`: hand-written instruction-stream examples and functional checks.
- `program.py`: optional PyTorch-to-TOSA examples; requires `torch_mlir`.
- `verify.py`: dependency-light direct-TOSA checks used by the Makefile.
- `feedback.py`: the pre-synthesis VREG/VMEM access-cost objective.
- `synth.py`: runs Vitis HLS C-synthesis and rebuilds the per-unit `(ii, depth)`
  table from the measured report.
- `ppa.py`: the co-design objective — synthesize, re-measure, score cycles.
- `chia_agent/`: the generate -> synthesize -> score -> keep-or-rewind loop.

The backend switch is explicit and contains no device-specific compiler fork:

```python
from examples.accelerator.tinytpu.microarch import export_backend

cpu = export_backend("cpu")
hls = export_backend("vitis").hls_code
rtl = export_backend("rtl").verilog
```

The CLI scaffolds the corresponding project with `--backend cpu`,
`--backend vitis`, or `--backend rtl`. Vitis-specific AXI pragmas are applied
only on the Vitis branch; RTLGen uses its own device model.

The `vneg` extension is intentionally small but crosses every layer: its ISA
semantics select `tosa.negate`, its opcode is 9 (leaving existing encodings
unchanged), its VPU behavior is synthesizable, and both an oracle program and a
direct-TOSA test validate it. Use the same four edits for a new operation.

The compiler accepts static-shape TOSA and performs pattern selection, tiling,
buffer allocation, routing, accumulation, and spilling. The ISA still exposes
fixed 8-element vectors and a fixed 4x4 matmul instruction, but larger GEMMs
whose dimensions satisfy the explicit tile policy are decomposed automatically.
Non-divisible dimensions currently fail loudly instead of silently producing a
partial tile.

## Clean build

From the repository root, initialize the pinned LLVM/CIRCT dependencies and
build in the required Conda environment:

```bash
git submodule update --init --recursive
conda activate allo
bash scripts/build-mlir.sh externals/llvm-project Release gcc g++ --fresh
bash scripts/build-circt.sh externals/circt externals/llvm-project/build Release gcc g++
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$PWD/externals/circt/ext" pip install -v -e .
```

`nanobind` is constrained below version 3 for compatibility with the pinned
MLIR Python bindings, and the wheel bundles the OR-Tools shared library needed
by RTL scheduling. No runtime `LD_LIBRARY_PATH` is required.

## Important integration rule

Keep instruction operand order identical in `isa.py` and `microarch.py`.
`dma_store` is the one intentional adapter: the ISA encodes `(bram source,
dram destination, length)`, while its hardware helper is named `(dram address,
bram address, length)`, so the decoder swaps the first two operands.
