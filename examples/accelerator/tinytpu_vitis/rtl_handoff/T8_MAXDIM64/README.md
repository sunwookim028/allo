# TinyTPU-isa, T=8, MAXDIM=64 -- the second array size

Vitis HLS 2023.2 generated Verilog for the ASIC synthesis-only
handoff (DC, freepdk-45nm, memories as flip-flops, no P&R).
Enter the flow at `sv2v` with `sv2v_manifest.f`; the top module
is `tinytpu_isa`.

## Configuration

- `T` = 8
- `MAXDIM` = 64
- `SPAD_ROWS` = 512
- `NVR` = 512
- `NAR` = 136
- `QD` = 16
- `IMEM_SIZE` = 56
- `DMA_WORDS` = 1
- `data type` = int8 x int8 -> int32, mvout clips to int8

Emitted from allo commit `92f0618ff40b35d3f5344a744302485940468de7`.

`main` moved to `01a55a22` while this was synthesising, so the emitted
`kernel.cpp` was regenerated at both commits and compared: **byte-identical**
(md5 `2a1f9ca33a623ad6fa66b5208771332d`). The commits between them move the
`QD` default from `ip/params.py` to `isa_spec.json`, which `microarch_isa.py`
already overrode with the same 16, and add a reduction-tree IP that
`tinytpu_isa` does not instantiate. This RTL is therefore current for
`01a55a22` too, checked rather than assumed.

## Measured on this configuration

Cycles are Vitis `cosim` (xsim), `ap_start` to `ap_done`,
`-m_axi_latency 0` unless stated, bit-exact against numpy.

**Measured on THIS export**, with every variable that matters
set explicitly -- `TPU_MAXDIM` defaults to 64, not 16, and
`TPU_QD` defaults to 16 since `63ee6ec7`, so the defaults are
not a configuration anyone can reconstruct from prose:

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
export PYTHONPATH=<repo root>
# No TPU_* knob may arrive from the shell: a leftover one silently
# changes the configuration without changing the command.
for v in $(env | grep -o '^TPU_[A-Z_]*'); do unset "$v"; done
cd examples/accelerator/tinytpu_vitis

# The cycles below. TPU_TB unset = the DEFAULT performance testbench,
# the only mode published cycle counts come from.
TPU_T=8 TPU_MAXDIM=64 \
TPU_SHAPES=8x8x8,16x16x8,16x16x16,32x32x32,48x48x48,64x64x64,64x32x64,32x64x32 \
TPU_PRJ=$PWD/export_t8.prj python cosim.py

# The correctness gate, at the SAME configuration.
TPU_T=8 TPU_MAXDIM=64 python stress_isa.py
```

- 8x8x8: **286** cycles
- 16x16x8: **425** cycles
- 16x16x16: **490** cycles
- 32x32x32: **1410** cycles
- 48x48x48: **3443** cycles
- 64x64x64: **6916** cycles
- 64x32x64: **4356** cycles
- 32x64x32: **2434** cycles

Vitis `csynth` on `xcu280-fsvh2892-2L-e`, 3.33 ns target:

- FF: 44303
- LUT: 70471
- BRAM: 62
- DSP: 58
- target clock period: 3.33 ns
- estimated clock period: 2.431 ns (411.4 MHz)

## What an ASIC number from this can and cannot say

The agreed scope is **DC synthesis only**: memories as
flip-flops (`sram_mode='none'`), FreePDK-45nm with
`view-tiny`, `topographical=True`. That yields **relative cell
area** and nothing else -- no absolute area, no routed timing,
no power, no DRC or LVS. The FPGA figures above are the
reference point the design is known by, not something an ASIC
flow should reproduce: a 45 nm standard-cell area has no
relationship to a BRAM count.

- **Flip-flop memories are not this design's memories.** On
  the FPGA the scratchpad, vector registers and accumulator
  are block RAM; mapped to flops they will **dominate the cell
  area**, and the resulting number then says more about the
  memory treatment than about the datapath. A comparison
  against any flow that used provided SRAM macros needs the
  same memory treatment on both sides.
- **The useful comparison is between our own variants**,
  synthesised identically -- the shipped design against the
  burst-widened candidate against T=8 -- and not between these
  and another project's numbers.
- **There is no testbench here.** The design's testbench is
  C++, generated per shape by `cosim.py`, and drives the
  design through its AXI interfaces; it is not synthesisable
  and would not help a synthesis-only flow.

## Notes

- **This is the `QD=16` export.** Channel depth became 16 in `63ee6ec7`; the directory that stood here before was emitted at `QD=8` and its cycle row (493 at 16x16x16, 7083 at 64x64x64) belonged to that RTL. The three `T4_*` directories beside this one are **still `QD=8` exports**, so no FF/LUT or cycle ratio may be formed across this directory and those until they are re-exported -- the two sides would be different designs.
- 8x8 array: 64 PE kernel instances plus 64 weight loaders.
- **A previous export of this variant was incomplete** -- 223 `.v` files with no `tinytpu_isa.v` at all, from a `csynth` that had not finished, and a manifest generated from that partial set which was therefore self-consistent. `export_rtl.py` now refuses an export whose named top module is undefined or whose resource record is empty.
- Peak is 64 MAC/cycle, FOUR TIMES the T=4 builds, so **its cycles are not comparable with theirs**; read the fraction-of-peak tables in `docs/source/designs/benchmarks.rst`.
- `stress_isa.py` at this exact configuration (`TPU_T=8 TPU_MAXDIM=64`): STRESS OK: 630/630 runs exact. A correct-looking export of a broken design is worse than no export, so the correctness gate is part of the export record, not a separate errand.
- Vitis HLS 2023.2, `csynth_design`, part `xcu280-fsvh2892-2L-e`, 3.33 ns target, `wrap_io=False`, `configs={'align_value': 64}`, `config_interface -m_axi_max_widen_bitwidth 512`.
- Vitis emits one four-line `.dat` memory-initialisation file for the sequencer's LOOP_DEPTH-deep `iv_now` RAM; it sits beside the Verilog and is deliberately absent from the manifest (it is data, not a compile unit). No module here references it -- verified, zero `$readmemh` hits -- so it is inert for a synthesis-only flow.
- Layout is FLAT and identical across all four variants: `sv2v_manifest.f`, `MANIFEST.json`, `README.md` and the `.v` files side by side, no `rtl/` subdirectory, manifest entries bare and relative to the manifest's own directory. No `isdir` probing is needed.
- `export_rtl.py --manifests T8_MAXDIM64` regenerates all three file lists and both stubs from the `.v` files here, and on an unchanged directory it is a **byte-identical no-op** -- that is the self-check that the mechanism was used correctly. It was not idempotent before this export: it read its own `_stub.v` output back in as source, which listed the scratchpad and the accumulator twice in the full list and grew each stub by a blank line per run.

`sv2v_manifest.f` and `MANIFEST.json` are generated by
`examples/accelerator/tinytpu_vitis/export_rtl.py` from the
module instantiation graph of these files. Vitis emits no
compile-order file for a dataflow region, so the graph is the
only source; do not hand-edit either.
