# TinyTPU-isa, shipped configuration — RTL for ASIC synthesis

Generated Verilog for the one design on `main`, handed over so a synthesis flow
on another host can read it from git without running Vitis.

## What this is

| | |
| --- | --- |
| Top module | `tinytpu_isa` |
| Configuration | `T = 4` (4x4 array), `MAXDIM = 16`, int8 operands, int32 accumulation |
| Emitted from | `examples/accelerator/tinytpu_vitis/microarch_isa.py` at the commit that adds this directory |
| Emitted by | Vitis HLS 2023.2, `csynth_design`, `xcu280-fsvh2892-2L-e`, 3.33 ns target |
| Build options | `wrap_io=False`, `configs={"align_value": 64}`, `config_interface -m_axi_max_widen_bitwidth 512` |
| Files | 146 `.v` plus one `.dat` memory-initialisation file |
| Manifest | `sv2v_manifest.f`, dependency order, leaves first, top last |

## Measured, for the numbers a synthesis run should be compared against

Cycles, RTL co-simulation, five benchmark shapes:

    4x4x4   16x16x16   32x32x32   48x48x48   64x64x64
      172        262        418        484        686

FPGA resources and timing from the same `csynth` that emitted this RTL,
re-measured when this directory was created:

| BRAM_18K | DSP | FF | LUT | URAM | Target | Estimated |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 14 | 17,481 | 26,583 | 0 | 3.33 ns | **2.431 ns** |

These are **FPGA** figures for a UltraScale+ part. They are here as the
reference point the design is known by, not as something an ASIC flow should
reproduce — a 45 nm standard-cell area has no relationship to a BRAM count.

## What an ASIC number from this can and cannot say

The agreed scope on the synthesis side is **DC synthesis only**: memories as
flip-flops (`sram_mode='none'`), FreePDK-45nm with `view-tiny`,
`topographical=True`. That yields **relative cell area** and nothing else — no
absolute area, no routed timing, no power, no DRC or LVS.

Two consequences worth stating before anyone compares:

- **Flip-flop memories are not this design's memories.** On the FPGA the
  scratchpad, vector registers and accumulator are block RAM; mapped to flops
  they will dominate the cell area and the resulting number says more about the
  memory treatment than about the datapath. A comparison against any flow that
  used provided SRAM macros needs the same memory treatment on both sides.
- **The useful comparison is between our own variants**, synthesised
  identically — the shipped design against the burst-widening candidate against
  `T = 8` — not between this and a different project's numbers.

## Regenerating it

This is generated output and is committed only so it can travel. To re-emit it
from source, on a host with Vitis HLS 2023.2:

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
cd examples/accelerator/tinytpu_vitis
TPU_PRJ=/tmp/rtl.prj python - <<'PY'
import cosim
s = cosim.customize(cosim.tinytpu_isa); cosim.schedule(s)
s.build(target="vitis_hls", mode="csyn", project=cosim.PRJ,
        wrap_io=False, configs={"align_value": 64})
cosim.patch_axi_depths(cosim.PRJ)
open(cosim.PRJ + "/tb.cpp", "w").write(cosim.testbench(*cosim.SHAPES[0]))
cosim.vitis(cosim.PRJ, cosim.TCL_SYN, "csynth.log")
PY
# RTL lands in $TPU_PRJ/out.prj/solution1/syn/verilog
```

Vitis is not on `PATH` on the development host; `cosim.py` hardcodes
`/opt/xilinx/Vitis_HLS/2023.2` and sets `LDFLAGS = "-B/usr/bin"` to work around
Vitis's binutils 2.37 against this system's glibc. Without that flag the link
fails.

There is **no testbench here.** The design's testbench is C++, generated per
shape by `cosim.py`, and it drives the design through its AXI interfaces; it is
not synthesisable Verilog and would not help a synthesis-only flow.
