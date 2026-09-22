# TinyTPU-isa through standard-cell synthesis

Design Compiler synthesis of the Vitis-generated Verilog, on FreePDK45/NanGate
45 nm. It exists to compare **our own variants against each other**, synthesised
identically. It is not an ASIC verdict on any of them.

## What this gives up

- **Relative cell area only.** No place and route, so no real area and no routed
  timing.
- **Memories are flip-flops** (`sram_mode='none'`). The scratchpad, vector
  registers and accumulator become registers, which is why 80% of the area is
  non-combinational. This **overstates** the cost of any change that buys cycles
  with more on-chip memory, which is exactly what the burst-widened variant
  does. It also makes these numbers **not comparable** with runs that used SRAM
  macros, including `TestNPU-core` and `allo-mininpu-v2` in the same flow.
- **Power is a default-toggle estimate.** No activity data, so indicative only.
- No DRC, no LVS, no signoff timing.

## Results

| variant | total cell area | non-comb. | comb. | seq. cells | worst slack | violating paths |
| --- | --- | --- | --- | --- | --- | --- |
| shipped T=4, MAXDIM=16 (current, `T4_MAXDIM16_shipped_baseline`) | **1,136,598** | 906,098 (79.7%) | 230,501 | 200,561 | **+0.21 ns** | 0 |
| *superseded* — an earlier export, before memories were derived from MAXDIM | 1,271,692 | 1,016,187 (79.9%) | 255,505 | 224,987 | +0.18 ns | 0 |

Clock 3.33 ns on `ap_clk`, the target the RTL was emitted at. The current
design meets it with 62 logic levels on a 3.08 ns critical path and zero hold
violations, in 37 min on zhang-21. The superseded row is kept because it was
published before being replaced: it describes a netlist that no longer exists,
and the current design is 10.6 % smaller. Power from DC here is indicative
only (default toggle rates, no activity data); the superseded run's 57.1 mW is
not carried over.

Reports are under `reports/<variant>/`: the QoR and power reports verbatim, an
area summary (full report is 716 KB, the reference report 4 MB; both stay in the
build directory), and mflowgen's `synthesis-metrics.json`.

## Reproducing

The flow is mflowgen at `~/allo-asic` on zhang-21; `construct-commercial.py`
here is the design, copied from `~/allo-asic/designs/allo-tinytpu-isa/`.

```bash
source /etc/profile.d/modules.sh
module load synopsys-dc-W-2024.09 synopsys-2024 cadence-231
source /opt/anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate /scratch/users/sk3463/envs/asicflow
export SNPSLMD_LICENSE_FILE=27020@en-license-05.coecis.cornell.edu

mkdir -p /scratch/users/sk3463/build_tinytpu_t4 && cd $_
mflowgen run --design ~/allo-asic/designs/allo-tinytpu-isa/construct-commercial.py
make 4          # synopsys-dc-synthesis
```

Build directories belong on `/scratch` (local), not NFS. `TINYTPU_RTL` points
the design at another variant's directory, which must hold `rtl/` and
`sv2v_manifest.f`.

Versions: mflowgen **0.8.0** at commit `aee0e5d6` (PyPI 0.7.0 lacks the `Node`
construct the node library needs), sv2v **v0.0.10-0-g87642c0** (installed, not
used — see below), DC **W-2024.09**, ADK `freepdk-45nm` **view-standard**.

## Three settings that differ from the template, and why

1. **`normalize_rtl: False`.** sv2v converts all 146 files fine, but it emits
   the top module's `(* CORE_GENERATION_INFO = ... *)` attribute and the
   `module tinytpu_isa (` keyword on one line, and the collector's check is
   `^\s*module\s+<top>`, so it reports "normalized RTL does not contain top
   module" and stops. Vitis emits Verilog-2001, so skipping sv2v is right on the
   merits and keeps the RTL byte-identical.
2. **`view-standard`, not `view-tiny`.** view-tiny has no `rtk-tech.tf`, no
   `*/SDFF*` scan cells (which the project's ADK overlay requires) and no
   driving cell; DC runs but leaves unmapped GTECH.
3. **`clock_port: 'ap_clk'`**, which is what Vitis emits.

## Two defects in the flow, found here

- **The top-module check rejects any attributed top module.** Vitis attributes
  every top module it emits, so `normalize_rtl: True` cannot work for this RTL
  until the check tolerates an attribute on the `module` line.
- **The `freepdk-45nm` ADK node is not idempotent and mutates its own source
  tree.** It runs `mv <view>/adk.tcl <view>/adk-base.tcl; cp adk-overlay.tcl
  <view>/adk.tcl` against `~/allo-asic/adks/...`, not a build copy. A second run
  moves the overlay onto `adk-base.tcl`, so `adk.tcl` sources itself and DC
  fails with "too many nested evaluations (infinite loop?)". Recover with
  `git checkout` in `~/allo-asic` (the original `adk.tcl` is a symlink to
  `../pkgs/base/adk.tcl`) and delete the leftover `adk-base.tcl`. `view-standard`
  is unaffected, being unpacked fresh in the build directory.
