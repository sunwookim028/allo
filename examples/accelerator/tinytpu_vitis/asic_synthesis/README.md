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

| variant | total cell area | non-comb. | comb. | seq. cells | worst slack | violating | wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T=4, MAXDIM=16, shipped (`T4_MAXDIM16_shipped_baseline`) | **1,136,598** | 906,098 (79.7%) | 230,501 | 200,561 | **+0.21 ns** | 0 | 37 min |
| T=4, MAXDIM=64, shipped (`T4_MAXDIM64_shipped`) | **1,865,314** | 1,503,790 (80.6%) | 361,524 | 333,189 | **+0.21 ns** | 0 | 56 min |
| T=8, MAXDIM=64 (`T8_MAXDIM64`) | **2,481,926** | 1,982,554 (79.9%) | 499,371 | 438,922 | **+0.20 ns** | 0 | 72 min |
| T=4, MAXDIM=64, burst-widened, banked (`T4_MAXDIM64_burstwiden`) | see below | | | | | | |
| *superseded* — an earlier export, before memories were derived from MAXDIM (`superseded_export_T4_MAXDIM16`) | 1,271,692 | 1,016,187 (79.9%) | 255,505 | 224,987 | +0.18 ns | 0 | 47 min |

Every run: identical settings, all close timing at 3.33 ns, all about 80%
non-combinational. The T=8 run was done by the second zhang-21 session
(`allo-zhang-21 [ebafc1]`) from `main` 287e4b68, same settings and tool
versions.

### The two clean comparisons

| what it isolates | pair | result |
| --- | --- | --- |
| operand space, MAXDIM 16 → 64 | shipped vs shipped, T=4 fixed | **+64.1%** cell area (+65.9% non-comb, +56.9% comb) |
| array size, T 4 → 8 | MAXDIM=64 fixed | **1.33x** cell area |

**Never quote `T8_MAXDIM64` against the MAXDIM=16 baseline** (2.18x) without
naming both changes: that ratio is the array doubling *and* the fourfold
operand space, and it has already been withdrawn once for travelling without
them.

### What the memory treatment does to these numbers

The operand-space pair states it exactly. Vitis measures the same change as
**+2.4% FF** (17,075 → 17,488) and +30% BRAM (40 → 52); this flow measures
**+64.1% cell area**, 66% of it non-combinational. Both are right: the change
is almost entirely memory, and with `sram_mode='none'` the BRAM axis is what
cell area renders. Read every row here as "BRAM plus logic, all in flops" — a
memory-dominated change looks roughly 27x worse on the FF axis than the FPGA's
own FF count says. All four designs land at ~80% non-combinational, which is
why the *ratios* above transfer between them while the absolute areas do not
transfer to any design with SRAM macros.

### Cycles these areas belong beside

Re-verified bit-exact on the current design: T=4 at MAXDIM=16 is
**171 / 261 / 417 / 483 / 685**; T=8 at MAXDIM=64 is **285 / 424 / 493** at
8x8x8, 16x16x8 and 16x16x16, and **7083** at 64x64x64.

### The burst-widened variant needed a dual-write-port memory

The first export of `T4_MAXDIM64_burstwiden` **could not be synthesised**. DC
refused it in 2 minutes:

```
Error: tinytpu_isa_dma_ld_0_1_rbA_RAM_AUTO_1R1W.v:60: Net 'ram[0][31]' or a
directly connected net is driven by more than one source, and not all drivers
are three-state. (ELAB-366)
```

Vitis had satisfied the widened loop's writes by emitting `rbA` as a **true
dual-write-port RAM** — two `always @(posedge clk)` blocks writing one array —
while still naming the module `_1R1W`. Auditing every RAM module in all four
variants found exactly one such module, only in that variant. An FPGA block RAM
has two write ports, so this is free there; standard cells have no such
primitive, and with memories as registers it is a real multi-driver. **The
refusal and the +123% BRAM figure are the same fact on two substrates**: the
widening buys cycles with something free on FPGA that costs a memory macro on
ASIC. It was re-exported with the buffer cyclically banked by 16, one writer per
bank, at identical cycles — and the banked form costs 100 BRAM against the
refused form's 116, so the legal version is also the cheaper one.

Clock 3.33 ns on `ap_clk`, the target the RTL was emitted at. The superseded
row is kept because it was published before being replaced: it describes a
netlist that no longer exists, and the current design is 10.6% smaller and one
cycle faster at every shape — both traceable to deriving the memory sizes from
MAXDIM. Power from DC is indicative only (default toggle rates, no activity
data); the superseded run's 57.1 mW is not carried over.

Reports are under `reports/<variant>/`: the QoR and power reports verbatim, an
area summary (full report is 716 KB, the reference report 4 MB; both stay in the
build directory), and mflowgen's `synthesis-metrics.json`.

## The Gemmini side of the same flow

`../gemmini_rtl/` holds Gemmini's accelerator — the `Gemmini` module and its
local memories, with Rocket, the caches, the buses and the DRAM model cut —
elaborated at DIM=4 and DIM=8, int8/int32, at a memory capacity matched to ours
so that `sram_mode='none'` means the same thing on both sides. It is for **these
identical settings**: DC W-2024.09, FreePDK45 `view-standard`, 3.33 ns,
topographical, flatten effort 3.

Four runs, not two: each directory carries `sv2v_manifest.f` (with memories)
and `sv2v_manifest_nomem.f` (memory arrays black-boxed). The logic-only figure
is the headline, because Gemmini's memories and ours are different sizes; our
own designs need the matching run, omitting their seven `*_RAM_*` modules, for
the pair to mean anything.

Two differences from the RTL above, both measured: this RTL **is**
SystemVerilog (firtool emits packed multidimensional arrays outside any
`ifdef`, so `normalize_rtl: True` or `analyze -format sverilog` is required,
unlike for Vitis output), and `SYNTHESIS` must be defined at read time. The
top-module check is not a problem — `module Gemmini(` carries a comment, not an
attribute. See `../gemmini_rtl/README.md` and
`docs/source/designs/gemmini_comparison.rst`, "Area".

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
