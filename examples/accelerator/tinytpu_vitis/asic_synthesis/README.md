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
| T=4, MAXDIM=64, burst-widened, **banked** (`T4_MAXDIM64_burstwiden`) | **3,254,024** | 2,628,257 (80.8%) | 625,767 | 581,616 | **+0.21 ns** | 0 | 98 min |
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
| burst widening | `T4_MAXDIM64_shipped` vs `T4_MAXDIM64_burstwiden`, MAXDIM=64 fixed | **+74.4%** cell area (+74.8% non-comb, +73.1% comb) |

The burst-widening row is the design decision this set exists for. It buys
−720 cycles at 48³ and −960 at 64³ — 55-61% of the steady-state deficit
against Gemmini — for **+74.4% cell area here**, against Vitis's +43% FF and
+92% BRAM for the same change. As everywhere in this table, the BRAM axis is
what cell area renders, so +74.4% is the flip-flop-memory price of the
widening and not the price a design with SRAM macros would pay. Compare it
only against `T4_MAXDIM64_shipped`: the pair differs in the widening alone.

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

**Every export here, and therefore every area in this file, is the `QD=8`
design** — the default channel depth until `63ee6ec7` (2026-09-24). Read the
cycles below as belonging to that RTL, which is the point of stating them:
they are the pair, and re-quoting them next to today's row would mismatch the
two. The shipped row is now **175 / 265 / 421 / 482 / 674**; re-pricing the
areas against it needs a fresh DC run, which has not been done.

Re-verified bit-exact on the design as exported: T=4 at MAXDIM=16 is
**171 / 261 / 417 / 483 / 685**; T=8 at MAXDIM=64 is **285 / 424 / 493** at
8x8x8, 16x16x8 and 16x16x16, and **7083** at 64x64x64.

**`T8_MAXDIM64/` no longer holds that RTL.** It was re-exported at `QD=16`
against `main` 92f0618f, and the directory now carries **286 / 425 / 490** at
those three shapes and **6916** at 64x64x64 (`+1 / +1 / -3 / -167`), with FF
43911 -> 44303 and LUT 70281 -> 70471. The `T8_MAXDIM64` area row above was
measured on the superseded `QD=8` export, which survives only in git history
(the stubbing commit `e3cc595b`); **it is not the area of the RTL now in that
directory**, and the T 4 -> 8 ratio in the next table pairs it with `QD=8`
T=4 exports, so it stays valid as a `QD=8`-to-`QD=8` comparison and must not
be re-quoted against a fresh T=8 run. The other three directories are
untouched `QD=8` exports.

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

## Logic-only runs: why the memory stubs exist

Omitting a memory module from the file list does **not** black-box it. DC cannot
resolve the reference and the flow treats that as fatal:

```
Warning: Unable to resolve reference 'mem_ext' in 'mem'.  (LINK-5)
Error: failed to link design Gemmini
```

`tools/make_stubs.py` copies each omitted module's header verbatim from its own
source — ports only, comments stripped, no body — and writes `<name>_stub.v`,
which is appended to the logic-only file list. It refuses if a header cannot be
found rather than inventing one.

With stubs the memory ports terminate in a zero-area black box instead of
dangling, so a logic-only figure excludes the array **and the array's own
interface**, while the logic that drives the memories — address generation,
enables, write masks — is retained and counted. The same rule must be applied
on both sides of any logic-only comparison, or the two numbers are not
measuring the same boundary.

## Where the burst widening's area actually goes

`report_area -hierarchy` on the pair, per top-level instance:

| instance | MAXDIM=64 shipped | banked widened | change |
| --- | --- | --- | --- |
| `gmem1_m_axi_U` | 56,605 | 744,092 | **13.1x** |
| `gmem2_m_axi_U` | 56,580 | 745,280 | **13.2x** |
| `gmem0_m_axi_U` | 703,513 | 703,151 | — |
| `dma_ld_0_1_U0` | 376,260 | 388,577 | +3.3% |
| `spm_0_U0` | 188,053 | 188,299 | — |
| `vru_0_U0` | 188,017 | 188,046 | — |
| `accu_0_U0` | 103,857 | 103,837 | — |

**99.1% of the +1,388,710 delta is the two AXI adapters** `gmem1` and `gmem2`
growing 13x. The scratchpad, the vector registers and the accumulator do not
move at all, and `dma_ld`'s own buffers account for 0.9%.

So "+74.4% for the widening" is **not** the operand memories getting bigger. It
is the cost of widening two 32-bit master ports to the same width as `gmem0`:
the adapters' outstanding-transaction buffering scales with the data width, and
on the FPGA that buffering is block RAM. The design question this raises is
narrower and more tractable than "is the widening worth it": whether both
operand ports need widening, or whether one wide port and a shared buffer would
buy the same cycles.
