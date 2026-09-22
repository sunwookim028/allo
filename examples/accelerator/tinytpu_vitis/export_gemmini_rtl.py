# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Gemmini's ACCELERATOR, not its SoC, for the same DC flow ours goes to.

`export_rtl.py` hands zhang-21 our design; this hands it the opponent, in the
same shape, so the two areas are comparable by construction. Chipyard elaborates
a whole SoC -- Rocket, L1, L2, buses, DRAM model, debug -- and our design has
none of that, so the only fair unit is the `Gemmini` module: the RoCC
accelerator with its mesh, scratchpad, accumulator, controllers, queues and its
own DMA and TLB. That module is cut out by taking the TRANSITIVE CLOSURE of the
module instantiation graph from `Gemmini`, which is derived from the elaborated
Verilog rather than asserted about it.

Two file lists come out of each export, from one set of files:

* `sv2v_manifest.f` -- the whole accelerator, memories included. Gemmini's
  memories arrive as behavioural flop arrays (`split_mem_ext` and friends,
  `reg [7:0] ram [0:N]`), which is exactly `sram_mode='none'` on our side.
* `sv2v_manifest_nomem.f` -- the same list with the four memory-array modules
  replaced by port-compatible EMPTY modules, so DC reports the accelerator's
  LOGIC area. That figure is the one that survives the memory treatment, and it
  costs a second `make` rather than a second export.

  **The stubs are not optional.** Merely omitting a module's source does not
  give DC a black box, it gives DC an unresolved reference -- `Unable to
  resolve reference 'mem_ext' in 'mem'` (LINK-5) -- and the link fails. Each
  stub's header is copied verbatim from the real module so it cannot disagree
  about a port, and the result is a better boundary than dangling nets: the
  figure keeps everything that DRIVES the memories, and excludes the arrays and
  the arrays' own interfaces.

    python export_gemmini_rtl.py --manifests    # rewrite lists and stubs only

**The memories must be capacity-matched before this means anything.** Stock
Gemmini carries 256 KiB of scratchpad and 64 KiB of accumulator against our
10.1 KiB; flip-flopping that is 2.6 M registers and measures the memory
treatment and nothing else. `Int8Dim{4,8}AreaGemminiRocketConfig` cut it to
8 KiB / 4 KiB, which `designs/gemmini_comparison.rst` has already measured to be
cycle-neutral at every published shape. This script REFUSES an export whose
scratchpad is bigger than that, because the refusal is cheaper than the run.

    python export_gemmini_rtl.py                 # both configurations
    python export_gemmini_rtl.py --only DIM4

The configurations must have been elaborated first, in a chipyard shell:

    cd ~/chipyard && source env.sh && cd sims/verilator
    make verilog CONFIG=Int8Dim4AreaGemminiRocketConfig -j16

which also REWRITES `gemmini_params.h` in place -- restore the committed
snapshot afterwards (`gemmini/gemmini_params.dim4.h`).
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEST = os.path.join(HERE, "gemmini_rtl")
CHIPYARD = os.environ.get("CHIPYARD", os.path.expanduser("~/chipyard"))
GENERATED = os.path.join(CHIPYARD, "sims", "verilator", "generated-src")
TOP = "Gemmini"

sys.path.insert(0, HERE)
from export_rtl import (ExportError, compile_order, write_stubs,  # noqa: E402
                        _MODULE, _INST, _KW)

#: Gemmini's local memories, as firtool emits them: a wrapper module per
#: `SyncReadMem` (`mem`, `mem_0`) instantiating a `*_ext` blackbox that the
#: memory-model file implements as a behavioural flop array. Dropping the
#: `*_ext` half from a file list leaves the wrappers as empty black boxes,
#: which is how the logic-only figure is taken without editing any RTL.
MEM_ARRAY = re.compile(r"^(split_)?mem(_\d+)?_ext$")

#: Refuse a capacity that cannot be synthesised as flops beside ours. The
#: matched configs carry 98,304 bits (8 KiB scratchpad + 4 KiB accumulator);
#: stock Gemmini's 320 KiB is 2,621,440, against our whole T=4 design's 200,561
#: sequential cells, and the area would measure the memory treatment.
MAX_MEM_BITS = 32 * 8192

CONFIGS = {
    "DIM4": {
        "config": "Int8Dim4AreaGemminiRocketConfig",
        "dest": "DIM4_int8_capmatched",
        "against": "TinyTPU-isa T=4, MAXDIM=16 (`rtl_handoff/T4_MAXDIM16_shipped_baseline`)",
        "params": {
            "meshRows / meshColumns": "4 (DIM = 4)",
            "tileRows / tileColumns": "1",
            "inputType": "SInt(8.W)",
            "accType": "SInt(32.W)",
            "spatialArrayOutputType": "SInt(20.W)",
            "dataflow": "BOTH (weight- and output-stationary; only WS is exercised)",
            "sp_capacity": "8 KiB in 4 banks (stock: 256 KiB)",
            "acc_capacity": "4 KiB in 2 banks (stock: 64 KiB)",
            "tlb_size": "4 entries",
            "dma_buswidth / dma_maxbytes": "128 bits / 64 bytes",
        },
        "cycles": {
            "4x4x4": 161, "8x8x8": 220, "12x12x12": 347,
            "16x16x8": 391, "16x16x16": 593,
        },
    },
    "DIM8": {
        "config": "Int8Dim8AreaGemminiRocketConfig",
        "dest": "DIM8_int8_capmatched",
        "against": "TinyTPU-isa T=8, MAXDIM=64 (`rtl_handoff/T8_MAXDIM64`)",
        "params": {
            "meshRows / meshColumns": "8 (DIM = 8)",
            "tileRows / tileColumns": "1",
            "inputType": "SInt(8.W)",
            "accType": "SInt(32.W)",
            "spatialArrayOutputType": "SInt(20.W)",
            "dataflow": "BOTH (weight- and output-stationary; only WS is exercised)",
            "sp_capacity": "8 KiB in 4 banks (stock: 256 KiB)",
            "acc_capacity": "4 KiB in 2 banks (stock: 64 KiB)",
            "tlb_size": "4 entries",
            "dma_buswidth / dma_maxbytes": "128 bits / 64 bytes",
        },
        "cycles": {},
    },
}

#: What is inside the boundary and why, stated per subsystem so the cut can be
#: argued with rather than taken on trust. Every name here is a module in the
#: closure; the closure itself is computed, not this list.
INSIDE = [
    ("mesh", "`Mesh`, `Tile`, `PE`, `MacUnit`, `MeshWithDelays`, "
             "`TransposePreloadUnroller`, `AlwaysOutTransposer` -- the "
             "systolic array and its skew registers. Our `pe`/`accu` units."),
    ("scratchpad", "`Scratchpad`, `ScratchpadBank`, `mem`/`mem_ext` -- 8 KiB "
                   "of int8 operand storage. Our `spad` plus vector registers."),
    ("accumulator", "`AccumulatorMem`, `TwoPortSyncMem`, `mem_0`/`mem_0_ext`, "
                    "`AccumulatorScale`, `AccPipe`, `ScalePipe` -- 4 KiB of "
                    "int32. Our `ar`."),
    ("controllers", "`ExecuteController`, `LoadController`, `StoreController`, "
                    "`ReservationStation`, `LoopMatmul*` -- decode, dependency "
                    "tracking and the hardware loop. Our `sequencer`."),
    ("DMA", "`StreamReader`, `StreamWriter`, `BeatMerger`, `XactTracker`, "
            "`DMACommandTracker`, `TLBuffer`, `TLXbar` -- the accelerator's "
            "own TileLink master and its buffering. Our `dma_ld`/`dma_st` "
            "and their AXI interfaces."),
    ("queues", "`Queue*`, `Pipeline*`, `MultiHeadedQueue`, `TagQueue`, "
               "`RRArbiter`, `WeightedArbiter`, `ram_*` -- the small "
               "flop-backed memories the controllers run on."),
]

CUT = [
    ("Rocket core and its tile", "`RocketTile`, `CSRFile`, `FPU`, `Frontend`, "
     "the branch predictor. Our design has no host: its program is in DRAM and "
     "it starts on `ap_start`. Gemmini has no decoder of its own -- it is a "
     "RoCC accelerator and the CPU issues its instructions -- so keeping the "
     "core would be measuring a CPU."),
    ("L1 I/D and the L2", "`rockettile_dcache_*`, `rockettile_icache_*`, "
     "`cc_banks_*`, `cc_dir_*`, `InclusiveCache`. Our operands come straight "
     "from DRAM over AXI, uncached."),
    ("SoC buses and peripherals", "`SystemBus`, `MemoryBus`, `PeripheryBus`, "
     "`TLXbar`s outside the accelerator, CLINT, PLIC, debug, bootrom, "
     "`TSIToTileLink`, the serial adapter and clock/reset infrastructure."),
    ("DRAM model and test harness", "`SimDRAM`/`TestHarness`. Ours is "
     "`-m_axi_latency 0`; neither side synthesises its memory system."),
]

#: The one place the boundary is genuinely arguable, reported both ways rather
#: than resolved silently.
TLB_MODULES = ["FrontendTLB", "DecoupledTLB", "DTLB_2", "PMAChecker",
               "PMPChecker_s6", "OptimizationBarrier_TLBEntryData"]


def instantiations(text):
    """`{module: times instantiated}` for one file, with the shared pattern's
    two false positives removed.

    `_INST` finds `Name inst (`, and Verilog's `for (logic ... i = ...)` and
    `if (reset)` both fit that after backtracking -- as `fo`/`r` and `i`/`f`.
    They are harmless when the result is only used to order files, which is
    what `export_rtl.py` does with it, but this export REPORTS the modules
    nothing defines, and two phantoms in that list would teach a reader to
    ignore it. A real instantiation begins the line; a backtracked one does
    not, so comparing against the line's first token is enough.
    """
    defined = set(_MODULE.findall(text))
    out = {}
    for m in _INST.finditer(text):
        used = m.group(1)
        if used in _KW or used in defined:
            continue
        head = text[m.start():m.start() + len(m.group(0))].split()
        if not head or head[0] != used:      # `for (logic ... i` / `if (x)`
            continue
        out[used] = out.get(used, 0) + 1
    return out


def _split_mems(text):
    """A chipyard `*.mems.v` holds every memory in the SoC in one file. Split
    it per module so the closure takes Gemmini's and leaves the caches'."""
    out = {}
    for chunk in re.split(r"(?=^module\s)", text, flags=re.M):
        m = _MODULE.search(chunk)
        if m:
            out[m.group(1)] = chunk
        elif chunk.strip():
            out.setdefault("__preamble__", "")
            out["__preamble__"] += chunk
    out.pop("__preamble__", None)
    return out


def collect(src):
    """Every module the elaboration emitted, as `{module: (filename, text)}`.

    Extensions are kept honest. firtool's output stays `.sv` because it IS
    SystemVerilog -- packed multidimensional arrays (`wire [7:0][6:0]`) and
    assignment patterns (`'{3'h5, 3'h0, ...}`) appear in `CounterFile`,
    `LoopMatmulStC` and `RRArbiter` outside any `ifdef`, so "Chisel emits
    Verilog" is not true of this RTL and sv2v (or a SystemVerilog reader) IS
    needed, unlike the Vitis export. Chipyard's memory-model file is plain
    Verilog and its modules are split out as `.v`, one per file.
    """
    gen = os.path.join(src, "gen-collateral")
    if not os.path.isdir(gen):
        raise ExportError(f"{gen} does not exist -- elaborate the config first")
    mods = {}
    for name in sorted(os.listdir(gen)):
        if not name.endswith((".v", ".sv")):
            continue
        text = open(os.path.join(gen, name), errors="replace").read()
        if name.endswith(".mems.v"):
            for mod, chunk in _split_mems(text).items():
                mods[mod] = (mod + ".v", chunk)
        else:
            for mod in _MODULE.findall(text):
                mods[mod] = (name, text)
    return mods


def closure(mods, top=TOP):
    """The files reachable from `top` through instantiation, and the modules
    that nothing defines."""
    if top not in mods:
        raise ExportError(
            f"the elaboration defines no `module {top}` ({len(mods)} modules "
            f"present). Either the config did not elaborate or firtool renamed "
            f"the accelerator; do not hand-pick a substitute.")
    files, seen, missing, queue = {}, set(), set(), [top]
    while queue:
        mod = queue.pop()
        if mod in seen:
            continue
        seen.add(mod)
        name, text = mods[mod]
        files[name] = text
        for used in instantiations(text):
            if used not in mods:
                missing.add(used)
            elif used not in seen:
                queue.append(used)
    return files, sorted(seen), sorted(missing)


def instance_counts(mods, top=TOP):
    """How many times each module is instantiated in the whole `top` hierarchy.

    `.top.mems.conf` describes a memory module once however many banks use it,
    so a per-module sum under-counts the scratchpad by its bank count. What
    decides whether flip-flop memories are affordable is the instance count, so
    that is what is computed: 1 for the top, and each child's count multiplied
    through its parent's.
    """
    counts, order, seen = {top: 1}, [top], {top}
    while order:
        mod = order.pop(0)
        _name, text = mods[mod]
        for child, n in instantiations(text).items():
            if child not in mods:
                continue
            counts[child] = counts.get(child, 0) + counts[mod] * n
            if child not in seen:
                seen.add(child)
                order.append(child)
    return counts


def mem_inventory(src, config, modules, counts):
    """Geometry, instance count and total flop bits of the accelerator's
    memory arrays.

    This is the `resources` record: the analogue of the Vitis FF/LUT/BRAM
    figures on our side, and the thing whose absence means the elaboration
    never finished. Geometry comes from the elaboration's own
    `.top.mems.conf`, filtered to the memories reachable from `Gemmini`.
    """
    conf = os.path.join(src, f"chipyard.harness.TestHarness.{config}.top.mems.conf")
    want = {m for m in modules if MEM_ARRAY.match(m) and not m.startswith("split_")}
    mems, bits = {}, 0
    if os.path.exists(conf):
        for line in open(conf):
            f = line.split()
            if len(f) >= 8 and f[0] == "name" and f[1] in want:
                depth, width, n = int(f[3]), int(f[5]), counts.get(f[1], 0)
                mems[f[1]] = {"depth": depth, "width": width, "ports": f[7],
                              "instances": n, "bits": depth * width * n}
                bits += depth * width * n
    return mems, bits


def hardware(mems):
    """`DIM`, `BANK_ROWS` and `ACC_ROWS` read back out of the RTL.

    Every elaboration rewrites `gemmini_params.h` in place, so the header on
    disk describes whichever config was elaborated LAST and is worthless as
    provenance for an earlier one. The scratchpad's own geometry is not:
    a bank is DIM int8 elements wide, and an accumulator row is DIM int32.
    """
    sp, acc = mems.get("mem_ext"), mems.get("mem_0_ext")
    if not sp or not acc:
        return {}
    return {"DIM": sp["width"] // 8,
            "BANK_NUM": sp["instances"],
            "BANK_ROWS": sp["depth"],
            "ACC_ROWS": acc["depth"] * acc["instances"],
            "ACC_DIM": acc["width"] // 32}


def manifest(dest, name, order, skip=(), stubs=None):
    drop = set(skip)
    kept = [f for f in order if f not in drop]
    with open(os.path.join(dest, name), "w") as fh:
        fh.write(f"# Gemmini accelerator RTL. Top module: {TOP}\n"
                 f"# Paths are relative to THIS FILE's directory, which also\n"
                 f"# holds the .v files -- the layout is flat, no rtl/ subdir.\n"
                 f"# Dependency order, leaves first, {TOP} last. GENERATED by\n"
                 f"# export_gemmini_rtl.py from the module instantiation graph\n"
                 f"# of these files -- do not edit.\n"
                 f"# READ AS SYSTEMVERILOG, WITH `+define+SYNTHESIS`. firtool\n"
                 f"# emits packed multidimensional arrays and assignment\n"
                 f"# patterns outside any `ifdef`, so Verilog-2001 rejects\n"
                 f"# this; sv2v or `analyze -format sverilog` is required.\n")
        if drop:
            fh.write(f"# LOGIC-ONLY LIST. {len(drop)} memory-array module(s)\n"
                     f"# are replaced by port-compatible EMPTY modules, so the\n"
                     f"# area reported is the accelerator's LOGIC alone. The\n"
                     f"# stubs are NOT optional: merely omitting a module does\n"
                     f"# not give DC a black box, it gives DC an unresolved\n"
                     f"# reference and the link fails (LINK-5).\n")
            for f in sorted(drop):
                fh.write(f"#   {f} -> {(stubs or {}).get(f, '(NO STUB)')}\n")
            fh.write(f"# The figure keeps everything that DRIVES the memories\n"
                     f"# -- address generation, enables, write masks -- and\n"
                     f"# excludes the arrays and the arrays' own interfaces.\n")
        fh.write("# '#' comments and a leading '!' excludes a file.\n")
        for f in order:
            fh.write(f"{(stubs or {})[f]}\n" if f in drop else f"{f}\n")
    return kept


def readme(dest, key, cfg, modules, mems, bits, hdr, commit, chipyard_sha,
           gemmini_sha, missing):
    mem_files = sorted(m for m in modules if MEM_ARRAY.match(m))
    lines = [f"# Gemmini accelerator, int8/int32, DIM={hdr.get('DIM', '?')}, "
             f"capacity-matched", "",
             "Chisel-elaborated RTL for the ASIC synthesis-only handoff",
             "(DC, freepdk-45nm `view-standard`, memories as flip-flops, no",
             "P&R) -- the **same flow and the same settings** our own variants",
             f"go through, so the areas compare. The top module is `{TOP}`.",
             "",
             "## Two things this export needs that ours did not", "",
             "1. **sv2v IS needed here** -- or a SystemVerilog reader. \"Chisel",
             "   emits Verilog\" is not true of firtool 1.75.0: `CounterFile`,",
             "   `LoopMatmulStC` and `RRArbiter` use packed multidimensional",
             "   arrays (`wire [7:0][6:0] _GEN_1`) and assignment patterns",
             "   (`'{3'h5, 3'h0, ...}`) **outside any `ifdef`**. Verilator",
             "   rejects all four file lists under `--language 1364-2001` with",
             "   6-7 syntax errors and accepts them as SystemVerilog. So either",
             "   set `normalize_rtl: True` and let sv2v convert, or read with",
             "   `analyze -format sverilog`. The files are named `.sv` for that",
             "   reason; the memory models, which are chipyard's and are plain",
             "   Verilog-2001, are `.v`.",
             "2. **`SYNTHESIS` must be defined** when the files are read. With",
             "   it, `plusarg_reader` collapses to `assign out = DEFAULT` and",
             "   the randomisation scaffolding disappears; without it DC meets",
             "   `$value$plusargs` inside an `initial` block and 173 `logic`",
             "   declarations under `ENABLE_INITIAL_REG_`.", "",
             "The **top-module check is fine**: `module Gemmini(` carries a",
             "trailing comment, not an attribute, so the collector's",
             "`^\\s*module\\s+<top>` pattern matches -- the defect that forced",
             "`normalize_rtl: False` on the Vitis export does not apply here.",
             "",
             "Verified on this host: all four file lists (both configurations x",
             "full and logic-only) pass `verilator --lint-only -DSYNTHESIS",
             "--top-module Gemmini` with no error.", "",
             f"This is the opponent for **{cfg['against']}**.", "",
             "## What is inside the boundary", "",
             "The cut is the transitive closure of the module instantiation",
             f"graph from `{TOP}`, computed from the elaborated Verilog -- not a",
             f"hand-picked list. {len(modules)} modules, {len(mem_files)} of",
             "them memory arrays.", ""]
    for what, why in INSIDE:
        lines.append(f"- **{what}** -- {why}")
    lines += ["", "## What was cut, and why", "",
              "Chipyard elaborates an SoC. Our design is an accelerator with no",
              "host, no cache and no core, so everything below is outside the",
              "comparable unit; leaving any of it in would compare a CPU with a",
              "matrix unit.", ""]
    for what, why in CUT:
        lines.append(f"- **{what}** -- {why}")
    lines += ["",
              "### The one ambiguous cut: Gemmini's own TLB", "",
              "`FrontendTLB` is **inside** this export, because it is inside the",
              "`Gemmini` module: Gemmini's DMA issues virtual addresses and the",
              "4-entry TLB translates them. Ours takes physical addresses over",
              "AXI and has no translation at all, so a reader may reasonably",
              "want it out. It is not cut here, because cutting it would mean",
              "editing Gemmini's RTL; instead **report it separately**: DC's",
              "`report_area -hierarchy` gives the `FrontendTLB` instance's area",
              "directly, and both the with-TLB and without-TLB figures should be",
              "quoted. The modules are " +
              ", ".join(f"`{m}`" for m in TLB_MODULES) + ".", "",
              "### Kept although unexercised, and why", "",
              "- **The conv pipeline** (`LoopConv*`, `Im2Col`, `PixelRepeater`,",
              "  `ZeroWriter`) and **the output-stationary datapath**",
              "  (`dataflow = BOTH`) are never used by a GEMM. They are kept",
              "  because removing them is tuning the opponent; both make Gemmini",
              "  look bigger, which is the direction that does not flatter us.",
              "- **The fp32 scaling pipeline** (`MulAddRecFN_e8_s24`,",
              "  `INToRecFN_*`, `RecFNToIN_*`, `RoundAnyRawFNToRecFN_*`) comes",
              "  from `defaultConfig`'s `mvin_scale_args`, which are `Float`.",
              "  Gemmini's default int8 accelerator carries an fp32 multiplier",
              "  per mvin lane and in the accumulator scale; we have none. Same",
              "  reasoning, same direction -- but this one is worth naming in",
              "  any area quote, because it is a real block and not a corner.",
              "- **`TLMonitor_43`..`_47`** are TileLink protocol assertions with",
              "  no outputs. DC optimises them to nothing; they are kept so the",
              "  file list is the closure and not the closure minus a judgement.",
              "", "## Configuration", ""]
    for k, v in cfg["params"].items():
        lines.append(f"- `{k}` = {v}")
    lines += ["",
              "The delta from Gemmini's own `defaultConfig` is **four fields**:",
              "`meshRows`/`meshColumns` (the array size, which is the point),",
              "`has_training_convs = false`, and `sp_capacity`/`acc_capacity`.",
              "`tileRows`/`tileColumns` also appear in the `copy()` at the values",
              "they already hold. Everything else -- `spad_read_delay`,",
              "`tile_latency`, `mesh_output_delay`, `dataflow`, the reservation",
              "station and queue depths, `max_in_flight_mem_reqs` -- stays at",
              "Gemmini's defaults.", "",
              "### Why the capacity was matched, and what that gives up", "",
              "Stock Gemmini has a **256 KiB scratchpad and a 64 KiB",
              "accumulator**. Both sides are synthesised with",
              "`sram_mode='none'`, so a stock Gemmini would put **2,621,440",
              "flip-flops** on the scale where our whole T=4 design has 200,561",
              "sequential cells: the number would measure the memory treatment",
              "and nothing else.", "",
              "8 KiB of scratchpad and 4 KiB of accumulator -- 98,304 bits --",
              "is what this carries instead. 4 KiB is the smallest accumulator",
              "inside the envelope the cycle-neutrality sweep in",
              "`designs/gemmini_comparison.rst` already covers; 2 KiB is legal",
              "but sits exactly on the binding half-capacity limit for a 16x16",
              "int32 tile, where the tiling search has never been run.", "",
              "**Against which of our builds that is a match, measured from the",
              "RTL rather than from the prose:**", "",
              "| our variant | operand + accumulator storage | vs Gemmini's 12 KiB |",
              "| --- | --- | --- |",
              "| `T4_MAXDIM16_shipped_baseline` | 2.5 KiB (0.25 + 0.25 + 2.0) | Gemmini has **4.8x** |",
              "| `T4_MAXDIM64_shipped` | 10.1 KiB (4 + 4 + 2.1) | Gemmini has 1.19x |",
              "| `T8_MAXDIM64` | 12.25 KiB (4 + 4 + 4.25) | **matched to 2 %** |", "",
              "The 4 KiB / 4 KiB / 2.1 KiB figures this project quotes for its",
              "own memories describe **`MAXDIM=64`**, not the `MAXDIM=16`",
              "baseline whose 1,136,598 is published: at `MAXDIM=16` the",
              "scratchpad and vector registers are 64 rows of 32 bits each,",
              "256 bytes apiece. So the capacity match is near-exact against",
              "`T8_MAXDIM64` and against `T4_MAXDIM64_shipped`, and it is",
              "**4.8x in Gemmini's disfavour** against the MAXDIM=16 baseline.",
              "That is why the logic-only figure, not the full one, is the",
              "headline.", ""]
    lines += [
              "**This is legitimate only because the shrink is cycle-neutral.**",
              "`tiled_matmul_auto`'s own tiling search issues exactly one",
              "`loop_ws` at every published shape from 256/64 KB down to 4/4 KB,",
              "so none of the five Gemmini cycle numbers moves. It stops being",
              "neutral at 32x32x32, so this argument does not extend to larger",
              "shapes without being re-measured.", "",
              "What it gives up: the 256 KiB scratchpad is a **real capability**",
              "that this export deliberately removes. The area below answers",
              "*what does the same amount of local memory cost in each design*,",
              "not *what does the shipped Gemmini cost*. Anyone quoting it",
              "against a published Gemmini area is quoting the wrong thing.", "",
              "## The memories in this export", ""]
    for m, d in sorted(mems.items()):
        lines.append(f"- `{m}`: {d['depth']} x {d['width']} bits x "
                     f"{d['instances']} instances = {d['bits']:,} bits, "
                     f"ports `{d['ports']}`")
    lines += [f"- total memory-array storage: **{bits:,} bits** "
              f"({bits / 8192:.1f} KiB), which is {bits:,} flip-flops under",
              f"  `sram_mode='none'` -- against our T=4 design's 200,561",
              "  sequential cells. Stock Gemmini would be 2,621,440.", "",
              "They arrive as behavioural flop arrays (`reg [7:0] ram [0:N]`",
              "inside `split_*_ext`), which is what `sram_mode='none'` means on",
              "our side too. A second file list is provided so the same export",
              "yields a memory-free figure:", "",
              "| file list | what DC sees | what the area means |",
              "| --- | --- | --- |",
              "| `sv2v_manifest.f` | everything | accelerator + 12 KiB of flops |",
              "| `sv2v_manifest_nomem.f` | `mem`/`mem_0` as empty black boxes |"
              " accelerator LOGIC only |", "",
              "The logic-only figure is the one that is invariant to both the",
              "memory treatment and the capacity choice, so it is the safer of",
              "the two to quote. Our side needs the same pair to be comparable:",
              "the matching omission is every `*_RAM_*` module in",
              "`rtl_handoff/`.", "",
              "## Provenance", ""]
    lines += [f"- chipyard `{chipyard_sha}`, gemmini `{gemmini_sha}`",
              f"- elaborated by `make verilog CONFIG={cfg['config']}`",
              "- hardware read back out of the RTL's own memory geometry: " +
              ", ".join(f"`{k}`={v}" for k, v in sorted(hdr.items())) +
              " -- **not** from `gemmini_params.h`, which every elaboration",
              "  rewrites in place and which therefore describes whichever",
              "  config was elaborated last",
              f"- exported by `export_gemmini_rtl.py` at allo `{commit}`"]
    if cfg["cycles"]:
        lines += ["", "## Cycles these areas belong beside", "",
                  "Gemmini's accelerator-only window (`gemmini/allo_bare5.c`):",
                  "`rdcycle` -> 5 `config`s -> one hardware `loop_ws` -> `fence`",
                  "-> `rdcycle`, under Verilator. **Measured at stock capacity**",
                  "-- unchanged here by the cycle-neutrality argument above, but",
                  "not re-measured on this RTL.", ""]
        for k, v in cfg["cycles"].items():
            lines.append(f"- {k}: **{v}** cycles")
    if missing:
        lines += ["", "## Undefined modules", "",
                  "Nothing in the closure defines these; DC will treat them as",
                  "black boxes:", ""]
        lines += [f"- `{m}`" for m in missing]
    lines += ["", "`sv2v_manifest.f`, `sv2v_manifest_nomem.f` and",
              "`MANIFEST.json` are GENERATED from the module instantiation",
              "graph of these files; do not hand-edit them.", ""]
    with open(os.path.join(dest, "README.md"), "w") as fh:
        fh.write("\n".join(lines))


def sha(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def export(key):
    cfg = CONFIGS[key]
    src = os.path.join(GENERATED, f"chipyard.harness.TestHarness.{cfg['config']}")
    mods = collect(src)
    files, modules, missing = closure(mods)
    order = compile_order(files, top=TOP)
    counts = instance_counts(mods)
    mems, bits = mem_inventory(src, cfg["config"], modules, counts)
    hdr = hardware(mems)

    # ---- THE GUARDS, the same two `export_rtl.write_design` makes ----
    definers = [n for n, t in files.items()
                if re.search(r"^\s*module\s+%s\s*[(#;]" % re.escape(TOP), t, re.M)]
    if not definers:
        raise ExportError(
            f"no file in the closure defines `module {TOP}` ({len(files)} "
            f"files). A manifest built from a file set is self-consistent "
            f"however partial that set is, so this checks the INTENT instead.")
    resources = {"modules": len(modules), "files": len(order),
                 "memories": mems, "memory_bits": bits, "hardware": hdr,
                 "instances": {m: counts.get(m, 0)
                               for m in ("Tile", "PE", "MacUnit",
                                         "ScratchpadBank", "AccumulatorMem",
                                         "MulAddRecFN_e8_s24", "FrontendTLB")}}
    if not mems or not bits or not hdr:
        raise ExportError(
            f"empty resource record for {key}: memories={mems!r} bits={bits} "
            f"hardware={hdr!r}. An elaboration that produced no "
            f"`.top.mems.conf` entry for the scratchpad or the accumulator did "
            f"not finish -- the DC area would land beside a blank where our "
            f"variants have Vitis figures.")
    if bits > MAX_MEM_BITS:
        raise ExportError(
            f"{key} carries {bits:,} bits of memory array "
            f"({bits / 8192:.0f} KiB), over the {MAX_MEM_BITS // 8192} KiB "
            f"ceiling. As flip-flops that is {bits:,} registers against our "
            f"200,561, and the area would measure the memory treatment. "
            f"Elaborate `{cfg['config']}`, not a stock-capacity config.")
    want_dim = int(cfg["params"]["meshRows / meshColumns"].split()[0])
    if hdr["DIM"] != want_dim or hdr["ACC_DIM"] != want_dim:
        raise ExportError(
            f"{key} should be DIM={want_dim} but the scratchpad is "
            f"{hdr['BANK_ROWS']}x{hdr['DIM'] * 8} bits and the accumulator row "
            f"is {hdr['ACC_DIM'] * 32}: this RTL is a different configuration. "
            f"Re-elaborate `{cfg['config']}` -- and note that the DIM is read "
            f"from the memory geometry here, not from `gemmini_params.h`, "
            f"which every elaboration rewrites in place.")

    dest = os.path.join(DEST, cfg["dest"])
    os.makedirs(dest, exist_ok=True)
    for stale in os.listdir(dest):
        if stale.endswith((".v", ".sv", ".f")):
            os.remove(os.path.join(dest, stale))
    for name, text in files.items():
        with open(os.path.join(dest, name), "w") as fh:
            fh.write(text)
    nomem = sorted(n for n in order
                   if MEM_ARRAY.match(os.path.splitext(n)[0]))
    manifest(dest, "sv2v_manifest.f", order)
    logic = manifest(dest, "sv2v_manifest_nomem.f", order, skip=nomem)

    meta = {
        "top": TOP,
        "top_defined_in": definers[0],
        "chisel_config": cfg["config"],
        "config": cfg["params"],
        "files": len(order),
        "files_logic_only": len(logic),
        "layout": "flat",
        "manifests": {"full": "sv2v_manifest.f",
                      "logic_only": "sv2v_manifest_nomem.f"},
        "memory_array_files": nomem,
        "tlb_modules": TLB_MODULES,
        "undefined_modules": missing,
        "language": "sverilog",
        "read_options": ["+define+SYNTHESIS"],
        "sv2v_required": True,
        "resources": resources,
        "chipyard": sha(CHIPYARD),
        "gemmini": sha(os.path.join(CHIPYARD, "generators", "gemmini")),
        "allo": sha(ROOT),
    }
    with open(os.path.join(dest, "MANIFEST.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    readme(dest, key, cfg, modules, mems, bits, hdr, sha(ROOT),
           meta["chipyard"], meta["gemmini"], missing)
    print(f"{key}: {len(order)} files, {len(modules)} modules, "
          f"{bits:,} memory bits -> {dest}")
    return meta


def regenerate_manifests(dest):
    """Rewrite both file lists, and the memory stubs, for a directory that is
    already exported -- no chipyard elaboration needed.

    The stubs arrived after these directories shipped. Omitting a module's
    source does NOT give DC a black box; it gives DC an unresolved reference
    and the link fails (``Unable to resolve reference 'mem_ext' in 'mem'``,
    LINK-5). The logic-only run needs each omitted module to exist and be
    empty, and the empty module's header is copied verbatim from the real one
    so it cannot disagree about a port.
    """
    meta = json.load(open(os.path.join(dest, "MANIFEST.json")))
    files = {n: open(os.path.join(dest, n), errors="replace").read()
             for n in sorted(os.listdir(dest))
             if n.endswith((".v", ".sv")) and not n.endswith("_stub.v")}
    order = compile_order(files, top=TOP)
    dropped = [n for n in order if MEM_ARRAY.match(os.path.splitext(n)[0])]
    if sorted(dropped) != sorted(meta["memory_array_files"]):
        raise ExportError(
            f"{dest}: the files matching {MEM_ARRAY.pattern!r} are "
            f"{sorted(dropped)} but MANIFEST.json records "
            f"{sorted(meta['memory_array_files'])}. Regenerating against a "
            f"different set than the one already synthesised would silently "
            f"change what the logic-only area measures.")
    stubs = write_stubs(dest, files, dropped)
    manifest(dest, "sv2v_manifest.f", order)
    for name in ("sv2v_manifest_nomem.f", "sv2v_manifest_nomem_stubbed.f"):
        manifest(dest, name, order, skip=dropped, stubs=stubs)
    meta["manifests"] = {"full": "sv2v_manifest.f",
                         "logic_only": "sv2v_manifest_nomem.f",
                         "logic_only_alias": "sv2v_manifest_nomem_stubbed.f"}
    meta["memory_stubs"] = {k: stubs[k] for k in sorted(stubs)}
    with open(os.path.join(dest, "MANIFEST.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"{os.path.basename(dest)}: {len(order)} files, {len(dropped)} "
          f"stubbed -> " + ", ".join(sorted(stubs.values())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=sorted(CONFIGS), action="append")
    ap.add_argument("--manifests", action="store_true",
                    help="rewrite the file lists and stubs for directories "
                         "that already shipped; no elaboration needed")
    args = ap.parse_args()
    keys = args.only or sorted(CONFIGS)
    for key in keys:
        if args.manifests:
            regenerate_manifests(os.path.join(DEST, CONFIGS[key]["dest"]))
        else:
            export(key)


if __name__ == "__main__":
    main()
