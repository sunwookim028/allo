# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a configuration's generated Verilog for the ASIC synthesis handoff.

zhang-21 has a synthesis-only mflowgen flow (DC on freepdk-45nm, memories as
flip-flops, no place-and-route) that enters at `sv2v` and takes RTL through
git. This writes one directory per configuration containing the `.v`/`.sv`
files, an `sv2v_manifest.f` in dependency order, the top module name, and a
README with the configuration's parameters, the commit it was emitted from and
its measured cycles and resources -- so the DC area numbers can be compared
against something rather than standing alone.

**The manifest is GENERATED, not written by hand.** Order comes from the
module instantiation graph of the files themselves, topologically sorted with
the leaves first, so it is derived from the project's own output and cannot
drift from it. Vitis emits one module per file for a dataflow region and no
compile-order file of its own, which is why the graph is the only source.

    python export_rtl.py                    # the three handoff configurations
    python export_rtl.py --only ship16

Used as a library by `latency_grid.py`, which exports from a project it was
going to synthesize anyway rather than paying for a second synthesis.
"""

import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEST = os.path.join(ROOT, "examples", "accelerator", "tinytpu_vitis", "rtl_handoff")
TOP = "tinytpu_isa"

# `module <name>` and `<Name> <inst> (` / `<Name> #(... ) <inst> (`.
_MODULE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.M)
_INST = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:#\s*\([^;]*?\)\s*)?"
                   r"([A-Za-z_]\w*)\s*\(", re.M)
# Verilog keywords that the instantiation pattern would otherwise pick up.
_KW = {"module", "endmodule", "input", "output", "inout", "reg", "wire",
       "always", "assign", "if", "else", "case", "begin", "end", "for",
       "while", "initial", "function", "task", "generate", "parameter",
       "localparam", "integer", "real", "genvar", "logic", "bit", "return",
       "posedge", "negedge", "or", "and", "not", "signed", "unsigned",
       "default", "endcase", "wait", "repeat", "forever", "disable"}


def compile_order(files, top=None):
    """Leaves first: a topological sort of the module instantiation graph.

    `files` maps a filename to its text. A file is emitted only after every
    file defining a module it instantiates. A cycle (which Verilog should not
    have) degrades to alphabetical for the files involved rather than
    raising -- an unusable manifest is worse than a slightly wrong order,
    and `sv2v` will say so.

    `top` names the module to emit last; `export_gemmini_rtl.py` passes
    `Gemmini`, which is why it is a parameter rather than the module constant.
    """
    top = top or TOP
    defines, uses = {}, {}
    for name, text in files.items():
        mods = set(_MODULE.findall(text))
        for m in mods:
            defines[m] = name
        cand = {m for m, _inst in _INST.findall(text)}
        uses[name] = cand - _KW - mods
    order, seen, stack = [], set(), set()

    def visit(name):
        if name in seen:
            return
        if name in stack:           # a cycle: stop descending
            return
        stack.add(name)
        for mod in sorted(uses.get(name, ())):
            dep = defines.get(mod)
            if dep and dep != name:
                visit(dep)
        stack.discard(name)
        seen.add(name)
        order.append(name)

    # The top last, so start from everything else and finish at the top.
    top_file = defines.get(top)
    for name in sorted(files):
        if name != top_file:
            visit(name)
    if top_file:
        visit(top_file)
    return order


#: Files Vitis emits beside the RTL that are DATA, not compile units: memory
#: initialisation contents. They must travel with the design but must not go
#: in the manifest, or the Verilog parser will choke on them. Today the only
#: one is a four-line `.dat` for the sequencer's LOOP_DEPTH-deep `iv_now` RAM,
#: and no emitted module references it (verified: no `$readmemh` anywhere in
#: the export), so it is inert for a synthesis-only flow -- but it is copied
#: rather than dropped, because "inert today" is not a property to rely on
#: silently. An earlier version of this function filtered on `.v`/`.sv` and
#: lost it.
AUX_EXT = (".dat", ".mif", ".mem", ".hex", ".coe")


class ExportError(Exception):
    """An export that would have shipped something unsynthesisable."""


#: **The operand scratchpad and the accumulator** -- the two arrays a real
#: implementation would build out of SRAM macros. Dropping these modules from a
#: file list leaves their instantiations as empty black boxes, so DC reports the
#: design's LOGIC area with the memory treatment taken out of it. That figure is
#: the only one that survives `sram_mode='none'`, and it is the one the Gemmini
#: comparison leads with.
#:
#: **The criterion is semantic, not syntactic, and it is the same criterion on
#: both sides**: drop the scratchpad and the accumulator, keep everything else.
#: `export_gemmini_rtl.py` drops `mem_ext`/`mem_0_ext` (with their `split_*`
#: halves), which ARE Gemmini's scratchpad and accumulator and nothing else.
#: Here that is `spad` -- one module, instantiated twice, for the scratchpad and
#: the vector registers -- and `ar`. Everything else stays on BOTH sides: our
#: `rbA` DMA read buffers stay because Gemmini's DMA buffering lives in
#: `BeatMerger`/`XactTracker` as plain registers and stays; our sequencer's
#: `ib`/`iv_now`/`lp_start`/`lp_trip` stay because Gemmini's control-path RAMs
#: (`ram_2x147` and friends) stay.
#:
#: The regex is per-toolchain because the NAMES are per-toolchain -- Vitis
#: writes `<unit>_<array>_RAM_AUTO_1R1W`, firtool writes `mem_ext` -- but what
#: it selects is the same two arrays. A structural rule (any module declaring
#: an array-of-reg) was rejected: it would also take Gemmini's depth-2 queue
#: RAMs, which are flops in any implementation, and no depth threshold
#: separates those from our depth-4 `lp_trip`.
MEM_ARRAY = re.compile(r"_(spad|ar)_RAM_")

#: Bits each dropped module holds, for the record that makes the two sides
#: checkable against each other rather than merely parallel.
_MEM_GEOM = (re.compile(r"parameter\s+AddressRange\s*=\s*(\d+)"),
             re.compile(r"parameter\s+DataWidth\s*=\s*(\d+)"))


#: The layout, stated once because it was briefly in doubt: **flat**. Every
#: variant directory holds `sv2v_manifest.f`, `MANIFEST.json`, `README.md` and
#: the `.v` files side by side, with NO `rtl/` subdirectory, and the manifest
#: entries are bare filenames that resolve relative to the manifest's own
#: directory. All three variants have always been this shape; the only
#: inconsistency was against a separate, now-deleted `asic/shipped_t4` export
#: that put files under `rtl/` while still listing them bare -- two
#: conventions that do not compose. Flat is kept because it is what is
#: deployed and what the synthesis sessions are running against; moving it
#: while runs are queued would buy nothing.
RTL_SUBDIR = ""


def instance_counts(files, top=None):
    """How many times each module is instantiated in the whole `top` hierarchy.

    A module's own file says nothing about how many of it there are, and the
    scratchpad module is instantiated twice -- once as `spad`, once as the
    vector registers -- so a per-module sum would halve the storage it holds.
    """
    top = top or TOP
    defines = {}
    for name, text in files.items():
        for m in _MODULE.findall(text):
            defines[m] = name
    counts, queue, seen = {top: 1}, [top], {top}
    while queue:
        mod = queue.pop(0)
        text = files[defines[mod]]
        defined = set(_MODULE.findall(text))
        here = {}
        for m in _INST.finditer(text):
            used = m.group(1)
            if used in _KW or used in defined or used not in defines:
                continue
            head = text[m.start():m.start() + len(m.group(0))].split()
            if not head or head[0] != used:   # `for (...` / `if (...` backtracks
                continue
            here[used] = here.get(used, 0) + 1
        for child, n in here.items():
            counts[child] = counts.get(child, 0) + counts[mod] * n
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return counts


def memory_files(files):
    """The files holding the scratchpad and accumulator arrays. See MEM_ARRAY."""
    return sorted(n for n in files if MEM_ARRAY.search(n))


def memory_bits(files, dropped):
    """`{filename: bits}` for the dropped arrays, instances included."""
    counts = instance_counts(files)
    out = {}
    for n in dropped:
        mods = _MODULE.findall(files[n])
        depth = _MEM_GEOM[0].search(files[n])
        width = _MEM_GEOM[1].search(files[n])
        if mods and depth and width:
            out[n] = (int(depth.group(1)) * int(width.group(1))
                      * counts.get(mods[0], 0))
    return out


#: Lines that may still belong to a module's declaration after the port list
#: closes. Vitis writes non-ANSI headers -- `module M (a, b);` and then
#: `parameter`/`input`/`output` declarations, with the port widths written in
#: terms of the parameters -- so a stub must carry those too or it will not
#: compile. firtool writes ANSI headers and this matches nothing.
_DECL = re.compile(r"^\s*(parameter|localparam|input|output|inout)\b")


def stub_source(text, module):
    """A port-compatible empty module, copied VERBATIM from `text`.

    **Omitting a module's source does not give DC a black box, it gives DC an
    unresolved reference** -- ``Unable to resolve reference 'mem_ext' in
    'mem'. (LINK-5)``, and the link fails. A logic-only run needs the module to
    exist and be empty. The header is copied rather than written so the stub
    cannot disagree with the real module about a port name, a width or a
    parameter default; if the header cannot be found, this returns None and the
    caller refuses, because a hand-written stub is exactly the unauditable
    thing this whole comparison is trying not to depend on.
    """
    m = re.search(r"^[ \t]*module\s+%s\b" % re.escape(module), text, re.M)
    if not m:
        return None
    close = text.find(");", m.start())
    semi = text.find(";", m.start())
    end = (close + 2) if close != -1 and close < semi else (semi + 1)
    if end <= 0:
        return None
    head, rest = text[m.start():end], text[end:]
    kept = []
    for line in rest.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            kept.append(line)
            continue
        if _DECL.match(line):
            kept.append(line)
            continue
        break
    while kept and not kept[-1].strip():
        kept.pop()
    body = "\n".join(kept)
    return (f"// GENERATED by export_rtl.py -- a port-compatible EMPTY module.\n"
            f"// The logic-only synthesis run needs `{module}` to link and to\n"
            f"// contribute zero area. Its ports terminate here instead of\n"
            f"// dangling, so the area figure keeps everything that DRIVES the\n"
            f"// memory -- address generation, enables, write masks -- and\n"
            f"// excludes the array and the array's own interface. Header\n"
            f"// copied verbatim from the real module; do not edit.\n"
            f"{head}\n{body}\nendmodule\n")


def write_stubs(dest, files, dropped):
    """Write `<module>_stub.v` for each omitted file. Returns {orig: stub}."""
    out = {}
    for n in dropped:
        mods = _MODULE.findall(files[n])
        if not mods:
            raise ExportError(f"{n} defines no module; cannot stub it")
        src = stub_source(files[n], mods[0])
        if src is None:
            raise ExportError(
                f"could not find the declaration of `{mods[0]}` in {n}, so no "
                f"stub can be copied from it. Do NOT hand-write one: a stub "
                f"that disagrees with the real module about a port silently "
                f"changes what the logic-only area measures.")
        stub = f"{mods[0]}_stub.v"
        with open(os.path.join(dest, stub), "w") as fh:
            fh.write(src)
        out[n] = stub
    return out


def write_manifest(dest, name, order, skip=(), bits=None, stubs=None):
    drop = set(skip)
    with open(os.path.join(dest, name), "w") as fh:
        fh.write(f"# TinyTPU-isa RTL for sv2v. Top module: {TOP}\n"
                 f"# Paths are relative to THIS FILE's directory, which also\n"
                 f"# holds the .v files -- the layout is flat, no rtl/ subdir.\n"
                 f"# Dependency order, leaves first, {TOP} last.\n"
                 f"# Order is advisory for Verilog-2001: no file uses\n"
                 f"# `include or a macro defined elsewhere, so any order\n"
                 f"# compiles. GENERATED by export_rtl.py from the module\n"
                 f"# instantiation graph of these files -- do not edit.\n")
        if drop:
            total = sum((bits or {}).values())
            fh.write(f"# LOGIC-ONLY LIST. The scratchpad and the accumulator\n"
                     f"# are replaced by port-compatible EMPTY modules, so DC\n"
                     f"# reports this design's logic area with the memory\n"
                     f"# treatment taken out of it. The same criterion drops\n"
                     f"# mem_ext/mem_0_ext on Gemmini's side -- the scratchpad\n"
                     f"# and the accumulator, nothing else, on both. DMA\n"
                     f"# buffers and control-path RAMs stay on both sides.\n"
                     f"# Replaced here ({total:,} bits):\n")
            for n in sorted(drop):
                fh.write(f"#   {n}  ({(bits or {}).get(n, 0):,} bits)\n"
                         f"#     -> {(stubs or {}).get(n, '(NO STUB)')}\n")
            fh.write(f"# The figure keeps everything that DRIVES the memories\n"
                     f"# -- address generation, enables, write masks -- and\n"
                     f"# excludes the arrays and the arrays' own interfaces,\n"
                     f"# on both sides, by the same mechanical rule. Compare\n"
                     f"# logic-only against logic-only only.\n"
                     f"# The stubs are NOT optional: merely omitting a module\n"
                     f"# does not give DC a black box, it gives DC an\n"
                     f"# unresolved reference and the link fails (LINK-5).\n")
        fh.write("# '#' comments and a leading '!' excludes a file.\n")
        for n in order:
            if n in drop:
                fh.write(f"{RTL_SUBDIR}/{(stubs or {})[n]}\n" if RTL_SUBDIR
                         else f"{(stubs or {})[n]}\n")
                continue
            fh.write(f"{RTL_SUBDIR}/{n}\n" if RTL_SUBDIR else f"{n}\n")


def regenerate_manifests(dest):
    """Rewrite both file lists for an ALREADY-exported directory.

    The logic-only list arrived after these directories shipped, and re-running
    `csynth` to get it would re-emit RTL that is byte-identical at best and
    subtly different at worst. This regenerates both lists from the `.v` files
    already in `dest`, by the same code path the export uses, so the two lists
    are consistent with each other and with Gemmini's pair.

    **`_stub.v` files are OUR OWN OUTPUT and must not be read back as input.**
    This function runs on a directory it has already written into, and a stub
    is named after the module it replaces -- so it matches `MEM_ARRAY` just as
    the real file does. Reading them back made the run non-idempotent in two
    ways: the full list grew to 435 entries and defined the scratchpad and the
    accumulator TWICE, once from the stub and once from the real module, with
    the stub first (a duplicate-module elaboration error, and silently the
    wrong design if a tool takes the first definition); and each stub was
    re-stubbed from itself, gaining one blank line per run. Excluding them
    makes regenerating an unchanged directory a byte-identical no-op, which is
    the only way this is checkable.
    """
    files = {n: open(os.path.join(dest, n), errors="replace").read()
             for n in sorted(os.listdir(dest))
             if n.endswith((".v", ".sv")) and not n.endswith("_stub.v")}
    if not files:
        raise ExportError(f"{dest} holds no .v files")
    if not any(re.search(r"^\s*module\s+%s\s*[(#;]" % re.escape(TOP), t, re.M)
               for t in files.values()):
        raise ExportError(f"no file in {dest} defines `module {TOP}`")
    order = compile_order(files)
    dropped = memory_files(files)
    if not dropped:
        raise ExportError(
            f"{dest} has no module matching {MEM_ARRAY.pattern!r}: a logic-only "
            f"list that drops nothing is not a logic-only list, it is a copy "
            f"of the full one, and it would land beside Gemmini's as though "
            f"the two had been cut the same way.")
    bits = memory_bits(files, dropped)
    stubs = write_stubs(dest, files, dropped)
    write_manifest(dest, "sv2v_manifest.f", order)
    for name in ("sv2v_manifest_nomem.f", "sv2v_manifest_nomem_stubbed.f"):
        write_manifest(dest, name, order, skip=dropped, bits=bits, stubs=stubs)
    print(f"{os.path.basename(dest)}: {len(order)} files, "
          f"{len(dropped)} stubbed, {sum(bits.values()):,} bits removed -> " +
          ", ".join(f"{s} ({bits[n]:,})" for n, s in sorted(stubs.items())))
    return dropped, bits, stubs


def write_design(src_verilog, dest, meta=None, resources=None):
    """Copy `src_verilog`'s RTL and data files into `dest/rtl/`, write the
    manifest and MANIFEST.json, and REFUSE an export that is not synthesisable.

    The two assertions at the end are the point of this function, not
    decoration. A manifest generated from the file set it describes is
    self-consistent no matter what is missing from that set, so checking the
    manifest against the files proves nothing -- `T8_MAXDIM64` shipped 223
    files and 223 matching entries with **no `tinytpu_isa.v` at all**, because
    its `csynth` had not finished, and every internal check passed. What
    catches that is checking both against the INTENT: the named top module
    must actually be defined, and the resource record must not be empty (an
    empty one is the signature of a csynth whose report never appeared, which
    is the same failure).
    """
    rtl = os.path.join(dest, RTL_SUBDIR) if RTL_SUBDIR else dest
    os.makedirs(rtl, exist_ok=True)
    names = sorted(f for f in os.listdir(src_verilog)
                   if f.endswith((".v", ".sv")))
    if not names:
        raise ExportError(f"{src_verilog} contains no .v/.sv files")
    aux = sorted(f for f in os.listdir(src_verilog) if f.endswith(AUX_EXT))
    files = {}
    for n in names:
        with open(os.path.join(src_verilog, n), errors="replace") as fh:
            files[n] = fh.read()
        shutil.copyfile(os.path.join(src_verilog, n), os.path.join(rtl, n))
    for n in aux:
        shutil.copyfile(os.path.join(src_verilog, n), os.path.join(rtl, n))
    order = compile_order(files)

    # ---- THE GUARDS ----
    definers = [n for n, t in files.items()
                if re.search(r"^\s*module\s+%s\s*[(#;]" % re.escape(TOP),
                             t, re.M)]
    if not definers:
        raise ExportError(
            f"no file in {src_verilog} defines `module {TOP}` "
            f"({len(names)} .v files present, ending at {names[-1]!r}). "
            f"This is what an unfinished csynth looks like: the RTL is "
            f"partial and the manifest built from it is self-consistent "
            f"anyway. Re-run the synthesis; do not hand-add a top.")
    if not resources:
        raise ExportError(
            f"no resource record for {dest}: an empty `resources` means the "
            f"csynth report was never produced, which usually means the same "
            f"unfinished csynth. The DC area would land beside a blank where "
            f"the other variants have Vitis figures.")

    dropped = memory_files(files)
    bits = memory_bits(files, dropped)
    stubs = write_stubs(dest, files, dropped)
    write_manifest(dest, "sv2v_manifest.f", order)
    for name in ("sv2v_manifest_nomem.f", "sv2v_manifest_nomem_stubbed.f"):
        write_manifest(dest, name, order, skip=dropped, bits=bits, stubs=stubs)
    meta = dict(meta or {})
    meta["manifests"] = {"full": "sv2v_manifest.f",
                         "logic_only": "sv2v_manifest_nomem.f",
                         "logic_only_alias": "sv2v_manifest_nomem_stubbed.f"}
    meta["memory_array_files"] = sorted(dropped)
    meta["memory_stubs"] = {k: stubs[k] for k in sorted(stubs)}
    meta["memory_bits"] = sum(bits.values())
    meta["top"] = TOP
    meta["top_defined_in"] = definers[0]
    meta["files"] = len(order)
    meta["layout"] = "flat" if not RTL_SUBDIR else RTL_SUBDIR
    meta["aux_files"] = aux
    meta["resources"] = resources
    # A module loading memory contents at elaboration would make an aux file
    # load-bearing; today none does, and this records which it was.
    meta["readmemh_users"] = sorted(
        n for n, t in files.items() if "readmemh" in t)
    with open(os.path.join(dest, "MANIFEST.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return meta


def readme(dest, title, params, commit, cycles, resources, notes,
           invocation=None):
    """`invocation` is the shell line these cycles were measured with, and it
    is REQUIRED.

    A cycle row carried over from a previous export, or measured at whichever
    defaults the shell happened to hold, is this project's most-repeated
    mistake: `TPU_MAXDIM` defaults to 64 and not 16, and `TPU_QD` defaulted to
    8 until `63ee6ec7` and to 16 after it, so "I ran cosim" does not name a
    configuration. Printing the command beside the numbers makes the two
    checkable against each other by anyone reading the file, and refusing
    without it means no future export can quietly omit it.
    """
    if not invocation:
        raise ExportError(
            "readme() needs the exact invocation the cycles were measured "
            "with. Numbers without the command that produced them are how "
            "this directory came to quote a pre-QD=16 row.")
    lines = [f"# {title}", "",
             "Vitis HLS 2023.2 generated Verilog for the ASIC synthesis-only",
             "handoff (DC, freepdk-45nm, memories as flip-flops, no P&R).",
             "Enter the flow at `sv2v` with `sv2v_manifest.f`; the top module",
             f"is `{TOP}`.", "",
             "## Configuration", ""]
    for k, v in params.items():
        lines.append(f"- `{k}` = {v}")
    lines += ["", f"Emitted from allo commit `{commit}`.", "",
              "## Measured on this configuration", "",
              "Cycles are Vitis `cosim` (xsim), `ap_start` to `ap_done`,",
              "`-m_axi_latency 0` unless stated, bit-exact against numpy.",
              "",
              "**Measured on THIS export**, with every variable that matters",
              "set explicitly -- `TPU_MAXDIM` defaults to 64, not 16, and",
              "`TPU_QD` defaults to 16 since `63ee6ec7`, so the defaults are",
              "not a configuration anyone can reconstruct from prose:", "",
              "```bash"] + invocation.strip().splitlines() + ["```", ""]
    for k, v in cycles.items():
        lines.append(f"- {k}: **{v}** cycles")
    lines += ["", "Vitis `csynth` on `xcu280-fsvh2892-2L-e`, 3.33 ns target:", ""]
    for k, v in resources.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## What an ASIC number from this can and cannot say", "",
              "The agreed scope is **DC synthesis only**: memories as",
              "flip-flops (`sram_mode='none'`), FreePDK-45nm with",
              "`view-tiny`, `topographical=True`. That yields **relative cell",
              "area** and nothing else -- no absolute area, no routed timing,",
              "no power, no DRC or LVS. The FPGA figures above are the",
              "reference point the design is known by, not something an ASIC",
              "flow should reproduce: a 45 nm standard-cell area has no",
              "relationship to a BRAM count.", "",
              "- **Flip-flop memories are not this design's memories.** On",
              "  the FPGA the scratchpad, vector registers and accumulator",
              "  are block RAM; mapped to flops they will **dominate the cell",
              "  area**, and the resulting number then says more about the",
              "  memory treatment than about the datapath. A comparison",
              "  against any flow that used provided SRAM macros needs the",
              "  same memory treatment on both sides.",
              "- **The useful comparison is between our own variants**,",
              "  synthesised identically -- the shipped design against the",
              "  burst-widened candidate against T=8 -- and not between these",
              "  and another project's numbers.",
              "- **There is no testbench here.** The design's testbench is",
              "  C++, generated per shape by `cosim.py`, and drives the",
              "  design through its AXI interfaces; it is not synthesisable",
              "  and would not help a synthesis-only flow.", ""]
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes]
    lines += ["", "`sv2v_manifest.f` and `MANIFEST.json` are generated by",
              "`examples/accelerator/tinytpu_vitis/export_rtl.py` from the",
              "module instantiation graph of these files. Vitis emits no",
              "compile-order file for a dataflow region, so the graph is the",
              "only source; do not hand-edit either.", ""]
    with open(os.path.join(dest, "README.md"), "w") as fh:
        fh.write("\n".join(lines))


def commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def build_and_export(name, env_extra, dest, shapes):
    """csynth one configuration, export its Verilog, delete the project."""
    prj = os.path.join(HERE, f"export_{name}.prj")
    env = dict(os.environ, TPU_PRJ=prj, **env_extra)
    for v in ("TPU_TB", "TPU_SET", "TPU_SHAPES", "TPU_AXI_LATENCY"):
        env.pop(v, None)
    child = (
        "import os, sys; sys.path.insert(0, %r)\n"
        "from examples.accelerator.tinytpu_vitis import cosim\n"
        "from examples.accelerator.tinytpu_vitis.microarch_isa import "
        "tinytpu_isa, schedule\n"
        "from allo.dataflow import customize\n"
        "prj = cosim.PRJ\n"
        "s = customize(tinytpu_isa); schedule(s)\n"
        "s.build(target='vitis_hls', mode='csyn', project=prj, wrap_io=False,"
        " configs={'align_value': 64})\n"
        "cosim.patch_axi_depths(prj)\n"
        "open(os.path.join(prj, 'tb.cpp'), 'w').write('int main(){return 0;}')\n"
        "cosim.vitis(prj, cosim.TCL_SYN, 'csynth.log')\n"
        "print('CSYNTH DONE')\n" % ROOT)
    try:
        subprocess.run([sys.executable, "-c", child], env=env,
                       capture_output=True, text=True)
        src = os.path.join(prj, "out.prj/solution1/syn/verilog")
        rpt = os.path.join(
            prj, "out.prj/solution1/syn/report/tinytpu_isa_csynth.rpt")
        res = {}
        if os.path.exists(rpt):
            sys.path.insert(0, HERE)
            from csynth_sweep import parse
            res = parse(open(rpt, errors="replace").read())
        meta = write_design(src, dest, meta={"config": env_extra},
                            resources=res)
        return meta, res
    finally:
        shutil.rmtree(prj, ignore_errors=True)


if __name__ == "__main__":
    if "--manifests" in sys.argv:
        targets = [a for a in sys.argv[1:] if not a.startswith("-")]
        if not targets:
            targets = sorted(d for d in os.listdir(DEST)
                             if os.path.isdir(os.path.join(DEST, d)))
        for t in targets:
            regenerate_manifests(t if os.path.isabs(t)
                                 else os.path.join(DEST, t))
        sys.exit(0)
    print(f"exporting into {DEST}")
    print("Run `latency_grid.py --keep-verilog <dir>` to export the "
          "burst-widened variant from a build it makes anyway.")
    print("Run `export_rtl.py --manifests [dir ...]` to rewrite both file "
          "lists for directories that already shipped.")
