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


def compile_order(files):
    """Leaves first: a topological sort of the module instantiation graph.

    `files` maps a filename to its text. A file is emitted only after every
    file defining a module it instantiates. A cycle (which Verilog should not
    have) degrades to alphabetical for the files involved rather than
    raising -- an unusable manifest is worse than a slightly wrong order,
    and `sv2v` will say so.
    """
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
    top_file = defines.get(TOP)
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

    with open(os.path.join(dest, "sv2v_manifest.f"), "w") as fh:
        fh.write(f"# TinyTPU-isa RTL for sv2v. Top module: {TOP}\n"
                 f"# Paths are relative to THIS FILE's directory, which also\n"
                 f"# holds the .v files -- the layout is flat, no rtl/ subdir.\n"
                 f"# Dependency order, leaves first, {TOP} last.\n"
                 f"# Order is advisory for Verilog-2001: no file uses\n"
                 f"# `include or a macro defined elsewhere, so any order\n"
                 f"# compiles. GENERATED by export_rtl.py from the module\n"
                 f"# instantiation graph of these files -- do not edit.\n"
                 f"# '#' comments and a leading '!' excludes a file.\n")
        for n in order:
            fh.write(f"{RTL_SUBDIR}/{n}\n" if RTL_SUBDIR else f"{n}\n")
    meta = dict(meta or {})
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


def readme(dest, title, params, commit, cycles, resources, notes):
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
              "`-m_axi_latency 0` unless stated, bit-exact against numpy.", ""]
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
    print(f"exporting into {DEST}")
    print("Run `latency_grid.py --keep-verilog <dir>` to export the "
          "burst-widened variant from a build it makes anyway.")
