# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the adder tree costs, and whether it is legal on standard cells.

Two questions, and the second one is the one this project has already been
burned by. `csynth_design` answers the first -- FF/LUT/BRAM/DSP and the
estimated clock, read out of the synthesis report exactly as `csynth_sweep.py`
reads TinyTPU's. The second is the ELAB-366 audit: Vitis will satisfy a loop
that writes one array from two places by emitting a **true dual-write-port
RAM**, while still naming the module `_1R1W`. An FPGA block RAM has two write
ports so this is free there; standard cells have no such primitive and Design
Compiler refuses the netlist outright. A unit that passes csynth is therefore
not yet a unit that passes the ASIC flow, and nothing but a scan of the
generated Verilog tells the two apart.

    python reduce_csynth.py                  # the default tree, csynth + audit
    python reduce_csynth.py 8:2 16:4 32:8    # RED_LANES:RED_GROUPS pairs
    python reduce_csynth.py --keep           # do not delete the project

The project is deleted as soon as it has been parsed and audited: the disk on
this host is the binding constraint. The report is copied out first, into
`csynth_reports/`, so a parsing bug costs a re-parse and not a re-synthesis.
"""

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

REPORTS = os.path.join(HERE, "csynth_reports")
VITIS = "/opt/xilinx/Vitis_HLS/2023.2/settings64.sh"
CLOCK_NS = 3.33

TCL = """open_project out.prj -reset
open_solution -reset solution1 -flow_target vivado
set_top %s
add_files kernel.cpp
set_part {xcu280-fsvh2892-2L-e}
create_clock -period %s
csynth_design
exit
"""


def parse(text):
    """FF/LUT/BRAM/DSP and the estimated clock. The two traps that have each
    cost a re-synthesis are `csynth_sweep.parse`'s, and this reads by the same
    rules: columns BY NAME, and only the `== Utilization Estimates` SUMMARY
    block, which ends at `+ Detail:`."""
    out = {"ok": False}
    start = text.find("== Utilization Estimates")
    if start < 0:
        return out
    block = text[start:]
    end = block.find("+ Detail:")
    if end > 0:
        block = block[:end]
    cols = None
    for ln in block.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) > 2 and cells[0] == "Name" and "LUT" in cells:
            cols = cells
            continue
        if not cols or not cells or cells[0] != "Total":
            continue
        for name, v in zip(cols[1:], cells[1:]):
            v = v.replace("~", "").replace("%", "").strip()
            if v.lstrip("-").isdigit():
                out[name] = int(v)
        out["ok"] = True
    clock = re.search(r"\|\s*ap_clk\s*\|\s*([\d.]+)\s*ns\s*\|\s*([\d.]+)\s*ns"
                      r"\s*\|\s*([\d.]+)\s*ns", text)
    if clock:
        out["target_ns"] = float(clock.group(1))
        out["estimated_ns"] = float(clock.group(2))
        out["uncertainty_ns"] = float(clock.group(3))
    # The tree's OWN row, out of the per-instance table: the rig's feeder and
    # sink and Vitis's five m_axi shims are not the unit, and quoting the
    # region's total as the tree's area is the mistake the ASIC page already
    # withdrew one ratio for.
    unit = re.search(r"\|\s*reduce_tree_\w*_U0\s*\|\s*reduce_tree\w*\s*\|"
                     r"\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", text)
    if unit:
        out["unit"] = dict(zip(("BRAM_18K", "DSP", "FF", "LUT"),
                               (int(g) for g in unit.groups())))
    return out


def loop(prj, top):
    """The tree loop's achieved II and iteration latency -- the closest thing
    this flow has to a MEASURED latency to hold the declared RED_DEPTH
    against. It is a schedule, not silicon, and it is one-sided: it says the
    RTL is no faster than this, not that anything books it correctly (see the
    UNIT_LATENCY row of ip/placeholders.py)."""
    reports = os.path.join(prj, "out.prj", "solution1", "syn", "report")
    if not os.path.isdir(reports):
        return None
    for name in os.listdir(reports):
        if not (name.startswith("reduce_tree") and "Pipeline" in name
                and name.endswith(".rpt")):
            continue
        text = open(os.path.join(reports, name), errors="replace").read()
        row = re.search(r"\|-\s*VITIS_LOOP[\w]*\s*\|\s*\d+\|\s*\d+\|"
                        r"\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", text)
        if row:
            return {"iteration_latency": int(row.group(1)),
                    "II_achieved": int(row.group(2)),
                    "II_target": int(row.group(3))}
    return None


# `reg [W-1:0] name [0:N-1];` -- a MEMORY, as against a scalar register with a
# part-select, which `data_buf[i*W +: W] <= ...` is and which is not a memory
# at all. Matching writes without this gate reports every bit-sliced register
# in the design.
_ARRAY_DECL = re.compile(
    r"^\s*reg\b[^;]*?\[[^\]]+\]\s*(\w+)\s*\[[^\]]+\]\s*;", re.M)
_ALWAYS = re.compile(r"always\s*@\s*\(\s*posedge\b")
_ARRAY_WRITE = re.compile(r"^\s*(\w+)\s*\[[^\]]+\]\s*(?:\[[^\]]+\]\s*)*<=")

# Vitis emits these for the region's own interfaces, identically in every
# design it generates, including the shipped TinyTPU that this flow has
# already taken through Design Compiler. They are not the unit's structure and
# a finding in one says nothing about the unit.
_INFRASTRUCTURE = ("_m_axi", "_s_axi", "_control_s_axi")


def write_ports(verilog):
    """How many clocked blocks write each declared memory in one module.

    More than one is the ELAB-366 shape: `Net 'ram[0][31]' or a directly
    connected net is driven by more than one source`. Counting BLOCKS rather
    than statements is deliberate -- two writes in one `always` block are one
    port, muxed; two blocks are two drivers."""
    arrays = set(_ARRAY_DECL.findall(verilog))
    if not arrays:
        return {}
    owners = {}
    for i, part in enumerate(_ALWAYS.split(verilog)):
        if i == 0:
            continue                       # before the first always block
        for line in part.splitlines():
            m = _ARRAY_WRITE.match(line)
            if m and m.group(1) in arrays:
                owners.setdefault(m.group(1), set()).add(i)
    return {name: len(blocks) for name, blocks in owners.items()}


def audit(prj):
    """Every generated Verilog module, scanned for an array written from more
    than one clocked block."""
    rtl = os.path.join(prj, "out.prj", "solution1", "impl", "verilog")
    if not os.path.isdir(rtl):
        rtl = os.path.join(prj, "out.prj", "solution1", "syn", "verilog")
    if not os.path.isdir(rtl):
        return None, [], []
    findings, skipped = [], []
    for name in sorted(os.listdir(rtl)):
        if not name.endswith(".v"):
            continue
        text = open(os.path.join(rtl, name), errors="replace").read()
        for array, ports in sorted(write_ports(text).items()):
            if ports > 1:
                where = (skipped if any(tag in name for tag in _INFRASTRUCTURE)
                         else findings)
                where.append((name, array, ports))
    return rtl, findings, skipped


def build(lanes, groups, prj):
    from allo.dataflow import customize
    from examples.accelerator.tinytpu_vitis.ip.reduce import (
        DotTree, ReduceParams)

    params = ReduceParams(RED_LANES=lanes, RED_GROUPS=groups,
                          DOT_MAX=max(8, lanes))
    tree = DotTree(params, name=f"dot_tree_{lanes}_{groups}")
    s = customize(tree.region)
    tree.schedule(s)
    s.build(target="vitis_hls", mode="csyn", project=prj, wrap_io=False,
            configs={"align_value": 64})
    return params, tree.architecture.name


def run(lanes, groups, keep):
    prj = os.path.join(HERE, f"reduce_{lanes}_{groups}.prj")
    shutil.rmtree(prj, ignore_errors=True)
    params, top = build(lanes, groups, prj)
    open(os.path.join(prj, "run.tcl"), "w").write(TCL % (top, CLOCK_NS))
    with open(os.path.join(prj, "csynth.log"), "w") as f:
        subprocess.call(["bash", "-lc",
                         f"source {VITIS} && cd {prj} && vitis_hls -f run.tcl"],
                        stdout=f, stderr=subprocess.STDOUT)
    report = os.path.join(prj, "out.prj", "solution1", "syn", "report",
                          f"{top}_csynth.rpt")
    os.makedirs(REPORTS, exist_ok=True)
    text = ""
    if os.path.exists(report):
        text = open(report, errors="replace").read()
        shutil.copy(report, os.path.join(REPORTS, f"{top}_csynth.rpt"))
    numbers = parse(text)
    rtl, findings, skipped = audit(prj)

    print(f"\n{top}: RED_LANES={params.RED_LANES} RED_GROUPS={params.RED_GROUPS} "
          f"RED_DEPTH={params.RED_DEPTH} RED_TAP_LEVEL={params.RED_TAP_LEVEL} "
          f"RED_ACC_BITS={params.RED_ACC_BITS}")
    if not numbers.get("ok"):
        print("  csynth FAILED -- see", os.path.join(prj, "csynth.log"))
    else:
        unit = numbers.get("unit", {})
        print("  reduce_tree alone: " + "  ".join(
            f"{k}={unit[k]}" for k in ("LUT", "FF", "DSP", "BRAM_18K")
            if k in unit))
        print("  whole dot_tree rig: " + "  ".join(
            f"{k}={numbers[k]}" for k in ("LUT", "FF", "DSP", "BRAM_18K", "URAM")
            if k in numbers))
        print(f"  clock: target {numbers.get('target_ns')} ns, estimated "
              f"{numbers.get('estimated_ns')} ns "
              f"(uncertainty {numbers.get('uncertainty_ns')} ns)")
        timing = loop(prj, top)
        if timing:
            print(f"  tree loop: iteration latency "
                  f"{timing['iteration_latency']} cycles at II="
                  f"{timing['II_achieved']}, against a declared RED_DEPTH of "
                  f"{params.RED_DEPTH} adder levels")
    if rtl is None:
        print("  ASIC audit: NO RTL EMITTED -- csynth did not get that far")
    elif findings:
        print("  ASIC audit: FAILED -- a memory written from two clocked "
              "blocks is the ELAB-366 shape:")
        for name, array, ports in findings:
            print(f"    {name}: {array} written from {ports} blocks")
    else:
        print(f"  ASIC audit: {len(os.listdir(rtl))} files, no memory written "
              f"from more than one clocked block"
              + (f" ({len(skipped)} in Vitis's own interface shims, not "
                 f"scored)" if skipped else ""))
    if not keep:
        shutil.rmtree(prj, ignore_errors=True)
    return numbers.get("ok") and rtl is not None and not findings


def main(argv):
    keep = "--keep" in argv
    pairs = [a for a in argv if ":" in a] or ["8:2"]
    ok = True
    for pair in pairs:
        lanes, groups = (int(x) for x in pair.split(":"))
        ok &= bool(run(lanes, groups, keep))
    print("\nREDUCE CSYNTH " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
