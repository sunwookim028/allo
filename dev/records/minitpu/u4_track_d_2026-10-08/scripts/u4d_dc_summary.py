# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One row per DC run: cell area (total / comb / non-comb), registers (flip-flop
cells, by reset kind), worst slack, Fmax = 1 / (period - slack), DC wall and peak.

    python3 u4d_dc_summary.py <dc dir>   (reads <dc>/out/<run>/*.rpt and <dc>/out_<run>.log)
"""
import glob
import os
import re
import sys

d = sys.argv[1]
cols = ["run", "period_ns", "total_um2", "comb_um2", "noncomb_um2", "regs", "regs_reset", "slack_ns", "fmax_mhz",
        "dc_wall", "dc_peak_kB"]
print("\t".join(cols))
for o in sorted(glob.glob(os.path.join(d, "out", "*"))):
    run = os.path.basename(o)
    try:
        area = open(glob.glob(os.path.join(o, "*.area.rpt"))[0]).read()
        qor = open(glob.glob(os.path.join(o, "*.qor.rpt"))[0]).read()
        ref = open(glob.glob(os.path.join(o, "*.reference.rpt"))[0]).read()
    except IndexError:
        print(f"{run}\tmissing reports")
        continue
    g = lambda pat, s: (re.search(pat, s).group(1) if re.search(pat, s) else "-")  # noqa: E731
    log = open(os.path.join(d, f"out_{run}.log")).read()
    slacks = [float(x) for x in re.findall(r"Critical Path Slack:\s+(-?[\d.]+)", qor)]
    period = float(g(r"Critical Path Clk Period:\s+([\d.]+)", qor))
    slack = min(slacks) if slacks else float("nan")
    # flip-flops counted in the mapped netlist (the hierarchy below level 1 is
    # kept by ungroup -start_level 2, so the top's reference report hides them)
    regs = rst = 0
    net = open(glob.glob(os.path.join(o, "*.mapped.v"))[0]).read()
    for name in re.findall(r"^\s*(S?DFF\w*)\s+\S+\s*\(", net, re.M):
        regs += 1
        rst += int("R" in name[3:].split("_")[0] or "S" in name[3:].split("_")[0])
    print("\t".join(str(x) for x in [
        run, period, g(r"Total cell area:\s+([\d.]+)", area), g(r"Combinational area:\s+([\d.]+)", area),
        g(r"Noncombinational area:\s+([\d.]+)", area), regs, rst, f"{slack:.3f}", f"{1000 / (period - slack):.0f}",
        g(r"DC_EXIT \d+ (\S+)", log), g(r"peak (\d+) kB", log)]))
