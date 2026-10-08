# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One row per Catapult build: manifest (latency/II/status per kernel), Catapult's
own area score and slack (rtl.rpt), wall time and peak RSS, and the u4d_check
verdict. usage: python3 u4d_cat_summary.py <scratch> <name>..."""
import glob
import json
import os
import re
import sys

S = sys.argv[1]
print("\t".join(["build", "clock", "kernels latency/ii/status", "cat_area", "cat_slack", "wall", "peak_MB", "verdict"]))
for n in sys.argv[2:]:
    log = open(os.path.join(S, n + ".log")).read() if os.path.exists(os.path.join(S, n + ".log")) else ""
    g = lambda pat, s: (re.search(pat, s).group(1) if re.search(pat, s) else "-")  # noqa: E731
    wall = g(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", log)
    peak = g(r"Maximum resident set size \(kbytes\): (\d+)", log)
    peak = f"{int(peak) // 1024}" if peak != "-" else "-"
    clk = g(r"'clock_period': ([\d.]+)", log)
    try:
        m = json.load(open(os.path.join(S, n + ".prj", "latency.json")))
        ks = "; ".join(f"{k} {u.get('latency')}/{u.get('ii')}/{u.get('status')[:5]}" for k, u in sorted(m["units"].items()))
    except (OSError, KeyError, ValueError):
        ks = "no manifest"
    rpt = glob.glob(os.path.join(S, n + ".prj", "build", "Catapult", "*.v1", "rtl.rpt"))
    area = slack = "-"
    if rpt:
        r = open(rpt[0]).read()
        area = g(r"Total Area Score:\s+([\d.]+)", r)
        slack = g(r"Slack:\s+(-?[\d.]+)", r)
    chk = os.path.join(S, n + ".check")
    v = "-"
    if os.path.exists(chk):
        c = open(chk).read()
        mm = re.search(r"^(UNIT-\S+|NO-RTL)\s+\S+ \S+ (?:catapult-rtl\[\S+\] )?(\d+/\d+)?", c, re.M)
        v = f"{mm.group(1)} {mm.group(2) or ''}".strip() if mm else "-"
        off = re.search(r"row->token offset \[([^\]]*)\]; CYCLE-FOR-CYCLE (\w+)", c)
        if off:
            v += f" offset [{off.group(1)[:12]}] c4c={off.group(2)}"
        st = re.search(r"stall cycles (-?\d+)", c)
        if st:
            v += f" stall={st.group(1)}"
    print("\t".join([n, clk, ks, area, slack, wall, peak, v]))
