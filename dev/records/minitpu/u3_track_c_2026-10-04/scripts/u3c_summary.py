# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One row per Catapult project: top, area score and worst slack (rtl.rpt),
per-kernel manifest latency/II, wall (the BUILD line in <prj>.log).

    python3 u3c_summary.py <prj> ...
"""
import glob, json, os, re, sys

print("\t".join(["project", "top", "area_score", "slack_ns", "manifest latency/ii per kernel", "wall", "exit"]))
for prj in sys.argv[1:]:
    name = os.path.basename(prj.rstrip("/")).replace(".prj", "")
    v1 = sorted(glob.glob(os.path.join(prj, "build", "Catapult", "*.v1")))
    top = os.path.basename(v1[0])[:-3] if v1 else "-"
    area = slack = "-"
    if v1 and os.path.exists(os.path.join(v1[0], "rtl.rpt")):
        rpt = open(os.path.join(v1[0], "rtl.rpt"), errors="replace").read()
        m = re.search(r"TOTAL AREA \(After Assignment\):\s+([\d.]+)", rpt) or re.search(r"Total Area Score:\s+([\d.]+)", rpt)
        area = m.group(1) if m else "-"
        sl = re.findall(r"Slack\s*:?\s+(-?[\d.]+)", rpt) or re.findall(r"slack\s+(-?[\d.]+)", rpt)
        if sl:
            slack = f"{min(float(x) for x in sl):.3f}"
    lat = "-"
    lj = os.path.join(prj, "latency.json")
    if os.path.exists(lj):
        man = json.load(open(lj))
        lat = "; ".join(f"{k}: {u.get('latency')}/{u.get('ii')} {u.get('status')}" + (f" pin {u['declared']}" if u.get("declared") is not None else "")
                        for k, u in sorted(man["units"].items()) if not k.startswith(("src_", "sink_")) or len(man["units"]) <= 3)
    wall = ex = "-"
    lg = prj.rstrip("/")[:-4] + ".log"
    if os.path.exists(lg):
        t = open(lg, errors="replace").read()
        m = re.search(r"^BUILD .*: ok (\d+s)", t, re.M)
        if m:
            wall, ex = m.group(1), "ok"
        else:
            m = re.search(r"^BUILD .*: (\w+Error): (.{0,90})", t, re.M)
            ex = (m.group(1) + ": " + m.group(2)) if m else "?"
            m2 = re.search(r"\((\d+)s\)\s*$", t, re.M)
            wall = m2.group(1) + "s" if m2 else "-"
    print("\t".join([name, top, area, slack, lat, wall, ex]))
