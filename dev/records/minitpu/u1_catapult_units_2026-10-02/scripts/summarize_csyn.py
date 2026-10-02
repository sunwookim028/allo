"""One line per Catapult project: outcome, area score, slack, cycle.rpt, loop II.

    python summarize_csyn.py <prj> ...      (prints a tab-separated table)

Area is rtl.rpt's ``TOTAL AREA (After Assignment)`` (Catapult score units);
slack is rtl.rpt's first ``Slack:`` (the worst path); latency/throughput are
the ``Design Total`` row of cycle.rpt (``-1`` for a reset-action loop, pilot
F8); ``loop`` is the SCHD-43 message (II and flushing). ``errors`` are the
``# Error:`` lines of csyn.log, verbatim, first two.
"""
import glob, os, re, sys

print("\t".join(["name", "exit", "wall", "area", "slack_ns", "latency", "throughput", "loop", "errors"]))
for prj in sys.argv[1:]:
    name = os.path.basename(prj.rstrip("/")).replace(".prj", "")
    log = open(os.path.join(prj, "csyn.log"), errors="replace").read()
    m = re.search(r"CATAPULT_EXIT (\d+) WALL (\d+)s", log)
    ex, wall = (m.group(1), m.group(2)) if m else ("?", "?")
    v1 = sorted(glob.glob(os.path.join(prj, "build", "Catapult", "*.v1")))
    area = slack = lat = thr = "-"
    if v1 and os.path.exists(os.path.join(v1[0], "rtl.rpt")):
        r = open(os.path.join(v1[0], "rtl.rpt"), errors="replace").read()
        m = re.search(r"TOTAL AREA \(After Assignment\):\s+([\d.]+)", r)
        area = m.group(1) if m else "-"
        m = re.search(r"Slack:\s+(-?[\d.]+)", r)
        slack = f"{float(m.group(1)):.4f}" if m else "-"
    if v1 and os.path.exists(os.path.join(v1[0], "cycle.rpt")):
        c = open(os.path.join(v1[0], "cycle.rpt"), errors="replace").read()
        m = re.search(r"Design Total:\s+\d+\s+(-?\d+)\s+(\d+)", c)
        if m:
            lat, thr = m.group(1), m.group(2)
    loops = re.findall(r"Loop '([^']+)' is pipelined with initiation interval (\d+) and (no flushing|flushing)", log)
    loop = "; ".join(f"{l.split('/')[-1]} II={ii} {fl}" for l, ii, fl in loops) or "-"
    errs = [l.strip() for l in log.splitlines() if l.startswith("# Error:")][:2]
    print("\t".join([name, ex, wall, area, slack, lat, thr, loop, " | ".join(errs) or "-"]))
