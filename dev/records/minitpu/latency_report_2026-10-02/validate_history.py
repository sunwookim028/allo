# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catapult's latency manifest against the Verilator-measured RTL latency of
every historical U1 Catapult build still on disk (no Catapult run).

    $ALLO_PYTHON validate_history.py <measured.txt>... -- <project-or-dir>...

<measured.txt>: cmp_rtl.py output (``RTL-CMP <unit> <name>: ... latency
catapult {L: n} ...`` for the stream shape at ready period 0, ``bare: measured
latency L`` for Wire). <project>: a Catapult project dir (``<name>.prj`` or
``<name>``) holding ``build/Catapult/<top>.v1``. Prints one line per project
with both numbers and a verdict, then a summary.
"""
import ast, glob, os, re, sys

sys.path.insert(0, os.getcwd())
from allo.backend.catapult import catapult_latency_manifest  # noqa: E402

i = sys.argv.index("--")
meas_files, targets = sys.argv[1:i], sys.argv[i + 1:]
meas = {}
for fn in meas_files:
    for line in open(fn):
        m = re.match(r"RTL-CMP (\S+) (\S+): .*?latency catapult (\{.*?\}).*?([\d.]+) cyc/vector.*ready period (\d+)", line)
        if m and m.group(5) == "0":
            meas[m.group(2)] = (ast.literal_eval(m.group(3)), float(m.group(4)))
            continue
        m = re.match(r"RTL-CMP (\S+) (\S+): .*bare: measured latency (\d+)", line)
        if m:
            meas[m.group(2)] = ({int(m.group(3)): 0}, 1.0)
prjs = []
for t in targets:
    if glob.glob(os.path.join(t, "build", "Catapult", "*.v1")):
        prjs.append(t)
    else:
        prjs += sorted(p for p in glob.glob(os.path.join(t, "*")) if glob.glob(os.path.join(p, "build", "Catapult", "*.v1")))
tally = {}
for p in prjs:
    name = os.path.basename(p.rstrip("/")).replace(".prj", "")
    if name not in meas:
        continue
    sol = glob.glob(os.path.join(p, "build", "Catapult", "*.v1"))[0]
    man = catapult_latency_manifest(sol, os.path.join(p, "build", "catapult.log"))
    us = {k: u for k, u in man["units"].items() if k not in ("src_0", "sink_0", "f_0", "drv_0", "col_0")}
    hist, cpv = meas[name]
    measured = sorted(hist)
    if len(us) == 1:
        (k, u), = us.items()
        rep = u["latency"]
    else:  # several kernels in series (staged): the region's latency is not one kernel's
        k, u = "+".join(sorted(us)), {"status": "composite", "ii": None}
        rep = None
    if u["status"] == "scheduled":
        verdict = "OK" if measured == [rep] and abs(cpv - (u["ii"] or 1)) < 1e-9 else "MISMATCH"
    elif u["status"] == "unreliable":
        verdict = "FLAGGED" + ("(measured varies)" if len(measured) > 1 or cpv != 1.0 else "(but measured steady)")
    else:
        verdict = u["status"].upper()
    tally[verdict] = tally.get(verdict, 0) + 1
    print(f"{name:34s} {k:10s} clk={man['clock_period_ns']} reported L={rep} II={u.get('ii')} "
          f"[{u['status']}{' ' + u.get('port_style', '') if 'port_style' in u else ''}]  measured L={measured if len(measured) > 1 else measured[0]} "
          f"cyc/vec={cpv:g}  {verdict}" + (f"  ({u.get('reason')})" if u.get("reason") else ""))
print("SUMMARY", tally)
