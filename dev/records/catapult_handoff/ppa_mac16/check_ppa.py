#!/usr/bin/env python3
"""Check the ppa run against the criterion written before it. No allo import.

Usage:  python3 check_ppa.py [solution_dir] [catapult_stdout.log]
Default solution dir: ./Catapult/mac16.v1
Exit code 0 = PASS, 1 = FAIL. Everything it prints under "informative" is data to
report back, not part of the criterion.
"""
import os
import re
import sys

sol = sys.argv[1] if len(sys.argv) > 1 else os.path.join("Catapult", "mac16.v1")
log = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/ppa_mac16_stdout.log")

fails, info = [], []

# --- 1. power.rpt exists and its first block has a Dynamic row above zero ---------
power = os.path.join(sol, "power.rpt")
rows = {}
if not os.path.isfile(power):
    fails.append(f"1. no power report at {power}")
else:
    text = open(power).read()
    body = text.partition("Power Report")[2]
    # drop the rest of the "Power Report (uW)" title line, then stop at the end marker
    body = body.split("\n", 1)[1].split("End of Report")[0]
    block, first = None, None
    for line in body.splitlines():
        m = re.match(r"^\s+(Static|Dynamic|Total)((?:\s+-?[\d.]+){5})\s*$", line)
        if m:
            if block is not None:
                rows.setdefault(block, {})[m.group(1)] = [
                    float(x) for x in m.group(2).split()
                ]
            continue
        s = line.strip()
        if not s or set(s) <= set("- "):
            continue
        if "Memory" in line and "Combinational" in line:
            continue
        block = s
        if first is None:
            first = s                     # first block = the whole design
    dyn = rows.get(first, {}).get("Dynamic", [0] * 5)[-1]
    tot = rows.get(first, {}).get("Total", [0] * 5)[-1]
    sta = rows.get(first, {}).get("Static", [0] * 5)[-1]
    if dyn <= 0.0:
        fails.append(f"1. dynamic power is {dyn} uW: the switching sim produced no activity")
    else:
        info.append(f"power: {tot:.2f} uW total = {dyn:.2f} dynamic + {sta:.2f} static")

    # --- 2. the SAIF annotated the flops (activity came from simulation) ---------
    m = re.search(
        r"Switching Activity.*?\n(?:.*\n)*?\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s*$",
        text.partition("Power Report")[0],
        re.M,
    )
    if not m:
        fails.append("2. no switching-activity annotation table in power.rpt")
    elif float(m.group(2)) < 100.0:
        fails.append(f"2. only {m.group(2)}% of flop outputs annotated (expected 100.00)")
    else:
        info.append(f"annotation: {m.group(2)}% flops / {m.group(3)}% user nets")

# --- 3. the testbench ran and checked itself -------------------------------------
if not os.path.isfile(log):
    fails.append(f"3. no Catapult stdout log at {log} (redirect it there, see RUNME.md)")
else:
    txt = open(log, errors="replace").read()
    if "MAC16 TB errors=0" not in txt:
        got = re.search(r"MAC16 TB errors=\d+", txt)
        fails.append(f"3. testbench self-check missing or non-zero ({got.group(0) if got else 'no line'})")
    if "Simulation PASSED" not in txt:
        fails.append("3. SCVerify did not report 'Simulation PASSED'")

# --- 4. RTL was actually produced -------------------------------------------------
rtl = os.path.join(sol, "rtl.v")
if not (os.path.isfile(rtl) and "module mac16" in open(rtl, errors="replace").read()):
    fails.append(f"4. {rtl} missing or has no `module mac16`")

# --- informative only -------------------------------------------------------------
cyc = os.path.join(sol, "cycle.rpt")
if os.path.isfile(cyc):
    m = re.search(r"Design Total:\s+(\d+)\s+(-?\d+)\s+(-?\d+)", open(cyc).read())
    if m:
        info.append(f"cycle.rpt: latency {m.group(2)}, throughput {m.group(3)} (informative)")
rtlr = os.path.join(sol, "rtl.rpt")
if os.path.isfile(rtlr):
    m = re.search(r"TOTAL AREA \(After Assignment\):\s+([\d.]+)", open(rtlr).read())
    if m:
        info.append(f"rtl.rpt: total area {m.group(1)} score units (informative)")
for name, vals in list(rows.items())[1:4]:
    if "Total" in vals:
        info.append(f"instance {name}: {vals['Total'][-1]:.2f} uW (informative)")

for line in info:
    print("   " + line)
if fails:
    print("FAIL")
    for f in fails:
        print("   " + f)
    sys.exit(1)
print("PASS")
