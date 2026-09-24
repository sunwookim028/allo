#!/usr/bin/env python3
"""Check the TinyTPU-isa ppa run against the criterion written before it.

No `allo` import, no Catapult, no network: it reads only files the tools write.
Adapted from ../ppa_mac16/check_ppa.py, whose criterion passed on zhang-21 and
was then shown to go red on three doctored runs.

Usage:  python3 check_ppa.py [solution_dir] [catapult_stdout.log]
Default solution dir: ./Catapult/tinytpu_isa.v1
Default log:          $HOME/ppa_tinytpu_stdout.log   (see RUNME.md: it must be
                      OUTSIDE the project, because Catapult owns and overwrites
                      ./catapult.log)
Exit code 0 = PASS, 1 = FAIL. Everything under "informative" is data to report
back whatever the verdict; none of it is part of the criterion.

WHAT IS DELIBERATELY NOT CHECKED
  * Any absolute power figure. There is no prior for TinyTPU -- this run IS the
    first data point -- so a threshold would be invented, not derived. Only
    "dynamic power is greater than zero" is checked, because zero is the known
    silent failure (a switching step that simulated nothing).
  * Latency and throughput from cycle.rpt: reported, never gating. A NEGATIVE
    latency is legitimate for a free-running thread, and this design is eight
    of them.
  * Area: reported, never gating.
  * `clobbered=` from the testbench: reported, never gating. It counts cells of
    `C` outside the program's result region that changed. That is a property of
    SCVerify's memory wrapper on a four-array-port design, which nothing has
    exercised before; the arithmetic is what `errors=` covers.
"""
import os
import re
import sys

TOP = "tinytpu_isa"

sol = sys.argv[1] if len(sys.argv) > 1 else os.path.join("Catapult", f"{TOP}.v1")
log = (sys.argv[2] if len(sys.argv) > 2
       else os.path.expanduser("~/ppa_tinytpu_stdout.log"))

fails, info = [], []

# --- 1. power.rpt exists and its first block has a Dynamic row above zero -----
power = os.path.join(sol, "power.rpt")
rows = {}
if not os.path.isfile(power):
    fails.append(f"1. no power report at {power}")
else:
    text = open(power, errors="replace").read()
    body = text.partition("Power Report")[2]
    # drop the rest of the "Power Report (uW)" title line, then stop at the end
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
        fails.append(f"1. dynamic power is {dyn} uW: the switching sim "
                     f"produced no activity")
    else:
        info.append(f"power: {tot:.2f} uW total = {dyn:.2f} dynamic "
                    f"+ {sta:.2f} static")

    # --- 2. the SAIF annotated the flops (activity came from simulation) ------
    m = re.search(
        r"Switching Activity.*?\n(?:.*\n)*?\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s*$",
        text.partition("Power Report")[0],
        re.M,
    )
    if not m:
        fails.append("2. no switching-activity annotation table in power.rpt")
    elif float(m.group(2)) < 100.0:
        fails.append(f"2. only {m.group(2)}% of flop outputs annotated "
                     f"(expected 100.00)")
    else:
        info.append(f"annotation: {m.group(2)}% flops / {m.group(3)}% user nets")

# --- 3. the testbench ran, on the RTL, and checked itself --------------------
# The marker is `TINYTPU TB errors=<n> clobbered=<m>`, printed once at the end
# of CCS_MAIN. It is matched with a regex on the NUMBER, not as the substring
# "errors=0", so that "errors=01" or a line inside a longer word cannot pass,
# and the per-case lines (which say "wrong=") cannot be mistaken for it.
if not os.path.isfile(log):
    fails.append(f"3. no Catapult stdout log at {log} "
                 f"(redirect it there, see RUNME.md)")
else:
    txt = open(log, errors="replace").read()
    m = re.search(r"TINYTPU TB errors=(\d+) clobbered=(\d+)", txt)
    if not m:
        fails.append("3. the testbench's summary line is not in the log: it "
                     "did not run to the end (or did not run)")
    elif m.group(1) != "0":
        fails.append(f"3. testbench self-check failed: {m.group(0)}")
    else:
        info.append(f"testbench: {m.group(0)}"
                    + ("" if m.group(2) == "0" else
                       "  <- clobbered is informative, not a failure"))
        for line in re.findall(r"^#?\s*TINYTPU case .*$", txt, re.M):
            info.append("  " + line.strip())
    # SCVerify's own verdict, anchored to the line it writes it on rather than
    # taken as a bare substring: Catapult's Info lines quote a lot of text.
    if not re.search(r"^#?\s*Info:\s*scverify_top\S*:\s*Simulation PASSED",
                     txt, re.M):
        if re.search(r"scverify_top\S*:\s*Simulation FAILED", txt):
            fails.append("3. SCVerify reported 'Simulation FAILED'")
        else:
            fails.append("3. no 'Simulation PASSED' from scverify_top in the "
                         "log (the RTL simulation did not finish)")

# --- 4. RTL was actually produced --------------------------------------------
rtl = os.path.join(sol, "rtl.v")
if not os.path.isfile(rtl):
    fails.append(f"4. {rtl} missing")
elif not re.search(rf"^\s*module\s+{TOP}\b", open(rtl, errors="replace").read(),
                   re.M):
    fails.append(f"4. {rtl} has no `module {TOP}`")

# --- informative only ---------------------------------------------------------
cyc = os.path.join(sol, "cycle.rpt")
if os.path.isfile(cyc):
    m = re.search(r"Design Total:\s+(\d+)\s+(-?\d+)\s+(-?\d+)",
                  open(cyc, errors="replace").read())
    if m:
        info.append(f"cycle.rpt: latency {m.group(2)}, throughput {m.group(3)}"
                    f" (informative; negative = free-running, not a failure)")
rtlr = os.path.join(sol, "rtl.rpt")
if os.path.isfile(rtlr):
    m = re.search(r"TOTAL AREA \(After Assignment\):\s+([\d.]+)",
                  open(rtlr, errors="replace").read())
    if m:
        info.append(f"rtl.rpt: total area {m.group(1)} score units (informative)")
saif = os.path.join(sol, "switching_v", "test.saif")
if os.path.isfile(saif):
    info.append(f"saif: {os.path.getsize(saif)} bytes (informative)")
# The biggest instances by total power, which is the per-unit breakdown the
# whole exercise is for. Ten rows, not three: TinyTPU has eight units.
inst = [(v["Total"][-1], k) for k, v in list(rows.items())[1:] if "Total" in v]
for tot, name in sorted(inst, reverse=True)[:10]:
    info.append(f"instance {name}: {tot:.2f} uW (informative)")

for line in info:
    print("   " + line)
if fails:
    print("FAIL")
    for f in fails:
        print("   " + f)
    sys.exit(1)
print("PASS")
