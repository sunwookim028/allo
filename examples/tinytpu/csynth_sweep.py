# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a bigger TinyTPU-isa build costs: MAXDIM -> resources, one csynth each.

`cosim.py` answers "how many cycles at this shape" on ONE build. This answers
the other half of the question the benchmark set needs: raising MAXDIM so that
steady-state shapes fit is only interesting if it is affordable, and nothing in
the repo measured that.

One `csynth_design` per MAXDIM, in a project this script deletes as soon as it
has parsed the numbers -- the disk on this host is the binding constraint, not
the CPU. The REPORT is copied out before the project goes, into
`csynth_reports/`, so a parsing bug costs a re-parse rather than a
re-synthesis. Resources come from the synthesis report rather than from
`cosim_design`, because they are the one thing csynth reports exactly and
cycles are the one thing it cannot (every loop bound is runtime data, so it can
only print a worst-case bound; see `cosim.py`).

    python csynth_sweep.py                 # T=4 at 16, 32, 48, 64
    python csynth_sweep.py 16 64           # T=4 at just those two
    python csynth_sweep.py 8:32 8:64       # T:MAXDIM pairs
    python csynth_sweep.py --reparse       # re-read csynth_reports/, no Vitis

A configuration is `T:MAXDIM`, and the two parameters cost completely
different things, so the table should be read down each column separately:
MAXDIM only resizes memories, while **T changes the SHAPE of the generated
region** -- T*T kernel instances and T, T*T stream arrays -- so it is the one
that buys peak MACs/cycle and the one that costs logic.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPORTS = os.path.join(HERE, "csynth_reports")
DEFAULT = ["4:16", "4:32", "4:48", "4:64"]


def parse(text):
    """FF/LUT/BRAM/DSP and the estimated clock out of a csynth report.

    Two traps, both of which have already cost a re-synthesis, which is why
    this is a named function with its own comment:

      * **the columns are read BY NAME, not by position.** The order is not
        the same in every Vitis report, and a positional version read the URAM
        column as FF and printed `FF: 0, LUT: 0, DSP: 438`.
      * **only the `== Utilization Estimates` SUMMARY block counts.** The
        Detail sections below it carry their own `Name`-headed tables with
        their own `Total` rows -- the `* Register:` table's Total is
        `FF 3, LUT 0` -- and taking the last `Total` in the file reads one of
        those. The block ends at `+ Detail:`.
    """
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
        if not cols or not cells or cells[0] not in ("Total", "Available"):
            continue
        row = {}
        for name, v in zip(cols[1:], cells[1:]):
            v = v.replace("~", "").replace("%", "").strip()
            if v.lstrip("-").isdigit():
                row[name] = int(v)
        norm = {("BRAM" if k.startswith("BRAM") else k): v
                for k, v in row.items()}
        if cells[0] == "Total":
            for k in ("FF", "LUT", "DSP", "BRAM", "URAM"):
                if k in norm:
                    out[k] = norm[k]
        else:
            out["avail"] = norm
    m = re.search(r"\|\s*ap_clk\s*\|\s*([\d.]+)\s*ns\s*\|\s*([\d.]+)\s*ns"
                  r"\s*\|\s*([\d.]+)\s*ns", text)
    if m:
        out["target_ns"] = float(m.group(1))
        out["estimated_ns"] = float(m.group(2))
        out["uncertainty_ns"] = float(m.group(3))
    out["ok"] = "LUT" in out and "FF" in out
    return out


# Run in a child process per MAXDIM: `microarch_isa` reads every parameter at
# import time (that is what makes the region's memories static), so one
# interpreter cannot hold two configurations.
CHILD = r'''
import json, os, shutil, sys
sys.path.insert(0, %(root)r)
from examples.tinytpu import cosim
from examples.tinytpu.csynth_sweep import parse, REPORTS
from examples.tinytpu.microarch_isa import (
    tinytpu_isa, schedule, MAXDIM, T, SPAD_ROWS, NVR, NAR, IMEM_SIZE)
from allo.dataflow import customize

prj = cosim.PRJ
s = customize(tinytpu_isa)
schedule(s)
s.build(target="vitis_hls", mode="csyn", project=prj, wrap_io=False,
        configs={"align_value": 64})
cosim.patch_axi_depths(prj)
# csynth needs the testbench file to exist even though it does not run it.
open(os.path.join(prj, "tb.cpp"), "w").write("int main() { return 0; }\n")
cosim.vitis(prj, cosim.TCL_SYN, "csynth.log")

out = {"MAXDIM": MAXDIM, "T": T, "SPAD": SPAD_ROWS, "NVR": NVR, "NAR": NAR,
       "IMEM": IMEM_SIZE, "ok": False}
rpt = os.path.join(prj, "out.prj/solution1/syn/report/tinytpu_isa_csynth.rpt")
if os.path.exists(rpt):
    os.makedirs(REPORTS, exist_ok=True)
    dst = os.path.join(REPORTS, "T%%d_MAXDIM%%d.rpt" %% (T, MAXDIM))
    shutil.copyfile(rpt, dst)
    out["report"] = os.path.relpath(dst, %(root)r)
    out.update(parse(open(rpt, errors="replace").read()))
print("RESULT " + json.dumps(out))
'''


def run(t, maxdim):
    prj = os.path.join(HERE, f"csynth_T{t}_{maxdim}.prj")
    env = dict(os.environ, TPU_T=str(t), TPU_MAXDIM=str(maxdim), TPU_PRJ=prj)
    for v in ("TPU_SHAPES", "TPU_TB", "TPU_SET", "TPU_SPAD", "TPU_NVR",
              "TPU_NAR"):
        env.pop(v, None)
    try:
        r = subprocess.run([sys.executable, "-c", CHILD % {"root": ROOT}],
                           env=env, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[7:])
        return {"T": t, "MAXDIM": maxdim, "ok": False,
                "error": (r.stderr.strip().splitlines() or ["?"])[-1][:300]}
    finally:
        # The project is tens to hundreds of MB and several agents run Vitis on
        # a host with ~36 GB free; it goes as soon as the report is copied out.
        shutil.rmtree(prj, ignore_errors=True)


def derived(t, maxdim):
    """The memory sizes `microarch_isa` derives, without importing it (a
    `--reparse` must work with no configuration in the environment)."""
    oprows = (maxdim // t) * maxdim
    return {"SPAD": oprows, "NVR": oprows, "NAR": max(128, 2 * maxdim + 8)}


def table(rows):
    print("\n    T  MAXDIM  spad   nvr  nar |      FF      LUT  BRAM  DSP |"
          " est ns    MHz  peak MAC/cyc")
    for r in rows:
        if not r.get("ok"):
            print(f"  {r.get('T', 0):3d} {r['MAXDIM']:6d}  -- FAILED: "
                  f"{r.get('error', 'no report')}")
            continue
        ns = r.get("estimated_ns", float("nan"))
        print(f"  {r.get('T', 0):3d} {r['MAXDIM']:6d} {r.get('SPAD', 0):5d} "
              f"{r.get('NVR', 0):5d} {r.get('NAR', 0):4d} | {r['FF']:7d} "
              f"{r['LUT']:8d} {r.get('BRAM', 0):5d} {r.get('DSP', 0):4d} | "
              f"{ns:6.3f} {1000.0 / ns:6.1f} {r.get('T', 0) ** 2:13d}")
    if rows and rows[0].get("avail"):
        a = rows[0]["avail"]
        print(f"  available on xcu280: FF {a.get('FF')}  LUT {a.get('LUT')}  "
              f"BRAM {a.get('BRAM')}  DSP {a.get('DSP')}")


def main(argv):
    if "--reparse" in argv:
        rows = []
        # Both name forms: `T<t>_MAXDIM<d>.rpt` and the earlier `MAXDIM<d>.rpt`
        # (which is always T=4, the only configuration that existed then).
        for f in glob.glob(os.path.join(REPORTS, "*MAXDIM*.rpt")):
            m = re.search(r"(?:T(\d+)_)?MAXDIM(\d+)\.rpt$", f)
            r = {"T": int(m.group(1) or 4), "MAXDIM": int(m.group(2)),
                 "report": f}
            r.update(derived(r["T"], r["MAXDIM"]))
            r.update(parse(open(f, errors="replace").read()))
            rows.append(r)
        rows.sort(key=lambda r: (r["T"], r["MAXDIM"]))
        for r in rows:
            print("  " + json.dumps(r), flush=True)
        table(rows)
        return 0
    rows = []
    for spec in (argv or DEFAULT):
        t, _, d = spec.rpartition(":")
        row = run(int(t or 4), int(d))
        rows.append(row)
        print("  " + json.dumps(row), flush=True)
    table(rows)
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
