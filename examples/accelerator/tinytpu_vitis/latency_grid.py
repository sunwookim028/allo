# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does our RANKING of design variants survive the memory-latency range?

Every cycle count this design has ever published comes from Vitis cosim with
`-m_axi_latency 0`, i.e. a memory that answers immediately. That is a value we
CHOSE, not a memory system we measured -- there is no DRAM model anywhere in
the flow. So a variant that wins at 0 has been shown to win at one arbitrary
point of a knob, and for a variant whose whole benefit is *wider DMA bursts*
that is exactly the knob its advantage depends on.

This is not a hypothetical failure mode. MiniTPU's simulator ranked GEMM
templates BACKWARDS against a board A/B at its inferred 92-cycle value (a 45%
simulated cut measured 3.6% on hardware), and backwards again at 0. Two wrong
values, two wrong rankings, and the simulator was internally consistent and
confident both times; it was found only by running the A/B on real boards.

So the grid, per variant:

    (shipped, burst-widened) x (0, 16, 64, 88, 100 cycles) x (48^3, 64^3)

`-m_axi_latency` is a csynth-time setting, so each grid point is its own
synthesis; the project is deleted as soon as its numbers are read, because
/home is the binding constraint on this host.

**Report seconds, not only cycles.** Cycles say which simulator settings a
variant is good for; SECONDS say which real memory systems it is good for.
MiniTPU's ranking inverted between 213 ns and 490 ns, and it was the
nanosecond figure that let them say their board sits near the low end of that
window. Our own clock is not fixed either -- the shipped design estimates
2.431 ns and other variants have estimated 3.782 -- so a cycle-valued
threshold silently moves when the frequency does and a time-valued one does
not. Each row therefore carries the latency in ns at that build's OWN
estimated period, and the burst-widened variant's period is part of what is
being compared.

    python latency_grid.py                       # shipped, all 5 latencies
    python latency_grid.py --widen               # the burst-widened variant
    python latency_grid.py --lat 0,88 --shapes 64x64x64
"""

import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LATENCIES = [0, 16, 64, 88, 100]
SHAPES = ["48x48x48", "64x64x64"]
# Free space below which a grid point is refused rather than risked: several
# agents share this filesystem and a full one loses everyone's work.
MIN_FREE_GB = 12


def free_gb(path="/home"):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1 << 30)


def run_point(widen, lat, shapes, keep_verilog=None):
    tag = f"{'widen' if widen else 'ship'}_lat{lat}"
    prj = os.path.join(HERE, f"grid_{tag}.prj")
    env = dict(os.environ)
    env["TPU_MAXDIM"] = env.get("TPU_MAXDIM", "64")
    env["TPU_PRJ"] = prj
    env["TPU_AXI_LATENCY"] = str(lat)
    env["TPU_SHAPES"] = ",".join(shapes)
    if widen:
        env["TPU_DMA_WIDEN"] = "1"
    else:
        env.pop("TPU_DMA_WIDEN", None)
    env.pop("TPU_TB", None)
    out = {"variant": "widen" if widen else "shipped", "latency_cycles": lat}
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "cosim.py")],
                           env=env, capture_output=True, text=True, cwd=HERE)
        text = r.stdout
        # cosim.py prints "  16x16x16  cycles=879   TB ... mismatches = 0 / N"
        cyc, exact = {}, {}
        for line in text.splitlines():
            m = re.search(r"(\d+x\s*\d+x\s*\d+)\s+cycles=(\d+)", line)
            if m:
                shp = m.group(1).replace(" ", "")
                cyc[shp] = int(m.group(2))
                exact[shp] = "mismatches = 0" in line
        out["cycles"] = cyc
        out["bit_exact"] = exact
        out["ok"] = bool(cyc) and all(exact.values())
        if not out["ok"]:
            out["stderr"] = (r.stderr.strip().splitlines() or ["?"])[-1][:200]
        # The estimated period of THIS build, for the ns column.
        rpt = os.path.join(
            prj, "out.prj/solution1/syn/report/tinytpu_isa_csynth.rpt")
        if os.path.exists(rpt):
            sys.path.insert(0, HERE)
            from csynth_sweep import parse, REPORTS
            info = parse(open(rpt, errors="replace").read())
            out.update({k: info[k] for k in
                        ("FF", "LUT", "BRAM", "DSP", "estimated_ns")
                        if k in info})
            os.makedirs(REPORTS, exist_ok=True)
            shutil.copyfile(rpt, os.path.join(REPORTS, f"grid_{tag}.rpt"))
        if keep_verilog:
            export_verilog(prj, keep_verilog)
    finally:
        shutil.rmtree(prj, ignore_errors=True)
    return out


def export_verilog(prj, dest):
    """Copy the generated Verilog out before the project is deleted."""
    src = os.path.join(prj, "out.prj/solution1/syn/verilog")
    if not os.path.isdir(src):
        return
    sys.path.insert(0, HERE)
    from export_rtl import write_design
    write_design(src, dest)


def main(argv):
    widen = "--widen" in argv
    lats = LATENCIES
    shapes = SHAPES
    for i, a in enumerate(argv):
        if a == "--lat":
            lats = [int(x) for x in argv[i + 1].split(",")]
        if a == "--shapes":
            shapes = argv[i + 1].split(",")
    keep = None
    for i, a in enumerate(argv):
        if a == "--keep-verilog":
            keep = argv[i + 1]
    rows = []
    for lat in lats:
        if free_gb() < MIN_FREE_GB:
            print(f"  STOP: /home has {free_gb():.1f} GB free, below the "
                  f"{MIN_FREE_GB} GB floor. Remaining latencies {lats[lats.index(lat):]} "
                  f"NOT run.", flush=True)
            break
        row = run_point(widen, lat, shapes,
                        keep_verilog=keep if lat == 0 else None)
        rows.append(row)
        print("  " + json.dumps(row), flush=True)
    print(f"\n  variant={'widen' if widen else 'shipped'}")
    print("  lat_cyc   lat_ns  est_ns  BRAM  " +
          "  ".join(f"{s:>10s}" for s in shapes))
    for r in rows:
        ns = r.get("estimated_ns", float("nan"))
        print(f"  {r['latency_cycles']:7d} {r['latency_cycles'] * ns:8.1f} "
              f"{ns:7.3f} {r.get('BRAM', 0):5d}  " +
              "  ".join(f"{r.get('cycles', {}).get(s, 0):10d}" for s in shapes)
              + ("" if r.get("ok") else "   <-- NOT bit-exact"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
