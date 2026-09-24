#!/usr/bin/env python3
"""
csynth a dataflow region with the Catapult BUILD-SUBDIR workaround.

    python csyn_subdir.py <module> <region> [project_dir]
    e.g. python csyn_subdir.py router_rvn_chan router_rvn_c

WHY THIS EXISTS
---------------
`df.build(..., mode="csyn")()` runs Catapult with cwd == the project dir, i.e. the directory
that CONTAINS kernel.cpp. For Connections designs that is a known Catapult 2024.2 trap: the
`Connections::In`/`Out` ports degrade to raw sc_signals (CIN-124 fires on `in.rdy`), iomode
goes fixed, and scheduling explodes or dies with SCHD-30 "could not schedule even with
unlimited resources". `mode="cosim"` in allo/backend/hls.py already works around this by
running Catapult from a `cosb/` subdir; `mode="csyn"` (hls.py ~line 1235) does NOT.

So: let df.build WRITE the project (kernel.cpp + run.tcl), but do NOT call the module --
calling it is what launches the flat Catapult run. Launch Catapult ourselves from a build
subdir instead. run.tcl already begins `set sfd [file dir [info script]]` and refers to
"$sfd/kernel.cpp", so sources still resolve to the parent.

This is a workaround, not a fix. The real fix is to make mode="csyn" use a build subdir the
way mode="cosim" does.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys
import time
import shutil
import subprocess
import importlib

import allo.dataflow as df


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mod_name, region_name = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.abspath(__file__))
    prj = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        here, "csyn_out", f"{mod_name}_subdir")

    sys.path.insert(0, here)
    region = getattr(importlib.import_module(mod_name), region_name)

    os.makedirs(prj, exist_ok=True)
    print(f"[emit ] {mod_name}.{region_name} -> {prj}", flush=True)
    t0 = time.time()
    # Writes kernel.cpp + run.tcl. Deliberately NOT called -- calling runs Catapult flat.
    df.build(region, target="systemc", mode="csyn", project=prj)
    kcpp = os.path.join(prj, "kernel.cpp")
    with open(kcpp, encoding="utf-8") as f:
        nlines = sum(1 for _ in f)
    print(f"[emit ] OK  {nlines} lines of SystemC  ({time.time()-t0:.1f}s)", flush=True)

    # ALLO_DESIGN_TOP: synthesize a SUBMODULE instead of the whole region.
    #
    # WHY THIS MATTERS FOR EVERY AREA NUMBER WE REPORT. A @df.region() contains the design
    # kernels AND the testbench kernels (the injector/collector that own the host arrays),
    # plus the AlloMem memories backing those arrays. df.build points DESIGN_HIERARCHY at
    # the region top, so Catapult synthesizes all of it and the reported area includes the
    # harness. Comparing that against a reference whose top is the design alone (e.g.
    # MatchLib's WHVCRouterTop) overstates our area substantially -- the AlloMem shows up
    # by name in the Genus area report.
    #
    # Each kernel is emitted as its own SC_MODULE named <kernel>_0, whose ports are
    # Connections::In/Out bound to the region's channels, so it is a legal synthesis top
    # on its own. Setting ALLO_DESIGN_TOP=router_0 measures the design without the harness.
    #
    # The testbench is NOT removed -- csim and cosim still drive the full region. This only
    # changes what Catapult treats as the top, so verification and measurement can differ:
    #   csim/cosim  -> full region (needs drv/col to supply and check stimulus)
    #   area/Fmax   -> ALLO_DESIGN_TOP=<kernel>_0
    # Note cosim does NOT work against a submodule top: SCVerify wraps the design top and
    # the input<k>.data -> AlloMem stimulus path disappears. Run the two separately.
    design_top = os.environ.get("ALLO_DESIGN_TOP", "")
    if design_top:
        tcl = os.path.join(prj, "run.tcl")
        with open(tcl, encoding="utf-8") as f:
            s = f.read()
        old = f"directive set -DESIGN_HIERARCHY {region_name}"
        if old not in s:
            sys.exit(f"[csyn ] cannot retarget: '{old}' not in run.tcl")
        s = s.replace(old, f"directive set -DESIGN_HIERARCHY {design_top}", 1)
        with open(tcl, "w", encoding="utf-8") as f:
            f.write(s)
        print(f"[csyn ] DESIGN_HIERARCHY {region_name} -> {design_top} "
              f"(testbench kernels excluded from synthesis)", flush=True)

    syn = os.path.join(prj, "syn")          # <-- the whole point: a separate cwd
    os.makedirs(syn, exist_ok=True)
    cat = shutil.which("catapult") or os.path.join(
        os.environ.get("MGC_HOME", ""), "bin", "catapult")
    timeout = int(os.environ.get("ALLO_CSYN_TIMEOUT", "3600"))

    print(f"[csyn ] catapult in {syn} (timeout {timeout}s) ...", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(f"cd {syn}; {cat} -shell -f {prj}/run.tcl",
                           shell=True, capture_output=True, text=True, timeout=timeout)
        out, rc = (r.stdout or "") + (r.stderr or ""), r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") + \
              (e.stderr or b"").decode(errors="replace")
        rc = -1
    log = os.path.join(prj, "csyn.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(out)
    dt = time.time() - t0

    rtl = None
    for root, _, files in os.walk(syn):
        if "rtl.v" in files:
            rtl = os.path.join(root, "rtl.v")
            break
    verdict = "TIMEOUT" if rc == -1 else ("OK" if rc == 0 and rtl else "FAIL")
    print(f"[csyn ] {verdict}  {dt/60:.1f} min  rc={rc}", flush=True)
    print(f"         log: {log}", flush=True)
    if rtl:
        print(f"         RTL: {rtl}", flush=True)
    # The two failure signatures this workaround targets -- report them explicitly.
    for sig in ("CIN-124", "SCHD-30"):
        n = out.count(sig)
        if n:
            print(f"         !! {sig} x{n} (the degraded-port failure mode)", flush=True)
    sys.exit(0 if verdict == "OK" else 1)


if __name__ == "__main__":
    main()
