# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit one U1 unit variant for Vitis HLS csynth (Allo's vitis_hls target).

    $ALLO_PYTHON build_vitis.py <unit> <variant> <prj> <freq MHz> [--unroll L] [--pipeline L] [--n N]

Writes the project only (run ``vitis_hls -f run.tcl`` in it; ``run_vitis.sh``).
"""
import argparse, importlib, os, shutil, sys

sys.path.insert(0, os.getcwd())
import allo.dataflow as df  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("variant"); ap.add_argument("prj"); ap.add_argument("freq", type=float)
ap.add_argument("--unroll", action="append", default=[])
ap.add_argument("--pipeline", action="append", default=[])
ap.add_argument("--n", type=int, default=1024)
a = ap.parse_args()
u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
make, _ = u.VARIANTS[a.variant]
s = df.customize(make(a.n))
for lp in a.unroll:
    s.unroll(lp)
for lp in a.pipeline:
    s.pipeline(lp)
if os.path.isdir(a.prj):
    shutil.rmtree(a.prj)
s.build(target="vitis_hls", mode="csyn", project=a.prj, configs={"frequency": a.freq})
print("EMITTED", a.prj)
