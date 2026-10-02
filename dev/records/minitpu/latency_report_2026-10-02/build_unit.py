# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build one U1 unit variant through Allo's SystemC -> Catapult flow with the
latency-report prototype (no hand-patch: ``latency=`` is a build config).

    $ALLO_PYTHON build_unit.py <unit> <variant> <prj> [--clock NS]
        [--unroll LOOP ...] [--pipeline LOOP ...] [--latency KERNEL=L ...]
        [--synth-top KERNEL]

``<variant>`` is a key of the unit's ``VARIANTS`` or ``wire:<variant>`` (the
Wire-port forms committed in ``../u1_catapult_units_2026-10-02/scripts/wire``).
Prints Allo's ``[latency]`` lines; ``<prj>/latency.json`` is the manifest.
"""
import argparse, importlib, os, shutil, sys, time

sys.path.insert(0, os.getcwd())
import allo.dataflow as df  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("variant"); ap.add_argument("prj")
ap.add_argument("--clock", type=float, default=None)
ap.add_argument("--unroll", action="append", default=[])
ap.add_argument("--pipeline", action="append", default=[])
ap.add_argument("--latency", action="append", default=[])
ap.add_argument("--synth-top", default=None)
ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
n = a.n or len(u.stimulus())
if a.variant.startswith("wire:"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                    "u1_catapult_units_2026-10-02", "scripts", "wire"))
    make = importlib.import_module(f"wire_{a.unit}_{a.variant[5:]}").wire
else:
    make, _ = u.VARIANTS[a.variant]
s = df.customize(make(n))
for lp in a.unroll:
    s.unroll(lp)
for lp in a.pipeline:
    s.pipeline(lp)
cfg = {}
if a.clock:
    cfg["clock_period"] = a.clock
if a.synth_top:
    cfg["synth_top"] = a.synth_top
if a.latency:
    cfg["latency"] = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.latency}
if os.path.isdir(a.prj):
    shutil.rmtree(a.prj)
t0 = time.time()
try:
    mod = s.build(target="systemc", mode="csyn", project=a.prj, configs=cfg)
    mod()
    print(f"BUILD {a.unit} {a.variant} {cfg}: ok {time.time() - t0:.0f}s", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"BUILD {a.unit} {a.variant} {cfg}: {type(e).__name__}: {e} ({time.time() - t0:.0f}s)", flush=True)
