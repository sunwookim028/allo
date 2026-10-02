# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Characterize one MiniTPU RTL unit against its numpy references.

    python -m examples.minitpu.harness.characterize bf16_mul [acc24_add_pipe ...]

For each unit, on the unit's own stimulus:

* latency: measured from ``valid_o`` (``valid``), by a step probe
  (``bare``, and ``valid`` as a second measurement), against the declared;
* ``REF-MATCH k/n``: the RTL-semantics reference (``harness/ref.py``)
  against the RTL, bit for bit -- it must be n/n;
* the RTL's deviations from IEEE, by cause: every vector where the RTL and
  the unit's ``IEEE`` reference differ, grouped by the unit's
  ``DEVIATIONS`` rules. ``unexplained`` must be empty.

Needs Verilator, not Allo.
"""

import argparse
import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from examples.minitpu.harness import rtl  # noqa: E402


def _classify(stim, ieee, got, rules):
    groups = {}
    for i in np.flatnonzero(ieee != got):
        key = "unexplained"
        for name, rule in rules:
            if rule(stim[i], ieee[i], got[i]):
                key = name
                break
        groups.setdefault(key, []).append(i)
    return groups


def _fmt(row, w):
    return "(" + ",".join(f"{int(x):0{w}x}" for x in row) + ")"


def characterize(name):
    u = importlib.import_module(f"examples.minitpu.units.{name}")
    stim = u.stimulus()
    n = len(stim)
    w = (u.RTL.outputs[0][1] + 3) // 4
    t = time.time()
    out, cyc = rtl.run(u.RTL, stim.astype(np.uint64))
    got = out[:, 0].astype(np.int64)
    lat = sorted(set(cyc.tolist()))
    probe = None
    if u.RTL.shape != "comb":
        probe = rtl.probe_latency(u.RTL, *u.PROBE)
    trun = time.time() - t
    ok_lat = (u.RTL.shape == "bare" or lat == [u.RTL.latency]) and probe in (None, u.RTL.latency)
    how = {"comb": "combinational", "valid": f"valid_o {lat}", "bare": "sampled at declared"}[u.RTL.shape]
    print(f"== {name}: {u.RTL.top} ({u.RTL.shape}), {n} vectors, rtl {trun:.1f}s")
    print(f"   {'LATENCY-OK  ' if ok_lat else 'LATENCY-DIFF'} declared {u.RTL.latency} "
          f"[{u.LATENCY_SOURCE}]; measured: {how}" + (f", step probe {probe}" if probe is not None else ""))

    want = u.REF(*stim.T).astype(np.int64)
    k = int((want != got).sum())
    print(f"   {'REF-MATCH' if k == 0 else 'REF-DIFF '} {n - k}/{n}")
    for i in np.flatnonzero(want != got)[:5]:
        print(f"      {_fmt(stim[i], 4)}: ref {want[i]:0{w}x} rtl {got[i]:0{w}x}")

    ieee = u.IEEE(*stim.T).astype(np.int64)
    groups = _classify(stim, ieee, got, u.DEVIATIONS)
    print(f"   IEEE-DIFF {sum(map(len, groups.values()))}/{n}, by cause:")
    for cause, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ex = "; ".join(f"{_fmt(stim[i], 4)} ieee {ieee[i]:0{w}x} rtl {got[i]:0{w}x}" for i in idx[:2])
        print(f"      {len(idx):8d}  {cause}  e.g. {ex}")
    return k == 0 and ok_lat and "unexplained" not in groups


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("units", nargs="+")
    args = ap.parse_args(argv)
    ok = [characterize(name) for name in args.units]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
