# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Characterize one MiniTPU RTL unit against its numpy references.

    python -m examples.minitpu.harness.characterize bf16_mul [acc24_add_pipe ...]
    python -m examples.minitpu.harness.characterize vpu_fifo[:output] ...

For each unit, on the unit's own stimulus:

* latency: measured from ``valid_o`` (``valid``), by a step probe
  (``bare``, and ``valid`` as a second measurement), against the declared;
* ``REF-MATCH k/n``: the RTL-semantics reference (``harness/ref.py``)
  against the RTL, bit for bit -- it must be n/n;
* the RTL's deviations from IEEE, by cause: every vector where the RTL and
  the unit's ``IEEE`` reference differ, grouped by the unit's
  ``DEVIATIONS`` rules. ``unexplained`` must be empty.

A storage unit (``trace`` shape, U2) is characterized per instance
(``unit:instance``; all instances by default), on its directed and random
command traces and on the traces seeded from MiniTPU's tbs:

* ``SEED-STABLE``: every slot the reference calls defined is equal between
  two runs with different random initial state (``--x-initial unique``) --
  else the reference calls something defined that the RTL leaves open;
* ``REF-MATCH k/k defined``: the reference against the RTL on the defined
  slots, and an undefined-slot census by reason, each with how many of its
  slots the reference's guess still matched and how many moved with the seed;
* events (illegal or quirky commands, counted by the reference) and the RTL
  assertions that fired; a trace tagged legal must fire none;
* ``TB-REPLAY``: a seeded trace's RTL replay equals what the tb's own DUT
  showed, on the defined slots;
* latency: each port's step probe against the declared value.

Needs Verilator; a unit file with Allo variants also imports Allo.
"""

import argparse
import importlib
import os
import re
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


def _defined_diff(a, b, defined):
    """Rows of defined slots where two ``uint64[n, nw]`` columns differ."""
    return np.flatnonzero(defined & (a != b).any(axis=1))


def _trace_one(u, inst, label, cmd, legal, totals, seen=None):
    """One trace through the RTL (two seeds) and the reference; prints a line."""
    unit = u.INSTANCES[inst]
    cmd = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}  # uint64[n, nwords] per port
    r1 = rtl.run_trace(unit, cmd, seed=1)
    fired = list(rtl.last_asserts)
    r2 = rtl.run_trace(unit, cmd, seed=7)
    want, reason, events = u.REF(inst, cmd)
    n = len(next(iter(cmd.values())))
    bad_ref, bad_seed, ndef, nmask = 0, 0, 0, 0
    for p, col in r1.items():
        defined = reason[p] == ""
        ndef += int(defined.sum())
        nmask += int((~defined).sum())
        d = _defined_diff(col, want[p], defined)
        bad_ref += len(d)
        for i in d[:3]:
            print(f"      REF-DIFF {label} {p} cycle {i}: ref {rtl.unpack(want[p][i:i+1])[0]:x} "
                  f"rtl {rtl.unpack(col[i:i+1])[0]:x}")
        bad_seed += len(_defined_diff(col, r2[p], defined))
        for why in set(reason[p][~defined]):
            m = reason[p] == why
            c = totals["census"].setdefault((p, why), [0, 0, 0])
            c[0] += int(m.sum())
            c[1] += int((m & (col == want[p]).all(axis=1)).sum())
            c[2] += int((m & (col != r2[p]).any(axis=1)).sum())
    for k, v in events.items():
        totals["events"][k] = totals["events"].get(k, 0) + v
    msgs = sorted({re.sub(r"^\[\d+\] ", "", m) for _, m in fired})
    for m in msgs:
        totals["asserts"][m] = totals["asserts"].get(m, 0) + sum(1 for _, x in fired if x.endswith(m))
    replay = ""
    if seen is not None:
        k = tot = 0
        for p, col in r1.items():
            vals = rtl.unpack(col)
            for i, v in enumerate(seen[p]):
                if v is None or reason[p][i] != "":
                    continue
                tot += 1
                k += int(v == vals[i])
        replay = f" TB-REPLAY {k}/{tot}"
        totals["ok"] &= k == tot
    ok = bad_ref == 0 and bad_seed == 0 and (not legal or not fired)
    totals["ok"] &= ok
    print(f"   {'ok ' if ok else 'BAD'} {label:28s} {'legal  ' if legal else 'illegal'} {n:6d} cyc  "
          f"defined {ndef - bad_ref}/{ndef} masked {nmask}  seed-diff {bad_seed}  "
          f"asserts {len(fired)}{replay}"
          + ("  " + ", ".join(f"{k}={v}" for k, v in sorted(events.items())) if events else ""))
    return ok


def characterize_trace(u, name, insts):
    ok = True
    for inst in insts:
        unit = u.INSTANCES[inst]
        t = time.time()
        totals = {"census": {}, "events": {}, "asserts": {}, "ok": True}
        print(f"== {name}:{inst}: {unit.top} (trace) params={unit.params} defines={unit.defines}")
        for label, cmd, legal in u.traces(inst):
            _trace_one(u, inst, label, cmd, legal, totals)
        for label, sinst, cmd, seen, legal in (u.seeds() if hasattr(u, "seeds") else []):
            if sinst == inst:
                _trace_one(u, inst, label, cmd, legal, totals, seen)
        print("   undefined-slot census (slots, guess matched, moved with seed):")
        for (p, why), (k, g, s_) in sorted(totals["census"].items()):
            print(f"      {p:18s} {why:16s} {k:8d}  guess {g:8d}  seed-moved {s_:8d}")
        if totals["events"]:
            print("   events: " + ", ".join(f"{k}={v}" for k, v in sorted(totals["events"].items())))
        for m, c in sorted(totals["asserts"].items()):
            print(f"   assertion x{c}: {m}")
        lat_ok = True
        for label, declared, measured in u.probes(inst):
            good = declared == measured
            lat_ok &= good
            print(f"   {'LATENCY-OK  ' if good else 'LATENCY-DIFF'} {label}: declared {declared}, "
                  f"step probe {measured}")
        verdict = totals["ok"] and lat_ok
        print(f"   {'REF-MATCH' if totals['ok'] else 'REF-DIFF'} {name}:{inst} "
              f"({time.time() - t:.1f}s)")
        ok &= verdict
    return ok


def characterize(name):
    name, _, only = name.partition(":")
    u = importlib.import_module(f"examples.minitpu.units.{name}")
    if u.RTL.shape == "trace":
        return characterize_trace(u, name, [only] if only else list(u.INSTANCES))
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
