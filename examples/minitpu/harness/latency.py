# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a backend's latency manifest against the RTL's measured latency.

A backend that produces RTL writes ``<project>/latency.json`` (for Catapult:
``allo/backend/catapult.py:write_latency_manifest``): per kernel, the latency
and II its scheduler chose. The manifest is a *report*; the RTL is the
evidence. :func:`verdict` holds one to the other, so a latency table built from
manifests (an assembler's, ACT's) is only as good as a check that has passed.

Latency is ``rtl.py``'s count: edges from the input-accepting edge to the edge
after which the output is visible, with every port ready.

Bookings (README D-20): a geometry record's derived latency (``MxuGeometry.
push_to_valid``, ``TreeGeometry.latency``, ...) is a *booking* (D-10), the
number an assembler schedules against. :func:`check_booking` holds a
``{unit: booked}`` dict to the manifest's ``latency`` per unit, at one clock,
and prints ``BOOKING-MATCH`` / ``BOOKING-MISMATCH`` / ``BOOKING-UNCHECKED``.
Only a ``scheduled`` entry at the booking's clock is evidence; an unreliable
or absent entry, or another clock, is UNCHECKED, never a pass. The manifest's
backend (``tool``) and clock are printed on each line, so the verdict is per
(unit, backend, clock). From the shell::

    $ALLO_PYTHON -m examples.minitpu.harness.latency \\
        --bookings examples.minitpu.template.legality:BOOKINGS [--clock 3.33] <manifest>

``--bookings`` is a JSON object, a ``.json`` file, or ``module:attr`` (a dict
or a zero-argument callable); the exit status is 1 on any MISMATCH.
"""
import argparse
import importlib
import json
import os
import sys


def load(project):
    if os.path.isfile(project):
        project, name = os.path.split(project)
    else:
        name = "latency.json"
    with open(os.path.join(project, name), encoding="utf-8") as f:
        return json.load(f)


def verdict(manifest, kernel, measured, cycles_per_vector=None, declared=None):
    """``LATENCY-MATCH`` / ``LATENCY-MISMATCH`` / ``LATENCY-UNRELIABLE`` line.

    ``measured``: the latency histogram ``{L: count}`` measured on the RTL (or
    one int). ``cycles_per_vector``: measured rate, checked against the II.
    ``declared``: the unit's own contract (MiniTPU's), reported beside.
    """
    u = manifest["units"][kernel]
    if isinstance(measured, int):
        measured = {measured: 1}
    ms = sorted(measured)
    rep, ii, st = u.get("latency"), u.get("ii"), u.get("status")
    tail = f"reported {rep} ii={ii} ({st}), measured {ms if len(ms) > 1 else ms[0]}"
    if cycles_per_vector is not None:
        tail += f" at {cycles_per_vector:g} cyc/vector"
    if declared is not None:
        tail += f"; unit declares {declared}"
    if u.get("declared") is not None:
        tail += f"; pinned {u['declared']}"
    if st != "scheduled":
        # An unreliable report must not be consumed; it passes only as a flag.
        return f"LATENCY-UNRELIABLE {kernel}: {tail} -- {u.get('reason', '')}"
    ok = ms == [rep]
    if cycles_per_vector is not None and ii is not None:
        # measured cycles/vector includes one fill over the whole stimulus
        ok = ok and abs(cycles_per_vector - ii) < 0.01
    return f"LATENCY-{'MATCH' if ok else 'MISMATCH'} {kernel}: {tail}"


def composite(manifest, kernels, measured, cycles_per_vector=None):
    """Several kernels in series (one region): the rate is checked against the
    slowest kernel's II; the latency is *not* a manifest number. Each kernel's
    is, but the region adds one cycle per Connections FIFO hop and, when the
    stages' IIs differ, a backlog that grows with the stimulus."""
    us = [manifest["units"][k] for k in kernels]
    if isinstance(measured, int):
        measured = {measured: 1}
    ms = sorted(measured)
    lat = [u.get("latency") for u in us]
    iis = [u.get("ii") for u in us]
    bad = [k for k, u in zip(kernels, us) if u.get("status") != "scheduled"]
    head = "LATENCY-COMPOSITE"
    if bad:
        head = "LATENCY-UNRELIABLE"
    elif cycles_per_vector is not None and None not in iis:
        if abs(cycles_per_vector - max(iis)) >= 0.01:
            head = "LATENCY-MISMATCH"
    return f"{head} {'+'.join(kernels)}: kernel latencies {lat} (sum {sum(x or 0 for x in lat)}), " f"ii {iis} (rate max {max(x or 0 for x in iis)}), measured {ms if len(ms) > 1 else ms[0]}" + (
        f" at {cycles_per_vector:g} cyc/vector" if cycles_per_vector is not None else ""
    ) + (
        f"; unreliable: {bad}" if bad else ""
    )


def check_booking(manifest_path, bookings, clock=None):
    """Compare booked latencies with a manifest's measured ones, per unit.

    ``manifest_path``: a ``latency.json`` or the project directory holding it.
    ``bookings``: ``{unit: booked latency}``, the unit named as in the manifest
    (Catapult: ``<name>_0``). ``clock``: the clock period (ns) the booking is
    for; a manifest at another period proves nothing about it (Catapult picks
    the latency from the clock) and its units are UNCHECKED. ``None`` accepts
    the manifest's own clock.

    Prints one line per unit and returns ``{unit: (verdict, booked, manifest)}``
    with verdict ``"MATCH"``, ``"MISMATCH"`` or ``"UNCHECKED"`` (manifest value
    ``None`` when there is none).
    """
    m = load(manifest_path)
    tool, mclk = m.get("tool", "?"), m.get("clock_period_ns")
    where = f"[{tool} @ {mclk} ns]"
    out = {}
    for unit, booked in bookings.items():
        u = m.get("units", {}).get(unit)
        if u is None:
            why, rep = "status=absent", None
        elif clock is not None and (mclk is None or abs(mclk - clock) >= 1e-9):
            why, rep = f"clock={mclk} != booked {clock}", u.get("latency")
        elif u.get("status") != "scheduled":
            why, rep = f"status={u.get('status')}", u.get("latency")
        else:
            why, rep = None, u.get("latency")
        if why is not None:
            out[unit] = ("UNCHECKED", booked, rep)
            print(f"BOOKING-UNCHECKED {unit} ({why}) booked={booked} {where}")
            continue
        v = "MATCH" if rep == booked else "MISMATCH"
        out[unit] = (v, booked, rep)
        print(f"BOOKING-{v} {unit} booked={booked} manifest={rep} {where}")
    return out


def _bookings_arg(arg):
    if os.path.isfile(arg):
        with open(arg, encoding="utf-8") as f:
            return json.load(f)
    if arg.lstrip().startswith("{"):
        return json.loads(arg)
    mod, _, attr = arg.partition(":")
    obj = getattr(importlib.import_module(mod), attr)
    return obj() if callable(obj) else obj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest", help="latency.json, or the project directory")
    ap.add_argument("--bookings", required=True, help="JSON, a .json file or module:attr")
    ap.add_argument("--clock", type=float, default=None, help="booked clock period, ns")
    a = ap.parse_args(argv)
    res = check_booking(a.manifest, _bookings_arg(a.bookings), a.clock)
    return 1 if any(v[0] == "MISMATCH" for v in res.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
