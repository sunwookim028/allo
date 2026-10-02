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
"""
import json
import os


def load(project):
    with open(os.path.join(project, "latency.json"), encoding="utf-8") as f:
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
