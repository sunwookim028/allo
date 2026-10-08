# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agreed seam, calendar side: a composed instance's write-port calendar
as ``versions.list.<name>.deltas`` into MiniTPU's ``isa/latency.json``.

    $ALLO_PYTHON -m examples.minitpu.template.gen_isa_delta --base <isa/latency.json> \\
        [--version v1] [--calendar locked | --calendar selftimed --hop H] --name allo-locked \\
        [--write out.json] [--check out.json] [--check-base]

The seam (``u1_matrix.rst``, "Agreed seam with minitpu-comp, 2026-10-08"):
a list of ``{"what", "set": {dotted path: value}, "source"}``, one entry per
quantity, provenance in ``source``; **overrides only** (a path the base lacks
is refused); ``rtl_params.WB_W_*`` are ``W = unit latency + VPU_WB_STAGES``,
the claim cycle, never the unit latency. Values come from
``template/calendar.Calendar`` (derived from the bound units' declared
latencies, D-20); nothing here is a constant.

This is the calendar half of the generator whose MXU half is the prototype in
``dev/records/minitpu/minitpu_rtl_mxu_2026-10-08/scripts/gen_isa_delta.py``;
that file's ``get`` (the dotted-path reader that enforces overrides-only) and
its ``--check`` comparison are imported, not copied.

``--check-base`` holds the derived values to the base JSON resolved at
``--version`` (the top level is ``v1``; another version applies its deltas
first): ``DELTA-MATCH`` when every ``WB_W_*`` the calendar implies equals the
pinned value -- i.e. the instance *is* that version on the write port --
else ``DELTA-DIFF`` naming each quantity.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_PROTO = os.path.join(ROOT, "dev", "records", "minitpu", "minitpu_rtl_mxu_2026-10-08", "scripts",
                      "gen_isa_delta.py")


def _proto():
    spec = importlib.util.spec_from_file_location("gen_isa_delta_mxu", _PROTO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


get = _proto().get  # overrides-only dotted-path reader (raises KeyError on a path the base lacks)

WHAT = {
    "WB_W_VLD": "vld claims the VREG write port: VMEM compute read + request register + WB stages",
    "WB_W_ALU": "ALU result claims the VREG write port: vpu_alu latency + WB stages",
    "WB_W_SFU": "SFU result claims the VREG write port: sfu latency + WB stages",
    "WB_W_REDUCE": "vredsum/vredmax claim the write port: tree root latency + WB stages",
    "WB_W_LANE_REDUCE": "vlanered claims the write port: tree tap latency + WB stages",
    "WB_W_TXOUT": "vtxout claims the write port: transpose read register + WB stages",
    "WB_W_MPOP_FIRST": "vmatpop (result waiting) claims the write port: pop engine register + WB stages",
    "WB_W_MPOP_LAST": "vmatpop's last beat: WB_W_MPOP_FIRST + MXU_POP_BEATS - 1",
}


def resolve(base, version):
    """The base JSON with ``version``'s deltas applied (``v1`` is the top level)."""
    out = copy.deepcopy(base)
    if "versions" not in base:  # a pre-versions JSON (b3ba0a4d's docs/isa_latency.json) is its own base
        assert version in (None, "v1"), f"{version}: this JSON has no versions section"
        return out
    if version in (None, base["versions"].get("default"), "v1"):
        return out
    entry = base["versions"]["list"][version]
    for d in entry.get("deltas") or []:
        for path, value in d["set"].items():
            keys = path.split(".")
            cur = out
            for k in keys[:-1]:
                cur = cur[k]
            assert keys[-1] in cur, f"{version}: delta path {path} not in the base"
            cur[keys[-1]] = value
    return out


def _allo_commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short=8", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def calendar_deltas(base, cal, *, measured=None, allo_commit="?", pin="b3ba0a4d", backend="cycle-locked"):
    """``{"deltas": [...], "unresolved": [...]}`` for one calendar.

    ``measured``: an optional ``{key: value}`` of the same quantities measured on
    the composed instance (csim/RTL); a disagreement with the derived value
    goes to ``unresolved`` instead of the delta -- the seam's "pin to a
    measurement" rule -- so a derived number is never published against a
    measurement that refutes it."""
    vals = cal.rtl_params()
    src = (f"allo {allo_commit}, template/calendar.Calendar '{cal.name}' ({backend}; "
           f"hop {cal.hop}, WB_STAGES {cal.WB_STAGES}, bound: "
           + ", ".join(f"{s.cls} {s.latency}" for s in cal.sources) + f"), MiniTPU pin {pin}")
    out, unresolved = [], []
    for key, v in vals.items():
        path = f"rtl_params.{key}"
        get(base, path)  # overrides only: refuses a path the base lacks
        if measured is not None and key in measured and measured[key] != v:
            unresolved.append({"what": path, "derived": v, "measured": measured[key],
                               "why": "the composed instance's measurement disagrees with the derived booking"})
            continue
        out.append({"what": WHAT[key], "set": {path: v}, "source": src})
    return {"deltas": out, "unresolved": unresolved,
            "note": "generated by examples/minitpu/template/gen_isa_delta.py; overrides only"}


def check_base(base, cal, version):
    """Derived ``WB_W_*`` vs the pinned JSON resolved at ``version``."""
    res = resolve(base, version)
    rows = []
    for key, v in cal.rtl_params().items():
        rows.append((key, v, get(res, f"rtl_params.{key}")))
    return rows


def main(argv=None):
    from examples.minitpu.template.calendar import Calendar

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="MiniTPU isa/latency.json (master >= f1e978e)")
    ap.add_argument("--version", default="v1", help="the base version the delta is relative to")
    ap.add_argument("--calendar", choices=("locked", "selftimed"), default="locked")
    ap.add_argument("--hop", type=int, default=0, help="selftimed: measured extra issue->mux cycles per source")
    ap.add_argument("--measured", help="JSON {WB_W_*: value} measured on the composed instance")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--write")
    ap.add_argument("--check")
    ap.add_argument("--check-base", action="store_true")
    a = ap.parse_args(argv)
    with open(a.base, encoding="utf-8") as f:
        base = json.load(f)
    cal = Calendar()
    if a.calendar == "selftimed":
        cal = cal.selftimed(a.hop)
    name = a.name or f"allo-{cal.name}"
    meas = json.load(open(a.measured, encoding="utf-8")) if a.measured else None
    backend = a.backend or ("cycle-locked (I1/W1), simulator + SystemC csim" if cal.hop == 0
                            else f"self-timed, {a.hop} cycles per Stream hop")
    block = {"versions": {"list": {name: calendar_deltas(base, cal, measured=meas, allo_commit=_allo_commit(),
                                                         backend=backend)}}}
    rc = 0
    if a.check_base:
        bad = [(k, d, p) for k, d, p in check_base(base, cal, a.version) if d != p]
        n = len(cal.rtl_params())
        if bad:
            rc = 1
            print(f"DELTA-DIFF {name} vs {a.version}: " + ", ".join(f"{k} derived {d} pinned {p}" for k, d, p in bad)
                  + f" ({n - len(bad)}/{n} equal)")
        else:
            print(f"DELTA-MATCH {name} vs {a.version}: {n}/{n} rtl_params.WB_W_* equal to the pinned JSON")
    if a.check:
        proto = _proto()  # noqa: F841 -- same comparison as the MXU prototype's --check
        have = json.load(open(a.check, encoding="utf-8"))
        want = block["versions"]["list"][name]["deltas"]
        got = get(have, f"versions.list.{name}.deltas")
        strip = lambda ds: [{"what": d["what"], "set": d["set"]} for d in ds]  # noqa: E731
        if strip(want) != strip(got):
            print("DELTA-MISMATCH", json.dumps(strip(want)), "vs", json.dumps(strip(got)))
            rc = 1
        else:
            print(f"DELTA-MATCH versions.list.{name}: {len(got)} entries re-derived equal")
    if not a.check and not a.check_base or a.write:
        text = json.dumps(block, indent=1)
        if a.write:
            with open(a.write, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        else:
            print(text)
    return rc


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    sys.exit(main())
