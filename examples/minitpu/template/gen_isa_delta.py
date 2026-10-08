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
``template/wb_calendar.Calendar`` (derived from the bound units' declared
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


RECORD = "sunwookim028/allo dev/records/minitpu/u4_track_b_2026-10-08.rst"
BASE = "v1"  # every delta here is relative to the top-level base, v1 (b3ba0a4d's timing; minitpu-comp 2026-10-08)

# The issue-edge offset of the MXU contract, MEASURED on b3ba0a4d (scripts/mxu_offset.py in RECORD):
# a vmatpush issued at row t streams its rows into mxu.input_push_i at t+1 .. t+4 (NUM_SUBLANES rows),
# the group is valid PUSH_TO_VALID (82) after the last push, at t+86; a vmatpop issued at p reaches
# mxu.output_pop_i at p+1 (the pop engine's register). result_latency.vmatpush is issue-to-issue (asm.py
# _M_RESULT_LATENCY: "cycles from a vmatpush to its first result"; isa_latency.json: "to the first
# vmatpop that finds its result waiting"), so it is PUSH_TO_VALID + ISSUE_TO_LAST_PUSH - POP_REGISTER.
ISSUE_TO_LAST_PUSH = 4
POP_REGISTER = 1
MXU_ISSUE_OFFSET = ISSUE_TO_LAST_PUSH - POP_REGISTER  # 3: 85 = 82 + 3 at b3ba0a4d

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


def calendar_deltas(base, cal, *, measured=None, allo_commit="?", pin="b3ba0a4d", backend="cycle-locked",
                    unresolved_keys=None):
    """``{"deltas": [...], "unresolved": [...]}`` for one calendar.

    ``measured``: an optional ``{key: value}`` of the same quantities measured on
    the composed instance (csim/RTL); a disagreement with the derived value
    goes to ``unresolved`` instead of the delta -- the seam's "pin to a
    measurement" rule -- so a derived number is never published against a
    measurement that refutes it. ``unresolved_keys``: ``{WB_W_*: why}`` for
    quantities whose bound latency has no scheduled manifest (D-10: not
    consumed); they go to ``unresolved`` with the value they would have."""
    vals = cal.rtl_params()
    src = (f"allo {allo_commit} ({RECORD}), relative to base {BASE}; template/calendar.Calendar '{cal.name}' ({backend}; "
           f"hop {cal.hop}, WB_STAGES {cal.WB_STAGES}, bound: "
           + ", ".join(f"{s.cls} {s.latency}" for s in cal.sources) + f"), MiniTPU pin {pin}")
    out, unresolved = [], []
    for key, v in vals.items():
        path = f"rtl_params.{key}"
        get(base, path)  # overrides only: refuses a path the base lacks
        if unresolved_keys and key in unresolved_keys:
            unresolved.append({"what": path, "would_be": v, "why": unresolved_keys[key]})
            continue
        if measured is not None and key in measured and measured[key] != v:
            unresolved.append({"what": path, "derived": v, "measured": measured[key],
                               "why": "the composed instance's measurement disagrees with the derived booking"})
            continue
        out.append({"what": WHAT[key], "set": {path: v}, "source": src})
    return {"describes": f"MiniTPU b3ba0a4d with the Allo calendar '{cal.name}' ({backend})",
            "deltas": out, "unresolved": unresolved,
            "note": "generated by examples/minitpu/template/gen_isa_delta.py; overrides only, relative to v1"}


def mxu_deltas(base, meas, *, allo_commit="?", pin="b3ba0a4d", origin=""):
    """The Allo-MXU version (D-21), on the same definition as the base: the
    MXU-port measurements of ``minitpu_rtl_mxu_2026-10-08`` (``measured_*.json``)
    moved to the issue edge by the offset measured here. ``issue_interval.vmatpop``
    is issue-to-issue too; the pop register is in both pops, so the port's
    pop -> next-valid interval is the issue interval unchanged."""
    lanes = base["profile"]["num_lanes"]
    src = (f"allo {allo_commit} ({RECORD}), relative to base {BASE}; MXU-port measurement {origin} "
           f"({meas.get('tb')} on {meas.get('rtl')}, DIM {meas.get('dim')}"
           + (f", NOT the profile's {lanes} lanes" if meas.get("dim") != lanes else "")
           + f"), moved to the issue edge by {MXU_ISSUE_OFFSET} = last push +{ISSUE_TO_LAST_PUSH} - pop register "
             f"{POP_REGISTER} (measured on b3ba0a4d); MiniTPU pin {pin}")
    out, unresolved = [], []

    def put(what, path, value):
        get(base, path)
        out.append({"what": what, "set": {path: value}, "source": src})

    if (meas.get("tokens_per_cycle") or 1.0) < 1.0:
        unresolved.append({"what": "every latency quantity", "tokens_per_cycle": meas["tokens_per_cycle"]})
    else:
        put("vmatpush result latency, issue to the first vmatpop that finds its result: MXU push->valid "
            f"{meas['push_to_valid']} + {MXU_ISSUE_OFFSET}", "matrix.result_latency.vmatpush",
            meas["push_to_valid"] + MXU_ISSUE_OFFSET)
        put("vmatpop issue interval, issue to issue (the port's pop -> next valid)", "matrix.issue_interval.vmatpop",
            meas["pop_interval"])
    first = get(base, "rtl_params.WB_W_MPOP_FIRST")
    put("vmatpop writeback beats: WB_W_MPOP_LAST = WB_W_MPOP_FIRST + beats - 1", "rtl_params.WB_W_MPOP_LAST",
        first + meas["pop_beats"] - 1)
    put("per-lane output FIFO depth in result rows", "resources.mxu_output_fifo.depth", meas["output_fifo_depth"])
    unresolved.append({"what": "matrix.weight_switch.span",
                       "why": "not measured (needs tb_matrix_weight_pipelining's scenario at the mxu boundary)"})
    return {"describes": f"MiniTPU b3ba0a4d with the Allo MXU at DIM {meas.get('dim')} (D-21 inverse angle)",
            "deltas": out, "unresolved": unresolved,
            "note": "generated by examples/minitpu/template/gen_isa_delta.py; overrides only, relative to v1"}


def check_base(base, cal, version):
    """Derived ``WB_W_*`` vs the pinned JSON resolved at ``version``."""
    res = resolve(base, version)
    rows = []
    for key, v in cal.rtl_params().items():
        rows.append((key, v, get(res, f"rtl_params.{key}")))
    return rows


def main(argv=None):
    from examples.minitpu.template.wb_calendar import Calendar

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
    ap.add_argument("--mxu-measured", help="minitpu_rtl_mxu measured_*.json: emit the Allo-MXU version instead")
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
    if a.mxu_measured:
        assert a.version == BASE, f"deltas are relative to {BASE}"
        name = a.name or "allo-mxu"
        mm = json.load(open(a.mxu_measured, encoding="utf-8"))
        block = {"versions": {"list": {name: mxu_deltas(base, mm, allo_commit=_allo_commit(),
                                                        origin=os.path.relpath(a.mxu_measured, ROOT))}}}
    else:
        block = {"versions": {"list": {name: calendar_deltas(base, cal, measured=meas, allo_commit=_allo_commit(),
                                                             backend=backend)}}}
    rc = 0
    if a.check_base and not a.mxu_measured:
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
