#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``results.json`` for every committed run, from the reports themselves.

Why this exists: a figure that is typed by hand can go stale silently, and moving
the typing from a docs page into a JSON file would only move the problem. So
nothing here is written by hand -- every field is parsed out of the reports the
synthesis run produced, and re-running this tool on unchanged reports must
produce byte-identical output.

What it reads, per ``reports/<variant>/``:

* ``area_summary.rpt``    -- the area split, and the cell counts DC prints there
* ``*.mapped.qor.rpt``    -- design WNS/TNS, violating paths, worst path slack
* ``synthesis-metrics.json`` -- status, wall seconds, start and finish
* ``settings.json``       -- if present, the run's settings snapshot, passed through

``--capture-settings <variant>=<build-dir>`` writes that snapshot for a run whose
build tree is still on disk: the DC and mflowgen parameters, the ``stdcells.db``
md5, and the clock port. The checksum and the port are there because two runs can
agree on every parameter and still resolve a different library, and because
Gemmini is constrained on ``clock`` where our designs use ``ap_clk`` -- a real
difference that reads as a discrepancy to anything comparing settings blindly.

Run: python asic_synthesis/tools/extract_results.py [--check]
``--check`` writes nothing and exits 1 if any results.json is missing or stale,
which is what CI should run.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASIC = os.path.dirname(HERE)
REPORTS = os.path.join(ASIC, "reports")

# "Combinational area:   499371.386185"
AREA_FIELDS = {
    "combinational": "Combinational area",
    "buf_inv": "Buf/Inv area",
    "noncombinational": "Noncombinational area",
    "macro_black_box": "Macro/Black Box area",
    "total_cell": "Total cell area",
}
COUNT_FIELDS = {
    "combinational_cells": "Number of combinational cells",
    "sequential_cells": "Number of sequential cells",
    "macros": "Number of macros/black boxes",
    "buf_inv_cells": "Number of buf/inv",
    "references": "Number of references",
    "ports": "Number of ports",
    "nets": "Number of nets",
    "cells": "Number of cells",
}
QOR_DESIGN = re.compile(
    r"Design\s+(\(Hold\)\s+)?WNS:\s*(-?[\d.]+)\s+TNS:\s*(-?[\d.]+)"
    r"\s+Number of Violating Paths:\s*(\d+)")
QOR_SLACK = re.compile(r"Critical Path Slack:\s*(-?[\d.]+)")
QOR_COUNT = re.compile(r"(Leaf Cell Count|Sequential Cell Count|"
                       r"Combinational Cell Count|Macro Count):\s*(\d+)")


def _number(line):
    tail = line.split(":", 1)[1].strip()
    try:
        return float(tail.split()[0])
    except (IndexError, ValueError):
        return None


def parse_area(path):
    out = {"area": {}, "counts": {}}
    with open(path, errors="replace") as fh:
        for line in fh:
            for key, label in AREA_FIELDS.items():
                if line.startswith(label):
                    value = _number(line)
                    if value is not None:
                        out["area"][key] = value
            for key, label in COUNT_FIELDS.items():
                if line.startswith(label):
                    value = _number(line)
                    if value is not None:
                        out["counts"][key] = int(value)
    return out


def parse_qor(path):
    """Design-level timing, plus the worst slack across path groups.

    DC prints one Critical Path Slack per path group and a single Design WNS
    line; the worst group slack is the honest headline, so take the minimum
    rather than whichever appears first.
    """
    out = {}
    slacks = []
    with open(path, errors="replace") as fh:
        for line in fh:
            hit = QOR_DESIGN.search(line)
            if hit:
                which = "hold" if hit.group(1) else "setup"
                out[which] = {"wns": float(hit.group(2)),
                              "tns": float(hit.group(3)),
                              "violating_paths": int(hit.group(4))}
            hit = QOR_SLACK.search(line)
            if hit:
                slacks.append(float(hit.group(1)))
            hit = QOR_COUNT.search(line)
            if hit:
                out.setdefault("counts", {})[
                    hit.group(1).lower().replace(" ", "_")] = int(hit.group(2))
    if slacks:
        out["worst_path_group_slack"] = min(slacks)
        out["path_group_slacks"] = slacks
    return out


def capture_settings(build_dir):
    """The run's settings snapshot, read out of a build tree that still exists."""
    step = build_dir
    if not os.path.isdir(os.path.join(step, "inputs")):
        matches = [d for d in sorted(os.listdir(build_dir))
                   if d.endswith("-synopsys-dc-synthesis")]
        if not matches:
            sys.exit(f"no synthesis step under {build_dir}")
        step = os.path.join(build_dir, matches[0])

    wanted = ("clock_period", "clock_port", "topographical", "flatten_effort",
              "design_name", "top_module", "sram_mode", "normalize_rtl",
              "nthreads", "gate_clock", "max_fanout", "high_effort_area_opt",
              "input_delay_fraction", "output_delay_fraction",
              "max_transition_fraction", "clock_uncertainty",
              "uniquify_with_design_name", "write_svsim_wrapper", "adk",
              "adk_view", "sv2v_defines", "manifest", "design_path")
    # The synthesis step's own configure.yml, then every other step's, because
    # the RTL path and manifest are set on the collector rather than on DC.
    params = {}
    others = sorted(glob.glob(os.path.join(os.path.dirname(step), "*",
                                           "configure.yml")))
    for cfg in [os.path.join(step, "configure.yml")] + others:
        if not os.path.isfile(cfg):
            continue
        with open(cfg, errors="replace") as fh:
            for line in fh:
                hit = re.match(r"\s*([a-z_0-9]+):\s*(\S.*?)\s*$", line)
                if hit and hit.group(1) in wanted:
                    params.setdefault(hit.group(1), hit.group(2))

    out = {"parameters": params}

    # The clock port as the SDC actually constrained it, not as a parameter
    # claimed it -- this is the field that distinguishes Gemmini's `clock` from
    # our `ap_clk` without anyone having to remember.
    sdc = os.path.join(step, "outputs", "design.sdc")
    if os.path.isfile(sdc):
        with open(sdc, errors="replace") as fh:
            for line in fh:
                hit = re.search(r"create_clock\s+\[get_ports\s+(\S+?)\]"
                                r".*?-period\s+([\d.]+)", line)
                if hit:
                    out["constrained_clock"] = {"port": hit.group(1),
                                                "period_ns": float(hit.group(2))}
                    break

    # The RTL this run consumed, as data rather than as a directory name. Every
    # mis-dated figure in this project has been prose -- a README asserting old
    # cycles, an area row placed beside cycles from other RTL -- so the export's
    # own parameters and measured cycles are recorded here, and a checker can
    # refuse a figure whose QD disagrees with the cycles beside it.
    design_path = params.get("design_path", "")
    manifest = params.get("manifest", "")
    rtl_dir = design_path if os.path.isdir(design_path) else os.path.dirname(manifest)
    if os.path.isdir(rtl_dir):
        rtl = {"dir": rtl_dir, "manifest": os.path.basename(manifest) or None}
        if os.path.isfile(manifest):
            with open(manifest, "rb") as fh:
                rtl["manifest_md5"] = hashlib.md5(fh.read()).hexdigest()
        mf = os.path.join(rtl_dir, "MANIFEST.json")
        if os.path.isfile(mf):
            with open(mf) as fh:
                m = json.load(fh)
            for key in ("params", "config", "cycles", "files", "memory_bits",
                        "stress_isa", "top"):
                if key in m:
                    rtl[key] = m[key]
            if "params" not in rtl and "config" not in rtl:
                rtl["params_source"] = "MANIFEST.json carries neither"
        else:
            rtl["params_source"] = "no MANIFEST.json in the RTL directory"
        readme = os.path.join(rtl_dir, "README.md")
        if os.path.isfile(readme):
            with open(readme, errors="replace") as fh:
                hit = re.search(r"Emitted from allo commit `([0-9a-f]{7,40})`",
                                fh.read())
            if hit:
                rtl["allo_commit"] = hit.group(1)
        out["rtl"] = rtl

    db = os.path.join(step, "inputs", "adk", "stdcells.db")
    if os.path.isfile(db):
        digest = hashlib.md5()
        with open(db, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        out["stdcells_db_md5"] = digest.hexdigest()

    out["build_dir"] = os.path.abspath(build_dir)
    return out


def results_for(variant, problems):
    """Everything known about one run, from its committed reports alone.

    Anything missing or unparseable is recorded in ``problems`` and aborts the
    run rather than being skipped: a results.json that quietly omits a variant
    is an instrument failing open, which is the one failure mode nobody notices.
    """
    d = os.path.join(REPORTS, variant)
    area = os.path.join(d, "area_summary.rpt")
    if not os.path.isfile(area):
        problems.append(f"{variant}: no area_summary.rpt")
        return None
    out = {"variant": variant}
    out.update(parse_area(area))
    if "total_cell" not in out["area"]:
        problems.append(f"{variant}: area_summary.rpt has no total cell area")
    for required in ("combinational", "noncombinational"):
        if required not in out["area"]:
            problems.append(f"{variant}: area_summary.rpt has no {required} area")

    qor = [f for f in sorted(os.listdir(d)) if f.endswith(".mapped.qor.rpt")]
    if not qor:
        problems.append(f"{variant}: no *.mapped.qor.rpt, so no timing")
    else:
        out["timing"] = parse_qor(os.path.join(d, qor[0]))
        if "worst_path_group_slack" not in out["timing"]:
            problems.append(f"{variant}: {qor[0]} has no Critical Path Slack")
        if "setup" not in out["timing"]:
            problems.append(f"{variant}: {qor[0]} has no Design WNS line")

    metrics = os.path.join(d, "synthesis-metrics.json")
    if os.path.isfile(metrics):
        with open(metrics) as fh:
            m = json.load(fh)
        out["run"] = {k: m[k] for k in
                      ("status", "returncode", "wall_seconds",
                       "started_at", "finished_at") if k in m}

    settings = os.path.join(d, "settings.json")
    if os.path.isfile(settings):
        with open(settings) as fh:
            out["settings"] = json.load(fh)

    return out


def dump(obj):
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 if any results.json is missing "
                         "or does not match its reports")
    ap.add_argument("--capture-settings", metavar="VARIANT=BUILD_DIR",
                    action="append", default=[],
                    help="write reports/VARIANT/settings.json from a build tree")
    args = ap.parse_args()

    for spec in args.capture_settings:
        if "=" not in spec:
            sys.exit(f"--capture-settings wants VARIANT=BUILD_DIR, got {spec}")
        variant, build = spec.split("=", 1)
        target = os.path.join(REPORTS, variant)
        if not os.path.isdir(target):
            sys.exit(f"no committed reports for {variant}")
        path = os.path.join(target, "settings.json")
        with open(path, "w") as fh:
            fh.write(dump(capture_settings(build)))
        print(f"wrote {os.path.relpath(path, ASIC)}")

    variants = [v for v in sorted(os.listdir(REPORTS))
                if os.path.isdir(os.path.join(REPORTS, v))]
    if not variants:
        sys.exit(f"no run directories under {REPORTS}")

    problems = []
    parsed = {v: results_for(v, problems) for v in variants}
    if problems:
        print("REPORTS INCOMPLETE -- nothing written:")
        for line in problems:
            print(f"  {line}")
        print("\nFix the report, or remove the directory if the run was abandoned."
              "\nA partial results.json would hide the gap rather than show it.")
        return 1

    index, stale = {}, []
    for variant in variants:
        res = parsed[variant]
        text = dump(res)
        path = os.path.join(REPORTS, variant, "results.json")
        if args.check:
            current = open(path).read() if os.path.isfile(path) else None
            if current != text:
                stale.append(variant)
        else:
            with open(path, "w") as fh:
                fh.write(text)
        rtl = res.get("settings", {}).get("rtl", {})
        design = dict(rtl.get("config") or {})
        design.update(rtl.get("params") or {})
        # Exports have used both TPU_T and T; normalise so the index is uniform.
        # An export predating the QD parameter simply has no QD here, which is
        # what dates it -- the absence is the signal, so it is not filled in.
        picked = {}
        for key in ("T", "MAXDIM", "QD"):
            for name in (key, f"TPU_{key}"):
                if name in design:
                    picked[key] = design[name]
                    break
        index[variant] = {
            # The design parameters travel with the area, so a figure can never
            # be quoted beside cycles from a different configuration unnoticed.
            "design_params": picked or None,
            "allo_commit": rtl.get("allo_commit"),
            "manifest": rtl.get("manifest"),
            "total_cell_area": res["area"].get("total_cell"),
            "noncombinational": res["area"].get("noncombinational"),
            "worst_path_group_slack":
                res.get("timing", {}).get("worst_path_group_slack"),
            "violating_paths":
                res.get("timing", {}).get("setup", {}).get("violating_paths"),
            "wall_seconds": res.get("run", {}).get("wall_seconds"),
        }

    index_text = dump({"runs": index})
    index_path = os.path.join(ASIC, "results.json")
    if args.check:
        current = open(index_path).read() if os.path.isfile(index_path) else None
        if current != index_text:
            stale.append("<index>")
        if stale:
            print("STALE OR MISSING results.json: " + ", ".join(stale))
            print("Run extract_results.py (no --check) and commit the result.")
            return 1
        print(f"results.json OK for {len(index)} runs")
        return 0

    with open(index_path, "w") as fh:
        fh.write(index_text)
    print(f"wrote results.json for {len(index)} runs, and the index")
    for variant, row in sorted(index.items()):
        area = row["total_cell_area"]
        print(f"  {variant:<34} {area:>14,.2f}" if area else f"  {variant:<34}"
              "  (no area)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
