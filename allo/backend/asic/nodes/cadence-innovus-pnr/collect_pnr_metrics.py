#!/usr/bin/env python3
"""Collect stable runtime, physical, verification, and timing metrics."""

import argparse
import json
import re
from pathlib import Path


def last_number(pattern, text):
    matches = re.findall(pattern, text, re.I | re.M)
    return float(matches[-1]) if matches else None


def timing_summary(path):
    if not path.is_file():
        return {"wns_ns": None, "tns_ns": None, "violating_paths": None}
    text = path.read_text(errors="replace")
    wns = last_number(r"WNS\s*\(ns\):\|\s*([-+0-9.eE]+)", text)
    tns = last_number(r"TNS\s*\(ns\):\|\s*([-+0-9.eE]+)", text)
    paths = last_number(r"Violating Paths:\|\s*(\d+)", text)
    return {"wns_ns": wns, "tns_ns": tns,
            "violating_paths": int(paths) if paths is not None else None}


def collect(root, wall_seconds, stop_after_step="none", drc_check_policy="error",
            hold_target_slack=0.005):
    reports = root / "reports"
    log = (root / "logs/run.log").read_text(errors="replace")
    stages = {}
    stage_path = reports / "pnr-stage-times.rpt"
    if stage_path.is_file():
        for line in stage_path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                stages[fields[0]] = float(fields[1])
    preplace = (reports / "preplace.summary").read_text(errors="replace")
    summary_name = "signoff.summary" if stop_after_step == "none" else f"{stop_after_step}.summary"
    final = (reports / summary_name).read_text(errors="replace")
    drc_path = reports / "innovus-drc.rpt"
    drc = drc_path.read_text(errors="replace") if drc_path.is_file() else ""
    drc_matches = re.findall(r"Verification Complete\s*:\s*(\d+)\s+(?:Viols|Violations)", drc, re.I)
    drc_count = int(drc_matches[-1]) if drc_matches else None
    antenna_path = reports / "innovus-antenna.rpt"
    antenna = antenna_path.read_text(errors="replace") if antenna_path.is_file() else ""
    hold = timing_summary(reports / "signoff_hold.summary")
    hold["target_slack_ns"] = hold_target_slack
    hold["target_met"] = hold["wns_ns"] is not None and hold["wns_ns"] >= hold_target_slack
    return {
        "schema_version": 1, "node": "cadence-innovus-pnr", "status": "passed",
        "run_mode": "full" if stop_after_step == "none" else "early_stop",
        "completed_step": "signoff" if stop_after_step == "none" else stop_after_step,
        "wall_seconds": wall_seconds, "stage_wall_seconds": stages,
        "placement": {
            "initial_density_percent": last_number(r"Density:\s*([0-9.]+)%", preplace),
            "final_density_percent": last_number(r"Density:\s*([0-9.]+)%", final),
            "instance_count": int(v) if (v := last_number(r"Total instances in design:\s*(\d+)", log)) is not None else None,
        },
        "timing": {"hold_target_slack_ns": hold_target_slack,
                   "postcts_hold": timing_summary(reports / "postcts_hold_hold.summary"),
                   "postroute_setup": timing_summary(reports / "postroute_setup.summary"),
                   "postroute_hold": timing_summary(reports / "postroute_hold_hold.summary"),
                   "signoff_setup": timing_summary(reports / "signoff.summary"),
                   "signoff_hold": hold},
        "drc": {"policy": drc_check_policy,
                "status": "not_run" if stop_after_step != "none" else "clean" if drc_count == 0 else "violations" if drc_count is not None else "unavailable",
                "violation_count": drc_count},
        "antenna": {"status": "not_run" if stop_after_step != "none" else "skipped" if "skipped by antenna_check_policy" in antenna else "error" if "ERROR:" in antenna else "passed"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall-seconds", required=True, type=float)
    parser.add_argument("--stop-after-step", default="none")
    parser.add_argument("--drc-check-policy", choices=("error", "report"), default="error")
    parser.add_argument("--hold-target-slack", type=float, default=0.005)
    args = parser.parse_args()
    Path("outputs/pnr-metrics.json").write_text(json.dumps(
        collect(Path("."), args.wall_seconds, args.stop_after_step,
                args.drc_check_policy, args.hold_target_slack), indent=2) + "\n")


if __name__ == "__main__":
    main()
