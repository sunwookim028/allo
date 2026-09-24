#!/usr/bin/env python3
"""Run representative macro power jobs in parallel and aggregate by reuse."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path.cwd()
WORK = ROOT / "work"
REPORTS = ROOT / "outputs" / "macro-power-reports"


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def parse_power(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    labels = {
        "switching_w": r"Net Switching Power\s*=\s*([0-9.eE+-]+)",
        "internal_w": r"Cell Internal Power\s*=\s*([0-9.eE+-]+)",
        "leakage_w": r"Cell Leakage Power\s*=\s*([0-9.eE+-]+)",
        "total_w": r"Total Power\s*=\s*([0-9.eE+-]+)",
    }
    result = {}
    for key, pattern in labels.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"could not parse {key} from {path}")
        result[key] = float(match.group(1))
    return result


def run_class(entry: dict, activity: dict) -> dict:
    class_id = entry["macro_class_id"]
    start = time.monotonic()
    worker = WORK / class_id
    print(f"[macro-power] starting {class_id}", flush=True)
    shutil.rmtree(worker, ignore_errors=True)
    worker.mkdir(parents=True)
    for name in ["START.tcl", "run.sh", "prepare_pt_sdc.py"]:
        shutil.copy2(ROOT / name, worker / name)
    shutil.copytree(ROOT / "scripts", worker / "scripts")
    inputs = worker / "inputs"
    inputs.mkdir()
    views = entry["views"]
    registry = ROOT / "inputs" / "macro-registry"
    link(registry / views["verilog"]["path"], inputs / "design.vcs.v")
    link(registry / views["sdc"]["path"], inputs / "design.pt.sdc")
    link(registry / views["spef"]["path"], inputs / "design.spef.gz")
    link(ROOT / "inputs" / "macro-activity" / activity["saif"], inputs / "run.saif")
    link(ROOT / "inputs" / "adk", inputs / "adk")
    srams = ROOT / "inputs" / "srams"
    if srams.exists():
        link(srams, inputs / "srams")
    env = os.environ.copy()
    env.update({
        "design_name": entry["top_module"],
        "saif_instance": activity["vcd_scope"],
        "analysis_mode": "averaged",
        "zero_delay_simulation": "False",
        "lib_op_condition": os.environ.get("lib_op_condition", "undefined"),
        "order": "designer-interface.tcl,setup-session.tcl,read-design.tcl,report-timing.tcl,report-power.tcl",
    })
    log_path = worker / "batch-power.log"
    with log_path.open("w") as log:
        prepare_rc = subprocess.run(
            ["python3", "prepare_pt_sdc.py", "inputs/design.pt.sdc",
             "prepared-design.pt.sdc"],
            cwd=worker, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        ).returncode
        if prepare_rc == 0:
            rc = subprocess.run(
                ["bash", "run.sh"], cwd=worker, env=env,
                stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            ).returncode
        else:
            rc = prepare_rc
    report = worker / "reports" / f"{entry['top_module']}.power.rpt"
    passed = rc == 0 and report.is_file()
    metrics = parse_power(report) if passed else {}
    destination = REPORTS / class_id
    destination.mkdir(parents=True, exist_ok=True)
    for source in (worker / "reports").glob("*") if (worker / "reports").exists() else []:
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    shutil.copy2(worker / "batch-power.log", destination / "batch-power.log")
    reuse = int(entry["reuse_count"])
    result = {
        "macro_class_id": class_id, "top_module": entry["top_module"],
        "representative_instance": activity["representative_instance"],
        "reuse_count": reuse, "status": "passed" if passed else "failed",
        "returncode": rc, "representative_power_w": metrics,
        "weighted_power_w": {key: value * reuse for key, value in metrics.items()},
        "wall_seconds": round(time.monotonic() - start, 3),
    }
    print(
        f"[macro-power] {result['status']} {class_id} "
        f"({result['wall_seconds']:.3f}s); log: {log_path}",
        flush=True,
    )
    return result


def main() -> None:
    batch_start = time.monotonic()
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(REPORTS, ignore_errors=True)
    WORK.mkdir()
    REPORTS.mkdir(parents=True)
    registry = json.loads((ROOT / "inputs/macro-registry/index.json").read_text())
    activity_doc = json.loads((ROOT / "inputs/macro-activity/index.json").read_text())
    activities = {x["macro_class_id"]: x for x in activity_doc.get("entries", [])}
    entries = registry.get("macros", [])
    missing = [x["macro_class_id"] for x in entries if x["macro_class_id"] not in activities]
    if missing:
        raise ValueError("missing macro activity for: " + ", ".join(missing))
    requested = max(1, int(os.environ.get("macro_power_max_workers", "4")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(requested, max(1, len(entries)))) as pool:
        results = list(pool.map(lambda x: run_class(x, activities[x["macro_class_id"]]), entries))
    failed = [x for x in results if x["status"] != "passed"]
    keys = ["switching_w", "internal_w", "leakage_w", "total_w"]
    macro_totals = {key: sum(x.get("weighted_power_w", {}).get(key, 0.0) for x in results) for key in keys}
    top = parse_power(ROOT / "inputs/power.rpt")
    combined = {key: top[key] + macro_totals[key] for key in keys}
    summary = {
        "schema_version": 1, "class_count": len(entries),
        "node": "commercial-batch-macro-power",
        "status": "passed" if not failed else "failed",
        "wall_seconds": round(time.monotonic() - batch_start, 3),
        "represented_instance_count": sum(int(x["reuse_count"]) for x in entries),
        "failed_count": len(failed), "top_level_power_w": top,
        "weighted_macro_power_w": macro_totals, "combined_power_w": combined,
        "classes": results,
    }
    (ROOT / "outputs/macro-power-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["Allo hierarchical aggregate power (W)", "",
             "Component                 Switching       Internal        Leakage          Total",
             f"Top-level shell       {top['switching_w']:14.6g} {top['internal_w']:14.6g} {top['leakage_w']:14.6g} {top['total_w']:14.6g}",
             f"Weighted macros       {macro_totals['switching_w']:14.6g} {macro_totals['internal_w']:14.6g} {macro_totals['leakage_w']:14.6g} {macro_totals['total_w']:14.6g}",
             f"Combined estimate     {combined['switching_w']:14.6g} {combined['internal_w']:14.6g} {combined['leakage_w']:14.6g} {combined['total_w']:14.6g}", ""]
    (ROOT / "outputs/aggregate-power.rpt").write_text("\n".join(lines))
    if failed:
        raise RuntimeError("macro power failed for: " + ", ".join(x["macro_class_id"] for x in failed))


if __name__ == "__main__":
    main()
