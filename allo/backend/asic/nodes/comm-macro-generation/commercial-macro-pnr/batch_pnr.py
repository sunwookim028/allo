#!/usr/bin/env python3
"""Run one isolated Innovus macro-hardening worker per synthesized class."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path.cwd()
INPUT_BATCH = ROOT / "inputs" / "synthesis-batch"
OUTPUT_BATCH = ROOT / "outputs" / "pnr-batch"
WORK_ROOT = ROOT / "work"


def symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def run_entry(entry: dict) -> dict:
    entry_id = entry["id"]
    worker = WORK_ROOT / entry_id
    destination = OUTPUT_BATCH / "entries" / entry_id
    shutil.rmtree(worker, ignore_errors=True)
    shutil.copytree(ROOT / "worker", worker)
    inputs = worker / "inputs"
    inputs.mkdir()
    artifacts = entry["artifacts"]
    symlink(INPUT_BATCH / artifacts["netlist"], inputs / "design.v")
    symlink(INPUT_BATCH / artifacts["sdc"], inputs / "design.sdc")
    symlink(INPUT_BATCH / entry["pin_intent_tcl"], inputs / "pin-intent.tcl")
    symlink(INPUT_BATCH / entry["pin_intent"], inputs / "pin-intent.json")
    symlink(ROOT / "inputs" / "adk", inputs / "adk")
    srams = ROOT / "inputs" / "srams"
    if srams.exists():
        symlink(srams, inputs / "srams")

    env = os.environ.copy()
    env["design_name"] = entry["top_module"]
    log_path = worker / "batch-driver.log"
    with log_path.open("w") as log:
        process = subprocess.run(
            ["bash", "run.sh"],
            cwd=worker,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    required = [
        "design.gds.gz",
        "design-merged.gds",
        "design.vcs.v",
        "design.lvs.v",
        "design.lef",
        "design.pt.sdc",
    ]
    missing = [name for name in required if not (worker / "outputs" / name).exists()]
    passed = process.returncode == 0 and not missing
    status = {
        "id": entry_id,
        "top_module": entry["top_module"],
        "status": "passed" if passed else "failed",
        "returncode": process.returncode,
        "missing_outputs": missing,
        "work_dir": str(worker),
    }
    if passed:
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True)
        for directory in ["outputs", "reports", "logs", "results"]:
            source = worker / directory
            if source.exists():
                shutil.copytree(source, destination / directory, symlinks=False)
        result = {
            **entry,
            "stage": "pnr",
            "artifacts": {
                "netlist": f"entries/{entry_id}/outputs/design.vcs.v",
                "sdc": f"entries/{entry_id}/outputs/design.pt.sdc",
                "spef": f"entries/{entry_id}/outputs/design.spef.gz",
                "sdf": f"entries/{entry_id}/outputs/design.sdf",
                "lef": f"entries/{entry_id}/outputs/design.lef",
                "gds": f"entries/{entry_id}/outputs/design.gds.gz",
                "merged_gds": f"entries/{entry_id}/outputs/design-merged.gds",
                "lvs_netlist": f"entries/{entry_id}/outputs/design.lvs.v",
            },
        }
        (destination / "entry.json").write_text(json.dumps(result, indent=2) + "\n")
    status_dir = OUTPUT_BATCH / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{entry_id}.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main() -> None:
    index = json.loads((INPUT_BATCH / "index.json").read_text())
    entries = index.get("entries", [])
    if not entries:
        raise RuntimeError("synthesis batch contains no entries")
    shutil.rmtree(OUTPUT_BATCH, ignore_errors=True)
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    OUTPUT_BATCH.mkdir(parents=True)
    WORK_ROOT.mkdir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(entries)) as executor:
        statuses = list(executor.map(run_entry, entries))
    passed = {item["id"] for item in statuses if item["status"] == "passed"}
    output_entries = [
        json.loads((OUTPUT_BATCH / "entries" / entry["id"] / "entry.json").read_text())
        for entry in entries
        if entry["id"] in passed
    ]
    output_index = {
        **{key: value for key, value in index.items() if key not in {"entries", "status"}},
        "stage": "pnr",
        "entries": output_entries,
        "status": statuses,
    }
    (OUTPUT_BATCH / "index.json").write_text(json.dumps(output_index, indent=2) + "\n")
    status_doc = {
        "total": len(statuses),
        "passed": len(passed),
        "failed": len(statuses) - len(passed),
        "entries": statuses,
    }
    (ROOT / "outputs" / "pnr-status.json").write_text(
        json.dumps(status_doc, indent=2) + "\n"
    )
    if len(passed) != len(statuses):
        raise RuntimeError(
            "macro PNR failed for: "
            + str([item["id"] for item in statuses if item["status"] != "passed"])
        )


if __name__ == "__main__":
    main()
