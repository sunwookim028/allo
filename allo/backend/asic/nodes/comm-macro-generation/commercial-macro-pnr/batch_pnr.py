#!/usr/bin/env python3
"""Run one isolated Innovus macro-hardening worker per synthesized class."""

from __future__ import annotations

import concurrent.futures
import json
import os
import signal
import shutil
import subprocess
import sys
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
    required_inputs = {
        "synthesis netlist": INPUT_BATCH / artifacts["netlist"],
        "synthesis SDC": INPUT_BATCH / artifacts["sdc"],
        "pin-intent Tcl": INPUT_BATCH / entry["pin_intent_tcl"],
        "pin-intent JSON": INPUT_BATCH / entry["pin_intent"],
    }
    missing_inputs = [
        f"{label}: {path}" for label, path in required_inputs.items() if not path.is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            f"macro PNR input validation failed for {entry_id}: "
            + "; ".join(missing_inputs)
        )
    symlink(required_inputs["synthesis netlist"], inputs / "design.v")
    symlink(required_inputs["synthesis SDC"], inputs / "design.sdc")
    symlink(required_inputs["pin-intent Tcl"], inputs / "pin-intent.tcl")
    symlink(required_inputs["pin-intent JSON"], inputs / "pin-intent.json")
    symlink(ROOT / "inputs" / "adk", inputs / "adk")
    srams = ROOT / "inputs" / "srams"
    if srams.exists():
        symlink(srams, inputs / "srams")

    env = os.environ.copy()
    env["design_name"] = entry["top_module"]
    timeout_seconds = int(os.environ.get("worker_timeout_seconds", "21600"))
    log_path = worker / "batch-driver.log"
    print(
        f"[commercial-macro-pnr] {entry_id}: starting Innovus "
        f"for {entry['top_module']}",
        flush=True,
    )
    timed_out = False
    with log_path.open("w") as log:
        process = subprocess.Popen(
            ["bash", "run.sh"],
            cwd=worker,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(
                timeout=timeout_seconds if timeout_seconds > 0 else None
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    print(
        f"[commercial-macro-pnr] {entry_id}: finished with return code "
        f"{returncode}{' (timeout)' if timed_out else ''}",
        flush=True,
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
    passed = returncode == 0 and not timed_out and not missing
    status = {
        "id": entry_id,
        "top_module": entry["top_module"],
        "status": "passed" if passed else "failed",
        "returncode": returncode,
        "timed_out": timed_out,
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


def emit_worker_logs(entries: list[dict]) -> None:
    """Replay every worker log to stdout for mflowgen-run.log."""
    print("\n===== BEGIN COMMERCIAL MACRO PNR WORKER LOGS =====", flush=True)
    for entry in entries:
        entry_id = entry["id"]
        worker = WORK_ROOT / entry_id
        logs = sorted(worker.rglob("*.log"))
        if not logs:
            print(f"\n===== WORKER {entry_id}: NO LOG FILES FOUND =====", flush=True)
        for path in logs:
            relative = path.relative_to(worker)
            print(f"\n===== BEGIN WORKER {entry_id} LOG {relative} =====", flush=True)
            with path.open(errors="replace") as source:
                shutil.copyfileobj(source, sys.stdout)
            print()
            print(f"===== END WORKER {entry_id} LOG {relative} =====", flush=True)
    print("===== END COMMERCIAL MACRO PNR WORKER LOGS =====", flush=True)


def main() -> None:
    index = json.loads((INPUT_BATCH / "index.json").read_text())
    entries = index.get("entries", [])
    if not entries:
        raise RuntimeError("synthesis batch contains no entries")
    shutil.rmtree(OUTPUT_BATCH, ignore_errors=True)
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    OUTPUT_BATCH.mkdir(parents=True)
    WORK_ROOT.mkdir()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(entries)) as executor:
            statuses = list(executor.map(run_entry, entries))
    finally:
        emit_worker_logs(entries)
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
