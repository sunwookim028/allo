#!/usr/bin/env python3
"""Run DRC and LVS for every hardened macro, concurrently and in isolation."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path.cwd()
INPUT_BATCH = ROOT / "inputs" / "pnr-batch"
OUTPUT_BATCH = ROOT / "outputs" / "verified-pnr-batch"
WORK_ROOT = ROOT / "work"


def symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def run_tool(directory: Path, env: dict[str, str]) -> tuple[int, float]:
    start = time.monotonic()
    with (directory / "driver.log").open("w") as log:
        returncode = subprocess.run(
            ["bash", "run.sh"],
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    return returncode, round(time.monotonic() - start, 3)


def run_entry(entry: dict) -> dict:
    worker_start = time.monotonic()
    entry_id = entry["id"]
    worker = WORK_ROOT / entry_id
    shutil.rmtree(worker, ignore_errors=True)
    shutil.copytree(ROOT / "worker", worker)
    env = os.environ.copy()
    env["design_name"] = entry["top_module"]
    env.setdefault("drc_nthreads", "4")
    env.setdefault("drc_rule_deck", "calibre-drc-block.rule")
    env.setdefault("drc_env_setup", "undefined")
    env.setdefault("lvs_nthreads", "4")
    env.setdefault("lvs_power_name", "VDD")
    env.setdefault("lvs_ground_name", "VSS")
    env.setdefault("lvs_hcells_file", "")
    env.setdefault("lvs_connect_names", "")
    env.setdefault("lvs_verify_netlist", "1")
    env.setdefault("lvs_extra_spice_include", "")
    artifacts = entry["artifacts"]
    for tool in ["drc", "lvs"]:
        inputs = worker / tool / "inputs"
        inputs.mkdir()
        symlink(ROOT / "inputs" / "adk", inputs / "adk")
        symlink(INPUT_BATCH / artifacts["merged_gds"], inputs / "design_merged.gds")
        srams = ROOT / "inputs" / "srams"
        if srams.exists():
            symlink(srams, inputs / "srams")
    symlink(INPUT_BATCH / artifacts["lvs_netlist"], worker / "lvs/inputs/design.lvs.v")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        drc_future = executor.submit(run_tool, worker / "drc", env)
        lvs_future = executor.submit(run_tool, worker / "lvs", env)
        drc_rc, drc_seconds = drc_future.result()
        lvs_rc, lvs_seconds = lvs_future.result()
    passed = drc_rc == 0 and lvs_rc == 0
    status = {
        "id": entry_id,
        "top_module": entry["top_module"],
        "status": "passed" if passed else "failed",
        "drc_returncode": drc_rc,
        "lvs_returncode": lvs_rc,
        "work_dir": str(worker),
        "drc_seconds": drc_seconds,
        "lvs_seconds": lvs_seconds,
        "wall_seconds": round(time.monotonic() - worker_start, 3),
    }
    if passed:
        destination = OUTPUT_BATCH / "entries" / entry_id
        shutil.copytree(INPUT_BATCH / "entries" / entry_id, destination, symlinks=False)
        verification = destination / "verification"
        verification.mkdir()
        for source, name in [
            (worker / "drc/drc.results", "drc.results"),
            (worker / "drc/drc.summary", "drc.summary"),
            (worker / "drc/drc.log", "drc.log"),
            (worker / "lvs/lvs.report", "lvs.report"),
            (worker / "lvs/lvs.log", "lvs.log"),
        ]:
            if source.exists():
                shutil.copy2(source, verification / name)
        result_path = destination / "entry.json"
        result = json.loads(result_path.read_text())
        result["stage"] = "physical_verify"
        result["verification"] = {
            "drc_summary": f"entries/{entry_id}/verification/drc.summary",
            "lvs_report": f"entries/{entry_id}/verification/lvs.report",
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    status_dir = OUTPUT_BATCH / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{entry_id}.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def emit_worker_logs(entries: list[dict]) -> None:
    """Replay every worker log to stdout for mflowgen-run.log."""
    print("\n===== BEGIN COMMERCIAL PHYSICAL VERIFY WORKER LOGS =====", flush=True)
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
    print("===== END COMMERCIAL PHYSICAL VERIFY WORKER LOGS =====", flush=True)


def main() -> None:
    batch_start = time.monotonic()
    index = json.loads((INPUT_BATCH / "index.json").read_text())
    entries = index.get("entries", [])
    shutil.rmtree(OUTPUT_BATCH, ignore_errors=True)
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    OUTPUT_BATCH.mkdir(parents=True)
    WORK_ROOT.mkdir()
    if not entries:
        output_index = {
            **{key: value for key, value in index.items() if key not in {"entries", "status"}},
            "stage": "physical_verify", "entries": [], "status": [],
        }
        (OUTPUT_BATCH / "index.json").write_text(json.dumps(output_index, indent=2) + "\n")
        status_doc = {"status": "bypassed", "total": 0, "passed": 0, "failed": 0, "entries": []}
        (ROOT / "outputs/physical-verify-status.json").write_text(json.dumps(status_doc, indent=2) + "\n")
        metrics = {
            "schema_version": 1, "node": "commercial-batch-physical-verify", "status": "bypassed",
            "wall_seconds": round(time.monotonic() - batch_start, 3), "workers": [],
            "aggregate": {"worker_count": 0, "maximum_worker_seconds": 0.0, "sum_worker_seconds": 0.0},
        }
        (ROOT / "outputs/macro-physical-verify-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("Macro physical verification bypassed: batch contains zero entries.")
        return
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
        "stage": "physical_verify",
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
    (ROOT / "outputs/physical-verify-status.json").write_text(
        json.dumps(status_doc, indent=2) + "\n"
    )
    worker_seconds = [item["wall_seconds"] for item in statuses]
    metrics = {
        "schema_version": 1,
        "node": "commercial-batch-physical-verify",
        "status": "passed" if len(passed) == len(statuses) else "failed",
        "wall_seconds": round(time.monotonic() - batch_start, 3),
        "workers": statuses,
        "aggregate": {
            "worker_count": len(statuses),
            "maximum_worker_seconds": max(worker_seconds, default=0.0),
            "sum_worker_seconds": round(sum(worker_seconds), 3),
        },
    }
    (ROOT / "outputs/macro-physical-verify-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    if len(passed) != len(statuses):
        raise RuntimeError(
            "macro physical verification failed for: "
            + str([item["id"] for item in statuses if item["status"] != "passed"])
        )


if __name__ == "__main__":
    main()
