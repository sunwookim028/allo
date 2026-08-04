#!/usr/bin/env python3
"""Run timing signoff and Liberty/DB extraction for each hardened macro."""

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
INPUT_BATCH = ROOT / "inputs" / "verified-pnr-batch"
OUTPUT_BATCH = ROOT / "outputs" / "signoff-batch"
WORK_ROOT = ROOT / "work"


def symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def stage_inputs(directory: Path, entry: dict) -> None:
    inputs = directory / "inputs"
    inputs.mkdir()
    artifacts = entry["artifacts"]
    symlink(INPUT_BATCH / artifacts["netlist"], inputs / "design.vcs.v")
    symlink(INPUT_BATCH / artifacts["sdc"], inputs / "design.pt.sdc")
    spef = INPUT_BATCH / artifacts["spef"]
    if spef.exists():
        symlink(spef, inputs / "design.spef.gz")
    symlink(ROOT / "inputs" / "adk", inputs / "adk")
    srams = ROOT / "inputs" / "srams"
    if srams.exists():
        symlink(srams, inputs / "srams")


def command(
    command: list[str], cwd: Path, env: dict[str, str], log_name: str
) -> tuple[int, float]:
    start = time.monotonic()
    with (cwd / log_name).open("w") as log:
        returncode = subprocess.run(
            command,
            cwd=cwd,
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
    destination = OUTPUT_BATCH / "entries" / entry_id
    shutil.rmtree(worker, ignore_errors=True)
    shutil.copytree(ROOT / "worker", worker)
    timing = worker / "timing"
    libdb = worker / "libdb"
    stage_inputs(timing, entry)
    stage_inputs(libdb, entry)
    (timing / "reports").mkdir()
    (libdb / "logs").mkdir()
    (libdb / "reports").mkdir()
    (libdb / "results").mkdir()

    env = os.environ.copy()
    env.update(
        {
            "design_name": entry["top_module"],
            "corner_setup": os.environ.get("corner_setup", "typical"),
            "corner_hold": os.environ.get("corner_hold", "bc"),
            "order": "read_design.tcl,extract_model.tcl,write-interface-timing.tcl",
        }
    )
    timing_rc, timing_seconds = command(
        ["pt_shell", "-file", "pt.tcl"], timing, env, "pt.log"
    )
    lib_rc, lib_seconds = command(
        ["pt_shell", "-file", "START.tcl", "-output_log_file", "logs/pt.log"],
        libdb,
        env,
        "pt-driver.log",
    )
    lc_rc = -1
    if lib_rc == 0:
        lc_rc, lc_seconds = command(
            ["lc_shell", "-f", "scripts/lib2db.tcl"], libdb, env, "logs/lib2db.log"
        )
    else:
        lc_seconds = 0.0

    expected = {
        "sdf": timing / "design.sdf",
        "lib": libdb / f"{entry['top_module']}.lib",
        "db": libdb / f"{entry['top_module']}.db",
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    passed = timing_rc == 0 and lib_rc == 0 and lc_rc == 0 and not missing
    status = {
        "id": entry_id,
        "top_module": entry["top_module"],
        "status": "passed" if passed else "failed",
        "timing_returncode": timing_rc,
        "lib_returncode": lib_rc,
        "lc_returncode": lc_rc,
        "missing_outputs": missing,
        "work_dir": str(worker),
        "timing_seconds": timing_seconds,
        "liberty_extraction_seconds": lib_seconds,
        "liberty_compile_seconds": lc_seconds,
        "wall_seconds": round(time.monotonic() - worker_start, 3),
    }
    if passed:
        shutil.rmtree(destination, ignore_errors=True)
        views = destination / "views"
        reports = destination / "reports"
        views.mkdir(parents=True)
        reports.mkdir()
        verification_source = INPUT_BATCH / "entries" / entry_id / "verification"
        if verification_source.exists():
            shutil.copytree(verification_source, destination / "verification")
        pnr_reports = INPUT_BATCH / "entries" / entry_id / "reports"
        if pnr_reports.exists():
            for source in pnr_reports.glob("*"):
                if source.is_file():
                    shutil.copy2(source, reports / f"pnr-{source.name}")
        for key in ["netlist", "sdc", "spef", "lef", "gds", "lvs_netlist"]:
            source = INPUT_BATCH / entry["artifacts"][key]
            if source.exists():
                suffix = {
                    "netlist": "macro.v",
                    "sdc": "macro.sdc",
                    "spef": "macro.spef.gz",
                    "lef": "macro.lef",
                    "gds": "macro.gds.gz",
                    "lvs_netlist": "macro.lvs.v",
                }[key]
                shutil.copy2(source, views / suffix)
        shutil.copy2(expected["sdf"], views / "macro.sdf")
        shutil.copy2(expected["lib"], views / "macro.lib")
        shutil.copy2(expected["db"], views / "macro.db")
        for source_dir, prefix in [(timing / "reports", "timing"), (libdb / "reports", "lib")]:
            for source in source_dir.glob("*"):
                if source.is_file():
                    shutil.copy2(source, reports / f"{prefix}-{source.name}")
        result = {
            **entry,
            "stage": "signoff",
            "views": {
                "verilog": f"entries/{entry_id}/views/macro.v",
                "liberty": f"entries/{entry_id}/views/macro.lib",
                "db": f"entries/{entry_id}/views/macro.db",
                "lef": f"entries/{entry_id}/views/macro.lef",
                "gds": f"entries/{entry_id}/views/macro.gds.gz",
                "spef": f"entries/{entry_id}/views/macro.spef.gz",
                "sdf": f"entries/{entry_id}/views/macro.sdf",
                "sdc": f"entries/{entry_id}/views/macro.sdc",
                "lvs_verilog": f"entries/{entry_id}/views/macro.lvs.v",
            },
        }
        (destination / "entry.json").write_text(json.dumps(result, indent=2) + "\n")
    status_dir = OUTPUT_BATCH / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{entry_id}.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def emit_worker_logs(entries: list[dict]) -> None:
    """Replay every worker log to stdout for mflowgen-run.log."""
    print("\n===== BEGIN COMMERCIAL BATCH SIGNOFF WORKER LOGS =====", flush=True)
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
    print("===== END COMMERCIAL BATCH SIGNOFF WORKER LOGS =====", flush=True)


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
            "stage": "signoff", "entries": [], "status": [],
        }
        (OUTPUT_BATCH / "index.json").write_text(json.dumps(output_index, indent=2) + "\n")
        status_doc = {"status": "bypassed", "total": 0, "passed": 0, "failed": 0, "entries": []}
        (ROOT / "outputs" / "signoff-status.json").write_text(json.dumps(status_doc, indent=2) + "\n")
        metrics = {
            "schema_version": 1, "node": "commercial-batch-signoff", "status": "bypassed",
            "wall_seconds": round(time.monotonic() - batch_start, 3), "workers": [],
            "aggregate": {"worker_count": 0, "maximum_worker_seconds": 0.0, "sum_worker_seconds": 0.0},
        }
        (ROOT / "outputs" / "macro-signoff-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("Macro signoff bypassed: batch contains zero entries.")
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
        "stage": "signoff",
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
    (ROOT / "outputs" / "signoff-status.json").write_text(
        json.dumps(status_doc, indent=2) + "\n"
    )
    worker_seconds = [item["wall_seconds"] for item in statuses]
    metrics = {
        "schema_version": 1,
        "node": "commercial-batch-signoff",
        "status": "passed" if len(passed) == len(statuses) else "failed",
        "wall_seconds": round(time.monotonic() - batch_start, 3),
        "workers": statuses,
        "aggregate": {
            "worker_count": len(statuses),
            "maximum_worker_seconds": max(worker_seconds, default=0.0),
            "sum_worker_seconds": round(sum(worker_seconds), 3),
        },
    }
    (ROOT / "outputs/macro-signoff-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    if len(passed) != len(statuses):
        raise RuntimeError(
            "macro signoff failed for: "
            + str([item["id"] for item in statuses if item["status"] != "passed"])
        )


if __name__ == "__main__":
    main()
