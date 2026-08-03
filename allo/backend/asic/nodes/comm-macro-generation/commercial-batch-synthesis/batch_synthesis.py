#!/usr/bin/env python3
"""Run one isolated Design Compiler worker per selected macro class."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path.cwd()
INPUT_BATCH = ROOT / "inputs" / "macro-batch"
OUTPUT_BATCH = ROOT / "outputs" / "synthesis-batch"
WORK_ROOT = ROOT / "work"


def environment(entry: dict) -> dict[str, str]:
    env = os.environ.copy()
    values = {
        "design_name": entry["top_module"],
        "clock_period": os.environ.get("clock_period", "10.0"),
        "saif_instance": "undefined",
        "flatten_effort": os.environ.get("flatten_effort", "0"),
        "topographical": os.environ.get("topographical", "True"),
        "nthreads": os.environ.get("nthreads", "4"),
        "high_effort_area_opt": os.environ.get("high_effort_area_opt", "False"),
        "gate_clock": os.environ.get("gate_clock", "True"),
        "uniquify_with_design_name": os.environ.get(
            "uniquify_with_design_name", "True"
        ),
        "suppress_msg": os.environ.get("suppress_msg", "False"),
        "suppressed_msg": os.environ.get("suppressed_msg", ""),
        "write_svsim_wrapper": os.environ.get("write_svsim_wrapper", "False"),
        "order": ",".join(
            [
                "designer-interface.tcl",
                "setup-session.tcl",
                "read-design.tcl",
                "constraints.tcl",
                "make-path-groups.tcl",
                "compile-options.tcl",
                "compile.tcl",
                "generate-results.tcl",
                "reporting.tcl",
            ]
        ),
    }
    env.update(values)
    return env


def symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def fingerprint(entry: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(entry, sort_keys=True).encode())
    digest.update((INPUT_BATCH / entry["rtl"]).read_bytes())
    digest.update((INPUT_BATCH / entry["constraints"]).read_bytes())
    for name in [
        "clock_period",
        "flatten_effort",
        "topographical",
        "nthreads",
        "high_effort_area_opt",
        "gate_clock",
        "uniquify_with_design_name",
    ]:
        digest.update(f"{name}={os.environ.get(name, '')}".encode())
    return digest.hexdigest()


def collect_worker(worker: Path, destination: Path, entry: dict, fp: str) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    for directory in ["outputs", "reports", "logs", "results"]:
        source = worker / directory
        if source.exists():
            shutil.copytree(source, destination / directory, symlinks=False)
    collateral = destination / "collateral"
    collateral.mkdir()
    shutil.copy2(INPUT_BATCH / entry["pin_intent"], collateral / "pin-intent.json")
    shutil.copy2(INPUT_BATCH / entry["pin_intent_tcl"], collateral / "pin-intent.tcl")
    result = {
        **entry,
        "stage": "synthesis",
        "fingerprint": fp,
        "pin_intent": f"entries/{entry['id']}/collateral/pin-intent.json",
        "pin_intent_tcl": f"entries/{entry['id']}/collateral/pin-intent.tcl",
        "artifacts": {
            "netlist": f"entries/{entry['id']}/outputs/design.v",
            "sdc": f"entries/{entry['id']}/outputs/design.sdc",
            "spef": f"entries/{entry['id']}/outputs/design.spef.gz",
            "svf": f"entries/{entry['id']}/outputs/design.svf",
        },
    }
    (destination / "entry.json").write_text(json.dumps(result, indent=2) + "\n")


def run_entry(entry: dict) -> dict:
    entry_id = entry["id"]
    worker = WORK_ROOT / entry_id
    destination = OUTPUT_BATCH / "entries" / entry_id
    shutil.rmtree(worker, ignore_errors=True)
    shutil.copytree(ROOT / "worker", worker)
    inputs = worker / "inputs"
    inputs.mkdir()
    symlink(INPUT_BATCH / entry["rtl"], inputs / "design.v")
    symlink(INPUT_BATCH / entry["constraints"], inputs / "constraints.tcl")
    symlink(ROOT / "inputs" / "adk", inputs / "adk")
    srams = ROOT / "inputs" / "srams"
    if srams.exists():
        symlink(srams, inputs / "srams")

    fp = fingerprint(entry)
    log_path = worker / "batch-driver.log"
    with log_path.open("w") as log:
        process = subprocess.run(
            ["bash", "run.sh"],
            cwd=worker,
            env=environment(entry),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = {
        "id": entry_id,
        "top_module": entry["top_module"],
        "status": "passed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "fingerprint": fp,
        "work_dir": str(worker),
    }
    if process.returncode == 0:
        required = [worker / "outputs" / "design.v", worker / "outputs" / "design.sdc"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            status.update(status="failed", reason=f"missing outputs: {missing}")
        else:
            collect_worker(worker, destination, entry, fp)
    shutil.copy2(log_path, worker / "batch-driver.saved.log")
    status_dir = OUTPUT_BATCH / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{entry_id}.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main() -> None:
    index = json.loads((INPUT_BATCH / "index.json").read_text())
    entries = index.get("entries", [])
    if not entries:
        raise RuntimeError("macro batch contains no entries")
    shutil.rmtree(OUTPUT_BATCH, ignore_errors=True)
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    OUTPUT_BATCH.mkdir(parents=True)
    WORK_ROOT.mkdir()

    # Intentionally launch the whole batch. Each commercial process has an
    # isolated work directory; license and scheduler limits remain external.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(entries)) as executor:
        statuses = list(executor.map(run_entry, entries))

    passed = {status["id"] for status in statuses if status["status"] == "passed"}
    output_entries = []
    for entry in entries:
        if entry["id"] in passed:
            output_entries.append(
                json.loads(
                    (OUTPUT_BATCH / "entries" / entry["id"] / "entry.json").read_text()
                )
            )
    output_index = {
        **{key: value for key, value in index.items() if key != "entries"},
        "stage": "synthesis",
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
    (ROOT / "outputs" / "synthesis-status.json").write_text(
        json.dumps(status_doc, indent=2) + "\n"
    )
    if len(passed) != len(statuses):
        failed = [status["id"] for status in statuses if status["status"] != "passed"]
        raise RuntimeError(f"macro synthesis failed for: {failed}")


if __name__ == "__main__":
    main()
