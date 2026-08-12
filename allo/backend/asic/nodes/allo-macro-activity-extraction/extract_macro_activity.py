#!/usr/bin/env python3
"""Extract one representative, instance-scoped SAIF for every macro class."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    inputs = Path("inputs")
    outputs = Path("outputs")
    activity_dir = outputs / "macro-activity"
    shutil.rmtree(activity_dir, ignore_errors=True)
    activity_dir.mkdir(parents=True)

    plan = json.loads((inputs / "assembly-plan.json").read_text())
    registry = json.loads((inputs / "macro-registry.json").read_text())
    registered = {item["macro_class_id"]: item for item in registry.get("macros", [])}
    tb = os.environ.get("testbench_name", "allo_generated_testbench")
    dut = os.environ.get("dut_name", "dut")
    records = []

    for macro_class in plan.get("classes", []):
        class_id = macro_class["macro_class_id"]
        if class_id not in registered:
            raise ValueError(f"assembly class is absent from macro registry: {class_id}")
        members = macro_class.get("members", [])
        if not members:
            raise ValueError(f"macro class has no representative member: {class_id}")
        member = members[0]
        stable_instance = member.get("stable_instance_name")
        paths = member.get("hierarchical_paths", [])
        instance = paths[0] if paths else f"{plan['top_module']}/{member['instance_name']}"
        # Hardened instances are emitted under stable_instance_name during RTL
        # assembly. Fall back to the original hierarchical path for plans made
        # before stable names were recorded.
        if stable_instance:
            relative = stable_instance
        else:
            top_prefix = f"{plan['top_module']}/"
            relative = instance[len(top_prefix):] if instance.startswith(top_prefix) else instance
        scope = "/".join([tb, dut, relative]).replace(".", "/")
        saif = activity_dir / f"{class_id}.saif"
        log = activity_dir / f"{class_id}.vcd2saif.log"
        command = [
            "vcd2saif", "-input", str((inputs / "run.vcd").resolve()),
            "-output", str(saif.resolve()), "-instance", scope,
        ]
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        passed = result.returncode == 0 and saif.is_file() and saif.stat().st_size > 0
        records.append({
            "macro_class_id": class_id,
            "canonical_module": macro_class["canonical_module"],
            "representative_instance": member["instance_name"],
            "vcd_scope": scope,
            "saif": saif.name,
            "reuse_count": int(registered[class_id]["reuse_count"]),
            "status": "passed" if passed else "failed",
            "returncode": result.returncode,
            "log": log.name,
        })

    failed = [item for item in records if item["status"] != "passed"]
    manifest = {
        "schema_version": 1,
        "class_count": len(records),
        "extracted_count": len(records) - len(failed),
        "failed_count": len(failed),
        "entries": records,
    }
    text = json.dumps(manifest, indent=2) + "\n"
    (activity_dir / "index.json").write_text(text)
    (outputs / "macro-activity-manifest.json").write_text(text)
    if failed:
        raise RuntimeError("SAIF extraction failed for: " + ", ".join(x["macro_class_id"] for x in failed))


if __name__ == "__main__":
    main()
