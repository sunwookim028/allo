#!/usr/bin/env python3
"""Remove large cross-node artifacts after every consumer has completed."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
BUILD_ROOT = ROOT.parent
OUTPUT = ROOT / "outputs" / "final-cleanup-metrics.json"
VCD_NODE_SUFFIXES = (
    "commercial-rtl-sim",
    "commercial-ffgl-sim",
    "commercial-bagl-sim",
    "allo-rtl-sim",
    "allo-ffgl-sim",
    "allo-bagl-sim",
)


def enabled_parameter() -> bool:
    value = os.environ.get("cleanup_enabled", "True").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError("cleanup_enabled must be True or False")
    return value == "true"


def find_node(suffix: str) -> Path | None:
    matches = [
        path for path in BUILD_ROOT.iterdir()
        if path.is_dir() and re.fullmatch(rf"\d+-{re.escape(suffix)}", path.name)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple build nodes match {suffix}: {matches}")
    return matches[0] if matches else None


def main() -> None:
    if not (BUILD_ROOT / ".mflowgen").is_dir():
        raise RuntimeError(f"refusing to finalize non-mflowgen directory: {BUILD_ROOT}")

    enabled = enabled_parameter()
    removed = []
    if enabled:
        for suffix in VCD_NODE_SUFFIXES:
            node = find_node(suffix)
            if node is None:
                continue
            path = node / "outputs" / "run.vcd"
            if not path.is_file() and not path.is_symlink():
                continue
            size = path.lstat().st_size if path.is_symlink() else path.stat().st_size
            path.unlink()
            removed.append({
                "node": suffix,
                "path": str(path.relative_to(BUILD_ROOT)),
                "apparent_size_bytes": size,
            })

    deleted = sum(item["apparent_size_bytes"] for item in removed)
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "schema_version": 1,
        "node": "asic-flow-finalize",
        "status": "passed" if enabled else "disabled",
        "cleanup_enabled": enabled,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "deleted_bytes": deleted,
        "deleted_gib": round(deleted / (1024 ** 3), 6),
        "removed": removed,
    }, indent=2) + "\n")
    print(f"Final cleanup removed {len(removed)} VCDs ({deleted} bytes)")


if __name__ == "__main__":
    main()
