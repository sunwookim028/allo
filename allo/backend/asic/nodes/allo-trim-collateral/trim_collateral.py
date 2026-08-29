#!/usr/bin/env python3
"""Conservatively trim recomputable collateral from a completed ASIC build."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
BUILD_ROOT = ROOT.parent
OUTPUTS = ROOT / "outputs"


def boolean_parameter(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def apparent_size_bytes(path: Path) -> int:
    """Sum stored files without following symbolic links."""
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    total = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                else:
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def node_directory(suffix: str) -> Path | None:
    matches = [
        path for path in BUILD_ROOT.iterdir()
        if path.is_dir() and re.fullmatch(rf"\d+-{re.escape(suffix)}", path.name)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple build nodes match {suffix}: {matches}")
    return matches[0] if matches else None


def target(node_suffix: str, relative: str) -> Path | None:
    node = node_directory(node_suffix)
    return node / relative if node is not None else None


def removal_targets() -> list[tuple[str, Path]]:
    specifications = [
        ("full-chip LVS extraction database", "commercial-full-chip-lvs", "svdb"),
        ("PrimeTime detailed activity report", "synopsys-pt-power", "reports/top.activity.pre.rpt"),
        ("published PrimeTime detailed activity report", "synopsys-pt-power", "outputs/power-reports/top.activity.pre.rpt"),
        ("PrimeTime saved session", "synopsys-pt-power", "outputs/primetime.session"),
        ("BAGL waveform", "allo-bagl-sim", "outputs/run.vcd"),
        ("BAGL compiled simulator database", "allo-bagl-sim", "simv.daidir"),
        ("BAGL compiled simulator objects", "allo-bagl-sim", "csrc"),
        ("FFGL waveform", "allo-ffgl-sim", "outputs/run.vcd"),
        ("FFGL compiled simulator database", "allo-ffgl-sim", "simv.daidir"),
        ("FFGL compiled simulator objects", "allo-ffgl-sim", "csrc"),
        ("Allo compilation scratch", "allo-asic-compilation", "work"),
        ("macro PNR worker scratch", "commercial-macro-pnr", "work"),
        ("macro physical-verification worker scratch", "commercial-batch-physical-verify", "work"),
        ("macro signoff worker scratch", "commercial-batch-signoff", "work"),
        ("macro power worker scratch", "commercial-batch-macro-power", "work"),
    ]
    result = []
    for description, suffix, relative in specifications:
        path = target(suffix, relative)
        if path is not None:
            result.append((description, path))
    for cache_name in ("__pycache__", ".pytest_cache"):
        for path in BUILD_ROOT.rglob(cache_name):
            result.append((f"generated {cache_name}", path))
    return result


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    if not (BUILD_ROOT / ".mflowgen").is_dir():
        raise RuntimeError(f"refusing to trim non-mflowgen directory: {BUILD_ROOT}")
    enabled = boolean_parameter("trim_collateral", False)
    before = apparent_size_bytes(BUILD_ROOT)
    records = []
    if enabled:
        for description, path in removal_targets():
            if not path.exists() and not path.is_symlink():
                continue
            try:
                relative = path.relative_to(BUILD_ROOT)
            except ValueError as error:
                raise RuntimeError(f"trim target escapes build root: {path}") from error
            size = apparent_size_bytes(path)
            remove(path)
            records.append({
                "path": str(relative),
                "description": description,
                "apparent_size_bytes": size,
            })
    after = apparent_size_bytes(BUILD_ROOT)
    OUTPUTS.mkdir(exist_ok=True)
    metrics = {
        "schema_version": 1,
        "node": "allo-trim-collateral",
        "status": "passed" if enabled else "disabled",
        "trim_collateral": enabled,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_root": str(BUILD_ROOT),
        "before_apparent_size_bytes": before,
        "before_apparent_size_gib": round(before / (1024 ** 3), 6),
        "after_apparent_size_bytes": after,
        "after_apparent_size_gib": round(after / (1024 ** 3), 6),
        "deleted_bytes": max(0, before - after),
        "deleted_gib": round(max(0, before - after) / (1024 ** 3), 6),
        "removed_count": len(records),
        "removed": records,
    }
    (OUTPUTS / "trim-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"Collateral trim {metrics['status']}: "
        f"{metrics['before_apparent_size_gib']} -> "
        f"{metrics['after_apparent_size_gib']} GiB"
    )


if __name__ == "__main__":
    main()
