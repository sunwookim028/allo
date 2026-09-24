#!/usr/bin/env python3
"""Remove an explicitly allowlisted node's private scratch after success."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def size_bytes(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(
        item.lstat().st_size
        for item in path.rglob("*")
        if item.is_symlink() or item.is_file()
    )


def validate_target(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe cleanup target: {value!r}")
    if relative.parts[0] in {"inputs", "outputs"}:
        raise ValueError(f"cleanup cannot remove declared inputs/outputs: {value!r}")
    target = root / relative
    target.resolve(strict=False).relative_to(root.resolve())
    return target


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args()

    enabled_text = os.environ.get("cleanup_enabled", "True").strip().lower()
    if enabled_text not in {"true", "false"}:
        raise ValueError("cleanup_enabled must be True or False")
    enabled = enabled_text == "true"

    root = Path.cwd()
    output = root / "outputs" / args.output
    output.parent.mkdir(exist_ok=True)
    removed = []
    if enabled:
        for value in args.targets:
            target = validate_target(root, value)
            if not target.exists() and not target.is_symlink():
                continue
            count = size_bytes(target)
            remove(target)
            removed.append({"path": value, "apparent_size_bytes": count})

    deleted = sum(item["apparent_size_bytes"] for item in removed)
    output.write_text(json.dumps({
        "schema_version": 1,
        "node": args.node,
        "status": "passed" if enabled else "disabled",
        "cleanup_enabled": enabled,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "deleted_bytes": deleted,
        "deleted_gib": round(deleted / (1024 ** 3), 6),
        "removed": removed,
    }, indent=2) + "\n")
    state = "enabled" if enabled else "disabled"
    print(f"{args.node} cleanup ({state}): removed {len(removed)} targets, {deleted} bytes")


if __name__ == "__main__":
    main()
