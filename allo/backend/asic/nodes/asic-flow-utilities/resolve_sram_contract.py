#!/usr/bin/env python3
"""Validate an SRAM contract and publish exact, consumer-ready view lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VIEWS = ("verilog", "liberty", "database", "lef", "gds", "spice")


def resolved_views(contract_path: Path, root: Path) -> tuple[int, dict[str, list[str]]]:
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != "sram-collateral-contract":
        raise ValueError("unsupported SRAM contract schema")
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported SRAM contract version")
    entries = contract.get("srams")
    num_srams = contract.get("num_srams")
    if not isinstance(entries, list) or not isinstance(num_srams, int):
        raise ValueError("SRAM contract requires integer num_srams and an srams list")
    if num_srams != len(entries):
        raise ValueError("num_srams does not match the number of SRAM entries")

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SRAM package does not exist: {root}")
    result = {view: [] for view in VIEWS}
    names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError("each SRAM entry requires a string name")
        if entry["name"] in names:
            raise ValueError(f"duplicate SRAM entry: {entry['name']}")
        names.add(entry["name"])
        views = entry.get("views")
        if not isinstance(views, dict):
            raise ValueError(f"SRAM {entry['name']} has no view registry")
        for view in VIEWS:
            paths = views.get(view)
            if not isinstance(paths, list) or (num_srams and not paths):
                raise ValueError(f"SRAM {entry['name']} has an invalid {view} view list")
            for relative in paths:
                if not isinstance(relative, str) or not relative:
                    raise ValueError(f"SRAM {entry['name']} has an invalid {view} path")
                candidate = (root / relative).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    raise ValueError(f"SRAM view escapes the package: {relative}")
                if not candidate.is_file():
                    raise FileNotFoundError(f"registered SRAM view does not exist: {candidate}")
                result[view].append(str(candidate))
    return num_srams, {view: sorted(set(paths)) for view, paths in result.items()}


def tcl_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("$", "\\$")
    escaped = escaped.replace("[", "\\[").replace("]", "\\]").replace('"', '\\"')
    return f'"{escaped}"'


def write_tcl(path: Path, num_srams: int, views: dict[str, list[str]]) -> None:
    lines = [f"set sram_contract_num_srams {num_srams}"]
    for view in VIEWS:
        values = " ".join(tcl_quote(item) for item in views[view])
        lines.append(f"set sram_{view}_files [list {values}]")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--view", choices=VIEWS)
    group.add_argument("--tcl-output", type=Path)
    args = parser.parse_args()

    num_srams, views = resolved_views(args.contract, args.root)
    if args.view:
        for path in views[args.view]:
            print(path)
    else:
        write_tcl(args.tcl_output, num_srams, views)


if __name__ == "__main__":
    main()
