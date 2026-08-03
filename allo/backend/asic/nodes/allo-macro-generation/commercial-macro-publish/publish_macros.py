#!/usr/bin/env python3
"""Validate and publish the hardened canonical macro registry."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


REQUIRED_VIEWS = ["verilog", "liberty", "db", "lef", "gds", "spef", "sdf", "sdc"]
REQUIRED_LEF_SYMMETRIES = {"X", "Y", "R90"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tcl_quote(value: object) -> str:
    return "{" + str(value).replace("\\", "\\\\").replace("}", "\\}") + "}"


def lef_symmetries(path: Path, macro_name: str) -> list[str]:
    text = path.read_text(errors="replace")
    macro = re.search(
        rf"(?ms)^\s*MACRO\s+{re.escape(macro_name)}\s*$"
        rf"(?P<body>.*?)^\s*END\s+{re.escape(macro_name)}\s*$",
        text,
    )
    if macro is None:
        raise ValueError(f"LEF does not define expected macro {macro_name}: {path}")
    match = re.search(r"(?m)^\s*SYMMETRY\s+([^;]+);", macro.group("body"))
    if match is None:
        raise ValueError(f"LEF macro {macro_name} has no SYMMETRY declaration")
    symmetries = match.group(1).split()
    missing = REQUIRED_LEF_SYMMETRIES - set(symmetries)
    if missing:
        raise ValueError(
            f"LEF macro {macro_name} cannot support Stage-2 D4 tiling; "
            f"missing symmetries {sorted(missing)}, found {symmetries}"
        )
    return symmetries


def main() -> None:
    source_root = Path("inputs/signoff-batch")
    output_root = Path("outputs/macro-registry")
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True)
    index = json.loads((source_root / "index.json").read_text())
    manifest = json.loads(Path("inputs/asic-manifest-final.json").read_text())
    manifest_classes = {
        group["macro_class_id"]: group for group in manifest.get("macro_groups", [])
    }
    published = []
    for entry in index.get("entries", []):
        class_id = entry["id"]
        if class_id not in manifest_classes:
            raise ValueError(f"signoff entry {class_id} is absent from ASIC manifest")
        if entry.get("reuse_count") != manifest_classes[class_id].get("member_count"):
            raise ValueError(f"reuse count changed for {class_id}")
        destination = output_root / class_id
        destination.mkdir()
        view_records = {}
        for view in REQUIRED_VIEWS:
            relative = entry.get("views", {}).get(view)
            if not relative:
                raise ValueError(f"{class_id} does not publish required {view} view")
            source = source_root / relative
            if not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"missing or empty {view} view for {class_id}: {source}")
            suffix = source.name.split("macro", 1)[-1]
            target = destination / f"{entry['top_module']}{suffix}"
            shutil.copy2(source, target)
            view_records[view] = {
                "path": f"{class_id}/{target.name}",
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
            }
        reports_dir = destination / "reports"
        reports_dir.mkdir()
        for source_dir in [
            source_root / "entries" / class_id / "reports",
            source_root / "entries" / class_id / "verification",
        ]:
            if source_dir.exists():
                for source in source_dir.glob("*"):
                    if source.is_file():
                        shutil.copy2(source, reports_dir / source.name)
        symmetry = lef_symmetries(
            destination / Path(view_records["lef"]["path"]).name,
            entry["top_module"],
        )
        record = {
            "macro_class_id": class_id,
            "top_module": entry["top_module"],
            "reuse_count": entry["reuse_count"],
            "rtl_hash": entry["rtl_hash"],
            "representative_semantic_id": entry["representative_semantic_id"],
            "member_modules": entry["member_modules"],
            "members": entry["members"],
            "member_placements": entry.get("member_placements", []),
            "port_maps": entry["port_maps"],
            "lef_symmetry": symmetry,
            "views": view_records,
            "reports": sorted(
                f"{class_id}/reports/{path.name}" for path in reports_dir.glob("*")
            ),
        }
        (destination / "macro.json").write_text(json.dumps(record, indent=2) + "\n")
        published.append(record)

    if not published:
        raise ValueError("signoff batch contains no publishable macro entries")
    registry = {
        "schema_version": 1,
        "stage": "hardened_macro_registry",
        "macro_count": len(published),
        "macros": published,
    }
    text = json.dumps(registry, indent=2) + "\n"
    (output_root / "index.json").write_text(text)
    Path("outputs/macro-registry.json").write_text(text)
    tcl = [
        "set allo_asic_macro_registry_schema_version 1",
        "set allo_asic_hardened_macro_classes [list "
        + " ".join(tcl_quote(item["macro_class_id"]) for item in published)
        + "]",
    ]
    for item in published:
        key = item["macro_class_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", key):
            raise ValueError(f"macro class ID is not Tcl-array safe: {key}")
        tcl.append(f"set allo_asic_hardened_macro_top({key}) {tcl_quote(item['top_module'])}")
        tcl.append(f"set allo_asic_hardened_macro_reuse({key}) {item['reuse_count']}")
        tcl.append(
            f"set allo_asic_hardened_macro_lef_symmetry({key}) "
            f"[list {' '.join(tcl_quote(value) for value in item['lef_symmetry'])}]"
        )
        for view, detail in item["views"].items():
            tcl.append(
                f"set allo_asic_hardened_macro_view({key},{view}) "
                f"{tcl_quote(detail['path'])}"
            )
    Path("outputs/macro-registry.tcl").write_text("\n".join(tcl) + "\n")
    message = f"Published {len(published)} hardened canonical macro classes.\n"
    Path("outputs/macro-publish.log").write_text(message)
    print(message, end="")


if __name__ == "__main__":
    main()
