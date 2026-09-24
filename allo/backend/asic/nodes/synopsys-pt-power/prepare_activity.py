#!/usr/bin/env python3
"""Validate and describe the explicitly selected PrimeTime activity source."""

from __future__ import annotations

import json
import os
from pathlib import Path


INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
SOURCES = {
    "bagl_vcd": {"format": "vcd", "path": "inputs/run.vcd", "rtl": False, "zero_delay": False},
    "ffgl_vcd": {"format": "vcd", "path": "inputs/run.vcd", "rtl": False, "zero_delay": True},
    "rtl_vcd": {"format": "vcd", "path": "inputs/run.vcd", "rtl": True, "zero_delay": True},
    "saif": {"format": "saif", "path": "inputs/run.saif", "rtl": False, "zero_delay": False},
}


def is_text_vcd(path: Path) -> bool:
    with path.open("rb") as stream:
        header = stream.read(4096)
    if b"\0" in header:
        return False
    return any(marker in header for marker in (b"$date", b"$version", b"$timescale", b"$scope"))


def tcl_atom(value: str) -> str:
    return "{" + value.replace("}", "\\}") + "}"


def main() -> None:
    source_name = os.environ.get("activity_source", "bagl_vcd").strip()
    if source_name not in SOURCES:
        raise ValueError(
            f"activity_source must be one of {', '.join(SOURCES)}, got {source_name!r}"
        )
    source = dict(SOURCES[source_name])
    activity_path = Path(source["path"])
    if not activity_path.is_file() or activity_path.stat().st_size == 0:
        raise FileNotFoundError(f"selected {source_name} activity is missing or empty: {activity_path}")
    if source["format"] == "vcd" and not is_text_vcd(activity_path):
        raise ValueError(f"selected {source_name} activity is not a textual VCD: {activity_path}")
    if source_name == "rtl_vcd" and not (INPUTS / "design.namemap").is_file():
        raise FileNotFoundError("rtl_vcd requires inputs/design.namemap")

    analysis_mode = os.environ.get("analysis_mode", "averaged")
    if analysis_mode not in {"averaged", "time_based"}:
        raise ValueError("analysis_mode must be averaged or time_based")
    if analysis_mode == "time_based" and source["format"] != "vcd":
        raise ValueError("time_based power analysis requires a VCD activity source")

    strip_path = os.environ.get("saif_instance", "auto").strip()
    contract_used = False
    if strip_path == "auto":
        contract_path = INPUTS / "testbench-contract.json"
        if not contract_path.is_file():
            raise FileNotFoundError("saif_instance=auto requires inputs/testbench-contract.json")
        contract = json.loads(contract_path.read_text())
        strip_path = f'{contract["testbench_top"]}/{contract["dut_instance"]}'
        contract_used = True
    if not strip_path or any(character.isspace() for character in strip_path):
        raise ValueError(f"invalid activity strip path: {strip_path!r}")

    result = {
        "schema_version": 1,
        "activity_source": source_name,
        "activity_format": source["format"],
        "activity_path": source["path"],
        "analysis_mode": analysis_mode,
        "strip_path": strip_path,
        "strip_path_from_testbench_contract": contract_used,
        "rtl_name_mapping": source["rtl"],
        "zero_delay": source["zero_delay"],
    }
    Path("activity-source.json").write_text(json.dumps(result, indent=2) + "\n")
    Path("activity-config.tcl").write_text("\n".join([
        f"set ptpx_strip_path {tcl_atom(strip_path)}",
        f"set ptpx_activity_source {tcl_atom(source_name)}",
        f"set ptpx_activity_format {tcl_atom(source['format'])}",
        f"set ptpx_activity_is_rtl {'True' if source['rtl'] else 'False'}",
        f"set ptpx_activity_is_zero_delay {'True' if source['zero_delay'] else 'False'}",
    ]) + "\n")


if __name__ == "__main__":
    main()
