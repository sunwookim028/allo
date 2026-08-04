#!/usr/bin/env python3
"""Build a dependency-free ASIC-flow summary from emitted reports and metrics."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
INPUTS = ROOT / "inputs"
REGISTRY_ROOT = INPUTS / "macro-registry"
OUTPUTS = ROOT / "outputs"

METRIC_FILES = [
    "allo-compilation-metrics.json",
    "macro-synthesis-metrics.json",
    "macro-pnr-metrics.json",
    "macro-physical-verify-metrics.json",
    "macro-signoff-metrics.json",
    "full-chip-synthesis-metrics.json",
    "full-chip-pnr-metrics.json",
    "full-chip-gdsmerge-metrics.json",
    "full-chip-drc-metrics.json",
    "full-chip-lvs-metrics.json",
]


def read_optional(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def parse_lef_size(text: str) -> tuple[float | None, float | None]:
    match = re.search(
        r"\bSIZE\s+([-+0-9.eE]+)\s+BY\s+([-+0-9.eE]+)\s*;", text
    )
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)


def parse_path_slack(text: str) -> float | None:
    values = [
        float(value)
        for value in re.findall(
            r"slack\s+\((?:MET|VIOLATED)\)\s+([-+0-9.eE]+)", text, re.I
        )
    ]
    return min(values) if values else None


def parse_global_timing(text: str) -> dict:
    result = {
        "setup_wns_ns": None,
        "setup_tns_ns": None,
        "hold_wns_ns": None,
        "hold_tns_ns": None,
    }
    sections = re.split(r"(?=Setup violations|Hold violations)", text, flags=re.I)
    for section in sections:
        kind = "setup" if section.lower().startswith("setup") else (
            "hold" if section.lower().startswith("hold") else None
        )
        if kind is None:
            continue
        for label in ("WNS", "TNS"):
            match = re.search(rf"^\s*{label}\s+([-+0-9.eE]+)", section, re.M)
            if match:
                result[f"{kind}_{label.lower()}_ns"] = float(match.group(1))
    if re.search(r"No setup violations found", text, re.I):
        result["setup_wns_ns"] = 0.0
        result["setup_tns_ns"] = 0.0
    if re.search(r"No hold violations found", text, re.I):
        result["hold_wns_ns"] = 0.0
        result["hold_tns_ns"] = 0.0
    return result


def parse_drc_count(text: str) -> int | None:
    match = re.search(r"TOTAL DRC Results Generated:\s*(\d+)", text, re.I)
    return int(match.group(1)) if match else None


def parse_innovus_violation_count(text: str) -> int | None:
    if re.search(r"skipped by antenna_check_policy=off", text, re.I):
        return None
    matches = re.findall(
        r"Verification Complete\s*:\s*(\d+)\s+Viols", text, re.I
    )
    return int(matches[-1]) if matches else None


def parse_lvs(text: str) -> str:
    if re.search(r"\bINCORRECT\b", text, re.I):
        return "failed"
    if re.search(r"\bCORRECT\b", text, re.I):
        return "passed"
    return "unavailable"


def parse_density(text: str) -> float | None:
    match = re.search(r"bins with density\s*>\s*[^=]+?=\s*([0-9.]+)\s*%", text)
    return float(match.group(1)) if match else None


def parse_innovus_area(text: str, design_name: str) -> dict:
    """Parse the top row of an Innovus ``report_area -verbose`` report."""
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != design_name:
            continue
        numeric = fields[1:]
        if len(numeric) != 10:
            continue
        try:
            values = [float(value) for value in numeric]
        except ValueError:
            continue
        total_area = values[1]
        macro_area = values[8]
        physical_only_area = values[9]
        return {
            "full_chip_report_total_area_um2": total_area,
            "linked_macro_abstract_area_um2": macro_area,
            "remaining_top_standard_cell_area_um2": max(
                0.0, total_area - macro_area
            ),
            "physical_only_cell_area_um2": physical_only_area,
        }
    return {
        "full_chip_report_total_area_um2": None,
        "linked_macro_abstract_area_um2": None,
        "remaining_top_standard_cell_area_um2": None,
        "physical_only_cell_area_um2": None,
    }


def percentage(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 6)


def coverage_metrics(intent: dict, area: dict, macros: list[dict]) -> dict:
    core = intent.get("core", {})
    width = core.get("width")
    height = core.get("height")
    core_area = (
        float(width) * float(height)
        if width is not None and height is not None
        else None
    )
    placements = intent.get("placements")
    footprint = None
    instance_count = None
    if isinstance(placements, list):
        pe_macros = [item for item in placements if item.get("kind") == "pe"]
        instance_count = len(pe_macros)
        footprint = sum(
            float(item["width"]) * float(item["height"])
            for item in pe_macros
        )
    internal_areas = [
        item.get("instantiated_internal_standard_cell_area_um2")
        for item in macros
    ]
    hardened_internal_area = (
        sum(internal_areas)
        if internal_areas and all(value is not None for value in internal_areas)
        else (0.0 if not macros else None)
    )
    remaining_area = area["remaining_top_standard_cell_area_um2"]
    equivalent_total = (
        hardened_internal_area + remaining_area
        if hardened_internal_area is not None and remaining_area is not None
        else None
    )
    return {
        "placed_macro_instance_count": instance_count,
        "instantiated_macro_footprint_area_um2": (
            round(footprint, 6) if footprint is not None else None
        ),
        "core_area_um2": round(core_area, 6) if core_area is not None else None,
        "physical_macro_coverage_percent": percentage(footprint, core_area),
        **area,
        "hardened_macro_internal_standard_cell_area_um2": (
            round(hardened_internal_area, 6)
            if hardened_internal_area is not None
            else None
        ),
        "equivalent_total_standard_cell_area_um2": (
            round(equivalent_total, 6) if equivalent_total is not None else None
        ),
        "logic_hardening_coverage_percent": percentage(
            hardened_internal_area, equivalent_total
        ),
    }


def load_stage_metrics() -> list[dict]:
    stages = []
    for name in METRIC_FILES:
        path = INPUTS / name
        if path.is_file():
            value = json.loads(path.read_text())
            value["source"] = name
            stages.append(value)
        else:
            stages.append({"node": name.removesuffix("-metrics.json"), "status": "unavailable", "source": name})
    return stages


def summarize_macro(macro: dict) -> dict:
    macro_id = macro["macro_class_id"]
    reports = REGISTRY_ROOT / macro_id / "reports"
    lef_info = macro.get("views", {}).get("lef", {})
    width, height = parse_lef_size(read_optional(REGISTRY_ROOT / lef_info.get("path", "")))
    global_reports = list(reports.glob("*report_global_timing.report"))
    timing = parse_global_timing(read_optional(global_reports[0])) if global_reports else parse_global_timing("")
    for kind in ("setup", "hold"):
        detailed = list(reports.glob(f"*timing.{kind}.rpt"))
        if detailed:
            detailed_slack = parse_path_slack(read_optional(detailed[0]))
            if detailed_slack is not None:
                timing[f"{kind}_wns_ns"] = detailed_slack
    drc_reports = list(reports.glob("*drc.summary"))
    lvs_reports = list(reports.glob("*lvs.report"))
    density_reports = list(reports.glob("*density.rpt"))
    area_reports = list(reports.glob("*signoff.area.rpt"))
    internal_area = None
    if area_reports:
        internal_area = parse_innovus_area(
            read_optional(area_reports[0]), macro.get("top_module", "")
        )["full_chip_report_total_area_um2"]
    reuse_count = macro.get("reuse_count")
    return {
        "macro_class_id": macro_id,
        "top_module": macro.get("top_module"),
        "reuse_count": reuse_count,
        "width_um": width,
        "height_um": height,
        "physical_area_um2": round(width * height, 6) if width is not None and height is not None else None,
        "internal_standard_cell_area_um2": internal_area,
        "instantiated_internal_standard_cell_area_um2": (
            round(internal_area * reuse_count, 6)
            if internal_area is not None and reuse_count is not None
            else None
        ),
        "density_over_threshold_percent": parse_density(read_optional(density_reports[0])) if density_reports else None,
        **timing,
        "drc_results": parse_drc_count(read_optional(drc_reports[0])) if drc_reports else None,
        "lvs_status": parse_lvs(read_optional(lvs_reports[0])) if lvs_reports else "unavailable",
        "power": {"status": "unavailable", "reason": "no power-analysis report is emitted in Stage 1"},
    }


def tcl_atom(value: object) -> str:
    if value is None:
        return "unavailable"
    text = str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return "{" + text + "}"


def write_tcl(summary: dict) -> None:
    coverage = summary["coverage"]
    lines = [
        "# Generated by allo-asic-flow-summary; sourceable by Tcl.",
        f"set allo_asic_flow_summary_schema_version {summary['schema_version']}",
        f"set allo_asic_report_design_name {tcl_atom(summary['run']['design_name'])}",
        f"set allo_asic_rtl_top_module {tcl_atom(summary['run']['rtl_top_module'])}",
        f"set allo_asic_implementation_style {tcl_atom(summary['implementation_style'])}",
        f"set allo_asic_macro_generation_bypassed {1 if summary['macro_generation_bypassed'] else 0}",
        f"set allo_asic_macro_count {summary['macro_count']}",
        f"set allo_asic_physical_macro_coverage_percent {tcl_atom(coverage['physical_macro_coverage_percent'])}",
        f"set allo_asic_logic_hardening_coverage_percent {tcl_atom(coverage['logic_hardening_coverage_percent'])}",
        f"set allo_asic_full_chip_drc_results {tcl_atom(summary['full_chip_verification']['drc_results'])}",
        f"set allo_asic_calibre_non_antenna_results {tcl_atom(summary['full_chip_verification']['calibre_non_antenna_results'])}",
        f"set allo_asic_calibre_antenna_results {tcl_atom(summary['full_chip_verification']['calibre_antenna_results'])}",
        f"set allo_asic_calibre_antenna_policy {tcl_atom(summary['full_chip_verification']['calibre_antenna_policy'])}",
        f"set allo_asic_innovus_route_drc_results {tcl_atom(summary['full_chip_verification']['innovus_route_drc_results'])}",
        f"set allo_asic_innovus_antenna_results {tcl_atom(summary['full_chip_verification']['innovus_antenna_results'])}",
        f"set allo_asic_full_chip_lvs_status {tcl_atom(summary['full_chip_verification']['lvs_status'])}",
    ]
    stage_names = []
    for stage in summary["stages"]:
        name = stage["node"]
        stage_names.append(name)
        lines.append(f"set allo_asic_stage_status({name}) {tcl_atom(stage.get('status'))}")
        lines.append(f"set allo_asic_stage_wall_seconds({name}) {tcl_atom(stage.get('wall_seconds'))}")
    lines.append("set allo_asic_stage_names [list " + " ".join(tcl_atom(x) for x in stage_names) + "]")
    ids = []
    for macro in summary["macros"]:
        macro_id = macro["macro_class_id"]
        ids.append(macro_id)
        for key, value in macro.items():
            if not isinstance(value, dict):
                lines.append(f"set allo_asic_macro({macro_id},{key}) {tcl_atom(value)}")
    lines.append("set allo_asic_macro_ids [list " + " ".join(tcl_atom(x) for x in ids) + "]")
    (OUTPUTS / "flow-summary.tcl").write_text("\n".join(lines) + "\n")


def write_text(summary: dict) -> None:
    lines = [
        "Allo ASIC flow summary",
        f"Design: {summary['run']['design_name']}",
        f"RTL top module: {summary['run']['rtl_top_module']}",
        "",
        "Stage runtimes:",
    ]
    for stage in summary["stages"]:
        lines.append(f"  {stage['node']}: {stage.get('status')} ({stage.get('wall_seconds', 'unavailable')} s)")
    verification = summary["full_chip_verification"]
    coverage = summary["coverage"]
    lines.extend([
        "",
        "Macro coverage:",
        f"  Physical footprint/core: {coverage['instantiated_macro_footprint_area_um2']} / {coverage['core_area_um2']} um^2 ({coverage['physical_macro_coverage_percent']}%)",
        f"  Hardened logic/equivalent total cell area: {coverage['hardened_macro_internal_standard_cell_area_um2']} / {coverage['equivalent_total_standard_cell_area_um2']} um^2 ({coverage['logic_hardening_coverage_percent']}%)",
        f"  Remaining top-level standard-cell area: {coverage['remaining_top_standard_cell_area_um2']} um^2",
        "",
        "Full-chip verification:",
        f"  DRC results: {verification['drc_results']}",
        f"  Calibre non-antenna results: {verification['calibre_non_antenna_results']}",
        f"  Calibre antenna results: {verification['calibre_antenna_results']} (policy={verification['calibre_antenna_policy']})",
        f"  Innovus route DRC results: {verification['innovus_route_drc_results']}",
        f"  Innovus antenna results: {verification['innovus_antenna_results']}",
        f"  LVS status: {verification['lvs_status']}",
        "",
        "Hardened macros:",
    ])
    for macro in summary["macros"]:
        lines.append(
            f"  {macro['macro_class_id']}: reuse={macro['reuse_count']} area={macro['physical_area_um2']} um^2 "
            f"density_over_threshold={macro['density_over_threshold_percent']}% "
            f"setup_wns/tns={macro['setup_wns_ns']}/{macro['setup_tns_ns']} ns "
            f"hold_wns/tns={macro['hold_wns_ns']}/{macro['hold_tns_ns']} ns "
            f"DRC={macro['drc_results']} LVS={macro['lvs_status']}"
        )
    lines.extend(["", "Power: unavailable (Stage 1 currently emits no power report)."])
    (OUTPUTS / "flow-summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    registry = json.loads((INPUTS / "macro-registry.json").read_text())
    macros = [summarize_macro(item) for item in registry.get("macros", [])]
    stages = load_stage_metrics()
    drc_policy_path = INPUTS / "drc-policy.json"
    drc_policy = (
        json.loads(drc_policy_path.read_text()) if drc_policy_path.is_file() else {}
    )
    physical_intent = json.loads(read_optional(INPUTS / "physical-intent.json") or "{}")
    area = parse_innovus_area(
        read_optional(INPUTS / "full-chip-area.rpt"),
        os.environ.get("design_name", "undefined"),
    )
    report_design_name = os.environ.get("report_design_name", "").strip()
    if not report_design_name or report_design_name == "undefined":
        report_design_name = os.environ.get("design_name", "undefined")
    summary = {
        "schema_version": 1,
        "source_policy": "reports_and_explicit_metrics_only",
        "implementation_style": registry.get(
            "implementation_style", "hierarchical_macros"
        ),
        "macro_generation_bypassed": bool(
            registry.get("bypass_macro_generation", False)
        ),
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "design_name": report_design_name,
            "rtl_top_module": os.environ.get("design_name", "undefined"),
        },
        "parameters": {
            "clock_period": float(os.environ.get("clock_period", "10.0")),
            "macro_clock_period": float(
                os.environ.get("macro_clock_period", "8.0")
            ),
            "min_macro_reuse": int(os.environ.get("min_macro_reuse", "2")),
            "bypass_macro_generation": bool(
                registry.get("bypass_macro_generation", False)
            ),
            "antenna_check_policy": os.environ.get(
                "antenna_check_policy", "report"
            ),
        },
        "macro_count": len(macros),
        "macro_physical_area_um2": round(
            sum(item["physical_area_um2"] or 0.0 for item in macros), 6
        ),
        "coverage": coverage_metrics(physical_intent, area, macros),
        "stages": stages,
        "stages_by_node": {item["node"]: item for item in stages},
        "macros": macros,
        "full_chip_verification": {
            "drc_results": parse_drc_count(read_optional(INPUTS / "drc.summary")),
            "calibre_non_antenna_results": drc_policy.get("non_antenna_results"),
            "calibre_antenna_results": drc_policy.get("antenna_results"),
            "calibre_antenna_policy": drc_policy.get("antenna_check_policy"),
            "innovus_route_drc_results": parse_innovus_violation_count(
                read_optional(INPUTS / "innovus-drc.rpt")
            ),
            "innovus_antenna_results": parse_innovus_violation_count(
                read_optional(INPUTS / "innovus-antenna.rpt")
            ),
            "lvs_status": parse_lvs(read_optional(INPUTS / "lvs.report")),
        },
    }
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "flow-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_tcl(summary)
    write_text(summary)


if __name__ == "__main__":
    main()
