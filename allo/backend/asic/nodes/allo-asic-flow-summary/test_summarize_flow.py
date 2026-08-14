"""Unit tests for report parsing without commercial tools."""

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize_flow.py")
SPEC = importlib.util.spec_from_file_location("summarize_flow", MODULE_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_report_parsers():
    assert SUMMARY.parse_lef_size("SIZE 12.5 BY 9.25 ;") == (12.5, 9.25)
    assert SUMMARY.parse_path_slack(
        "slack (MET) 0.120\nslack (VIOLATED) -0.030\n"
    ) == -0.03
    timing = SUMMARY.parse_global_timing(
        "No setup violations found.\nHold violations\nWNS -0.030 0\nTNS -0.052 0\n"
    )
    assert timing == {
        "setup_wns_ns": 0.0,
        "setup_tns_ns": 0.0,
        "hold_wns_ns": -0.03,
        "hold_tns_ns": -0.052,
    }
    assert SUMMARY.parse_drc_count("TOTAL DRC Results Generated: 7 (7)") == 7
    assert SUMMARY.parse_lvs("# CORRECT #") == "passed"
    assert SUMMARY.parse_lvs("# INCORRECT #") == "failed"
    assert SUMMARY.parse_density(
        "default core: bins with density > 0.750 = 44.61 % ( 1 / 2 )"
    ) == 44.61
    assert SUMMARY.parse_innovus_area(
        "top 100 250.0 10 5 20 100 0 5 75 0\n", "top"
    ) == {
        "full_chip_report_total_area_um2": 250.0,
        "linked_macro_abstract_area_um2": 75.0,
        "remaining_top_standard_cell_area_um2": 175.0,
        "physical_only_cell_area_um2": 0.0,
    }


def test_summary_outputs_are_json_tcl_and_text(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    registry_root = inputs / "macro-registry"
    macro_dir = registry_root / "macro_alpha_test"
    reports = macro_dir / "reports"
    reports.mkdir(parents=True)
    (macro_dir / "macro.lef").write_text("MACRO m\n SIZE 10 BY 20 ;\nEND m\n")
    (reports / "timing-report_global_timing.report").write_text(
        "No setup violations found.\nNo hold violations found.\n"
    )
    (reports / "macro.timing.setup.rpt").write_text("slack (MET) 0.125\n")
    (reports / "macro.timing.hold.rpt").write_text("slack (MET) 0.075\n")
    (reports / "drc.summary").write_text("TOTAL DRC Results Generated: 0 (0)\n")
    (reports / "lvs.report").write_text("# CORRECT #\n")
    (reports / "pnr-signoff.area.rpt").write_text(
        "m 20 50.0 1 1 10 30 0 8 0 0\n"
    )
    registry = {
        "macros": [{
            "macro_class_id": "macro_alpha_test",
            "top_module": "m",
            "reuse_count": 4,
            "implementation_contract_hash": "contract123",
            "equivalence_method": "specialized_mlir_emitted_hls_contract",
            "rtl_audit_status": "agree",
            "rtl_audit_hashes": ["rtl123"],
            "rtl_hash": "rtl123",
            "views": {"lef": {"path": "macro_alpha_test/macro.lef"}},
        }]
    }
    inputs.mkdir(exist_ok=True)
    (inputs / "macro-registry.json").write_text(json.dumps(registry))
    for name in SUMMARY.METRIC_FILES:
        (inputs / name).write_text(json.dumps({
            "node": name.removesuffix("-metrics.json"),
            "status": "passed",
            "wall_seconds": 1.5,
        }))
    (inputs / "drc.summary").write_text(
        "TOTAL DRC Results Generated: 0 (0)\n"
    )
    (inputs / "innovus-drc.rpt").write_text(
        "Verification Complete : 0 Viols.\n"
    )
    (inputs / "innovus-antenna.rpt").write_text(
        "Verification Complete : 12 Viols.\n"
    )
    (inputs / "drc-policy.json").write_text(json.dumps({
        "antenna_check_policy": "report",
        "antenna_results": 12,
        "non_antenna_results": 0,
    }))
    (inputs / "lvs.report").write_text("# CORRECT #\n")
    (inputs / "physical-intent.json").write_text(json.dumps({
        "core": {"width": 100.0, "height": 80.0},
        "placements": [
            {"kind": "pe", "width": 10.0, "height": 20.0},
            {"kind": "pe", "width": 10.0, "height": 20.0},
            {"kind": "sram", "width": 20.0, "height": 20.0},
        ],
    }))
    (inputs / "full-chip-area.rpt").write_text(
        "top 100 1000.0 10 5 20 100 0 5 250.0 0\n"
    )
    monkeypatch.setattr(SUMMARY, "INPUTS", inputs)
    monkeypatch.setattr(SUMMARY, "REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(SUMMARY, "OUTPUTS", outputs)
    monkeypatch.setenv("design_name", "top")
    monkeypatch.setenv("report_design_name", "gemm-4x4")
    SUMMARY.main()
    result = json.loads((outputs / "flow-summary.json").read_text())
    assert result["run"]["design_name"] == "gemm-4x4"
    assert result["run"]["rtl_top_module"] == "top"
    assert result["full_chip_verification"]["innovus_route_drc_results"] == 0
    assert result["full_chip_verification"]["innovus_antenna_results"] == 12
    assert result["macros"][0]["physical_area_um2"] == 200.0
    assert result["macros"][0]["setup_wns_ns"] == 0.125
    assert result["macros"][0]["hold_wns_ns"] == 0.075
    assert result["macros"][0]["implementation_contract_hash"] == "contract123"
    assert result["macros"][0]["equivalence_method"] == (
        "specialized_mlir_emitted_hls_contract"
    )
    assert result["macros"][0]["rtl_audit_status"] == "agree"
    assert result["full_chip_verification"] == {
        "drc_results": 0,
        "calibre_non_antenna_results": 0,
        "calibre_antenna_results": 12,
        "calibre_antenna_policy": "report",
        "innovus_route_drc_results": 0,
        "innovus_antenna_results": 12,
        "lvs_status": "passed",
    }
    assert result["coverage"] == {
        "placed_macro_instance_count": 2,
        "instantiated_macro_footprint_area_um2": 400.0,
        "core_area_um2": 8000.0,
        "physical_macro_coverage_percent": 5.0,
        "full_chip_report_total_area_um2": 1000.0,
        "linked_macro_abstract_area_um2": 250.0,
        "remaining_top_standard_cell_area_um2": 750.0,
        "physical_only_cell_area_um2": 0.0,
        "hardened_macro_internal_standard_cell_area_um2": 200.0,
        "equivalent_total_standard_cell_area_um2": 950.0,
        "logic_hardening_coverage_percent": 21.052632,
    }
    assert "allo_asic_macro_count 1" in (outputs / "flow-summary.tcl").read_text()
    assert "Power: unavailable" in (outputs / "flow-summary.txt").read_text()


def test_summary_classifies_explicit_flat_bypass(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    registry_root = inputs / "macro-registry"
    registry_root.mkdir(parents=True)
    (inputs / "macro-registry.json").write_text(json.dumps({
        "implementation_style": "flat",
        "bypass_macro_generation": True,
        "macros": [],
    }))
    for name in SUMMARY.METRIC_FILES:
        (inputs / name).write_text(json.dumps({
            "node": name.removesuffix("-metrics.json"),
            "status": "bypassed" if name.startswith("macro-") else "passed",
            "wall_seconds": 0.0,
        }))
    monkeypatch.setattr(SUMMARY, "INPUTS", inputs)
    monkeypatch.setattr(SUMMARY, "REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(SUMMARY, "OUTPUTS", outputs)
    SUMMARY.main()
    result = json.loads((outputs / "flow-summary.json").read_text())
    assert result["implementation_style"] == "flat"
    assert result["macro_generation_bypassed"] is True
    assert result["macro_count"] == 0
    assert "allo_asic_macro_generation_bypassed 1" in (
        outputs / "flow-summary.tcl"
    ).read_text()
